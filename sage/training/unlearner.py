import time
import wandb

import torch
import torch.optim as optim

from .metrics import get_metric
from .optimizer import get_optimizer
from ..models.model_util import load_model, load_unlearn_models, save_checkpoint
from ..data.dataloader import get_dataloader


def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        end = '' if original_fn.__name__=='train' else '\n'
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec ", end=end)
        return result

    return wrapper_fn


def disp_metrics(trn_loss, trn_metrics,
                 val_loss, val_metrics):

    metrics = ' | '.join(map(lambda k: f'{k[0].split("_")[1].upper()} {k[1]:.2f}', list(trn_metrics.items())))
    print(f'TRAIN :: LOSS {trn_loss:.3f} | {metrics}')
    metrics = ' | '.join(map(lambda k: f'{k[0].split("_")[1].upper()} {k[1]:.2f}', list(val_metrics.items())))
    print(f'VALID :: LOSS {val_loss:.3f} | {metrics}')


def run(cfg, checkpoint: dict=None):

    (encoder, regressor, domainer), cfg.device = load_unlearn_models(cfg)
    train_dataloader = get_dataloader(cfg, test=False)
    valid_dataloader = get_dataloader(cfg, test=True)
    print(f"TOTAL TRAIN {len(train_dataloader.dataset)} | VALID {len(valid_dataloader.dataset)}")

    if checkpoint is not None:
        
        # CHECKPOINT DICT SHOULD CONTAIN
        #   + resume_epoch: epoch that user wants to start with
        # TODO
        #   + make path loading ...
        start = checkpoint['resume_epoch']

    else:
        start = 0
        
    optimizers = (
        get_optimizer([encoder, regressor], cfg.regression), # REGRESSION
        get_optimizer([domainer], cfg.domain), # DOMAIN PREDICTOR
        get_optimizer([encoder], cfg.confusion), # CONFUSION
    )

    best_mae = float('inf')
    stop_count = 0
    models = (encoder, regressor, domainer)
    for e in range(start, cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        trn_loss, trn_metrics, trn_pred = train(models, optimizers, cfg)
        val_loss, val_metrics, tst_pred = valid(models, cfg)
        disp_metrics(trn_loss, trn_metrics, val_loss, val_metrics)
        wandb.log(dict(
            trn_metrics,
            train_loss=trn_loss,
            valid_loss=val_loss,
            **val_metrics
         ))
        
        model_name = f'{cfg.model_name}_ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{val_metrics["valid_mae"]:.2f}.pt'
        if best_mae > val_metrics['valid_mae']:
            
            stop_count = 0
            best_mae = val_metrics['valid_mae']
            # save_checkpoint(model.state_dict(), model_name, is_best=best_mae)

        elif stop_count >= cfg.early_patience:

            print(f'Early stopped with {stop_count} / {cfg.early_patience} at EPOCH {e}')
            break

        else:
            if cfg.verbose_period % (e + 1) == 0:
                # save_checkpoint(model.state_dict(), model_name, is_best=False)
                pass

            # To prevent training being stopped even before expected performance
            if best_mae < cfg.mae_threshold:
                stop_count += 1

        torch.cuda.empty_cache()

    if cfg.debug:
        if cfg.run_debug.return_all:
            return model, (trn_loss, trn_metrics, trn_pred), (val_loss, val_metrics, tst_pred)

    wandb.config.update(cfg)
    wandb.finish()

    return model


@logging_time
def train(models, optimizers, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=False)
    gt_age = torch.tensor(dataloader.dataset.data_ages)
    gt_src = torch.tensor(dataloader.dataset.data_src)

    device = cfg.device

    encoder, regressor, domainer = models
    opt_reg, opt_dom, opt_conf = optimizers

    encoder.train()
    regressor.train()
    domainer.train()
    age_preds, src_preds, losses = [], [], []
    for i, (x, y, d) in enumerate(dataloader):

        try: 
            x, y, d = x.to(device), y.to(device), d.to(device)

        except FileNotFoundError as e:
            print(e)
            time.sleep(20)
            continue

        # STEP 1. FEATURE EXTRACT
        opt_reg.zero_grad()
        embedded = encoder.forward(x).to(device)
        y_pred = regressor(embedded)
        loss = get_metric(y_pred, y, cfg.loss)
        loss.backward()
        opt_reg.step()

        # STEP 2. DOMAIN PREDICT
        opt_dom.zero_grad()
        d_pred = domainer(embedded)
        loss_dm = cfg.dom_eta * get_metric(d_pred, d, 'ce')
        loss_dm.backward()
        opt_dom.step()

        # STEP 3. CONFUSION LOSS
        opt_conf.zero_grad()
        d_pred = domainer(embedded)
        loss_conf = cfg.conf_eta * get_metric(d_pred, d, 'confusion')
        loss_conf.backward()
        opt_conf.step()

        age_preds.append(y_pred.cpu())
        src_preds.append(d_pred.cpu())
        losses.append(loss.item())

        del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    _metrics = {f'train_{metric}': get_metric(ages, gt_age, metric) for metric in cfg.metrics}
    # TODO: add ACCURACY
    loss = sum(losses) / len(losses)

    return loss, _metrics, ages


@logging_time
def valid(models, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=True)
    trues = torch.tensor(dataloader.dataset.data_ages)

    device = cfg.device

    encoder, regressor, domainer = models

    encoder.eval()
    regressor.eval()
    domainer.eval()
    age_preds, src_preds, losses = [], [], []
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y, d) in enumerate(dataloader):

            try:
                x, y, d = x.to(device), y.to(device), d.to(device)

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                continue

            # STEP 1. FEATURE EXTRACT
            embedded = encoder.forward(x).to(device)
            y_pred = regressor(embedded)
            loss = get_metric(y_pred, y, cfg.loss)
            loss.backward()

            # STEP 2. DOMAIN PREDICT
            d_pred = domainer(embedded)
            loss_dm = cfg.dom_eta * get_metric(d_pred, d, 'ce')
            loss_dm.backward()

            # STEP 3. CONFUSION LOSS
            d_pred = domainer(embedded)
            loss_conf = cfg.conf_eta * get_metric(d_pred, d, 'confusion')
            loss_conf.backward()

            age_preds.append(y_pred.cpu())
            src_preds.append(d_pred.cpu())
            losses.append(loss.item())

            del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    _metrics = {f'valid_{metric}': get_metric(ages, trues, metric) for metric in cfg.metrics}
    loss = sum(losses) / len(losses)
    
    return loss, _metrics, ages