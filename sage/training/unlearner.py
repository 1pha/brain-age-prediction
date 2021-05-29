import time
import wandb

import torch
import torch.optim as optim

from .metrics import get_metric
from .optimizer import get_optimizer
from ..models.model_util import load_unlearn_models, save_checkpoint
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

    # TRAIN
    metrics = ' | '.join(map(lambda k: f'{k[0].split("_")[1].upper()} {k[1]:.2f}', list(trn_metrics.items())))
    if isinstance(trn_loss, tuple): # PRINT OPTION WHEN UNLEARNING
        print(f'TRAIN :: REG {trn_loss[0]:.3f} DOM {trn_loss[1]:.3f} CONF {trn_loss[2]:.3f} \n{metrics}')

    else:
        print(f'TRAIN :: LOSS {trn_loss:.3f} | {metrics}')

    # VALID
    metrics = ' | '.join(map(lambda k: f'{k[0].split("_")[1].upper()} {k[1]:.2f}', list(val_metrics.items())))
    if isinstance(val_loss, tuple):
        print(f'TRAIN :: REG {val_loss[0]:.3f} DOM {val_loss[1]:.3f} CONF {val_loss[2]:.3f} \n{metrics}')
    else:
        print(f'VALID :: LOSS {val_loss:.3f} | {metrics}')


def run(cfg, checkpoint: dict=None):

    (encoder, regressor, domainer), cfg.device = load_unlearn_models(cfg.unlearn_cfg)
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
        
    unlearn_cfg = cfg.unlearn_cfg
    optimizers = (
        get_optimizer([encoder, regressor], unlearn_cfg.opt_age), # AGE PREDICTOR
        get_optimizer([domainer], unlearn_cfg.opt_dom), # DOMAIN PREDICTOR
        get_optimizer([encoder], unlearn_cfg.opt_conf), # CONFUSION
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
            train_reg_loss=trn_loss[0],
            valid_reg_loss=val_loss[0],

            train_dom_loss=trn_loss[1],
            valid_dom_loss=val_loss[1],

            train_conf_loss=trn_loss[2],
            valid_conf_loss=val_loss[2],

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
            return models, (trn_loss, trn_metrics, trn_pred), (val_loss, val_metrics, tst_pred)

    wandb.config.update(cfg)
    wandb.finish()

    return models


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
    age_preds, src_preds = [], []
    reg_loss, dom_loss, conf_loss = [], [], []
    for i, (x, y, d) in enumerate(dataloader):

        if cfg.debug and cfg.run_debug.verbose_all:
            print(f'{i}th Batch.')

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
        loss = get_metric(y_pred.squeeze(), y, cfg.loss)
        loss.backward(retain_graph=False)
        opt_reg.step()

        # STEP 2. DOMAIN PREDICT
        opt_dom.zero_grad()
        d_pred = domainer(embedded.detach())
        loss_dm = cfg.unlearn_cfg.alpha * get_metric(d_pred, d, 'ce')
        loss_dm.backward(retain_graph=False)
        opt_dom.step()

        # STEP 3. CONFUSION LOSS
        # del embedded
        # embedded = encoder.forward(x).to(device)

        opt_conf.zero_grad()
        d_pred = domainer(embedded.detach())
        loss_conf = cfg.unlearn_cfg.beta * get_metric(d_pred, d, 'confusion')
        loss_conf.backward()
        opt_conf.step()

        age_preds.append(y_pred.cpu())
        src_preds.append(d_pred.cpu())

        reg_loss.append(loss.item())
        dom_loss.append(loss_dm.item())
        conf_loss.append(loss_conf.item())

        del x, y, y_pred, d_pred
        torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach().squeeze()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    _metrics = {f'train_{metric}': get_metric(ages, gt_age, metric) for metric in cfg.metrics}
    # TODO: add ACCURACY
    reg_loss, dom_loss, conf_loss = map(lambda x: sum(x) / len(x), [reg_loss, dom_loss, conf_loss])

    return (reg_loss, dom_loss, conf_loss), _metrics, ages


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
    age_preds, src_preds = [], []
    reg_loss, dom_loss, conf_loss = [], [], []
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
            loss = get_metric(y_pred.squeeze(), y, cfg.loss)

            # STEP 2. DOMAIN PREDICT
            d_pred = domainer(embedded)
            loss_dm = cfg.unlearn_cfg.alpha * get_metric(d_pred, d, 'ce')

            # STEP 3. CONFUSION LOSS
            d_pred = domainer(embedded)
            loss_conf = cfg.unlearn_cfg.beta * get_metric(d_pred, d, 'confusion')

            age_preds.append(y_pred.cpu())
            src_preds.append(d_pred.cpu())
            
            reg_loss.append(loss.item())
            dom_loss.append(loss_dm.item())
            conf_loss.append(loss_conf.item())

            del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach().squeeze()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    _metrics = {f'valid_{metric}': get_metric(ages, trues, metric) for metric in cfg.metrics}
    reg_loss, dom_loss, conf_loss = map(lambda x: sum(x) / len(x), [reg_loss, dom_loss, conf_loss])
    
    return (reg_loss, dom_loss, conf_loss), _metrics, ages