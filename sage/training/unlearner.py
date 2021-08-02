import time
import wandb

import torch

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
                 val_loss, val_metrics,
                 trn_dom,  val_dom):

    LOSS = ['REG', 'DOM', 'CONF']
    def _format(key, value, option=None):

        if option == 'split':
            _f = lambda k, v: f'{k.split("_")[1].upper():5} {v:7.2f}'

        elif option is None:
            _f = lambda k, v: f'{k:5}{v:7.2f}'

        return ' | '.join(map(_f, key, value))

    # TRAIN
    metrics = _format(trn_metrics.keys(), trn_metrics.values(), option='split')
    if isinstance(trn_loss, tuple): # PRINT OPTION WHEN UNLEARNING
        loss = _format(LOSS, trn_loss)
        dom  = _format(trn_dom.keys(), trn_dom.values(), option='split')
        print(f'\tTRAIN :: \n\t{loss}\n\t{dom}\n\t{metrics}')

    else:
        print(f'\tTRAIN :: LOSS {trn_loss:.3f} | {metrics}')

    # VALID
    metrics = _format(val_metrics.keys(), val_metrics.values(), option='split')
    if isinstance(val_loss, tuple): # PRINT OPTION WHEN UNLEARNING
        loss = _format(LOSS, val_loss)
        dom  = _format(val_dom.keys(), val_dom.values(), option='split')
        print(f'\tVALID :: \n\t{loss}\n\t{dom}\n\t{metrics}')
    else:
        print(f'\tVALID :: LOSS {val_loss:.3f} | {metrics}')


def run(cfg, checkpoint: dict=None):

    (encoder, regressor, domainer), cfg.device = load_unlearn_models(cfg.unlearn_cfg)
    models = {
        'encoder': encoder,
        'regressor': regressor,
        'domainer': domainer,
    }
    train_dataloader = get_dataloader(cfg, test=False)
    valid_dataloader = get_dataloader(cfg, test=True)
    print(f"TOTAL TRAIN {len(train_dataloader.dataset)} | VALID {len(valid_dataloader.dataset)}")

    if checkpoint is not None:
        
        # CHECKPOINT DICT SHOULD CONTAIN
        #   + resume_epoch: epoch that user wants to start with
        # TODO
        #   + make path loading ...
        start = checkpoint['resume_epoch']
        for pipe, pth in checkpoint:
            models[pipe].load_state_dict(torch.load(pth))

    else:
        start = 0
        
    unlearn_cfg = cfg.unlearn_cfg
    optimizers = (
        get_optimizer([encoder, regressor], unlearn_cfg.opt_age), # AGE PREDICTOR
        get_optimizer([domainer], unlearn_cfg.opt_dom), # DOMAIN PREDICTOR
        get_optimizer([encoder], unlearn_cfg.opt_conf), # CONFUSION
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_mae = float('inf')
    stop_count = 0
    for e in range(start, cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        cfg.unlearn_cfg = set_point(cfg.unlearn_cfg, e)
        trn_loss, (trn_metrics, trn_dom), trn_pred = train(models.values(), optimizers, scaler, cfg)
        val_loss, (val_metrics, val_dom), tst_pred = valid(models.values(), cfg)
        disp_metrics(trn_loss, trn_metrics, val_loss, val_metrics, trn_dom, val_dom)
        wandb.log(dict(
            {**trn_metrics, **val_metrics,
             **trn_dom,     **val_dom,},
            
            train_reg_loss=trn_loss[0],
            valid_reg_loss=val_loss[0],

            train_dom_loss=trn_loss[1],
            valid_dom_loss=val_loss[1],

            train_conf_loss=trn_loss[2],
            valid_conf_loss=val_loss[2],
         ))
        
        model_name = f'ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{val_metrics["valid_mae"]:.2f}.pt'
        if best_mae > val_metrics['valid_mae']:
            
            stop_count = 0
            best_mae = val_metrics['valid_mae']
            save_checkpoint(states=models, model_dir=cfg.RESULT_PATH, model_name=model_name)

        elif stop_count >= cfg.early_patience:

            # print(f'Early stopped with {stop_count} / {cfg.early_patience} at EPOCH {e}')
            if cfg.unlearn_cfg.opt_conf.use == False:
                print(f'Fully pretrained {stop_count} / {cfg.early_patience} at EPOCH {e}')
                cfg.unlearn_cfg.opt_conf.use = True
                cfg.unlearn_cfg.opt_conf.point = e

            else:
                print(f'Early stopped at {stop_count} / {cfg.early_patience} at EPOCH {e}')
                break

        else:
            if cfg.verbose_period % (e + 1) == 0:
                save_checkpoint(states=models, model_dir=cfg.RESULT_PATH, model_name=model_name)

            # To prevent training being stopped even before expected performance
            if best_mae < cfg.mae_threshold:
                stop_count += 1

    if cfg.debug:
        if cfg.run_debug.return_all:
            return models, (trn_loss, trn_metrics, trn_pred), (val_loss, val_metrics, tst_pred)

    cfg.best_mae = best_mae
    wandb.config.update(cfg)
    wandb.finish()

    return models


@logging_time
def train(models, optimizers, scaler, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=False)
    gt_age = torch.tensor(dataloader.dataset.data_ages, dtype=torch.float)
    gt_src = torch.tensor(dataloader.dataset.data_src)

    device = cfg.device

    encoder, regressor, domainer = models
    opt_reg, opt_dom, opt_conf = optimizers

    encoder.train()
    regressor.train()
    domainer.train()
    age_preds, src_preds = [], []
    reg_loss, dom_loss, conf_loss = [], [], []
    with torch.autograd.set_detect_anomaly(True):
        for i, (x, y, d) in enumerate(dataloader):

            if cfg.debug and cfg.run_debug.verbose_all:
                print(f'{i}th Batch.')

            try: 
                x, y, d = map(lambda x: x.to(device), (x, y, d))

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                continue
            
            # STEP 1. FEATURE EXTRACT
            if cfg.unlearn_cfg.opt_age.use:

                with torch.cuda.amp.autocast(cfg.use_amp):
                    embedded = encoder.forward(x)
                    y_pred = regressor(embedded)
                    loss = get_metric(y_pred.squeeze(), y, cfg.loss)
                
                retain_graph = cfg.unlearn_cfg.opt_conf.use
                scaler.scale(loss).backward(retain_graph=retain_graph)
                scaler.step(opt_reg)
                scaler.update()
                opt_reg.zero_grad()

                reg_loss.append(loss.item())
                age_preds.append(y_pred.cpu())


            # STEP 2. DOMAIN PREDICT
            if cfg.unlearn_cfg.opt_dom.use:

                with torch.cuda.amp.autocast(cfg.use_amp):
                    d_pred = domainer(embedded.detach())
                    loss_dm = cfg.unlearn_cfg.alpha * get_metric(d_pred, d, 'ce')

                scaler.scale(loss_dm).backward()
                scaler.step(opt_dom)
                scaler.update()
                opt_dom.zero_grad()

                dom_loss.append(loss_dm.item())
                src_preds.append(d_pred.cpu())


            # STEP 3. CONFUSION LOSS
            if cfg.unlearn_cfg.opt_conf.use:

                # del embedded
                # embedded = encoder.forward(x)
                # embedded.grad.zero_()
                # for param in encoder.parameters():
                #     param.detach()
                
                with torch.cuda.amp.autocast(cfg.use_amp):
                    d_pred_conf = domainer(embedded.clone())
                    loss_conf = cfg.unlearn_cfg.beta * get_metric(d_pred_conf, d, 'confusion')

                scaler.scale(loss_conf).backward()
                scaler.step(opt_conf)
                scaler.update()
                opt_conf.zero_grad()

                del loss, loss_conf
                conf_loss.append(loss_conf.item())

            else:
                del loss

            del x, y, y_pred, d_pred
            torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach().squeeze()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    age_metrics = {f'train_{metric}': get_metric(ages, gt_age, metric) for metric in cfg.metrics}
    src_metrics = {f'train_{metric}': get_metric(srcs, gt_src, metric) for metric in cfg.unlearn_cfg.metrics}
    reg_loss, dom_loss, conf_loss = map(agg_loss, [reg_loss, dom_loss, conf_loss])

    return (reg_loss, dom_loss, conf_loss), (age_metrics, src_metrics), (ages, srcs)


@logging_time
def valid(models, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=True)
    gt_age = torch.tensor(dataloader.dataset.data_ages, dtype=torch.float)
    gt_src = torch.tensor(dataloader.dataset.data_src)

    device = cfg.device

    encoder, regressor, domainer = models

    encoder.eval()
    regressor.eval()
    domainer.eval()
    age_preds, src_preds = [], []
    reg_loss, dom_loss, conf_loss = [], [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i, (x, y, d) in enumerate(dataloader):

                try:
                    x, y, d = x.to(device), y.to(device), d.to(device)

                except FileNotFoundError as e:
                    print(e)
                    time.sleep(20)
                    continue

                # STEP 1. FEATURE EXTRACT
                if cfg.unlearn_cfg.opt_age.use:
                    embedded = encoder.forward(x).to(device)
                    y_pred = regressor(embedded)
                    loss = get_metric(y_pred.squeeze(), y, cfg.loss)

                    reg_loss.append(loss.item())
                    age_preds.append(y_pred.cpu())

                # STEP 2. DOMAIN PREDICT
                if cfg.unlearn_cfg.opt_dom.use:
                    d_pred = domainer(embedded)
                    loss_dm = cfg.unlearn_cfg.alpha * get_metric(d_pred, d, 'ce')
                    dom_loss.append(loss_dm.item())
                    src_preds.append(d_pred.cpu())

                # STEP 3. CONFUSION LOSS
                if cfg.unlearn_cfg.opt_conf.use:
                    d_pred = domainer(embedded)
                    loss_conf = cfg.unlearn_cfg.beta * get_metric(d_pred, d, 'confusion')
                    conf_loss.append(loss_conf.item())

                del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    ages = torch.cat(age_preds).detach().squeeze()
    srcs = torch.cat(src_preds).detach()

    # Gather all metrics
    age_metrics = {f'valid_{metric}': get_metric(ages, gt_age, metric) for metric in cfg.metrics}
    src_metrics = {f'valid_{metric}': get_metric(srcs, gt_src, metric) for metric in cfg.unlearn_cfg.metrics}
    reg_loss, dom_loss, conf_loss = map(agg_loss, [reg_loss, dom_loss, conf_loss]) 
    
    return (reg_loss, dom_loss, conf_loss), (age_metrics, src_metrics), (ages, srcs)


def agg_loss(loss: list):

    if loss: # IF THERE IS SOMETHING ...
        return sum(loss) / len(loss)
    
    else: # NO, THEN RETURN 0 FOR LOSS
        return 0


def set_point(cfg, e): # UNLEARN CFG

    TASKS = ['opt_age', 'opt_dom', 'opt_conf']
    for t in TASKS:
        cfg[t].use = e >= cfg[t].point
        print(f' {t}: {cfg[t].use} ', end='')
    print()

    return cfg