import os
import time
import wandb
import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import torch

from .optimizer import get_optimizer
from .metrics import get_metric
from ..config import save_config
from ..models.model_util import load_model, save_checkpoint
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

    model, cfg.device = load_model(cfg, verbose=False)
    train_dataloader = get_dataloader(cfg, test=False)
    valid_dataloader = get_dataloader(cfg, test=True)
    
    trn_gt , val_gt  = map(lambda x: x.dataset.data_ages, [train_dataloader, valid_dataloader])
    trn_src, val_src = map(lambda x: x.dataset.data_src,  [train_dataloader, valid_dataloader])
    idx2src = {v: k for k, v in train_dataloader.dataset.src_map.items()}
    print(f"TOTAL TRAIN {len(train_dataloader.dataset)} | VALID {len(valid_dataloader.dataset)}")

    if checkpoint is not None:
        
        # CHECKPOINT DICT SHOULD CONTAIN
        #   + path: weight saved pth
        #   + resume_epoch: epoch that user wants to start with
        model_path = checkpoint['path']
        model.load_state_dict(torch.load(model_path)) 
        start = checkpoint['resume_epoch']

    else:
        start = 0
        
    optimizer = get_optimizer(model, cfg) # AGE PREDICTOR

    best_mae = float('inf')
    stop_count = 0
    wandb.watch(model)
    for e in range(start, cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        trn_loss, trn_metrics, trn_pred = train(model, optimizer, cfg, train_dataloader)
        val_loss, val_metrics, tst_pred = valid(model, cfg, valid_dataloader)
        disp_metrics(trn_loss, trn_metrics, val_loss, val_metrics)

        # PLOT
        if cfg.plot:
            df = pd.DataFrame({
                'Ground Truth': trn_gt + val_gt,
                'Predicted Age': torch.cat([trn_pred, tst_pred]),
                'Source': list(map(lambda x: idx2src[x], trn_src + val_src)),
                'Phase': ['Train'] * len(trn_src) + ['Valid'] * len(val_src),
            })
            source_plot = reg_plot(df, 'Source', cfg.model_name)
            phase_plot  = reg_plot(df, 'Phase',  cfg.model_name)
            wandb.log(dict(
                trn_metrics,
                train_loss=trn_loss,
                valid_loss=val_loss,
                source_plot=source_plot,
                phase_plot=phase_plot,
                **val_metrics
            ))

        else: # WITHOUT PLOT
            wandb.log(dict(
                trn_metrics,
                train_loss=trn_loss,
                valid_loss=val_loss,
                **val_metrics
            ))
        
        model_name = f'{cfg.model_name}_ep{e+1}_mae{val_metrics["valid_mae"]:.2f}.pth'
        if best_mae > val_metrics['valid_mae']:
            
            stop_count = 0
            best_mae = val_metrics['valid_mae']
            save_checkpoint(model.state_dict(), model_name, model_dir=cfg.RESULT_PATH)

        elif stop_count >= cfg.early_patience:
 
            print(f'Early stopped with {stop_count} / {cfg.early_patience} at EPOCH {e}')
            break

        else:
            if (e + 1) % cfg.verbose_period == 0:
                save_checkpoint(model.state_dict(), model_name, model_dir=cfg.RESULT_PATH)

            # To prevent training being stopped even before expected performance
            if best_mae < cfg.mae_threshold:
                stop_count += 1

        torch.cuda.empty_cache()

    if cfg.debug:
        if cfg.run_debug.return_all:
            return model, (trn_loss, trn_metrics, trn_pred), (val_loss, val_metrics, tst_pred)

    cfg.best_mae = best_mae
    save_config(cfg, os.path.join(cfg.RESULT_PATH, 'config.yml'))
    wandb.config.update(cfg)
    wandb.finish()

    return model


@logging_time
def train(model, optimizer, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=False)
    trues = torch.tensor(dataloader.dataset.data_ages)

    device = cfg.device

    model.train()
    preds, losses = [], []
    for i, (x, y) in enumerate(dataloader):

        try: 
            x, y = x.to(device), y.to(device)

        except FileNotFoundError as e:
            print(e)
            time.sleep(20)
            continue

        optimizer.zero_grad()

        y_pred = model.forward(x).squeeze()
        preds.append(y_pred.cpu())

        loss = get_metric(y_pred, y, cfg.loss)

        # Track down results
        losses.append(loss.item())

        if cfg.lamb:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += cfg.lamb * l2_reg

        loss.backward()
        optimizer.step()

        del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    preds = torch.cat(preds).detach()

    # Gather all metrics
    _metrics = {f'train_{metric}': get_metric(preds, trues, metric) for metric in cfg.metrics}
    loss = sum(losses) / len(losses)

    return loss, _metrics, preds


@logging_time
def valid(model, cfg, dataloader=None):

    if dataloader is None:
        dataloader = get_dataloader(cfg, test=True)
    trues = torch.tensor(dataloader.dataset.data_ages)

    device = cfg.device
    
    model.eval()
    preds, losses = [], []
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            try:
                x, y = x.to(device), y.to(device)

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                continue

            y_pred = model.forward(x).to(device).squeeze()
            preds.append(y_pred.cpu())

            loss = get_metric(y_pred, y, 'mse')

            # Track down results
            losses.append(loss.item())

            if cfg.lamb:
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += cfg.lamb * l2_reg

            del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    preds = torch.cat(preds).detach()

    # Gather all metrics
    _metrics = {f'valid_{metric}': get_metric(preds, trues, metric) for metric in cfg.metrics}
    loss = sum(losses) / len(losses)
    
    return loss, _metrics, preds


def reg_plot(df, hue, model_name, filter:list =None):

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title(f'True-Prediction plot - {hue}\n{model_name}', size=14)

    iteration = df[hue].unique() if filter is None else filter
    for it in iteration:
        sns.regplot(
            data=df[df[hue] == it],
            x='Ground Truth',
            y='Predicted Age',
            ax=ax, label=it, fit_reg=True
        )
    ax.legend(title=hue)
    return ax