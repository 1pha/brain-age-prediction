import time
import wandb

import torch
import torch.optim as optim

from .metrics import get_metric
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

    model, cfg.device = load_model(cfg, verbose=True)

    if checkpoint is not None:
        
        # CHECKPOINT VARIALBE SHOULD CONTAIN
        #   + path: weight saved pth
        #   + resume_epoch: epoch that user wants to start with
        model_path = checkpoint['path']
        model.load_state_dict(torch.load(model_path))
        start = checkpoint['resume_epoch']

    else:
        start = 0
        
    # TODO add optimizer config
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_mae = cfg.mae_threshold
    stop_count = 0
    for e in range(start, cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        trn_loss, trn_metrics, trn_pred = train(model, optimizer, cfg)
        val_loss, val_metrics, tst_pred = valid(model, cfg)
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
            save_checkpoint(cfg, model_name, is_best=best_mae)

        elif stop_count >= cfg.early_patience:

            print(f'Early stopped with {stop_count} / {cfg.early_patience} at EPOCH {e}')
            break

        else:
            if cfg.verbose_period % (e + 1) == 0:
                save_checkpoint(cfg, model_name, is_best=False)
            stop_count += 1

        torch.cuda.empty_cache()

    if cfg.debug:
        if cfg.run_debug.return_all:
            return model, (trn_loss, trn_metrics, trn_pred), (val_loss, val_metrics, tst_pred)

    return model


@logging_time
def train(model, optimizer, cfg):

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

        y_pred = model.forward(x).to(device).squeeze()
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
def valid(model, cfg):

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