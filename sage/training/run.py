import time
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from .losses import get_metric
from ..models.model_util import load_model, save_checkpoint
from ..data.data_util import get_dataloader

def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        end = '' if original_fn.__name__=='train' else '\n'
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec ", end=end)
        return result

    return wrapper_fn

class AgeTrainer:

    def __init__(self, cfg):

        pass


def run(cfg, checkpoint: dict=None):

    model, cfg.device = load_model(cfg.model_name, verbose=False)

    if checkpoint is not None:
        
        model_path = checkpoint['path']
        model.load_state_dict(torch.load(model_path))

        start = checkpoint['resume_epoch']

    else:
        start = 0
        
    # TODO add optimizer config
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_mae = -1
    stop_count = 0
    for e in range(start, cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        trn_metrics, trn_pred = train(model, optimizer, cfg)
        val_metrics, tst_pred = valid(model, cfg)
        
        if best_mae > val_metrics['mae']:
            
            stop_count = 0
            best_mae = val_metrics['mae']
            model_name = f'{cfg.model_name}_ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{best_mae:.3f}.pt'
            save_checkpoint(cfg.get_dict(), model_name, is_best=True)

        elif stop_count >= cfg.early_patience:

            print(f'Early stopped with {stop_count} / {cfg.early_patience} at EPOCH {e}')
            break

        else:
            stop_count += 1

        torch.cuda.empty_cache()

    return model


@logging_time
def train(model, optimizer, cfg, augment=False):

    dataloader = get_dataloader(augment=augment, test=False)

    device = cfg.device

    model.train()
    predictions = []
    losses = []
    _metrics = {metric: [] for metric in cfg.metrics}
    for i, (x, y) in enumerate(dataloader):

        try: 
            x, y = x.to(device), y.to(device)

        except FileNotFoundError as e:
            print(e)
            time.sleep(20)
            continue

        optimizer.zero_grad()

        y_pred = model.forward(x).to(device).squeeze()
        predictions.append(y_pred.cpu())

        loss = get_metric(y_pred, y, cfg.loss)

        # Track down results
        losses.append(loss.item())
        for metric in cfg.metrics:
            _metrics[metric].append(get_metric(y_pred, y, metric))

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
    predictions = torch.cat(predictions).detach().numpy()

    # Gather all metrics
    _metrics = {metric: sum(_v)/len(_v) for metric, _v in _metrics.items()}
    loss = sum(losses) / len(losses)

    return loss, _metrics, predictions


@logging_time
def valid(model, cfg):

    dataloader = get_dataloader(augment=False, test=True)

    device = cfg.device
    
    model.eval()
    predictions = []
    losses = []
    _metrics = {metric: [] for metric in cfg.metrics}
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            try:
                x, y = x.to(device), y.to(device)

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                continue

            y_pred = model.forward(x).to(device).squeeze()
            predictions.append(y_pred.cpu())

            loss = get_metric(y_pred, y, 'mse')

            # Track down results
            losses.append(loss.item())
            for metric in cfg.metrics:
                _metrics[metric].append(get_metric(y_pred, y, metric))

            if cfg.lamb:
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += cfg.lamb * l2_reg

            del x, y, y_pred

    torch.cuda.empty_cache()

    # Gather Prediction results
    predictions = torch.cat(predictions).detach().numpy()

    # Gather all metrics
    _metrics = {metric: sum(_v)/len(_v) for metric, _v in _metrics.items()}

    loss = sum(losses) / len(losses)
    
    return loss, _metrics, predictions