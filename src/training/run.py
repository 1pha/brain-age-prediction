import time
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from .losses import get_metric
from ..data.data_util import get_dataloader 
from ..models.model_util import load_model, save_checkpoint

def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        end = '' if original_fn.__name__=='train' else '\n'
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec ", end=end)
        return result

    return wrapper_fn

def make_df(data, label):
    
    preds, trues = data
    return pd.DataFrame({
        'True': list(map(float, trues)),
        'Prediction': list(map(float, preds)),
        'Label': [label] * len(trues)
    })

class AgeTrainer:

    def __init__(self, cfg):

        pass


def run(cfg, fold, db=None, mlflow=None):

    model, cfg.device = load_model(cfg.model_name, verbose=False)

    # TODO add optimizer config
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_mae = -1
    fold = None
    for e in range(cfg.epochs):
        
        print(f'Epoch {e + 1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        trn_metrics, trn_res = train(model, optimizer, cfg)
        aug_metrics, aug_res = train(model, optimizer, cfg, augment=True)
        val_metrics, tst_res = valid(model, cfg, fold=fold)
        
        if best_mae < val_metrics['mae']:
            
            best_mae = val_metrics['mae']
            model_name = f'{cfg.model_name}_ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{best_mae:.3f}.pt'
            save_checkpoint(cfg.get_dict(), model_name, is_best=True)
            
        df = pd.concat([make_df(trn_res, 'Train'),
                        make_df(aug_res, 'Aug'),
                        make_df(tst_res, 'Valid')], ignore_index=True)

        torch.cuda.empty_cache()

    mlflow.end_run()

    return model, (trn_res, tst_res)


@logging_time
def train(model, optimizer, loss_fns, CFG, augment=False):

    dataloader = get_dataloader(augment=augment, test=False)

    device = CFG.device

    model.train()
    predictions, targets = [], []
    losses = []
    _metrics = {metric: [] for metric in CFG.metrics}
    for i, (x, y) in enumerate(dataloader):

        try: 
            x, y = x.to(device), y.to(device)

        except FileNotFoundError as e:
            print(e)
            time.sleep(20)
            pass

        optimizer.zero_grad()

        y_pred = model.forward(x).to(device).squeeze()
        predictions.append(y_pred.cpu())
        targets.append(y.cpu())

        loss = loss_fns['mse'](y_pred, y)

        # Track down results
        losses.append(loss)
        for metric in CFG.metrics:
            _metrics[metric].append(get_metric(y_pred, y, metric))

        if CFG.lamb:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += CFG.lamb * l2_reg

        loss.backward()
        optimizer.step()

        del x, y, y_pred

    torch.cuda.empty_cache()

    predictions = torch.cat(predictions).detach().numpy()
    targets = torch.cat(targets).detach().numpy()
    _metrics = {metric: sum(_v)/len(_v) for metric, _v in _metrics.items()}

    return _metrics, (predictions, targets)


@logging_time
def valid(CFG):

    dataloader = get_dataloader(augment=False, test=True)

    device = CFG.device
    
    model.eval()
    predictions, targets = [], []
    losses = []
    _metrics = {metric: [] for metric in CFG.metrics}
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            try:
                x, y = x.to(device), y.to(device)

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                pass

            y_pred = model.forward(x).to(device).squeeze()
            predictions.append(y_pred.cpu())
            targets.append(y.cpu())

            loss = loss_fns['mse'](y_pred, y)

            # Track down results
            losses.append(loss)
            for metric in CFG.metrics:
                _metrics[metric].append(get_metric(y_pred, y, metric))

            if CFG.lamb:
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += CFG.lamb * l2_reg

            del x, y, y_pred

    torch.cuda.empty_cache()

    predictions = torch.cat(predictions).detach().numpy()
    targets = torch.cat(targets).detach().numpy()
    _metrics = {metric: sum(_v)/len(_v) for metric, _v in _metrics.items()}
    
    return _metrics, (predictions, targets)