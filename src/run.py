import time
from itertools import chain
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F
from .data_util import * 
from .losses import RMSELoss, fn_lst


def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec  ", end='')
        return result

    return wrapper_fn

def make_df(data, label):
    
    preds, trues = data
    return pd.DataFrame({
        'True': list(map(float, trues)),
        'Prediction': list(map(float, preds)),
        'Label': [label] * len(trues)
    })


@logging_time
def train(model, optimizer, loss_fns, DP, CFG, fold=None, augment=False):

    dset = MyDataset(CFG, augment=augment, fold=fold)
    dataloader = DataLoader(dset, batch_size=CFG.batch_size)

    device = CFG.device

    model.train()
    predictions, targets = [], []
    for i, (x, y) in enumerate(dataloader):

        if CFG.resize:
            x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

        else:
            x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred = model.forward(x).to(device).squeeze(1)
        predictions.append(y_pred.cpu())
        targets.append(y.cpu())

        loss = loss_fns['mse'](y_pred, y)

        # Track down results
        DP.loss.batch_update(loss.item(), 1)
        DP.mae.batch_update(loss_fns['mae'](y_pred, y).item(), 1)
        DP.rmse.batch_update(loss_fns['rmse'](y_pred, y).item(), 1)

        if CFG.lamb:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += CFG.lamb * l2_reg

        loss.backward()
        optimizer.step()

        del x, y, y_pred

    torch.cuda.empty_cache()
    DP.loss.clear()
    DP.mae.clear()
    DP.rmse.clear()
    DP.refresh()

    predictions = torch.cat(predictions).detach().numpy()
    targets = torch.cat(targets).detach().numpy()

    return model, DP, (predictions, targets)


@logging_time
def eval(model, loss_fns, DP, CFG, fold=None):

    dset = MyDataset(CFG, augment=False, fold=fold)
    dataloader = DataLoader(dset, batch_size=CFG.batch_size)

    device = CFG.device
    
    model.eval()
    predictions, targets = [], []
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            if CFG.resize:
                x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

            else:
                x, y = x.to(device), y.to(device)

            y_pred = model.forward(x).to(device).squeeze(1)
            predictions.append(y_pred.cpu())
            targets.append(y.cpu())

            loss = loss_fns['mse'](y_pred, y)

            # Track down results
            DP.loss.batch_update(loss.item(), 1)
            DP.mae.batch_update(loss_fns['mae'](y_pred, y).item(), 1)
            DP.rmse.batch_update(loss_fns['rmse'](y_pred, y).item(), 1)

            if CFG.lamb:
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += CFG.lamb * l2_reg

            del x, y, y_pred

    torch.cuda.empty_cache()
    DP.loss.clear()
    DP.mae.clear()
    DP.rmse.clear()
    DP.refresh()

    predictions = torch.cat(predictions).detach().numpy()
    targets = torch.cat(targets).detach().numpy()
    
    return model, DP, (predictions, targets)
