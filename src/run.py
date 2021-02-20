import time
from itertools import chain
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F
import torch.optim as optim
from .data_util import * 
from .losses import RMSELoss, fn_lst
from .architectures.model_util import load_model, save_checkpoint

import sys
sys.path.append('../')
from DeepNotion.build import write_db


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

def run(cfg, fold, db=None, mlflow=None):

    model, cfg.device = load_model(cfg.model_name, verbose=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trn_dp, aug_dp, tst_dp = DataPacket(), DataPacket(), DataPacket()
    aug_dp.delete('corr')

    best_mae = cfg.best_mae
    fold = None
    for e in range(cfg.epochs):
        
        start_time = time.time()
        print(f'Epoch {e+1} / {cfg.epochs}, BEST MAE {best_mae:.3f}')
        cfg.test = False
        model, trn_dp, trn_res = train(model, optimizer, fn_lst, trn_dp, cfg, fold=fold)
        model, aug_dp, aug_res = train(model, optimizer, fn_lst, aug_dp, cfg, fold=fold, augment=True)
        cfg.test = True
        model, tst_dp, tst_res = eval(model, fn_lst, tst_dp, cfg, fold=fold)
        elapsed_time = round(time.time() - start_time, 3)
        
        if best_mae > tst_dp.mae[-1]:
            
            best_mae = tst_dp.mae[-1]
            model_name = f'{cfg.model_name}_ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{best_mae:.3f}.pt'
            save_checkpoint(cfg.get_dict(), model_name, is_best=True)
            
        df = pd.concat([make_df(trn_res, 'Train'),
                        make_df(aug_res, 'Aug'),
                        make_df(tst_res, 'Valid')], ignore_index=True)
        
        trn_dp.corr.update(df[df['Label'] == 'Train'].corr().Prediction['True'])
        trn_dp.refresh()
        tst_dp.corr.update(df[df['Label'] == 'Valid'].corr().Prediction['True'])
        tst_dp.refresh()

        if e % 1 == 0:
            trn_dp.info('train')
            aug_dp.info('augme')
            tst_dp.info('valid')

        if e % 5 == 0:
            plt.title(f"L1 Losses among epochs, {e}th")
            plt.plot(list(trn_dp.loss), label='Train')
            plt.plot(list(tst_dp.loss), label='Valid')
            plt.grid(); plt.legend()

            sns.lmplot(data=df, x='True', y='Prediction', hue='Label')
            plt.grid()
            plt.show()

            if db is not None:
                data = gather_data(e=e, time=elapsed_time, cfg=cfg,
                                    train=trn_dp, valid=tst_dp, aug=aug_dp)
                write_db(db, data)
            
        if mlflow:
            metrics = mlflow_data(time=elapsed_time, train=trn_dp, valid=tst_dp, aug=aug_dp)
            mlflow.log_metrics(metrics, e)

        torch.cuda.empty_cache()

    mlflow.end_run()
    return model, (trn_dp, aug_dp, tst_dp), (trn_res, tst_res)

@logging_time
def train(model, optimizer, loss_fns, DP, CFG, fold=None, augment=False):

    dset = MyDataset(CFG, augment=augment, fold=fold)
    dataloader = DataLoader(dset, batch_size=CFG.batch_size)

    device = CFG.device

    model.train()
    predictions, targets = [], []
    for i, (x, y) in enumerate(dataloader):

        try: 
            if CFG.resize:
                x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

            else:
                x, y = x.to(device), y.to(device)

        except FileNotFoundError as e:
            print(e)
            time.sleep(20)
            pass

        optimizer.zero_grad()

        y_pred = model.forward(x).to(device).squeeze(1)
        predictions.append(y_pred.cpu())
        targets.append(y.cpu())

        loss = loss_fns['mse'](y_pred, y)

        if i % 8 == 0 and cfg.debug:
            fig, ax = plt.subplots()
            ax.set_title(f"Augment {i}, MAE {loss_fns['mae'](y_pred, y).item()}")
            ax.imshow(x.detach().cpu()[0][0][:, 48, :].T, cmap='gray', origin='lower')
            plt.show()

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
def valid(model, loss_fns, DP, CFG, fold=None):

    dset = MyDataset(CFG, augment=False, fold=fold)
    dataloader = DataLoader(dset, batch_size=CFG.batch_size)

    device = CFG.device
    
    model.eval()
    predictions, targets = [], []
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            try: 
                if CFG.resize:
                    x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

                else:
                    x, y = x.to(device), y.to(device)

            except FileNotFoundError as e:
                print(e)
                time.sleep(20)
                pass

            y_pred = model.forward(x).to(device).squeeze(1)
            predictions.append(y_pred.cpu())
            targets.append(y.cpu())

            loss = loss_fns['mse'](y_pred, y)

            if i % 3 == 0 and cfg.debug:
                fig, ax = plt.subplots()
                ax.set_title(f"Eval {i}, MAE {loss_fns['mae'](y_pred, y).item()}")
                ax.imshow(x.detach().cpu()[0][0][:, 48, :].T, cmap='gray', origin='lower')
                plt.show()
             
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