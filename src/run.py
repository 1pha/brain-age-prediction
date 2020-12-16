from itertools import chain
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as cf
from sklearn.metrics import classification_report

import torch
import torch.nn.functional as F

def run(model, epochs, train_loader, test_loader,
        optimizer, loss_fn, device,
        resize=64, summary=None, scheduler=None, verbose=True):

    trn_losses, tst_losses = [], []
    best_acc = 0

    for e in epochs:

        # TRAIN
        trn_bth_loss = 0
        trn_trues, trn_preds = [], []
        model.train()
        for i, (x, y) in enumerate(train_loader):

            if resize:
                x, y = F.interpolate(x, size=(64, 64, 64)).to(device), y.to(device)
            
            else:
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model.forward(x).to(device)

            trn_trues.append(y.to('cpu'))
            trn_preds.append(y_pred.to('cpu'))

            loss = loss_fn(y_pred.squeeze(1), y)
            del x, y, y_pred

            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            trn_bth_loss += loss.item()

            if i % 20 == 0:
                print(f'{i:<4}th Batch. Loss: {loss.item():.3f}')


        torch.cuda.empty_cache()
        # COLLECT TRAIN RESULTS
        ### loss
        trn_losses.append(trn_bth_loss / len(train_loader))

        ### collect trues/predictions
        # TODO:: trn_trues doesn't need to be calcaulted like this(since it is a real true value...)
        trn_trues = np.array(list(chain(*trn_trues)))
        trn_preds = (np.array(list(chain(*trn_preds))) >= .5) * 1

        ### calculate accuracy and classification report
        trn_acc = sum(trn_trues==trn_preds) / len(trn_trues)
        trn_rep = classification_report(trn_trues, trn_preds, target_names=["Old", "Young"])
        

        # TEST
        tst_bth_loss = 0
        model.eval()
        tst_trues, tst_preds = [], []
        with torch.no_grad(): # to not give loads on GPU... :(
            for i, (x, y) in enumerate(test_loader):
                if resize:
                    x, y = F.interpolate(x, size=(64, 64, 64)).to(device), y.to(device)
            
                else:
                    x, y = x.to(device), y.to(device)

                y_pred = model.forward(x).to(device)

                tst_trues.append(y.to('cpu'))
                tst_preds.append(y_pred.to('cpu'))

                loss = loss_fn(y_pred.squeeze(1), y)
                del x, y, y_pred

                tst_bth_loss += loss.item()

        torch.cuda.empty_cache()
        # COLLECT TRAIN RESULTS
        ### loss
        tst_losses.append(tst_bth_loss / len(test_loader))

        ### collect trues/predictions
        tst_trues = np.array(list(chain(*tst_trues)))
        tst_preds = (np.array(list(chain(*tst_preds))) >= .5) * 1

        ### calculate accuracy and classification report
        tst_acc = sum(tst_trues==tst_preds) / len(tst_trues)
        tst_rep = classification_report(tst_trues, tst_preds, target_names=["Old", "Young"])


        # to tensorboard
        if summary:
            summary.add_scalars('loss/BCE_loss',
                                {'Train Loss': trn_losses[-1],
                                 'Valid Loss': tst_losses[-1]}, e)
            summary.add_scalars('acc/accuracy',
                                {'Train Acc': trn_acc,
                                 'Valid Acc': tst_acc}, e)

            summary.add_pr_curve('pr_curve/train', trn_trues, trn_preds, 0)
            summary.add_pr_curve('pr_curve/test', tst_trues, tst_preds, 0)

            if scheduler:
                summary.add_scalar('lr', scheduler.get_last_lr(), e)


        # save when valid accuracy hits the best
        if best_acc + .02 < tst_acc:
            date = f'{datetime.now().strftime("%Y-%m-%d_%H%M")}'
            fname = f"./models/{date}_{tst_acc:.3f}_model.pth"
            torch.save(model, fname)
            best_acc = max(tst_acc, best_acc)


        # print results
        if verbose:
            print(f'EPOCHS {e}')
            print(f'TRAIN :: [LOSS] {trn_losses[-1]:.3f} | VALID :: [LOSS] {tst_losses[-1]:.3f}')
            print(f'TRAIN :: [ACC%] {trn_acc:.3f} | VALID :: [ACC%] {tst_acc:.3f}')
            print(f'TRAIN :: [REPORT]\n{trn_rep}')
            print(f'VALID :: [REPORT]\n{tst_rep}')
            print(f'BEST ACC FOR TEST :: {best_acc}')

    return model, (trn_losses, tst_losses), (trn_trues, trn_preds), (tst_trues, tst_preds)

def run_reg(model, epochs, train_loader, test_loader,
            optimizer, loss_fn, device,
            resize=64, summary=None, scheduler=None, verbose=True):

    
    trn_losses, tst_losses = [], []
    best_loss = 100
    for e in epochs:

        # TRAIN
        trn_bth_loss = 0
        trn_trues, trn_preds = [], []
        model.train()
        for i, (x, y) in enumerate(train_loader):

            if resize:
                x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

            else:
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model.forward(x).to(device)

            trn_trues.append(y.to('cpu'))
            trn_preds.append(y_pred.to('cpu'))

            loss = torch.sqrt(loss_fn(y_pred.squeeze(1), y))
            del x, y, y_pred

            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            trn_bth_loss += loss.item()

        torch.cuda.empty_cache()
        
        ### loss
        trn_losses.append(trn_bth_loss / len(train_loader))

        ### collect trues/predictions
        trn_trues = list(chain(*trn_trues))
        trn_preds = list(chain(*trn_preds))

            
        # TEST
        tst_bth_loss = 0
        model.eval()
        tst_trues, tst_preds = [], []
        with torch.no_grad(): # to not give loads on GPU... :(
            for i, (x, y) in enumerate(test_loader):
                if resize:
                    x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

                else:
                    x, y = x.to(device), y.to(device)

                y_pred = model.forward(x).to(device)

                tst_trues.append(y.to('cpu'))
                tst_preds.append(y_pred.to('cpu'))

                loss = torch.sqrt(loss_fn(y_pred.squeeze(1), y))
                del x, y, y_pred

                tst_bth_loss += loss.item()

        torch.cuda.empty_cache()
        ### loss
        tst_losses.append(tst_bth_loss / len(test_loader))

        ### collect trues/predictions
        tst_trues = list(chain(*tst_trues))
        tst_preds = list(chain(*tst_preds))
        
        reg_df = pd.DataFrame({
            'True': list(map(float, trn_trues + tst_trues)),
            'Prediction': list(map(float, trn_preds + tst_preds)),
            'Label': ['train'] * 250 + ['test'] * 62
        })

        if verbose:

            print(f'EPOCHS {e}')
            print(f'RMSE :: [TRAIN] {trn_losses[-1]:.3f} | [VALID] {tst_losses[-1]:.3f}')
            
            sns.lmplot(data=reg_df, x='True', y='Prediction', hue='Label')
            plt.grid()
            plt.show()

            if e % 20 == 0:
                plt.plot(trn_losses, label='Train')
                plt.plot(tst_losses, label='Valid')
                plt.title(f"RMSE Losses among epochs, {e}th")
                plt.grid()
                plt.legend()
            
        if best_loss - .02 > tst_losses[-1]:
            
            date = f'{datetime.now().strftime("%Y-%m-%d_%H%M")}'
            fname = f"./models/{date}_{tst_losses[-1]:.3f}_model.pth"
            torch.save(model, fname)
            best_loss = min(tst_losses[-1], best_loss)

        if summary:
            summary.add_scalars('loss/RMSE_loss',
                                {'Train Loss': trn_losses[-1],
                                 'Valid Loss': tst_losses[-1]}, e)

def run_folds(model, epochs, train_loader, test_loader,
              optimizer, loss_fn, device, folds,
              resize=64, summary=None, scheduler=None, verbose=True):

    best_loss = 10

    trn_fold_losses, tst_fold_losses = [], []
    trn_fold_corrs, tst_fold_corrs = [], []
    for fold in folds:
        
        train_dset = MyDataset(task_type='age', fold=fold)
        train_loader = DataLoader(train_dset, batch_size=8)
        
        trn_losses, tst_losses = [], []
        for e in epochs:

            # TRAIN
            trn_bth_loss = 0
            trn_trues, trn_preds = [], []
            model.train()
            for i, (x, y) in enumerate(train_loader):

                if resize:
                    x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

                else:
                    x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                y_pred = model.forward(x).to(device)

                trn_trues.append(y.to('cpu'))
                trn_preds.append(y_pred.to('cpu'))

                loss = loss_fn(y_pred.squeeze(1), y)
                del x, y, y_pred

                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()

                trn_bth_loss += loss.item()

            torch.cuda.empty_cache()

            ### loss
            trn_losses.append(trn_bth_loss / len(train_loader))

            ### collect trues/predictions
            trn_trues = list(chain(*trn_trues))
            trn_preds = list(chain(*trn_preds))


            # TEST
            tst_bth_loss = 0
            model.eval()
            tst_trues, tst_preds = [], []
            with torch.no_grad(): # to not give loads on GPU... :(
                for i, (x, y) in enumerate(test_loader):
                    if resize:
                        x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

                    else:
                        x, y = x.to(device), y.to(device)

                    y_pred = model.forward(x).to(device)

                    tst_trues.append(y.to('cpu'))
                    tst_preds.append(y_pred.to('cpu'))

                    loss = loss_fn(y_pred.squeeze(1), y)
                    del x, y, y_pred

                    tst_bth_loss += loss.item()

            torch.cuda.empty_cache()
            ### loss
            tst_losses.append(tst_bth_loss / len(test_loader))

            ### collect trues/predictions
            tst_trues = list(chain(*tst_trues))
            tst_preds = list(chain(*tst_preds))

            reg_df = pd.DataFrame({
                'True': list(map(float, trn_trues + tst_trues)),
                'Prediction': list(map(float, trn_preds + tst_preds)),
                'Label': ['train'] * len(trn_trues) + ['test'] * len(tst_trues)
            })

            trn_corr = reg_df[reg_df['Label'] == 'train'].corr().Prediction['True']
            tst_corr = reg_df[reg_df['Label'] == 'test' ].corr().Prediction['True']

            print(f'FOLD {fold}')
            print(f'EPOCHS {e}')
            print(f'RMSE :: [TRAIN] {trn_losses[-1]:.3f} | [VALID] {tst_losses[-1]:.3f}')
            print(f'CORR :: [TRAIN] {trn_corr:.3f} | [VALID] {tst_corr:.3f}')

            sns.lmplot(data=reg_df, x='True', y='Prediction', hue='Label')
            plt.grid()
            plt.show()

            if e % 20 == 0:
                plt.plot(trn_losses, label='Train')
                plt.plot(tst_losses, label='Valid')
                plt.title(f"L1 Losses among epochs, {e}th")
                #plt.ylim(0, 500)
                plt.grid()
                plt.legend()
        
        trn_fold_losses.append(trn_losses)
        trn_fold_corrs.append(trn_corr)
        tst_fold_losses.append(tst_losses)
        tst_fold_corrs.append(tst_corr)

    return model, (trn_fold_losses, tst_fold_losses), (trn_fold_corrs, tst_fold_corrs)


def make_df(data, label):
    
    trues, preds = data
    return pd.DataFrame({
        'True': list(map(float, trues)),
        'Prediction': list(map(float, preds)),
        'Label': [label] * len(trues)
    })

def train(model, dataloader, resize, device,
          loss_fn, mae_fn, rmse_fn,
          losses, maes, rmses,
          optimizer, scheduler, lamb):
    
    bth_loss, bth_mae, bth_rmse = 0, 0, 0
    trues, preds = [], []
    model.train()
    for i, (x, y) in enumerate(dataloader):

        if resize:
            x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

        else:
            x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred = model.forward(x).to(device)

        trues.append(y.to('cpu'))
        preds.append(y_pred.to('cpu'))

        # Loss
        loss = loss_fn(y_pred.squeeze(1), y)
        
        if lamb:
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += lamb * l2_reg
        
        # Metrics
        mae = mae_fn(y_pred.squeeze(1), y)
        rmse = rmse_fn(y_pred.squeeze(1), y)

        del x, y, y_pred

        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()

        bth_loss += loss.item()
        bth_mae  += mae.item()
        bth_rmse += rmse.item()

    torch.cuda.empty_cache()

    ### loss
    M = len(dataloader)
    losses.append(bth_loss / M)
    maes.append(bth_mae / M)
    rmses.append(bth_rmse / M)

    ### collect trues/predictions
    trues = list(chain(*trues))
    preds = list(chain(*preds))
    
    return model, (losses, maes, rmses), (trues, preds)

def eval(model, dataloader, resize, device,
          loss_fn, mae_fn, rmse_fn,
        losses, maes, rmses):
    
    bth_loss, bth_mae, bth_rmse = 0, 0, 0
    trues, preds = [], []
    model.eval()
    with torch.no_grad(): # to not give loads on GPU... :(
        for i, (x, y) in enumerate(dataloader):

            if resize:
                x, y = F.interpolate(x, size=(96, 96, 96)).to(device), y.to(device)

            else:
                x, y = x.to(device), y.to(device)

            y_pred = model.forward(x).to(device)

            trues.append(y.to('cpu'))
            preds.append(y_pred.to('cpu'))

            # Loss
            loss = loss_fn(y_pred.squeeze(1), y)

            # Metrics
            mae = mae_fn(y_pred.squeeze(1), y)
            rmse = rmse_fn(y_pred.squeeze(1), y)

            del x, y, y_pred

            bth_loss += loss.item()
            bth_mae  += mae.item()
            bth_rmse += rmse.item()

    torch.cuda.empty_cache()

    ### loss
    M = len(dataloader)
    losses.append(bth_loss / M)
    maes.append(bth_mae / M)
    rmses.append(bth_rmse / M)

    ### collect trues/predictions
    trues = list(chain(*trues))
    preds = list(chain(*preds))
    
    return model, (losses, maes, rmses), (trues, preds)

def info(state, fold, epoch, loss, mae, rmse, corr):

    print(f'FOLD {fold}', end='')
    print(f'MSE  :: [TEST] {loss[-1]:.3f}')
    print(f'MAE  :: [TEST] {mae[-1]:.3f}')
    print(f'RMSE :: [TEST] {rmse[-1]:.3f}')
    print(f'CORR :: [TEST] {corr:.3f}')
