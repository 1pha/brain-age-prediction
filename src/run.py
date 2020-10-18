from itertools import chain

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

    for e in epochs:

        if scheduler: scheduler.step()

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

            trn_bth_loss += loss.item()

            if i % 20 == 0:
                print(f'{i:<4}th Batch. Loss: {loss.item():.3f}')

        torch.cuda.empty_cache()
        trn_trues = np.array(list(chain(*trn_trues)))
        trn_preds = (np.exp(list(chain(*trn_preds))) >= .5) * 1
        trn_losses.append(trn_bth_loss / len(train_loader))

        # TEST
        tst_bth_loss = 0
        model.eval()
        tst_trues, tst_preds = [], []
        with torch.no_grad():
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

        tst_losses.append(tst_bth_loss / len(test_loader))
        tst_trues = np.array(list(chain(*tst_trues)))
        tst_preds = (np.exp(list(chain(*tst_preds))) >= .5) * 1
        torch.cuda.empty_cache()

        if summary:
            summary.add_scalars('loss/MSE_loss',
                                {'Train Loss': trn_losses[-1],
                                 'Valid Loss': tst_losses[-1]}, e)

            summary.add_pr_curve('pr_curve/train', trn_trues, trn_preds, e)
            summary.add_pr_curve('pr_curve/test', tst_trues, tst_preds, e)

            if scheduler:
                summary.add_scalar('lr', scheduler.get_last_lr(), e)

        if verbose:
            print(f'EPOCHS {e} | TRAIN :: [LOSS] {trn_losses[-1]:.3f} | VALID :: [LOSS] {tst_losses[-1]:.3f}')
            print(f'[TRAIN - REPORT]\n{classification_report(trn_trues, trn_preds)}')
            print(f'[TEST  - REPORT]\n{classification_report(tst_trues, tst_preds)}')

    return model, (trn_losses, tst_losses)