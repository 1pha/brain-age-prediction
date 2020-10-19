from itertools import chain
from datetime import datetime

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
        trn_preds = (np.array(list(chain(*trn_preds))) >= .5) * 1
        trn_acc = sum(trn_trues==trn_preds) / len(trn_trues)
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
        tst_preds = (np.array(list(chain(*tst_preds))) >= .5) * 1
        tst_acc = sum(tst_trues==tst_preds) / len(tst_trues)
        torch.cuda.empty_cache()

        if summary:
            summary.add_scalars('loss/BCE_loss',
                                {'Train Loss': trn_losses[-1],
                                 'Valid Loss': tst_losses[-1]}, e)
            summary.add_scalars('acc/accuracy',
                                {'Train Acc': sum(trn_trues==trn_preds) / len(trn_trues),
                                 'Valid Acc': sum(tst_trues==tst_preds) / len(tst_trues)}, e)

            summary.add_pr_curve('pr_curve/train', trn_trues, trn_preds, e)
            summary.add_pr_curve('pr_curve/test', tst_trues, tst_preds, e)

            if scheduler:
                summary.add_scalar('lr', scheduler.get_last_lr(), e)

        if best_acc + .02 < tst_acc:
            date = f'{datetime.now().strftime("%Y-%m-%d_%H%M")}'
            fname = f"./models/{date}_{tst_acc:.3f}_model.pth"
            torch.save(model, fname)
            best_acc = max(tst_acc, best_acc)

        if verbose:
            print(f'EPOCHS {e}')
            print(f'TRAIN :: [LOSS] {trn_losses[-1]:.3f} | VALID :: [LOSS] {tst_losses[-1]:.3f}')
            print(f'TRAIN :: [ACC%] {sum(trn_trues==trn_preds) / len(trn_trues):.3f} | VALID :: [ACC%] {sum(tst_trues==tst_preds) / len(tst_trues):.3f}')
            print(f'[TRAIN - REPORT]\n{classification_report(trn_trues, trn_preds, target_names=["Old", "Young"])}')
            print(f'[TEST  - REPORT]\n{classification_report(tst_trues, tst_preds, target_names=["Old", "Young"])}')

    return model, (trn_losses, tst_losses), (trn_trues, trn_preds), (tst_trues, tst_preds)