import torch
import numpy as np

def run(model, epochs, train_loader, test_loader,
        optimizer, loss_fn, device, summary=None, verbose=True):

    trn_losses, tst_losses = [], []

    for e in epochs:

        trn_bth_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model.forward(x).to(device)

            loss = loss_fn(y_pred, y)
            del x, y, y_pred

            loss.backward()
            optimizer.step()

            trn_bth_loss += loss.item()

            if i % 20 == 0:
                print(f'{i:<4}th Batch. Loss: {loss.item():.3f}')

        torch.cuda.empty_cache()
        trn_losses.append(trn_bth_loss / len(train_loader))

        tst_bth_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)

                y_pred = model.forward(x).to(device)

                loss = loss_fn(y_pred, y)
                del x, y, y_pred

                tst_bth_loss += loss.item()

        tst_losses.append(tst_bth_loss / len(test_loader))
        torch.cuda.empty_cache()

        if summary:
            summary.add_scalars('loss/MSE_loss',
                                {'Train Loss': trn_losses[-1],
                                 'Valid Loss': tst_losses[-1]}, e)

        if verbose:
            print(f'EPOCHS {e} | TRAIN :: [LOSS] {trn_losses[-1]:.3f} | VALID :: [LOSS] {tst_losses[-1]:.3f}')

    return model, (trn_losses, tst_losses)