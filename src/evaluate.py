from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch

def eval(model, loader, device):

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):

            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x).to(device)

            trues.append(y.to('cpu'))
            preds.append(y_pred.to('cpu'))

            del x, y, y_pred
            torch.cuda.empty_cache()

    return trues, preds


def loss_plot(trn_losses, tst_losses, epochs, loss_type):

    plt.plot(trn_losses)
    plt.plot(tst_losses)
    plt.legend(['Train Loss', 'Val Loss'])
    plt.title('Loss / Epochs')
    plt.xlabel('Epochs')
    step = np.arange(0, epochs.stop, step=int(epochs.stop/10) if int(epochs.stop/10) else 1)
    plt.xticks(np.arange(0, epochs.stop, step=step),
               [x for x in np.arange(0, epochs.stop, step=step)])
    plt.ylabel(f'{loss_type} Loss')
    plt.grid()
    plt.show()


def result_plot(task_type, trues, preds, title='Train', fname=None):

    preds = np.exp(list(chain(*preds)))
    trues = np.array(list(chain(*trues)))

    if task_type == 'reg':
        cut = max(max(abs(preds)), max(abs(trues)))
        # cuts = [-cut, cut]
        cuts = [0, cut]

        plt.figure(figsize=(9, 9))
        plt.scatter(preds, trues)
        plt.xlim(*cuts)
        plt.ylim(*cuts)
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.title(f'{title} Dataset Prediction')
        plt.grid()
        domain = np.linspace(*cuts, 100)
        plt.plot(domain, domain, c='black')
        if fname:
            plt.savefig(f'./result/{fname}_{title}_reg.png')
        plt.show()

    elif task_type == 'binary':

        pass