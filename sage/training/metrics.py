import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class confusion_loss(nn.Module):
    def __init__(self, task=0):
        super(confusion_loss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss

class accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return torch.sum(y_pred == y) / len(y_pred)

def get_metric(y_pred, y, metric: str):

    '''
    y_pred, y: given with CPU, gradient DETACHED TORCH TENSOR
    '''

    metric = metric.lower()
    if metric == 'r2':
        return r2_score(y.numpy(), y_pred.numpy())
    return {
        'mse': nn.MSELoss(),
        'rmse': RMSELoss(),
        'mae': nn.L1Loss(),
        'corr': lambda _p, _t: pearsonr(_p, _t)[0],
        'ce': nn.CrossEntropyLoss(),
        'confusion': confusion_loss(),
        'acc': accuracy(),
    }[metric](y_pred, y)


if __name__=="__main__":

    y_pred = torch.tensor([1, 2, 3], dtype=torch.float16)
    y_true = torch.tensor([4, 5, 6], dtype=torch.float16)

    print(get_metric(y_pred, y_true, 'r2'))