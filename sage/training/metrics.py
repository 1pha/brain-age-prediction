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
        'corr': lambda _p, _t: pearsonr(_p, _t)[0]
    }[metric](y_pred, y)


if __name__=="__main__":

    y_pred = torch.tensor([1, 2, 3], dtype=torch.float16)
    y_true = torch.tensor([4, 5, 6], dtype=torch.float16)

    print(get_metric(y_pred, y_true, 'r2'))