import torch
import torch.nn as nn
from sklearn.metrics import r2_score

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

def get_metric(y_pred, y, metric: str):

    metric = metric.lower()
    if metric == 'r2':
        return r2_score(y, y_pred)
    return {
        'mse': nn.MSELoss(),
        'rmse': RMSELoss(),
        'mae': nn.L1Loss()
    }[metric](y_pred, y).item()


if __name__=="__main__":

    y_pred = torch.tensor([1, 2, 3], dtype=torch.float16)
    y_true = torch.tensor([4, 5, 6], dtype=torch.float16)

    print(get_metric(y_pred, y_true, 'r2'))