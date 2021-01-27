import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

fn_lst = {
    'mse': nn.MSELoss(),
    'rmse': RMSELoss(),
    'mae': nn.L1Loss()
}