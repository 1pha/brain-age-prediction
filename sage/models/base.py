import torch
from torch import nn


class ModelBase(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 criterion: nn.Module,):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
    
    def forward(self, brain: torch.Tensor, age: torch.Tensor):
        pred = self.backbone(brain)
        loss = self.criterion(pred, age)
        
        return dict(loss=loss,
                    reg_pred=pred.detach().cpu(),
                    reg_target=age.detach().cpu())

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
