import torch
from torch import nn

from sage.utils import get_logger
from .utils import find_conv_modules


logger = get_logger(name=__file__)


class ModelBase(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 criterion: nn.Module,
                 name: str,
                 task: str = "reg"):
        super().__init__()
        logger.info("Start Initiating model %s", name.upper())
        self.backbone = backbone
        self.criterion = criterion
        self.NAME = name
        self.TASK = task

    def forward(self, brain: torch.Tensor, age: torch.Tensor):
        pred = self.backbone(brain).squeeze()
        loss = self.criterion(pred, age)
        return dict(loss=loss,
                    reg_pred=pred.detach().cpu(),
                    reg_target=age.detach().cpu())
        
    def _forward(self, brain: torch.Tensor):
        return self.backbone(brain)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_from_checkpoint(self, ckpt: str):
        ckpt = torch.load(ckpt)["state_dict"]
        def parse_ckpt(s: str):
            s = ".".join(s.split(".")[1:])
            return s
        ckpt = {parse_ckpt(k): v for k, v in ckpt.items()}
        self.load_state_dict(ckpt)
        
    def conv_layers(self):
        if hasattr(self.backbone, "conv_layers"):
            return self.backbone.conv_layers()
        else:
            return find_conv_modules(self.backbone)


class ResNet(ModelBase):
    def __init__(self,
                 backbone: nn.Module,
                 criterion: nn.Module,
                 name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)


class ConvNext(ModelBase):
    def __init__(self,
                 backbone: nn.Module,
                 criterion: nn.Module,
                 name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)
