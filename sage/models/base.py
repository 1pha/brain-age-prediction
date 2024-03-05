import torch
from torch import nn

from sage.utils import get_logger
from .utils import find_conv_modules
from .model_zoo.sfcn import num2vect


logger = get_logger(name=__file__)


class ModelBase(nn.Module):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__()
        logger.info("Start Initiating model %s", name.upper())
        # self.backbone = torch.compile(backbone)
        self.backbone = backbone
        self.criterion = criterion
        self.NAME = name

    def _forward(self, brain: torch.Tensor):
        return self.backbone(brain)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_from_checkpoint(self, ckpt: str):
        ckpt = torch.load(ckpt)["state_dict"]
        def parse_ckpt(s: str):
            # This is to remove "model." prefix from pytorch_lightning
            s = ".".join(s.split(".")[1:])
            return s
        ckpt = {parse_ckpt(k): v for k, v in ckpt.items()}
        self.load_state_dict(ckpt)

    def conv_layers(self):
        if hasattr(self.backbone, "conv_layers"):
            return self.backbone.conv_layers()
        else:
            return find_conv_modules(self.backbone)


class ClsBase(ModelBase):
    def forward(self, brain: torch.Tensor, age: torch.Tensor):
        pred = self.backbone(brain).squeeze()
        loss = self.criterion(pred, age.long())
        return dict(loss=loss, pred=pred.detach().cpu(), target=age.detach().cpu().long())


class RegBase(ModelBase):
    def forward(self, brain: torch.Tensor, age: torch.Tensor):
        pred = self.backbone(brain).squeeze()
        loss = self.criterion(pred, age.float())
        return dict(loss=loss, pred=pred.detach().cpu(), target=age.detach().cpu())


class ResNet(RegBase):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)


class ConvNext(RegBase):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)


class ResNetCls(RegBase):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)


class ConvNextCls(RegBase):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)


class SFCNModel(ModelBase):
    def __init__(self, backbone: nn.Module, criterion: nn.Module, name: str):
        super().__init__(backbone=backbone, criterion=criterion, name=name)
        # TODO: bin_range interval and backbones' output_dim should be matched,
        # but they are separately hard-coded!
        self.num2vect_kwargs = dict(bin_range=(40, 100), bin_step=1, sigma=1)
    
    def forward(self, brain: torch.Tensor, age: torch.Tensor):
        _pred = self.backbone(brain)
        age_y, bc = num2vect(age.cpu().numpy(), **self.num2vect_kwargs)
        
        device = brain.device
        loss = self.criterion(_pred, torch.tensor(age_y, device=device))
        
        pred = _pred.cpu().clone().detach().exp().numpy() @ bc
        pred = torch.tensor(pred)
        return dict(loss=loss,
                    pred=pred.detach().cpu(),
                    target=age.detach().cpu())
