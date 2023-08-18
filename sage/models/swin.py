import torch
from torch import nn

from .base import ModelBase
from sage.utils import get_logger


logger = get_logger(name=__name__)

SWIN_CKPT = "./assets/weights/model_swinvit.pt"

class SwinViT(ModelBase):
    """ SwinViT from MONAI
    backbone: _target_=monai.networks.nets.swin_unetr.SwinTransformer
    """
    def __init__(self,
                 backbone: nn.Module,
                 criterion: nn.Module,
                 name: str = "swin_vit",
                 pretrained: bool = True):
        super().__init__(backbone=backbone, criterion=criterion, name=name)
        if pretrained:
            self.load_pretrained(ckpt=SWIN_CKPT)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=4),
            nn.Linear(768, 1)
        )
    
    def load_pretrained(self, ckpt: str):
        ckpt = torch.load(SWIN_CKPT)
        def parse(s: str):
            if ".fc" in s:
                s = s.replace(".fc", ".linear")
            return s.split("module.")[1]
        exclude = {'contrastive_head.bias',
                    'contrastive_head.weight',
                    'convTrans3d.bias',
                    'convTrans3d.weight',
                    'norm.bias',
                    'norm.weight',
                    'rotation_head.bias',
                    'rotation_head.weight'}
        ckpt = {parse(k): v for k, v in ckpt["state_dict"].items()
                if parse(k) not in exclude}
        self.backbone.load_state_dict(ckpt)
        logger.info("Successfully loaded checkpoint")
        
    def forward(self, brain: torch.Tensor, age: torch.Tensor = None):
        # Resulting image: (batch_size, num_channels, 3, 3, 3)
        x_out = self.backbone(brain)[-1]
        pred = self.pool(x_out).squeeze()
        if age is None:
            # For GuidedBackprop
            return pred.unsqueeze(dim=0)
        else:
            loss = self.criterion(pred, age)
            
            return dict(loss=loss,
                        reg_pred=pred.detach().cpu(),
                        reg_target=age.detach().cpu())
