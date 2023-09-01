from pathlib import Path
from typing import Callable

import omegaconf
import numpy as np
import torch
from torch import nn
from captum.attr import (
    LayerAttribution,
    GuidedBackprop,
    IntegratedGradients,
    LayerGradCam,
    LRP
)

import sage
from sage.trainer import PLModule
from sage.constants import MNI_SHAPE
from .utils import margin_mni_mask, z_norm, top_q, load_np


logger = sage.utils.get_logger(name=__name__)


class XPLModule(PLModule):
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: omegaconf.DictConfig,
                 metrics: dict,
                 ### Additional Arguments ###
                 target_layer_index: int = -2,
                 top_k_percentile: float = 0.95,
                 xai_method: str = "gbp",
                 baseline: bool = False,
                 ############################
                 mask: Path | str | torch.Tensor = None,
                 mask_threshold: float = 0.1,
                 test_loader: torch.utils.data.DataLoader = None,
                 predict_loader: torch.utils.data.DataLoader = None,
                 log_train_metrics: bool = False,
                 augmentation: omegaconf.DictConfig = None,
                 scheduler: omegaconf.DictConfig = None,
                 load_model_ckpt: str = None,
                 load_from_checkpoint: str = None,
                 separate_lr: dict = None):

        super().__init__(model,
                         train_loader,
                         valid_loader,
                         optimizer,
                         metrics,
                         mask,
                         mask_threshold,
                         test_loader,
                         predict_loader,
                         log_train_metrics,
                         augmentation,
                         scheduler,
                         load_model_ckpt,
                         load_from_checkpoint,
                         separate_lr)

        # Dataloader sanity check
        if self.predict_dataloader:
            assert self.predict_dataloader.batch_size == 1, "Predict dataloader should have batch_size=1 for XPL"
        if self.test_dataloader:
            assert self.test_dataloader.batch_size == 1, "Test dataloader should have batch_size=1 for XPL"

        self.smaller_mask = margin_mni_mask()
        self.target_layer_index = target_layer_index
        self.top_k_percentile = top_k_percentile
        self.xai_method = xai_method
        self.baseline = load_np(fname=baseline)
        
        self.configure_xai(model=self.model,
                           xai_method=xai_method,
                           target_layer_index=target_layer_index)

    def setup(self, stage):
        super().setup(stage)
        if self.baseline is not None and isinstance(self.baseline, np.ndarray):
            self.baseline = torch.tensor(self.no_augment(self.baseline[None, ...]))
        
    def _configure_xai(self,
                       model: nn.Module | Callable,
                       xai_method: str = "gbp",
                       target_layer_index: int = 1):
        forward_func = model._forward
        if xai_method == "gcam":
            xai = LayerGradCam(forward_func=model.forward_func,
                               layer=model.conv_layers()[target_layer_index])
        elif xai_method == "gbp":
            xai = GuidedBackprop(model=forward_func)
        elif xai_method == "ig":
            xai = IntegratedGradients(forward_func=forward_func)
        elif xai_method == "lrp":
            xai = LRP(model=model)
        else:
            breakpoint()
        return xai
        
    def configure_xai(self,
                      model: nn.Module,
                      xai_method: str = "gbp",
                      target_layer_index: int = 1) -> None:
        name = model.NAME.lower()
        if name in ["resnet10t", "convnext-base", "convnext-tiny"]:
            # self.model = model.backbone
            self.xai = self._configure_xai(model=self.model,
                                           xai_method=xai_method,
                                           target_layer_index=target_layer_index)

        elif name == "swin_vit":
            if xai_method != "gcam":
                self.xai = self._configure_xai(model=self.model,
                                               xai_method=xai_method,
                                               target_layer_index=target_layer_index)
            else:
                breakpoint()
                
    def upsample(self,
                 tensor: torch.Tensor,
                 target_shape: tuple = None,
                 interpolate_mode: str = "trilinear",
                 return_np: bool = True,
                 apply_margin_mask: bool = True) -> np.ndarray | torch.Tensor:
        """ Upsamples a given tensor to target_shape
        through captum.attrs.LayerAttribution.interpolate """
        if apply_margin_mask:
            assert self.smaller_mask.shape == target_shape,\
                "Given target_shape is not same as a smaller mask saved as attribute: "\
                f"target_shape: {target_shape} | smaller_mask: {self.smaller_mask.shape}"
        
        target_shape = target_shape or MNI_SHAPE
        upsampled = LayerAttribution.interpolate(layer_attribution=tensor,
                                                 interpolate_dims=target_shape,
                                                 interpolate_mode=interpolate_mode)
        upsampled = upsampled.cpu().detach().squeeze()
        if return_np:
            upsampled = upsampled.numpy()
        if apply_margin_mask:
            assert return_np
            upsampled *= self.smaller_mask
        return upsampled

    def forward(self, batch: dict, mode: str = "test") -> dict:
        try:
            augmentor = self.augmentor if mode == "train" else self.no_augment
            brain = torch.tensor(augmentor(batch["brain"]))
            if self.xai_method == "ig" and self.baseline is not None:
                attr_kwargs = dict(baselines=self.baseline.to(self.device))
            else:
                attr_kwargs = dict()
            
            attr: torch.Tensor = self.xai.attribute(brain, **attr_kwargs)
            attr: torch.Tensor = z_norm(attr)
            attr: np.ndarray = self.upsample(tensor=attr,
                                             target_shape=MNI_SHAPE,
                                             interpolate_mode="trilinear",
                                             return_np=True, apply_margin_mask=True)
            top_attr: np.ndarray = top_q(arr=attr,
                                         q=self.top_k_percentile,
                                         use_abs=True,
                                         return_bool=False)
            while attr.ndim > 3:
                attr = attr[0]
            while top_attr.ndim > 3:
                top_attr = top_attr[0]
            return {
                "attr": attr,
                "top_attr": top_attr
            }

        except RuntimeError as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e)
            breakpoint()
            raise e
    
    def on_predict_start(self) -> None:
        """ Initialize attribute """
        self.attr = np.zeros(shape=self.smaller_mask.shape)
        self.top_attr = np.zeros(shape=self.smaller_mask.shape)
        
    def predict_step(self,
                     batch: dict,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> np.ndarray:
        attrs: np.ndarray = self.forward(batch, mode="test")
        self.attr += attrs["attr"]
        self.top_attr += attrs["top_attr"]
        # This is a hack to make lightning work
        return torch.zeros(size=(1,), requires_grad=True)

    def on_predict_end(self) -> np.ndarray:
        self.attr /= len(self.predict_dataloader)
        self.top_attr /= len(self.predict_dataloader)