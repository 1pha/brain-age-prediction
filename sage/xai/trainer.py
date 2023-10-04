from collections import defaultdict
from pathlib import Path
from typing import Callable
import os
import json

import captum.attr as ca
import numpy as np
import torch
from torch import nn
import omegaconf

import sage
from sage.trainer import PLModule
from . import nilearn_plots as nilp_
from . import atlas_overlap as ao
from . import atlas as A
from . import utils
try:
    import sage.constants as C
except ImportError:
    import meta_brain.router as C



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
                 baseline: bool = None,
                 atlas: str = "dkt",
                 ############################
                 test_loader: torch.utils.data.DataLoader = None,
                 predict_loader: torch.utils.data.DataLoader = None,
                 log_train_metrics: bool = False,
                 augmentation: omegaconf.DictConfig = None,
                 scheduler: omegaconf.DictConfig = None,
                 load_model_ckpt: str = None,
                 load_from_checkpoint: str = None,
                 separate_lr: dict = None,
                 save_dir: Path = None):

        super().__init__(model,
                         train_loader,
                         valid_loader,
                         optimizer,
                         metrics,
                         test_loader,
                         predict_loader,
                         log_train_metrics,
                         augmentation,
                         scheduler,
                         load_model_ckpt,
                         load_from_checkpoint,
                         separate_lr,
                         save_dir)

        # Dataloader sanity check
        if self.predict_dataloader:
            assert self.predict_dataloader.batch_size == 1, "Predict dataloader should have batch_size=1 for XPL"
        if self.test_dataloader:
            assert self.test_dataloader.batch_size == 1, "Test dataloader should have batch_size=1 for XPL"

        self.smaller_mask = utils.margin_mni_mask()
        self.target_layer_index = target_layer_index
        self.top_k_percentile = top_k_percentile
        self.xai_method = xai_method
        self.baseline = baseline
        
        self.configure_xai(model=self.model,
                           xai_method=xai_method,
                           target_layer_index=target_layer_index)
        
        self.atlas = A.get_atlas(atlas_name=atlas)

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
            xai = ca.LayerGradCam(forward_func=forward_func,
                                  layer=model.conv_layers()[target_layer_index])
        elif xai_method == "gcam_avg":
            xai = [ca.LayerGradCam(forward_func=forward_func, layer=layer) for layer in model.conv_layers()[-20:]] # TODO
        elif xai_method == "gbp":
            xai = ca.GuidedBackprop(model=model.backbone)
        elif xai_method == "ig":
            xai = ca.IntegratedGradients(forward_func=forward_func)
        elif xai_method == "lrp":
            xai = ca.LRP(model=model.backbone)
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
        
        target_shape = target_shape or C.MNI_SHAPE
        upsampled = ca.LayerAttribution.interpolate(layer_attribution=tensor,
                                                    interpolate_dims=target_shape,
                                                    interpolate_mode=interpolate_mode)
        upsampled = upsampled.cpu().detach().squeeze()
        if return_np:
            upsampled = upsampled.numpy()
        if apply_margin_mask:
            assert return_np
            upsampled *= self.smaller_mask
            np.nan_to_num(x=upsampled, copy=False) # inplace
        return upsampled
    
    def attribute(self, brain: torch.Tensor) -> np.ndarray:
        # For integrated gradients with baseline (average brain) given.
        if self.xai_method == "ig" and self.baseline is not None:
            attr_kwargs = dict(baselines=self.baseline.to(self.device))
        else:
            attr_kwargs = dict()

        if self.xai_method == "gcam_avg":
            attrs = []
            # GCAM_AVG will have list of attributers
            for xai in self.xai:
                attr = xai.attribute(brain)
                attr: torch.Tensor = utils.z_norm(attr)
                attr: np.ndarray = self.upsample(tensor=attr, target_shape=C.MNI_SHAPE,
                                                 interpolate_mode="trilinear",
                                                 return_np=True, apply_margin_mask=True)
                attrs.append(attr)
            attrs = torch.from_numpy(attrs)
            attr = torch.stack(attrs).mean(dim=0)
        else:
            attr: torch.Tensor = self.xai.attribute(brain, **attr_kwargs)
            attr: torch.Tensor = utils.z_norm(attr)
            attr: np.ndarray = self.upsample(tensor=attr, target_shape=C.MNI_SHAPE,
                                             interpolate_mode="trilinear",
                                             return_np=True, apply_margin_mask=True)
        return attr

    def forward(self, batch: dict, mode: str = "test") -> dict:
        try:
            augmentor = self.augmentor if mode == "train" else self.no_augment
            brain = torch.tensor(augmentor(batch["brain"]))
            attr: np.ndarray = self.attribute(brain=brain)

            # Get projection list
            xai_dict, _ = ao.calculate_overlaps(arr=attr, atlas=self.atlas, use_torch=True, device=brain.device,
                                                plot_raw_sal=False, plot_bargraph=False, plot_projection=False)

            # Get top_attribute
            top_attr: np.ndarray = utils.top_q(arr=attr, q=self.top_k_percentile,
                                               use_abs=True, return_bool=False)

            while attr.ndim > 3:
                attr = attr[0]
            while top_attr.ndim > 3:
                top_attr = top_attr[0]

            return {"attr": attr,
                    "top_attr": top_attr,
                    "xai_dict": xai_dict}

        except Exception as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e)
            breakpoint()

    def on_predict_start(self) -> None:
        """ Initialize attribute """
        self.attr = np.zeros(shape=self.smaller_mask.shape)
        self.top_attr = np.zeros(shape=self.smaller_mask.shape)
        # dict: {roi1: [X1, X2, ... Xn],
        #        roi2: [X2, X2, ... Xn], ...}
        self.xai_dict = defaultdict(list)
        
    def predict_step(self,
                     batch: dict,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> np.ndarray:
        attrs: np.ndarray = self.forward(batch, mode="test")
        
        self.attr += attrs["attr"]
        self.top_attr += attrs["top_attr"]
        for k in attrs["xai_dict"]:
            self.xai_dict[k].append(attrs["xai_dict"][k])
        
        # This is a hack to make lightning work
        return torch.zeros(size=(1,), requires_grad=True)

    def on_predict_end(self) -> None:
        self.attr /= len(self.predict_dataloader)
        self.top_attr /= len(self.predict_dataloader)
        
    def save_result(self, root_dir: Path):
        logger.info("Start saving here %s", root_dir)
        os.makedirs(name=root_dir, exist_ok=True)

        # Save attrs
        np.save(file=root_dir / "attrs.npy", arr=self.attr)
        np.save(file=root_dir / "top_attr.npy", arr=self.top_attr)

        # Save plots
        nilp_.plot_glass_brain(arr=self.attr, save=root_dir / "attr_glass.png", colorbar=True)
        nilp_.plot_overlay(arr=self.attr, save=root_dir / "attr_anat.png", display_mode="mosaic")
        
        nilp_.plot_glass_brain(arr=self.top_attr, save=root_dir / "top_glass.png", colorbar=True)
        nilp_.plot_overlay(arr=self.top_attr, save=root_dir / "top_anat.png", display_mode="mosaic")
        
        breakpoint()
        # Save Individual Projection Result
        with (root_dir / "xai_dict_indiv.json").open(mode="w") as f:
            json.dump(obj=self.xai_dict, fp=f, indent="\t")
        
        # Save Total Projection Result
        xai_dict, agg_saliency = ao.calculate_overlaps(arr=self.top_attr, atlas=self.atlas,
                                                       root_dir=root_dir, title=root_dir.stem)
        with (root_dir / "xai_dict.json").open(mode="w") as f:
            json.dump(obj=xai_dict, fp=f, indent="\t")
