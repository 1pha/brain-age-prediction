from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Dict
import os
import json

import captum.attr as ca
import numpy as np
import torch
from torch import nn
import omegaconf

import sage
from sage.trainer import PLModule
from .lrp.innvestigator import InnvestigateModel
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
                 baseline: bool = False,
                 atlas: str = "dkt",
                 xai_init_kwarg: dict = None,
                 xai_call_kwarg: dict = None,
                 ############################
                 test_loader: torch.utils.data.DataLoader = None,
                 predict_loader: torch.utils.data.DataLoader = None,
                 log_train_metrics: bool = False,
                 manual_lr: bool = False,
                 augmentation: omegaconf.DictConfig = None,
                 scheduler: omegaconf.DictConfig = None,
                 load_model_ckpt: str = None,
                 load_from_checkpoint: str = None,
                 separate_lr: dict = None,
                 save_dir: Path = None):
        super().__init__(model=model,
                         train_loader=train_loader,
                         valid_loader=valid_loader,
                         optimizer=optimizer,
                         metrics=metrics,
                         test_loader=test_loader,
                         predict_loader=predict_loader,
                         log_train_metrics=log_train_metrics,
                         manual_lr=manual_lr,
                         augmentation=augmentation,
                         scheduler=scheduler,
                         load_model_ckpt=load_model_ckpt,
                         load_from_checkpoint=load_from_checkpoint,
                         separate_lr=separate_lr,
                         save_dir=save_dir)
        self.init_transforms(augmentation=augmentation)

        self.smaller_mask = utils.margin_mni_mask(return_pt=True)
        self.target_layer_index = target_layer_index
        self.top_k_percentile = top_k_percentile
        self.xai_method = xai_method
        self.baseline = baseline
        
        self.configure_xai(model=self.model, xai_method=xai_method,
                           target_layer_index=target_layer_index,
                           xai_init_kwarg=xai_init_kwarg, xai_call_kwarg=xai_call_kwarg)
        
        self.atlas = A.get_atlas(atlas_name=atlas)

    def setup(self, stage):
        super().setup(stage)
        if self.baseline and isinstance(self.baseline, np.ndarray):
            self.baseline = torch.tensor(self.no_augment(self.baseline[None, ...]))

    def _configure_xai(self,
                       model: nn.Module | Callable,
                       xai_method: str = "gbp",
                       target_layer_index: int = -1,
                       xai_init_kwarg: dict = None,
                       xai_call_kwarg: dict = None):
        forward_func = model._forward
        if xai_method == "gcam":
            xai = ca.LayerGradCam(forward_func=forward_func,
                                  layer=model.conv_layers()[target_layer_index])
        elif xai_method == "ggcam":
            xai = ca.GuidedGradCam(model=model.backbone, layer=model.conv_layers()[target_layer_index])
        elif xai_method == "gcam_avg":
            xai = [ca.LayerGradCam(forward_func=forward_func, layer=layer) for layer in model.conv_layers()[-20:]] # TODO
        elif xai_method == "ggcam_avg":
            xai = [ca.GuidedGradCam(model=model.backbone, layer=layer) for layer in model.conv_layers()[-20:]] # TODO
        elif xai_method == "gradxinput":
            xai = ca.InputXGradient(forward_func=model.backbone)
        elif xai_method == "deeplift":
            xai = ca.DeepLift(model=model.backbone)
        elif xai_method == "deepliftshap":
            xai = ca.DeepLiftShap(model=model.backbone)
        elif xai_method == "gbp":
            xai = ca.GuidedBackprop(model=model.backbone)
        elif xai_method == "deconv":
            xai = ca.Deconvolution(model=model.backbone)
        elif xai_method == "ig":
            xai = ca.IntegratedGradients(forward_func=forward_func)

        elif xai_method == "lrp":
            # XXX: Device setup is not done in lightning module instantiation
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            lrp_default_kwarg = dict(lrp_exponent=2, method="b-rule", beta=.5, device=device)
            if xai_init_kwarg is None:
                xai_init_kwarg = lrp_default_kwarg
            elif isinstance(xai_init_kwarg, dict):
                xai_init_kwarg.update(**lrp_default_kwarg)
            xai = InnvestigateModel(model.backbone, **xai_init_kwarg)

        elif xai_method.startswith("smooth"):
            if xai_method == "smooth_gbp":
                attr_mtd = ca.GuidedBackprop(model=model.backbone)
            elif xai_method == "smooth_gi":
                attr_mtd = ca.InputXGradient(forward_func=model.backbone)
            xai = ca.NoiseTunnel(attribution_method=attr_mtd)
            if xai_call_kwarg is None:
                xai_call_kwarg = dict(nt_type="smoothgrad", nt_samples=10)

        else:
            breakpoint()
        self.xai_call_kwarg = dict() if xai_call_kwarg is None else xai_call_kwarg
        logger.info("Start XAI Inference of %s", xai_method.upper())
        return xai

    def configure_xai(self,
                      model: nn.Module,
                      xai_method: str = "gbp",
                      target_layer_index: int = 1,
                      xai_init_kwarg: dict = None,
                      xai_call_kwarg: dict = None) -> None:
        name = model.NAME.lower()
        if name in ["resnet10", "resnet18", "resnet34", "convnext-tiny", "convnext-small", "convnext-base"]:
            self.xai = self._configure_xai(model=self.model, xai_method=xai_method,
                                           target_layer_index=target_layer_index,
                                           xai_init_kwarg=xai_init_kwarg,
                                           xai_call_kwarg=xai_call_kwarg)

        elif name == "swin_vit":
            if xai_method != "gcam":
                self.xai = self._configure_xai(model=self.model, xai_method=xai_method,
                                               target_layer_index=target_layer_index,
                                               xai_init_kwarg=xai_init_kwarg,
                                               xai_call_kwarg=xai_call_kwarg)
            else:
                breakpoint()

    def upsample(self,
                 tensor: torch.Tensor,
                 target_shape: tuple = None,
                 interpolate_mode: str = "trilinear",
                 return_np: bool = False,
                 apply_margin_mask: bool = True) -> np.ndarray | torch.Tensor:
        B = tensor.size(0)
        if B > 1:
            upsampled = []
            for brain in tensor:
                brain = brain.unsqueeze(dim=0)
                brain = self._upsample(tensor=brain, target_shape=target_shape,
                                       interpolate_mode=interpolate_mode, return_np=return_np,
                                       apply_margin_mask=apply_margin_mask)
                upsampled.append(torch.from_numpy(brain))
            upsampled = torch.stack(upsampled, dim=0)
        else:
            upsampled = self._upsample(tensor=tensor, target_shape=target_shape,
                                       interpolate_mode=interpolate_mode, return_np=return_np,
                                       apply_margin_mask=apply_margin_mask)
        return upsampled

    def _upsample(self,
                  tensor: torch.Tensor,
                  target_shape: tuple = None,
                  interpolate_mode: str = "trilinear",
                  return_np: bool = True,
                  apply_margin_mask: bool = True) -> torch.Tensor | np.ndarray:
        """ Upsamples a given tensor to target_shape
        through captum.attrs.LayerAttribution.interpolate """
        if apply_margin_mask:
            assert self.smaller_mask.shape == target_shape,\
                "Given target_shape is not same as a smaller mask saved as attribute: "\
                f"target_shape: {target_shape} | smaller_mask: {self.smaller_mask.shape}"
        
        target_shape = target_shape or C.MNI_SHAPE
        # upsampled: (B, C, H', W', D')
        upsampled = ca.LayerAttribution.interpolate(layer_attribution=tensor,
                                                    interpolate_dims=target_shape,
                                                    interpolate_mode=interpolate_mode)
        # upsampled: (B, H', W', D')
        upsampled = upsampled.cpu().detach().squeeze()
        if apply_margin_mask:
            upsampled *= self.smaller_mask
            upsampled = torch.nan_to_num(x=upsampled, nan=0.0) # inplace
        if return_np:
            upsampled = upsampled.numpy()
        return upsampled

    def attribute(self, brain: torch.Tensor) -> torch.Tensor:
        # For integrated gradients with baseline (average brain) given.
        # brain: (B, C, H, W, D)
        if hasattr(self, "xai_call_kwarg"):
            attr_kwargs = self.xai_call_kwarg
        if self.xai_method == "ig" and self.baseline:
            attr_kwargs = attr_kwargs["baselines"] = self.baseline.to(self.device)
            
        if hasattr(brain, "as_tensor"):
            brain = brain.as_tensor()

        if self.xai_method in ["gcam_avg", "ggcam_avg"]:
            attrs = []
            # GCAM_AVG will have list of attributers
            for xai in self.xai:
                attr = xai.attribute(brain)
                attr: torch.Tensor = utils.z_norm(attr)
                attr: torch.Tensor = self.upsample(tensor=attr, target_shape=C.MNI_SHAPE,
                                                 interpolate_mode="trilinear",
                                                 return_np=False, apply_margin_mask=True)
                attrs.append(attr)
            attr = torch.stack(attrs, dim=0).mean(dim=0)
        else:
            attr: torch.Tensor = self.xai.attribute(brain, **attr_kwargs) # (B, C, H, W, D)
            attr: torch.Tensor = utils.z_norm(attr) # (B, C, H, W, D)
            attr: torch.Tensor = self.upsample(tensor=attr, target_shape=C.MNI_SHAPE,
                                             interpolate_mode="trilinear",
                                             return_np=False, apply_margin_mask=True)
        return attr

    def calculate_overlaps(self,
                           attr: torch.Tensor,
                           atlas: ao.Bunch,
                           device: str | torch.device) -> Dict[str, List[float] | float]:
        B = attr.size(0)
        if (attr.ndim > 3) and (B > 1):
            # Multi-batch
            xai_dict: Dict[str, List[float]] = defaultdict(list)
            for _attr in attr:
                _xai_dict, _ = ao.calculate_overlaps(arr=_attr, atlas=atlas, use_torch=True, device=device,
                                                    plot_raw_sal=False, plot_bargraph=False, plot_projection=False)
                for k in _xai_dict:
                    val = float(_xai_dict[k])
                    xai_dict[k].append(val)
        else:
            xai_dict, _ = ao.calculate_overlaps(arr=attr, atlas=atlas, use_torch=True, device=device,
                                                plot_raw_sal=False, plot_bargraph=False, plot_projection=False)
            xai_dict = {k: float(v) for k, v in xai_dict.items()}
        return xai_dict

    def forward(self, batch: dict, mode: str = "test") -> Dict[str, torch.Tensor | dict]:
        try:
            # augmentor = self.augmentor if mode == "train" else self.no_augment
            aug = getattr(self, f"{'train' if mode == 'train' else 'valid'}_transforms")
            brain = aug(batch["brain"]) # (B, C, H, W, D)

            # Calculate Attribute & Get Top-k attribute
            attr: torch.Tensor = self.attribute(brain=brain) # (B, H', W', D') or (H', W', D')
            top_attr: np.ndarray = utils.top_q(arr=attr, q=self.top_k_percentile, use_abs=True,
                                               return_bool=False) # (B, H', W', D') or (H', W', D')
            # When attr is batch-inferred
            if attr.ndim == 4:
                attr = attr.sum(dim=0)
            if top_attr.ndim == 4:
                top_attr = top_attr.sum(dim=0)

            # Get projection list
            xai_dict = self.calculate_overlaps(attr=attr, atlas=self.atlas, device=brain.device)
            top_xai_dict = self.calculate_overlaps(attr=top_attr, atlas=self.atlas, device=brain.device)
            return dict(attr=attr, top_attr=top_attr, xai_dict=xai_dict, top_xai_dict=top_xai_dict)

        except Exception as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e) 
            breakpoint()

    def on_predict_start(self) -> None:
        """ Initialize attribute """
        self.attr = torch.zeros_like(self.smaller_mask)
        self.top_attr = torch.zeros_like(self.smaller_mask)
        # dict: {roi1: [X1, X2, ... Xn],
        #        roi2: [X2, X2, ... Xn], ...}
        self.xai_dict = defaultdict(list)
        self.top_xai_dict = defaultdict(list)

    def predict_step(self,
                     batch: dict,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> np.ndarray:
        result: dict = self.forward(batch, mode="test")

        self.attr += result["attr"]
        self.top_attr += result["top_attr"]
        for k in result["xai_dict"]:
            val = result["xai_dict"][k]
            if isinstance(val, float):
                self.xai_dict[k].append(val)
            elif isinstance(val, list):
                self.xai_dict[k].extend(val)

        for k in result["top_xai_dict"]:
            val = result["top_xai_dict"][k]
            if isinstance(val, float):
                self.top_xai_dict[k].append(val)
            elif isinstance(val, list):
                self.top_xai_dict[k].extend(val)
        # This is a hack to make lightning work
        return torch.zeros(size=(1,), requires_grad=True)

    def on_predict_end(self) -> None:
        # XXX: Note that we divide by len(dataset) NOT len(dataloader).
        # This is possible since we summated attr/top_attr one-by-one even in multi-batch cases
        self.attr /= len(self.predict_dataloader.dataset)
        self.top_attr /= len(self.predict_dataloader.dataset)

    def save_result(self, root_dir: Path):
        logger.info("Start saving here: %s", root_dir)
        os.makedirs(name=root_dir, exist_ok=True)

        # Save attrs
        np.save(file=root_dir / "attrs.npy", arr=self.attr)
        np.save(file=root_dir / "top_attr.npy", arr=self.top_attr)

        # Save plots
        nilp_.plot_glass_brain(arr=self.attr, save=root_dir / "attr_glass.png", colorbar=True)
        nilp_.plot_overlay(arr=self.attr, save=root_dir / "attr_anat.png", display_mode="mosaic")

        nilp_.plot_glass_brain(arr=self.top_attr, save=root_dir / "top_glass.png", colorbar=True)
        nilp_.plot_overlay(arr=self.top_attr, save=root_dir / "top_anat.png", display_mode="mosaic")

        # Save Individual Projection Result
        with (root_dir / "xai_dict_indiv.json").open(mode="w") as f:
            json.dump(obj=self.xai_dict, fp=f, indent="\t")
        with (root_dir / "top_xai_dict_indiv.json").open(mode="w") as f:
            json.dump(obj=self.top_xai_dict, fp=f, indent="\t")

        # Save Total Projection Result
        xai_dict, agg_saliency = ao.calculate_overlaps(arr=self.top_attr, atlas=self.atlas,
                                                       use_torch=True, root_dir=root_dir,
                                                       title=root_dir.stem)
        with (root_dir / "xai_dict.json").open(mode="w") as f:
            json.dump(obj=xai_dict, fp=f, indent="\t")
