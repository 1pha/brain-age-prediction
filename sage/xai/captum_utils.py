from functools import partial
from pathlib import Path

from captum.attr import LayerGradCam, LayerAttribution
import numpy as np
from hydra.utils import instantiate as inst
import omegaconf
import monai.transforms as mt
import nilearn.plotting as nilp
import torch
from torch.utils.data import Dataset

from sage.trainer.utils import load_mask
from sage.data import no_augment
from .utils import load_mni152_template, _nifti, _mni, MNI_SHAPE, margin_mni_mask


smaller_mask = margin_mni_mask()


def get_brain(idx: int,
              aug: mt.Compose,
              dataset: Dataset,
              verbose: bool = True):
    batch = dataset[idx]
    if verbose:
        print("age: ", batch["age"])
    brain = aug(batch["brain"][None, ...])
    age = torch.tensor(batch["age"])
    return brain, age


def read_config(weight_path: Path) -> omegaconf.DictConfig:
    config_file = weight_path / ".hydra" / "config.yaml"
    with open(config_file, mode="r") as f:
        conf = omegaconf.OmegaConf.load(f)
    return conf


def instantiate_xai(config: omegaconf.OmegaConf):
    """ Use `xai` key """
    weight_path = Path(config.weight_path)
    config = read_config(weight_path)
    
    dataset: Dataset = inst(config.dataset, mode="test")
    model: torch.nn.Module = inst(config.model)
    mask: torch.Tensor = load_mask(mask_path="assets/mask.npy",
                                   mask_threshold=config.module.mask_threshold)
    augmentor: mt.Compose = no_augment(spatial_size=(96, 96, 96), mask=mask)
    _get_brain: callable = partial(get_brain, dataset=dataset, aug=augmentor)
    
    for idx in range(len(dataset)):
        brain, age = _get_brain(idx)
        
        
def get_attr(model: torch.nn.Module,
             brain: torch.Tensor,
             age: torch.Tensor,
             **kwargs) -> callable:
    _get_attr = {
        "resnet": get_resnet_attr,
        "swin_vit": get_swinvit_attr,
    }[model.NAME]
    
            
def get_swinvit_attr():
    return


def get_resnet_attr(model: torch.nn.Module,
                    brain: torch.Tensor,
                    age: torch.Tensor,
                    target_layer_index: int = 1) -> np.ndarray:
    
    result = model(brain=brain, age=age)
    
    nilp.plot_anat(_nifti(brain[0][0].numpy()))
    
    target_layer_index = 1
    layer_gc = LayerGradCam(model, model.conv_layers()[target_layer_index])

    attr = layer_gc.attribute(brain)
    upsampled_attr = (
        LayerAttribution.interpolate(
            attr, MNI_SHAPE, interpolate_mode="trilinear"
        )
        .cpu()
        .detach()
    ).numpy() * smaller_mask
    return upsampled_attr