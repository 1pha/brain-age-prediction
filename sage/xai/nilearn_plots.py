import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.plotting as nilp
import nibabel as nib
from nilearn.datasets import load_mni152_template
import torch

from sage.utils import get_logger
from .utils import _mni, _nifti
from .atlas import get_atlas


logger = get_logger(name=__file__)

UNIT_Y = 25 / 116


def load_affine(target_affine: str | np.ndarray = "cerebra") -> np.ndarray:
    if isinstance(target_affine, str):
        atlas = get_atlas(atlas_name=target_affine,
                          return_mni=False if target_affine == "cerebra" else True)
        target_affine = atlas.nii.affine
    elif isinstance(target_affine, np.ndarray):
        assert target_affine.ndim == 2, "`target_affine` should be 2-dimensional 4*4 matrix"
        assert target_affine.shape == (4, 4), "`target_affine` should be 2-dimensional 4*4 matrix"
    return target_affine


def check_nii(arr: np.ndarray | torch.Tensor | nib.nifti1.Nifti1Image,
              target_mni: bool = True,
              target_affine: np.ndarray = None):
    """ Returns nii for a given array """
    if isinstance(arr, np.ndarray | torch.Tensor):
        arr = _tensorfy(arr=arr)
        if target_mni and target_affine is None:
            # If target_affine is given, this should be ignored
            nifti = _mni(arr)
        else:
            assert target_affine is not None, f"Please provide target_affine. Given {target_affine}"
            nifti = _nifti(arr=arr, affine=target_affine)
    elif isinstance(arr, nib.nifti1.Nifti1Image):
        nifti = arr
    else:
        raise
    return nifti


def _tensorfy(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    """ Convert torch.tensor or numpy array into 3-dim numpy array"""
    # torch.tensor given
    if isinstance(arr, torch.Tensor):
        while arr.ndim > 3:
            arr = arr.squeeze()
        arr: np.ndarray = arr.cpu().detach().numpy()
    elif isinstance(arr, np.ndarray):
        while arr.ndim > 3:
            arr = arr[0]
    else:
        logger.warn("Provide torch.tensor or numpy ndarray")
        raise
    return arr


def plot_brain(arr: torch.Tensor | np.ndarray,
               use_mni: bool = False,
               save: str = None, **kwargs):
    arr = _tensorfy(arr)
    nifti = _mni if use_mni else _nifti
    display = nilp.plot_anat(anat_img=nifti(arr),
                             output_file=save, **kwargs)
    return display, save


def plot_overlay(arr: torch.Tensor | np.ndarray,
                 use_mni: bool = True,
                 target_affine: np.ndarray = None,
                 bg: np.ndarray | nib.nifti1.Nifti1Image = None,
                 display_mode: str = "ortho",
                 save: str = None, alpha: float = 0.7, **kwargs):    
    """ Works similar with `nilp.plot_roi`
    However this first plots an original brain
    and overlays with `nilp.add_overlay`. """
    arr = check_nii(arr=arr, target_mni=use_mni, target_affine=target_affine)
    if bg is None:
        bg = load_mni152_template()
    title = kwargs.pop("title", "")
    display = nilp.plot_anat(anat_img=bg, display_mode=display_mode, title=title)
    display.add_overlay(arr, alpha=alpha, **kwargs)
    if save is not None:
        display.savefig(save)
    return display, save


def plot_glass_brain(arr: torch.Tensor | np.ndarray,
                     use_mni: bool = True,
                     target_affine: np.ndarray = None,
                     save: str = None, **kwargs):
    arr = check_nii(arr=arr, target_mni=use_mni, target_affine=target_affine)
    display = nilp.plot_glass_brain(arr, output_file=save, **kwargs)
    return display, save


def plot_roi(roi_img: np.ndarray | nib.nifti1.Nifti1Image,
             bg_img: np.ndarray | nib.nifti1.Nifti1Image,
             target_affine: str | np.ndarray = "cerebra",
             display_mode: str = "mosaic", **kwargs):
    target_affine = load_affine(target_affine=target_affine)
    roi_img = check_nii(roi_img, target_affine=target_affine)
    bg_img = check_nii(bg_img, target_affine=target_affine)
    
    dp = nilp.plot_roi(roi_img=roi_img, bg_img=bg_img, display_mode=display_mode, **kwargs)
    return dp

def brain_barplot(xai_dict: dict,
                  title: str = "",
                  sort_values: bool = True,
                  save: str = None) -> None:
    fig, ax = plt.subplots(figsize=(12, UNIT_Y * len(xai_dict)))
    
    ax.set_title(title)
    df = pd.DataFrame({"Regions": xai_dict.keys(),
                       "Saliency": xai_dict.values()})
    if sort_values:
        df = df.sort_values(by="Saliency")
    sns.barplot(data=df, x="Saliency", y="Regions", ax=ax)
    fig.savefig(fname=save)
