import torch
import numpy as np
import nilearn.plotting as nilp
from nilearn.datasets import load_mni152_template

from sage.utils import get_logger
from .utils import _mni, _nifti


logger = get_logger(name=__file__)


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
                 use_mni: bool = True, scale_factor: float = 1e+5,
                 save: str = None, alpha: float = 0.7, **kwargs):
    arr = np.abs(_tensorfy(arr)) * scale_factor
    nifti = _mni if use_mni else _nifti
    display = nilp.plot_anat(anat_img=load_mni152_template(), **kwargs)
    display.add_overlay(nifti(arr), alpha=alpha)
    return display, save