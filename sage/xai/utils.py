from pathlib import Path

import torch
import nibabel
import numpy as np
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from captum.attr import LayerAttribution


MNI_AFFINE = load_mni152_template().affine
MNI_SHAPE = load_mni152_brain_mask().get_fdata().shape


def load_np(fname: str | np.ndarray | Path):
    if isinstance(fname, Path | str):
        arr = np.abs(np.load(fname))
    elif isinstance(fname, np.ndarray):
        arr = np.abs(fname)
    elif fname is False:
        return None

    while arr.ndim > 3:
        arr = arr[0]
    return arr


def top_q(arr: np.ndarray,
          q: float = 0.95,
          use_abs: bool = True,
          return_bool: bool = False):
    arr = load_np(arr)

    mask = ~np.isnan(arr)
    arr = arr * mask

    mask = arr > np.nanquantile(np.abs(arr) if use_abs else arr, q=q)
    return mask.astype(np.int32) if return_bool else arr * mask


def z_norm(tensor) -> torch.Tensor:
    if tensor.ndim == 5: # (N, 1, 96, 96, 96)
        z_normed = torch.cat([_z_norm(t[0])[None, None, ...] for t in tensor], dim=0)
    elif tensor.ndim == 4: # (1, 96, 96, 96)
        z_normed = _z_norm(tensor[0])
    elif tensor.ndim == 3:
        z_normed = _z_norm(tensor)
    return z_normed


def _z_norm(tensor: torch.tensor):
    mu = torch.nanmean(tensor)
    sigma = torch.std(tensor)
    return (tensor - mu) / sigma


def __z_norm(tensor):
    assert tensor.ndim == 3, f"Give 3-dimensional tensor. Given {tensor.shape}."
    mu = np.nanmean(tensor)
    sigma = np.nanstd(tensor)
    return (tensor - mu) / sigma


def boolify(arr: np.ndarray) -> np.ndarray:
    bool_mask = ~np.isnan(arr) & (arr > 0)
    return bool_mask


def average(tensor, dim=0):
    N = tensor.shape[dim]
    return tensor.sum(dim) / N


def margin_mni_mask():
    """ This loads smaller brain mask of mni
    Why do we need smaller mask?
        - Since some models watch dura maters on the edges
          or some brains not having proper skull-stripping,
          it was critical to delete some of margins (2 voxels) from edge 
    """
    mask = load_mni152_brain_mask().get_fdata()
    mni_shape = mask.shape
    _int = 1
    s = tuple([shape + _int * 2 for shape in mni_shape])
    smaller_mask = np.zeros(s)

    smaller_mask[_int:s[0]-_int, _int:s[1]-_int, _int:s[2]-_int] = mask
    smaller_mask = LayerAttribution.interpolate(torch.tensor(smaller_mask)[None, None, ...],
                                                mni_shape,
                                                interpolate_mode="trilinear")[0][0].numpy()
    smaller_mask[smaller_mask < 0.5] = np.nan
    smaller_mask[smaller_mask > 0.5] = 1.
    return smaller_mask


def upsample(arr: np.ndarray | torch.Tensor,
             return_mni: bool = True,
             target_shape: tuple = MNI_SHAPE,
             interpolate_mode="trilinear") -> np.ndarray:
    """ If you're translating atlas, it is recommended to use 'nearest' as interpolation_mode"""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
        
    while arr.ndim > 3:
        arr = arr[0]
    
    if return_mni:
        target_shape = MNI_SHAPE
    
    arr = torch.tensor(arr)
    arr = LayerAttribution.interpolate(layer_attribution=arr[None, None, ...],
                                       interpolate_dims=MNI_SHAPE,
                                       interpolate_mode=interpolate_mode)[0][0].numpy()
    return arr


_nifti = lambda arr: nibabel.nifti1.Nifti1Image(arr, np.eye(4))
_mni = lambda arr: nibabel.nifti1.Nifti1Image(arr, MNI_AFFINE)