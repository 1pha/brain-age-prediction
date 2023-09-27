import copy
import gc
from pathlib import Path
from warnings import warn

import torch
import nibabel as nib
import numpy as np
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import new_img_like
from captum.attr import LayerAttribution

from sage.utils import profile

try:
    import sage.constants as C
except ImportError:
    import meta_brain.router as C


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


def top_q(arr: np.ndarray, q: float = 0.95, use_abs: bool = True, return_bool: bool = False):
    if use_abs:
        quantile_value = np.nanquantile(np.abs(arr), q=q)
    else:
        quantile_value = np.nanquantile(arr, q=q)

    mask = arr > quantile_value
    if return_bool:
        return mask.astype(np.int32)
    else:
        return arr * mask


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


def align(arr: np.ndarray, affine: np.ndarray = C.MNI_AFFINE) -> nib.nifti1.Nifti1Image:
    _arr: np.ndarray = _safe_get_data(_mni(arr), ensure_finite=True)
    arr_nifti = new_img_like(ref_niimg=_mni(arr), data=_arr, affine=affine)
    return arr_nifti


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
             target_shape: tuple = C.MNI_SHAPE,
             interpolate_mode="trilinear") -> np.ndarray:
    """ If you're translating atlas, it is recommended to use 'nearest' as interpolation_mode"""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
        
    while arr.ndim > 3:
        arr = arr[0]
    
    if return_mni:
        target_shape = C.MNI_SHAPE
    
    arr = torch.tensor(arr)
    arr = LayerAttribution.interpolate(layer_attribution=arr[None, None, ...],
                                       interpolate_dims=target_shape,
                                       interpolate_mode=interpolate_mode)[0][0].numpy()
    return arr


def _get_data(img):
    # copy-pasted from
    # https://github.com/nipy/nibabel/blob/de44a105c1267b07ef9e28f6c35b31f851d5a005/nibabel/dataobj_images.py#L204 # noqa
    #
    # get_data is removed from nibabel because:
    # see https://github.com/nipy/nibabel/wiki/BIAP8
    if img._data_cache is not None:
        return img._data_cache
    data = np.asanyarray(img._dataobj)
    img._data_cache = data
    return data


def _safe_get_data(img, ensure_finite=False, copy_data=False):
    """Get the data in the image without having a side effect \
    on the Nifti1Image object.

    Parameters
    ----------
    img: Nifti image/object
        Image to get data.

    ensure_finite: bool
        If True, non-finite values such as (NaNs and infs) found in the
        image will be replaced by zeros.

    copy_data: bool, default is False
        If true, the returned data is a copy of the img data.

    Returns
    -------
    data: numpy array
        nilearn.image.get_data return from Nifti image.
    """
    if copy_data:
        img = copy.deepcopy(img)

    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()

    data = _get_data(img)
    if ensure_finite:
        non_finite_mask = np.logical_not(np.isfinite(data))
        if non_finite_mask.sum() > 0:  # any non_finite_mask values?
            warn(
                "Non-finite values detected. "
                "These values will be replaced with zeros."
            )
            data[non_finite_mask] = 0

    return data


_nifti = lambda arr, affine=np.eye(4): nib.nifti1.Nifti1Image(arr, affine)
_mni = lambda arr: nib.nifti1.Nifti1Image(arr, C.MNI_AFFINE)
