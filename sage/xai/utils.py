import torch
import nibabel
import numpy as np
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from captum.attr import LayerAttribution


MNI_AFFINE = load_mni152_template().affine
MNI_SHAPE = load_mni152_brain_mask().get_fdata().shape


def z_norm(tensor):
    if tensor.ndim == 5: # (N, 1, 96, 96, 96)
        z_normed = torch.cat([_z_norm(t[0])[None, None, ...] for t in tensor], dim=0)
    elif tensor.ndim == 4: # (1, 96, 96, 96)
        z_normed = _z_norm(tensor[0])
    elif tensor.ndim == 3:
        z_normed = _z_norm(tensor)
    return z_normed


def _z_norm(tensor):
    assert tensor.ndim == 3, f"Give 3-dimensional tensor. Given {tensor.shape}."
    mu = np.nanmean(tensor)
    sigma = np.nanstd(tensor)
    return (tensor - mu) / sigma


def average(tensor, dim=0):
    N = tensor.shape[dim]
    return tensor.sum(dim) / N


def margin_mni_mask():
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


_nifti = lambda arr: nibabel.nifti1.Nifti1Image(arr, np.eye(4))
_mni = lambda arr: nibabel.nifti1.Nifti1Image(arr, MNI_AFFINE)