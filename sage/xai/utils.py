import torch
import nibabel
import numpy as np
from nilearn.datasets import load_mni152_template

MNI_AFFINE = load_mni152_template().affine

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

_nifti = lambda arr: nibabel.nifti1.Nifti1Image(arr, np.eye(4))
_mni = lambda arr: nibabel.nifti1.Nifti1Image(arr, MNI_AFFINE)