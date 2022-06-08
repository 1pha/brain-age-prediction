import torch
import nibabel
import numpy as np

def z_norm(tensor):

    if tensor.ndim == 5:
        z_normed = torch.cat([_z_norm(t[0])[None, None, ...] for t in tensor], dim=0)
        return z_normed

def _z_norm(tensor):

    assert tensor.ndim == 3, f"Give 3-dimensional tensor. Given {tensor.shape}."

    mu = tensor.mean()
    sigma = tensor.var().sqrt()

    return (tensor - mu) / sigma

def average(tensor, dim=0):

    N = tensor.shape[dim]
    return tensor.sum(dim) / N

_nifti = lambda arr: nibabel.nifti1.Nifti1Image(arr, np.eye(4))