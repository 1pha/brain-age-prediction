import ast
from pathlib import Path
from typing import Tuple, TypeVar

import numpy as np
from sklearn.utils import Bunch
import nibabel
from nilearn.image import load_img
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_harvard_oxford

from .utils import upsample, MNI_SHAPE


nii = TypeVar(name="nii", bound=nibabel.nifti1.Nifti1Image)


def get_atlas(atlas_name: str,
              atlas_kwargs: dict = {},
              return_mni: bool = True,
              target_shape: tuple = MNI_SHAPE,
              interpolate_mode: str = "nearest"):
    atlas_map, indices, labels = {
        "aal": _get_aal(),
        "oxford": _get_ho(**atlas_kwargs),
        "cerebra": _get_cerebra(),
    }[atlas_name]
    
    if return_mni:
        arr = upsample(arr=atlas_map.get_fdata(),
                       return_mni=return_mni,
                       target_shape=target_shape,
                       interpolate_mode=interpolate_mode)
    indices = _literal_eval(lst=indices)
    assert len(indices) == len(labels),\
        f"# Labels and indices should be same: len(indices)={len(indices)}, len(labels)={len(labels)}"
    return Bunch(array=arr, indices=indices, labels=labels, nii=atlas_map)
        
    
def _load_map(maps: str | nii) -> nii:
    if isinstance(maps, str):
        atlas_map = load_img(maps)
    elif isinstance(maps, nii.__bound__):
        # TODO: weird type checking
        atlas_map = maps
    else:
        raise
    return atlas_map


def _literal_eval(lst: list):
    def le(s: str | int):
        if isinstance(s, str):
            s = ast.literal_eval(s)
        else:
            pass
        return s
    lst = list(map(le, lst))
    return lst


def _get_cerebra():
    pass

    
def _get_ho(atlas_name="cortl-maxprob-thr50-1mm") -> Tuple[nii, list, list]:
    atlas = fetch_atlas_harvard_oxford(atlas_name=atlas_name)
    atlas_map = _load_map(maps=atlas.maps)
    
    indices = np.unique(atlas.maps.get_fdata())
    labels = atlas.labels
    return atlas_map, indices, labels
    
def _get_aal() -> Tuple[nii, list, list]:
    atlas = fetch_atlas_aal()
    atlas_map = _load_map(maps=atlas.maps)

    indices = atlas.indices    
    labels = atlas.labels
    return atlas_map, indices, labels

