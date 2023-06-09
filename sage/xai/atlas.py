import ast
from pathlib import Path
from typing import Tuple, TypeVar

import numpy as np
import nibabel
from nilearn.image import load_img
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_harvard_oxford
import pandas as pd
from sklearn.utils import Bunch

from .utils import upsample, MNI_SHAPE


nii = TypeVar(name="nii", bound=nibabel.nifti1.Nifti1Image)


def get_cerebra():
    cerebra = get_atlas(atlas_name="cerebra",
                        return_mni=False)
    return cerebra


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
    else:
        arr = atlas_map.get_fdata()
    indices = _literal_eval(lst=indices)
    assert len(indices) == len(labels),\
        f"# Labels and indices should be same: len(indices)={len(indices)}, len(labels)={len(labels)}"
    return Bunch(array=arr, indices=indices, labels=labels, nii=atlas_map)
        
    
def _load_map(maps: str | nii) -> nii:
    if isinstance(maps, str | Path):
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


def _get_cerebra(fname: str = "mni_icbm152_CerebrA_tal_nlin_sym_09c.nii",
                 label_fname: str = "CerebrA_LabelDetails.csv",
                 root: Path = Path("assets/mni_icbm152_nlin_sym_09c_CerebrA_nifti")) -> Tuple[nii, list, list]:
    
    fname = root / fname
    atlas_map = _load_map(maps=fname)
    
    # Process Labels
    labels = pd.read_csv(root / label_fname)
    
    LABEL_COL, INDEX_COL = "Label Name", "Index"
    rhl = labels[[LABEL_COL, "RH Label"]]
    rhl[LABEL_COL] = rhl["Label Name"].apply(lambda s: f"{s}_R")
    rhl = rhl.rename({"RH Label": INDEX_COL}, axis=1)

    lhl = labels[[LABEL_COL, "LH Labels"]]
    lhl[LABEL_COL] = lhl["Label Name"].apply(lambda s: f"{s}_L")
    lhl = lhl.rename({"LH Labels": INDEX_COL}, axis=1)
    
    labels = pd.concat([rhl, lhl], axis=0).reset_index(drop=True)
    indices = labels[INDEX_COL].values.tolist()
    labels = labels[LABEL_COL].values.tolist()
    
    return atlas_map, indices, labels

    
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

