import ast
from pathlib import Path
from typing import Tuple, TypeVar, List
import types

import numpy as np
import nibabel
from nilearn.image import load_img
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_harvard_oxford
import pandas as pd
from sklearn.utils import Bunch

from . import utils
from sage.utils import get_logger
try:
    import meta_brain.router as C
except ImportError:
    import sage.constants as C


nii = TypeVar(name="nii", bound=nibabel.nifti1.Nifti1Image)
logger = get_logger(name=__file__)


def get_cerebra():
    cerebra = get_atlas(atlas_name="cerebra",
                        return_mni=False)
    return cerebra


def get_atlas(atlas_name: str,
              atlas_kwargs: dict = {},
              return_mni: bool = True,
              target_shape: tuple = C.MNI_SHAPE,
              interpolate_mode: str = "nearest"):
    logger.info("Load %s atlas.", atlas_name)
    atlas_map, indices, labels = {
        "aal": _get_aal,
        "oxford": _get_ho,
        "cerebra": _get_cerebra,
        "dkt": _get_dkt,
    }[atlas_name](*atlas_kwargs)

    if return_mni:
        arr = utils.upsample(arr=atlas_map.get_fdata(),
                             return_mni=return_mni,
                             target_shape=target_shape,
                             interpolate_mode=interpolate_mode)
    else:
        arr = atlas_map.get_fdata()
    indices = _literal_eval(lst=indices)
    assert len(indices) == len(labels),\
        f"# Labels and indices should be same: len(indices)={len(indices)}, len(labels)={len(labels)}"

    array_obj = Bunch(array=arr, indices=indices, labels=labels, nii=atlas_map)
    array_obj.get_roi_index = types.MethodType(get_roi_index, array_obj)
    array_obj.get_roi_name = types.MethodType(get_roi_name, array_obj)
    return array_obj


def get_roi_index(self, roi_name: int) -> int:
    idx = self.labels.index(roi_name)
    idx = self.indices[idx]
    return idx


def get_roi_name(self, index: int) -> str:
    idx = self.indices.index(index)
    label = self.labels[idx]
    return label


    
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


def _get_dkt() -> Tuple[nii, list, list]:
    # TODO: Fix hard-coded path
    atlas = load_img(img=C.DKT_ATLAS)
    atlas = utils.align(arr=atlas.get_fdata(), affine=C.BIOBANK_AFFINE)
    
    meta = C.DKT_META
    indices = meta.SegId.tolist()
    labels = meta.StructName.tolist()
    return atlas, indices, labels
