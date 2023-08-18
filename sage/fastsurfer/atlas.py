import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
import nilearn as nil
import nilearn.plotting as nilp

from sage.utils import get_logger
from sage.constants import BIOBANK_AFFINE, MNI_SHAPE
from sage.xai.utils import _nifti
import sage.xai.nilearn_plots as nilp_
from .constants import SEG_ROOT, ASEG_AFFINE
from .utils import _get_segmask
from .stats import get_name_bysegid, get_idx_byname, get_seg_results


logger = get_logger(name=__file__)


def generate_proba_asegdkt(root_dir: Path = SEG_ROOT) -> Dict[int, np.ndarray]:
    seg_results = get_seg_results(root_dir=root_dir)
    seg_map = _get_segmask(path=seg_results[0]).get_fdata()
    seg_indices = np.unique(seg_map)
    
    # Initialize seg_map
    proba_atlas = dict()
    for idx in seg_indices:
        proba_atlas[idx] = (seg_map == idx).astype(np.int16)

    pbar = tqdm(iterable=seg_results[1:], desc="# Scans left")
    for seg in pbar:
        seg_map = _get_segmask(path=seg).get_fdata()
        for idx in seg_indices:
            proba_atlas[idx] += (seg_map == idx).astype(np.int16)
    return proba_atlas


def load_proba_asegdkt(resampled: bool = True,
                       cached: bool = True,
                       affine: np.ndarray = None) -> Dict[int, np.ndarray]:
    with open("./assets/proba_atlas.pkl", mode="rb") as f:
        proba_atlas = pickle.load(f)
    if resampled:
        if cached:
            with open("./assets/proba_atlas_resampled.pkl", mode="rb") as f:
                proba_atlas = pickle.load(f)
            return proba_atlas
        logger.info("Resampling ATLAS")
        affine = affine if affine is not None else ASEG_AFFINE
        pbar = tqdm(iterable=proba_atlas.keys(), desc="Going through RoIs ...")
        for key in pbar:
            if key == 0:
                continue
            atlas = proba_atlas[key] / 4581
            atlas = np.clip(a=atlas, a_min=0, a_max=1)
            atlas = nil.image.resample_img(_nifti(atlas, affine=affine),
                                        target_affine=BIOBANK_AFFINE,
                                        target_shape=MNI_SHAPE,
                                        interpolation='linear').get_fdata()
            proba_atlas[key] = atlas
    return proba_atlas


def _load_fs(root_dir: Path = SEG_ROOT):
    seg_results = get_seg_results(root_dir=root_dir)
    fs = _nifti(_get_segmask(path=seg_results[0]).get_fdata(),
                affine=ASEG_AFFINE)
    return fs


def calculate_overlaps(arr: np.ndarray,
                       atlas: Dict[int, np.ndarray],
                       title: str = "",
                       use_abs: bool = True,
                       vmin: float = None, vmax: float = None,
                       plot_raw_sal: bool = True,
                       plot_bargraph: bool = True,
                       plot_brains: bool = True) -> Tuple[dict, np.ndarray]:
    
    if use_abs:
        arr = np.abs(arr)
        logger.info("Overlaps with absolute values")
    else:
        logger.info("Overlaps with raw values")
        
    if plot_raw_sal and plot_brains:
        _title = f"{title}_RAW Mask"
        nilp_.plot_glass_brain(arr=arr, target_affine=BIOBANK_AFFINE,
                               colorbar=True, title=_title, plot_abs=False) # Always draw raw

    xai_dict = dict()
    pbar = tqdm(iterable=atlas.keys(), desc="Going through RoIs ...")
    for key in pbar:
        if key == 0:
            # Skip background
            continue
        _atlas = atlas[key]
        roi_mask = _atlas * arr
        roi_val = np.nansum(roi_mask)
        num_nonzero = np.nansum(_atlas)
        label = get_name_bysegid(idx=key)
        xai_dict[label] = roi_val / num_nonzero
        
    if plot_bargraph:
        nilp_.brain_barplot(xai_dict=xai_dict, title=title)
    
    sample = nil.image.resample_img(img=_load_fs(),
                                   target_affine=BIOBANK_AFFINE,
                                   target_shape=MNI_SHAPE,
                                   interpolation='linear')
    agg_saliency = np.zeros_like(sample.get_fdata())

    pbar = tqdm(iterable=xai_dict, desc="Spread values to Brain ROI ...")
    for label in pbar:
        val = xai_dict[label]        
        idx = get_idx_byname(name=label)
        agg_saliency[np.where(sample.get_fdata() == int(idx))] = val

    if plot_brains:
        nilp_.plot_glass_brain(arr=agg_saliency,
                               target_affine=BIOBANK_AFFINE, title=title,
                               vmin=vmin, vmax=vmax,
                               colorbar=True, plot_abs=use_abs)
        nilp_.plot_overlay(arr=agg_saliency,
                           target_affine=BIOBANK_AFFINE,
                           display_mode="mosaic",
                           threshold=0.25, title=title,
                           cmap=nilp.cm.red_transparent if use_abs else nilp.cm.bwr,
                           colorbar=True)
    return xai_dict, agg_saliency