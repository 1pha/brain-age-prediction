""" Compare saliency maps over ATLAS. 
Possible outcomes:
- Saliency value bar-graph
- Region-wise maps (glass / anat)

They can be inferred with absolute or raw values"""
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.image import resample_img, new_img_like
import nilearn.plotting as nilp
from sklearn.utils import Bunch
from tqdm import tqdm

from sage.utils import get_logger
from . import nilearn_plots as nilp_
from .utils import _mni, _safe_get_data
from .atlas import get_atlas

try:
    import sage.constants as C
except ImportError:
    import meta_brain.router as C


logger = get_logger(name=__file__)

ASSET_DIR = Path("assets/weights")


def load_sal(path: str = "resnet10t-aug",
             mask: str = "nomask",
             xai: str = "ig",
             top_k: float = 0.95,
             load_top: bool = True,
             root_dir: Path = ASSET_DIR) -> np.ndarray:
    """ Loads saliency and their metadata
    Given arguments will be used as follows
    Loads attr.npy from following dir
    root / $path / $mask / $xai$top_k / $load_top_attr.npy 
    - $mask: e.g. mask, no-mask, sigma=0.5, sigma=1.0
    """
    
    npy_dir = root_dir / \
              path / \
              mask / \
              f"{xai}k{top_k}" / \
              f"{'top_attr' if load_top else 'attrs'}.npy"
    saliency = np.load(file=npy_dir)
    return saliency


def align(arr: np.ndarray) -> nib.nifti1.Nifti1Image:
    _arr: np.ndarray = _safe_get_data(_mni(arr), ensure_finite=True)
    arr_nifti = new_img_like(_mni(arr), _arr, C.MNI_AFFINE)
    return arr_nifti


def resample_sal(arr: np.ndarray,
                 atlas: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """ Resample a given array to target atlas.
    When resampling, given array will be converted to mni space with nibabel. """
    arr_nifti = align(arr)
    resampled = resample_img(img=arr_nifti,
                             target_affine=atlas.affine,
                             target_shape=atlas.shape)
    return resampled

    
def calculate_overlaps(arr: np.ndarray,
                       atlas: Bunch,
                       title: str = "",
                       use_abs: bool = True,
                       vmin: float = None, vmax: float = None,
                       plot_raw_sal: bool = True,
                       plot_bargraph: bool = True,
                       plot_brains: bool = True) -> dict:
    ### Setups ###
    # Load proper mask
    if isinstance(arr, nib.nifti1.Nifti1Image):
        mask_ = arr.get_fdata()
    elif isinstance(arr, np.ndarray):
        mask_ = arr.copy()
        
    # Load atlas
    if isinstance(atlas, str):
        atlas = get_atlas(atlas_name=atlas,
                          return_mni=False if atlas == "cerebra" else True)
    
    if use_abs:
        mask_ = np.abs(mask_)
        logger.info("Overlaps with absolute values")
    else:
        logger.info("Overlaps with raw values")
        
    if plot_raw_sal and plot_brains:
        _title = f"{title}_RAW Mask"
        nilp_.plot_glass_brain(arr=mask_, target_affine=atlas.nii.affine,
                               colorbar=True, title=_title, plot_abs=False) # Always draw raw

    # 1. Calculate values over regions
    xai_dict = dict()
    pbar = tqdm(iterable=zip(atlas.indices, atlas.labels),
                total=len(atlas.indices),
                desc="Aggregating values across ROIs")
    for idx, label in pbar:
        roi_mask = atlas.array == idx
        
        # Norm
        num_nonzero = np.count_nonzero(roi_mask)
        roi_val = np.nansum(roi_mask * mask_)
        xai_dict[label] = roi_val / num_nonzero
    
    if plot_bargraph:
       nilp_.brain_barplot(xai_dict=xai_dict, title=title)
        
    # 2. Map dict on brain
    agg_saliency = np.zeros_like(atlas.array)

    pbar = tqdm(iterable=xai_dict, desc="Spread values to Brain ROI ...")
    for label in pbar:
        val = xai_dict[label]        
        idx = atlas.labels.index(label)
        idx = atlas.indices[idx]
        agg_saliency[np.where(atlas.array == int(idx))] = val
        
    if plot_brains:
        nilp_.plot_glass_brain(arr=agg_saliency,
                               target_affine=atlas.nii.affine, title=title,
                               vmin=vmin, vmax=vmax,
                               colorbar=True, plot_abs=use_abs)
        nilp_.plot_overlay(arr=agg_saliency,
                           target_affine=atlas.nii.affine,
                           display_mode="mosaic",
                           threshold=0.25, title=title,
                           cmap=nilp.cm.red_transparent if use_abs else nilp.cm.bwr,
                           colorbar=True)
        
    return xai_dict, agg_saliency
