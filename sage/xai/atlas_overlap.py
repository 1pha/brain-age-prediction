""" Compare saliency maps over ATLAS. 
Possible outcomes:
- Saliency value bar-graph
- Region-wise maps (glass / anat)

They can be inferred with absolute or raw values"""
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import nibabel as nib
import nilearn.plotting as nilp
from nilearn.image import resample_img, new_img_like
from sklearn.utils import Bunch
from tqdm import tqdm
import torch

from sage.utils import get_logger
from . import nilearn_plots as nilp_
from . import atlas as A
from . import utils

try:
    import sage.constants as C
except ImportError:
    import meta_brain.router as C


logger = get_logger(name=__file__)

ASSET_DIR = Path("assets/weights")


def get_path(path: str,
             mask: str = "nomask",
             xai: str = "ig",
             top_k: float = 0.95,
             load_top: bool = True,
             root_dir: Path = ASSET_DIR) -> Path:
    """ Returns path for saliency map"""
    npy_dir = root_dir / \
              path / \
              mask / \
              f"{xai}k{top_k}" / \
              f"{'top_attr' if load_top else 'attrs'}.npy"
    return npy_dir


def load_sal(path: str = "resnet10t-aug",
             mask: str = "nomask",
             xai: str = "ig",
             top_k: float = 0.95,
             load_top: bool = True,
             root_dir: Path = ASSET_DIR) -> Tuple[np.ndarray, Path]:
    """ Loads saliency and their metadata
    Given arguments will be used as follows
    Loads attr.npy from following dir
    root / $path / $mask / $xai$top_k / $load_top_attr.npy 
    - $mask: e.g. mask, no-mask, sigma=0.5, sigma=1.0
    """
    
    npy_dir = get_path(path=path, mask=mask, xai=xai,
                       top_k=top_k, load_top=load_top, root_dir=root_dir)
    saliency = np.load(file=npy_dir)
    return saliency, npy_dir


def align(arr: np.ndarray) -> nib.nifti1.Nifti1Image:
    _arr: np.ndarray = utils._safe_get_data(utils._mni(arr), ensure_finite=True)
    arr_nifti = new_img_like(utils._mni(arr), _arr, C.MNI_AFFINE)
    return arr_nifti


def resample_sal(arr: np.ndarray,
                 atlas: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """ Resample a given array to target atlas.
    When resampling, given array will be converted to mni space with nibabel. """
    arr_nifti = utils.align(arr)
    resampled = resample_img(img=arr_nifti,
                             target_affine=atlas.affine,
                             target_shape=atlas.shape)
    return resampled


def flatten_to_dict(arr: np.ndarray,
                    atlas: Bunch,
                    use_torch: bool = False,
                    device: str = "cpu",
                    use_abs: bool = True,
                    verbose: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
    """ Given a saliency array,
    calculate representative value for each RoI
    """
    if isinstance(arr, nib.nifti1.Nifti1Image):
        mask_ = arr.get_fdata()
    elif isinstance(arr, np.ndarray):
        mask_ = arr.copy()
    elif isinstance(arr, torch.Tensor):
        # Monai `MetaTensor` would also get caught here.
        mask_ = arr.clone()
        use_torch = True

    if use_abs:
        mask_ = np.abs(mask_)

    xai_dict = dict()
    pbar = tqdm(iterable=zip(atlas.indices, atlas.labels),
                total=len(atlas.indices), leave=verbose,
                desc="Aggregating values across ROIs")
    if use_torch:
        atlas_ = torch.from_numpy(atlas.array).to(device)
        mask_ = torch.from_numpy(mask_).to(device) if isinstance(mask_, np.ndarray) else mask_.to(device)
        for idx, label in pbar:
            roi_mask = atlas_ == idx
            # Norm
            num_nonzero = torch.sum(roi_mask)
            roi_val = torch.nansum(roi_mask * mask_)
            xai_dict[label] = float((roi_val / num_nonzero).cpu().numpy())
        mask_ = mask_.cpu().numpy()

    else:
        for idx, label in pbar:
            roi_mask = atlas.array == idx
            # Norm
            num_nonzero = np.sum(roi_mask)
            roi_val = np.nansum(roi_mask * mask_)
            xai_dict[label] = roi_val / num_nonzero
    return xai_dict, mask_


def project_to_atlas(atlas: Bunch,
                     xai_dict: dict,
                     title: str = "",
                     use_abs: bool = True,
                     vmin: float = None, vmax: float = None,
                     threshold: float = 0.25,
                     root_dir: Path | str = None,
                     verbose: bool = False) -> np.ndarray:
    agg_saliency = np.zeros_like(atlas.array)

    pbar = tqdm(iterable=xai_dict, desc="Spread values to Brain ROI ...", leave=verbose)
    for label in pbar:
        val = xai_dict[label]
        idx = atlas.labels.index(label)
        idx = atlas.indices[idx]
        agg_saliency[np.where(atlas.array == int(idx))] = val

    save = root_dir / "proj_glass.png" if root_dir is not None else None
    nilp_.plot_glass_brain(arr=agg_saliency,
                           target_affine=atlas.nii.affine, title=title, cmap=nilp.cm.bwr,
                           vmin=vmin, vmax=vmax, colorbar=True, plot_abs=use_abs, save=save)

    save = root_dir / "proj_mosaic.png" if root_dir is not None else None
    if (vmin is None) or (vmax is None):
        _max = np.abs(agg_saliency).max()
        vmin, vmax = -_max, _max
    nilp_.plot_overlay(arr=agg_saliency, target_affine=atlas.nii.affine, vmin=vmin, vmax=vmax,
                       display_mode="mosaic", threshold=threshold, title=title, colorbar=True,
                       cmap=nilp.cm.red_transparent if use_abs else nilp.cm.bwr, save=save)
    return agg_saliency


def calculate_overlaps(arr: np.ndarray,
                       atlas: Bunch = None,
                       title: str = "",
                       use_torch: bool = False,
                       device: str = "cpu",
                       use_abs: bool = True,
                       vmin: float = None, vmax: float = None,
                       root_dir: Path | str = None,
                       plot_raw_sal: bool = True,
                       plot_bargraph: bool = True,
                       plot_projection: bool = True,
                       verbose: bool = False,) -> Tuple[Dict[str, float], np.ndarray]:
    # Load atlas if not loaded
    if isinstance(atlas, str) or (atlas is None):
        atlas = A.get_atlas(atlas_name=atlas, return_mni=False if atlas == "cerebra" else True)

    xai_dict, mask_ = flatten_to_dict(arr=arr, atlas=atlas,
                                      use_torch=use_torch, device=device, use_abs=use_abs)
    if plot_raw_sal:
        _title = f"{title}_RAW Mask"
        save = root_dir / "raw_mask.png" if root_dir is not None else root_dir
        nilp_.plot_glass_brain(arr=mask_, target_affine=atlas.nii.affine,
                               colorbar=True, title=_title, plot_abs=False) # Always draw raw

    if plot_bargraph:
        save = root_dir / "bargraph.png" if root_dir is not None else root_dir
        nilp_.brain_barplot(xai_dict=xai_dict, title=title, save=save)
        
    if plot_projection:
        agg_saliency = project_to_atlas(atlas=atlas, xai_dict=xai_dict, root_dir=root_dir,
                                        title=title, use_abs=use_abs, vmin=vmin, vmax=vmax)
    else:
        agg_saliency = None

    return xai_dict, agg_saliency
