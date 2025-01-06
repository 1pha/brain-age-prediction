import numpy as np
import nibabel as nib
import nilearn.plotting as nilp
from nilearn.datasets import load_mni152_template
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import pandas as pd
sns.set_theme()

import constants as C


_nifti = lambda arr, affine=np.eye(4): nib.nifti1.Nifti1Image(arr, affine)
_mni = lambda arr: nib.nifti1.Nifti1Image(arr, C.MNI_AFFINE)


def _tensorfy(arr: np.ndarray) -> np.ndarray:
    """ Convert torch.tensor or numpy array into 3-dim numpy array"""
    # torch.tensor given
    if isinstance(arr, np.ndarray):
        while arr.ndim > 3:
            arr = arr[0]
    else:
        print("Provide torch.tensor or numpy ndarray")
        raise
    return arr


def check_nii(arr: np.ndarray | nib.nifti1.Nifti1Image,
              target_mni: bool = True,
              target_affine: np.ndarray = None):
    """ Returns nii for a given array """
    if isinstance(arr, np.ndarray):
        arr = _tensorfy(arr=arr)
        if target_mni and target_affine is None:
            # If target_affine is given, this should be ignored
            nifti = _mni(arr)
        else:
            assert target_affine is not None, f"Please provide target_affine. Given {target_affine}"
            nifti = _nifti(arr=arr, affine=target_affine)
    elif isinstance(arr, nib.nifti1.Nifti1Image):
        nifti = arr
    else:
        raise
    return nifti


def plot_corr(corr: pd.DataFrame, subtitle: str = "", subtitle_size: str | int = "large",
              hide_triu: bool = True, ax=None,
              cbar_size: float = 0.7, use_cbar: bool = True):
    if hide_triu:
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False
    else:
        mask = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    hm = sns.heatmap(corr, mask=mask, ax=ax,
                     vmin=-1, vmax=1, cmap="coolwarm",
                     cbar_kws={"shrink": cbar_size}, cbar=use_cbar,
                     annot=True, fmt=".2f", annot_kws={"size": 9},
                     square=True, linewidth=0.5)
    for i, model_name in enumerate(corr.index):
        model_name = model_name.split(" ")[0]
        if i == 0:
            prev = model_name
            continue
        if prev != model_name:
            if hide_triu:
                hm.axhline(i, xmin=0, xmax=i / len(corr), color="black", linewidth=1.2)
                hm.axvline(i, ymin=0, ymax=(len(corr) - i) / len(corr), color="black", linewidth=1.2)
            else:
                hm.axhline(i, color="black", linewidth=1.2)
        prev = model_name
    ax.set_title(subtitle, size=subtitle_size)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_brain(arr: np.ndarray, use_mni: bool = True, save: str = None, **kwargs):
    arr = _tensorfy(arr)
    nifti = _mni if use_mni else _nifti
    display = nilp.plot_anat(anat_img=nifti(arr), output_file=save, **kwargs)
    return display, save


def plot_overlay(arr: np.ndarray,
                 use_mni: bool = True,
                 target_affine: np.ndarray = None,
                 bg: np.ndarray | nib.nifti1.Nifti1Image = None,
                 display_mode: str = "ortho",
                 cut_coords: tuple = None, title: str = "", title_size: str | float = "x-large",
                 save: str = None, alpha: float = 0.7, **kwargs):    
    """ Works similar with `nilp.plot_roi`
    However this first plots an original brain (or the MNI template)
    and overlays with `nilp.add_overlay`. """
    arr = check_nii(arr=arr, target_mni=use_mni, target_affine=target_affine)
    if bg is None:
        bg = load_mni152_template()
    display = nilp.plot_anat(anat_img=bg, display_mode=display_mode, cut_coords=cut_coords)
    display.add_overlay(arr, alpha=alpha, **kwargs)
    display.title(text=title, size=title_size)
    if save is not None:
        display.savefig(save)
    return display, save


def gaussian_blur(arr: np.ndarray, sigma: float = 1.0, **kwargs) -> np.ndarray:
    blurred = gaussian_filter(input=arr, sigma=sigma, **kwargs)
    return blurred


def thresholding(arr: np.ndarray, q: float) -> np.ndarray:
    """ Applies quantile based thresholding.
    Logic throws away values under the threshold """
    threshold = np.quantile(a=arr, q=q)
    mask = arr < threshold
    arr[mask] = 0.
    return arr


def preprocess_saliency(arr: np.ndarray,
                        sigma: float = 1.0, blur_kwargs: dict = {},
                        q: float = 0.8,
                        use_mni: bool = True, display_mode: str = "mosaic",
                        title: str = "", title_size: str | float = "x-large"):
    """ Preprocessing saliencies to provide to doctors.
    Takes 3d raw brain saliency maps as input and
    1. Apply gaussian blurring
    2. Apply thresholding
    3. Plots mosaic view and save.
    """
    # 1. Apply Gaussian blur

    # 2. Thresholding
    arr = thresholding(arr=arr, q=q)
    arr = gaussian_blur(arr=arr, sigma=sigma, **blur_kwargs)
    
    # 3. Plot Mosaic view on MNI template
    dp, _ = plot_overlay(arr=arr, use_mni=use_mni, display_mode=display_mode,
                         title=title, title_size=title_size)
    return arr, dp


def to_square(df: pd.DataFrame, xai_key: str = None, method: str = None) -> pd.DataFrame:
    """Convert dataframe into square form for heatmap plots"""
    if not (xai_key is None and method is None):
        subdf = df[(df[C.XCOL] == xai_key) & (df["Similarity Method"].str.contains(method))]
    else:
        subdf = df.copy()
    subdf = subdf[["Similarity", "Source", "Target"]]

    ids = np.unique(subdf[['Source', 'Target']])
    matrix = pd.DataFrame(data=np.nan, index=ids, columns=ids)
    for _, row in subdf.iterrows():
        matrix.at[row['Source'], row['Target']] = row['Similarity']
        matrix.at[row['Target'], row['Source']] = row['Similarity']  # if the matrix is symmetric
    return matrix


def heatmap(rob: pd.DataFrame, method: str, auto_cbar: bool = True, save: str = "") -> None:
    assert method in {"spearmanr", "cossim"}, f"Method: {method}"
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 12))
    vmin = None if auto_cbar else -1
    vmax = None if auto_cbar else 1
    for idx, xai in enumerate(C.XAI_METHODS):
        r, c = idx % 2, idx // 2
        _ax = ax[r, c]

        mat = to_square(df=rob, xai_key=xai, method=method)
        mask = np.triu(np.ones_like(mat, dtype=bool))
        
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_bad(color="none")
        sns.heatmap(mat, mask=mask, cmap=cmap, ax=_ax, vmin=vmin, vmax=vmax, square=True, linewidth=0.,)
        _ax.set_title(C.XAI_METHODS_MAPPER.get(xai, xai), size="x-large")
        _ax.grid(False)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    else:
        fig.suptitle(f"Similarity method: {method}, Average of patient-wise Robustness", size=35)