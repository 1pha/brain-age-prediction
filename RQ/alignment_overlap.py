import json
from collections import defaultdict
from typing import List, Dict, Union
from pathlib import Path

from tqdm import tqdm
from scipy.ndimage import maximum_filter, binary_erosion, distance_transform_edt
import numpy as np
import nibabel as nib
import nilearn.image as nili
from nilearn.datasets import load_mni152_brain_mask

import constants as C
import weight_parser as wp
from metadata import load_vbm, load_vbm_adni


BRAINMASK = load_mni152_brain_mask()


def get_masked_topq(brain: np.ndarray | nib.Nifti1Image,
                    mask: np.ndarray,
                    q_percentile: float) -> np.ndarray | nib.Nifti1Image:
    """
    3D array에서 mask가 True인 영역 내의 상위 q-percentile 값만 남기고 나머지는 0으로 만듭니다.
    
    Parameters:
        brain: 3차원 numpy array
        mask: array_3d와 같은 shape의 boolean mask
        q_percentile: 상위 몇 퍼센트를 선택할지 지정 (0-100 사이 값)
    
    Returns:
        result: 원본과 같은 shape의 array. mask 영역 내 상위 q-percentile 값만 유지
    """
    if isinstance(brain, nib.Nifti1Image):
        brain = brain.get_fdata()
        affine = brain.affine
    else:
        affine = None
    result = np.zeros_like(brain)
    
    # mask가 True인 부분의 값들만 추출
    if mask is None:
        mask = load_mni152_brain_mask()
    if isinstance(mask, nib.Nifti1Image):
        mask = mask.get_fdata().astype(bool)
    masked_values = brain[mask]

    if len(masked_values) == 0:  # mask가 모두 False인 경우
        return result
    
    # q-percentile에 해당하는 threshold 값 계산
    threshold = np.percentile(masked_values, 100 - q_percentile)
    
    # threshold보다 큰 값들의 mask 생성
    top_mask = (brain >= threshold) & mask
    
    # threshold를 넘는 값들만 결과 array에 복사
    result[top_mask] = brain[top_mask]
    
    if affine is not None:
        result = nib.nifti1.Nifti1Image(dataobj=result, affine=affine)

    return result


def compare_brain_maps(target_map: Union[np.ndarray, nib.Nifti1Image],
                       saliency_map: Union[np.ndarray, nib.Nifti1Image],
                       brain_mask: Union[np.ndarray, nib.Nifti1Image],
                       q_percentiles: List[float]) -> Dict[str, List[float]]:
    """
    Compare VBM t-map with saliency map using various metrics for different q-percentiles.
    All comparisons are done within the brain mask region only.
    
    Args:
        target_map: VBM t-statistics map
        saliency_map: Interpretability method's saliency map
        brain_mask: Binary mask defining brain regions
        q_percentiles: List of percentile thresholds to use
        
    Returns:
        Dictionary containing lists of metrics for each q_percentile,
        including robust metrics (robust ASD and HD95)
    """

    # Convert inputs to numpy arrays if needed
    if isinstance(target_map, nib.Nifti1Image):
        target_map = target_map.get_fdata()
    if isinstance(saliency_map, nib.Nifti1Image):
        saliency_map = saliency_map.get_fdata()
    if isinstance(brain_mask, nib.Nifti1Image):
        brain_mask = brain_mask.get_fdata()
    
    # Apply brain mask to both maps
    masked_target = target_map * brain_mask
    masked_saliency = saliency_map * brain_mask

    brain_voxels = brain_mask > 0
    target_nonzero = (masked_target != 0)
    total_target_voxels = np.sum(target_nonzero)
    
    # Initialize results dictionary (추가: robust_asd와 hd95)
    results = {
        'overlap_count': [],
        'overlap_percentage': [],
        'dice': [],
        'iou': [],
        'spatial_correlation': [],
        'peak_overlap': [],
        'target_mean': [],
        'weighted_dice': [],
        'asd': [],
        'hd': [],
        'robust_asd': [],
        'hd95': []
    }
    
    # Helper function for peak detection
    def get_local_maxima(data: np.ndarray, mask: np.ndarray, size: int = 3) -> np.ndarray:
        mask_bool = mask.astype(bool)
        masked_data = data * mask_bool
        local_max = maximum_filter(masked_data, size=size) == masked_data
        threshold = np.percentile(masked_data[mask_bool], 95)
        return local_max & mask_bool & (masked_data > threshold)
    
    # Weighted Dice function
    def weighted_dice_score(pred, target, target_map):
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        target_map = target_map.astype(np.float32)
        
        # Normalize target_map to [0, 1]
        norm_target_map = (target_map - np.min(target_map)) / (np.max(target_map) - np.min(target_map) + 1e-8)
        
        intersection = np.sum(norm_target_map * pred * target)
        pred_sum = np.sum(norm_target_map * pred)
        target_sum = np.sum(norm_target_map * target)
        
        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
        return dice

    # Helper function to extract surface voxels from a binary segmentation
    def get_surface_voxels(segmentation: np.ndarray) -> np.ndarray:
        seg_bool = segmentation.astype(bool)
        eroded = binary_erosion(seg_bool)
        surface = seg_bool & ~eroded
        return surface

    # Robust metrics functions
    def robust_asd(distances1: np.ndarray, distances2: np.ndarray) -> float:
        """Compute the robust average surface distance using the median of distances."""
        if distances1.size + distances2.size > 0:
            combined = np.concatenate((distances1, distances2))
            return float(np.median(combined))
        else:
            return 0.0

    def hd95(distances1: np.ndarray, distances2: np.ndarray) -> float:
        """
        Compute the 95th percentile Hausdorff Distance (HD95) between two surfaces.
        Outlier distances are excluded by using the 95th percentile.
        """
        if distances1.size > 0:
            hd95_1 = np.percentile(distances1, 95)
        else:
            hd95_1 = 0.0
        if distances2.size > 0:
            hd95_2 = np.percentile(distances2, 95)
        else:
            hd95_2 = 0.0
        return float(max(hd95_1, hd95_2))
    
    # Calculate metrics for each q_percentile
    for q in q_percentiles:
        # Compute threshold for saliency map within brain mask
        brain_saliency = masked_saliency[brain_voxels]
        nonzero_saliency = brain_saliency[brain_saliency != 0]
        saliency_threshold = np.percentile(nonzero_saliency, 100 - q) if len(nonzero_saliency) > 0 else 0
        
        # Create binary mask for current threshold
        current_saliency_mask = (masked_saliency >= saliency_threshold) & (masked_saliency > 0)
        total_saliency_voxels = np.sum(current_saliency_mask)
        
        # Calculate overlap between target and current saliency mask
        overlap_mask = np.logical_and(target_nonzero, current_saliency_mask)
        overlap_count = np.sum(overlap_mask)
        overlap_percentage = overlap_count / total_target_voxels if total_target_voxels > 0 else 0
        dice = 2 * overlap_count / (total_target_voxels + total_saliency_voxels) if (total_target_voxels + total_saliency_voxels) > 0 else 0
        iou = overlap_count / (total_target_voxels + total_saliency_voxels - overlap_count) if (total_target_voxels + total_saliency_voxels - overlap_count) > 0 else 0

        # Spatial correlation and target mean in overlapping regions
        if overlap_count > 0:
            target_values = masked_target[overlap_mask]
            saliency_values = masked_saliency[overlap_mask]
            if np.std(target_values) > 0 and np.std(saliency_values) > 0:
                try:
                    correlation_matrix = np.corrcoef(target_values, saliency_values)
                    spatial_corr = correlation_matrix[0, 1]
                    if np.isnan(spatial_corr):
                        spatial_corr = 0.0
                except:
                    spatial_corr = 0.0
            else:
                spatial_corr = 0.0
            target_mean = np.mean(target_values)
        else:
            spatial_corr = 0.0
            target_mean = 0.0

        # Calculate peak overlap using local maxima
        brain_mask_bool = brain_mask.astype(bool)
        current_saliency_mask_bool = current_saliency_mask.astype(bool)
        target_peaks = get_local_maxima(masked_target, brain_mask_bool)
        saliency_peaks = get_local_maxima(masked_saliency, current_saliency_mask_bool & brain_mask_bool)
        peak_overlap = (np.sum(np.logical_and(target_peaks, saliency_peaks)) / np.sum(target_peaks)
                        if np.sum(target_peaks) > 0 else 0)
        
        weighted_dice = weighted_dice_score(current_saliency_mask, target_nonzero, masked_target)

        # ===== Robust ASD and HD95 계산 =====
        # 이진 분할 (target: 실제 영역, saliency: 해석 방법 결과)
        target_seg = target_nonzero
        saliency_seg = current_saliency_mask

        # 각 분할의 표면(경계) 추출
        target_surface = get_surface_voxels(target_seg)
        saliency_surface = get_surface_voxels(saliency_seg)
        
        # target_surface에서 saliency_surface까지의 최소 거리 계산
        if np.sum(saliency_surface) > 0:
            dt_saliency = distance_transform_edt(~saliency_surface)
            distances_target_to_saliency = dt_saliency[target_surface]
        else:
            distances_target_to_saliency = np.array([])
        
        # saliency_surface에서 target_surface까지의 최소 거리 계산
        if np.sum(target_surface) > 0:
            dt_target = distance_transform_edt(~target_surface)
            distances_saliency_to_target = dt_target[saliency_surface]
        else:
            distances_saliency_to_target = np.array([])
        
        # 기존 ASD와 HD
        if (distances_target_to_saliency.size + distances_saliency_to_target.size) > 0:
            asd = (np.sum(distances_target_to_saliency) + np.sum(distances_saliency_to_target)) / \
                  (distances_target_to_saliency.size + distances_saliency_to_target.size)
            hd = max(np.max(distances_target_to_saliency) if distances_target_to_saliency.size > 0 else 0, 
                     np.max(distances_saliency_to_target) if distances_saliency_to_target.size > 0 else 0)
        else:
            asd = 0.0
            hd = 0.0
        
        # Robust metrics: robust ASD (median)와 HD95 (95번째 백분위수)
        robust_asd_val = robust_asd(distances_target_to_saliency, distances_saliency_to_target)
        hd95_val = hd95(distances_target_to_saliency, distances_saliency_to_target)
        # =========================================
        
        # 결과 저장 (모든 값은 Python float로 저장)
        results['overlap_count'].append(float(overlap_count))
        results['overlap_percentage'].append(float(overlap_percentage))
        results['dice'].append(float(dice))
        results['iou'].append(float(iou))
        results['spatial_correlation'].append(float(spatial_corr))
        results['peak_overlap'].append(float(peak_overlap))
        results['target_mean'].append(float(target_mean))
        results['weighted_dice'].append(float(weighted_dice))
        results['asd'].append(float(asd))
        results['hd'].append(float(hd))
        results['robust_asd'].append(robust_asd_val)
        results['hd95'].append(hd95_val)
    
    return results

def calculate_all_alignment(xai_methods: List[str] = C.XAI_METHODS,
                            models: List[str] = C.MODELS,
                            seeds: List[int] = C.SEEDS,
                            is_cls: bool = False,
                            verbose: bool = False):
    # VBM affine is currently bank affine
    affine, shape = C.BIOBANK_AFFINE, BRAINMASK.shape
    def preprocess_vbm(img: nib.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """Resamples a given nifti image into a biobank affine and shape
        """
        # Fillna
        img_nan20_nii = nib.nifti1.Nifti1Image(np.nan_to_num(img.get_fdata(), nan=0), affine=img.affine)
        # Above threshold
        img_nii = nili.resample_img(img=img_nan20_nii, target_affine=affine, target_shape=shape)
        img_arr = np.where(img_nii.get_fdata() >= C.UKB_FEW_THD, img_nii.get_fdata(), 0)
        return nib.nifti1.Nifti1Image(img_arr, affine=affine)
        
    # Create `vbm_merge` that will be used as a 'ground-truth'
    if is_cls:
        # Case of ADNI
        vbm = load_vbm_adni()["nii"]
        target = preprocess_vbm(img=vbm)
    else:
        vbms = load_vbm()
        vbm_yo = preprocess_vbm(img=vbms["young2old_nii"])
        vbm_oy = preprocess_vbm(img=vbms["old2young_nii"])
        vbm_merge = np.maximum(vbm_yo.get_fdata(), vbm_oy.get_fdata())
        vbm_merge = np.where(vbm_merge >= C.UKB_FEW_THD_TWO, vbm_merge, 0)
        target = nib.nifti1.Nifti1Image(vbm_merge, affine=affine)

    for xai_method in xai_methods:
        for model_name in models:
            for seed in seeds:
                if is_cls:
                    w = wp.WeightsCls(model_name=f"{model_name}-binary", seed=seed, xai_method=xai_method,
                                      base_dir=C.ADNI_WEIGHT_DIR, verbose=verbose)
                else:
                    w = wp.Weights(model_name=model_name, seed=seed, xai_method=xai_method,
                                   base_dir=C.WEIGHT_DIR, verbose=verbose)

                attrs = w.load_attributes()
                attr, top_attr = attrs["attrs"], attrs["top_attr"]

                qs = [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 75, 100]
                print(f"{xai_method} & {model_name} | {seed} Top-q perc value")
                try:
                    attr_result = compare_brain_maps(target_map=target,
                                                    saliency_map=attr,
                                                    brain_mask=BRAINMASK,
                                                    q_percentiles=qs)
                    with open(w.xai_path / "vbm_sim_attr.json", mode="w") as f:
                        json.dump(obj=attr_result, fp=f, indent="\t")
                    
                    top_attr_result = compare_brain_maps(target_map=target,
                                                        saliency_map=top_attr,
                                                        brain_mask=BRAINMASK,
                                                        q_percentiles=qs)
                    with open(w.xai_path / "vbm_sim_top_attr.json", mode="w") as f:
                        json.dump(obj=top_attr_result, fp=f, indent="\t")
                except IndexError:
                    print(f"\t\tFailed: {xai_method} & {model_name} | {seed}, all zero")
                    continue
