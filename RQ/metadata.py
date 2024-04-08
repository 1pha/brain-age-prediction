import json
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import numpy as np

import constants as C
import weight_parser as wp


def load_vbm(base_dir: Path = C.VBM_DIR) -> Dict[str, Dict[str, float]]:
    """ Loads VBM stats result for young2old and old2young """
    with (base_dir / "12_old_young" / "vbm_dict.json").open(mode="r") as f:
        cont_12_dict = json.load(f)
    with (base_dir / "21_young_old" / "vbm_dict.json").open(mode="r") as f:
        cont_21_dict = json.load(f)
    return dict(young2old=cont_12_dict, old2young=cont_21_dict)

    
def load_fastsurfer(base_dir: Path = C.FS_DIR) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """ Loads SpearmanR result on age vs. Voxel/Intensity respectively """
    with (base_dir / "fastsurfer_vox_spearmanr(vs_age).json").open(mode="r") as f:
        fs_vox = json.load(f)
    with (base_dir / "fastsurfer_int_spearmanr(vs_age).json").open(mode="r") as f:
        fs_int = json.load(f)
    return dict(fastsurfer_volume_dict=fs_vox,
                fastsurfer_intensity_dict=fs_int)


def load_metadata(vbm_dir: Path = C.VBM_DIR,
                  fs_dir: Path = C.FS_DIR) -> Dict[str, Dict[str, float] | Dict[str, Tuple[float, float]]]:
    vbm_dicts = load_vbm(base_dir=vbm_dir)
    fs_dicts = load_fastsurfer(base_dir=fs_dir)
    return {"VBM Young-to-old": vbm_dicts["young2old"],
            "VBM Old-to-Young": vbm_dicts["old2young"],
            "Fastsurfer Voxel": fs_dicts["fastsurfer_volume_dict"],
            "Fastsurfer Intensity": fs_dicts["fastsurfer_intensity_dict"]}


def load_interps() -> Dict[str, Dict[str, List[np.ndarray]]]:
    interps = defaultdict(dict)
    for xai_method in C.XAI_METHODS:
        for model_name in C.MODELS:
            if xai_method == "ig" and model_name == "convnext-base":
                continue
            seeds = [42, 43, 44] if model_name != "convnext-base" else [42, 43]
            lst = [wp.Weights(model_name=model_name, seed=seed, xai_method=xai_method, verbose=False).normalize_df()\
                   for seed in seeds]
            interps[xai_method][model_name] = lst
    return interps


def load_robustness(base_dir: Path = C.