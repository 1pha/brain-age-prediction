import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd

import calc
import constants as C
import weight_parser as wp


def _open_json(fname: Path | str, check_keys: bool = True) -> dict:
    with open(fname, mode="r") as f:
        dct = json.load(f)
    if check_keys:
        dct = {k: dct[k] for k in C.ROI_COLUMNS}
    return dct


def _open_pkl(fname: Path | str):
    with open(fname, mode="rb") as f:
        dct = pickle.load(f)
    return dct


def load_testdata(fname: Path = C.TESTFILE) -> pd.DataFrame:
    df = pd.read_csv(fname)
    return df


def load_vbm(base_dir: Path = C.VBM_DIR) -> Dict[str, Dict[str, float]]:
    """ Loads VBM stats result for young2old and old2young
    Threshold is 4.52"""
    cont_12_dict = _open_json(fname=base_dir / "12_old_young" / "vbm_dict.json")
    cont_21_dict = _open_json(fname=base_dir / "21_young_old" / "vbm_dict.json")

    cont_12_nii = nib.load(filename=base_dir / "12_old_young" / "threshold.nii")
    cont_21_nii = nib.load(filename=base_dir / "21_young_old" / "threshold.nii")
    return dict(young2old=cont_12_dict, old2young=cont_21_dict,
                young2old_nii=cont_12_nii, old2young_nii=cont_21_nii,)
    
    
def _process_single_combination(
    xai_method: str, model_name: str, seed: int, use_adni: bool = False
) -> Tuple[str, Dict, Dict]:
    """단일 조합(xai_method, model_name, seed)에 대해 처리하는 함수"""
    pk = f"{xai_method}|{model_name}|{seed}"
    attr_result = {}
    tattr_result = {}
    
    if use_adni:
        w = wp.WeightsCls(model_name=f"{model_name}-binary", seed=seed, xai_method=xai_method,
                          base_dir=C.ADNI_WEIGHT_DIR,
                          strict=False, verbose=False)
    else:
        w = wp.Weights(model_name=model_name, seed=seed, xai_method=xai_method,
                       strict=False, verbose=False)
    
    # vbm_sim_attr.json 처리
    print(pk)
    try:
        attr_q = _open_json(w.xai_path / "vbm_sim_attr.json", check_keys=False)
        attr_result = attr_q
    except FileNotFoundError:
        print(f"{pk} attrs not found")
    
    # vbm_sim_top_attr.json 처리
    try:
        tattr_q = _open_json(w.xai_path / "vbm_sim_top_attr.json", check_keys=False)
        tattr_result = tattr_q
    except FileNotFoundError:
        print(f"{pk} tattrs not found")
        
    return pk, attr_result, tattr_result


def load_vbm_overlap(xai_methods: List[str] = C.XAI_METHODS,
                     models: List[str] = C.MODELS,
                     seeds: List[int] = C.SEEDS,
                     use_adni: bool = False,
                     parallelize: bool = False,
                     n_jobs: int = -1):
    """
    vbm 중첩 데이터를 로드합니다. 선택적으로 병렬 처리를 사용할 수 있습니다.
    
    Parameters:
    -----------
    xai_methods: 사용할 XAI 메소드 리스트
    models: 사용할 모델 리스트
    seeds: 사용할 시드 값 리스트
    parallelize: 병렬 처리 사용 여부 (기본값: False)
    n_jobs: 병렬 처리에 사용할 CPU 코어 수 (-1은 모든 코어 사용, parallelize=True일 때만 사용됨)
    
    Returns:
    --------
    attrs, tattrs: 속성과 상위 속성 데이터를 담은 두 개의 딕셔너리
    """
    attrs = {}
    tattrs = {}
    
    if parallelize or n_jobs > 1:
        # 병렬 처리 방식
        try:
            # joblib이 설치되어 있는지 확인
            from joblib import Parallel, delayed
            
            # 모든 조합 생성
            combinations = [(xai_method, model_name, seed) 
                            for xai_method in xai_methods 
                            for model_name in models 
                            for seed in seeds]
            
            # 병렬 처리 실행
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_combination)(xai_method, model_name, seed, use_adni)
                for xai_method, model_name, seed in combinations
            )
            
            # 결과 처리
            for pk, attr_result, tattr_result in results:
                if attr_result:
                    attrs[pk] = attr_result
                if tattr_result:
                    tattrs[pk] = tattr_result
                    
        except ImportError:
            # joblib이 설치되어 있지 않은 경우 순차 처리로 대체
            print("Warning: joblib is not installed. Falling back to sequential processing.")
            parallelize = False
    
    else:
        # 순차 처리 방식 (원래 코드)
        for xai_method in xai_methods:
            for model_name in models:
                for seed in seeds:
                    pk, attr_result, tattr_result = _process_single_combination(
                        xai_method=xai_method, model_name=model_name,
                        seed=seed, use_adni=use_adni
                    )
                    attrs[pk] = attr_result
                    tattrs[pk] = tattr_result
    
    return attrs, tattrs
                

def load_vbm_adni(base_dir: Path = C.VBM_DIR) -> Dict[str, Dict[str, float]]:
    """ Loads VBM stats result for young2old and old2young """
    # vbm_dict = _open_json(fname=base_dir / "adni" / "vbm_dict.json")
    cont_nii = nib.load(filename=base_dir / "adni" / "threshold.nii")
    return dict(nii=cont_nii)


def load_vbm_age(base_dir: Path = C.VBM_DIR) -> pd.DataFrame:
    """Loads target patients used for VBM: their filelist and ages"""
    with (base_dir / "meta" / "old.txt").open(mode="r") as f:
        old = [[s.strip() for s in _.split(",")] for _ in f.readlines()][1:]
    with (base_dir / "meta" / "young.txt").open(mode="r") as f:
        young = [[s.strip() for s in _.split(",")] for _ in f.readlines()][1:]
    vbm_age = old + young
    vbm_age = pd.DataFrame(vbm_age, columns=["fname", "age"])
    vbm_age.age = vbm_age.age.astype(float).astype(int)
    return vbm_age


def load_fastsurfer(base_dir: Path = C.FS_DIR) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """ Loads SpearmanR result on age vs. Voxel/Intensity respectively """
    fs_vox = _open_json(base_dir / "fastsurfer_vox_spearmanr(vs_age).json")
    fs_int = _open_json(base_dir / "fastsurfer_int_spearmanr(vs_age).json")
    return dict(fastsurfer_volume_dict=fs_vox,
                fastsurfer_intensity_dict=fs_int)


def load_fastsurfer_adni(base_dir: Path = C.FS_DIR) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """ Loads SpearmanR result on age vs. Voxel/Intensity respectively """
    fs_vox_ttest = _open_json(base_dir / "fastsurfer_vox_ttest(adni).json")
    fs_vox_mwu = _open_json(base_dir / "fastsurfer_vox_mwu(adni).json")
    fs_int_ttest = _open_json(base_dir / "fastsurfer_int_ttest(adni).json")
    fs_int_mwu = _open_json(base_dir / "fastsurfer_int_mwu(adni).json")
    return dict(fastsurfer_volume_ttest_dict=fs_vox_ttest,
                fastsurfer_volume_mwu_dict=fs_vox_mwu,
                fastsurfer_intensity_ttest_dict=fs_int_ttest,
                fastsurfer_intensity_mwu_dict=fs_int_mwu)


def load_occlusion(base_dir: Path = C.OCC_DIR) -> Dict[str, Dict[str, float]]:
    occ_dict = {}
    for metric_key in ["MSE", "MAE", "R2"]:
        xai_dict = _open_json(fname=base_dir / f"{metric_key}_rel.json")
        occ_dict[metric_key] = xai_dict

    df = pd.read_csv(base_dir / "roi_metrics.csv")
    occ_dict["Absolute value"] = df
    return occ_dict


def load_metadata(
    vbm_dir: Path = C.VBM_DIR, fs_dir: Path = C.FS_DIR, occ_dir: Path = C.OCC_DIR
) -> Dict[str, Dict[str, float] | Dict[str, Tuple[float, float]]]:
    vbm_dicts = load_vbm(base_dir=vbm_dir)
    # Merged VBM
    merged_vbm = {k: max(abs(vbm_dicts["young2old"][k]), abs(vbm_dicts["old2young"][k])) \
        for k in vbm_dicts["young2old"].keys()}
    fs_dicts = load_fastsurfer(base_dir=fs_dir)
    occ_dicts = load_occlusion(base_dir=occ_dir)
    return {"VBM Young-to-old": vbm_dicts["young2old"],
            "VBM Old-to-Young": vbm_dicts["old2young"],
            "VBM": merged_vbm,
            "Fastsurfer Voxel": fs_dicts["fastsurfer_volume_dict"],
            "Fastsurfer Intensity": fs_dicts["fastsurfer_intensity_dict"]}
            # "Occlusion MSE": occ_dicts["MSE"],
            # "Occlusion MAE": occ_dicts["MAE"],
            # "Occlusion R2": occ_dicts["R2"],
            # "Occlusion absolute": occ_dicts["Absolute value"]}


def load_metadata_adni(
    vbm_dir: Path = C.VBM_DIR, fs_dir: Path = C.FS_DIR, occ_dir: Path = C.OCC_DIR
) -> Dict[str, Dict[str, float] | Dict[str, Tuple[float, float]]]:
    vbm_dicts = load_vbm_adni(base_dir=vbm_dir)
    fs_dicts = load_fastsurfer_adni(base_dir=fs_dir)
    return {"VBM ADNI": vbm_dicts["vbm"],
            "Fastsurfer Voxel t-test": fs_dicts["fastsurfer_volume_ttest_dict"],
            # "Fastsurfer Voxel MWU": fs_dicts["fastsurfer_volume_mwu_dict"],
            "Fastsurfer Intensity t-test": fs_dicts["fastsurfer_intensity_ttest_dict"],
            # "Fastsurfer Intensity MWU": fs_dicts["fastsurfer_intensity_mwu_dict"]
    }


def load_interps(load_indiv: bool = True,
                 normalize: bool = True,
                 mask: np.ndarray = None,
                 xai_methods: List[str] = C.XAI_METHODS,
                 models: List[str] = C.MODELS,
                 seeds: List[int] = C.SEEDS,
                 base_dir: str | Path = C.WEIGHT_DIR,
                 is_cls: bool = False,
                 strict: bool = True,
                 verbose: bool = False) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Hierarchy
    method1 (key):
        model1 (key): value=[
            seed1: (#Test scans, #RoI)
            seed2: (#Test scans, #RoI)
            ...
        ]
        model2 (key): value=[
            seed1: (#Test scans, #RoI)
            seed2: (#Test scans, #RoI)
            ...
        ]
    ...
    method2:
        model1:
            seed1: (#Test scans, #RoI)
            seed2: (#Test scans, #RoI)
            ...
        model2:
            seed1: (#Test scans, #RoI)
            seed2: (#Test scans, #RoI)
            ...
        ...
    ...
    """
    interps = defaultdict(dict)
    print(f"Loop over XAI    : {xai_methods}")
    print(f"Loop over MODELS : {models}")
    print(f"Loop over seeds  : {seeds}")
    if mask is not None:
        num_pat = mask.sum() if mask.dtype == bool else len(mask)
        print(f"Loop over {num_pat} patients")
    for xai_method in xai_methods:
        for model_name in models:
            lst = []
            for seed in seeds:
                if is_cls:
                    w = wp.WeightsCls(model_name=f"{model_name}-binary", seed=seed, xai_method=xai_method,
                                      base_dir=base_dir, strict=strict, verbose=verbose)
                else:
                    w = wp.Weights(model_name=model_name, seed=seed, xai_method=xai_method,
                                   base_dir=base_dir, strict=strict, verbose=verbose)
                if load_indiv:
                    dct = w.normalize_df(mask=mask) if normalize \
                          else w.load_xai_dict_indiv(mask=mask, return_np=True)
                else:
                    if mask is not None:
                        print(f"Mask given but will be ignored with `load_indiv=False` flag")
                    dct = w.xai_dict
                lst.append(dct)
            interps[xai_method][model_name] = lst
    return interps


def calculate_robustness(interps: Dict[str, Dict[str, List[np.ndarray]]]) -> pd.DataFrame:
    intra_cossim = calc.intra_robustness(interps=interps, method="cossim")
    intra_spear = calc.intra_robustness(interps=interps, method="spearmanr")
    intra_cka = calc.intra_robustness(interps=interps, method="cka")
    intra = pd.concat([intra_cossim, intra_spear, intra_cka])

    inter_cossim = calc.inter_robustness(interps=interps, method="cossim")
    inter_spear = calc.inter_robustness(interps=interps, method="spearmanr")
    inter_cka = calc.inter_robustness(interps=interps, method="cka")
    inter = pd.concat([inter_cossim, inter_spear, inter_cka])

    df = pd.concat([intra, inter])
    df.reset_index(drop=True, inplace=True)
    return df


def load_robustness(base_dir: Path = C.ANALYSIS_DIR,
                    scratch: bool = False,
                    verbose: bool = False,
                    is_cls: bool = False,
                    save: str = "") -> pd.DataFrame:
    if scratch:
        # Generate from scratch
        INTERPS = load_interps(verbose=verbose, base_dir=C.WEIGHT_DIR, is_cls=is_cls)
        df = calculate_robustness(interps=INTERPS)
        if save:
            df.to_csv(path_or_buf=save, index=False)
    else:
        fname = C.ASSET_DIR / "full_robustness_250312.csv"
        print(f"Load from {fname}")
        df = pd.read_csv(fname)
    return df


def load_robustness_xai(base_dir: Path = C.ANALYSIS_DIR,
                        scratch: bool = False,
                        verbose: bool = False,
                        is_cls: bool = False,
                        save: str = "") -> pd.DataFrame:
    if scratch:
        # Generate from scratch
        INTERPS = load_interps(verbose=verbose, is_cls=is_cls)
        df = calc.inter_robustness_xai(interps=INTERPS, method="spearmanr")
        if save:
            df.to_csv(path_or_buf=save, index=False)
    else:
        fname = C.ASSET_DIR / "full_xai_robustness_250106.csv"
        print(f"Load from {fname}")
        df = pd.read_csv(fname)
    return df


def load_alignment(base_dir: Path = C.ANALYSIS_DIR,
                   interps: Dict[str, Dict[str, List[np.ndarray]]] = None,
                   metadata: dict = None,
                   scratch: bool = False,
                   save: str = "") -> pd.DataFrame:
    """Alignment: Pre-calculated spearman of
    Saliency map against Fastsurfer"""
    if scratch:
        if interps is None:
            # Loads biobank
            interps = load_interps(load_indiv=True, normalize=True, verbose=False)
        if metadata is None:
            # Loads biobank
            metadata = load_metadata()
        
        alignment = calc.calculate_all_alignment(interps=interps, metadata=metadata)
        stack = []
        for key, _df in alignment.items():
            stack.append(_df)
        stack = pd.concat(stack)
        stack = stack.reset_index(drop=True)
        alignment = stack
        if save:
            alignment.to_csv(save, index=False)
    else:
        fname = C.ASSET_DIR / "full_alignment_241113.csv"
        # fname = C.ASSET_DIR / "full_alignment_241218_threshold.csv"
        print(f"Load from {fname}")
        alignment = pd.read_csv(fname)

    # cond = alignment["Similarity Method"].str.contains("Occlusion M")
    # alignment.loc[cond, C.YCOL] = - alignment[cond][C.YCOL].values

    cond = alignment["Similarity Method"] == "Fastsurfer Intensity"
    alignment.loc[cond, C.YCOL] = - alignment[cond][C.YCOL].values
    return alignment
