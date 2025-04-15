"""Calculating similarities
Deep learning XAI methods: Matrix in (#RoI, #Test patitnet)
Conventional stats methods: Vector in (#RoI, )

Deep XAI comparison should return 

Possible combinations
- mat vs. mat
- mat vs. vec
- vec vs. vec
"""
from typing import List, Dict, Tuple
from itertools import combinations

import pandas as pd
import numpy as np
from numpy import linalg
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform

import constants as C


def is_norm(v: np.ndarray, normalize: bool = False) -> np.ndarray | bool:
    assert v.ndim == 1, f"Provide vector. Input ndim={v.ndim}"
    norm = linalg.norm(x=v)
    _is_norm: bool = np.isclose(a=norm, b=1)
    if normalize:
        if not _is_norm:
            v = v / norm
        return v
    else:
        return _is_norm
    
    
def norm_mat(mat: np.ndarray, axis: int = 1, eps: float = 1e-8):
    nr, nc = mat.shape
    norm = np.linalg.norm(x=mat, axis=1).repeat(nc).reshape(nr, nc)
    mat = mat / (norm + eps)
    mat = np.nan_to_num(x=mat, nan=0, posinf=0, neginf=0)
    return mat


def _cossim_vv(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """ Vectors should be norm """
    vec1, vec2 = is_norm(v=vec1, normalize=True), is_norm(v=vec2, normalize=True)
    sim = vec1 @ vec2
    return sim


def _cossim_mv(mat: np.ndarray, vec: np.ndarray, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate simlarities of simlarity between mat row vector vs a single vector
    mat: (# RoIs, # Test scans)
    vec: (# RoIs)
    Function will return vector of similarities in (# Test scans)
    
    To make mat @ vec possible, mat should be transpose.
    This function checks possibility of matrix multiplication. """
    assert mat.ndim == 2, f"Entry matrix is not a matrix. Check ndim: {mat.ndim}"
    assert vec.ndim == 1, f"Entry vector is not a vector. Check ndim: {vec.ndim}"
    able = mat.shape[1] == vec.shape[0]
    if not able:
        mat = mat.T

    mat = norm_mat(mat)
    simvec = mat @ vec # (#scans)
    return np.nanmean(simvec) if return_avg else simvec


def _cossim_mm(mat1: np.ndarray, mat2: np.ndarray,
               num_patients: int = C.NUM_TEST, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate similarities between matrix and matrix
    Note that both matrix should have (# RoIs, # Test scans)
    The calculation is expected to return (# Test scans, # Test scans)
    
    Since we need diagnoal elements only, which indicates similarity between same scans,
    this will return vector of similarity in (# Test scans)
    Or with return_avg flag, we will get single float of similarity """
    # Sanity check
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert num_patients in mat1.shape, f"Entry matrix1 does not have num_patients shape ({num_patients}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert num_patients in mat2.shape, f"Entry matrix2 does not have num_patients shape ({num_patients}). Check shape: {mat2.shape}"

    if mat1.shape[0] != num_patients:
        # mat1: (# Test scans, # RoIs)
        mat1 = mat1.T
    if mat2.shape[0] != num_patients:
        # mat2: (# RoIs, # Test Scans)
        mat2 = mat2.T

    mat1, mat2 = norm_mat(mat=mat1), norm_mat(mat=mat2)
    simmat = mat1 @ mat2.T
    simvec = np.diag(simmat) # (#Testscans, )
    return np.nanmean(simvec) if return_avg else simvec


def _spear_vv(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """ Vectors should be norm """
    vec1, vec2 = is_norm(v=vec1, normalize=True), is_norm(v=vec2, normalize=True)
    stat, pval = spearmanr(a=vec1, b=vec2)
    return stat


def _spear_mv(mat: np.ndarray, vec: np.ndarray, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate simlarities of simlarity between mat row vector vs a single vector
    mat: (# RoIs, # Test scans)
    vec: (# RoIs)
    Function will return vector of similarities in (# Test scans)
    
    To make mat @ vec possible, mat should be transpose.
    This function checks possibility of matrix multiplication.
    
    TODO: Check C.NUM_TEST not done here. """
    # Sanity check
    assert mat.ndim == 2, f"Entry matrix is not a matrix. Check ndim: {mat.ndim}"
    assert vec.ndim == 1, f"Entry vector is not a vector. Check ndim: {vec.ndim}"

    able = mat.shape[1] == vec.shape[0]
    if not able:
        mat = mat.T
    simvec = np.array([_spear_vv(vec1=_vec, vec2=vec) for _vec in mat]) # (#scans)
    return np.nanmean(simvec) if return_avg else simvec


def _spear_mm(mat1: np.ndarray, mat2: np.ndarray,
              num_patients: int = C.NUM_TEST, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate similarities between matrix and matrix
    Note that both matrix should have (# RoIs, # Test scans)
    The calculation is expected to return (# Test scans, # Test scans)
    
    Since we need diagnoal elements only, which indicates similarity between same scans,
    this will return vector of similarity in (# Test scans)
    Or with return_avg flag, we will get single float of similarity """
    # Sanity check
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert num_patients in mat1.shape, f"Entry matrix1 does not have num_patients shape ({num_patients}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert num_patients in mat2.shape, f"Entry matrix2 does not have num_patients shape ({num_patients}). Check shape: {mat2.shape}"

    if mat1.shape[0] != num_patients:
        # mat1: (# Test scans, # RoIs)
        mat1 = mat1.T
    if mat2.shape[0] != num_patients:
        # mat2: (# RoIs, # Test Scans)
        mat2 = mat2.T
    simvec = np.array([_spear_vv(vec1=vec1, vec2=vec2) for vec1, vec2 in zip(mat1, mat2)])
    return np.nanmean(simvec) if return_avg else simvec


def _cka_mm(mat1: np.ndarray, mat2: np.ndarray, sigma: float = 1.0,
            num_patients: int = C.NUM_TEST, return_avg: bool = False) -> float | np.ndarray:
    # Sanity check
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert num_patients in mat1.shape, f"Entry matrix1 does not have num_patients shape ({num_patients}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert num_patients in mat2.shape, f"Entry matrix2 does not have num_patients shape ({num_patients}). Check shape: {mat2.shape}"

    if mat1.shape[0] != num_patients:
        # mat1: (# Test scans, # RoIs)
        mat1 = mat1.T
    if mat2.shape[0] != num_patients:
        # mat2: (# RoIs, # Test Scans)
        mat2 = mat2.T
    
    # 마스크 적용하여 NaN 제거
    # Calculate RBF kernel matrices
    if sigma is None:
        # Median heuristic for kernel width
        X_norm = np.sqrt(np.sum(X**2, axis=1))
        distances = squareform(pdist(X_norm.reshape(-1, 1)))
        sigma = np.median(distances[distances > 0])
        
    # Apply RBF kernel
    gamma = 1 / (2 * sigma**2)
    K = mat1 @ mat1.T
    L = mat2 @ mat2.T
    
    # Center kernel matrices
    n = K.shape[0]
    H = np.eye(n) - 1/n * np.ones((n, n))
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Calculate HSIC
    hsic = np.sum(K_centered * L_centered)
    
    # Calculate normalization
    normalization = np.sqrt(np.sum(K_centered * K_centered) * np.sum(L_centered * L_centered))
    
    if normalization < 1e-10:
        return 0.0
    
    return hsic / normalization


def compute_RDM(X):
    """
    Compute Representational Dissimilarity Matrix.
    
    Args:
        X: Matrix of shape (num_samples, num_features)
        
    Returns:
        RDM of shape (num_samples, num_samples)
    """
    return squareform(pdist(X, metric='correlation'))


def _rsa_mm(mat1: np.ndarray, mat2: np.ndarray, method: str = "spearmanr",
            num_patients: int = C.NUM_TEST, return_avg: bool = False) -> float | np.ndarray:
    # Sanity check
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert num_patients in mat1.shape, f"Entry matrix1 does not have num_patients shape ({num_patients}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert num_patients in mat2.shape, f"Entry matrix2 does not have num_patients shape ({num_patients}). Check shape: {mat2.shape}"
    
    # Compute RDMs
    rdm_X = compute_RDM(mat1)
    rdm_Y = compute_RDM(mat2)
    
    # Extract upper triangular part (excluding diagonal)
    triu_indices = np.triu_indices_from(rdm_X, k=1)
    rdm_X_vec = rdm_X[triu_indices]
    rdm_Y_vec = rdm_Y[triu_indices]
    
    # Compute correlation
    if method == 'spearman':
        corr, p = spearmanr(rdm_X_vec, rdm_Y_vec)
    else:  # pearson
        corr, p = pearsonr(rdm_X_vec, rdm_Y_vec)
    
    return corr, p


def simcalc(interp1: np.ndarray, interp2: np.ndarray,
            method: str = "spearmanr", return_avg: bool = False) -> np.ndarray | float:
    """ Similarity calculation betwen matrix/vector """
    def d2n(v: np.ndarray | dict):
        """Safe dict2numpy"""
        if isinstance(v, dict):
            v = np.array([v[k] for k in v])
        return v
    interp1, interp2 = d2n(interp1), d2n(interp2)
    dt1, dt2 = interp1.ndim, interp2.ndim
    
    func_name = {"spearmanr": "spear", "cossim": "cossim",
                 "cka": "cka", "rsa": "rsa"}[method]
    if (dt1, dt2) == (1, 1):
        _simcalc = lambda i1, i2: eval(f"_{func_name}_vv")(vec1=i1, vec2=i2)
    elif (dt1, dt2) == (2, 1):
        _simcalc = lambda i1, i2: eval(f"_{func_name}_mv")(mat=i1, vec=i2, return_avg=return_avg)
    elif (dt1, dt2) == (2, 2):
        if (C.NUM_TEST in interp1.shape) and (C.NUM_TEST in interp2.shape):
            num_patients = C.NUM_TEST
        else:
            # Assume interp1.shape[0] is num_patients
            num_patients = interp1.shape[0]
        _simcalc = lambda i1, i2: eval(f"_{func_name}_mm")(mat1=i1, mat2=i2,
                                                           num_patients=num_patients,
                                                           return_avg=return_avg)
    else:
        print(f"Please check dimensions of input vectors: {interp1.shape}, {interp2.shape}")
        return None

    sim = _simcalc(i1=interp1, i2=interp2)
    return sim


def _group_aligncalc(group: List[np.ndarray], vector: np.ndarray) -> List[float]:
    """Alignment calculation between Saliency maps from certain model group (seeds)
    vs conventional vector from method.
    
    Saliency mat: (# Test brain, # RoI)
    conventional vec: (# RoI)
    
    Each seed will be calculated via `_spear_mv`
    result: mat @ vec -> average -> float
    Will always calculate SpearmanR and no L2-sim,
    since yielded values does not come from similar ways
    """
    sims = [simcalc(interp1=g, interp2=vector, return_avg=True) for g in group]
    return sims


def _group_simcalc(group: List[np.ndarray],
                   target_group: List[np.ndarray] = None,
                   method: str = "spearmanr",
                   return_avg: bool = True) -> Tuple[List[float], List[Tuple[int, int]]]:
    """Calculates similarity between groups
    Group: Experiemnts with different seeds within the same models
    Returns similarities of
    1. Similarity In-between models: will return nC2 (if num_seeds(=n)=3, returns len=3 list)
    2. Similarty vs. Model groups: return num_seeds * num_seeds' list of similarities
    """
    sims, corresponding_index = [], []
    if target_group is None:
        # Intra-calculation: list of similarities between same model architecture
        comb_index = combinations(iterable=range(len(group)), r=2)
        for idx1, idx2 in comb_index:
            sim = simcalc(interp1=group[idx1], interp2=group[idx2],
                          method=method, return_avg=return_avg)
            if return_avg:
                sims.append(sim)
            else:
                sims.extend(sim)
            corresponding_index.append((idx1, idx2))
    else:
        for idx1 in range(len(group)):
            for idx2 in range(len(target_group)):
                sim = simcalc(interp1=group[idx1], interp2=target_group[idx2],
                              method=method, return_avg=return_avg)
                if return_avg:
                    sims.append(sim)
                else:
                    sims.extend(sim)
                corresponding_index.append((idx1, idx2))
    if return_avg:
        assert len(sims) == len(corresponding_index),\
            f"Make sure number of similarity is same as index pairs"
    else:
        assert len(sims) / C.NUM_TEST == len(corresponding_index),\
            f"Make sure number of similarity is same as index pairs"
    return sims, corresponding_index


def convert_dict2df(dct: dict,
                    method: str | List[str],
                    col_name: str | List[str] = C.HUECOL) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(dct, orient="index").T
    df = df.melt(value_vars=df.columns.tolist())
    if isinstance(method, str):
        df[col_name] = method
    elif isinstance(method, list):
        for col, met in zip(col_name, method):
            # single element that applies to all rows
            # OR list of elements equals to full df
            df[col] = met

    df = df.rename({"variable": C.XCOL, "value": C.YCOL}, axis=1)
    df = df.replace(to_replace=C.XAI_METHODS_MAPPER)
    return df


def intra_robustness(interps: Dict[str, Dict[str, List[np.ndarray]]],
                     method: str = "spearmanr") -> pd.DataFrame:
    """For input, refer to metadata.py/load_interps()
    This will get internal robustness, which refers to list of similarities between
    same models with difference seeds.
    
    This will in turn generate the following:
        XAI Method | Similarity | Similarity Method
    1   IG         | 0.87       | `method`
    ...
    """
    print(f"Start calculating Intra-robustness: {method}")
    rd = []
    for xai in interps:
        print(f"\t{xai}")
        for model in interps[xai]:
            sim, index_pair = _group_simcalc(group=interps[xai][model],
                                             method=method, return_avg=True)
            for _sim, (idx1, idx2) in zip(sim, index_pair):
                data = [xai, _sim, f"Intra: {method}", f"{model}-{idx1}", f"{model}-{idx2}"]
                rd.append(data)
    df = pd.DataFrame(rd, columns=[C.XCOL, C.YCOL, "Similarity Method", "Source", "Target"])
    return df


def inter_robustness(interps: Dict[str, Dict[str, List[np.ndarray]]],
                     method: str = "spearmanr") -> pd.DataFrame:
    """For input, refer to metadata.py/load_interps()
    This will get internal robustness, which refers to list of similarities between
    same models with difference seeds.
    
    This will in turn generate the following:
        XAI Method | Similarity | Similarity Method
    1   IG         | 0.87       | `method`
    ...
    """
    rd = [] # Robustness Dictionary
    print(f"Start calculating Inter-robustness: {method}")
    for xai in interps:
        print(f"\t{xai}")
        comb_index = combinations(iterable=interps[xai].keys(), r=2)
        for model, tgt_model in comb_index:
            sim, index_pair = _group_simcalc(group=interps[xai][model],
                                             target_group=interps[xai][tgt_model],
                                             method=method, return_avg=True)
            for _sim, (idx1, idx2) in zip(sim, index_pair):
                data = [xai, _sim, f"Inter: {method}", f"{model}-{idx1}", f"{tgt_model}-{idx2}"]
                rd.append(data)
    df = pd.DataFrame(rd, columns=[C.XCOL, C.YCOL, "Similarity Method", "Source", "Target"])
    return df


def inter_robustness_xai(interps: Dict[str, Dict[str, List[np.ndarray]]],
                         method: str = "spearmanr") -> pd.DataFrame:
    """For input, refer to metadata.py/load_interps()
    Robustness with fixed model-seed, compare across saliency methods
    
    This will in turn generate the following:
        Model      | Similarity | Similarity Method | Source   | Target
    1   resnet10-0 | 0.87       | `method`          | DeepLIFT | IG
    ...
    """
    rd = [] # Robustness Dictionary
    print(f"Start calculating Inter-robustness (XAI): {method}")
    
    xai_list = list(interps.keys())
    assert xai_list, f"Empty xai list. Check interpretability matrix `interps`"
    model_list = list(interps[xai_list[0]].keys())
    # No assertion code that forces all models saliency methods exist in different xai
    NUM_SEEDS = range(10)
    for model in model_list:
        for seed in NUM_SEEDS:
            xai_array = [interps[xai][model][seed] for xai in xai_list]
            sim, index_pair = _group_simcalc(group=xai_array, method=method, return_avg=True)
            for _sim, (idx1, idx2) in zip(sim, index_pair):
                data = [f"{model}-{seed}", _sim, method, xai_list[idx1], xai_list[idx2]]
                rd.append(data)
    df = pd.DataFrame(rd, columns=["Model", C.YCOL, "Similarity Method", "Source", "Target"])
    return df


def group_alignment(interps: Dict[str, Dict[str, List[np.ndarray]]],
                    vector: np.ndarray, method: str, mask: np.ndarray = None) -> pd.DataFrame:
    """Calculates alignment between conventional methods and saliency vector
    """
    ad = dict()
    models = []
    for xai in interps:
        print(f"{xai}")
        sims = []
        for model in interps[xai]:
            group: np.ndarray = interps[xai][model]
            if isinstance(group, dict):
                group = np.array([group[key] for key in C.ROI_COLUMNS])
            # similarities from all seeds per `model`
            if mask is not None:
                # group: (# RoI, # Test subj)
                group, v = [g[:, mask] for g in group], vector[mask]
            else:
                v = vector.copy()
            sim: List[float] = _group_aligncalc(group=group, vector=v)
            sims.extend(sim)

            _m = [f"{model}-{idx}" for idx in range(len(sim))]
            models.extend(_m)
        ad[xai] = sims
    # For saliency vs. conventional, method will be the single meta_key
    df = convert_dict2df(dct=ad, method=[method, models], col_name=[C.HUECOL, "Model Key"])
    return df


def calculate_all_alignment(
    interps: Dict[str, Dict[str, List[np.ndarray]]],
    metadata: Dict[str, Dict[str, float] | Dict[str, Tuple[float, float]]],
    use_threshold: bool = True,
) -> Dict[str, pd.DataFrame]:
    def _check_type(vector: np.ndarray) -> np.ndarray:
        """Metadata in general will be a vector but in some cases they are in a different format
        Check the dataformat of metadata and convert it to analyzable format
        Assume p-values are given.
        """
        if isinstance(vector, dict):
            # VBM stats
            key = list(vector.keys())[0]
            val = vector[key]
            if isinstance(val, float):
                vector = np.array([vector[key] for key in C.ROI_COLUMNS])
            elif isinstance(val, tuple | list):
                # Fastsurfer values contain {RoI: (slope, pval)}
                # Return p-value
                # P-value: needs reverse
                vector = - np.array([vector[key][0] for key in C.ROI_COLUMNS])
        elif isinstance(vector, np.ndarray):
            pass
        else:
            vector = None
        return vector

    ad = dict()
    for meta_key in metadata:
        vector = metadata[meta_key]
        vector = _check_type(vector=vector)
        if vector is None:
            # Non analyzable vector
            print(f"Skip {meta_key}.")
            continue
        
        # Finding Threshold
        if meta_key in ["VBM Young-to-old", "VBM Old-to-Young"]:
            # UKB VBM Threshold given
            # Threshold based on t-statistics from vbm result.
            # 인줄 알았으나 p-val이 들어있네.. 그거미만으로 가야..?
            threshold = C.UKB_FEW_THD
            mask = vector > threshold
            breakpoint()
        elif meta_key == "VBM ADNI":
            # ADNI VBM Threshold given
            # Threshold based on t-statistics from vbm result.
            threshold = C.ADNI_FEW_THD
            mask = vector > threshold
        elif meta_key.startswith("Fastsurfer"):
            # Fastsurfer tests given
            # Threshold based on p-value
            threshold = 0.05
            mask = np.array([metadata[meta_key][key][1] < threshold for key in C.ROI_COLUMNS])
        else:
            if use_threshold:
                print(f"No availble threhsolding for given {meta_key}")
                raise
            mask = None

        sim_df: pd.DataFrame = group_alignment(interps=interps,
                                               vector=vector,
                                               method=meta_key,
                                               mask=mask if use_threshold else None)
        ad[meta_key] = sim_df
    return ad
