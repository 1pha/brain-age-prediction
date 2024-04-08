"""Calculating similarities
Deep learning XAI methods: Matrix in (#RoI, #Test patitnet)
Conventional stats methods: Vector in (#RoI, )

Deep XAI comparison should return 

Possible combinations
- mat vs. mat
- mat vs. vec
- vec vs. vec
"""
from typing import List, Dict
from itertools import combinations

import pandas as pd
import numpy as np
from numpy import linalg
from scipy.stats import spearmanr

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


def _cossim_mm(mat1: np.ndarray, mat2: np.ndarray, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate similarities between matrix and matrix
    Note that both matrix should have (# RoIs, # Test scans)
    The calculation is expected to return (# Test scans, # Test scans)
    
    Since we need diagnoal elements only, which indicates similarity between same scans,
    this will return vector of similarity in (# Test scans)
    Or with return_avg flag, we will get single float of similarity """
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert C.NUM_TEST in mat1.shape, f"Entry matrix1 does not have C.NUM_TEST shape ({C.NUM_TEST}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert C.NUM_TEST in mat2.shape, f"Entry matrix2 does not have C.NUM_TEST shape ({C.NUM_TEST}). Check shape: {mat2.shape}"
    if mat1.shape[0] != C.NUM_TEST:
        # mat1: (# Test scans, # RoIs)
        mat1 = mat1.T

    if mat2.shape[0] != C.NUM_TEST:
        # mat2: (# RoIs, # Test Scans)
        mat2 = mat2.T

    mat1, mat2 = norm_mat(mat=mat1), norm_mat(mat=mat2)
    simmat = mat1 @ mat2.T
    simvec = np.diag(simmat) # (# Test scans)
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


def _spear_mm(mat1: np.ndarray, mat2: np.ndarray, return_avg: bool = False) -> float | np.ndarray:
    """ Calculate similarities between matrix and matrix
    Note that both matrix should have (# RoIs, # Test scans)
    The calculation is expected to return (# Test scans, # Test scans)
    
    Since we need diagnoal elements only, which indicates similarity between same scans,
    this will return vector of similarity in (# Test scans)
    Or with return_avg flag, we will get single float of similarity """
    # Sanity check
    assert mat1.ndim == 2, f"Entry matrix1 is not a matrix. Check ndim: {mat1.ndim}"
    assert C.NUM_TEST in mat1.shape, f"Entry matrix1 does not have C.NUM_TEST shape ({C.NUM_TEST}). Check shape: {mat1.shape}"
    assert mat2.ndim == 2, f"Entry matrix2 is not a matrix. Check ndim: {mat2.ndim}"
    assert C.NUM_TEST in mat2.shape, f"Entry matrix2 does not have C.NUM_TEST shape ({C.NUM_TEST}). Check shape: {mat2.shape}"

    if mat1.shape[0] != C.NUM_TEST:
        # mat1: (# Test scans, # RoIs)
        mat1 = mat1.T
    if mat2.shape[0] != C.NUM_TEST:
        # mat2: (# RoIs, # Test Scans)
        mat2 = mat2.T
    simvec = np.array([_spear_vv(vec1=vec1, vec2=vec2) for vec1, vec2 in zip(mat1, mat2)])
    return np.nanmean(simvec) if return_avg else simvec


def simcalc(interp1: np.ndarray, interp2: np.ndarray,
            method: str = "spearmanr", return_avg: bool = False) -> np.ndarray | float:
    """ Similarity calculation betwen matrix/vector """
    dt1, dt2 = interp1.ndim, interp2.ndim
    
    func_name = {"spearmanr": "spear", "cossim": "cossim"}[method]
    if (dt1, dt2) == (1, 1):
        _simcalc = lambda i1, i2: eval(f"_{func_name}_vv")(vec1=i1, vec2=i2)
    elif (dt1, dt2) == (2, 1):
        _simcalc = lambda i1, i2: eval(f"_{func_name}_mv")(mat=i1, vec=i2, return_avg=return_avg)
    elif (dt1, dt2) == (2, 2):
        _simcalc = lambda i1, i2: eval(f"_{func_name}_mm")(mat1=i1, mat2=i2, return_avg=return_avg)
    else:
        print(f"Please check dimensions of input vectors: {interp1.shape}, {interp2.shape}")
        return None

    sim = _simcalc(i1=interp1, i2=interp2)
    return sim


def _group_simcalc(group: List[np.ndarray], target_group: List[np.ndarray] = None,
                   method: str = "spearmanr", return_avg: bool = True) -> List[float]:
    """Calculates similarity between groups
    Group: Experiemnts with different seeds within the same models
    Returns similarities of
    1. Similarity In-between models: will return nC2 (if num_seeds(=n)=3, returns len=3 list)
    2. Similarty vs. Model groups: return num_seeds * num_seeds' list of similarities
    """
    sims = []
    if target_group is None:
        comb_index = combinations(iterable=range(len(group)), r=2)
        for idx1, idx2 in comb_index:
            sim = simcalc(interp1=group[idx1], interp2=group[idx2],
                          method=method, return_avg=return_avg)
            sims.append(sim)
    else:
        for idx1 in range(len(group)):
            for idx2 in range(len(target_group)):
                sim = simcalc(interp1=group[idx1], interp2=target_group[idx2],
                              method=method, return_avg=return_avg)
                sims.append(sim)
    return sims


def convert_dict2df(dct: dict, method: str) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(dct, orient="index").T
    df = df.melt(value_vars=df.columns.tolist())
    df[C.HUECOL] = method
    
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
    rd = dict() # Robustness Dictionary
    for xai in interps:
        print(f"{xai}")
        sims = []
        for model in interps[xai]:
            sim: List[float] = _group_simcalc(group=interps[xai][model],
                                              method=method, return_avg=True)
            sims.extend(sim)
        rd[xai] = sims
    df = convert_dict2df(dct=rd, method=method)
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
    rd = dict() # Robustness Dictionary
    for xai in interps:
        print(f"{xai}")
        sims = []
        for model in interps[xai]:
            for tgt_model in interps[xai]:
                if model != tgt_model:
                    sim: List[float] = _group_simcalc(group=interps[xai][model],
                                                      target_group=interps[xai][tgt_model],
                                                      method=method, return_avg=True)
            sims.extend(sim)
        rd[xai] = sims
    df = convert_dict2df(dct=rd, method=method)
    return df
