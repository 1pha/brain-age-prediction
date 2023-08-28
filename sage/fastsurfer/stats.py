import os
from typing import Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.stats import linregress

from sage.data import UKBDataset
from .constants import META, SEG_ROOT
from .utils import get_seg


def get_name(idx: int) -> str:
    return META.iloc[idx].values[-1]


def get_name_bysegid(idx: int) -> str:
    idx = int(idx)
    return META[META.SegId == idx].StructName.iloc[0]


def get_idx_byname(name: str) -> int:
    return META[META.StructName == name].SegId.iloc[0]


def linefit(x, y) -> Tuple[np.ndarray, np.ndarray, dict]:
    r = linregress(x, y)
    xseq = np.linspace(min(x), max(x), num=100)
    return xseq, r.intercept + r.slope*xseq, r


def get_seg_results(root_dir: Path = SEG_ROOT) -> list:
    seg_results = sorted(list(root_dir.glob("*")))
    seg_results = list(filter(os.path.isdir, seg_results))
    return seg_results


def collect_stats(root_dir: Path = SEG_ROOT):
    """ Collecting stats file from fastsurfer result """
    seg_results = get_seg_results(root_dir=root_dir)
    stats = get_seg(path=seg_results[0])["stat"]
    stats = stats.drop(["Index", "SegId", "StructName"], axis=1).values.astype(np.float32)
    stats = stats[None, ...]

    pbar = tqdm(iterable=seg_results[1:], desc="Collecting Stats...")
    for path in pbar:
        _stat = get_seg(path=path)["stat"]
        _stat = _stat.drop(["Index", "SegId", "StructName"], axis=1).values.astype(np.float32)
        _stat = _stat[None, ...]
        
        stats = np.vstack([stats, _stat])
        
    # Sort stats by age, in ascending order
    ds = UKBDataset(mode="test")
    scan_names = list(map(lambda x: str(x.stem), ds.files))
    mask = ds.labels.fname.isin(scan_names)
    ages = ds.labels[mask].reset_index(drop=True)
    sort_age = ages.sort_values(by="age").index.tolist()
    stats = stats[sort_age]
    return stats
