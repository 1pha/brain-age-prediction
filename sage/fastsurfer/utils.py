from typing import Union
from pathlib import Path

import pandas as pd
import nibabel as nib


def _split(s: Union[str, list]) -> list:
    """ Splitting stat files
    Since each line contains """
    if isinstance(s, str):
        s = s.split(" ")
    s = list(filter(lambda x: x, s))
    s[-1] = s[-1].rstrip("\n")
    return s


def _get_segmask(path: Path):
    fname = path / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    segmask = nib.load(fname)
    return segmask


def _get_stat(path: Path):
    """ Read .stat files
    Get table only """
    fname = path / "stats" / "aseg+DKT.stats"
    with open(file=fname, mode="r") as f:
        stats = f.readlines()[-101:]
    cols = _split(stats.pop(0))[2:]
    stats = list(map(lambda x: _split(x), stats))
    
    df = pd.DataFrame(stats, columns=cols)
    return df


def get_seg(path: Path) -> dict:
    seg_mask = _get_segmask(path=path)
    seg_stat = _get_stat(path=path)
    return dict(mask=seg_mask, stat=seg_stat)



