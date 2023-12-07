import json
import logging
import math
from pathlib import Path
import pickle
from time import time
from typing import Dict, Tuple, Iterable

import numpy as np


def get_logger(name: str = None, filehandler: bool = False):
    name = name or __name__
    logging.basicConfig()
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    if filehandler:
        fname = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}.log"
        logger.addHandler(logging.FileHandler(filename=fname))
    return logger


logger = get_logger(name=__file__)


def load_json(path: Path):
    with path.open(mode="r") as f:
        dct = json.load(f)
    return dct


def load_pkl(path: Path):
    with path.open(mode="rb") as f:
        dct = pickle.load(f)
    return dct


def to_list(dct: Dict[str, float | Tuple[float, float]]) -> list:
    """ Converts dictionary into flatten list, omitting nan values.
    This is for statisitcal tests """
    init_key = list(dct.keys())[0]
    if isinstance(dct[init_key], float):
        lst = [v for v in dct.values() if not math.isnan(v)]
    elif isinstance(dct[init_key], Iterable):
        # Assume first element is the mean value
        lst = [v[0] for v in dct.values() if not math.isnan(v[0])]
    else:
        logger.warn("Check your dict: %s", dct)
    return lst


def check_nan(lst: list) -> bool:
    """" Warn if nan is contained inside a given list.
    Does NOT raise any errors, only the warning . """
    is_nan = np.any(np.isnan(lst))
    return is_nan