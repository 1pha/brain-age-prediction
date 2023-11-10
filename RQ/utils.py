import json
import logging
from pathlib import Path
import pickle
from time import time


def get_logger(name: str = None, filehandler: bool = False):
    name = name or __name__
    logging.basicConfig()
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    if filehandler:
        fname = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}.log"
        logger.addHandler(logging.FileHandler(filename=fname))
    return logger


def load_json(path: Path):
    with path.open(mode="r") as f:
        dct = json.load(f)
    return dct


def load_pkl(path: Path):
    with path.open(mode="rb") as f:
        dct = pickle.load(f)
    return dct