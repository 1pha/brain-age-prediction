__all__ = ["get_today"]

import os
import ast
import time
import random
import logging
import importlib
from pathlib import Path
from datetime import datetime as dt
from typing import FrozenSet
import cProfile
import functools

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn


def parse_bool(s: str) -> bool:
    if s in ["true", "t", "y", 1, "1"]:
        return True
    elif s in ["false", "f", "no", "n", "0", 0, "None", "none"]:
        return False
    elif s in ["True", "False"]:
        return ast.literal_eval(s)
    else:
        return s


def get_logger(name: str = None, filehandler: bool = False):
    name = name or __name__
    logging.basicConfig()
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    if filehandler:
        fname = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}.log"
        logger.addHandler(logging.FileHandler(filename=fname))
    return logger


def check_exists(path: str | Path) -> str | Path:
    assert os.path.exists(path), f"{path} does not exist."
    return path


def count_parameters(model: nn.Module):
    # Reference
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # If the above line is uncommented, we get the following RuntimeError:
    #  max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
    torch.backends.cudnn.benchmark = False


def parse_hydra(config: omegaconf, **kwargs):
    """ Sometimes hydra.utils.instantiate does not work.
    Use this function to instantiate a new class
    """
    _cfg: list = config.pop("_target_").split(".")
    lib, cls_ = ".".join(_cfg[:-1]), _cfg[-1]
    cls_ = importlib.import_module(lib)
    cls_ = getattr(lib, cls_)
    config.update(kwargs)
    inst = cls_(**config)
    return inst


def get_func_name(config: omegaconf.DictConfig) -> str:
    if "_target_" in config:
        target = config._target_
        name = target.split(".")[-1]
    else:
        name = ""
    return name


def load_hydra(config_name: str,
               config_path: str = "config",
               overrides: FrozenSet[str] = None):
    with hydra.initialize(config_path=config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


def get_today():
    td = dt.today()
    return (
        str(td.year)
        + str(td.month).zfill(2)
        + str(td.day).zfill(2)
        + "-"
        + str(td.hour).zfill(2)
        + str(td.minute).zfill(2)
    )


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()

        result = original_fn(*args, **kwargs)

        end_time = time.time()
        end = "" if original_fn.__name__ == "train" else "\n"
        print(f"[{original_fn.__name__}] {end_time - start_time:.1f} sec ", end=end)
        return result

    return wrapper_fn


def stratified_sample_df(df, col, n_samples):
    """ For stratified sampling by columns
    Just put -
        df       : pandas.DataFrame
        col      : string of the column,
        n_samples: the number of samples you want from each column category
    """
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=42))
    df_.index = df_.index.droplevel(0)
    return df_


def path_maker(row):
    brain_id = row.id
    src = row.src

    if src == "Oasis3":
        SUFFIX = ".nii-brainmask.nii"

    else:
        SUFFIX = "-brainmask.nii"

    ROOT = "../../brainmask_nii/"
    path = ROOT + brain_id + SUFFIX
    return path if os.path.exists(path) else brain_id


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        print(f"Profiling results for {func.__name__}:")
        profiler.print_stats(sort='cumulative')
        return result
    return wrapper
