__all__ = ["get_today"]

import os
import random
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch


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


def seed_everything(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
    """
    For stratified sampling by columns
    Just put -
        df       : pandas.DataFrame
        col      : string of the column,
        n_samples: the number of samples you want from each column category
    """
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=42))
    df_.index = df_.index.droplevel(0)
    return df_


# label = pd.read_csv("../rsc/age_ixidlbsoas13.csv", index_col=0)


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


# FNAMES = stratified_sample_df(label, "src", 3).apply(path_maker, axis=1).values
