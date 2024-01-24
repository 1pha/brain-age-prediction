""" Includes model inference, output sorting utils
"""
import math
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_theme()
import torch
import torchmetrics.functional as tmf

from sage.utils import get_logger


logger = get_logger(name=__file__)


def tune(batch_size: int = 64,
         logging_interval: int = None,
         lr_frequency: int = None,
         accumulate_grad_batches: int = None,
         multiplier: int = 1,
         BASE_BATCH: int = 32):
    """ Tune logging interval to log same number for varying batch sizes
    BASE: with batch size 64, logging_interval 50
    i.e. with a given batch size 4, freq step should increase to 16
    
    Tune learning rate stepping frequency
    to step same number for varying batch sizes (logging)
    BASE: with batch size 64, steps 1
    i.e. with a given batch size 4, freq step should increase to 16
    """
    ratio = BASE_BATCH / batch_size
    if logging_interval is not None:
        BASE_INTERVAL = 50
        logging_interval = round(BASE_INTERVAL * ratio)
        return logging_interval
        
    if lr_frequency is not None:
        assert accumulate_grad_batches is not None, f"Please provide `accumulate_grad_batches`"
        lr_frequency = round(lr_frequency * ratio / accumulate_grad_batches)
        return lr_frequency
    
    if multiplier is not None:
        BASE_MULTIPLIER = 1
        multiplier = round(BASE_MULTIPLIER * ratio)
        return multiplier
    
    return ratio


def _sort_outputs(outputs):
    try:
        result = dict()
        keys: list = outputs[0].keys()
        for key in keys:
            data = outputs[0][key]
            if data.ndim == 0:
                # Scalar value result
                result[key] = torch.stack([o[key] for o in outputs if key in o])
            elif data.ndim in [1, 2]:
                # Batched 
                result[key] = torch.concat([o[key] for o in outputs if key in o])
    except:
        breakpoint()
    return result


def timestamp(fmt: str = "%y%m%d_%H%M") -> str:
    now = datetime.now()
    now = now.strftime(fmt)
    return now


def finalize_inference(prediction: list,
                       name: str,
                       root_dir: Path = Path(".")) -> None:
    """ Takes non-sorted prediction (=list of dicts)
        1. Flatten predictions into dict
        2. Saves & Logs prediction
        3. Plot prediction with jointplot / kde plot and save plots
    """
    root_dir = Path(root_dir)

    # 1. Sort Prediction
    prediction = _sort_outputs(prediction)

    # 2. Save PRediction
    save_name = f"{name}.pkl"
    logger.info("Save prediction as %s", save_name)
    with open(root_dir / save_name, "wb") as f:
        pickle.dump(prediction, f)

    # 2. Log Predictions
    run_name = save_name[:-4] + "_" + timestamp()
    preds, target = prediction["pred"], prediction["target"]
    if name.startswith("C"):
        logger.info("Classification data given:")
        _cls_inference(preds=preds, target=target, root_dir=root_dir, run_name=run_name)
    elif name[0] in set(["R", "M"]):
        logger.info("Regression data given:")
        _reg_inference(preds=preds, target=target, root_dir=root_dir, run_name=run_name)
        _get_norm_cf_reg(preds=preds, target=target, root_dir=root_dir, run_name=run_name)
    else:
        logger.info("Failed to inference. Check the run name for the task.")


def _reg_inference(preds, target, root_dir, run_name) -> None:
    mse = tmf.mean_squared_error(preds=preds, target=target)
    mae = tmf.mean_absolute_error(preds=preds, target=target)
    r2 = tmf.r2_score(preds=preds, target=target)
    logger.info("Results as follow:")
    logger.info("MSE: %.3f", mse)
    logger.info("MAE: %.3f", mae)
    logger.info("R2 : %.4f", r2)

    # 3. Plot Jointplot
    data = pd.DataFrame({"Prediction": preds.numpy(),
                         "Target": target.numpy()})
    
    p = sns.jointplot(data=data,
                      x="Prediction", y="Target",
                      xlim=[43, 87], ylim=[43, 87])
    p.fig.suptitle(run_name)
    p.fig.tight_layout()
    p.savefig(root_dir / f"{run_name}-joint.png")

    # 3. Plot KDE plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(data=data, ax=ax)
    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(root_dir / f"{run_name}-kde.png")
    

def _get_norm_cf_reg(preds, target, root_dir, run_name) -> None:
    """Calculate normalized confusion matrix for regression result.
    Calculating number of bins is done autmoatically.
    1. Number of bins: between 5 and 10
    2. Interval of each bin: 5 or 10.
    """
    AGEMIN, AGEMAX = 0, 100
    pmin_, pmax_ = preds.min(), preds.max()
    tmin_, tmax_ = target.min(), target.max()
    min_, max_ = max(AGEMIN, min(tmin_, pmin_)), min(AGEMAX, max(tmax_, pmax_))
    if math.isnan(min_):
        min_ = AGEMIN
    if math.isnan(max_):
        max_ = AGEMIN
    int_ = max_ - min_ # interval

    # Check two intervals 5 and 10
    bin_ = 5 if int_ // 5 <= 10 else 10

    # Bounds
    lb, ub = int(min_), int(max_)
    while (lb % bin_) != 0:
        lb -= 1
    while (ub % bin_) != 0:
        ub += 1
    bins = [lb + bin_ * idx for idx in range(0, (ub - lb) // bin_ + 1)]
    labels = [f"{left}-{right - 1}" for left, right in zip(bins, bins[1:])]

    cut_kwargs = dict(bins=bins, labels=labels, include_lowest=True)
    _preds = pd.cut(x=preds, **cut_kwargs)
    _target = pd.cut(x=target, **cut_kwargs)
    if (_preds.isna().sum() + _target.isna().sum()) == 0:
        fig, ax = plt.subplots(figsize=(13, 6), ncols=2)
        labelsize = "large"
        titlesize = "xx-large"
        cmap = "Blues"

        # Normalized Confusion Matrix
        cf = confusion_matrix(_preds, _target)
        sns.heatmap(cf, annot=True, fmt="d",
                    xticklabels=_target.categories, cmap=cmap, ax=ax[0])
        ax[0].set_xlabel("Target", size=labelsize)
        ax[0].set_yticklabels(_preds.categories, rotation=270)
        ax[0].set_ylabel("Prediction", size=labelsize)

        # 0-1 row-wise Noramlized Confusion.
        norm_cf = (cf.T / cf.sum(axis=1)).T
        norm_cf = np.nan_to_num(x=norm_cf, nan=0.0)
        sns.heatmap(norm_cf, annot=True, fmt="0.2f",
                    xticklabels=_target.categories, cmap=cmap, ax=ax[1])
        ax[1].set_xlabel("Target", size=labelsize)
        ax[1].set_yticklabels(_preds.categories, rotation=270)
        ax[1].set_ylabel("Prediction", size=labelsize)
        
        fig.suptitle(run_name, size=titlesize)
        fig.tight_layout()
        fig.savefig(root_dir / f"{run_name}-cf.png")


def _cls_inference(preds, target, root_dir, run_name) -> None:
    metrics_input = dict(preds=preds,
                         target=target.int(),
                         task="binary")
    acc = tmf.accuracy(**metrics_input)
    f1 = tmf.f1_score(**metrics_input)
    auroc = tmf.auroc(**metrics_input)
    logger.info("Results as follow:")
    logger.info("ACC  : %.3f", acc)
    logger.info("F1   : %.3f", f1)
    logger.info("AUROC: %.4f", auroc)
    
    cf = tmf.confusion_matrix(**metrics_input)
    p = sns.heatmap(cf, annot=True, fmt="d")
    p.set_title(run_name)
    plt.savefig(root_dir / f"{run_name}-cf.png")


def brain2augment(brain: torch.Tensor) -> torch.Tensor:
    """ Monai transforms is intended to take a single data as input.
    However, if this augmentation is implemented inside the dataloader,
    this will invoke high memory usage.
    I have splitted this outside the dataloader, and included inside the LightningModule.
    
    Because monai.transforms expects a single array with multi-channel image to be fed,
    we remove channel but instead put batch inside.
    However, this will instead apply the same augmentation to all batches.
    
    In short, this will convert a given tensor into (B, H, W, D)."""
    orig_ndim = brain.ndim
    if orig_ndim == 3:
        # (H, W, D) -> (1, H, W, D), making brain to batch=1
        brain = brain.unsqueeze(dim=0)
    elif orig_ndim == 4:
        # (B, H, W, D)
        pass
    elif orig_ndim == 5:
        # (B, C, H, W, D)
        # Kill channel
        C = brain.shape[1]
        assert C == 1, f"Brain should have single-channel: #channels = {C}"
        # Specify dimension in case batch_size is 1 (i.e. (1, 1, H, W, D))
        brain = brain.squeeze(dim=1)
        assert brain.ndim == 4, f"Output brain should be ndim 4"
    return brain


def augment2brain(brain: torch.Tensor) -> torch.Tensor:
    """Return to .ndim=5
    """
    orig_ndim = brain.ndim
    if orig_ndim == 3:
        # (H, W, D) -> (1, 1, H, W, D)
        brain = brain.unsqueeze(dim=0).unsqueeze(dim=0)
    elif orig_ndim == 4:
        # (B, H, W, D) -> (B, 1, H, W, D)
        brain = brain.unsqueeze(dim=1)
    elif orig_ndim == 5:
        # (B, C, H, W, D)
        # Kill channel
        C = brain.shape[1]
        assert C == 1, f"Brain should have single-channel: #channels = {C}"
    return brain
