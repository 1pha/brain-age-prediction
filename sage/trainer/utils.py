""" Includes model inference, output sorting utils
"""
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
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
    run_name = save_name[:-4]
    preds, target = prediction["pred"], prediction["target"]
    if name.startswith("C"):
        logger.info("Classification data given:")
        _cls_infrence(preds=preds, target=target, root_dir=root_dir, run_name=run_name)
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
    min_, max_ = target.min(), target.max()
    min_, max_ = max(AGEMIN, min_), min(AGEMAX, max_)
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
    labels = [f"{left+1}-{right}" for left, right in zip(bins, bins[1:])]
    
    _preds = pd.cut(x=preds, bins=bins, labels=labels)
    _target = pd.cut(x=target, bins=bins, labels=labels)
    breakpoint()


def _cls_infrence(preds, target, root_dir, run_name) -> None:
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
