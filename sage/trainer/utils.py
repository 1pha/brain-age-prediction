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
         BASE_BATCH: int = 64):
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
        BASE_FREQ = 1
        assert accumulate_grad_batches is not None, f"Please provide `accumulate_grad_batches`"
        lr_frequency = round(BASE_FREQ * ratio / accumulate_grad_batches)
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
    elif name.startswith("R"):
        logger.info("Regression data given:")
        _reg_infrence(preds=preds, target=target, root_dir=root_dir, run_name=run_name)
    else:
        logger.info("Failed to inference. Check the run name for the task.")


def _reg_infrence(preds, target, root_dir, run_name) -> None:
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
