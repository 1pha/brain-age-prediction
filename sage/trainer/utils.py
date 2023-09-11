""" Includes model inference, output sorting utils
"""
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import torch
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score

from sage.utils import get_logger


logger = get_logger(name=__file__)


def tune_logging_interval(logging_interval: int = 50,
                          batch_size: int = 64) -> int:
    """ Tune logging interval to log same number for varying batch sizes
    BASE: with batch size 64, logging_interval 50
    i.e. with a given batch size 4, freq step should increase to 16
    """
    BASE_BATCH = 64
    BASE_INTERVAL = 50
    if batch_size == BASE_BATCH and logging_interval == BASE_INTERVAL:
        return logging_interval
    else:
        ratio = BASE_BATCH / batch_size
        logging_interval = round(BASE_INTERVAL * ratio)
        return logging_interval
    
    
def tune_lr_interval(lr_frequency: int = 1,
                     batch_size: int = 64) -> int:
    """ Tune learning rate stepping frequency
    to step same number for varying batch sizes (logging)
    BASE: with batch size 64, steps 1
    i.e. with a given batch size 4, freq step should increase to 16
    """
    BASE_FREQ = 1
    BASE_BATCH = 50
    if lr_frequency == BASE_FREQ and batch_size == BASE_BATCH:
        return lr_frequency
    else:
        ratio = BASE_BATCH / batch_size
        lr_frequency = round(BASE_FREQ * ratio)
        return lr_frequency


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
    preds, target = prediction["reg_pred"], prediction["reg_target"]
    mse = mean_squared_error(preds=preds, target=target)
    mae = mean_absolute_error(preds=preds, target=target)
    r2 = r2_score(preds=preds, target=target)
    logger.info("Results as follow:")
    logger.info("MSE: %.3f", mse)
    logger.info("MAE: %.3f", mae)
    logger.info("MSE: %.4f", r2)

    # 3. Plot Jointplot
    data = pd.DataFrame({
        "Prediction": preds.numpy(),
        "Target": target.numpy(),
    })
    
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
