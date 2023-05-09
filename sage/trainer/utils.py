from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np
import torch
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score

from sage.utils import get_logger


logger = get_logger(name=__file__)


def tune_logging_interval(logging_interval: int = 50,
                          batch_size: int = 64) -> int:
    """ Tune logging interval to log same number for varying batch sizes """
    BASE_BATCH = 64
    BASE_INTERVAL = 50
    if batch_size == BASE_BATCH and logging_interval == BASE_INTERVAL:
        return logging_interval
    else:
        ratio = batch_size / BASE_BATCH
        logging_interval = round(BASE_INTERVAL * ratio)
        return logging_interval
        

def load_mask(mask_path: str | Path = None,
              mask_threshold: float = 0.1):
    if not mask_path:
        return None
    else:
        if isinstance(mask_path, Path | str):
            mask = np.load(mask_path)
        elif isinstance(mask_path, np.ndarray):
            mask = mask_path
        else:
            raise
        # 4D-tensor: applied before channel unsqueezing
        mask = torch.nn.functional.interpolate(input=torch.tensor(mask)[None, None, ...],
                                                size=(96, 96, 96), mode="trilinear").squeeze(dim=0)
        mask = mask < (mask_threshold or 0.1)
        return mask
    

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
                       name: str) -> None:
    """ Takes non-sorted prediction (=list of dicts)
        1. Flatten predictions into dict
        2. Saves & Logs prediction
        3. Plot prediction with jointplot / kde plot and save plots
    """
    
    # 1. Sort Prediction
    prediction = _sort_outputs(prediction)
    
    # 2. Save PRediction
    save_name = f"{name}.pkl"
    logger.info("Save prediction as %s", save_name)
    with open(save_name, "wb") as f:
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
    p.savefig(f"{run_name}-joint.png")
    
    # 3. Plot KDE plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(data=data, ax=ax)
    fig.title(run_name)
    fig.tight_layout()
    fig.savefig(f"{run_name}-kde.png")
