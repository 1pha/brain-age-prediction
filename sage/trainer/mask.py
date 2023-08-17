""" Mask-related utilties, including followings
"""
import os
from pathlib import Path

import numpy as np
import torch

from sage.utils import get_logger

logger = get_logger(name=__file__)
MASK_BASE = Path("./assets/masks/")


def load_mask(mask_path: str | Path = None,
              mask_threshold: float = 0.1):
    mask_path = mask_aliasing(mask_path=mask_path)
    if mask_path is None:
        # Mask not given
        return None
    else:
        if isinstance(mask_path, Path | str):
            mask = np.load(mask_path)
        elif isinstance(mask_path, np.ndarray):
            mask = mask_path
        else:
            raise
        # 4D-tensor: applied before channel unsqueezing
        mask = torch.tensor(mask)[None, None, ...].float()
        mask = torch.nn.functional.interpolate(input=mask,
                                                size=(96, 96, 96), mode="trilinear").squeeze(dim=0)
        mask = mask < (mask_threshold or 0.1)
        return mask
    
    
def mask_aliasing(mask_path: str | Path = None) -> None | np.ndarray:
    """ Naming can be
    1. Naive path that is actual path to the loading
    2. Alias to mask: contains with certain format :: {model_name}-{xai_method}-{sigma}
        e.g.
        - model_name: resnet10t
        - xai_method: ig0.99-0.995 (Format of {method}{agg_threshold}_{top_threshold})
        - sigma: sigma0.5 (values used to do Gaussian blur sigma{threshold:.1f})
    """
    if (not mask_path) or mask_path in ["False", "None"]:
        return None

    if isinstance(mask_path, Path) or os.path.exists(mask_path):
        mask = np.load(mask_path)
    elif isinstance(mask_path, str):
        model_name, xai_method, sigma = mask_path.split("-")
        if model_name == "resnet":
            model_name = "resnet/resnet10t-aug-nomask"
            if not xai_method:
                xai_method = "ig0.99_0.995"
        
        mask_path = MASK_BASE / f"{model_name}-{xai_method}-{sigma}.npy"
        mask = np.load(mask_path)
    elif isinstance(mask_path, np.ndarray):
        mask = mask_path
    else:
        logger.info("Not a valid format")
        breakpoint()
    logger.info("Successfully loaded mask from %s", mask_path)
    breakpoint()
    return mask
