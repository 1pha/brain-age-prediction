""" Mask-related utilties, including followings
"""
import os
from pathlib import Path

import numpy as np
import torch


MASK_BASE = Path("./asets/masks/")


def load_mask(mask_path: str | Path = None,
              mask_threshold: float = 0.1):
    if (not mask_path) or mask_path in ["False", "None"]:
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
    
    
def mask_aliasing(mask_path: str | Path = None) -> Path:
    """ Naming can be
    1. Naive path that is actual path to the loading
    2. Alias to mask: contains with certain format :: {model_name}-{xai_method}-{sigma}=0.5
        e.g.
        - model_name: resnet10t
        - xai_method: ig0.99 (includes method and threshold value)
        - sigma: 0.5 (values used to do Gaussian blur)
    """
    
    if isinstance(mask_path, Path) or os.path.exists(mask_path):
        return mask_path
    elif isinstance(mask_path, str):
        model_name, xai_method, sigma = mask_path.split("_")
        if model_name == "resnet":
            model_name = "resnet10t-aug-nomask"
            