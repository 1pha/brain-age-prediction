import os
import time
from typing import Any, NewType

import torch

Logger = NewType("Logger", Any)


def save_checkpoint(
    model: torch.nn.Module, model_name: str, output_dir: str, logger: Logger
):

    try:
        os.makedirs(os.path.join(output_dir, "ckpts"), exist_ok=True)
        fname = os.path.join(output_dir, "ckpts", model_name)
        torch.save(model.state_dict(), fname)
        logger.info(f"Successfully saved model as {fname}.")
    except:
        logger.warn(f"Failed to save model as {fname}.")


def walltime(original_fn):
    def wrapper_fn(self, *args, **kwargs):
        start_time = time.time()
        result = original_fn(self, *args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    return wrapper_fn
