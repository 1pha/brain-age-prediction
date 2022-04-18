import os
import time
from typing import Any, NewType

import torch

Logger = NewType("Logger", Any)


def save_checkpoint(
    model: torch.nn.Module, model_name: str, output_dir: str, logger: Logger
):

    try:
        os.makedirs(os.path.join(output_dir, "/ckpts"), exist_ok=True)
        fname = os.path.join(output_dir, "/ckpts/", model_name)
        torch.save(model.state_dict(), fname)
        logger.info(f"Successfully saved model as {fname}.")
    except:
        logger.warn(f"Failed to save model as {fname}.")


def walltime(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()

        result = original_fn(*args, **kwargs)

        end_time = time.time()
        end = "" if original_fn.__name__ == "train" else "\n"
        print(f"[{original_fn.__name__}] {end_time - start_time:.1f} sec ", end=end)
        return result

    return wrapper_fn
