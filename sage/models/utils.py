import math
from typing import List

import torch
import torch.nn as nn

from .model_zoo import build_resnet


def find_conv_modules(model: nn.Module) -> List[nn.Module]:
    conv_modules = []
    def _find_conv_modules(module: nn.Module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_modules.append(module)
        
        for child_module in module.children():
            _find_conv_modules(child_module)

    _find_conv_modules(model)
    return conv_modules


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(training_args, logger):
    """
    TODO:
        It is just calling resnet directly without reading arguments.
        Need additional changes for future usage.
    """

    name = training_args.model_name
    logger.info(f"{name.capitalize()} was chosen.")
    if name == "resnet":
        model = build_resnet()

    params = count_params(model)
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)

    if torch.__version__.startswith("1.13"):
        if torch.backends.mps.is_available():
            device = "mps"
            model = model.to(device)

    logger.info(f"{name.capitalize()} has #params: {millify(params)}.")
    training_args.num_params = params
    return model


millnames = ["", " K", " M", " B", " T"]


def millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


if __name__ == "__main__":

    model = build_resnet().to("mps")
    # model.to("mps")
    device = next(model.parameters()).device
    print(device)
