import math
import os

from .model_zoo import build_convit, build_convnext, build_resnet


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_args, logger):
    """
    TODO:
        It is just calling resnet directly without reading arguments.
        Need additional changes for future usage.
    """

    name = model_args.model_name
    if name == "resnet":
        model = build_resnet()

    elif name == "convit":
        model = build_convit(model_args)

    elif name == "convnext":
        model = build_convnext(model_args)

    params = count_params(model)
    logger.info(f"{model_args.model_name.capitalize()} has #params: {millify(params)}.")
    return build_resnet()


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
