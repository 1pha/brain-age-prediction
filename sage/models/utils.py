import math
import torch

from .model_zoo import build_convit, build_convnext, build_resnet, convit_list


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(training_args, logger):
    """
    TODO:
        It is just calling resnet directly without reading arguments.
        Need additional changes for future usage.
    """

    name = training_args.model_name
    if name == "resnet":
        model = build_resnet()

    elif name in convit_list:
        model = build_convit(training_args)

    elif name == "convnext":
        model = build_convnext(training_args)

    params = count_params(model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    logger.info(f"{training_args.model_name.capitalize()} has #params: {millify(params)}.")
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
