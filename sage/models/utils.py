import math

import torch

from .model_zoo import (
    build_cait,
    build_convit,
    build_convnext,
    build_resnet,
    build_repvgg,
    cait_list,
    convit_list,
    convnext_list,
)


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

    elif name == "repvgg":
        model = build_repvgg(name)

    elif name in convit_list:
        model = build_convit(name)

    elif name in convnext_list:
        model = build_convnext(name)

    elif name in cait_list:
        model = build_cait(name)

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
