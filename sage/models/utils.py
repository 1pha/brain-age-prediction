import math
import os

import torch
from torchsummary import summary

from .model_zoo import build_resnet

# from .naive_models.dinsdale import *
# from .naive_models.efficientnet import EfficientNet3D
# from .naive_models.levakov_96 import *
# from .naive_models.res_sfcn import *
# from .naive_models.residual_vanilla import *
# from .naive_models.resnet import *
# from .naive_models.sequential import *
# from .naive_models.sfcn import *
# from .naive_models.vanilla import *
# from .unlearning.convit import ConvitArguments, VisionTransformer
# from .unlearning.prediction_layer import *
# from .unlearning.resnet_wo_linear import load_resnet
# from .unlearning.vanilla_dinsdale import VanillaConv


def load_model(cfg=None, gpu=True, verbose=True):

    model_name = cfg.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.force_cpu:
        device = torch.device("cpu")
    print(f"Model {model_name.capitalize()} is selected.")

    if model_name == "resnet" or model_name == "resnet_no_maxpool":

        opt = Option()
        model = generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.shortcut_type,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool,
            widen_factor=opt.resnet_widen_factor,
            start_channels=cfg.start_channels,
        )

        if model_name == "resnet_no_maxpool":
            model.no_max_pool = True

    elif model_name == "levakov":
        model = Levakov(task_type="age")

    elif model_name == "dinsdale":
        model = Dinsdale(1, 1, 2)

    elif model_name == "sfcn":
        model = SFCN(cfg)

    elif model_name == "vanilla":
        model = Vanilla3d(cfg)

    elif model_name == "vanilla_residual":
        model = Residual(cfg)

    elif model_name == "vanilla_residual_past":
        model = ResidualPast(cfg)

    elif model_name == "res_sfcn":
        model = ResSFCN(cfg)

    elif model_name == "efficientnet-b0":
        model = EfficientNet3D.from_name(
            "efficientnet-b0", override_params={"num_classes": 1}, in_channels=1
        )

    else:
        return None

    if gpu:
        model.to(device)

    if verbose:
        print(summary(model, input_size=(1, 96, 96, 96)))

    return model, device


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_unlearn_models(cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use {device} as a device.")

    encoder = ENCODERS[cfg.encoder.name](cfg.encoder).to(device)
    vector_size = encoder(torch.zeros((2, 1, 96, 96, 96)).to(device)).shape
    assert len(vector_size) == 2  # It should be 1-dim vector with batch (=2dim)
    print(f"Output from encoder is {vector_size[1]}.")
    cfg.regressor.init_node = vector_size[1]
    cfg.domainer.init_node = vector_size[1]
    regressor = PREDICTORS[cfg.regressor.name](cfg.regressor).to(device)
    domainer = PREDICTORS[cfg.domainer.name](cfg.domainer).to(device)

    cfg.encoder.num_params = count_num_params(encoder)
    cfg.regressor.num_params = count_num_params(regressor)
    cfg.domainer.num_params = count_num_params(domainer)
    cfg.num_params = sum(
        [cfg.encoder.num_params, cfg.regressor.num_params, cfg.domainer.num_params]
    )

    return (encoder, regressor, domainer), device


def load_models(*cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use {device} as a device.")

    models = []
    total_params = 0
    for i, config in enumerate(cfg):

        if config is None:
            models.append(None)
            continue

        if i == 0:  # ENCODER

            if config.name == "convit":
                arg_config = ConvitArguments(**config.config).config
                encoder = ENCODERS[config.name](**arg_config).to(device)

            else:
                arg_config = config.config
                encoder = ENCODERS[config.name](**arg_config).to(device)

            vector_size = encoder(torch.zeros((2, 1, 96, 96, 96)).to(device)).shape
            assert len(vector_size) == 2  # It should be 1-dim vector with batch (=2dim)
            print(f"Output from encoder is {vector_size[1]}.")
            num_params = count_num_params(encoder)
            config.num_params = num_params
            total_params += num_params
            models.append(encoder)

        elif i == 1:

            config.init_node = vector_size[1]
            regressor = PREDICTORS[config.name](config).to(device)
            num_params = count_num_params(regressor)
            config.num_params = num_params
            total_params += num_params
            models.append(regressor)

        elif i == 2:
            config.init_node = vector_size[1]
            domainer = PREDICTORS[config.name](config).to(device)
            num_params = count_num_params(domainer)
            config.num_params = num_params
            total_params += num_params
            models.append(domainer)

    print(f"Total Number of parameters: {total_params}")

    return models, device


def save_checkpoint(states, model_name, model_dir="./models/"):

    print("Saving ...")

    # MAKE DIRECTORY
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # MODEL NOT STATE_DICT
    if isinstance(states, nn.Module):
        states = states.state_dict()

    torch.save(states, os.path.join(model_dir, model_name))


def multimodel_save_checkpoint(states, model_name, model_dir="./models/"):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    for name, s in states.items():

        if isinstance(s, nn.Module):
            s = s.state_dict()

        _model_dir = os.path.join(model_dir, name)
        os.makedirs(_model_dir, exist_ok=True)
        fname = os.path.join(_model_dir, model_name)
        if not os.path.exists(fname):
            torch.save(s, fname)

    print("Saved Models")


def build_model(model_args, logger):
    """
    TODO:
        It is just calling resnet directly without reading arguments.
        Need additional changes for future usage.
    """

    model = build_resnet()
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


if __name__ == "__main__":

    model = "resnet"
    model, device = load_model(model)
