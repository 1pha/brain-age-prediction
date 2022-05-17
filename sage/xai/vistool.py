import os
from glob import glob

import numpy as np
import torch
from IPython.display import clear_output

from ..config import load_config
from ..data.dataloader import get_dataloader
from ..training.trainer import MRITrainer
from .auggrad import AugGrad
from .cams import CAM
from .smoothgrad import SmoothGrad
from .utils import Assembled, plot_vismap


def deprecate(func):
    print(f"This {(func.__name__)} function is no longer supported since version=0.2")

    def class_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return class_wrapper


def get_weight_dict(prefix):

    return {
        model_name: sorted(
            glob(f"{prefix}/{model_name}/*.pt"),
            key=lambda x: int(x.split("\\ep")[-1].split("_")[0]),
        )
        for model_name in ["encoder", "regressor"]
    }


class VisTool:

    __version__ = "0.3"
    __date__ = "Sep 11. 2021"
    CAMS = {
        "gcam": CAM,
        "sgrad": SmoothGrad,
        "agrad": AugGrad,
    }

    def __init__(self, cfg=None, model=None, cam_type="agrad", PREFIX=None, **kwargs):

        """
        cfg:
            Configuration dict. necessary
        model:
            PyTorch Model. Doesn't need to be trained (Better to give pretrained).
            You can pass a pretrained weights through load_weight method
        cam_type:
            type in which cam to use. 3 options for Apr 7. - gcam, sgrad, agrad.
            which stands for GradCAM, SmoothGrad, AugGrad respectively
        """

        if cfg is None and model is None and PREFIX is not None:
            self.setup(PREFIX)

        else:
            self.cfg = cfg
            self.model = model

        self.cfg.batch_size = 1
        self.cam_type = cam_type
        self.vis_tool = VisTool.CAMS[self.cam_type](self.cfg, self.model, **kwargs)

    def setup(self, PREFIX):

        self.cfg = load_config(f"{PREFIX}/config.yml")
        self.cfg.batch_size = 1

        trainer = MRITrainer(self.cfg)
        encoder = trainer.models["encoder"]
        regressor = trainer.models["regressor"]
        self.model = Assembled(encoder, regressor)
        del trainer

        self.train_dataloader = get_dataloader(self.cfg, sampling="train")
        self.valid_dataloader = get_dataloader(self.cfg, sampling="valid")
        self.test_dataloader = get_dataloader(self.cfg, sampling="test")

    def load_weight(self, pth):

        """
        Load pretraind weights to model.
        Either use
            dict:
                that contains {model_name: weight_path}
            str:
                directly .pth
        """

        try:
            print(f"Load '{pth}'")
            if isinstance(pth, dict):
                self.model.load_weight(pth)

            elif isinstance(pth, str):
                self.model.load_state_dict(torch.load(pth))

            print("Weights successfully loaded!")

        except:
            print("An error occurred during loading weights.")
            raise

    def __call__(
        self,
        x=None,
        y=None,
        dataloader=None,
        visualize=False,
        slice_index=48,
        weight=None,
        prefix=None,  # DEPRECATED
        layer_index=None,
        save=False,
    ):
        """
        This method interacts with `trainer`.
        Arguments:
            x: torch.tensor
                A single brain image with 5D (batch=1, channel=1, Height, Width, Depth)
                If not 5D, function will automatically convert it to desired shape.
            y: torch.tensor
                A single age (or any target) in float contained in tensor

            dataloader: torch.utils.data.DataLoader | bool
                If "True" given, use attribute dataloader
                Dataloader from torch that yields (x, y) pair.
                If dataloader returns x, y, d, this will be processed internally.
            average: bool, default=True # FURTHER IMPLEMENTED
                If dataloader is given, return an averaged visual map of all the brains
                contained inside the dataloader.

            *Note: One of (x, y) or dataloader should be given.

            visualize: bool
                Whehter to show saliency map on the prompt or not.
                If True, `plot_vistool` will be executed.
            title: str, default=None, optional # FURTHER IMPLEMENTED
                If given, this will be used as title during visualization
            slice_index: int|list, default=48
                Select which slice to visualize. Either integer or list of integers be given.

            weight: dict, default=None, optional
                Dictionary that constructed with {model_name: weight_path}.
                Model of attribute would be set with a given weight.
                Note that this should be a single checkpoint of a model.
                To use multiple model, please use `path` instead.
            prefix: dict, default=None <- DEPRECATED
                directory of path that contains a total checkpoint of single run.
                Directory must follow the next architecture
                    prefix/
                        -encoder
                        -regressor
                        -domainer (optional)

            layer_index: int|list default=None
                Select a layer/layers to retrieve a saliency map.
                If None is given, find all layers' visual map

            save: bool default=False
                Whether to save a saliency map.
                If True, all maps will be saved depending on the arguments.
                + prefix (multiple model) given
                    All layers/timestamps of 3D voxel with size (W, H, D) will be saved as the following
                    prefix/
                        vismap_{date}/
                            layers/
                                layer1/
                                    weight1.npy
                                    weight2.npy
                                    ...
                                layer2/
                                ...
                            gifs/
                                layer1.gif
                                layer2.gif
                                ...
                + weight given <- DEPRECATED
                    prefix/
                        ep{epoch}_layer1.npy
                        ep{epoch}_layer2.npy
                        ...


        """
        if dataloader == "train":
            dataloader = self.train_dataloader

        elif dataloader == "valid":
            dataloader = self.valid_dataloader

        elif dataloader == "test":
            dataloader = self.test_dataloader

        else:
            assert isinstance(
                dataloader, torch.utils.data.dataloader.DataLoader
            ), "Please give external dataloader, or 'train'/'valid' as a string"

        def run():

            if x is not None and y is not None:  # (x, y) pair given
                if dataloader is not None:
                    print(
                        "Don't need to pass dataloader if x and y is given. "
                        "This dataloader will be ignored."
                    )

                vismap = self.run_vistool(x, y, layer_index=layer_index)
                brain = x

            elif dataloader is not None:  # dataloader given.
                if x is not None and y is not None:
                    print(
                        "(x, y) pair overpowers in priority against dataloader. "
                        "VisMap with a single (x, y) pair will be returned"
                    )

                vismap = [
                    np.zeros(self.cfg.resize)
                    for l in range(len(self.vis_tool.conv_layers))
                ]
                brain = torch.zeros((1, 1, *self.cfg.resize))
                for _x, _y, _ in dataloader:

                    _vismap = self.run_vistool(_x, _y, layer_index=layer_index)
                    brain += _x
                    for i, v in enumerate(_vismap):
                        vismap[i] += v

            if visualize:
                for idx, layer in enumerate(vismap):
                    # plot_vismap(brain, layer, slc=slice_index, title=f"{idx}th layer.")
                    plot_vismap(
                        "template", layer, slc=slice_index, title=f"{idx}th layer."
                    )
                    clear_output()

            return vismap

        if prefix is not None:
            weights = get_weight_dict(prefix)
            vismaps = list()
            for encoder_weight, regressor_weight in zip(
                weights["encoder"], weights["regressor"]
            ):

                self.load_weight(
                    {
                        "encoder": encoder_weight,
                        "regressor": regressor_weight,
                    }
                )
                layer_vismap = run()
                vismaps.append(layer_vismap)

                if save:
                    os.makedirs(f"{prefix}/layers", exist_ok=True)
                    weight_name = encoder_weight.split("\\")[-1].split(".pt")[0]
                    for layer_idx, layer in enumerate(layer_vismap):
                        save_path = f"{prefix}/layers/layer{layer_idx}"
                        os.makedirs(save_path, exist_ok=True)
                        np.save(
                            f"{save_path}/weight{weight_name}_layer{layer_idx}.npy",
                            layer,
                        )

        if weight is not None:
            self.load_weight(weight)
            return run()

        elif prefix is not None:

            weights = get_weight_dict(prefix)
            vismaps = list()
            for encoder_weight, regressor_weight in zip(
                weights["encoder"], weights["regressor"]
            ):

                self.load_weight(
                    {
                        "encoder": encoder_weight,
                        "regressor": regressor_weight,
                    }
                )
                vismaps.append(run())

            return vismaps  # [VISMAP_EP1(=[LAYER1, LAYER2, ...]), VISMAP_EP2, ...]

        else:
            print("None of weight neither prefix is given.")
            return

    def run_vistool(self, x, y, layer_index=None, **kwargs):

        self.model.to(self.cfg.device)
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        vismap = self.vis_tool(
            x, y, layer_index=layer_index, **kwargs
        )  # Should return (1, 96, 96, 96) visualization map

        return vismap
