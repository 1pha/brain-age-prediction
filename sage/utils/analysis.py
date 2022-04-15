import os
from glob import glob
from itertools import chain

import easydict
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from scipy.stats import ttest_ind

from sage.config import load_config


class FileSelector:
    def __init__(self, _type="naive"):

        self._type = _type
        self.ROOT = {
            "naive": "../resnet256_naive_checkpoints",
            "augment": "../resnet256_augmentation_checkpoints",
            "naive_nonreg": "../resnet256_naive_nonreg_checkpoints",
            "augment_nonreg": "../resnet256_augmentation_nonreg_checkpoints",
        }[_type]

        self.runs_dir = sorted(glob(self.ROOT + "/*"))  # contains all runs
        self.selector = "encoder"

        # Possible Selectors
        #   + encoder, domainer, regressor
        #   + npy_maps, npy_mm, npy_std
        #       - Possible layers: layer0 ~ layer8

    def __len__(self):
        return len(self.runs_dir)

    def __getitem__(self, idx: int):
        return sorted(glob(f"{self.runs_dir[idx]}/{self.selector}/*.pt"))

    def __next__(self):

        idx = 0
        idx += 1
        yield self.__getitem__(idx)

    def get_config(self, idx: int = 0):
        return load_config(f"{self.runs_dir[idx]}/config.yml")

    def set_selector(self, selector: str):
        self.selector = selector
        # logger.info(f"Selector set to {selector}. Now everything gets selected from {self.runs_dir[idx]}/{self.selector}/")

    def get(self, selector, idx: int = 0):

        # Get list of checkpoints (.pt) or

        path = f"{self.runs_dir[idx]}/{selector}/"
        files = list(
            chain(*[glob(path + extension) for extension in ["*.pt", "*.npy"]])
        )
        return sorted(files)

    def get_test_result(self, idx: int = 0):

        path = f"{self.runs_dir[idx]}/test.yml"
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def get_config(self, idx: int = 0):
        path = f"{self.runs_dir[idx]}/config.yml"
        with open(path, "r") as f:
            return easydict.EasyDict(yaml.load(f, Loader=yaml.Loader))

    @property
    def test_gt(self):
        path = f"data/test_gt_age.yml"
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.Loader)


def MAE(true, pred):
    return np.sum(np.abs(true - pred)) / len(true)


def check_existence(input, selector):

    idx, epoch = input

    # takes idx and epoch
    #   idx: indicates index of the run
    #   epoch: indicates epochs for selected run

    try:
        return os.path.exists(selector.get("npy_std/layer0", idx)[epoch])
    except:
        print(idx, epoch)
        # return None


def cherry_picker(input, selector):

    idx, epoch = input

    # takes idx and epoch
    #   idx: indicates index of the run
    #   epoch: indicates epochs for selected run

    try:
        return np.load(selector.get("npy_std/layer0", idx)[epoch])
    except:
        print(f"idx: {idx}, epoch:{epoch}")
        # return None


class Result:
    def __init__(self, data, _type="naive"):

        self._type = _type
        self.raw_data = data
        self.run_names = []
        for idx, (run_name, result) in enumerate(data.items()):
            setattr(self, f"result_{str(idx).zfill(3)}", result)
            self.run_names.append(run_name)
        self.epoch_organize()

    def __getitem__(self, idx):

        if isinstance(idx, int):
            return getattr(self, f"result_{str(idx).zfill(3)}")
        elif isinstance(idx, str):
            return self.raw_data[idx]
        else:
            raise f"Please give integer index or run_name (consisted of date). Given {idx}"

    def __getslice__(self, i, j):

        return [getattr(self, f"result_{str(idx).zfill(3)}") for idx in range(i, j)]

    def get_runname(self, idx):
        return self.run_names[idx]

    def __len__(self):
        return len(self.run_names)

    def epoch_organize(self):

        self.epoch_pivot = {}
        for idx, run_name in enumerate(self.run_names):

            # data of list[tuples, ...]
            data = self.raw_data[run_name]
            for d in data:
                e, mae = d
                self.epoch_pivot.setdefault(e, []).append(mae)

    @property
    def mean(self):
        return {e: np.mean(v) for e, v in self.epoch_pivot.items()}

    @property
    def std(self):
        return {e: np.std(v) for e, v in self.epoch_pivot.items()}

    def filterout_mae(self, func):
        if isinstance(func, str):

            if func == "first":
                return [e[0][1] for e in self.raw_data.values()]
            elif func == "last":
                return [e[-1][1] for e in self.raw_data.values()]
            elif func == "best":
                return [min(_[1] for _ in e) for e in self.raw_data.values()]


def transform(result):

    """
    turn [(e0, mae0), (e1, mae1), ... ] form into (list of epochs), (list of maes)
    """

    return [_[0] for _ in result], [_[1] for _ in result]


def group_stats(naive, augment, info=None):

    if info:
        print(info)
    print(f"Naive  : {np.mean(naive):.3f} ± {np.std(naive):.3f}")
    print(f"Augment: {np.mean(augment):.3f} ± {np.std(augment):.3f}")
    t_stat, p_val = stats.ttest_ind(naive, augment)
    print(f"Statistics: {t_stat:.2f} p-value: {p_val}\n")


def save2nifti(npy, savename, overwrite=False, size=(207, 256, 215)):

    """
    Takes saliency map with shape of (96, 96, 96)
    """
    if npy.ndim == 4:
        print(f"Came across with array dimension of {npy.ndim}, {npy.shape}.")
        print(f"Shrink with average on axis=0")
        npy = np.mean(npy, axis=0)

    # Resize
    resized_nifti = (
        F.interpolate(torch.tensor(npy[None, None, ...]), size=size)
        .squeeze()
        .squeeze()
        .numpy()
    )

    # Define Affine Matrix
    affine = np.array(
        [
            [0.73746312, 0.0, 0.0, -75.7625351],
            [0.0, 0.73746312, 0.0, -110.7625351],
            [0.0, 0.0, 0.73746312, -71.7625351],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    if os.path.exists(savename):
        if overwrite is False:
            print("File with same name exists, please allow overwrite to replace.")
            return

    nib.save(
        nib.Nifti1Image(
            resized_nifti,
            affine,
        ),
        savename,
    )
    print("Successfully saved.")
