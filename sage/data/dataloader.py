# BASICS
import os
import logging
from itertools import permutations

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .preprocess import *
from ..config import load_config

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AUGMENTATION
try:
    import torchio as tio
except:
    pass

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(extension):

    return {"npy": np.load, "nii": lambda x: nib.load(x).get_fdata()}[extension]


class BrainAgeDataset(Dataset):
    def __init__(self, cfg, sampling="train"):

        """
        CONFIG file should contain .csv file and -
        that .csv file should contain 'path' columns that contains full absolute path of the file

        sampling: str
            either 'train', 'valid', 'test'
            Test will always be the same with seed=42,
            while train/valid will depend on the configuration seed

        ROOT is the path of database.
        In this folder, we need - (**VERY IMPORTANT**)
            - label.csv
            - data_config.yml: should contain -
                - data extension
                - preprocessing method
                - maximum volume (for mni)
            - data
        """

        # INITIAL SETUP
        logger.info("Initialize dataset.")

        self.cfg = cfg
        ROOT = cfg.root
        SEED = cfg.seed
        # self.data_cfg = load_config(os.path.join(ROOT, "data_config.yml"))  # -> Edict
        # Temporal Exception code
        if os.path.exists(os.path.join(ROOT, "data_config.yml")):
            self.data_cfg = load_config(os.path.join(ROOT, "data_config.yml"))  # -> Edict
        else:
            self.data_cfg = load_config("/workspace/brainmask_mni/data_config.yml")
        self.load = get_loader(extension=self.data_cfg.extension)

        self.augment = cfg.augment
        self.augment_replacement = cfg.augment_replacement
        if self.augment == True and self.augment_replacement == True:
            augment = input(
                "Please choose one of augmentation between - concatenate / replacement"
            )
            if augment in [None, " ", "replacement", "r"]:
                self.augment = False
                self.augment_replacement = True

            elif augment in ["concatenate", "concat", "old"]:
                self.augment = True
                self.augment_replacement = False
        self.sampling = sampling  # TRAIN / VALID / TEST

        # DO NOT AUGMENT WHEN VALID/TEST
        if sampling is not "train":
            self.augment = False
            self.augment_replacement = False

        # DEBUG SETUP
        self.debug = cfg.debug
        for d in cfg.data_debug:
            setattr(self, d, cfg.data_debug[d] if self.debug else False)
        if not cfg.debug:  # FLUSHOUT DEBUG ATTRS
            cfg.data_debug = []

        # VALIDATION SET SHOULD NOT DO AUGMENTATION
        if sampling in ["valid", "test"]:
            self.augment = False

        # LABEL FILE
        self.label_file = pd.read_csv(os.path.join(ROOT, "label.csv"))

        # EXCLUDE UNUSED SOURCE DATABASES
        self.label_file = self.label_file[
            self.label_file["src"].apply(lambda x: x not in cfg.unused_src)
        ]

        # IF PARTIAL (TO UES ONLY SOME DATA WHEN DEBUG)
        if cfg.partial < 1:
            self.label_file = self.label_file[: int(len(self.label_file) * cfg.partial)]

        if os.path.exists("G:/"):
            if not os.path.exists("G:/My Drive"):
                logger.info(
                    "Since G:/My Drive doesn't exist in thie environment, use G:/내 드라이브."
                )
                self.label_file["abs_path"] = self.label_file["abs_path"].apply(
                    lambda x: x.replace("G:\My Drive", "G:\내 드라이브")
                )

        elif os.path.exists("/workspace/"):
            self.label_file["abs_path"] = self.label_file["abs_path"].apply(
                lambda x: x.replace(
                    "G:\\My Drive\\brain_data\\", "/workspace/"
                )
            )
            self.label_file["abs_path"] = self.label_file["abs_path"].apply(
                lambda x: x.replace("\\", "/")
            )

        elif os.path.exists("/home/hoesung/hoesung_save2/daehyun/"):
            self.label_file["abs_path"] = self.label_file["abs_path"].apply(
                lambda x: x.replace(
                    "G:\\My Drive\\brain_data\\", "/home/hoesung/hoesung_save2/daehyun/"
                )
            )
            self.label_file["abs_path"] = self.label_file["abs_path"].apply(
                lambda x: x.replace("\\", "/")
            )
        assert (
            sum(self.label_file["abs_path"].apply(os.path.exists))
            == self.label_file.shape[0]
        )

        self.src_map = {
            src: i for i, src in enumerate(sorted(self.label_file.src.unique()))
        }

        # SPLIT DATA
        trn, tst = train_test_split(
            self.label_file, test_size=0.1, random_state=42  # FIXATED SEED
        )
        trn, val = train_test_split(trn, test_size=cfg.test_size, random_state=cfg.seed)

        trn_idx, trn_age, trn_src = trn["abs_path"], trn["age"], trn["src"]
        val_idx, val_age, val_src = val["abs_path"], val["age"], val["src"]
        tst_idx, tst_age, tst_src = tst["abs_path"], tst["age"], tst["src"]

        if (
            self.augment
        ):  # AUGMENTATION THROUGH CONCATENATE ((AT MOST) DOUBLES # OF SAMPLES)
            self.aug_proba = cfg.aug_proba
            self.aug_intensity = cfg.aug_intensity

            aug_idx = trn_idx.apply(lambda x: x + "aug")
            aug_age = trn_age
            aug_src = trn_src

        elif self.augment_replacement:  # AUGMENTATION WITH REPLACEMENT
            self.aug_intensity = cfg.aug_intensity

        else:  # NO AUGMENTATION
            self.aug_proba = []
            self.aug_intensity = []
            self.transformation = lambda x: x

        # SETUP DATA_FILES
        if sampling == "train":  # TRAIN SET

            # TODO: AUGRATIO
            self.data_files = (
                shuffle(pd.concat([trn_idx, aug_idx]), random_state=SEED)
                if self.augment
                else trn_idx
            )
            self.data_ages = (
                shuffle(pd.concat([trn_age, aug_age]), random_state=SEED)
                if self.augment
                else trn_age
            )
            self.data_src = (
                shuffle(pd.concat([trn_src, aug_src]), random_state=SEED)
                if self.augment
                else trn_src
            )

        elif sampling == "valid":  # VALIDATION SET
            self.data_files = val_idx
            self.data_ages = val_age
            self.data_src = val_src

        elif sampling == "test":  # TEST SET
            self.data_files = tst_idx
            self.data_ages = tst_age
            self.data_src = tst_src

        self.data_files = self.data_files.to_list()
        self.data_ages = self.data_ages.to_list()
        self.data_src = list(
            map(lambda s: self.src_map[s], self.data_src.to_list())
        )  # RETURN MAPPER

    def __len__(self):
        return len(self.data_files)

    def __getitem__(
        self, idx
    ):  # -> ((1, W', H', D'), age: torch.tensor.float, domain: torch.tensor.long)

        """
        PIPELINE
        1. LOAD BRAIN (x = self.load(self.data_files[idx])) -> np.ndarray: (1, W, H, D)
            - absolute path(:str) is given to load method.

        2. PREPROESS BRAIN (x = self.preprocess(x)) -> torch.tensor: (1, W', H', D')
            - preprocess with certain process
            2+. AUGMENTATION (x = self.transformation(x)) -> torch.tensor: (1, W', H', D')

        3. RETURN (BRAIN, AGE)
            - (torch.tensor(x, dtype=torch.float), torch.tensor(self.data_ages[idx]).float())
        """

        fpath = self.data_files[idx]
        aug = True if fpath.endswith("aug") else False
        if aug:
            fpath = fpath[:-3]

        x = self.load(fpath)  # 3D (W, H, D)
        x = self.maxcut(x)  # 3D (W, H, D)
        x = self.preprocess(x)  # 4D (1, W', H', D')
        if aug or self.augment_replacement:
            x = self.transformation(x)  # 4D (1, W', H', D')

        return (
            x,
            torch.tensor(self.data_ages[idx]).float(),
            torch.tensor(self.data_src[idx]).long(),
        )

    def maxcut(self, x):
        """
        For brains that has many blanks.
        Should explicity give maxcut with tuples of tuples ((w, W), (h, H), (d, D))
        """

        maxcut = self.data_cfg.maxcut if self.data_cfg.maxcut else None
        if maxcut is not None:
            (w, W), (h, H), (d, D) = maxcut
            return x[w:W, h:H, d:D]
        else:
            return x

    def preprocess(self, x):  # -> (1, W', H', D')

        """
        Given with raw brain np.ndarray
            -> return desired output 4D shape (1, W', H', D')

        MAY CONTAIN
            1. SCALING
            2. RESIZING
            3. ROTATION (IF NEEDED)
        """

        # 1. SCALING
        size = x.shape
        x = (
            get_scaler(self.data_cfg.scaler)
            .fit_transform(x.reshape(-1, 1))
            .reshape(*size)
        )

        # 2. RESIZING
        # PRIORITY: cfg > data_cfg <- cfg is set later than data_cfg
        resize = self.data_cfg.resize if self.cfg.resize is None else self.cfg.resize
        if not resize is None:
            # (1, 1, *resize) (5D) -> because F.interpolate requires 5D tensor for 3D tensor to be torted
            x = F.interpolate(torch.tensor(x)[None, None, ...], size=resize)
            x = x.squeeze(0).float()  # -> (1, *resize) (4D)

        else:
            x = torch.tensor(x)[None, ...].float()

        return x

    def setup_augmentation(self, cfg=None):

        """
        Not setup yet.
        """

        transforms = [
            tio.RandomAffine(),
            tio.RandomFlip(axes=["left-right"]),
            tio.RandomElasticDeformation(),
        ]

        tfm_combinations = []
        for idx in range(len(transforms)):

            for combination in permutations(transforms, idx + 1):
                tfm_combinations.append(tio.Compose(list(combination)))

    def transformation(self, x: torch.tensor):  # -> torch.Tensor (1, W', H', D')

        """
        x must be given with torch.tensor with (1, W', H', D')
        """

        transform = {
            "affine": tio.RandomAffine(),
            "flip": tio.RandomFlip(axes=["left-right"]),
            "elastic_deform": tio.RandomElasticDeformation(),
        }

        # TODO: Normalize probability but order of probabilities should be handled!
        # e.g. if probability order is [flip, ela, affine], then it will not give the expected output
        p = list(
            map(
                lambda x: x / sum(self.cfg.aug_proba.values()),
                self.cfg.aug_proba.values(),
            )
        )
        aug_choice = np.random.choice(list(transform.keys()), p=p)

        # if self.aug_verbose:
        #     print(f"Augmentation Choice: {aug_choice.capitalize()}")

        x = transform[aug_choice](x)

        return x

    def configuration(self):

        return self.cfg, self.data_cfg


def get_dataloader(cfg, sampling="train", return_dataset=False, pin_memory=True):
    """
    Just giving cfg.registration will find a proper path
    """

    cfg.root = {
        "tlrc": "H:/My Drive/brain/age_prediction/brainmask_tlrc",
        "mni": "/workspace/brainmask_mni",
        # "mni": "/home/hoesung/hoesung_save2/daehyun/brainmask_mni",
        # "mni": "H:/My Drive/brain/age_prediction/brainmask_mni",
        "raw": "H:/My Drive/brain/age_prediction/brainmask_nii",
    }[cfg.registration]

    dataset = BrainAgeDataset(cfg, sampling=sampling)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=pin_memory)
    return dataloader if not return_dataset else dataset
