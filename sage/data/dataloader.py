import json
import os
from typing import Any, List, NewType

Arguments = NewType("Arguments", Any)
Logger = NewType("Logger", Any)

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset

# Augmentation Library
try:
    import torchio as tio
except:
    pass


class Identity:
    """
    Identity Scaler (returns X)
    """

    def __init__(self):
        pass

    def fit_transform(self, X: np.ndarray):
        return X


def get_scaler(scaler: str):
    return {"minmax": MinMaxScaler, "zscore": StandardScaler, "identity": Identity}[
        scaler
    ]()


def get_loader(extension: str):
    return {"npy": np.load, "nii": lambda x: nib.load(x).get_fdata()}[extension]


class BrainAgeDataset(Dataset):
    def __init__(
        self, data_args: Arguments, misc_args: Arguments, sampling: str, logger: Logger
    ):

        """
        CONFIG file should contain .csv file and -
        that .csv file should contain 'path' columns that contains full absolute path of the file

        sampling: str
            either 'train', 'valid', 'test'
            Test will always be the same with seed=42,
            while train/valid will depend on the configuration seed

        In this `data_path` folder, we need - (**VERY IMPORTANT**)
            - label.csv
            - data_config.yml: should contain -
                - data extension
                - preprocessing method
                - maximum volume (for mni)
            - data
        """

        # INITIAL SETUP
        self.logger = logger
        self.logger.info("Initialize dataset.")

        # Run methods
        self._load_dataset_config(data_args)
        self._load_label(data_args, misc_args.exclude_source)
        self._split_data(data_args, sampling, misc_args.seed)
        self._augmentation_setup(data_args)

    def _load_dataset_config(self, data_args: Arguments):

        """
        Loads dataset configuration file (json).
        This configuration depends on the dataset and lies in the same directory of brain scans.
        """

        # Check Dataset Configuration file Validity
        try:
            dataset_config_fname = os.path.join(
                data_args.data_path, data_args.config_file
            )
            self.logger.debug(f"Load dataset config. PATH: {dataset_config_fname}")
            with open(dataset_config_fname) as f:
                self.dataset_config = json.load(f)
        except:
            self.logger.warn(
                f"Data configuration cannot be found. Check {dataset_config_fname}."
            )
        self.load = get_loader(extension=self.dataset_config["extension"])

    def _load_label(self, data_args: Arguments, exclude_source: List[str]):

        label_fname = os.path.join(data_args.data_path, data_args.label_file)
        self.logger.debug(f"Load label file. PATH: {label_fname}")

        # Load Label File
        self.label_file = pd.read_csv(label_fname)

        # Exclude sources if needed.
        if exclude_source is not None:
            self.label_file = self.label_file[
                self.label_file["src"].apply(lambda x: x not in exclude_source)
            ]

        # Check if all data is possible.
        assert (
            sum(self.label_file["abs_path"].apply(os.path.exists))
            == self.label_file.shape[0]
        ), f"Possible Paths: {sum(self.label_file['abs_path'].apply(os.path.exists))} != #Rows: {self.label_file.shape[0]}"

        self.return_age_range = data_args.return_age_range

    def _split_data(self, data_args: Arguments, sampling: str, seed: int):

        self.logger.debug(f"Start spliting with seed {seed}")

        # Data split
        trn, tst = train_test_split(
            self.label_file, test_size=0.1, random_state=42  # FIXATED SEED
        )
        trn, val = train_test_split(
            trn, test_size=data_args.validation_ratio, random_state=seed
        )

        trn_idx, trn_age = trn["abs_path"], trn["age"]
        val_idx, val_age = val["abs_path"], val["age"]
        tst_idx, tst_age = tst["abs_path"], tst["age"]

        # AUGMENTATION THROUGH CONCATENATE ((AT MOST) DOUBLES # OF SAMPLES)
        if data_args.augmentation == "concat":
            self.augmentation = "concat"
            aug_idx, aug_age = trn_idx.apply(lambda x: x + "aug"), trn_age

        elif data_args.augmentation == "replace":
            self.augmentation = "replace"

        else:
            self.augmentation = False

        # Set data
        if sampling == "train":  # TRAIN SET

            self.data_files = (
                shuffle(pd.concat([trn_idx, aug_idx]), random_state=seed)
                if self.augmentation == "concat"
                else trn_idx
            )
            self.data_ages = (
                shuffle(pd.concat([trn_age, aug_age]), random_state=seed)
                if self.augmentation == "concat"
                else trn_age
            )

        elif sampling == "valid":  # VALIDATION SET
            self.augmentation = False
            self.data_files, self.data_ages = val_idx, val_age

        elif sampling == "test":  # TEST SET
            self.augmentation = False
            self.data_files, self.data_ages = tst_idx, tst_age

        self.data_files = self.data_files.to_list()
        self.data_ages = self.data_ages.to_list()

        self.logger.info(
            f"Successfully setup {self.__len__()} brains for {sampling.capitalize()}"
        )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx: int):  # -> ((1, W', H', D'), age: torch.tensor.float)

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
        aug = True if fpath.endswith("aug") else False  # for aug concat case.
        if aug:
            fpath = fpath[:-3]

        x = self.load(fpath)  # 3D (W, H, D)
        x = self.maxcut(x)  # 3D (W, H, D)
        x = self.preprocess(x)  # 4D (1, W', H', D')
        if aug or self.augmentation == "replace":
            x = self.transform(x)  # 4D (1, W', H', D')

        y = torch.tensor(self.data_ages[idx]).float()
        if self.return_age_range == "shrink":
            y /= 100

        return x, y

    def maxcut(self, x: torch.Tensor):
        """
        For brains that has many blanks.
        Should explicity give maxcut with tuples of tuples ((w, W), (h, H), (d, D))
        """

        if self.dataset_config["maxcut"] is not None:
            (w, W), (h, H), (d, D) = self.dataset_config["maxcut"]
            return x[w:W, h:H, d:D]
        else:
            return x

    def preprocess(self, x: torch.Tensor):  # -> (1, W', H', D')

        """
        Given with raw brain np.ndarray
            -> return desired output 4D shape (1, W', H', D')
        """

        # 1. Scale
        size = x.shape
        x = (
            get_scaler(self.dataset_config["scaler"])
            .fit_transform(x.reshape(-1, 1))
            .reshape(*size)
        )

        # 2. Resize
        resize = self.dataset_config["resize"]
        if not resize is None:
            # (1, 1, *resize) (5D) -> because F.interpolate requires 5D tensor for 3D tensor to be torted
            x = F.interpolate(torch.tensor(x)[None, None, ...], size=resize)
            x = x.squeeze(0).float()  # -> (1, *resize) (4D)

        else:
            x = torch.tensor(x)[None, ...].float()

        return x

    def _augmentation_setup(self, data_args: Arguments):

        self.logger.debug("Setting up augmentation.")

        # self._transform = tio.Compose(
        #     {
        #         tio.RandomFlip(axes=["left-right"]),
        #         tio.OneOf(
        #             {
        #                 tio.RandomAffine(),
        #                 tio.RandomElasticDeformation(),
        #             }
        #         ),
        #         tio.OneOf(
        #             {
        #                 tio.RandomGamma((-0.5, 0.5)),
        #                 tio.RandomBiasField(0.3),
        #                 tio.RandomBlur((0, 1)),
        #             }
        #         ),
        #     }
        # )
        self._transform = tio.OneOf(
            {
                tio.RandomAffine(),
                tio.RandomFlip(axes=["left-right"]),
                tio.RandomElasticDeformation(),
            }
        )

    def transform(self, x: torch.Tensor):  # -> torch.Tensor (1, W', H', D')

        """
        x must be given with torch.tensor with (1, W', H', D')
        """

        x = self._transform(x)
        return x


def get_dataloader(
    data_args: Arguments, misc_args: Arguments, sampling: str, logger: Logger
):

    dataset = BrainAgeDataset(data_args, misc_args, sampling, logger)
    dataloader = DataLoader(
        dataset, batch_size=data_args.batch_size, pin_memory=data_args.pin_memory
    )
    return dataloader


def get_dataloaders(data_args: Arguments, misc_args: Arguments, logger: Logger):

    _dataloaders = []
    for sampling in ["train", "valid", "test"]:
        _dataloaders.append(get_dataloader(data_args, misc_args, sampling, logger))
    return _dataloaders
