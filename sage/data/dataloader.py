import os
from pathlib import Path
from typing import Any, Dict, NewType, Tuple, List

import h5py
import hydra
import omegaconf
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchio as tio
import monai.transforms as mt
from monai.data.meta_tensor import MetaTensor

from sage.utils import get_logger
try:
    import meta_brain.router as C
except ImportError:
    import sage.constants as C


logger = get_logger(name=__name__)


def open_scan(fname: str) -> Tuple[np.array, dict]:
    suffix = Path(fname).suffix
    meta = None
    if suffix == ".npy":
        arr = open_npy(fname)
    elif suffix == ".h5":
        arr, meta = open_h5(fname=fname)
    elif suffix in {".nii", ".nii.gz"} or str(fname).endswith(".nii.gz"):
        arr = nib.load(filename=fname)
        arr = arr.get_fdata()
    return arr, meta


def open_npy(fname: str) -> np.array:
    arr = np.load(file=fname)
    return arr


def open_h5(fname: str) -> Tuple[np.ndarray, dict]:
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
        except:
            meta = None
    return arr, meta


def open_h5_nifti(fname: str) -> nib.nifti1.Nifti1Image:
    """ Open `.h5` file and returns into Nifti file.
    Affine and other meta information will be stored in the nifti objects.
    If affine is not found, we will use constants.BIOBANK_AFFINE as default. """
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
            
            affine_last = np.array([0, 0, 0, 1], dtype=np.float32)
            affine = np.stack([meta["srow_x"], meta["srow_y"], meta["srow_z"], affine_last])
        except:
            meta = None
            affine = C.BIOBANK_AFFINE
    nii = nib.nifti1.Nifti1Image(dataobj=arr, affine=affine)
    return nii


class DatasetBase(Dataset):
    """ This Dataset class takes `.csv` labels with following scheme
    cols: pid (primary_key) | label | abspath
    
    - Removal of duplicates will base on `pid`
    - `__getitem__` will base on this csv
    - Sanity checking on abspath will be done.
    
    - Make sure to use split csv for test data. Test data will NOT be splitted
    - If dataset needs to be splitted based on their unique patient, provide `pid_col` 
    """
    NAME = ""
    def __init__(self,
                 root: Path | str,
                 label_name: str,
                 mode: str,
                 valid_ratio: float,
                 path_col: str,
                 pk_col: str,
                 pid_col: str,
                 label_col: str,
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        """ Here we treat same pid with scans from different timeline as independent dataset.
        Since multiple scans have different ages """
        logger.info("Setting up %s Dataset", self.NAME)
        self.path_col, self.pk_col, self.label_col = path_col, pk_col, label_col

        root = Path(root)
        labels: pd.DataFrame = self.load_labels(root=root, label_name=label_name, mode=mode)
        labels: pd.DataFrame = self.remove_duplicates(labels=labels)
        if mod_col and modality:
            labels: pd.DataFrame = self.filter_data(labels=labels, col=mod_col, leave=modality)
        self.mode = mode

        if mode != "test":
            labels: pd.DataFrame = self.split_data(labels=labels, valid_ratio=valid_ratio,
                                                    pid_col=pid_col, mode=mode, seed=seed)
        
        self.sanity_check(labels=labels, path_col=path_col)
        self.labels = labels
        self.init_transforms(augmentation=augmentation)
        logger.info("Total %s files of %s exist", len(self), mode)

    def load_labels(self, root: Path, label_name: str = None, mode: str = None) -> pd.DataFrame:
        """ Load `.csv` """
        labels = pd.read_csv(root / label_name)
        return labels

    def remove_duplicates(self, labels: pd.DataFrame) -> pd.DataFrame:
        return labels

    def _exclude_data(self,
                      lst: pd.DataFrame,
                      root: Path,
                      exclusion_fname: str = "exclusion.csv") -> List[Path]:
        return lst
    
    def filter_data(self, labels: pd.DataFrame, col: str, leave: List[str]) -> pd.DataFrame:
        cond = labels[col].isin(set(leave))
        labels = labels[cond]
        return labels

    def split_data(self,
                   labels: pd.DataFrame,
                   valid_ratio: float = 0.1,
                   pid_col: str = "",
                   mode: str = "train",
                   seed: int = 42) -> pd.DataFrame:
        # Data split, used fixated seed
        if pid_col:
            pid = labels[pid_col].unique().tolist()
            trn_pid, val_pid = train_test_split(pid, test_size=valid_ratio, random_state=seed)
            trn = labels[labels[pid_col].isin(trn_pid)]
            val = labels[labels[pid_col].isin(val_pid)]
        else:    
            trn, val = train_test_split(labels, test_size=valid_ratio, random_state=seed)
        labels = {"train": trn, "valid": val}.get(mode, None)
        if labels is None:
            logger.exception(msg=f"Invalide mode given: {mode}")
            raise
        return labels
    
    def sanity_check(self, labels: pd.DataFrame, path_col: str):
        paths = labels[path_col]
        existence = paths.apply(os.path.exists)
        exists = existence.sum()
        num_file = len(paths)
        assert exists == num_file, f"There are {num_file - exists} files that does not exist.\n{existence.tolist()}"

    def init_transforms(self, augmentation: str, spatial_size: tuple = (160, 192, 160)) -> None:
        # Currently unused and implemented inside the trainer
        # If one needs to use `torch.compile`, then transform should happen inside the torch.dataset
        if isinstance(augmentation, str):
            if augmentation == "monai":
                if self.mode == "train":
                    self.transforms = mt.Compose([
                        mt.Resize(spatial_size=spatial_size),
                        mt.ScaleIntensity(channel_wise=True),
                        mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
                        mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
                        mt.RandAxisFlip(prob=0.5),
                        mt.RandZoom(prob=0.4, min_zoom=0.9, max_zoom=1.4, mode="trilinear"),  
                    ])
                else:
                    self.transforms = mt.Compose([
                        mt.Resize(spatial_size=spatial_size),
                        mt.ScaleIntensity(channel_wise=True)
                    ])

            elif augmentation == "torchio":
                self.transforms = tio.Compose(transforms=[
                    tio.Resize(target_shape=spatial_size),
                    tio.RescaleIntensity(),
                    tio.RandomGamma(log_gamma=(0.5, 2.0)),
                    tio.RandomSwap(num_iterations=20, patch_size=8),
                    tio.RandomFlip(axes=[0, 1, 2]),
                    tio.RandomAffine(scales=(0.9, 1.4))
                ])
        else:
            self.transforms = mt.Identity()

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        data: dict = self.labels.iloc[idx].to_dict()
        arr, _ = open_scan(data[self.path_col])
        arr = torch.from_numpy(arr).type(dtype=torch.float32)

        age: int = data[self.label_col]
        age = torch.tensor(age, dtype=torch.long)
        return arr, age

    def get_tensor(self, tensor: torch.Tensor | MetaTensor) -> torch.Tensor:
        if isinstance(tensor, MetaTensor):
            # MetaTensor is not suitable for torch.compile
            tensor = tensor.as_tensor()
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr, age = self._load_data(idx=idx) # (H, W, D)
        arr = arr.unsqueeze(dim=0) # (C, H, W, D)
        return dict(brain=arr, age=age)

    def __len__(self) -> int:
        return len(self.labels)


class UKBDataset(DatasetBase):
    """ There are 45,800 .h5 scans inside the biobank directory.
    However, there is a disjoint set between labels files PID and actual .h5 files
    
    biobank .h5: 45,800
    label files: 40,940
        trainval: 37,911
        test_labels: 3,029

    trainval - biobank = 1,236
    test - biobank = 0 (all test_labels are included in actual h5)
    biobank - (trainval | test) = 6,096
    
    Until Feb 20, 2024, the logic of loading the scan due to this discrepancy was as follows
    1. Fetch all `.h5` scans in the biobank dir
    2. Load labels `ukb_trainval_age.csv`/`ukb_test_age.csv`
    3. Leave files that exist in the label file. Train/val split will be made w/ filelist from 2.
    
    From Feb 21, 2024, the label `.csv` file will include only existing files,
    no longer sanity checking.
    """
    NAME = "UKBiobank"
    def __init__(self,
                 root: Path | str = "./biobank",
                 label_name: str = None,
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 path_col: str = "abs_path",
                 pk_col: str = "fname",
                 pid_col: str = "",
                 label_col: str = "age",
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)

    def load_labels(self, root: Path, label_name: str = None, mode: str = None) -> pd.DataFrame:
        """ Load `.csv` """
        if label_name is None:
            assert mode is not None, f"Provide `mode` for UKB dataset"
            label_name = {"train": "ukb_trainval_age_exist240221.csv",
                          "valid": "ukb_trainval_age_exist240221.csv",
                          "test": "ukb_test_age_exist240221.csv"}[mode]
        labels = super().load_labels(root=root, label_name=label_name, mode=mode)
        return labels

    def remove_duplicates(self, labels: pd.DataFrame) -> pd.DataFrame:
        """ This method removes duplicate patients
        who have undergone multiple scans """
        _labels = labels.fname.apply(lambda s: s.split("_")[0])
        _dups_bool = _labels.duplicated(keep=False)
        labels = labels[~_dups_bool]
        return labels

    def _exclude_data(self,
                      lst: pd.DataFrame,
                      root: Path,
                      exclusion_fname: str = "exclusion.csv") -> List[Path]:
        try:
            exc = pd.read_csv(root / exclusion_fname, header=None)
            exclusion = set(exc.values.flatten().tolist())
            lst = [f for f in lst if f not in exclusion]
        except:
            logger.info("No exclusion file found. %s", root / exclusion_fname)
            pass
        return lst


class UKBClassification(UKBDataset):
    def __init__(self,
                 root: Path | str = "./biobank",
                 label_name: str = None,
                 young_threshold: int = 51,
                 old_threshold: int = 77,
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 exclusion_fname: str = "exclusion.csv",
                 seed: int = 42,
                 verbose: bool = False):
        self.thresholds = {"young": young_threshold, "old": old_threshold}
        self.verbose = verbose
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         exclusion_fname=exclusion_fname, seed=seed)

    def _age_filter(self, files: list) -> list:
        # Filter out self.files with age condition
        logger.info("Filter out ages")
        age = self.labels.age
        self.age_pids = self.labels[(age < self.thresholds["young"]) | (age > self.thresholds["old"])]
        age_pids = self.age_pids.fname.apply(lambda s: s.split("_")[0])
        all_pids = list(map(lambda s: s.stem.split("_")[0], files))
        filtered_files, passed = [], []
        for pid in age_pids:
            try:
                idx = all_pids.index(str(pid))
                filtered_files.append(files[idx])
            except ValueError:
                passed.append(pid)
                if self.verbose:
                    logger.info("\t\t %s was excluded.", pid)
        files = filtered_files
        logger.info("Total %s files will be used as train+valid scans", len(filtered_files))
        logger.info("Young: %s | Old: %s", (age < self.thresholds["young"]).sum(),
                                           (age > self.thresholds["old"]).sum())
        logger.info("#%s scans were excluded since they were not found as h5 files in biobank", len(passed))
        return files

    def split_data(self,
                   files: list,
                   valid_ratio: float = 0.1,
                   mode: str = "train",
                   seed: int = 42) -> pd.DataFrame:
        """ Override function.
        Filters out data with age. """
        files = self._age_filter(files=files)
        files = super().split_data(files=files, valid_ratio=valid_ratio, mode=mode, seed=seed)
        return files

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = super().__getitem__(idx=idx)
        age = data["age"]
        if age <= self.thresholds["young"]:
            age = 0
        elif age >= self.thresholds["old"]:
            age = 1
        else:
            raise
        data["age"] = age
        return data


def get_dataloaders(ds_cfg: omegaconf.DictConfig,
                    dl_cfg: omegaconf.DictConfig,
                    modes: list = ["train", "valid"]) -> Dict[str, Dataset]:
    dl_dict = {"train": None, "valid": None, "test": None}
    for mode in modes:
        _ds = hydra.utils.instantiate(ds_cfg, mode=mode)
        _dl = hydra.utils.instantiate(dl_cfg,
                                      dataset=_ds,
                                      shuffle=(True if mode == "train" else False))
        dl_dict[mode] = _dl
    return dl_dict
