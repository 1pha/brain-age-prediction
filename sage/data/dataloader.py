from pathlib import Path
from typing import Any, Dict, NewType, Tuple, List

Arguments = NewType("Arguments", Any)
Logger = NewType("Logger", Any)

import h5py
import hydra
import omegaconf
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sage.utils import get_logger
import sage.constants as sc


logger = get_logger(name=__name__)


def open_h5(fname: str) -> Tuple[np.ndarray, dict]:
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
        except:
            meta = None
    return arr, meta


def open_h5_nifti(fname: str) -> nib.nifti1.Nifti1Image:
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
            
            affine_last = np.array([0, 0, 0, 1], dtype=np.float32)
            affine = np.stack([meta["srow_x"], meta["srow_y"], meta["srow_z"], affine_last])
        except:
            meta = None
            affine = sc.BIOBANK_AFFINE
    nii = nib.nifti1.Nifti1Image(dataobj=arr, affine=affine)
    return nii

    
class UKBDataset(Dataset):
    def __init__(self,
                 root: Path | str = "./biobank",
                 label_name: str = None,
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 exclusion_fname: str = "exclusion.csv",
                 return_tensor: bool = True,
                 seed: int = 42,):
        """ Here we treat same pid with scans from different timeline as independent dataset.
        Since multiple scans have different ages """
        logger.info("Setting up UKBiobank Dataset")
        
        root = Path(root)
        if label_name is None:
            label_name = {"train": "ukb_trainval_age.csv",
                          "valid": "ukb_trainval_age.csv",
                          "test": "ukb_test_age.csv"}[mode]
        labels = pd.read_csv(root / label_name)
        self.labels = self.remove_duplicates(labels=labels)

        pids = set(self.labels.fname.unique())
        files = sorted(root.rglob("*.h5"))
        files = list(filter(lambda f: f.stem in pids, files))
        if mode != "test":
            files = self._split_data(files=files, valid_ratio=valid_ratio, mode=mode, seed=seed)
        self.files = self._exclude_data(lst=files, root=root, exclusion_fname=exclusion_fname)
        logger.info("Total %s files of %s h5 exist", len(self.files), mode)

        self.return_tensor = return_tensor

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
                      exclusion_fname: str = "exclusion.csv",):
        try:
            exc = pd.read_csv(root / exclusion_fname, header=None)
            exclusion = set(exc.values.flatten().tolist())
            lst = [f for f in lst if f not in exclusion]
        except:
            logger.info("No exclusion file found. %s", root / exclusion_fname)
            pass
        return lst

    def _split_data(self,
                    files: List[Path],
                    valid_ratio: float = 0.1,
                    mode: str = "train",
                    seed: int = 42) -> None:
        # Data split, used fixated seed
        trn, val = train_test_split(files, test_size=valid_ratio, random_state=seed)
        files = {"train": trn, "valid": val}.get(mode, None)
        if files is None:
            logger.exception(msg=f"Invalide mode given: {mode}")
            raise
        return files

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        arr, _ = open_h5(fname)
        age = self.labels.query(f"fname == '{fname.stem}'").age.iloc[0]
        if self.return_tensor:
            arr = torch.tensor(arr, dtype=torch.float32)
            age = torch.tensor(age, dtype=torch.long)
        return {
            "brain": arr,
            "age": age,
        }
        
    def __len__(self):
        return len(self.files)
    
    
class UKBClassification(UKBDataset):
    def __init__(self,
                 root: Path | str = "./biobank",
                 label_name: str = None,
                 young_threshold: int = 51,
                 old_threshold: int = 77,
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 exclusion_fname: str = "exclusion.csv",
                 return_tensor: bool = True,
                 seed: int = 42,
                 verbose: bool = False):
        self.thresholds = {"young": young_threshold, "old": old_threshold}
        self.verbose = verbose
        # TODO: Many variables are accessed by `self.`. Make sure they can be introduced
        if label_name is None:
            label_name = {"train": "ukb_trainval_age.csv",
                          "valid": "ukb_trainval_age.csv",
                          "test": "vbm.csv"}[mode]
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         exclusion_fname=exclusion_fname, return_tensor=return_tensor, seed=seed)

    def _age_filter(self, files: list):
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

    def _split_data(self,
                    files: list,
                    valid_ratio: float = 0.1,
                    mode: str = "train",
                    seed: int = 42) -> None:
        """ Override function.
        Filters out data with age. """
        files = self._age_filter(files=files)
        files = super()._split_data(files=files, valid_ratio=valid_ratio, mode=mode, seed=seed)
        return files

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx=idx)
        age = result["age"]
        if age <= self.thresholds["young"]:
            age = 0
        elif age >= self.thresholds["old"]:
            age = 1
        else:
            raise
        result["age"] = age
        return result


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
