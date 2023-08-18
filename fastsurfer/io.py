from pathlib import Path
from typing import Union, Tuple

import h5py
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import get_logger


logger = get_logger(name=__file__)


BIOBANK_AFFINE = np.array([[ -1.,  0.,  0.,   90.],
                           [  0.,  1.,  0., -126.],
                           [  0.,  0.,  1.,  -72.],
                           [  0.,  0.,  0.,    1.]], dtype=np.float32)


def open_h5(fname: str) -> Tuple[np.ndarray, dict]:
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
        except:
            meta = None
    return arr, meta


def open_h5_nifti(fname: str) -> Tuple[nib.nifti1.Nifti1Image, dict]:
    with h5py.File(name=fname, mode="r") as hf:
        arr = hf["volume"][:]
        try:
            meta = dict(hf.attrs)
            
            affine_last = np.array([0, 0, 0, 1], dtype=np.float32)
            affine = np.stack([meta["srow_x"], meta["srow_y"], meta["srow_z"], affine_last])
        except:
            meta = None
            affine = BIOBANK_AFFINE
    nii = nib.nifti1.Nifti1Image(dataobj=arr, affine=affine)
    return nii, meta


class UKBDataset(Dataset):
    def __init__(self,
                 root: Union[Path, str] = "./biobank",
                 label_name: str = "ukb_age_label.csv",
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 exclusion_fname: str = "exclusion.csv",
                 return_tensor: bool = True,
                 seed: int = 42,):
        logger.info("Setting up UKBiobank Dataset")
        self.return_tensor = return_tensor
        root = Path(root)
        self.files = list(root.rglob("*.h5"))
        self._split_data(valid_ratio=valid_ratio, seed=seed, mode=mode)
        
        self.files = self._exclude_data(lst=self.files, root=root, exclusion_fname=exclusion_fname)
        logger.info("Total %s files of %s h5 exist", len(self.files), mode)
        
        self.labels = pd.read_csv(root / label_name)
        
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
                    valid_ratio: float = 0.1,
                    mode: str = "train",
                    seed: int = 42) -> None:
        # Data split, used fixated seed
        trn, tst = train_test_split(self.files, test_size=0.1, random_state=42)
        trn, val = train_test_split(trn, test_size=valid_ratio, random_state=seed)
        self.files = {"train": trn,
                      "valid": val,
                      "test": tst}.get(mode, None)
        if self.files is None:
            logger.exception(msg=f"Invalide mode given: {mode}")
            raise
        
    def __getitem__(self, idx: int):
        """ Slightly different from brain-age-prediction UKBdataset """
        fname = self.files[idx]
        nii, meta = open_h5_nifti(fname=fname)
        age = self.labels.query(f"fname == '{fname.stem}'").age.iloc[0]
        return {"nii": nii,
                "age": age,
                "fname": fname,
                "meta": meta}
        
    def __len__(self):
        return len(self.files)
