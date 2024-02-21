from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from sage.data.dataloader import DatasetBase, open_scan
import sage.constants as C
from sage.utils import get_logger


logger = get_logger(name=__name__)


class PPMIClassification(DatasetBase):
    NAME = "PPMI_CLS"
    MAPPER2INT = dict(Control=0, PD=1, SWEDD=2, Prodromal=3)
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        
    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        data: dict = self.labels.iloc[idx].to_dict()
        arr, _ = open_scan(data[self.path_col])
        arr = torch.from_numpy(arr).type(dtype=torch.float32)

        label: str = data[self.label_col]
        label: int = self.MAPPER2INT.get(label, None)
        if label is None:
            logger.warn("Wrong label: %s\nData:%s", data[self.label_col], arr)
            raise
        label = torch.tensor(label, dtype=torch.long)
        return arr, label


class PPMIAgeRegression(DatasetBase):
    NAME = "PPMI_AGE"
    MAPPER2INT = dict(Control=0, PD=1, SWEDD=2, Prodromal=3)
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Age",
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        
    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        data: dict = self.labels.iloc[idx].to_dict()
        arr, _ = open_scan(data[self.path_col])
        arr = torch.from_numpy(arr).type(dtype=torch.float32)

        age: int = data[self.label_col]
        age = torch.tensor(age, dtype=torch.long)
        return arr, age
