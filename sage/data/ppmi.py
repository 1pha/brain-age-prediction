from pathlib import Path
from typing import Tuple, List

import torch

from sage.data.dataloader import DatasetBase, open_scan
import sage.constants as C
from sage.utils import get_logger


logger = get_logger(name=__name__)


class PPMIBase(DatasetBase):
    NAME = "PPMI"
    MAPPER2INT = dict(Control=0, PD=1, SWEDD=2, Prodromal=3)
    MOD_MAPPER = dict(t1="T1-anatomical", t2="T2 in T1-anatomical space")
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Description",
                 modality: List[str] = ["t1"],
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        modality = [self.MOD_MAPPER[m] for m in modality]
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        """ Make sure to properly return PPMI """
        raise NotImplementedError


class PPMIClassification(PPMIBase):
    NAME = "PPMI-CLS"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Description",
                 modality: List[str] = ["t1"],
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
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


class PPMIBinary(PPMIClassification):
    NAME = "PPMI-Binary"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_binary_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Description",
                 modality: List[str] = ["t1"],
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)


class PPMIAgeRegression(PPMIBase):
    NAME = "PPMI_AGE"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Image Data ID",
                 pid_col: str = "Subject",
                 label_col: str = "Age",
                 strat_col: str = None,
                 mod_col: str = "Description",
                 modality: List[str] = ["t1"],
                 exclusion_fname: str = "exclusion.csv",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        data: dict = self.labels.iloc[idx].to_dict()
        arr, _ = open_scan(data[self.path_col])
        arr = torch.from_numpy(arr).type(dtype=torch.float32)

        age: int = data[self.label_col]
        age = torch.tensor(age, dtype=torch.long)
        return arr, age
