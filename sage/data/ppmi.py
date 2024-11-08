from pathlib import Path
from typing import Tuple, List

import torch
import pandas as pd

from sage.data.dataloader import DatasetBase, open_scan
import sage.constants as C
from sage.utils import get_logger


logger = get_logger(name=__name__)


class PPMIBase(DatasetBase):
    NAME = "PPMI"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        if visit_col and target_visit:
            self.labels: pd.DataFrame = self.filter_data(labels=self.labels, col=visit_col,
                                                         leave=target_visit if isinstance(target_visit, list) else [target_visit])

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        """ Make sure to properly return PPMI """
        raise NotImplementedError

    def _exclude_data(self, labels: pd.DataFrame, pk_col: str, root: Path,
                      exclusion_fname: str = "donotuse-adni.txt") -> pd.DataFrame:
        """ TODO: Remove exclude from label """
        fp = root / exclusion_fname
        if not fp.exists():
            logger.warn("Exclusion file `%s` was given but not found. Skip exclusion", fp)
            return labels
        else:
            with open(file=root / exclusion_fname, mode="r") as f:
                exclude = [s.strip() for s in f.readlines()]
            exc = set(exclude)
            labels = labels[~labels[pk_col].isin(exc)]
            return labels


class PPMIClassification(PPMIBase):
    NAME = "PPMI-CLS"
    MAPPER2INT = {"Control": 0, "Prodromal": 1, "PD": 2}
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "Prodromal", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
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
    MAPPER2INT = {"Control": 0, "PD": 1}
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)


class PPMIClassificationT1(PPMIClassification):
    NAME = "PPMI-CLS-T1"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "Prodromal", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        self.labels: pd.DataFrame = self.filter_data(labels=self.labels, col="T1/T2", leave=["T1"])


class PPMIBinaryT1(PPMIBinary):
    NAME = "PPMI-BIN-T1"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        self.labels: pd.DataFrame = self.filter_data(labels=self.labels, col="T1/T2", leave=["T1"])


class PPMIClassificationT2(PPMIClassification):
    NAME = "PPMI-CLS-T2"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "Prodromal", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        self.labels: pd.DataFrame = self.filter_data(labels=self.labels, col="T1/T2", leave=["T2"])


class PPMIBinaryT2(PPMIBinary):
    NAME = "PPMI-BIN-T2"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Group",
                 strat_col: str = "Group",
                 mod_col: str = "Group",
                 modality: List[str] = ["Control", "Prodromal", "PD"],
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)
        self.labels: pd.DataFrame = self.filter_data(labels=self.labels, col="T1/T2", leave=["T2"])


class PPMIAgeRegression(PPMIBase):
    NAME = "PPMI_AGE"
    def __init__(self,
                 root: Path | str = C.PPMI_DIR,
                 label_name: str = "ppmi_labels_240610.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "fname",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "Age",
                 strat_col: str = "Age",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "",
                 target_visit: str | List[str] = "BL",
                 visit_col: str = "Visit",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         target_visit=target_visit, visit_col=visit_col,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        data: dict = self.labels.iloc[idx].to_dict()
        arr, _ = open_scan(data[self.path_col])
        arr = torch.from_numpy(arr).type(dtype=torch.float32)

        age: int = data[self.label_col]
        age = torch.tensor(age, dtype=torch.long)
        return arr, age
