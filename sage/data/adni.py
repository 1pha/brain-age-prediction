from pathlib import Path
from typing import Tuple, List

import torch
import pandas as pd
from overrides import overrides

from sage.data.dataloader import DatasetBase, open_scan
import sage.constants as C
from sage.utils import get_logger


logger = get_logger(name=__name__)


class ADNIBase(DatasetBase):
    NAME = "ADNI"
    MAPPER2INT = {"ADNI 2": 0, "ADNI 3": 1}
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Subject ID",
                 pid_col: str = "Subject ID",
                 label_col: str = "Phase",
                 strat_col: str = "Phase",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "donotuse-adni.txt",
                 augmentation: str = "monai",
                 seed: int = 42,):
        logger.warn("Please note that ADNI dataset label file should not have the exclusion file.")
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)

    def _load_data(self, idx: int) -> Tuple[torch.Tensor]:
        """ Make sure to properly return PPMI """
        raise NotImplementedError

    def _exclude_data(self, labels: pd.DataFrame, pk_col: str, root: Path,
                      exclusion_fname: str = "donotuse-adni.txt") -> pd.DataFrame:
        """ TODO: Remove exclude from label """
        fp = root / exclusion_fname
        if ~fp.exists():
            logger.warn("Exclusion file `%s` was given but not found. Skip exclusion", fp)
            return labels
        else:
            with open(file=root / exclusion_fname, mode="r") as f:
                exclude = [s.strip() for s in f.readlines()]
            exc = set(exclude)
            labels = labels[~labels[pk_col].isin(exc)]
            return labels


class ADNIClassification(ADNIBase):
    NAME = "ADNI-CLS"
    MAPPER2INT = {"ADNI 2": 0, "ADNI 3": 1}
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_label.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "abs_path",
                 pk_col: str = "Subject ID",
                 pid_col: str = "Subject ID",
                 label_col: str = "Phase",
                 strat_col: str = "Phase",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "donotuse-adni.txt",
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