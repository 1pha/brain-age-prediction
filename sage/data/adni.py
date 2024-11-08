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
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_screen_labels_Sept11_test15_2024.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "filepath",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "DX_bl",
                 strat_col: str = "DX_bl",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "",
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
    MAPPER2INT = {"CN": 0, "MCI": 1, "AD": 2}
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_screen_labels_Sept11_test15_2024.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "filepath",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "DX_bl",
                 strat_col: str = "DX_bl",
                 mod_col: str = "DX_bl",
                 modality: List[str] = ["CN", "MCI", "AD"],
                 exclusion_fname: str = "",
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


class ADNIBinary(ADNIClassification):
    NAME = "ADNI-Binary"
    MAPPER2INT = {"CN": 0, "AD": 1}
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_screen_labels_Sept11_test15_2024.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "filepath",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "DX_bl",
                 strat_col: str = "DX_bl",
                 mod_col: str = "DX_bl",
                 modality: List[str] = ["CN", "AD"],
                 exclusion_fname: str = "",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)


class ADNIFullClassification(ADNIClassification):
    NAME = "ADNI-ALL-CLS"
    MAPPER2INT = {"CN": 0, "SMC": 1, "EMCI": 2, "MCI": 3, "LMCI": 4, "AD": 5}
    def __init__(self,
                 root: Path | str = C.ADNI_DIR,
                 label_name: str = "adni_screen_labels_Sept11_test15_2024.csv",
                 mode: str = "train",
                 valid_ratio: float = .1,
                 path_col: str = "filepath",
                 pk_col: str = "Subject",
                 pid_col: str = "Subject",
                 label_col: str = "DX_bl",
                 strat_col: str = "DX_bl",
                 mod_col: str = None,
                 modality: List[str] = None,
                 exclusion_fname: str = "",
                 augmentation: str = "monai",
                 seed: int = 42,):
        super().__init__(root=root, label_name=label_name, mode=mode, valid_ratio=valid_ratio,
                         path_col=path_col, pk_col=pk_col, pid_col=pid_col, label_col=label_col,
                         strat_col=strat_col, mod_col=mod_col, modality=modality,
                         exclusion_fname=exclusion_fname, augmentation=augmentation, seed=seed)