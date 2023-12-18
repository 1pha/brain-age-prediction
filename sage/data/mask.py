from pathlib import Path
from typing import List, Tuple, Dict

import nilearn.image as nili
import numpy as np
import torch

from sage.data.dataloader import UKBDataset
from sage.xai.atlas import get_atlas
from sage.utils import get_logger
import sage.constants as C


logger = get_logger(name=__file__)


class UKB_MaskDataset(UKBDataset):
    def __init__(self,
                 atlas_name: str,
                 mask_idx: int | List[int],
                 reshape: Tuple = C.H5_SHAPE,
                 root: Path | str = "./biobank",
                 label_name: str = None,
                 mode: str = "train",
                 valid_ratio: float = 0.1,
                 exclusion_fname: str = "exclusion.csv",
                 return_tensor: bool = True,
                 seed: int = 42):
        super().__init__(root=root,
                         label_name=label_name,
                         mode=mode,
                         valid_ratio=valid_ratio,
                         exclusion_fname=exclusion_fname,
                         return_tensor=return_tensor,
                         seed=seed)
        self.initiate_atlas(atlas_name=atlas_name, mask_idx=mask_idx, reshape=reshape)

    def initiate_atlas(self, atlas_name: str, mask_idx: int | List[int], reshape: Tuple) -> None:
        atlas = get_atlas(atlas_name=atlas_name)
        logger.info("RoI(s) to be masked out: %s", mask_idx)
        if isinstance(mask_idx, int):
            mask_idx = [mask_idx]
        assert set(mask_idx).issubset(set(atlas.indices)),\
               f"{mask_idx} is not in the ATLAS RoI Indices!"
        self.rois = [atlas.get_roi_name(idx) for idx in mask_idx]
        self.mask_idx = mask_idx

        logger.info("RoI(s) to be masked out: %s", self.rois)

        # Reshape
        # In most cases, brain will be shaped as (182, 218, 182)
        if reshape and not (reshape == atlas.array.shape):
            logger.info("Start reshaping the atlas into %s", reshape)
            # If reshape is given and atlas is not aligned to the given shape,
            # reshape the atlas
            new_img = nili.resample_img(img=atlas.nii,
                                        target_affine=C.MNI_AFFINE,
                                        target_shape=reshape,
                                        interpolation="nearest")
            atlas.nii = new_img
            atlas.array = new_img.get_fdata()

        # Get mask idx
        self.atlas = atlas
        self.mask = np.isin(element=self.atlas.array, test_elements=self.mask_idx, assume_unique=True)
        assert self.mask.sum() > 0, f"Masking is not done correctly. Check the mask index: {self.mask_idx}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = super().__getitem__(idx=idx)
        data["brain"][self.mask] = 0.
        return data
