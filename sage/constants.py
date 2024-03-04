from pathlib import Path

from nilearn.datasets import load_mni152_brain_mask
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(dotenv_path=".env")

mni_template = load_mni152_brain_mask()
MNI_SHAPE = mni_template.get_fdata().shape
MNI_AFFINE = mni_template.affine
H5_SHAPE = (182, 218, 182)

BIOBANK_AFFINE = np.array([[ -1.,  0.,  0.,   90.],
                           [  0.,  1.,  0., -126.],
                           [  0.,  0.,  1.,  -72.],
                           [  0.,  0.,  0.,    1.]], dtype=np.float32)

# Reshape target size
SPATIAL_SIZE = (160, 192, 160)

DATA_BASE = Path(config.get("DATA_BASE", Path.home() / "data" / "hdd01" / "1pha"))
BIOBANK_PATH = DATA_BASE / "h5"

EXT_BASE = DATA_BASE / "brain"
PPMI_DIR = EXT_BASE / "PPMI"