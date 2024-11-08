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

DATA_BASE1 = Path(config.get("DATA_BASE", Path.home() / "data" / "hdd01" / "1pha"))
BIOBANK_PATH = DATA_BASE1 / "h5"
ADNI_DIR = DATA_BASE1 / "brain" / "ADNI_08_12_2024" / "3_reg"

DATA_BASE3 = Path.home() / "data" / "hdd03" / "1pha"
PPMI_DIR = DATA_BASE3 / "ppmi" / "PPMI_4_reg"
# PPMI_DIR = Path("~/workspace/brain-age-prediction/ppmi/PPMI_4_reg")
ADNI_DIR = Path("adni")