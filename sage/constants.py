from pathlib import Path

from nilearn.datasets import load_mni152_brain_mask
import numpy as np


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
BIOBANK_PATH = Path.home() / "brain-age-prediction" / "biobank"