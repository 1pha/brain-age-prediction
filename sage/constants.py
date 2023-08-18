from nilearn.datasets import load_mni152_brain_mask
import numpy as np


MNI_SHAPE = load_mni152_brain_mask().get_fdata().shape

BIOBANK_AFFINE = np.array([[ -1.,  0.,  0.,   90.],
                           [  0.,  1.,  0., -126.],
                           [  0.,  0.,  1.,  -72.],
                           [  0.,  0.,  0.,    1.]], dtype=np.float32)