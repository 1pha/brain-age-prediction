from nilearn.datasets import load_mni152_brain_mask
import numpy as np


mni_template = load_mni152_brain_mask()
MNI_SHAPE = mni_template.get_fdata().shape
MNI_AFFINE = mni_template.affine

BIOBANK_AFFINE = np.array([[ -1.,  0.,  0.,   90.],
                           [  0.,  1.,  0., -126.],
                           [  0.,  0.,  1.,  -72.],
                           [  0.,  0.,  0.,    1.]], dtype=np.float32)