from nilearn.datasets import load_mni152_brain_mask


MNI_SHAPE = load_mni152_brain_mask().get_fdata().shape