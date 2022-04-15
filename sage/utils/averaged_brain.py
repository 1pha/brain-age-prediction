from glob import glob

import nibabel as nib
import numpy as np
from tqdm import tqdm


def averaged_brain(path, file_ext="npy", save=False, all=None):

    brain_fnames = sorted(glob(f"{path}/*.{file_ext}"))
    if len(brain_fnames) == 0:

        print("Doesn't seem to get the path correctly")
        return None

    if all is None:
        pass
    else:
        assert isinstance(all, int)
        brain_fnames = brain_fnames[:all]

    if file_ext == "npy":
        load = np.load
    elif file_ext == "nii.gz":
        load = lambda b: nib.load(b).get_fdata()

    avg_brain = load(brain_fnames[0])
    for b in tqdm(brain_fnames):
        avg_brain += load(b)

    avg_brain /= len(brain_fnames)
    if save:
        np.save(f"{path}/{save}.npy", avg_brain)

    return avg_brain


if __name__ == "__main__":

    path = "../../../brainmask_tlrc/"
    averaged_brain(path, all=875, save="averaged_brain_no_oas3")
