import os
import matplotlib.pyplot as plt

import numpy as np

import torch
import nibabel as nib

from nilearn.datasets import load_mni152_template
from skimage.transform import resize


from .cams import *
from .smoothgrad import *
from .auggrad import *


def check_type(brain, resize_shape=(96, 96, 96), maxcut=None):

    """
    Converts a given 3D (or even more than 4D) array (either numpy.ndarray or torch.tensor)
    to a designated 3D shape

    This does
        - reduce dimension until 3D
        - if torch.tensor given, detach and fetch to CPU and turn to numpy array
        - rotate to have the brain in a designated direction
        - returns numpy array
    """

    if isinstance(brain, str):

        if brain == "template":

            brain = load_mni152_template().get_fdata()
            maxcut = (
                (
                    [8, 82],
                    [5, 104],
                    [0, 78],
                )
                if maxcut is None
                else maxcut
            )
            (w, W), (h, H), (d, D) = maxcut
            brain = brain[w:W, h:H, d:D]
            brain = resize(brain, resize_shape)
            brain = np.transpose(brain, (1, 2, 0))

    else:
        while brain.ndim > 3:
            brain = brain[0]

        if isinstance(brain, np.ndarray):
            brain = np.transpose(brain, (1, 2, 0))

        elif torch.is_tensor(brain):
            brain = brain.permute(1, 2, 0).data.cpu().numpy()

    return np.rot90(brain)


def plot_vismap(
    brain,
    vismap,
    masked=True,
    percentile_threshold=0.975,
    value_threshold=None,
    two_sided=True,
    slc=48,
    alpha=0.6,
    save=False,
    att_path=None,
    idx=None,
    title=None,
):

    """
    brain, vismap:
        Array that contains base template brain and saliency map

    masked:
        For non-normalized brains, this ables to cut out a value below the threshold.
        It will move out blues when overlaid
    threshold:
        values to be thrown out when masked is turned on
    slc:
        slice to plot up
    alpha:
        opacity for overlaid vismap
    save:
        save plots in './result/att_tmp_plots/'
    idx:
        when using visualizations during the training, able to show up which epoch
    """

    if masked:
        if percentile_threshold is not None:
            if two_sided:
                threshold = np.quantile(abs(vismap).reshape(-1), percentile_threshold)
            else:
                threshold = np.quantile(vismap.reshape(-1), percentile_threshold)

        elif value_threshold is not None:
            threshold = value_threshold

        vismap = np.ma.masked_where(vismap < threshold, vismap)

    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

    brain = check_type(brain)
    vismap = check_type(vismap)

    if title is not None:
        fig.suptitle(title)

    elif title is None and idx is not None:
        fig.suptitle(f"Epoch {idx}")

    elif title is None and idx is None:
        pass

    else:  # Title and Epoch both exists
        fig.suptitle(f"ep{idx} - {title}")

    fig.tight_layout()
    # axes[0].set_title('Saggital')
    axes[0].imshow(brain[slc, :, :], cmap="gray", interpolation="none")
    axes[0].imshow(vismap[slc, :, :], cmap="jet", interpolation="none", alpha=alpha)

    # axes[1].set_title('Coronal')
    axes[1].imshow(brain[:, slc, :], cmap="gray", interpolation="none")
    axes[1].imshow(vismap[:, slc, :], cmap="jet", interpolation="none", alpha=alpha)

    # axes[2].set_title('Horizontal')
    axes[2].imshow(brain[:, :, slc], cmap="gray", interpolation="none")
    axes[2].imshow(vismap[:, :, slc], cmap="jet", interpolation="none", alpha=alpha)

    if save:
        if not os.path.exists(att_path):
            os.mkdir(att_path)
        plt.savefig(f"{att_path}/{str(idx).zfill(3)}.png")
    plt.show()

    return fig


def convert2nifti(path, data, vismap):

    """
    path: path of original dataloaders', e.g. '../../brainmask_tlrc/PAL318_mpr_wave1_orig-brainmask_tlrc.npy'
    data: a single brain of 5-dim torch.tensor. Will be converted to numpy automatically
    vismap: attention map derived from any methods of - GradCAM, GBP, GuidedGCAM

    Does not return anything but instead saved 2 nifti files (registrated brain, visualization map) in
    ../../attmap_result_pairs/filename/*.nii.gz
    """

    ROOT = "../../attmap_result_pairs/"
    fname = brain_parser(path, full_path=False)[1]

    if not os.path.exists(f"{ROOT}{fname}"):
        os.mkdir(f"{ROOT}{fname}")

    try:
        # Make Affine
        affine = nib.load(brain_parser(path)).affine

        # Save vismap as nifti
        vismap_nifti = nib.Nifti1Image(vismap, affine)
        nib.save(vismap_nifti, f"{ROOT}{fname}/{fname}_attmap.nii.gz")

        # Save .npy brain as nifti
        brain = nib.Nifti1Image(data[0][0][0].numpy(), affine)
        nib.save(brain, f"{ROOT}{fname}/{fname}_brain.nii.gz")
        print("Saved")

    except:
        print("Error occurred")


def exp_parser(state):

    """
    Parses experiment path into date/epoch
    """

    date, pth_name = state.split("/")[-1].split("\\")
    model_name = pth_name.split("_ep")[0]
    epoch = pth_name.split("_ep")[-1].split("-")[0]

    return date, epoch


def brain_parser(path, full_path=True):

    """
    Parses path that contains registrated .npy file name into registrated .nii(NifTi) file
    full_path=True will return a single string, otherwise it will return a tuple of (root, .nii)
    """

    root = "/".join(path.split("/")[:2]) + "/brainmask_nii/"
    fname = path.split("/")[-1].split("_tlrc")[0] + ".nii"
    return root + fname if full_path else root, fname.split(".nii")[0]


def ep_mae_parser(_path):

    _path = _path.split("\\")[-1]

    ep = _path.split("_ep")[-1].split("-")[0]
    mae = _path.split("mae")[-1].split(".pth")[0]
    return f"EPOCH {ep.zfill(3)} | MAE {mae}"


def normalize(vismap, eps=1e-4):

    numer = vismap - np.min(vismap)
    denom = (vismap.max() - vismap.min()) + eps
    vismap = numer / denom
    vismap = (vismap * 255).astype("uint8")

    return vismap if len(vismap.shape) < 4 else vismap[0]


class Assembled(nn.Module):
    def __init__(self, encoder, regressor):

        super().__init__()
        self.encoder = encoder
        self.regressor = regressor

    def load_weight(self, weights: dict):

        for model_name, path in weights.items():

            if model_name == "encoder":
                self.encoder.load_state_dict(torch.load(path))

            elif model_name == "regressor":
                self.regressor.load_state_dict(torch.load(path))

        print("Weights successfully loaded!")

    def forward(self, x):

        out = self.encoder(x)
        out = self.regressor(out)

        return out

    @property
    def conv_layers(self):

        try:
            return self.encoder.conv_layers

        except:
            print("No conv_layers attribute supported for this model !")
            return
