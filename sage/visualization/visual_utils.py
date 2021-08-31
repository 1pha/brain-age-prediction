import os
import imageio
import matplotlib.pyplot as plt

from glob import glob
import numpy as np

import torch
from torch._C import R
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
try: 
    from models.model_util import load_model
    from data.data_util import DatasetPlus
except:
    pass

from .cams import *
from .smoothgrad import *
from .auggrad import *


def check_type(brain):

    while brain.nidm > 3:
        brain = brain[0]

    if isinstance(brain, np.ndarray):
        return brain

    elif isinstance(brain, torch.tensor):
        return brain.permute(1, 2, 0).data.cpu().numpy()

        


def plot_vismap(brain, vismap, masked=True, threshold=2,
                slc=48, alpha=.6, save=False, att_path=None, idx=None, title=None):

    '''
    TODO - automated reshaping function. NUMPY <-> TORCH.TENSOR and 3D <-> 5D
    brain:
        Should take 3d input
    vismap:
        Also should be 3d.
    
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
    '''

    if masked:
        vismap = np.ma.masked_where(vismap < threshold, vismap)
    
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

    brain = np.rot90(brain[0][0].permute(1, 2, 0).data.cpu().numpy())
    vismap = np.rot90(torch.tensor(vismap).permute(1, 2, 0).data.cpu().numpy())

    if title is not None:
        fig.suptitle(title)

    elif title is None and idx is not None:
        fig.suptitle(f'Epoch {idx}')

    elif title is None and idx is None:
        pass

    else: # Title and Epoch both exists
        fig.suptitle(f'ep{idx} - {title}')

    fig.tight_layout()
    # axes[0].set_title('Saggital')
    axes[0].imshow(brain[slc, :, :], cmap='gray', interpolation='none')
    axes[0].imshow(vismap[slc, :, :], cmap='jet', interpolation='none', alpha=alpha)
    
    # axes[1].set_title('Coronal')
    axes[1].imshow(brain[:, slc, :], cmap='gray', interpolation='none')
    axes[1].imshow(vismap[:, slc, :], cmap='jet', interpolation='none', alpha=alpha)

    # axes[2].set_title('Horizontal')
    axes[2].imshow(brain[:, :, slc], cmap='gray', interpolation='none')
    axes[2].imshow(vismap[:, :, slc], cmap='jet', interpolation='none', alpha=alpha)
    
    if save:
        if not os.path.exists(att_path):
            os.mkdir(att_path)
        plt.savefig(f'{att_path}/{str(idx).zfill(3)}.png')
    plt.show()


def convert2nifti(path, data, vismap):

    '''
    path: path of original dataloaders', e.g. '../../brainmask_tlrc/PAL318_mpr_wave1_orig-brainmask_tlrc.npy'
    data: a single brain of 5-dim torch.tensor. Will be converted to numpy automatically
    vismap: attention map derived from any methods of - GradCAM, GBP, GuidedGCAM
    
    Does not return anything but instead saved 2 nifti files (registrated brain, visualization map) in
    ../../attmap_result_pairs/filename/*.nii.gz 
    '''
    
    ROOT = '../../attmap_result_pairs/'
    fname = brain_parser(path, full_path=False)[1]
    
    if not os.path.exists(f'{ROOT}{fname}'):
        os.mkdir(f'{ROOT}{fname}')
    
    try:
        # Make Affine
        affine = nib.load(brain_parser(path)).affine

        # Save vismap as nifti
        vismap_nifti = nib.Nifti1Image(vismap, affine)
        nib.save(grad_cam, f'{ROOT}{fname}/{fname}_attmap.nii.gz')

        # Save .npy brain as nifti
        brain = nib.Nifti1Image(data[0][0][0].numpy(), affine)
        nib.save(brain, f'{ROOT}{fname}/{fname}_brain.nii.gz')
        print('Saved')
    
    except:
        print('Error occurred')


def exp_parser(state):

    '''
    Parses experiment path into date/epoch
    '''
    
    date, pth_name = state.split('/')[-1].split('\\')
    model_name = pth_name.split('_ep')[0]
    epoch = pth_name.split('_ep')[-1].split('-')[0]
    
    return date, epoch

def brain_parser(path, full_path=True):

    '''
    Parses path that contains registrated .npy file name into registrated .nii(NifTi) file
    full_path=True will return a single string, otherwise it will return a tuple of (root, .nii)
    '''
    
    root = '/'.join(path.split('/')[:2])+'/brainmask_nii/'
    fname = path.split('/')[-1].split('_tlrc')[0]+'.nii'
    return root + fname if full_path else root, fname.split('.nii')[0]


def ep_mae_parser(_path):
    
    _path = _path.split('\\')[-1]
    
    ep = _path.split('_ep')[-1].split('-')[0]
    mae = _path.split('mae')[-1].split('.pth')[0]
    return f'EPOCH {ep.zfill(3)} | MAE {mae}'


def normalize(vismap, eps=1e-4):

    numer = vismap - np.min(vismap)
    denom = (vismap.max() - vismap.min()) + eps
    vismap = numer / denom
    vismap = (vismap * 255).astype("uint8")

    return vismap if len(vismap.shape) < 4 else vismap[0]


class Camsual:

    '''
    DEPRECATED
    This is for GradCAM and a prototype
    Takes configuration file and saved models path then -
    1. Make a dataset (to pickup samples)
    2. Load Model architecture
    3. Consequently load models onto the architecture and then see the visuals
    '''

    def __init__(self, cfg, path):

        self.cfg = cfg

        # Load Dataset
        ds = DatasetPlus(cfg, augment=False)
        self.dl = DataLoader(ds, batch_size=1)

        self.model, self.device = load_model(cfg.model_name, verbose=False, cfg=cfg)

        ROOT = './result/models/'
        SUFFIX = '/*.pth'
        self.saved_models = sorted(glob(f'{ROOT}{path}{SUFFIX}'), \
                              key=lambda x: int(x.split('ep')[1].split('-')[0]))


    def visualize(self, layer_idx, data=None):

        if data is None:
            data = next(iter(self.dl))

        for idx, state in enumerate(self.saved_models):

            _, epoch = exp_parser(state)
            self.model.load_state_dict(torch.load(state))
            resized_cam = run_gradcam(self.model, data, self.cfg)[layer_idx]
            plot_vismap(data[0][0][0], resized_cam, alpha=.4, masked=False, save=True, epoch=epoch)

    
    def save_gif(self, gif_path: str):

        '''
        gif_path should be 
            1. string  
            2. that ends with .gif
        '''

        with imageio.get_writer(f'./result/gifs/{gif_path}', mode='I') as writer:
            for files in sorted(glob('./result/att_tmp_plots/*.png')):
                image = imageio.imread(files)
                writer.append_data(image)


class Assembled(nn.Module):


    def __init__(self, encoder, regressor):

        super().__init__()
        self.encoder = encoder
        self.regressor = regressor


    def load_weight(self, weights: dict):

        for model_name, path in weights.items():

            if model_name == 'encoder':
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
            print("No conv_layers supported for this model !")
            return