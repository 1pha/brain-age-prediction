import os
import imageio
import matplotlib.pyplot as plt

from glob import glob
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.models.model_util import load_model
from src.visualization.cams import *
from src.data.data_util import DatasetPlus


def plot_vismap(brain, vismap, masked=True, threshold=2,
                slc=48, alpha=.6, view=0, save=False, epoch=None):
    if masked:
        vismap = np.ma.masked_where(vismap < 2, vismap)
    
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

    fig.suptitle(f'Epoch {epoch}')

    axes[0].set_title('Saggital')
    axes[0].imshow(np.rot90(brain[slc, :, :]), cmap='gray', interpolation='none')
    axes[0].imshow(np.rot90(vismap[slc, :, :]), cmap='jet', interpolation='none', alpha=alpha)
    
    axes[1].set_title('Coronal')
    axes[1].imshow(np.rot90(brain[:, slc, :]), cmap='gray', interpolation='none')
    axes[1].imshow(np.rot90(vismap[:, slc, :]), cmap='jet', interpolation='none', alpha=alpha)

    axes[2].set_title('Horizontal')
    axes[2].imshow(np.rot90(brain[:, :, slc]), cmap='gray', interpolation='none')
    axes[2].imshow(np.rot90(vismap[:, :, slc]), cmap='jet', interpolation='none', alpha=alpha)
    
    if save:
        if not os.path.exists('./result/att_tmp_plots/'):
            os.mkdir('./result/att_tmp_plots/')
        plt.savefig(f'./result/att_tmp_plots/ep{str(epoch).zfill(3)}.png')
    plt.show()


def parser(state):
    
    date, pth_name = state.split('/')[-1].split('\\')
    model_name = pth_name.split('_ep')[0]
    epoch = pth_name.split('_ep')[-1].split('-')[0]
    
    return date, epoch


class Camsual:
    '''
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

            _, epoch = parser(state)
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