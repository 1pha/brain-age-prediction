import numpy as np
import pandas as pd
from glob import glob

from scipy.ndimage import shift
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

from nilearn.image import get_data, crop_img
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchio as tio

class AugGrad:

    def __init__(self, pretrained_model, cfg, n_samples=25):

        self.pretrained_model = pretrained_model
        self.n_samples = n_samples
        self.cfg = cfg
        self.augmentation = cfg.augmentation

        scales, degrees = cfg.aug_intensity['affine']
        num_control_points, max_displacement = cfg.aug_intensity['elastic_deform']
        self.transform = {
            'affine': tio.RandomAffine(scales=scales, degrees=degrees),
            'flip': tio.RandomFlip(axes=['left-right']),
            'elastic_deform': tio.RandomElasticDeformation(num_control_points=num_control_points, max_displacement=max_displacement)
        }

        p = list(self.augmentation.values())
        norm = sum(p)
        self.p = list(map(lambda x: x / norm, p))
        

    def __call__(self, x, y, normalize=True, verbose=False):

        x.requires_grad = True
        output = self.pretrained_model(x).squeeze()
        print(f'[true]: {int(y.data.cpu())}')
        print(f'[pred]: {float(output.data.cpu()):.3f}')
        output.backward()
        total_gradients = x.grad.data.cpu().numpy()
        x.requires_grad = False
        for sample in range(self.n_samples):
 
            aug_choice = np.random.choice(list(self.transform.keys()), p=self.p)
            x_aug = self.transform[aug_choice](x[0].cpu()).to(self.cfg.device)[None, ...]
            x_aug.requires_grad = True
            output = self.pretrained_model(x_aug).squeeze()
            if verbose:
                print(f'{sample}th [{aug_choice}]: {float(output.data.cpu()):.3f}')
            (output - y.to(self.cfg.device)).backward()

            total_gradients += x_aug.grad.data.cpu().numpy()

        avg_gradients = total_gradients[0, ...] / self.n_samples

        return self.normalize(avg_gradients) if normalize else avg_gradients

    def normalize(self, vismap, eps=1e-4):

        numer = vismap - np.min(vismap)
        denom = (vismap.max() - vismap.min()) + eps
        vismap = numer / denom
        vismap = (vismap * 255).astype("uint8")

        return vismap


if __name__=="__main__":

    # Load Config
    cfg = load_config()

    # Load Model
    cfg.model_name = 'vanilla_residual'
    model = load_model(cfg.model_name, verbose=False, cfg=cfg)

    # Dataloader
    train_dataset = DatasetPlus(cfg, augment=False)
    sample_dl = DataLoader(train_dataset, batch_size=1)

    data = next(iter(sample_dl))

    # Make SmoothGrad Instance
    agrad = AugGrad(model, cfg)

    # Forward a single brain with age to the SmoothGrad instance
    smooth_grad = agrad(data[0][None, ...], data[1], verbose=True)