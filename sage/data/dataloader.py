# BASICS
import os
import numpy as np
import pandas as pd

# SCIKIT-LEARN
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# PREPROCESS
from .preprocess import *

# CONFIG
from ..config import load_config

# MRI RELATED
import nibabel as nib

# TORCH
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AUGMENTATION
import torchio as tio


def get_loader(extension):

    return {
        'npy': np.load,
        'nii': lambda x: nib.load(x).get_fdata()
    }[extension]


class BrainAgeDataset(Dataset):

    def __init__(self, cfg, test=False):

        '''
            CONFIG file should contain .csv file and -
            that .csv file should contain 'path' columns that contains full absolute path of the file

            ROOT is the path of database.
            In this folder, we need - (**VERY IMPORTANT**)
                - label.csv
                - data_config.yml: should contain -
                    - data extension
                    - preprocessing method
                    - maximum volume (for mni)
                - data
        '''

        # INITIAL SETUP
        self.cfg = cfg
        ROOT = cfg.root
        SEED = cfg.seed
        self.data_cfg = load_config(os.path.join(ROOT, 'data_config.yml')) # -> Edict
        self.load = get_loader(extension=self.data_cfg.extension)
        self.augment = cfg.augment
        self.test = test

        # DEBUG SETUP
        self.debug = cfg.debug
        for d in cfg.data_debug:
            setattr(self, d, cfg.data_debug[d] if self.debug else False)
        if not cfg.debug: # FLUSHOUT DEBUG ATTRS
            cfg.data_debug = []

        # VALIDATION SET SHOULD NOT DO AUGMENTATION
        if test: self.augment = False

        # LABEL FILE
        self.label_file = pd.read_csv(os.path.join(ROOT, 'label.csv'))

        # EXCLUDE UNUSED SOURCE DATABASES
        self.label_file = self.label_file[self.label_file['src'].apply(lambda x: x not in cfg.unused_src)]

        # SPLIT DATA
        trn_idx, val_idx, trn_age, val_age = train_test_split(
            self.label_file['abs_path'],
            self.label_file['age'],
            test_size=cfg.test_size,
            random_state=SEED
        )

        # AUGMENTATION
        if self.augment:
            aug_idx = trn_idx.apply(lambda x: x + 'aug')
            aug_age = trn_age

        else: # NO AUGMENTATION
            cfg.aug_proba = []
            cfg.aug_intensity = []
            self.augmentation = lambda x: x

        # SETUP DATA_FILES
        if not test: # TRAIN SET
            # TODO: AUGRATIO
            self.data_files = shuffle(pd.concat([trn_idx, aug_idx]), random_state=SEED) \
                if self.augment else trn_idx
            self.data_ages = shuffle(pd.concat([trn_age, aug_age]), random_state=SEED) \
                if self.augment else trn_age

        else: # VALIDATION SET
            self.data_files = val_idx
            self.data_ages = val_age

        self.data_files = self.data_files.to_list()
        self.data_ages = self.data_ages.to_list()


    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, idx): # -> ((1, W', H', D'), age: torch.tensor.float)

        '''
        PIPELINE
        1. LOAD BRAIN (x = self.load(self.data_files[idx])) -> np.ndarray: (1, W, H, D)
            - absolute path(:str) is given to load method.

        2. PREPROESS BRAIN (x = self.preprocess(x)) -> torch.tensor: (1, W', H', D')
            - preprocess with certain process
            2+. AUGMENTATION (x = self.augmentation(x)) -> torch.tensor: (1, W', H', D')
                
        3. RETURN (BRAIN, AGE)
            - (torch.tensor(x, dtype=torch.float), torch.tensor(self.data_ages[idx]).float())
        '''

        fpath = self.data_files[idx]
        aug = True if fpath[-3:] == 'aug' else False
        if aug:
            fpath = fpath[:-3]

        x = self.load(fpath)   # 3D (W, H, D)
        x = self.maxcut(x)     # 3D (W, H, D)
        x = self.preprocess(x) # 4D (1, W', H', D')
        if aug:
            x = self.augmentation(x) # 4D (1, W', H', D')

        return x, torch.tensor(self.data_ages[idx]).float()


    def maxcut(self, x):
        '''
        For brains that has many blanks.
        Should explicity give maxcut with tuples of tuples ((w, W), (h, H), (d, D))
        '''
        
        maxcut = self.data_cfg.maxcut if self.data_cfg.maxcut else None
        if maxcut is not None:
            (w, W), (h, H), (d, D) = maxcut
            return x[w:W, h:H, d:D]
        else:
            return x


    def preprocess(self, x): # -> (1, W', H', D')

        '''
        Given with raw brain np.ndarray
            -> return desired output 4D shape (1, W', H', D')

        MAY CONTAIN
            1. SCALING
            2. RESIZING
            3. ROTATION (IF NEEDED)
        '''

        # 1. SCALING
        size = x.shape
        x = get_scaler(self.data_cfg.scaler).fit_transform(x.reshape(-1, 1)).reshape(*size)

        # 2. RESIZING
        # PRIORITY: cfg > data_cfg <- cfg is set later than data_cfg
        resize = self.data_cfg.resize if self.cfg.resize is None else self.cfg.resize
        if not resize is None:
            # (1, 1, *resize) (5D) -> because F.interpolate requires 5D tensor for 3D tensor to be torted
            x = F.interpolate(torch.tensor(x)[None, None, ...], size=resize) 
            x = x.squeeze(0).float() # -> (1, *resize) (4D)

        else:
            x = torch.tensor(x)[None, ...].float()

        return x


    def augmentation(self, x: torch.tensor): # -> torch.Tensor (1, W', H', D')
        
        '''
        x must be given with torch.tensor with (1, W', H', D')
        '''

        transform = {
            'affine': tio.RandomAffine(),
            'flip':   tio.RandomFlip(axes=['left-right']),
            'elastic_deform': tio.RandomElasticDeformation()
        }

        # TODO: Normalize probability but order of probabilities should be handled!
        # e.g. if probability order is [flip, ela, affine], then it will not give the expected output
        p = list(map(lambda x: x / sum(self.cfg.aug_proba.values()), self.cfg.aug_proba.values()))
        aug_choice = np.random.choice(list(transform.keys()), p=p)

        if self.aug_verbose:
            print(f'Augmentation Choice: {aug_choice.capitalize()}')

        x = transform[aug_choice](x)
        
        return x

    def configuration(self):

        return self.cfg, self.data_cfg


def get_dataloader(cfg, test=False, return_dataset=False):
    '''
    Just giving cfg.registration will find a proper path
    '''

    cfg.root = {
        'tlrc': 'G:/My Drive/brain_data/brainmask_tlrc',
        'mni': 'G:/My Drive/brain_data/brainmask_mni',
        'raw': 'G:/My Drive/brain_data/brainmask_nii',
    }[cfg.registration]

    dataset = BrainAgeDataset(cfg, test=test)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)
    return dataloader if not return_dataset else dataset


if __name__ == "__main__":

    pass