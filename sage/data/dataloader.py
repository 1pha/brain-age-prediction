# BASICS
import os
import numpy as np
import pandas as pd
from functools import partial

# SCIKIT-LEARN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# MRI RELATED
import nibabel as nib

# TORCH
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AUGMENTATION
import torchio as tio

class BrainAgeDataset(Dataset):

    def __init__(self, cfg, augment=False, test=False):

        '''
            CONFIG file should contain .csv file and -
            that .csv file should contain 'path' columns that contains full absolute path of the file
        '''

        # INITIAL SETUP
        self.cfg = cfg
        self.augment = augment
        self.test = test
        SEED = cfg.seed

        # DEBUG SETUP
        self.debug = cfg.debug
        for d in self._debug:
            setattr(self, d, self._debug[d] if self.debug else False)
        if not cfg.debug: # FLUSHOUT DEBUG ATTRS
            cfg._debug = []

        # VALIDATION SET SHOULD NOT DO AUGMENTATION
        if test: assert augment == False

        # LABEL FILE
        ROOT = cfg.root
        self.label_file = pd.read_csv(os.path.join(ROOT, 'label.csv'))

        # SPLIT DATA
        trn_idx, val_idx, trn_age, val_age = train_test_split(\
            X=self.label_file['abs_path'],
            y=self.label_file['age'],
            test_size=cfg.test_size,
            random_state=SEED
        )

        # AUGMENTATION
        if augment:
            aug_idx = trn_idx.apply(lambda x: x + 'aug')
            aug_age = trn_age

        else: # NO AUGMENTATION
            cfg.aug_proba = []
            cfg.aug_intensity = []
            self.augmentation = lambda x: x

        # SETUP DATA_FILES
        if not test: # TRAIN SET
            self.data_files = shuffle(pd.concat([trn_idx, aug_idx]), random_state=SEED) \
                if augment else trn_idx
            self.data_ages = shuffle(pd.concat([trn_age, aug_age]), random_state=SEED) \
                if augment else trn_age

        else: # VALIDATION SET
            self.data_files = val_idx
            self.data_ages = val_age

        self.data_files = self.data_files.to_list()
        self.data_ages = self.data_ages.to_list()


    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, idx):

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
            fpath = fpath[-3:]

        x = self.load(fpath)
        x = self.preprocess(x)
        if aug:
            x = self.augmentation(x)

        return x, torch.Tensor(self.data_ages[idx]).float()

    def load(self, fpath):

        '''
        only 2 extensions available
            .nii: nib.load().get_fdata()
            .npy: np.load()
        '''

        return nib.load(fpath).get_fdata()

    def preprocess(self, x):

        '''
        Given with raw brain np.ndarray
            -> return desired output shape (1, W', H', D')

        MAY CONTAIN
            1. SCALING
            2. RESIZING
            3. ROTATION (IF NEEDED)
        '''

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

        p = list(map(lambda x: x / sum(self.aug_proba), self.aug_proba))
        aug_choice = np.random.choice(list(transform.keys()), p=p)

        if self.aug_verbose:
            print(f'Augmentation Choice {aug_choice}')

        x = aug_choice(x)
        
        return x


class TLRCDataset(BrainAgeDataset):

    def __init__(self, cfg=None, augment=False, test=False):
        super().__init__(cfg, augment, test)


class MNIDataset(BrainAgeDataset):

    def __init__(self, cfg=None, augment=False, test=False):
        super().__init__(cfg, augment, test)
        

    def __getitem__(self, idx):

        pass

class RAWDataset(BrainAgeDataset):

    def __init__(self, cfg=None, augment=False, test=False):
        super().__init__(cfg, augment, test)



def get_dataloader(cfg, augment, test):

    if cfg.registration == 'mni':
        dataset = MNIDataset(cfg, augment=augment, test=test)

    elif cfg.registration == 'tlrc':
        dataset = TLRCDataset(cfg, augment=augment, test=test)

    elif cfg.registration == 'raw':
        dataset = RAWDataset(cfg, augment=augment, test=test)
    
    else: # DEPRECATED
        dataset = BrainAgeDataset(cfg, augment=augment, test=test)

    return DataLoader(dataset, batch_size=cfg.batch_size)

if __name__ == "__main__":

    pass