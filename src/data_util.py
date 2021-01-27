import numpy as np
import pandas as pd
from glob import glob

from scipy.ndimage import shift
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio

class MyDataset(Dataset):
    def __init__(self, CFG, fold=None, augment=None):

        self.scaler = CFG.scaler
        self.augment = augment
        
        RANDOM_STATE = CFG.seed
        np.random.seed(RANDOM_STATE)
            
        label_file = pd.read_csv(CFG.label_path, index_col=0)
        
        # idx = label_file[label_file['used'] == 1].index
        idx = label_file[label_file.src.map(lambda x: x not in ['Oasis3'])].index
        data_files = np.array(glob(CFG.npy_path))
        data_files.sort()
        data_files = data_files[idx].tolist()

        test_num = int(CFG.test_size * len(data_files))
        train_num = len(data_files) - test_num
        shuffled_index = np.random.permutation(len(data_files))

        train_fname = [data_files[i] for i in shuffled_index[:train_num]]
        test_fname = [data_files[i] for i in shuffled_index[-test_num:]]
        
        if CFG.test and fold is None:
            self.data_files = [data_files[i] for i in shuffled_index[-test_num:]]
            self.label_file = label_file[CFG.task_type].values[shuffled_index[-test_num:]]
            self.fname = label_file['id'].values[shuffled_index[-test_num:]]

        else:
            self.data_files = [data_files[i] for i in shuffled_index[:train_num]]
            self.label_file = label_file[CFG.task_type].values[shuffled_index[:train_num]]
            self.fname = label_file['id'].values[shuffled_index[:train_num]]
                
            if fold is not None:
                kfold = KFold(CFG.folds)
                for i, idx in enumerate(kfold.split(self.label_file)):
                    
                    if i == fold:

                        if test:
                            self.data_files = np.array(self.data_files)[idx[1]]
                            self.label_file = self.label_file[idx[1]]

                        else:
                            self.data_files = np.array(self.data_files)[idx[0]]
                            self.label_file = self.label_file[idx[0]]
                        break
        
        self.transform = tio.OneOf({
            tio.RandomAffine(),
            tio.RandomFlip(axes=['left-right']),
            tio.RandomElasticDeformation()
        })
                        

    def __getitem__(self, idx):
        
        if self.scaler == 'minmax':
            x = np.load(self.data_files[idx])
            x = MinMaxScaler().fit_transform(x.reshape(-1, 1)).reshape(141, 172, 110)

        else:
            x = np.load(self.data_files[idx])

        if self.augment:
            # x = torch.tensor(shift(x, shift=[1, 1, 1]))[None, :, :].float()
            x = self.transform(x[None, ...])

        else:
            x = torch.tensor(x)[None, :, :].float()
            
        y = torch.tensor(self.label_file[idx]).float()
        
        return x, y

    def __len__(self):
        return len(self.data_files)

class Data:

    def __init__(self, data=None):
        self.data = [] if data is None else data
        self.batch = []

    def update(self, val):
        self.data.append(val)

    def batch_update(self, val, bsz):
        self.batch.append((val, bsz))

    def clear(self):
        self.val = sum(b[0] for b in self.batch) / sum(b[1] for b in self.batch)
        self.update(self.val)
        self.batch = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return str(self.data)

class DataPacket:

    def __init__(self, names=None):
        self.data = {}        
        if names is None:
            for name in ['loss', 'mae', 'rmse', 'corr']:
                self.__setitem__(name, Data())

    def __setitem__(self, name, data):
        self.data[name] = list(data)
        return setattr(self, name, data)

    def refresh(self):
        for name in self.data:
            self.data[name] = list(getattr(self, name))

    def get_df(self):
        return pd.DataFrame(self.data)

    def info(self):
        for name, data in self.data.items():
            print(f'[{name.upper()}] {data[-1]:.3f}', end=' ')

def to_df(res: tuple):

    res = tuple()

if __name__ == "__main__":

    train_dset = MyDataset()
    test_dset = MyDataset(test=True)

    train_loader = DataLoader(train_dset, batch_size=8)
    test_loader = DataLoader(test_dset, batch_size=8)