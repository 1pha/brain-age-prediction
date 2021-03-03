import os
import numpy as np
import pandas as pd
from glob import glob

from scipy.ndimage import shift
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchio as tio

class MyDataset(Dataset):
    def __init__(self, CFG, fold=None, augment=None):

        self.scaler = CFG.scaler
        self.augment = augment
        self.cfg = CFG
        
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
          
        self.transform = {
            'affine': tio.RandomAffine(),
            'flip':   tio.RandomFlip(axes=['left-right']),
            'elastic_deform': tio.RandomElasticDeformation()
        }
                        
    def __getitem__(self, idx):
        
        if self.scaler == 'minmax':
            x = np.load(self.data_files[idx])
            x = MinMaxScaler().fit_transform(x.reshape(-1, 1)).reshape(141, 172, 110)

        else:
            x = np.load(self.data_files[idx])

        if self.augment:
            
            # x = torch.tensor(shift(x, shift=[1, 1, 1]))[None, :, :].float()
            p = list(self.cfg['augmentation'].values())
            aug_choice = np.random.choice(list(self.transform.keys()), p=p)
            if self.cfg['verbose_loader']:
                print(aug_choice)

            else: pass

            if aug_choice == 'elastic_deform':
                fname = self.data_files[idx].replace('/brainmask_tlrc', '/brainmask_elasticdeform')
                x = np.load(fname)

            else:
                x = self.transform[aug_choice](x[None, ...])

        else:
            x = torch.tensor(x)[None, :, :].float()
            
        y = torch.tensor(self.label_file[idx]).float()
        
        return x, y

    def __len__(self):
        return len(self.data_files)


class DatasetPlus(Dataset):

    def __init__(self, cfg=None, augment=False, test=False):

        self.cfg = cfg
        self.augment = augment
        self.test = test

        # If it's a testset then augment should set to be false
        if test: assert augment == False

        # set seed
        RANDOM_STATE = cfg.seed
        np.random.seed(RANDOM_STATE)

        label_file = pd.read_csv(cfg.label_path, index_col=0)
        # cfg.unused_src: list
        unused = label_file['src'].apply(lambda row: row.lower() in map(str.lower, cfg.unused_src))
        label_file.drop(label_file[unused].index, inplace=True)
        
        label_file.loc[:, 'id'] = label_file.loc[:, ['id', 'src']].apply(DatasetPlus.path_maker, axis=1)

        train_idx, test_idx, train_age, test_age = train_test_split(label_file.id,
                                                                    label_file.age,
                                                                    test_size=cfg.test_size,
                                                                    random_state=RANDOM_STATE)
        if augment:
            aug_idx = train_idx.apply(lambda x: x+'*') 
            aug_age = train_age   

        if not test: # Training set
            self.data_files  = shuffle(pd.concat([train_idx, aug_idx]), random_state=RANDOM_STATE) if augment else train_idx
            self.data_labels = shuffle(pd.concat([train_age, aug_age]), random_state=RANDOM_STATE) if augment else train_age

        else: # Test set
            self.data_files  = test_idx
            self.data_labels = test_age

        self.data_files  = self.data_files.to_list()
        self.data_labels = self.data_labels.to_list()

    @staticmethod
    def path_maker(row, ROOT=None, SUFFIX=None):
    
        brain_id = row.id
        src = row.src
        
        ROOT = '../../brainmask_tlrc/' if ROOT is None else ROOT
        SUFFIX = '-brainmask_tlrc.npy' if SUFFIX is None else SUFFIX
        
        if src == 'Oasis3':
            SUFFIX = '_tlrc.npy'
            
        path = ROOT + brain_id + SUFFIX
        return path if os.path.exists(path) else brain_id

    
    def __getitem__(self, idx):

        if os.path.exists(self.data_files[idx]): # original data
            x = np.load(self.data_files[idx])

        else: # augmented data
            x = self.augmentation(self.data_files[idx][:-1])

        # Preprocessing
        if self.cfg.preprocess['scaler']:
            
            if self.cfg.preprocess['scaler'] == 'minmax':
                x = MinMaxScaler().fit_transform(x.reshape(-1, 1)).reshape(141, 172, 110)

            elif self.cfg.preprocess['scaler'] == 'znorm':
                x = StandardScaler().fit_transform(x.reshape(-1, 1)).reshape(141, 172, 110)

        else:
            x = np.reshape(-1, 1).reshape(141, 172, 110)

        if self.cfg.preprocess['resize']:
            x = F.interpolate(torch.tensor(x)[None, None, ...], size=self.cfg.preprocess['resize'])
            x = x.squeeze(0).float()

        else:
            x = torch.tensor(x)[None, ...].float()

        return x, torch.tensor(self.data_labels[idx]).float()

    def augmentation(self, path):

        self.transform = {
            'affine': tio.RandomAffine(),
            'flip':   tio.RandomFlip(axes=['left-right']),
            'elastic_deform': tio.RandomElasticDeformation()
        }

        p = list(self.cfg['augmentation'].values())
        norm = sum(p)
        p = list(map(lambda x: x / norm, p))
        aug_choice = np.random.choice(list(self.transform.keys()), p=p)
        if self.cfg['verbose_loader']:
            print(aug_choice)

        else: pass

        if aug_choice == 'elastic_deform':
            fname = path.replace('/brainmask_tlrc', '/brainmask_elasticdeform')
            x = np.load(fname)

        else:
            x = np.load(path)
            x = self.transform[aug_choice](x[None, ...])
        
        return x

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

    def delete(self, name):
        del self.data[name]
        delattr(self, name)

    def get_df(self):
        return pd.DataFrame(self.data)

    def info(self, label=False):
        if label:
            print(f'[{label.upper()}]:: ', end='')
        for name, data in self.data.items():
            print(f'{name.capitalize()}={data[-1]:.3f}', end=' ')
        print()


def gather_data(e=None, f=None, **kwargs):

    data = dict()
    if e is not None:
        data['Epoch'] = str(e)

    if f is not None:
        data['Fold'] = str(f)

    for key, value in kwargs.items():

        if key == 'train':
            
            data['Train MAE'] = value.data['mae'][-1]
            data['Train LOSS'] = value.data['loss'][-1]
            data['Train RMSE'] = value.data['rmse'][-1]
            data['Train CORR'] = value.data['corr'][-1]

        elif key == 'aug':
            
            data['Aug MAE'] = value.data['mae'][-1]
            data['Aug LOSS'] = value.data['loss'][-1]
            data['Aug RMSE'] = value.data['rmse'][-1]

        elif key == 'valid':
            
            data['Valid MAE'] = value.data['mae'][-1]
            data['Valid LOSS'] = value.data['loss'][-1]
            data['Valid RMSE'] = value.data['rmse'][-1]
            data['Valid CORR'] = value.data['corr'][-1]

        elif key == 'test':
            
            data['Test MAE'] = value.data['mae'][-1]
            data['Test LOSS'] = value.data['loss'][-1]
            data['Test RMSE'] = value.data['rmse'][-1]
            data['Test CORR'] = value.data['corr'][-1]

        elif key == 'time':
            data['Elapsed Time'] = value
            
        elif key == 'cfg':
            data['Learning Rate'] = value.learning_rate

    return data


def mlflow_data(**kwargs):

    data = dict()

    for key, value in kwargs.items():

        if key == 'train':
            
            data['Train_MAE'] = value.data['mae'][-1]
            data['Train_LOSS'] = value.data['loss'][-1]
            data['Train_RMSE'] = value.data['rmse'][-1]
            data['Train_CORR'] = value.data['corr'][-1]

        elif key == 'aug':
            
            data['Aug_MAE'] = value.data['mae'][-1]
            data['Aug_LOSS'] = value.data['loss'][-1]
            data['Aug_RMSE'] = value.data['rmse'][-1]

        elif key == 'valid':
            
            data['Valid_MAE'] = value.data['mae'][-1]
            data['Valid_LOSS'] = value.data['loss'][-1]
            data['Valid_RMSE'] = value.data['rmse'][-1]
            data['Valid_CORR'] = value.data['corr'][-1]

        elif key == 'test':
            
            data['Test_MAE'] = value.data['mae'][-1]
            data['Test_LOSS'] = value.data['loss'][-1]
            data['Test_RMSE'] = value.data['rmse'][-1]
            data['Test_CORR'] = value.data['corr'][-1]

        elif key == 'time':
            data['Elapsed_Time'] = value
            
        elif key == 'cfg':
            data['Learning_Rate'] = value.learning_rate

    return data


if __name__ == "__main__":

    train_dset = MyDataset()
    test_dset = MyDataset(test=True)

    train_loader = DataLoader(train_dset, batch_size=8)
    test_loader = DataLoader(test_dset, batch_size=8)