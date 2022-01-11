import numpy as np
from glob import glob

class Files:

    def __init__(self, _type="naive"):

        self._type = _type
        self.ROOT = {
            "naive": "../resnet256_naive_checkpoints",
            "augment": "../resnet256_augmentation_checkpoints",
        }[_type]

class Result:

    def __init__(self, data, _type="naive"):

        self._type = _type
        self.raw_data = data
        self.run_names = []
        for idx, (run_name, result) in enumerate(data.items()):
            setattr(self, f"result_{str(idx).zfill(3)}", result)
            self.run_names.append(run_name)
        self.epoch_organize()
    
    def __getitem__(self, idx):

        if isinstance(idx, int):
            return getattr(self, f"result_{str(idx).zfill(3)}")
        elif isinstance(idx, str):
            return self.raw_data[idx]
        else:
            raise f"Please give integer index or run_name (consisted of date). Given {idx}"

    def __getslice__(self, i, j):

        return [getattr(self, f"result_{str(idx).zfill(3)}") for idx in range(i, j)]

    def get_runname(self, idx):
        return self.run_names[idx]

    def __len__(self):
        return len(self.run_names)

    def epoch_organize(self):
        
        self.epoch_pivot = {}
        for idx, run_name in enumerate(self.run_names):
            
            # data of list[tuples, ...]
            data = self.raw_data[run_name]
            for d in data:
                e, mae = d
                self.epoch_pivot.setdefault(e, []).append(mae)

    @property
    def mean(self):
        return {e: np.mean(v) for e, v in self.epoch_pivot.items()}

    @property
    def std(self):
        return {e: np.std(v) for e, v in self.epoch_pivot.items()}


def transform(result):

    """
    turn [(e0, mae0), (e1, mae1), ... ] form into (list of epochs), (list of maes)
    """

    return [_[0] for _ in result], [_[1] for _ in result]