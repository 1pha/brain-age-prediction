from pathlib import Path

from tqdm import tqdm
import numpy as np

from sage.data import UKBDataset


def get_ukb_average(mode: str = "train",
                    save_dir: Path = Path("assets/average"),
                    **ds_kwargs) -> np.ndarray:
    dataset = UKBDataset(mode=mode, **ds_kwargs)
    norm = len(dataset)
    
    avg = dataset[0]["brain"] / norm
    iterable = range(1, len(dataset))
    pbar = tqdm(iterable=iterable,
                total=len(iterable),
                desc=f"Mode: {mode}")
    for idx in pbar:
        avg += (dataset[idx]["brain"] / norm)
    
    np.save(file=save_dir / f"{mode}.npy",
            arr=avg)
    return avg
