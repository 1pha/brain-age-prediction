import os
import subprocess
import multiprocessing as mp
from pathlib import Path

import nibabel as nib
from torch.utils.data import Dataset

from fastsurfer.io import UKBDataset
from fastsurfer.utils import get_logger

logger = get_logger(name=__file__)

ENV = os.environ.copy()
ENV["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

BASE_DIR = Path("/Users/daehyuncho/codespace/brain-age-prediction/fastsurfer")


def run_fs(idx: int, ds: Dataset):
    sample = ds[idx]
    stem = sample["fname"].stem
    logger.info("Start %s", stem)
    
    # Save nii
    nib.save(img=sample["nii"], filename=BASE_DIR / "tmp_nii" / stem)
    
    # Run Fastsurfer
    try:
        subprocess.call(["./run_fastsurfer.sh",
                        "--t1", BASE_DIR / f"tmp_nii/{stem}.nii",
                        "--sd", BASE_DIR / "seg",
                        "--sid", stem,
                        "--seg_only", "--allow_root", "--no_cereb",
                        "--threads", "1", "--device", "mps"], env=ENV)
        subprocess.call(["rm", BASE_DIR / f"tmp_nii/{stem}.nii"])
        return True
    except:
        return stem


def main():
    subprocess.call(["cd", "/Users/daehyuncho/FastSurfer"])
    ds = UKBDataset(mode="test")
    with mp.Pool(processes=4) as p:
        p.starmap(run_fs, ((idx, ds) for idx in range(len(ds))))
        
    
if __name__=="__main__":
    logger.info("Remember to use `fastsurfer` conda environment to run this")
    main()