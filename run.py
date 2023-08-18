import os
import argparse
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
FS_DIR = Path("/Users/daehyuncho/FastSurfer")


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
                         "--threads", "1", "--device", "mps"],
                         env=ENV, cwd=FS_DIR)
        subprocess.call(["rm", BASE_DIR / f"tmp_nii/{stem}.nii"])
        return True
    except:
        raise


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_index", default=0, type=int, help="Index for brain age")
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--num_proc", default=6, type=int, help="num_proc for multiprocess")
    
    args = parser.parse_args()
    return args


def main(args):
    ds = UKBDataset(mode="test")
    with mp.Pool(processes=args.num_proc) as p:
        p.starmap(run_fs,
                  ((idx, ds) for idx in range(args.start_index, args.start_index + args.num_samples)))


if __name__=="__main__":
    logger.info("Remember to use `fastsurfer` conda environment to run this")
    args = parse_args()
    main(args)
