""" Convert h5 files to numpy
This will increase the storage amount,
but instead will boost up data loading speed.
Estimated storage for biobank is as follows:
- # of scans: 45k
- Storage per 20 scans to numpy: 1.1GB
- Storage estimated for 45k brains: 2.5TB

We will save this to /mnt/gpuHDD03, which is mounted as /home/data/hdd03/biobank_npy

* It took 8h 56m to convert 45,800 scans from h5 to npy.
"""
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import parmap as pm

from sage.data.dataloader import open_h5
from sage.constants import BIOBANK_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_workers", type=int, default=1, help="Number of multiprocessing workers.")
    parser.add_argument("--target_dir", type=str, default="/home/data/hdd03/biobank_npy", help="Target directory")
    parser.add_argument("--resume_index", type=int, default=0, help="Target index to resume")
    
    args = parser.parse_args()
    return args


def convert_h52npy(fname: Path, target_dir: Path):
    arr, _ = open_h5(fname=fname)
    target_fname = target_dir / f"{fname.stem}.npy"
    np.save(file=target_fname, arr=arr.astype(int))


def main(args):
    print("Start working on conversion")
    files = sorted(BIOBANK_PATH.rglob("*.h5"))
    if args.resume_index > 0:
        print(f"Resume index: {args.resume_index}")
        files = files[args.resume_index:]
    convert = partial(convert_h52npy, target_dir=Path(args.target_dir))
    result = pm.map(function=convert, iterable=files, pm_processes=args.num_workers, pm_pbar=True)
    return

if __name__=="__main__":
    args = parse_args()
    main(args)
