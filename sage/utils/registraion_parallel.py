import argparse
import time
from datetime import datetime
from glob import glob

from registration import *

parser = argparse.ArgumentParser(description="Registration work")
parser.add_argument(
    "--template",
    "-t",
    type=int,
    default=None,
    help="Which model to use, default=vanilla",
)
parser.add_argument(
    "--start", "-s", type=int, default=None, help="Where to start registration"
)
parser.add_argument(
    "--end", "-e", type=int, default=None, help="Where to end registration"
)
args = parser.parse_args()


if __name__ == "__main__":

    data_files = glob("../../../brainmask_nii/*.nii")
    data_files.sort()

    registrator = Registrator(cfg="registration.yml")
    start_time = time.time()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Initiated at {dt_string}.")
    for i, moving in enumerate(data_files[args.start : args.end]):

        print(
            f"{i}th brain | start={args.start} & end={args.end} | Progress {i / (args.end - args.start) * 100:.2f}%"
        )
        registrator(moving, save=True)
        print(f"Elapsed {time.time() - start_time:.1f} sec | Initiated at {dt_string}.")
