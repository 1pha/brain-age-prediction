import os
from glob import glob

import pandas as pd

ROOT = "G:/My Drive/brain_data/brainmask_*"  # <- Change here

label = pd.read_csv(f"{ROOT}/label.csv")
label = label.drop(["rel_path", "abs_path"], axis=1)

files = sorted(glob(ROOT + "/*.npy"))
label["rel_path"] = files
label["abs_path"] = label["rel_path"].apply(os.path.abspath)

label.to_csv(f"{ROOT}/label.csv", index=False)

if __name__ == "__main__":

    print("How to set label.csv for database. Just for reference.")
