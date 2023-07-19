from pathlib import Path
import pandas as pd
import numpy as np


SEG_ROOT = Path("./fastsurfer/seg/")
FS_ROOT = Path("fastsurfer")

META = pd.read_csv(FS_ROOT / "stats_meta.csv")

ASEG_AFFINE = np.array([[  -1.,    0.,    0.,  127.],
                        [   0.,    0.,    1., -145.],
                        [   0.,   -1.,    0.,  147.],
                        [   0.,    0.,    0.,    1.]])