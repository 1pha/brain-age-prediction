from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def remove_duplicates(files: List[Path]) -> List:
    """ Remove same patients with multiple scans,
    in order to be used as a longitudinal studies hold-out test data"""
    orig_files = sorted(files)
    files = sorted(map(lambda x: x.stem.split("_"), orig_files))
    files_df = pd.DataFrame(files)
    files_df.columns = ["pid", "val1", "count", "post"]
    
    long_bool = files_df.duplicated(subset="pid", keep=False)
    orig_files = np.array(orig_files)[~long_bool]
    return orig_files.tolist()
