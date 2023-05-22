""" Functions loading the saved anat/glass brain images and stack.
"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_imgs(root: Path = Path("assets/weights"),
              path: Path | List = Path("resnet10t-mask"),
              img_type: str = "glass",
              mask: bool = True,
              method: str = "gbp",
              top_k: float = None,
              indiv: bool = False) -> List[Path]:
    if indiv is False:
        top_k = None
    
    if img_type not in ["anat", "glass"]:
        img_type = "glass"
    
    indiv = "indiv" if indiv else "total"
    topk = f"-k{top_k}" if top_k is not None else ""
    mask = "mask" if mask else "no-mask"
    
    leaf_name = f"{method}-{indiv}{topk}"
    if isinstance(path, Path | str):
        path = [path]
    
    imgs = []
    for _path in path:
        img_path: Path = root / _path / mask / leaf_name
        imgs.extend(list(img_path.glob(f"{img_type}.png")))
    return imgs


def create_title(path: Path, img_dict: dict):
    
    method = img_dict.get("method")
    indiv = img_dict.get("indiv")
    mask = img_dict.get("mask")
    
    title = f"{str(path)}:: Infer Mask={mask} | Method={method}"
    return title
