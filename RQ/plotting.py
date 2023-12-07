import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_corr(corr: pd.DataFrame, subtitle: str = "", subtitle_size: str | int = "large",
              hide_triu: bool = True, ax=None,
              cbar_size: float = 0.7, use_cbar: bool = True):
    if hide_triu:
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False
    else:
        mask = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    hm = sns.heatmap(corr, mask=mask, ax=ax,
                     vmin=-1, vmax=1, cmap="coolwarm",
                     cbar_kws={"shrink": cbar_size}, cbar=use_cbar,
                     annot=True, fmt=".2f", annot_kws={"size": 9},
                     square=True, linewidth=0.5)
    for i, model_name in enumerate(corr.index):
        model_name = model_name.split(" ")[0]
        if i == 0:
            prev = model_name
            continue
        if prev != model_name:
            if hide_triu:
                hm.axhline(i, xmin=0, xmax=i / len(corr), color="black", linewidth=1.2)
                hm.axvline(i, ymin=0, ymax=(len(corr) - i) / len(corr), color="black", linewidth=1.2)
            else:
                hm.axhline(i, color="black", linewidth=1.2)
        prev = model_name
    ax.set_title(subtitle, size=subtitle_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
