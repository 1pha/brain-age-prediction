from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from tqdm import tqdm

original_path = glob("../../brainmask_tlrc/*.npy")
original_path.sort()


class VariationED:
    def __init__(self, cfg=(7, 7.5)):

        num_control_points, max_displacement = cfg
        self.cfg = {
            "num_control_points": num_control_points,  # default=7
            "max_displacement": max_displacement,  # default=7.5
        }

        self.transform = tio.RandomElasticDeformation(**self.cfg)

    def transform(self, original_brain):

        return self.transform(original_brain)

    def title(self):

        return f"Deformed, num_ctrl_pts={self.cfg['num_control_points']}, max_dspl={self.cfg['max_displacement']}"


def plot():

    original = np.load(original_path[0])[None, ...]

    cfg_combs = [
        (7, 7.5),  # default
        (7, 6.0),
        (7, 4.5),
    ]

    cfg_combs = [
        (7, 7.5),  # default
        (6, 7.5),
        (5, 7.5),
    ]

    cfg_combs = [
        (7, 7.5),  # default
        (5, 2.5),
        (5, 1.5),
    ]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    ax[0, 0].set_title("Original Brain")
    ax[0, 0].imshow(original[0][:, 86, :].T, cmap="gray", origin="lower")

    t = VariationED(cfg_combs[0])
    ax[0, 1].set_title(t.title())
    ax[0, 1].imshow(t.transform(original)[0][:, 86, :].T, cmap="gray", origin="lower")

    t = VariationED(cfg_combs[1])
    ax[1, 0].set_title(t.title())
    ax[1, 0].imshow(t.transform(original)[0][:, 86, :].T, cmap="gray", origin="lower")

    t = VariationED(cfg_combs[2])
    ax[1, 1].set_title(t.title())
    ax[1, 1].imshow(t.transform(original)[0][:, 86, :].T, cmap="gray", origin="lower")

    plt.show()


if __name__ == "__main__":

    elastic_deform = VariationED((5, 1.5))
    for file in tqdm(original_path):

        brain = np.load(file)
        fname = file.split("\\")[1][:-4]
        np.save(
            f"../../brainmask_elasticdeform/{fname}.npy",
            elastic_deform.transform(brain[None, ...]),
        )
