import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import warnings
from glob import glob
from pathlib import Path

warnings.simplefilter("ignore", UserWarning)
import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam
from einops import rearrange
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from sage.config import load_config
from sage.training.trainer import MRITrainer
from sage.visualization.vistool import Assembled

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# RESULT_DIR = "../resnet256_naive_checkpoints/"
RESULT_DIR = "../result/models/"
checkpoint_lists = sorted(glob(f"{RESULT_DIR}/*"))

checkpoint = Path(checkpoint_lists[0])


def load_model_ckpts(path: Path, epoch: int):

    epoch = str(epoch).zfill(3)
    ckpts = dict()
    for model_name in ("encoder", "regressor"):

        ckpt = list(path.glob(f"./{model_name}/ep{epoch}*.pt"))
        assert len(ckpt) == 1
        ckpts[model_name] = ckpt[0]

    mae = float(str(ckpt[0]).split("mae")[-1].split(".pt")[0])

    return ckpts, mae


for checkpoint in checkpoint_lists[-2:]:

    checkpoint = Path(checkpoint)
    cfg = load_config(Path(checkpoint, "config.yml"))
    cfg.registration = "mni"
    logger.info(f"Starting seed {cfg.seed}")
    cfg.force_cpu = True

    saliency_mm_dir = Path(f"{checkpoint}/npy_mm/")
    saliency_std_dir = Path(f"{checkpoint}/npy_std/")
    os.makedirs(saliency_mm_dir, exist_ok=True)
    os.makedirs(saliency_std_dir, exist_ok=True)
    trainer = MRITrainer(cfg)
    model = Assembled(trainer.models["encoder"], trainer.models["regressor"]).to("cuda")

    saliency_map_ep_naive = dict()
    for e in range(0, 151):
        try:
            ckpt_dict, mae = load_model_ckpts(checkpoint, e)
            model.load_weight(ckpt_dict)
            logger.info(f"Load checkpoint epoch={e} | mae={mae}")
        except:
            break

        saliency_map_ep_naive[e] = []
        for layer_idx, conv_layer in tqdm(enumerate(model.conv_layers()[:1])):

            layer_save_mm_dir = Path(f"{saliency_mm_dir}/layer{layer_idx}/")
            layer_save_std_dir = Path(f"{saliency_std_dir}/layer{layer_idx}/")

            os.makedirs(layer_save_mm_dir, exist_ok=True)
            os.makedirs(layer_save_std_dir, exist_ok=True)
            logger.info(f"Layer {layer_idx}: {conv_layer}")
            layer_gc = LayerGradCam(model, conv_layer)

            saliency_map = []
            for x, y, _ in trainer.test_dataloader:

                x, y = map(lambda x: x.to("cuda"), (x, y))
                attr = layer_gc.attribute(x)
                upsampled_attr = (
                    LayerAttribution.interpolate(
                        attr, (96, 96, 96), interpolate_mode="trilinear"
                    )
                    .cpu()
                    .detach()
                )
                saliency_map.append(upsampled_attr)

            saliency_map = rearrange(
                torch.cat(saliency_map, dim=0), "b c h w d -> b (c h w d)"
            ).numpy()
            num_test = saliency_map.shape[0]

            scaler = MinMaxScaler()
            saliency_minmax = rearrange(
                scaler.fit_transform(saliency_map.T),
                "(h w d) b -> d w h b",
                h=96,
                w=96,
                d=96,
            ).T
            saliency_map_avg = saliency_minmax.sum(axis=0) / num_test
            # saliency_map_ep_naive[e].append((conv_layer, saliency_map_avg))
            np.save(
                f"{layer_save_mm_dir}/ep{str(e).zfill(3)}_mae{mae}.npy",
                saliency_map_avg,
            )

            scaler = StandardScaler()
            saliency_std = rearrange(
                scaler.fit_transform(saliency_map.T),
                "(h w d) b -> d w h b",
                h=96,
                w=96,
                d=96,
            ).T
            saliency_map_avg = saliency_std.sum(axis=0) / num_test
            # saliency_map_ep_naive[e].append((conv_layer, saliency_map_avg))
            np.save(
                f"{layer_save_std_dir}/ep{str(e).zfill(3)}_mae{mae}.npy",
                saliency_map_avg,
            )

        clear_output()
