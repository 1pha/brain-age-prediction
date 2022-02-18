import os
import logging
from pathlib import Path
from glob import glob
import warnings
import yaml

warnings.simplefilter("ignore", UserWarning)
import torch

from sage.config import load_config
from sage.training.trainer import MRITrainer
from sage.visualization.vistool import Assembled

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RESULT_DIR = "../resnet256_naive_nonreg_checkpoints/"
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

def inference(checkpoint):

    checkpoint = Path(checkpoint)
    cfg = load_config(Path(checkpoint, "config.yml"))
    logger.info(f"Starting seed {cfg.seed}")

    trainer = MRITrainer(cfg)
    model = Assembled(trainer.models["encoder"], trainer.models["regressor"]).to("cuda")
    test_results = {}
    for e in range(150):
        try:
            ckpt_dict, mae = load_model_ckpts(checkpoint, e)
            model.load_weight(ckpt_dict)
            logger.info(f"Load checkpoint epoch={e} | mae={mae}")

            test_preds = []
            for x, y, _ in trainer.test_dataloader:
                x, y = map(lambda x: x.to("cuda"), (x, y))
                pred = model(x)
                test_preds.append(pred)

                del x, y
                torch.cuda.empty_cache()

            test_preds = torch.cat(test_preds).squeeze().tolist()
            test_results[e] = test_preds

        except Exception as e:
            logger.exception(e)
            break

    with open(Path(checkpoint, "test.yml"), "w") as f:
        yaml.dump(test_results, f)


if __name__=="__main__":
    for checkpoint in checkpoint_lists[-2:]:
        inference(checkpoint)
    # inference(checkpoint_lists[-2:])
    # print(checkpoint_lists[-1])