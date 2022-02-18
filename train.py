import wandb
import os
import logging

logging.basicConfig(
    format="%(asctime)s(%(levelname)s) %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from sage.config import load_config
from sage.args import parse_args
from sage.training.trainer import MRITrainer

NAMEDICT = {"vanillaconv": "VanillaConv", "resnet": "ResNet", "convit": "ConViT"}


if __name__ == "__main__":

    cfg = load_config()

    # TODO Need These to be generalized
    cfg.phase_config = {"epochs": [200], "update": [["reg"]]}
    cfg.encoder.name = "resnet"
    cfg.encoder.config.start_channels = 32  # THIS IS THE PROBLEM !!
    # cfg.registration = "non_registered"

    args = parse_args()
    cfg.update(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg.augment_replacement = bool(cfg.augment_replacement)
    if cfg.debug is True:
        logger.setLevel(logging.DEBUG)
    logger.info(f"Use GPU {cfg.gpu_num}")
    
    trainer = MRITrainer(cfg)

    run_name = cfg.run_name if cfg.get("run_name") else "DEFAULT NAME"

    wandb.login()
    wandb.init(
        project="3d_smri",
        config=vars(cfg),
        name=run_name,
        tags=[
            NAMEDICT[cfg.encoder.name],
            "aug_replacement" if cfg.augment_replacement is True else "naive",
            "save",
            f"seed {cfg.seed}",
        ],
    )

    trainer.run(cfg)
