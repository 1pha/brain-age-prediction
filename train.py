import wandb

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

    args = parse_args()
    cfg.update(args)

    trainer = MRITrainer(cfg)

    run_name = cfg.run_name if cfg.get("run_name") else "DEFAULT NAME"

    wandb.login()
    wandb.init(
        project="3d_smri",
        config=vars(cfg),
        name=run_name,
        tags=[NAMEDICT[cfg.encoder.name], "baseline", "save", f"seed {cfg.seed}"],
    )

    trainer.run(cfg)
