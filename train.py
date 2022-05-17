import math
import os

import wandb
from sage.config import (
    parse,
    get_logger,
    logger_conf,
)
from sage.data import get_dataloader
from sage.models import build_model
from sage.training.trainer import MRITrainer
from sage.utils import seed_everything, set_path


def run():

    data_args, training_args, misc_args = parse() # Parse Arguments
    seed_everything(misc_args.seed) # Fixate Seed

    # Set logger configuration. Change logger file to "/run.log"
    logger_conf["handlers"]["file_handler"]["filename"] = (
        misc_args.output_dir + "/run.log"
    )
    logger = get_logger(logger_conf)

    # Initiate wandb
    wandb.init(project="3dcnn_test", name=misc_args.run_name)

    # Build dataloaders
    train_dataloader = get_dataloader(data_args, misc_args, "train", logger)
    valid_dataloader = get_dataloader(data_args, misc_args, "valid", logger)
    test_dataloader = get_dataloader(data_args, misc_args, "test", logger)

    # Build Model
    model = build_model(training_args, logger)

    # Build Trainer
    trainer = MRITrainer(
        model,
        data_args=data_args,
        training_args=training_args,
        misc_args=misc_args,
        logger=logger,
        training_data=train_dataloader,
        validation_data=valid_dataloader,
    )

    # Start Training
    trainer.run()


if __name__ == "__main__":
    run()
