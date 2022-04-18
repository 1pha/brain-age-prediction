import math
import os

import wandb
from sage.config import (DataArguments, MiscArguments, ModelArguments,
                         TrainingArguments, argument_parser, get_logger,
                         logger_conf)
from sage.data import get_dataloader
from sage.models import build_model
from sage.training.trainer import MRITrainer
from sage.utils import seed_everything, set_path


def run():

    parser = argument_parser.CustomParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            MiscArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        misc_args,
    ) = parser.parse_args_into_dataclasses()

    # Fixate Seed
    seed_everything(misc_args.seed)

    # Set GPU device
    if misc_args.which_gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(misc_args.which_gpu)

    # Set saving path
    misc_args.output_dir, run_name = set_path(model_args, data_args, training_args, misc_args)

    # Set logger configuration. Change logger file to "/run.log"
    logger_conf["handlers"]["file_handler"]["filename"] = (
        misc_args.output_dir + "/run.log"
    )
    logger = get_logger(logger_conf)

    # Initiate wandb
    wandb.init(project="3dcnn_test", name=run_name)

    # Build dataloaders
    train_dataloader = get_dataloader(data_args, misc_args, "train", logger)
    if training_args.scheduler in ["linear_warmup", "cosine_linear_warmup"]:
        training_args.total_steps = int(
            math.ceil(len(train_dataloader.dataset) / data_args.batch_size)
        ) * (training_args.epochs)
        training_args.warmup_steps = training_args.total_steps // 10
    valid_dataloader = get_dataloader(data_args, misc_args, "valid", logger)
    test_dataloader = get_dataloader(data_args, misc_args, "test", logger)

    # Build Model
    model = build_model(model_args, logger)

    # Build Trainer
    trainer = MRITrainer(
        model,
        model_args=model_args,
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
