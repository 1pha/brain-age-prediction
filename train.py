import wandb
from sage.config import (
    argument_parser,
    get_logger,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    MiscArguments,
)
from sage.data.dataloader import get_dataloader
from sage.utils import seed_everything, set_path
from sage.training.trainer import MRITrainer
from sage.models import build_model

logger = get_logger()


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

    seed_everything(misc_args.seed)
    misc_args.output_dir = set_path(model_args, data_args, training_args, misc_args, logger)

    train_dataloader = get_dataloader(data_args, misc_args, "train", logger)
    valid_dataloader = get_dataloader(data_args, misc_args, "valid", logger)
    test_dataloader = get_dataloader(data_args, misc_args, "test", logger)
    model = build_model(model_args)

    trainer = MRITrainer(
        model,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        misc_args=misc_args,
        logger=logger,
        training_data=train_dataloader,
        validation_data=valid_dataloader
    )
    trainer.run()

    # run_name = cfg.run_name if cfg.get("run_name") else "DEFAULT NAME"

    # trainer.run(cfg)


if __name__ == "__main__":
    run()
