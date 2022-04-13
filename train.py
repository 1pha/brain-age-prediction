import wandb
from sage.config import argument_parser, config, logging_config
from sage.data.dataloader import get_dataloader

# from sage.training.trainer import MRITrainer
logger = logging_config.get_logger()

def run():

    parser = argument_parser.CustomParser(
        (
            config.ModelArguments,
            config.DataArguments,
            config.TrainingArguments,
            config.MiscArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        misc_args,
    ) = parser.parse_args_into_dataclasses()

    train_dataloader = get_dataloader(data_args, misc_args, "train", logger)
    # trainer = MRITrainer(cfg)

    # run_name = cfg.run_name if cfg.get("run_name") else "DEFAULT NAME"

    # trainer.run(cfg)


if __name__ == "__main__":
    run()
