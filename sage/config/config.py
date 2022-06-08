import json
import os
from dataclasses import dataclass, field


class BaseArgument:
    def to_dict(self):
        return vars(self)

    def save(self, output_dir):

        os.makedirs(f"{output_dir}/config", exist_ok=True)
        fname = f"{output_dir}/config/{self.get_name()}"
        configs = self.to_dict()
        with open(fname, "w") as f:
            json.dump(configs, f)

    def get_name(self):
        raise NotImplementedError

    def load(self, config: dict):

        for key, value in config.items():
            setattr(self, key, value)

    def load_json(self, config_file: str):

        with open(config_file, "r") as f:
            config = json.load(f)[self.get_name()]

        self.load(config)


@dataclass
class DataArguments(BaseArgument):
    """
    Configurations related to data/dataloader/augmentation
    """

    batch_size: int = field(
        default=16, metadata={"help": "Control batch size. Default=16"}
    )
    data_path: str = field(
        default="../brainmask_mni/", metadata={"help": "Root path of the data name."}
    )
    label_file: str = field(
        default="label.csv",
        metadata={"help": "Name of the label file. default=labels.csv"},
    )
    config_file: str = field(
        default="data_config.json",
        metadata={
            "help": "Name of the configuration files. Configurations here are for each dataset."
        },
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Boosts up GPU memory loading speed in dataloader."},
    )
    validation_ratio: float = field(
        default=0.2,
        metadata={
            "help": "Control ratio of the validation data. Please note that hold-out test size is always fixed to 10%, with seed 42 always."
        },
    )
    augmentation: str = field(
        default="replace",
        metadata={
            "help": "Choose which augmentation technique to use. It should be one of 'concat', 'replace' or 'false'"
        },
    )
    return_age_range: str = field(
        default="raw",
        metadata={
            "help": "Whether to use raw age or shrinked age. Use squash to divide by 100."
        },
    )

    def get_name(self):
        return "data_args"


@dataclass
class TrainingArguments(BaseArgument):
    """
    Configurations related to training setup.
    """

    model_name: str = field(
        default="resnet",
        metadata={
            "help": "Name of the model architecture or trained checkpoints that ends with '.pt' extension"
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Option to use mixed-precision to speed up training. However if your loss is too small, NaN values in gradients during training might occur."
        },
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Which optimizer to use. Default=adam"}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate. Default=1e-4"}
    )
    weight_decay: float = field(
        default=0, metadata={"help": "Weight decay for AdamW. Default=0"}
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "Set momentum values for optimizers that needs them."},
    )
    scheduler: str = field(
        default="",
        metadata={
            "help": "Which scheduler to use. Currently 'plateau', 'linear_warmup' and 'cosine_linear_warmup'."
        },
    )
    gamma: float = field(
        default=0.95,
        metadata={
            "help": "Factor multipled to learning rate every epoch step when using exponential decay scheduler."
        },
    )
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Percentage of total epochs to be warmed up."}
    )
    lr_patience: int = field(
        default=10, metadata={"help": "Patience for learning rate scheduler."}
    )
    epochs: int = field(default=200, metadata={"help": "Set how much epochs to train."})
    checkpoint_period: int = field(
        default=1,
        metadata={"help": "How much epochs to be used as a period between savings."},
    )
    early_patience: int = field(
        default=20,
        metadata={
            "help": "Set how much epochs to be await when validation loss does not improve."
        },
    )
    mae_threshold: float = field(
        default=8.0,
        metadata={
            "help": "In order to prevent models early stop before 'satisfying' performance, we set the threshold for early stopping."
        },
    )
    loss_fn: str = field(
        default="rmse",
        metadata={
            "help": "Designate loss functions as string. Note that MSE has too high value that makes model find hard to be optimized."
        },
    )
    # TODO: deal with multiple metrics
    metrics_fn: str = field(
        default="mae",
        metadata={"help": "List of metrics to measure during the training."},
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to train or not."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to train or not."})
    do_inference: bool = field(default=False, metadata={"help": "Infer test_data"})

    def get_name(self):
        return "training_args"

    def __post_init__(self):

        if not self.scheduler.endswith("warmup"):
            self.warmup_ratio = 0


@dataclass
class MiscArguments(BaseArgument):
    """
    Other minor configurations
    """

    seed: int = field(default=42, metadata={"help": "Which random seed to use."})
    debug: bool = field(
        default=False, metadata={"help": "Set to debug mode. Logs more output."}
    )
    force_cpu: bool = field(
        default=False,
        metadata={
            "help": "Force models/data to stay in CPU memory for some use cases."
        },
    )
    exclude_source: str = field(
        default=None,
        metadata={
            "help": "If one of labels shall not be used, put it here. Should be one of 'IXI', 'Dallas', 'Oas1' or 'Oas3'."
        },
    )
    data_proportion: float = field(
        default=0.1,
        metadata={"help": "Designate number/ratio of the total data when debug."},
    )
    output_path: str = field(
        default="../repvgg_result/", metadata={"help": "Root directory for results."}
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "Name of output directory. Final directory will be `output_path/output_dir/`. If None, dir name will be automatically made."
        },
    )
    overwrite_output: str = field(
        default=False,
        metadata={"help": "If set to True, allow output directory to be overwritten."},
    )
    which_gpu: int = field(
        default=1,
        metadata={
            "help": "Choose which gpu to use. -1 if you can deviate all of them."
        },
    )

    def get_name(self):
        return "misc_args"

    def __post_init__(self):

        # Set GPU Device
        if self.which_gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.which_gpu)


def arguments_to_dict(*args):

    arguments = dict()
    for a in args:
        arguments.update(**vars(a))
    return arguments


def parse():

    from .argument_parser import CustomParser

    parser = CustomParser(
        (
            DataArguments,
            TrainingArguments,
            MiscArguments,
        )
    )
    (
        data_args,
        training_args,
        misc_args,
    ) = parser.parse_args_into_dataclasses()

    from .path_utils import set_path

    misc_args.output_dir, misc_args.run_name = set_path(
        data_args, training_args, misc_args
    )
    return data_args, training_args, misc_args


if __name__ == "__main__":

    from argument_parser import CustomParser

    # parser = CustomParser((ModelArguments, MiscArguments))
    # model_args, misc_args = parser.parse_args_into_dataclasses()
    # print(arguments_to_dict(model_args, misc_args))
    # print(model_args.to_dict())
    # for v in iter(model_args):
    # print(v)
    # print(vars(model_args[0]))
