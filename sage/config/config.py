__all__ = ["ModelArguments", "DataArguments", "TrainingArguments", "MiscArguments"]

from dataclasses import asdict, dataclass, field
from typing import Union


@dataclass
class ModelArguments:

    """
    Model arguments.
    """

    model_name: str = field(
        default="resnet",
        metadata={
            "help": "Name of the model architecture or trained checkpoints that ends with '.pt' extension"
        },
    )


@dataclass
class DataArguments:
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
    affine_proba: float = field(
        default=0.33, metadata={"help": "Proportion of affine transform"}
    )
    flip_proba: float = field(
        default=0.33, metadata={"help": "Proportion of left-right flip transform"}
    )
    elasticdeform_proba: float = field(
        default=0.33, metadata={"help": "Proportion of elastic deformation transform"}
    )

    def __post_init__(self):
        self.affine_proba, self.flip_proba, self.elasticdeform_proba = self._normalize(
            self.affine_proba, self.flip_proba, self.elasticdeform_proba
        )

    def _normalize(self, *e):
        _list = [val for val in e]
        norm = sum(_list)
        new_list = list(map(lambda x: x / norm, _list))
        return new_list


@dataclass
class TrainingArguments:
    """
    Configurations related to training setup.
    """

    fp16: bool = field(
        default=True,
        metadata={"help": "Option to use mixed-precision to speed up training."},
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
    result_path: str = field(
        default="../result/",
        metadata={"help": "Where the checkpoints should be saved."},
    )
    epochs: int = field(default=100, metadata={"help": "Set how much epochs to train."})
    checkpoint_period: int = field(
        default=5,
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


@dataclass
class MiscArguments:
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


def arguments_to_dict(*args):

    arguments = dict()
    for a in args:
        arguments.update(**vars(a))
    return arguments


if __name__ == "__main__":

    from argument_parser import CustomParser

    parser = CustomParser((ModelArguments, MiscArguments))
    model_args, misc_args = parser.parse_args_into_dataclasses()
    print(arguments_to_dict(model_args, misc_args))
    # for v in iter(model_args):
    # print(v)
    # print(vars(model_args[0]))
