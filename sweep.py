import os
import ast
import argparse
from functools import partial
from typing import List, Callable

import yaml
import hydra
import omegaconf
import wandb

import sage


logger = sage.utils.get_logger(name=__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default="config", type=str, help="")
    parser.add_argument("--config_name", default="train.yaml", type=str, help="")
    parser.add_argument("--overrides", default="", type=str, help="")
    parser.add_argument("--version_base", default="1.1", type=str, help="")

    parser.add_argument("--sweep_cfg_name", default="sweep.yaml", type=str, help="")
    parser.add_argument("--wandb_project", default="brain-age", type=str, help="")
    parser.add_argument("--wandb_entity", default="1pha", type=str, help="")

    args = parser.parse_args()
    return args


def override_config(hydra_config: omegaconf.DictConfig,
                    update_dict: dict, config_path: str = "config") -> omegaconf.DictConfig:
    """
    hydra_config: Base config
    update_dict : Updated key-value pairs which should be merged into hydra_config.
                  This key contains "a.b.c" structure of keys, which makes it hard to merge
    """
    update_dict = dict(update_dict)
    for key, value in update_dict.items():
        key_list = key.split(".")
        nkeys = len(key_list)
        if nkeys == 1:
            # If no . found in key
            # This implies override from defaults
            _subcfg = load_yaml(config_path=f"{config_path}/{key}", config_name=f"{value}.yaml")
            hydra_config[key] = _subcfg
        else:
            _c = hydra_config[key_list[0]]
            for idx, _k in enumerate(key_list[1:]):
                if idx < nkeys - 2:
                    _c = _c[_k]
                else:
                    _c[_k] = value
    if "sweep" in hydra_config.hydra:
        # Configure directory for sweep. sweep_main_dir/subdir
        hydra_config.hydra.sweep.subdir = "_".join([f"{k}={v}" for k, v in update_dict.items()])
        dirpath = f"{hydra_config.hydra.sweep.dir}/{hydra_config.hydra.sweep.subdir}"
        hydra_config.callbacks.checkpoint.dirpath = dirpath
    return hydra_config


def load_default_hydra_config(config_path: str = "config",
                              config_name: str = "train.yaml",
                              version_base="1.1",
                              overrides: List[str] = []) -> omegaconf.DictConfig:
    with hydra.initialize(config_path=config_path, version_base=version_base):
        config = hydra.compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
    return config


def load_yaml(config_path: str = "config/sweep", config_name: str = "sweep.yaml") -> dict:
    with open(os.path.join(config_path, config_name), mode="r") as f:
        sweep_cfg = yaml.load(stream=f, Loader=yaml.FullLoader)
    return sweep_cfg


def main(config: omegaconf.DictConfig, config_path: str = "config") -> float:
    wandb.init(project="brain-age")
    logger.info("Sweep Config: %s", wandb.config)
    updated_config = override_config(hydra_config=config,
                                     update_dict=wandb.config,
                                     config_path=config_path)
    logger.info("Start Training")
    metric = sage.trainer.train(updated_config)
    return metric


if __name__=="__main__":
    args = parse_args()

    # Load hydra default configuration
    overrides = ast.literal_eval(args.overrides)
    config = load_default_hydra_config(config_path=args.config_path,
                                       config_name=args.config_name,
                                       overrides=overrides,
                                       version_base=args.version_base)
    func: Callable = partial(main, config=config, config_path=args.config_path)

    # Load wandb.sweep configuration and instantiation
    sweep_cfg = load_yaml(config_path=os.path.join(args.config_path, "sweep"),
                          config_name=args.sweep_cfg_name)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id=sweep_id, function=func)
