import os
import ast
from copy import deepcopy
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
    parser.add_argument("--version_base", default="1.3", type=str, help="")

    parser.add_argument("--sweep_cfg_name", default="sweep.yaml", type=str, help="")
    parser.add_argument("--wandb_project", default="brain-age", type=str,
                        help="Project name for training. Since we are using sweep, it is recommended to avoid `brain-age`\
                            and rather use dataset name as project name.")
    parser.add_argument("--wandb_entity", default="1pha", type=str, help="")
    parser.add_argument("--sweep_prefix", default="", type=str,
                        help="Prefix for sweep experiment run name.")

    args = parser.parse_args()
    return args


def load_hydra_config(config_path: str = "config",
                      config_name: str = "train.yaml",
                      version_base="1.3",
                      overrides: List[str] = [],
                      return_hydra_config: bool = False) -> omegaconf.DictConfig:
    with hydra.initialize(config_path=config_path, version_base=version_base):
        config = hydra.compose(config_name=config_name, overrides=overrides,
                               return_hydra_config=return_hydra_config)
    return config


def load_yaml(config_path: str = "config/sweep", config_name: str = "sweep.yaml") -> dict:
    with open(os.path.join(config_path, config_name), mode="r") as f:
        sweep_cfg = yaml.load(stream=f, Loader=yaml.FullLoader)
    return sweep_cfg


def override_config(hydra_config: omegaconf.DictConfig, update_dict: dict,
                    config_path: str = "config", prefix: str = "") -> omegaconf.DictConfig:
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
            _subcfg = load_hydra_config(config_path=f"{config_path}/{key}",
                                        config_name=f"{value}.yaml")
            hydra_config[key] = _subcfg
        else:
            _c = hydra_config[key_list[0]]
            for idx, _k in enumerate(key_list[1:]):
                if idx < nkeys - 2:
                    _c = _c[_k]
                else:
                    _c[_k] = value

    var_sweep = " ".join([f"{k[:3]}={v}" for k, v in update_dict.items()])
    ds_name = sage.utils.get_func_name(hydra_config.dataset) if hydra_config.get("dataset") else ""
    if "sweep" in hydra_config.get("hydra", []):
        # Configure directory for sweep. sweep_main_dir/subdir
        hydra_config.hydra.sweep.dir = f"{hydra_config.hydra.run.dir}-{ds_name}"
        hydra_config.hydra.sweep.subdir = var_sweep
        dirpath = f"{hydra_config.hydra.sweep.dir}/{var_sweep}"
        hydra_config.callbacks.checkpoint.dirpath = dirpath
        hydra_config.logger.name = f"{prefix}: {var_sweep}" if prefix else var_sweep

    return hydra_config


def main(config: omegaconf.DictConfig, config_path: str = "config", prefix: str = "") -> float:
    wandb.init(project="brain-age")
    _config = deepcopy(config)
    updated_config = override_config(hydra_config=_config, update_dict=wandb.config,
                                     config_path=config_path, prefix=prefix)
    wandb.run.name = updated_config.logger.name

    logger.info("Start Training")
    logger.info("Sweep Config: %s", wandb.config)
    metric = sage.trainer.train(updated_config)
    return metric


if __name__=="__main__":
    args = parse_args()

    # Load hydra default configuration
    overrides = ast.literal_eval(args.overrides or "[]")
    config = load_hydra_config(config_path=args.config_path,
                               config_name=args.config_name,
                               overrides=overrides,
                               version_base=args.version_base,
                               return_hydra_config=True)
    func: Callable = partial(main, config=config, config_path=args.config_path,
                             prefix=args.sweep_prefix)

    # Load wandb.sweep configuration and instantiation
    sweep_cfg = load_yaml(config_path=os.path.join(args.config_path, "sweep"),
                          config_name=args.sweep_cfg_name)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id=sweep_id, function=func)
