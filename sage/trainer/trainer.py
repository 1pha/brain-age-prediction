import os
import random
import subprocess
from typing import Any
from pathlib import Path

import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
import wandb

import sage
from sage.xai.nilearn_plots import plot_overlay, plot_glass_brain, plot_brain
from .utils import load_mask, finalize_inference, tune_logging_interval, tune_lr_interval


logger = sage.utils.get_logger(name=__name__)


class PLModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: omegaconf.DictConfig,
                 metrics: dict,
                 mask: Path | str | torch.Tensor = None,
                 mask_threshold: float = 0.1,
                 test_loader: torch.utils.data.DataLoader = None,
                 predict_loader: torch.utils.data.DataLoader = None,
                 log_train_metrics: bool = False,
                 augmentation: omegaconf.DictConfig = None,
                 scheduler: omegaconf.DictConfig = None,
                 load_model_ckpt: str = None,
                 load_from_checkpoint: str = None, # unused params but requires for instantiation
                 separate_lr: dict = None):
        super().__init__()
        self.model = model

        # Dataloaders
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.test_dataloader = test_loader
        self.predict_dataloader = predict_loader
        # num_training_steps for linear warmup scheduler from transformers
        # Fix n_epochs to 100
        num_training_steps = round(len(train_loader) * 100)

        # Optimizers
        self.opt_config = self._configure_optimizer(optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    separate_lr=separate_lr,
                                                    num_training_steps=num_training_steps)

        # Metrics Configuration: iterate through dict: {_target_: ...}
        metrics = MetricCollection(metrics=[
            hydra.utils.instantiate(metrics[m]) for m in metrics.keys() if "_target_" in metrics[m]
        ])
        self.in_metric = metrics.clone(prefix="train_")
        self.valid_metric = metrics.clone(prefix="valid_")

        if load_model_ckpt:
            logger.info("Load checkpoint from %s", load_model_ckpt)
            self.model.load_from_checkpoint(load_model_ckpt)

        self.log_train_metrics = log_train_metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.aug_config = augmentation
        self.mask = mask
        self.mask_threshold = mask_threshold
        
    def setup(self, stage):
        mask = load_mask(mask_path=self.mask,
                         mask_threshold=self.mask_threshold)
        self.no_augment = sage.data.no_augment(mask=mask)
        self.augmentor = hydra.utils.instantiate(self.aug_config, mask=mask)
        self.log_brain(return_path=False)
        
    def log_brain(self, return_path: bool = False, augment: bool = True):
        """ Logs sample brain to check how augmentation is applied. """
        ds = self.train_dataloader.dataset
        idx: int = random.randint(a=0, b=len(ds))
        brain: torch.Tensor = ds[idx]["brain"]
        brain = self.augmentor(brain[None, ...]) if augment else self.no_augment(brain[None, ...])
        
        tmp = "tmp.png"
        plot_brain(brain, save=tmp)
        try:
            wandb.log({"sample": wandb.Image(tmp)})
        except:
            logger.info("Not using wandb. Skip logging brain")
        
        if return_path:
            return tmp
        else:
            subprocess.run(args=["rm", tmp])

    def _configure_optimizer(self,
                             optimizer: omegaconf.DictConfig,
                             scheduler: omegaconf.DictConfig,
                             num_training_steps: int,
                             separate_lr: omegaconf.DictConfig = None):
        use_sch: bool = scheduler.scheduler is not False
        if separate_lr is not None:
            _opt_groups = []
            for _submodel, _lr in separate_lr.items():
                submodel = getattr(self.model, _submodel, None)
                if submodel is None:
                    logger.warn("separate_lr was given but submodel was not found: %s", _submodel)
                    opt_config = self._configure_optimizer(optimizer=optimizer,
                                                      scheduler=scheduler)
                    break
                _opt_groups.append(
                    {"params": submodel.parameters(), "lr": _lr}
                )
            _opt = sage.utils.parse_hydra(config=optimizer, params=_opt_groups)
            if use_sch:
                _sch = self.configure_scheduler(
                        optimizer=_opt,
                        scheduler=scheduler,
                        num_training_steps=num_training_steps
                       ) if use_sch else None
                opt_config = {"optimizer": _opt, "lr_scheduler": dict(**_sch)}
            else:
                opt_config = {"optimizer": _opt}
        else:
            opt = hydra.utils.instantiate(optimizer, params=self.model.parameters())
            sch = self.configure_scheduler(
                    optimizer=opt,
                    scheduler=scheduler,
                    num_training_steps=num_training_steps
                    ) if use_sch else None
            opt_config = {
                "optimizer": opt, "lr_scheduler": dict(**sch)
            } if use_sch else opt
        return opt_config

    def configure_scheduler(self,
                            optimizer: torch.optim.Optimizer,
                            scheduler: omegaconf.DictConfig,
                            num_training_steps: int = None):
        """ Do NOT feed `num_training_steps` if not required. """
        struct = {"optimizer": optimizer}
        try:
            sch = hydra.utils.instantiate(scheduler, scheduler=struct)
        except Exception as e:
            try:
                if num_training_steps:
                    struct.update({"num_training_steps": num_training_steps})
                sch = hydra.utils.instantiate(scheduler, scheduler=struct)
            except Exception as e:
                logger.exception(e)
                raise
        return sch

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        return self.opt_config

    def forward(self, batch, mode: str = "train"):
        try:
            """ model should return dict of
            {
                loss:
                cls_pred:
                cls_target:
            }
            """
            augmentor = self.augmentor if mode == "train" else self.no_augment
            batch["brain"] = augmentor(batch["brain"])
            batch["age"] = batch["age"].float()
            result: dict = self.model(**batch)
            return result
        except RuntimeError as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e)
            breakpoint()
            raise e
        
    def log_result(self, output: dict, unit: str = "step"):
        output = {f"{unit}/{k}": float(v) for k, v in output.items()}
        self.log_dict(dictionary=output,
                      on_step=unit == "step",
                      on_epoch=unit =="epoch")

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result: dict = self.forward(batch, mode="train")
        self.log(name="train_loss", value=result["loss"], prog_bar=True)
        
        if self.log_train_metrics:
            output: dict = self.train_metric(result["reg_pred"], result["reg_target"])
            self.log_result(output, unit="step")
            
            self.training_step_outputs.append(result)
        return result["loss"]

    def on_train_epoch_end(self):
        if self.log_train_metrics:
            output: dict = self.train_metric.compute()
            self.log_result(output, unit="epoch")
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result: dict = self.forward(batch, mode="valid")
        self.log(name="valid_loss", value=result["loss"], prog_bar=True)
        
        output: dict = self.valid_metric(result["reg_pred"], result["reg_target"])
        self.log_result(output, unit="step")
        
        self.validation_step_outputs.append(result)

    def on_validation_epoch_end(self):
        output: dict = self.valid_metric.compute()
        self.log_result(output, unit="epoch")
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        result: dict = self.forward(batch, mode="test")
        return result


def setup_trainer(config: omegaconf.DictConfig) -> pl.LightningModule:
    logger.info("Start Setting up")
    sage.utils.seed_everything(config.misc.seed)

    logger.info("Start instantiating dataloaders")
    dataloaders = sage.data.get_dataloaders(ds_cfg=config.dataset,
                                            dl_cfg=config.dataloader,
                                            modes=config.misc.modes)

    logger.info("Start intantiating Models & Optimizers")
    model = hydra.utils.instantiate(config.model)

    logger.info("Start instantiating Pytorch-Lightning Trainer")
    if config.module.get("load_from_checkpoint"):
        ckpt = config.module["load_from_checkpoint"]
        module = PLModule.load_from_checkpoint(ckpt,
                                               model=model,
                                               optimizer=config.optim,
                                               metrics=config.metrics,
                                               scheduler=config.scheduler,
                                               train_loader=dataloaders["train"],
                                               valid_loader=dataloaders["valid"],
                                               test_loader=dataloaders["test"],
                                               predict_loader=dataloaders["test"],
                                               **config.module)
    else:
        module = hydra.utils.instantiate(config.module,
                                         model=model,
                                         optimizer=config.optim,
                                         metrics=config.metrics,
                                         scheduler=config.scheduler,
                                         train_loader=dataloaders["train"],
                                         valid_loader=dataloaders["valid"],
                                         test_loader=dataloaders["test"],
                                         predict_loader=dataloaders["test"])
    return module, dataloaders


def tune(config: omegaconf.DictConfig) -> omegaconf.DictConfig:
    batch_size = config.dataloader.batch_size
    logging_interval = config.trainer.log_every_n_steps
    lr_frequency = config.scheduler.frequency
    
    # Tune logging interval
    config.trainer.log_every_n_steps = tune_logging_interval(logging_interval=logging_interval,
                                                             batch_size=batch_size)
    return config


def train(config: omegaconf.DictConfig) -> None:
    config: omegaconf.DictConfig = tune(config)
    logger = hydra.utils.instantiate(config.logger)
    module, dataloaders = setup_trainer(config)

    # Logger Setup
    logger.watch(module)
    config_update: bool = "version" in config.logger or config.trainer.devices > 1
    if config_update:
        # Skip config update when using resume checkpoint
        pass
    else:
        # Hard-code config uploading
        wandb.config.update(
            omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )

    # Callbacks
    callbacks: dict = hydra.utils.instantiate(config.callbacks)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer,
                                                  logger=logger,
                                                  callbacks=list(callbacks.values()))
    trainer.fit(model=module,
                train_dataloaders=dataloaders["train"],
                val_dataloaders=dataloaders["valid"])
    if dataloaders["test"]:
        prediction = trainer.test(ckpt_path="best", dataloaders=dataloaders["test"])
        finalize_inference(prediction=prediction,
                           name=config.logger.name)
    if config_update:
        wandb.config.update(omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    

def inference(config: omegaconf.DictConfig,
              root_dir: Path = None) -> None:
    if root_dir is None:
        root_dir = Path(config.callbacks.checkpoint.dirpath)
        
    module, dataloaders = setup_trainer(config)
    module.setup(stage=None)
    brain = module.log_brain(return_path=True, augment=False)
    subprocess.run(["mv", brain, f"{root_dir}/sample.png"])

    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    logger.info("Start prediction")
    prediction = trainer.predict(model=module, dataloaders=dataloaders["test"])

    task = config.module._target_
    # Infer Metrics
    if task == "sage.trainer.PLModule":
        finalize_inference(prediction=prediction,
                           name=config.logger.name,
                           root_dir=root_dir)

    elif task == "sage.xai.trainer.XPLModule":
        attr: np.ndarray = module.attr
        top_attr: np.ndarray = module.top_attr
        
        top_k: float = module.top_k_percentile
        postfix = module.xai_method + f"k{top_k:.2f}"
        
        root_dir = root_dir / postfix
        os.makedirs(name=root_dir, exist_ok=True)
        logger.info("Start saving here %s", root_dir)
        
        # Save attrs
        np.save(file=root_dir / "attrs.npy", arr=attr)
        np.save(file=root_dir / "top_attr.npy", arr=top_attr)
        
        # Save plots
        plot_glass_brain(arr=attr, save=root_dir / "attr_glass.png")
        plot_overlay(arr=attr, scale_factor=1, save=root_dir / "attr_anat.png")
        
        plot_glass_brain(arr=top_attr, save=root_dir / "top_glass.png")
        plot_overlay(arr=top_attr, scale_factor=1, save=root_dir / "top_anat.png")
