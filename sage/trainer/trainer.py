import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
import wandb
from monai import transforms as mt

import sage


logger = sage.utils.get_logger(name=__name__)


class PLModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: omegaconf.DictConfig,
                 metrics: dict,
                 scheduler: omegaconf.DictConfig = None,
                 load_from_checkpoint: str = None,
                 separate_lr: dict = None):
        super().__init__()
        self.model = model

        # Dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
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
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

        if load_from_checkpoint:
            logger.info("Load checkpoint from %s", load_from_checkpoint)
            self.load_from_checkpoint(load_from_checkpoint)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        self.augmentor = mt.Compose([
            mt.Resize(spatial_size=(96, 96, 96)),
            mt.ScaleIntensity(),
        ])

    def train_dataloader(self):
        return self.train_loader

    def valid_dataloader(self):
        return self.valid_loader

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

    def forward(self, batch):
        try:
            """ model should return dict of
            {
                loss:
                cls_pred:
                cls_target:
            }
            """
            # 5d-tensor
            batch["brain"] = self.augmentor(batch["brain"]).unsqueeze(dim=1)
            result: dict = self.model(**batch)
            return result
        except RuntimeError as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e)
            breakpoint()
            raise e

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result: dict = self.forward(batch)
        output: dict = self.train_metrics(result["reg_pred", "reg_target"])
        self.log_dict(output)
        self.training_step_outputs.append(result)
        return result["loss"]

    def on_train_epoch_end(self):
        output: dict = self.train_metric.compute()
        self.log_dict(output)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result: dict = self.forward(batch)
        output: dict = self.valid_metric(result["reg_pred", "reg_target"])
        self.log_dict(output)
        self.validation_step_outputs.append(result)

    def on_validation_epoch_end(self):
        output: dict = self.valid_metric.compute()
        self.log_dict(output)
        self.validation_step_outputs.clear()


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
    module = hydra.utils.instantiate(config.module,
                                      model=model,
                                      optimizer=config.optim,
                                      metrics=config.metrics,
                                      scheduler=config.scheduler,
                                      train_loader=dataloaders["train"],
                                      valid_loader=dataloaders["valid"])
    return module, dataloaders


def train(config: omegaconf.DictConfig) -> None:
    module, dataloaders = setup_trainer(config)

    # Logger Setup
    logger = hydra.utils.instantiate(config.logger)
    logger.watch(module)
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