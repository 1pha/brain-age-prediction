import random
import subprocess
from typing import Any, List, Dict
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
import wandb
import monai.transforms as mt
from monai.data.meta_tensor import MetaTensor

import sage
import sage.xai.nilearn_plots as nilp_
import sage.constants as C
from . import utils


logger = sage.utils.get_logger(name=__name__)


class PLModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: omegaconf.DictConfig,
                 metrics: dict,
                 test_loader: torch.utils.data.DataLoader = None,
                 predict_loader: torch.utils.data.DataLoader = None,
                 log_train_metrics: bool = False,
                 manual_lr: bool = False,
                 augmentation: omegaconf.DictConfig = None,
                 scheduler: omegaconf.DictConfig = None,
                 load_model_ckpt: str = None,
                 load_from_checkpoint: str = None, # unused params but requires for instantiation
                 separate_lr: dict = None,
                 task: str = None,
                 save_dir: str = None):
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
        self.train_metric = metrics.clone(prefix="train_")
        self.valid_metric = metrics.clone(prefix="valid_")

        if load_model_ckpt:
            logger.info("Load checkpoint from %s", load_model_ckpt)
            self.model.load_from_checkpoint(load_model_ckpt)

        self.log_train_metrics = log_train_metrics
        self.log_lr = manual_lr
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.prediction_step_outputs = []

        self.init_transforms(augmentation=augmentation)
        self.save_dir = Path(save_dir)
        self.task = task

    def setup(self, stage):
        self.log_brain(return_path=False)

    def init_transforms(self, augmentation: omegaconf.DictConfig):
        self.train_transforms = mt.Compose([
            mt.Lambda(func=utils.brain2augment),
            mt.Resize(spatial_size=augmentation.get("spatial_size", C.SPATIAL_SIZE)),
            mt.ScaleIntensity(channel_wise=True),
            mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
            mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
            mt.RandAxisFlip(prob=0.5),
            mt.RandZoom(prob=0.4, min_zoom=0.9, max_zoom=1.4, mode="trilinear"),  
            mt.Lambda(func=utils.augment2brain),
        ])
        self.valid_transforms = mt.Compose([
            mt.Lambda(func=utils.brain2augment),
            mt.Resize(spatial_size=augmentation.get("spatial_size", C.SPATIAL_SIZE)),
            mt.ScaleIntensity(channel_wise=True),
            mt.Lambda(func=utils.augment2brain),
        ])

    def log_brain(self, return_path: bool = False, augment: bool = True):
        """ Logs sample brain to check how augmentation is applied. """
        ds = self.train_dataloader.dataset
        idx: int = random.randint(a=0, b=len(ds))
        brain: torch.Tensor = ds[idx]["brain"]
        brain = self.train_transforms(brain) if augment else self.valid_transforms(brain)

        tmp = "tmp.png"
        nilp_.plot_brain(brain, save=tmp)
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
        if "total_steps" in scheduler["scheduler"]:
            struct["total_steps"] = num_training_steps
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

    def get_tensor(self, tensor: torch.Tensor | MetaTensor) -> torch.Tensor:
        """ monai.transforms return a internal tensor class called "Metatensor"
        This datatype may throw an error for some functions from time to time. (e.g torch.compile)
        Use this method to resolve the issue. """
        if isinstance(tensor, MetaTensor):
            # MetaTensor is not suitable for torch.compile
            tensor = tensor.as_tensor()
        return tensor

    def forward(self, batch, mode: str = "train"):
        try:
            """ model should return dict of
            {
                loss:
                cls_pred:
                cls_target:
            }
            """
            aug = getattr(self, f"{'train' if mode == 'train' else 'valid'}_transforms")
            batch["brain"] = aug(batch["brain"]).as_tensor()
            batch["age"] = batch["age"]
            result: dict = self.model(**batch)
            return result
        except RuntimeError as e:
            # For CUDA Device-side asserted error
            logger.warn("Given batch %s", batch)
            logger.exception(e)
            breakpoint()
            raise e
        
    def move_device(self,
                    result: Dict[str, torch.Tensor],
                    exclude_keys: List[str] = ["loss"]) -> Dict[str, torch.Tensor]:
        """ ModelBase returns a dictionary with model output,
        including prediction, logits, ground truth.
        However, these values shuold be shifted from cuda to cpu, due to VRAM consuming issue.
        This was moved back to cpu in ModelBase,
        but most torchmetrics functions require logits and ground truth to be on cuda
        (more precisely, to be on the same device with torchmetrics)
        Therefore, we keep other keys in cuda and move back to cpu after metric calculation.
        """
        for key in result:
            if key not in exclude_keys:
                result[key] = result[key].to("cpu")
        return result

    def log_confusion_matrix(self, result: dict):
        probs = result["pred"]
        if probs.ndim == 1:
            # Binary classification
            probs = torch.nn.functional.sigmoid(probs)
            probs = torch.stack([1-probs, probs]).T.cpu().numpy()
        else:
            probs = probs.cpu().detach()
        labels = result["target"].cpu().numpy()
        cf = wandb.plot.confusion_matrix(probs=probs, y_true=labels)
        roc = wandb.plot.roc_curve(y_true=labels, y_probas=probs)
        pr = wandb.plot.pr_curve(y_true=labels, y_probas=probs)
        self.logger.experiment.log({"confusion_matrix": cf, "roc_curve": roc, "pr_curve": pr})

    def log_result(self, output: dict, unit: str = "step", prog_bar: bool = False):
        output = {f"{unit}/{k}": float(v) for k, v in output.items()}
        self.log_dict(dictionary=output, 
                      on_step=unit == "step",
                      on_epoch=unit == "epoch",
                      prog_bar=prog_bar)

    def training_step(self, batch, batch_idx):
        result: dict = self.forward(batch, mode="train")
        self.log(name="train_loss", value=result["loss"], prog_bar=True)

        if self.log_train_metrics:
            output: dict = self.train_metric(result["pred"], result["target"])
            self.log_result(output=output, unit="step", prog_bar=False)
            self.training_step_outputs.append(self.move_device(result=result))

        if self.log_lr:
            # Since `ModelCheckpoint` cannot track learning rate automatically,
            # We log learning rate explicitly and monitor this
            lr = self.lr_schedulers().get_lr()[0]
            self.log(name="_lr", value=lr, on_step=True)
        return result["loss"]

    def on_train_epoch_end(self):
        if self.log_train_metrics:
            output: dict = self.train_metric.compute()
            self.log_result(output, unit="epoch")
            self.training_step_outputs.clear()
            self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        result: dict = self.forward(batch, mode="valid")
        self.log(name="valid_loss", value=result["loss"], prog_bar=True)
        output: dict = self.valid_metric(result["pred"], result["target"])
        self.log_result(output, unit="step", prog_bar=False)
        self.validation_step_outputs.append(self.move_device(result=result))

    def on_validation_epoch_end(self):
        output: dict = self.valid_metric.compute()
        self.log_result(output, unit="epoch", prog_bar=True)

        result = utils._sort_outputs(outputs=self.validation_step_outputs)
        if utils.check_classification(result=result):
            self.log_confusion_matrix(result=result)
        self.validation_step_outputs.clear()
        self.valid_metric.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        result: dict = self.forward(batch, mode="test")
        self.prediction_step_outputs.append(result)
        return result

    def on_predict_end(self):
        result = utils._sort_outputs(outputs=self.prediction_step_outputs)
        if utils.check_classification(result=result):
            self.log_confusion_matrix(result=result)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
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
        # This is used for training resume.
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
    config.trainer.log_every_n_steps = utils.tune(batch_size=batch_size,
                                                  logging_interval=logging_interval)
    config.scheduler.frequency = utils.tune(batch_size=batch_size, lr_frequency=lr_frequency,
                                            accumulate_grad_batches=config.trainer.accumulate_grad_batches)
    if "manual_ckpt" in config.callbacks:
        config.callbacks.manual_ckpt.multiplier = utils.tune(batch_size=batch_size,
                                                             multiplier=config.callbacks.manual_ckpt.multiplier)
    
    return config


def train(config: omegaconf.DictConfig) -> None:
    config: omegaconf.DictConfig = tune(config)
    _logger = hydra.utils.instantiate(config.logger)
    module, dataloaders = setup_trainer(config)

    # Logger Setup
    _logger.watch(module)
    config_update: bool = "version" in config.logger or config.trainer.devices > 1
    if config_update:
        # Skip config update when using resume checkpoint
        pass
    else:
        # Hard-code config uploading
        resolve = not "sweep" in config.hydra
        wandb.config.update(
            omegaconf.OmegaConf.to_container(config, resolve=resolve, throw_on_missing=resolve)
        )

    # Callbacks
    callbacks: dict = hydra.utils.instantiate(config.callbacks)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=_logger,
                                                  callbacks=list(callbacks.values()))
    trainer.fit(model=module,
                train_dataloaders=dataloaders["train"],
                val_dataloaders=dataloaders["valid"],
                ckpt_path=config.misc.get("ckpt_path", None))
    
    if dataloaders["test"]:
        logger.info("Test dataset given. Start inference on %s", len(dataloaders["test"].dataset))
        prediction = trainer.predict(ckpt_path="best", dataloaders=dataloaders["test"])
        metric = utils.finalize_inference(prediction=prediction, name=config.logger.name,
                                          root_dir=Path(config.callbacks.checkpoint.dirpath))
    elif dataloaders["valid"]:
        logger.info("Test dataset not found. Start inference on validation dataset %s",
                    len(dataloaders["valid"].dataset))
        prediction = trainer.predict(ckpt_path="best", dataloaders=dataloaders["valid"])
        metric = utils.finalize_inference(prediction=prediction, name=config.logger.name,
                                          root_dir=Path(config.callbacks.checkpoint.dirpath))
        
    if config_update:
        wandb.config.update(omegaconf.OmegaConf.to_container(config, resolve=True,
                                                             throw_on_missing=True))
    return metric


def inference(config: omegaconf.DictConfig,
              root_dir: Path = None) -> None:
    if root_dir is None:
        root_dir = Path(config.callbacks.checkpoint.dirpath)
        
    module, dataloaders = setup_trainer(config)
    module.setup(stage=None)
    brain = module.log_brain(return_path=True, augment=False)

    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    logger.info("Start prediction")
    prediction = trainer.predict(model=module, dataloaders=dataloaders["test"])

    task = config.module._target_
    if task == "sage.trainer.PLModule":
        # Infer Metrics
        utils.finalize_inference(prediction=prediction,
                                 name=config.logger.name,
                                 root_dir=root_dir)

    elif task == "sage.xai.trainer.XPLModule":
        # Infer Saliency maps
        postfix = module.xai_method + f"k{module.top_k_percentile:.2f}"

        bsz = config.dataloader.batch_size
        if bsz > 1:
            postfix = f"{postfix}-bsz{bsz}"
        root_dir = root_dir / postfix
        subprocess.run(["mv", brain, f"{root_dir}/sample.png"])
        module.save_result(root_dir=root_dir)
