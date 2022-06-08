import json
import pickle
import math
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

Arguments = NewType("Arguments", Any)
Logger = NewType("Logger", Any)

import torch

if (torch.__version__.startswith("1.13") and torch.backends.mps.is_available()) or (
    not torch.cuda.is_available()
):
    # Ignore warnings when accelerated through M1
    import warnings

    warnings.filterwarnings("ignore")

import wandb
from tqdm import tqdm

from .metrics import get_metric_fn
from .optimizer import construct_optimizer
from .scheduler import construct_scheduler, step_on_batch_list
from .utils import save_checkpoint, walltime


class MRITrainer:

    """ """

    __version__ = 0.4
    __date__ = "May 17. 2022"

    def __init__(
        self,
        model: torch.nn.Module,
        data_args: Arguments,
        training_args: Arguments,
        misc_args: Arguments,
        logger: Logger,
        training_data: torch.utils.data.DataLoader = None,
        validation_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
    ) -> None:

        arguments = locals()
        arguments.pop("self")
        self._allocate(**arguments)
        wandb.config.update(
            self.get_configs(
                training_args=training_args,
                data_args=data_args,
                misc_args=misc_args,
            )
        )
        if training_args.do_train:
            if training_args.scheduler in step_on_batch_list:
                training_args.total_steps = int(
                    math.ceil(len(training_data.dataset) / data_args.batch_size)
                ) * (training_args.epochs)
                training_args.warmup_steps = int(
                    training_args.total_steps * training_args.warmup_ratio
                )

        # Construct Optimizer
        self.optimizer = construct_optimizer(self.model, training_args, logger)
        self.scheduler = construct_scheduler(self.optimizer, training_args, logger)
        self._set_fp16(training_args.fp16)

        # Construct Loss functions & Metrics
        self.loss_fn = get_metric_fn(training_args.loss_fn)
        self.metrics_fn = get_metric_fn(training_args.metrics_fn)

        # Force devices to CPU is needed (for debugs)
        self._find_device()
        if misc_args.force_cpu:
            self._force_cpu()

        # Buckets to save results or failures.
        self.loading_failures = defaultdict(list)
        self.results = defaultdict(list)

    def _set_fp16(self, fp16: bool) -> None:

        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.logger.info(f"FP16: {str(fp16)}")

    def _force_cpu(self) -> None:

        """
        Forcibly place models to CPU for certain usage.
        """
        self.model.to("cpu")
        self.device = "cpu"
        self.logger.info("Device forced to use CPU.")

    def _find_device(self) -> None:

        """
        Find where the model locates - CPU or GPU and if multiple, which GPUs
        """
        self.device = next(self.model.parameters()).device
        self.logger.info(f"Use {self.device} as a device.")

    def _allocate(self, **kwargs) -> None:

        for arg, val in kwargs.items():
            if arg != "self":
                setattr(self, arg, val)

    def _reallocate(self, **kwargs) -> Dict:

        """
        Reallocate function arguments to attributes inside the instance if None given.
        None will be returned to each arguments if they are not attributes inside the instance.
        """

        reallocated_dict = {}
        for arg, val in kwargs.items():
            if arg == "self":
                pass

            elif val is None:
                try:
                    # Find attribute and reallocate
                    reallocated_dict[arg] = getattr(self, arg)
                except AttributeError:
                    # If attribute not found, allocate None
                    reallocated_dict[arg] = None

            else:
                # If proper value was already given, don't reallocate.
                reallocated_dict[arg] = val
        return reallocated_dict

    def run(
        self,
        model: torch.nn.Module = None,
        data_args: Arguments = None,
        training_args: Arguments = None,
        misc_args: Arguments = None,
        training_data: torch.utils.data.DataLoader = None,
        validation_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
    ) -> None:
        try:
            arguments = locals()
            arguments.pop("self")
            kwargs = self._reallocate(**arguments)
            self._run(**kwargs)

        except Exception as e:
            self.logger.exception(e)
            self.logger.warn(f"Error found. Return arguments and end training.")
            self.save_configs(**kwargs)

    def _run(
        self,
        model: torch.nn.Module,
        data_args: Arguments,
        training_args: Arguments,
        misc_args: Arguments,
        training_data: torch.utils.data.DataLoader = None,
        validation_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
    ) -> None:

        self.logger.info("Start Training")

        best_mae = float("inf")

        epochs = training_args.epochs
        stop_count, long_term_patience, elapsed_epoch_saved = 0, 0, 0
        _loss_fn, _metrics_fn = (
            training_args.loss_fn.upper(),
            training_args.metrics_fn.upper(),
        )
        wandb.watch(model)
        result = defaultdict(dict)
        for e in range(epochs):

            self.logger.info(f"EPOCH {e}/{epochs} | BEST MAE {best_mae:.3f}")

            if training_data is not None and training_args.do_train:
                (train_loss, train_metric), time_elapsed = self.train(
                    model, training_data
                )
                train_result = {"train_loss": train_loss, "train_metric": train_metric}
                self.logger.info(
                    f"Train:: {time_elapsed:>5.1f} sec | {_loss_fn} {train_loss:>6.3f} | {_metrics_fn} {train_metric:>6.3f}"
                )
                wandb.log(
                    train_result,
                    commit=False,
                )
                result[e].update(train_result)

            if validation_data is not None and training_args.do_eval:
                (valid_loss, valid_metric), time_elapsed = self.valid(
                    model, validation_data
                )
                valid_result = {"valid_loss": valid_loss, "valid_metric": valid_metric}
                self.logger.info(
                    f"Valid:: {time_elapsed:>5.1f} sec | {_loss_fn} {valid_loss:>6.3f} | {_metrics_fn} {valid_metric:>6.3f}"
                )
                wandb.log(valid_result, commit=False)
                result[e].update(valid_result)

            if test_data is not None and training_args.do_inference:
                (test_loss, test_metric), time_elapsed = self.valid(model, test_data)
                test_result = {"test_loss": test_loss, "test_metric": test_metric}
                self.logger.info(
                    f"Test :: {time_elapsed:>5.1f} sec | {_loss_fn} {test_loss:>6.3f} | {_metrics_fn} {test_metric:>6.3f}"
                )
                wandb.log(test_result, commit=False)
                result[e].update(test_result)

            scheduler_name = self.training_args.scheduler
            if scheduler_name == "":
                pass

            elif scheduler_name == "plateau":
                self.scheduler.step(valid_loss)

            elif scheduler_name not in step_on_batch_list:
                self.scheduler.step()

            wandb.log({"epoch": e, "lr": self.optimizer.param_groups[0]["lr"]})

            # Check performance improvement
            model_name = f"ep{str(e).zfill(3)}.pt"
            current_mae = valid_result["valid_metric"]
            # Yes Improvement
            if current_mae < best_mae:

                stop_count = 0
                best_mae = current_mae
                save_checkpoint(
                    model=self.model,
                    model_name=model_name,
                    output_dir=misc_args.output_dir,
                    logger=self.logger,
                )
                best_epoch = e

            # No Improvement
            else:
                # COUNT PATIENCE
                stop_count += 1

                # COUNT
                if stop_count >= training_args.early_patience:

                    # END OF PATIENCE
                    if best_mae < training_args.mae_threshold:
                        save_checkpoint(
                            model=self.model,
                            model_name=model_name,
                            output_dir=misc_args.output_dir,
                            logger=self.logger,
                        )
                        self.logger.info(
                            f"Early stopped at {stop_count} / {training_args.early_patience} at EPOCH {e}"
                        )
                        break

                    # IF THE FINAL RESULT IS NOT SATISFACTORY
                    # WAIT FOR ANOTHER LONG-TERM PATIENCE
                    else:
                        long_term_patience += 1

            # EVEN AFTER LONG-TERM PATIENCE - KILL
            if long_term_patience >= 3:
                save_checkpoint(
                    model=self.model,
                    model_name=model_name,
                    output_dir=misc_args.output_dir,
                    logger=self.logger,
                )
                self.logger.info(
                    f"Waited for 3 times and no better result {long_term_patience} / {3} at EPOCH {e}"
                )
                break

            # SAVING CRITERIA
            # WILL BE SAVED TWICE IF THERE WAS A PERFORMANCE IMPROVEMENT
            # BUT WON'T MATTER SINCE "save_checkpoint" SAVES ONLY WHEN MODEL IS NOT THERE
            if elapsed_epoch_saved == training_args.checkpoint_period:
                elapsed_epoch_saved = 1
                save_checkpoint(
                    model=self.model,
                    model_name=model_name,
                    output_dir=misc_args.output_dir,
                    logger=self.logger,
                )

            else:
                elapsed_epoch_saved += 1

            wandb.config.update(
                {"best_valid_epoch": best_epoch, "best_valid_metric": best_mae}, allow_val_change=True
            )

        # Finish training
        wandb.finish()
        training_args.best_valid_metric = best_mae
        training_args.best_valid_epoch = best_epoch
        self.save_configs(
            data_args=data_args, training_args=training_args, misc_args=misc_args
        )
        self.save_result(result)

    @walltime
    def train(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer = None,
        loss: str = None,
    ) -> Tuple[List, List]:

        """
        * Note *
            Unlearning phase removed in Apr 14. 2022
        """

        losses, preds = [], []
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            for i, (x, y) in enumerate(tqdm(dataloader)):

                self.logger.debug(f"train phase, {i}th batch.")

                try:
                    x, y = map(lambda obj: obj.to(self.device), (x, y))
                except FileNotFoundError as e:
                    self.logger.exception(e)
                    time.sleep(20)
                    self.loading_failures["train"].append((e, i))
                    continue

                with torch.cuda.amp.autocast(self.fp16):
                    loss, pred = self.step(
                        x, y, model=model, update=True, optimizer=optimizer
                    )
                    losses.append(float(loss.cpu().detach()))
                    preds.append(pred)

                torch.cuda.empty_cache()

        loss, metric = self.organize_result(losses, preds, dataloader)
        return loss, metric

    @walltime
    def valid(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[List, List]:

        losses, preds = [], []
        model.eval()
        for i, (x, y) in enumerate(tqdm(dataloader)):

            self.logger.debug(f"validation phase, {i}th batch.")

            try:
                x, y = map(lambda obj: obj.to(self.device), (x, y))
            except FileNotFoundError as e:
                self.logger.exception(e)
                time.sleep(20)
                self.loading_failures["valid"].append((e, i))
                continue

            with torch.no_grad():
                loss, pred = self.step(x, y, model=model, update=False)
                losses.append(float(loss.cpu().detach()))
                preds.append(pred)

            torch.cuda.empty_cache()

        loss, metric = self.organize_result(losses, preds, dataloader)
        return loss, metric

    def step(
        self,
        x: torch.FloatTensor,
        y: torch.IntTensor,
        model: torch.nn.Module,
        update: bool = True,
        optimizer: torch.optim.Optimizer = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pred = model(x)
        loss = self.calculate_loss(pred.squeeze(), y)

        if update:
            self.update_loss(loss, optimizer)

        return loss, pred

    def calculate_loss(self, y_pred, y_true, loss_fn: Callable = None) -> torch.Tensor:

        if loss_fn is None:
            loss_fn = self.loss_fn

        return loss_fn(y_pred, y_true)

    def update_loss(self, loss, optimizer=None) -> None:

        if optimizer is None:
            optimizer = self.optimizer

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()
        if self.training_args.scheduler in step_on_batch_list:
            self.scheduler.step()

    def load_checkpoint(self, checkpoint: str) -> None:

        self.model.load_state_dict(torch.load(checkpoint))
        self.logger.info(f"Successfully loaded model.")
        return self.model

    def organize_result(
        self,
        losses: list,
        preds: list,
        dataloader: torch.utils.data.DataLoader,
        metrics_fn: str = None,
    ) -> Tuple[float, float]:

        loss = sum(losses) / len(losses)
        preds = torch.cat(preds).cpu().detach().squeeze()
        gt = torch.tensor(self.get_ground_truth(dataloader)).squeeze()
        if self.data_args.return_age_range == "shrink":
            preds *= 100

        if metrics_fn is not None and isinstance(metrics_fn, str):
            metrics_fn = get_metric_fn(metrics)
        else:
            metrics_fn = self.metrics_fn
        metrics = float(metrics_fn(preds, gt))
        return loss, metrics

    def get_ground_truth(self, dataloader: torch.utils.data.DataLoader) -> list:
        return dataloader.dataset.data_ages

    def get_configs(self, **kwargs) -> None:

        configs = {
            a.get_name(): a.to_dict() for k, a in kwargs.items() if k.endswith("_args")
        }
        return configs

    def save_configs(self, **kwargs) -> None:

        output_dir = kwargs["misc_args"].output_dir
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, "config.json")
        configs = self.get_configs(**kwargs)
        with open(fname, "w") as f:
            json.dump(configs, f, indent=4, sort_keys=True)
        self.logger.info(f"Successfully saved configurations to {fname}")

    def save_result(self, result, misc_args=None) -> None:

        misc_args = misc_args or self.misc_args
        output_dir = misc_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, "result.pkl")
        with open(fname, "wb") as f:
            pickle.dump(result, f)
