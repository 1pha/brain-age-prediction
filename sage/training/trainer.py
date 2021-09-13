import time
import yaml
import wandb

from easydict import EasyDict as edict

import torch

from .metrics import get_metric
from .optimizer import get_optimizer
from ..models.model_util import load_models, multimodel_save_checkpoint
from ..data.dataloader import get_dataloader
import sys

sys.path.append("../../")
from utils.misc import seed_everything, logging_time, get_today
from utils.average_meter import AverageMeter


class MRITrainer:

    """
    This is an integrated trainer class that combines -
        1. Training the naive age
        2. Training with Unlearning strategy
        3. Extensible for future works
    """

    __version__ = 0.2
    __date__ = "Aug 19. 2021"

    def __init__(self, cfg, result_dir_suffix=None):

        self.cfg = cfg
        self.phase_dict = self.get_phase_dict()
        self.cfg.epochs = sum(len(_range) for _range in self.phase_dict.values())
        self.cfg.trainer_version = MRITrainer.__version__
        self.setup(cfg=cfg, result_dir_suffix=result_dir_suffix)
        self.failure_cases = {
            "train": [],
            "valid": [],
        }

    def setup(self, cfg=None, result_dir_suffix=None):

        """
        SETUP
            Set every options through configuration file.
            Pipelines are as follows
            1. Fixate seed
            2. Load Models & Optimizers
            3. Load Dataloader
            4. Load AMP Scaler (optional)
                - May not be able to use L2-Loss
        """

        cfg = self.cfg if cfg is None else cfg

        # 1. Fixate Seed
        seed_everything(seed=cfg.seed)

        # 2. Count Database
        cfg.domainer.num_dbs = (
            4 if cfg.unused_src[0] is None else 4 - len(cfg.unused_src)
        )

        # 2. SETUP MODEL & OPTIMIZER
        (encoder, regressor, domainer), cfg.device = load_models(
            cfg.encoder, cfg.regressor, cfg.domainer
        )

        if cfg.force_cpu:
            cfg.device = torch.device("cpu")
            for model in (encoder, regressor, domainer):
                model.to(cfg.device)

        self.models = {
            "encoder": encoder,
            "regressor": regressor,
            "domainer": domainer,
        }
        self.optimizers = {
            "regression": get_optimizer(
                [encoder, regressor], cfg.reg_opt
            ),  # AGE PREDICTOR
            "domain": get_optimizer([domainer], cfg.clf_opt),  # DOMAIN PREDICTOR
            "confusion": get_optimizer([encoder], cfg.unl_opt),  # CONFUSION
        }

        # 3. Load Dataloader
        self.train_dataloader = get_dataloader(cfg, sampling="train")
        self.gt_age_train = torch.tensor(
            self.train_dataloader.dataset.data_ages, dtype=torch.float
        )
        self.gt_src_train = torch.tensor(self.train_dataloader.dataset.data_src)

        self.valid_dataloader = get_dataloader(cfg, sampling="valid")
        self.gt_age_valid = torch.tensor(
            self.valid_dataloader.dataset.data_ages, dtype=torch.float
        )
        self.gt_src_valid = torch.tensor(self.valid_dataloader.dataset.data_src)

        self.test_dataloader = get_dataloader(cfg, sampling="test")
        self.gt_age_test = torch.tensor(
            self.test_dataloader.dataset.data_ages, dtype=torch.float
        )
        self.gt_src_test = torch.tensor(self.test_dataloader.dataset.data_src)

        print(
            f"TOTAL TRAIN {len(self.train_dataloader.dataset)} | VALID {len(self.valid_dataloader.dataset)} | TEST {len(self.test_dataloader.dataset)}"
        )

        # 4. AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        print(f"MIXED PRECISION:: {cfg.use_amp}")

        # 5. SAVE Directory
        self.save_dir = f"{cfg.RESULT_PATH}/{get_today()}_{cfg.encoder.name}"
        self.save_dir += (
            f"_{result_dir_suffix}" if result_dir_suffix is not None else ""
        )

    def run(self, cfg=None, checkpoint=None):

        cfg = self.cfg if cfg is None else cfg

        if checkpoint is not None:

            """
            checkpoint: dict = {
                'resume_epoch': start, # NECESSARY
                'models': {
                    'encoder': ~.pt,
                    ..
                }
            }
            """

            self.load_checkpoint(checkpoint["models"])
            offset = (
                checkpoint["resume_epoch"] if "resume_epoch" in checkpoint.keys() else 0
            )

        else:
            offset = 0

        best_mae = float("inf")
        for phase, (epochs, actions) in enumerate(zip(*self.cfg.phase_config.values())):

            stop_count, long_term_patience, elapsed_epoch_saved = 0, 0, 0
            for e in range(epochs):

                print(
                    f'{"-" * 5} Epoch {e + 1 + offset} / {cfg.epochs} (phase: {phase}) BEST MAE {best_mae:.3f} {"-" * 5}'
                )

                train_result = self.train(actions)
                valid_result = self.valid(actions)
                results = edict(  # AGGREGATE RESULTS
                    **train_result,
                    **valid_result,
                )
                self.prompt(results)
                wandb.log({**results})

                model_name = f"ep{str(e).zfill(3)}_mae{results.valid_mae:.2f}.pt"
                # PERFORMANCE IMPROVEMENT
                if results.valid_mae < best_mae:

                    stop_count = 0
                    best_mae = results.valid_mae
                    multimodel_save_checkpoint(
                        states=self.models,
                        model_dir=self.save_dir,
                        model_name=model_name,
                    )
                    best_epoch = e

                # NO PERFORMANCE IMPROVEMENT
                else:
                    # COUNT PATIENCE
                    stop_count += 1

                    # COUNT
                    if stop_count >= cfg.early_patience:

                        # END OF PATIENCE
                        if best_mae < cfg.mae_threshold:
                            multimodel_save_checkpoint(
                                states=self.models,
                                model_dir=self.save_dir,
                                model_name=model_name,
                            )
                            print(
                                f"Early stopped at {stop_count} / {cfg.early_patience} at EPOCH {e + offset}"
                            )
                            break

                        # IF THE FINAL RESULT IS NOT SATISFACTORY
                        # WAIT FOR ANOTHER LONG-TERM PATIENCE
                        else:
                            long_term_patience += 1

                # EVEN AFTER LONG-TERM PATIENCE - KILL
                if long_term_patience >= 3:
                    multimodel_save_checkpoint(
                        states=self.models,
                        model_dir=self.save_dir,
                        model_name=model_name,
                    )
                    print(
                        f"Waited for 3 times and no better result {long_term_patience} / {3} at EPOCH {e + offset}"
                    )
                    break

                # SAVING CRITERIA
                # WILL BE SAVED TWICE IF THERE WAS A PERFORMANCE IMPROVEMENT
                # BUT WON'T MATTER SINCE "save_checkpoint" SAVES ONLY WHEN MODEL IS NOT THERE
                if elapsed_epoch_saved == cfg.checkpoint_period:
                    elapsed_epoch_saved = 1
                    multimodel_save_checkpoint(
                        states=self.models,
                        model_dir=self.save_dir,
                        model_name=model_name,
                    )

                else:
                    elapsed_epoch_saved += 1

            offset += e

        # CHECK WITH TEST_DATALOADER
        self.load_checkpoint(
            {
                "encoder": f"{self.save_dir}/encoder/ep{str(best_epoch).zfill(3)}_mae{best_mae:.2f}.pt",
                "domainer": f"{self.save_dir}/domainer/ep{str(best_epoch).zfill(3)}_mae{best_mae:.2f}.pt",
                "regressor": f"{self.save_dir}/regressor/ep{str(best_epoch).zfill(3)}_mae{best_mae:.2f}.pt",
            }
        )
        test_results = self.valid(actions, test=True)
        cfg.test_mae = test_results["test_mae"]
        cfg.best_mae = best_mae
        wandb.config.update(cfg)
        wandb.finish()
        with open(f"{self.save_dir}/config.yml", "w") as y:
            yaml.dump(cfg, y)

    @logging_time
    def train(self, actions):

        """
        3 Phases of Training
            1. Pre-train Encoder (around 100+ epochs with Full 4 Database)
                - update encoder/age_regressor only

            2. Train Domain Predictor (around 10- epochs will do fine)
                - update domain_predictor only

            3. Do Unlearning (with confusion loss)
                - update encoder only
        """

        losses, ages, domains = (
            [AverageMeter(tag=action, train=True) for action in actions],
            [],
            [],
        )
        with torch.autograd.set_detect_anomaly(True):
            for i, (x, y, d) in enumerate(self.train_dataloader):

                for _, model in self.models.items():
                    model.train()

                if self.cfg.debug and self.cfg.run_debug.verbose_all:
                    print(f"{i}th Batch.")

                try:
                    x, y, d = map(lambda x: x.to(self.cfg.device), (x, y, d))

                except FileNotFoundError as e:
                    print(e)
                    time.sleep(20)
                    self.failure_cases["train"].append(i)
                    continue

                with torch.cuda.amp.autocast(self.cfg.use_amp):
                    for j, action in enumerate(actions):
                        loss, _ = {
                            "reg": self.update_age_reg,
                            "clf": self.update_domain_clf,
                            "unl": self.update_domain_conf,
                        }[action](x, y, d, update=True)
                        losses[j].append(float(loss.cpu().detach()))

                with torch.no_grad():
                    for _, model in self.models.items():
                        model.eval()
                    _, age = self.update_age_reg(x, y, d, update=False)
                    _, domain = self.update_domain_clf(x, y, d, update=False)

                ages.extend(age.cpu().detach().tolist())
                domains.extend(domain.cpu().detach().tolist())
            torch.cuda.empty_cache()

        results = {
            **self.agg_loss(losses),
            **self.gather_result(ages, "age", sampling="train"),
            **self.gather_result(domains, "domain", sampling="train"),
        }

        return results

    @logging_time
    def valid(self, actions, test=False):

        for _, model in self.models.items():
            model.eval()

        losses, ages, domains = (
            [AverageMeter(tag=action, train=False) for action in actions],
            [],
            [],
        )
        dataloader = self.valid_dataloader if not test else self.test_dataloader
        with torch.no_grad():
            for i, (x, y, d) in enumerate(dataloader):

                if self.cfg.debug and self.cfg.run_debug.verbose_all:
                    print(f"{i}th Batch.")

                try:
                    x, y, d = map(lambda x: x.to(self.cfg.device), (x, y, d))

                except FileNotFoundError as e:
                    print(e)
                    time.sleep(20)
                    self.failure_cases["valid"].append(i)
                    continue

                for j, action in enumerate(actions):
                    loss, _ = {
                        "reg": self.update_age_reg,
                        "clf": self.update_domain_clf,
                        "unl": self.update_domain_conf,
                    }[action](x, y, d, update=False)
                    losses[j].append(float(loss.cpu().detach()))

                _, age = self.update_age_reg(x, y, d, update=False)
                _, domain = self.update_domain_clf(x, y, d, update=False)

                ages.extend(age.cpu().detach().tolist())
                domains.extend(domain.cpu().detach().tolist())
            torch.cuda.empty_cache()

        results = {
            **self.agg_loss(losses),
            **self.gather_result(ages, "age", sampling="valid" if not test else "test"),
            **self.gather_result(
                domains, "domain", sampling="valid" if not test else "test"
            ),
        }

        return results

    def update_age_reg(self, x, y, d, update=True):

        embed = self.models["encoder"](x)
        y_pred = self.models["regressor"](embed)
        loss = get_metric(y_pred.squeeze(), y, "rmse")

        if update:
            self.update_loss(loss, self.optimizers["regression"])

        return loss, y_pred

    def update_domain_clf(self, x, y, d, update=True):

        embed = self.models["encoder"](x)
        d_pred = self.models["domainer"](embed)
        loss = self.cfg.alpha * get_metric(d_pred, d, "ce")

        if update:
            self.update_loss(loss, self.optimizers["domain"])

        return loss, d_pred

    def update_domain_conf(self, x, y, d, update=True):

        embed = self.models["encoder"](x)
        d_pred = self.models["domainer"](embed)
        loss = self.cfg.beta * get_metric(d_pred, d, "confusion")

        if update:
            self.update_loss(loss, self.optimizers["confusion"])

        return loss, d_pred

    def update_loss(self, loss, optimizer):

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()

    def load_checkpoint(self, checkpoint):

        for model_name, pth_path in checkpoint.items():
            self.models[model_name].load_state_dict(torch.load(pth_path))
            print(f"{model_name.capitalize()} is successfully loaded")

    def zero_grad(self, model):

        for param in model.parameters():
            param.grad = None

    def get_phase_dict(self):  # DEPRECATED

        phase_dict = dict()
        stop = 0
        for i, e in enumerate(self.cfg.phase_config.epochs):

            phase_dict[i] = range(stop, stop + e)
            stop += e

        return phase_dict

    def agg_loss(self, losses):

        loss_dict = dict()
        for l in losses:
            k, v = list(l.average.items())[0]
            loss_dict[k] = v
        return loss_dict

    def check_which_phase(self, e):

        if not hasattr(self, "phase_dict"):
            self.phase_dict = self.get_phase_dict()

        # WILL CONTAIN [('phase_n'), range(n1, n2)]
        where_e = list(
            filter(lambda args: e in args[1], enumerate(self.phase_dict.values()))
        )
        assert len(where_e) == 1
        return where_e[0][0]

    def gather_result(self, preds, datatype="age", sampling="train"):

        prefix = sampling

        if isinstance(preds, list):
            preds = torch.tensor(preds, dtype=torch.float).squeeze()

        if datatype == "age":

            """
            Calculate
            - Mean Absolute Error
            - Correlation
            - R Squared
            """
            metrics = ["mae", "corr", "r2"]
            gt = getattr(self, f"gt_age_{sampling}")

            return {
                f"{prefix}_{metric}": get_metric(preds, gt, metric)
                for metric in metrics
            }

        elif datatype == "domain":

            """
            Receive Domain predictor
            Calculate
            - AUC
            - Accuracy
            """

            metrics = ["auc", "acc"]
            if self.cfg.partial < 1:
                metrics = ["acc"]
            gt = getattr(self, f"gt_src_{sampling}")

            return {
                f"{prefix}_{metric}": get_metric(preds, gt, metric)
                for metric in metrics
            }

    def prompt(self, result):

        metrics = sorted(set(map(lambda x: x.split("_")[-1], result.keys())))

        def _prompt(prefix):

            print(f"{prefix.upper()}")
            count = 0
            for metric in metrics:

                key = f"{prefix}_{metric}"
                if result.get(key) is not None:
                    count += 1
                    end = "  |  " if count % 3 != 0 else "\n"
                    print(f"{metric:6s}: {result[key]:.4f}", end=end)

            print("")

        _prompt("train")
        _prompt("valid")
        print("")
