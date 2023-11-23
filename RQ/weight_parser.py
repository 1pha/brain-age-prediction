import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np

import constants as C
import utils

logger = utils.get_logger(name=__file__)


class Weights:
    """ Object that can easily load the followings:
        - Pre-trained weights / checkpoints
        - Configuration
        - XAI maps, attributes, graphs
        
    Access variables via
    - .imgs
    - .xai_dict
    - .xai_dict_indiv
    - .attrs
    - .top_attr
    """

    XAI_METHODS = {"gbp", "gcam_avg", "ig", "gradxinput", "lrp"}
    def __init__(self,
                 model_name: str = "resnet10",
                 seed: int = 42,
                 base_dir: Path = C.WEIGHT_DIR,
                 xai_method: str = "",
                 debug: bool = False):
        self.base_path: Path = base_dir / f"{model_name}-{seed}"
        if not self.base_path.exists():
            self.base_path = (self.base_path / ".." / model_name).resolve()
        logger.info("Set base_path as %s", self.base_path)
        self.debug = debug
        
        logger.info("Loading Basic Information")
        self.prediction = self.load_prediction()
        self.config = self.load_config()
        self.ckpt_dict = self.load_checkpoint()
        self.test_performance = self.load_test_performance()
        
        if xai_method:
            # One can explicitly set xai method after instantiation via `obj.load_xai` as well.
            self.load_xai(xai_method=xai_method)
            
    def __repr__(self) -> str:
        return " / ".join([f"{k}: {v}" for k, v in self.test_performance.items()])

    def load_prediction(self, name: str = None) -> dict:
        if name is not None:
            # Load if there is a designate prediction file
            prediction = utils.load_pkl(self.base_path / name)
            return prediction

        else:
            preds = list(self.base_path.glob("*.pkl"))
            if len(preds):
                prediction = utils.load_pkl(preds[0])
                return prediction
            else:
                logger.info("There is no prediction pickle file in %s.", self.base_path)
                logger.info("Check the following: %s", list(self.base_path.glob("*")))

    def load_config(self,
                    config_fname: str = "config.yaml",
                    config_path: str = ".hydra",
                    version_base: str = "1.1") -> dict:
        """ Load config file used for training. """
        _config_path = str(self.base_path / config_path)
        with hydra.initialize(config_path=os.path.relpath(_config_path), version_base=version_base):
            config = hydra.compose(config_name=config_fname)
        return config

    def load_checkpoint(self, step: int = None) -> Dict[str, List[tuple] | Path]:
        ckpts = list(self.base_path.glob("*.ckpt"))
        ckpt_dict = dict(steps=[])
        for _ckpt in ckpts:
            ckpt = str(_ckpt.stem)
            if ckpt.startswith("step"):
                # Checkpoints saved during training
                step = int(ckpt.lstrip("step").split("-")[0])
                train_mae = float(ckpt.split("-")[-1].lstrip("train_mae"))
                ckpt_dict["steps"].append((step, train_mae))

            elif ckpt == "last":
                # Last Checkpoint
                ckpt_dict["last"] = _ckpt

            elif ckpt[0].isdigit():
                # Best Checkpoint
                ckpt_dict["best"] = _ckpt
                step, mae = ckpt.split("-")
                mae = float(mae.lstrip("valid_mae"))
                ckpt_dict["best_valid_mae"] = [(int(step), mae)]
        return ckpt_dict
    
    def load_test_performance(self, log_fname: str = "train.log") -> Dict[str, float]:
        with open(self.base_path / log_fname, mode="r") as f:
            logs = f.readlines()
        
        perfs = dict(mse="MSE:", mae="MAE:", r2="R2 :")
        # Filterout lines with performances
        logs = [line for line in logs if any(p in line for p in perfs.values())]
        for log in logs:
            for metric, char in perfs.items():
                if isinstance(char, str) and char in log:
                    val = float(log.split(char)[-1])
                    perfs[metric] = val
        return perfs
    
    def set_path(self, xai_method: str, topk: float = 0.99) -> None:
        assert xai_method in self.XAI_METHODS, \
               f"Please provide valid xai_method in string: given {xai_method}" + \
               f"\nPossible methods: {self.XAI_METHODS}"
        xai_path = self.base_path / f"{xai_method}k{topk}"
        assert xai_path.exists(), f"Check if provided xai_path exists: {xai_path}"
        self.xai_path = xai_path

    def clear_path(self):
        self.xai_method = None

    def load_xai(self, xai_method: str, topk: float = 0.99) -> None:
        self.clear_path()
        self.set_path(xai_method=xai_method, topk=topk)
        logger.info("Loading XAI information from %s", self.xai_path)
        
        self.load_imgs()
        self.load_xai_dict()
        self.load_xai_dict_indiv()
        self.load_attributes()

    def load_imgs(self) -> Dict[str, Path]:
        imgs = self.xai_path.rglob("*.png")
        img_dict = {}
        for img in imgs:
            img_dict[img.stem] = img
        self.imgs = img_dict
        return img_dict

    def load_xai_dict(self) -> Dict[str, float]:
        xai_dict = utils.load_json(path=self.xai_path / "xai_dict.json")
        self.xai_dict = xai_dict
        return xai_dict

    def load_xai_dict_indiv(self) -> Dict[str, List[float]]:
        xai_dict_indiv = utils.load_json(path=self.xai_path / "xai_dict_indiv.json")
        self.xai_dict_indiv = xai_dict_indiv
        return xai_dict_indiv

    def load_attributes(self) -> Dict[str, np.ndarray]:
        self.attrs = np.load(self.xai_path / "attrs.npy")
        self.top_attr = np.load(self.xai_path / "top_attr.npy")
        return dict(attrs=self.attrs, top_attr=self.top_attr)


class WeightAvg:
    """ Pivoting across multiple seeds per run. """
    def __init__(self, model_name: str, xai_method: str = "", seeds: List[int] = [42]):
        """ Fetch multiple Weights of a fixated model_name & xai_method.
        e.g. For resnet10, collect xai_attributes from all seeds """
        logger.info("Load all seeds: %s", seeds)
        self.num_seeds = len(set(seeds))
        if self.num_seeds != len(seeds):
            logger.warn("Check seed list. There must be duplicates or other issues: %s", seeds)

        self._init_seed = seeds[0]
        self.seed_dict = dict()
        for seed in seeds:
            w = Weights(model_name=model_name, xai_method=xai_method, seed=seed)
            self.seed_dict[seed] = w

        self.aggregate(agg_xai=bool(xai_method))

    def __repr__(self) -> str:
        return "\n".join([repr(w) for w in self.seed_dict.values()])

    def aggregate(self, agg_xai: bool = True) -> None:
        logger.info("Aggregate across %s seeds", self.num_seeds)
        self.test_performance, self.test_performance_std = self._agg_test_performance()
        if agg_xai:
            self.attrs = self._agg_array(arr_name="attrs")
            self.top_attr = self._agg_array(arr_name="top_attr")
            self.xai_dict, self.xai_dict_std = self._agg_xai_dict()
            self.xai_dict_indiv, self.xai_dict_indiv_std = self._agg_xai_dict_indiv()
        
    def _agg_test_performance(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        test_performance = {k: [] for k in self.seed_dict[self._init_seed].test_performance}
        for seed in self.seed_dict:
            w = self.seed_dict[seed]
            tp: dict = w.test_performance
            for metric in test_performance:
                test_performance[metric].append(tp[metric])
        test_performance_mu = {k: np.mean(v) for k, v in test_performance.items()}
        test_performance_std = {k: np.std(v) for k, v in test_performance.items()}
        return test_performance_mu, test_performance_std

    def _agg_array(self, arr_name: str) -> np.ndarray:
        """ Aggregate arrays """
        arr = np.zeros_like(getattr(self.seed_dict[self._init_seed], arr_name))
        for seed in self.seed_dict:
            arr += getattr(self.seed_dict[seed], arr_name)
        arr /= self.num_seeds
        return arr

    def _agg_xai_dict(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Aggregation of Dict[str, float]
        Returns mean/std dict respectively. """
        dct = {k: [] for k in self.seed_dict[self._init_seed].xai_dict}
        for seed in self.seed_dict:
            tmp_dct = self.seed_dict[seed].xai_dict
            for k in dct:
                dct[k].append(tmp_dct[k])
        mean_dct = {k: np.mean(v) for k, v in dct.items()}
        std_dct = {k: np.std(v) for k, v in dct.items()}
        return mean_dct, std_dct

    def _agg_xai_dict_indiv(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Aggregation of Dict[str, List[float]] """
        dct = {k: [] for k in self.seed_dict[self._init_seed].xai_dict_indiv}
        for seed in self.seed_dict:
            tmp_dct = self.seed_dict[seed].xai_dict_indiv
            for k in dct:
                dct[k].append(tmp_dct[k])
        mean_dct = {k: np.mean(v, axis=0) for k, v in dct.items()}
        std_dct = {k: np.std(v, axis=0) for k, v in dct.items()}
        return mean_dct, std_dct
