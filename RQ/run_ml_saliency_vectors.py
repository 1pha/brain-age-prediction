import argparse
from collections import defaultdict
from datetime import datetime
import sys
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import pandas as pd

sys.path.append("/home/daehyun/codespace/brain-age-prediction/")
sys.path.append("/home/daehyun/codespace/brain-age-prediction/sage")

import RQ.metadata as m
import RQ.constants as C
PERF_COL = "performance"
SEED = 42


def _train_lgbm(X: np.ndarray, y: np.ndarray, reg: bool = True):
    LGBM_PARAMS = dict(reg_alpha=0.1, reg_lambda=0.1, num_leaves=50, n_estimators=200)
    
    y_gts, y_preds, fis = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (trn_idx, tst_idx) in enumerate(kf.split(X=X, y=y)):
        X_train, y_train = X[trn_idx], y[trn_idx]
        X_test, y_test = X[tst_idx], y[tst_idx]

        model = LGBMRegressor(**LGBM_PARAMS) if reg else LGBMClassifier(**LGBM_PARAMS)
        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_test)
        fi = model.feature_importances_

        y_gts.extend(y_test.tolist())
        y_preds.extend(y_pred.tolist())
        fis.append(fi)

    fis = np.stack(fis)
    if reg:
        mae = mean_absolute_error(y_true=y_gts, y_pred=y_preds)
        mse = mean_squared_error(y_true=y_gts, y_pred=y_preds)
        r2 = r2_score(y_true=y_gts, y_pred=y_preds)
        return dict(mae=mae, mse=mse, r2=r2, y_preds=y_preds)
    else:
        acc = accuracy_score(y_true=y_gts, y_pred=y_preds)
        f1 = f1_score(y_true=y_gts, y_pred=y_preds)
        auc = roc_auc_score(y_true=y_gts, y_score=y_preds)
        return dict(acc=acc, f1=f1, auc=auc, y_preds=y_preds)


def _train_knn(X: np.ndarray, y: np.ndarray, reg: bool = True):
    y_gts, y_preds, fis = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (trn_idx, tst_idx) in enumerate(kf.split(X=X, y=y)):
        X_train, y_train = X[trn_idx], y[trn_idx]
        X_test, y_test = X[tst_idx], y[tst_idx]

        model = KNeighborsRegressor() if reg else KNeighborsClassifier()
        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_test)

        y_gts.extend(y_test.tolist())
        y_preds.extend(y_pred.tolist())

    if reg:
        mae = mean_absolute_error(y_true=y_gts, y_pred=y_preds)
        mse = mean_squared_error(y_true=y_gts, y_pred=y_preds)
        r2 = r2_score(y_true=y_gts, y_pred=y_preds)
        return dict(mae=mae, mse=mse, r2=r2, y_preds=y_preds)
    else:
        acc = accuracy_score(y_true=y_gts, y_pred=y_preds)
        f1 = f1_score(y_true=y_gts, y_pred=y_preds)
        auc = roc_auc_score(y_true=y_gts, y_score=y_preds)
        return dict(acc=acc, f1=f1, auc=auc, y_preds=y_preds)


def train_ml(interps, gt, xai_key: str, model_key: str, _train: callable, reg: bool = True) -> dict:
    X_full = interps[xai_key][model_key]
    results = defaultdict(list)
    for X in X_full:
        result = _train(X=X, y=gt, reg=reg)
        for key in result:
            results[key].append(result[key])
        results[C.XCOL] = xai_key
        results["Model Key"] = model_key
    return results


def fit(interps, gt, fitter: str = "lgbm", reg: bool = True,
        xai_methods=C.XAI_METHODS, model_list=C.MODELS,
        output_name: str = None):
    _train: callable = {
        "lgbm": _train_lgbm, "knn": _train_knn
    }[fitter]

    # Generate ground truth
    y_true = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (_, tst_idx) in enumerate(kf.split(gt)):
        y_true.extend(gt[tst_idx].tolist())
    y_true = np.array(y_true)

    results = [] # List of dicts
    for xai_key in xai_methods:
        for model_key in model_list:
            print(f"\nXAI: {xai_key}\nMODEL: {model_key}\nMAE: ")
            result = train_ml(interps=interps, gt=gt, xai_key=xai_key, model_key=model_key,
                              _train=_train, reg=reg)
            n = len(interps[xai_key][model_key])
            result["XAI Method"] = [xai_key for _ in range(n)]
            result["Model Key"] = [f"{model_key}-{i}" for i in range(n)]
            result.pop('y_preds')
            results.append(pd.DataFrame(result))
    results = pd.concat(results, axis=0)
    if output_name:
        results.to_csv(output_name, index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ukb", type=str, help="Saliency vectors to fetch")
    parser.add_argument("--fitter", default="lgbm", type=str, help="Fitter: lgbm or knn")

    args = parser.parse_args()
    return args


def main(args):
    """Trains over saliency vectors (interps)
    """
    ikwargs, fit_kwargs = dict(), dict(fitter=args.fitter)
    if args.dataset == "ukb":
        from sage.data import UKBDataset
        test_dataset = UKBDataset(root="/home/daehyun/codespace/brain-age-prediction/biobank",
                                  mode="test")
        y = test_dataset.labels.age.values

    elif args.dataset == "adni":
        from sage.data.adni import ADNIBinary
        test_dataset = ADNIBinary(mode="test", root="/home/daehyun/codespace/brain-age-prediction/adni")
        y = test_dataset.labels.DX_bl.apply(test_dataset.MAPPER2INT.get).values
        
        ikwargs = dict(base_dir=Path("/home/daehyun/data/hdd03/1pha/adni_extra2/extra"),
                      is_cls=True, strict=False, verbose=False)
        fit_kwargs["reg"] = False
    else:
        raise
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    fit_kwargs["output_name"] = f"{args.fitter}_{args.dataset}_{current_time}.csv"

    interps = m.load_interps(**ikwargs)
    ml_result = fit(interps=interps, gt=y, **fit_kwargs)


if __name__=="__main__":
    args = parse_args()
    main(args)
    