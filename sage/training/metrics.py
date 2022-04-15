import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class confusion_loss(nn.Module):
    def __init__(self, task=0):
        super(confusion_loss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum, x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss


def accuracy(y, y_pred):

    y_pred = y_pred.argmax(axis=1)
    return sum(y == y_pred) / len(y)


def auc(y, y_pred):

    if y_pred.shape[1] == 2:
        y_pred = y_pred.argmax(axis=1)
        return roc_auc_score(y, y_pred)

    else:
        return roc_auc_score(y, y_pred, multi_class="ovr")


def get_metric(y_pred, y, metric: str):

    """
    y_pred, y: given with CPU, gradient DETACHED TORCH TENSOR
    """
    NUMPY_METRICS = {
        "r2": r2_score,
        "acc": accuracy,
        "auc": auc,
    }

    TORCH_METRICS = {
        "mse": nn.MSELoss(),
        "rmse": RMSELoss(),
        "mae": nn.L1Loss(),
        "corr": lambda _p, _t: pearsonr(_p, _t)[0],
        "ce": nn.CrossEntropyLoss(),
        "confusion": confusion_loss(),
    }

    metric = metric.lower()
    if metric in NUMPY_METRICS.keys():
        return NUMPY_METRICS[metric](y.numpy(), y_pred.numpy())

    elif metric in TORCH_METRICS.keys():
        return TORCH_METRICS[metric](y_pred, y)


def get_metric_fn(metric: str):

    NUMPY_METRICS = {
        "r2": r2_score,
        "acc": accuracy,
        "auc": auc,
    }

    TORCH_METRICS = {
        "mse": nn.MSELoss(),
        "rmse": RMSELoss(),
        "mae": nn.L1Loss(),
        "corr": lambda _p, _t: pearsonr(_p, _t)[0],
        "ce": nn.CrossEntropyLoss(),
        "confusion": confusion_loss(),
    }

    metric = metric.lower()
    if metric in NUMPY_METRICS.keys():
        fn = lambda y_pred, y: NUMPY_METRICS[metric](y.numpy(), y_pred.numpy())
        return fn

    elif metric in TORCH_METRICS.keys():
        return TORCH_METRICS[metric]


if __name__ == "__main__":

    y_pred = torch.tensor(
        [
            [0.1, 0.9, 0.8],
            [0.2, 0.3, 0.7],
            [0.4, 0.5, 0.1],
        ],
        dtype=torch.float16,
    )
    y_true = torch.tensor([1, 2, 1], dtype=torch.float16)

    print(get_metric(y_pred, y_true, "acc"))
