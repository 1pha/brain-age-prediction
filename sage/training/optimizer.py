from itertools import chain

import torch.nn as nn

try:
    from torch.optim import Adam, AdamW
except:
    from torch.optim import Adam


def get_optimizer(models, cfg):

    if isinstance(models, nn.Module):
        params = models.parameters()
    else:
        params = list(chain(*([list(m.parameters()) for m in models])))

    if cfg.optimizer == "adam":
        optimizer = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    optimizer.zero_grad()

    return optimizer
