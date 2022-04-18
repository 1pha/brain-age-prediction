from itertools import chain

import torch.nn as nn

try:
    from torch.optim import Adam, AdamW
except:
    from torch.optim import Adam


def construct_optimizer(model, training_args, logger=None):

    params = model.parameters()
    optimizer = training_args.optimizer
    learning_rate = training_args.learning_rate
    weight_decay = training_args.weight_decay
    momentum = training_args.momentum

    if logger is not None:
        logger.debug(f"Construct {optimizer} optimizer.")

    if optimizer == "adam":
        optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)

    elif optimizer == "adamW":
        optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()
    return optimizer
