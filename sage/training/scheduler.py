import torch.optim.lr_scheduler as lr


def construct_optimizer(optimizer, training_args):

    name = training_args.scheduler
    patience = training_args.patience

    if name == "plateau":
        scheduler = lr.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=patience, min_lr=1e-6, eps=1e-4
        )
        return scheduler

    elif name is None:
        return None
