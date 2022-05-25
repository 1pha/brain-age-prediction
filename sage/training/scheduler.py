import torch.optim.lr_scheduler as lr

step_on_batch_list = ["cosine_linear_warmup", "linear_warmup"]


def construct_scheduler(optimizer, training_args, logger=None):

    name = training_args.scheduler
    patience = training_args.lr_patience

    if logger is not None:
        logger.debug(f"Construct {name} learning rate scheduler.")

    if name == "plateau":
        scheduler = lr.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=patience, min_lr=1e-6, eps=1e-4
        )

    elif name == "cosine_linear_warmup":
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
        )

    elif name == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
        )

    elif name == "exp_decay":

        scheduler = lr.ExponentialLR(optimizer, gamma=training_args.gamma)

    elif name is None or name == "":
        scheduler = None

    return scheduler
