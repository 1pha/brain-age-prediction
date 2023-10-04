import pytorch_lightning as pl


def ckpt_steps(multiplier: int = 1) -> int:
    anchors = [250, 1_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    if multiplier > 1:
        anchors = [multiplier * a for a in anchors]

    while anchors:
        yield anchors.pop(0)

    interval = 500_000
    a = anchors[-1] + interval  # Start from 250,000 more than the last anchor
    while True:
        yield a
        a += interval * multiplier


def ckpt_epochs() -> int:
    anchors = [1, 5, 10, 15, 20, 25, 30, 50, 70, 100]
    while anchors:
        yield anchors.pop(0)

    interval = 50
    a = anchors[-1] + interval # Start from 250,000 more than the last anchor
    while True:
        yield a
        a += interval


class AnchorCheckpoint(pl.Callback):
    def __init__(self, multiplier: int = 1):
        super().__init__()
        self.save_steps = ckpt_steps(multiplier=multiplier)
        self.save_step = next(self.save_steps)
        
        self.save_epochs = ckpt_epochs()
        self.save_epoch = next(self.save_epochs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        save_dir = pl_module.save_dir
        current_global_step = trainer.global_step
        if current_global_step == self.save_step:
            metrics = trainer.callback_metrics
            train_mae = float(metrics["train_loss"])
            checkpoint_path = f"step{current_global_step}-train_mae{train_mae:.3f}.ckpt"
            trainer.save_checkpoint(save_dir / checkpoint_path, weights_only=True)
            self.save_step = next(self.save_steps)

    def on_validation_epoch_end(self, trainer, pl_module):
        save_dir = pl_module.save_dir
        current_epoch = trainer.current_epoch
        if current_epoch == self.save_step:
            metrics = trainer.callback_metrics
            valid_loss = float(metrics["valid_loss"])
            checkpoint_path = f"epoch{current_epoch}-valid_loss{valid_loss:.3f}.ckpt"
            trainer.save_checkpoint(save_dir / checkpoint_path, weights_only=True)
            self.self.epoch = next(self.save_epochs)
