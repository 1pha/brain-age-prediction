import pytorch_lightning as pl


def ckpt_saver(multiplier: int = 1) -> int:
    anchors = [250, 1_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    if multiplier > 1:
        anchors = [multiplier * a for a in anchors]
    
    while anchors:
        yield anchors.pop(0)
    
    a = anchors[-1]  # Start from 250,000 more than the last anchor
    while True:
        yield a
        a += 500_000 * multiplier


class ManualCheckpoint(pl.Callback):
    def __init__(self, multiplier: int = 1):
        super().__init__()
        self.save_steps = ckpt_saver(multiplier=multiplier)
        self.save_step = next(self.save_steps)

    def on_train_batch_end(self, trainer, pl_module):
        current_global_step = trainer.global_step
        if current_global_step % self.save_step == 0:
            metrics = trainer.callback_metrics
            checkpoint_path = f"{current_global_step}-valid_mae{metrics}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            self.save_step = next(self.save_steps)
