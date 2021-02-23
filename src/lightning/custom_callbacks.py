import pytorch_lightning as pl

class PrintCallback(pl.callbacks.Callback):

    def __init__(self, cfg):
        super(PrintCallback, self).__init__()
        self.cfg = cfg
        self.best_mae = 0
    
    def on_epoch_end(self, epoch, logs):

        print(f"Epoch: {epoch}")
        if epoch % self.cfg['verbose_period'] == 0:

            for k, v in logs.items():

                print(f'[{k.upper()}] {v:.3f}')

