import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.lr = self.cfg.learning_rate

        self.train_acc = pl.metrics.regression.MeanAbsoluteError()
        self.val_acc = pl.metrics.regression.MeanAbsoluteError()
        self.test_acc = pl.metrics.regression.MeanAbsoluteError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        return optimizer, scheduler

    def loss_fn(self, logits, y):

        loss = nn.MSELoss()(logits, y)
        if self.cfg.lamb:
            l2_reg = torch.tensor(0.).type_as(loss)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            
            loss += self.cfg.lamb * l2_reg
        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        if self.cfg.resize:
            x = F.interpolate(x, size=(96, 96, 96))
        
        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        if self.cfg.resize:
            x = F.interpolate(x, size=(96, 96, 96))

        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        if self.cfg.resize:
            x = F.interpolate(x, size=(96, 96, 96))

        logits = self(x).squeeze(1)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)