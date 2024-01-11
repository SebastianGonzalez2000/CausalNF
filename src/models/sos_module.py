import torch
from torch import optim
import lightning.pytorch as pl
import numpy as np

# define the LightningModule
class SOSModule(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def flow_loss(self, z, logdet, size_average=True):
        """If using Uniform as source distribution
        log_probs = 0
        """
        log_probs = (-0.5 * z.pow(2) - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        loss = -(log_probs + logdet).sum()
        if size_average:
            loss /= z.size(0)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        Z, logdet = self.model(x)

        loss = self.flow_loss(Z, logdet)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch
        Z, logdet = self.model(x)

        val_loss = self.flow_loss(Z, logdet)
        self.log("val_loss", val_loss)
        self.log("Learning Rate", self.scheduler.get_last_lr()[0])

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch
        Z, logdet = self.model(x)

        test_loss = self.flow_loss(Z, logdet)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//100, gamma=0.97)
        #return optimizer
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
