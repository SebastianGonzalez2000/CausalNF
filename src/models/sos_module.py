import torch
from torch import optim
from torch.nn.functional import gaussian_nll_loss
import lightning.pytorch as pl
import numpy as np
import math

# define the LightningModule
class SOSModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.model = model
        self.lr= lr
        self.weight_decay = weight_decay

    def flow_loss(self, z, logdet, size_average=True):
        """Assuming Standard Gaussian source distribution
        """
        log_probs = gaussian_nll_loss(z, torch.zeros_like(z), torch.ones_like(z), reduction='none')
        loss = -(log_probs + logdet).sum()
        if size_average:
            loss /= loss.shape[0]
        return loss
    
    def training_step_(self, batch, batch_idx):
        x = batch
        z, logdet = self.model(x)

        loss = self.flow_loss(z, logdet)

        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self.training_step_(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        loss = self.training_step_(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("Learning Rate", self.scheduler.get_last_lr()[0])


    def test_step(self, batch, batch_idx):
        # this is the test loop
        loss = self.training_step_(batch, batch_idx)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//100, gamma=0.97)
        #return optimizer
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
