import torch
from torch import optim
from torch.nn.functional import gaussian_nll_loss
import pytorch_lightning as pl
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from src.utils.utils import perm_2_mat

# define the LightningModule
class SOSModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4,
                 **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def setup(self, stage):
        d = self.trainer.datamodule.data.shape[-1]
        self.perm = np.arange(d) ## start at identity permutation
        self.perm_mat = perm_2_mat(self.perm).to(self.device)
        self.jac = torch.zeros(d, d).to(self.device)
    
    def forward(self, x):
        x = x @ self.perm_mat
        return self.model(x)

    def flow_loss(self, z, logdet, size_average=True):
        """Assuming Standard Gaussian source distribution
        """
        nll_loss = gaussian_nll_loss(z, torch.zeros_like(z), torch.ones_like(z), reduction='none')
        nll = nll_loss.sum(dim = len(nll_loss.shape)-1)
        loss = nll - logdet.squeeze()
        if size_average:
            return loss.mean(), loss.argmax()
        return loss.sum()
    
    def training_step_(self, batch, batch_idx):
        x = batch
        z, logdet = self.forward(x)
        loss, max_idx = self.flow_loss(z, logdet)

        return loss

    def get_transform(self):
        return lambda x: self.forward(x)[0]

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
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//10, gamma=0.97)
        #return optimizer
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    
    def return_adjacency(self):
        adj = torch.t(self.perm_mat) @ self.jac @ self.perm_mat
        return self.jac
    

# define the LightningModule
class CausalNFModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4,
                 **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def setup(self, stage):
        d = self.trainer.datamodule.data.shape[-1]
        self.perm = np.arange(d) ## start at identity permutation
        self.perm_mat = perm_2_mat(self.perm).to(self.device)
        self.jac = torch.zeros(d, d).to(self.device)
    
    def forward(self, x):
        output = {}
        n_flow = self.model()

        log_prob = n_flow.log_prob(x)
        output["log_prob"] = log_prob
        output["loss"] = -log_prob
        return output

    def get_transform(self):
        return self.model().transform
    
    def training_step_(self, batch, batch_idx):
        x = batch
        loss_dict = self.forward(x)
        loss_dict["loss"] = loss_dict["loss"].mean()
        return loss_dict

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss_dict = self.training_step_(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss_dict["loss"])
        return loss_dict

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        loss_dict = self.training_step_(batch, batch_idx)
        self.log("val_loss", loss_dict["loss"])
        self.log("Learning Rate", self.scheduler.get_last_lr()[0])

    def test_step(self, batch, batch_idx):
        # this is the test loop
        loss_dict = self.training_step_(batch, batch_idx)
        self.log("test_loss", loss_dict["loss"])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//10, gamma=0.97)
        #return optimizer
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
    
    def return_adjacency(self):
        adj = torch.t(self.perm_mat) @ self.jac @ self.perm_mat
        return self.jac
    


