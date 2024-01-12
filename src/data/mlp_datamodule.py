import torch
from torch import nn, from_numpy
import numpy as np

import torch
from torch import utils
import pytorch_lightning as pl

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from src.data.components.dep_matrix import dependency_matrix

def generate_dataset(n):
    layers = []

    # FORCING GRAPH STRUCTURES
    weight_matrix = np.linalg.inv(dependency_matrix)

    d = weight_matrix.shape[0]

    lin_layer = nn.Linear(d, d, bias=False)
    lin_layer.weight.data = torch.tensor(weight_matrix)
    layers.append(lin_layer)

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    eps = torch.from_numpy(np.random.normal(0, 1, (n,d)))

    mixing_net.eval()
    x = mixing_net(eps)

    # x = (x - x.mean(axis=0)) / (x.std(axis=0))
    x = x.numpy()

    return x


def make_sparse_mlp(sparsity_vec, n_layers, hidden_n = None):
    ### first layer
    d = sparsity_vec.shape[0]
    if hidden_n is None:
        hidden_n = d * 2
    fc1 = nn.Linear(d,hidden_n, bias = False)
    relu1 = nn.ReLU()
    nn.init.normal_(fc1.weight, 0, 1)
    sparse_index = torch.where(sparsity_vec == 0)
    fc1.weight.requires_grad = False
    for idx in sparse_index:
        fc1.weight[:,idx] = torch.tensor(0.0)
    layers = [fc1, relu1]
    print(fc1.weight.data)
    for _ in range(1, n_layers - 1):
        fc = nn.Linear(hidden_n, hidden_n)
        #nn.init.normal_(fc.params, 0, 1)
        relu = nn.ReLU()
        layers.extend([fc, relu])
    ### final layer
    fc_final = nn.Linear(hidden_n, 1)
    #nn.init.normal_(fc_final.params, 0, 1)
    layers.extend([fc_final])

    net = nn.Sequential(*layers)

    for param in net.parameters():
        param.requires_grad = False

    return net 

class sparse_MLP(nn.Module):
    def __init__(self, sparsity, n_layers):
        super().__init__()
        d = sparsity.shape[0]
        self.mlps = [make_sparse_mlp(sparsity[i,:], n_layers) for i in range(d)]
    def forward(self, z):
        x = torch.zeros_like(z)
        for idx, mlp in enumerate(self.mlps):
            if idx == 0:
                x[:,0] = mlp(z).squeeze()
            else:
                inp = torch.cat((x[:,:idx], z[:,(idx):]), dim = 1)
                x[:,idx] = mlp(inp).squeeze()
        # x = [mlp(z) for mlp in self.mlps]
        # x = torch.cat(x, dim = 1)
        return x
    

def generate_dataset_mlp(n):

    d = dependency_matrix.shape[0]
    mixing_net = sparse_MLP(torch.tensor(dependency_matrix), 3)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False
    
    eps = torch.randn((n,d))

    mixing_net.eval()
    x = mixing_net(eps)
    x = (x - torch.mean(x, dim=0)) / (torch.std(x, dim=0))

    return x


class MLPDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32, 
                 data_size = 30000,
                 permutation = "identity" ## identity or random
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.data_size = data_size
        self.jac_batch_size = data_size // 10
        assert permutation in ["identity", "random"], "permutation must be identity or random"
        self.permutation = permutation

    def prepare_data(self):
        self.data = generate_dataset_mlp(self.data_size) 
        print(self.data[0])
        if self.permutation == "random":
            self.data = self.data[..., torch.randperm(self.data.shape[-1])]
        print(self.data[0])
        print(torch.mean(self.data, dim = 0))
        print(torch.std(self.data, dim = 0))


    def setup(self, stage: str):
        self.mlp_train, self.mlp_val, self.mlp_test = utils.data.random_split(
            self.data, 
            [int(0.8*self.data_size), int(0.1*self.data_size), int(0.1*self.data_size)], 
            generator=torch.Generator().manual_seed(42)
        )
        '''
        self.mlp_train = self.data[:int(0.8*self.data_size)]
        self.mlp_val = self.data[int(0.8*self.data_size):int(0.9*self.data_size)]
        self.mlp_test = self.data[int(0.9*self.data_size):]
        '''

    def train_dataloader(self):
        out = utils.data.DataLoader(self.mlp_train, batch_size=self.batch_size)
        return out

    def val_dataloader(self):
        return utils.data.DataLoader(self.mlp_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return utils.data.DataLoader(self.mlp_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return utils.data.DataLoader(self.mlp_train, batch_size=self.jac_batch_size)


if __name__ == "__main__":
    
    data_module = MLPDataModule(permutation="random")
    data_module.prepare_data()
    data_module.setup("train")
    train_loader = data_module.train_dataloader()
    print(next(iter(train_loader))[0:10])