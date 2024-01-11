import torch
from torch import nn, from_numpy
import numpy as np

import os
import torch
from torch import utils
import lightning.pytorch as pl

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
    relu1 = nn.GELU()
    fc1.weight.requires_grad = False
    nn.init.uniform_(fc1.weight, 5.0, 10.0)
    sparse_index = torch.where(sparsity_vec == 0)
    for idx in sparse_index:
        fc1.weight[:,idx] = torch.tensor(0.0)
    layers = [fc1, relu1]
    for _ in range(1, n_layers - 1):
        fc = nn.Linear(hidden_n, hidden_n, bias = False)
        fc.weight.requires_grad = False
        nn.init.uniform_(fc.weight, 0, 1)
        relu = nn.GELU()
        layers.extend([fc, relu])
    ### final layer
    fc_final = nn.Linear(hidden_n, 1, bias=False)
    relu_final = nn.GELU()
    fc_final.weight.requires_grad = False
    nn.init.uniform_(fc_final.weight, 0, 1)
    layers.extend([fc_final, relu_final])

    net = nn.Sequential(*layers)

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
    mixing_net = sparse_MLP(torch.tensor(dependency_matrix), 2)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False
    
    eps = torch.randn((n,d))

    mixing_net.eval()
    x = mixing_net(eps)
    x = (x - torch.mean(x, dim=0)) / (torch.std(x, dim=0))

    return x


class MLPDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_size = 30000):
        super().__init__()
        self.batch_size = batch_size
        self.data_size = data_size
        self.jac_batch_size = data_size // 10

    def prepare_data(self):
        self.data = generate_dataset_mlp(self.data_size)

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
