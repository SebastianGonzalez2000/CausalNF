import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.data.components.dep_matrix import dependency_matrix

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    # TODO: READ MADE PAPER. Why do we mask this way?
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class ConditionerNet(nn.Module):
    def __init__(self, input_size, hidden_size, k, m, n_layers=1):
        super().__init__()
        self.k = k
        self.m = m
        self.input_size = input_size
        self.output_size = k * self.m * input_size + input_size
        self.network = self._make_net(
            input_size, hidden_size, self.output_size, n_layers)

    def _make_net(self, input_size, hidden_size, output_size, n_layers):
        if self.input_size > 1:
            input_mask = get_mask(
                input_size, hidden_size, input_size, mask_type='input')
            hidden_mask = get_mask(hidden_size, hidden_size, input_size)
            output_mask = get_mask(
                hidden_size, output_size, input_size, mask_type='output')

            network = nn.Sequential(
                MaskedLinear(input_size, hidden_size, input_mask), nn.ReLU(),
                MaskedLinear(hidden_size, hidden_size, hidden_mask), nn.ReLU(),
                MaskedLinear(hidden_size, output_size, output_mask))
        else:
            network = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, output_size))

        return network

    def forward(self, inputs):
        batch_size = inputs.size(0)
        params = self.network(inputs)
        i = self.k * self.m * self.input_size
        c = params[:, :i].view(batch_size, -1, self.input_size).transpose(1, 2).view(
            batch_size, self.input_size, self.k, self.m, 1)
        # Each latent variable has its own constant for its polynomial
        const = params[:, i:].view(batch_size, self.input_size)
        # TODO: What is this doing? Why C? Answer: You are squaring each polynomial
        C = torch.matmul(c, c.transpose(3, 4))
        return C, const


class SOSFlow(nn.Module):
    @staticmethod
    def power(z, m):
        return z ** (torch.arange(m).float().to(z.device))

    def __init__(self, hidden_size, k, r, input_size=dependency_matrix.shape[0], n_layers=1): 
        super().__init__()
        self.k = k
        self.m = r+1

        self.conditioner = ConditionerNet(
            input_size, hidden_size, k, self.m, n_layers)
        self.register_buffer('filter', self._make_filter())

    def _make_filter(self):
        n = torch.arange(self.m).unsqueeze(1)
        e = torch.ones(self.m).unsqueeze(1).long()
        filter = (n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1
        return filter.float()

    def forward(self, inputs, mode='direct'):
        batch_size, input_size = inputs.size(0), inputs.size(1)
        C, const = self.conditioner(inputs)
        X = SOSFlow.power(inputs.unsqueeze(-1), self.m).view(batch_size, input_size, 1, self.m,
                                                             1)  # bs x d x 1 x m x 1
        # Dividing by filter, mimics integrating polynomial
        # multiply transform by inputs because integral increases degree of polynomial by one
        Z = self._transform(X, C / self.filter) * inputs + const
        # Derivative of each polynomial is just inner sum over k
        logdet = torch.log(torch.abs(self._transform(X, C))
                           ).sum(dim=1, keepdim=True)
        return Z, logdet

    def _transform(self, X, C):
        # bs x d x k x m x 1
        CX = torch.matmul(C, X)
        # bs x d x k x 1 x 1
        XCX = torch.matmul(X.transpose(3, 4), CX)
        # bs x d
        summed = XCX.squeeze(-1).squeeze(-1).sum(-1)
        return summed