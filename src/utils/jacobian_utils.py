import numpy as np
import torch
from torch.autograd.functional import jacobian

from src.data.components.dep_matrix import dependency_matrix

def jacobian_dynamics(model, data):
    num_batches = 0
    d = dependency_matrix.shape[0]
    # TODO: Fix this. How to automatically instantiate this on GPU with lightning
    J = torch.zeros(d, d)#.cuda(0)

    for batch_idx, batch in enumerate(data):
        # TODO: Fix this. Why is the batch sent to CPU during callback?
        batch = batch#.cuda(0)
        num_batches += 1
        batchjac = mean_batch_jacobian(model, batch)
        J += batchjac

    J = J / num_batches
    return torch.transpose(torch.transpose(J, 0,1) / torch.max(J, dim=1).values, 0,1)

def mean_batch_jacobian(f, x, threshold = 1e-1):
    f_sum = lambda x: torch.sum(f.forward(x)[0], axis=0)
    batchjac = jacobian(f_sum, x).permute(1,0,2)
    batchjac = torch.abs(batchjac).mean(axis = 0)

    #return F.threshold(batchjac, threshold, 0)
    return batchjac