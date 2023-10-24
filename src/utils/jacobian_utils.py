import numpy as np
import torch
from torch.autograd.functional import jacobian

from src.data.components.dep_matrix import dependency_matrix

def jacobian_dynamics(model, data):
    n = len(data)
    d = dependency_matrix.shape[0]
    J = torch.zeros(d, d)

    for batch_idx, batch in enumerate(data):
        for x in batch:
            #J += torch.abs(model._jacob(x))
            x = x.view(1, d)
            jac = jacobian(lambda x: model.forward(x)[0], x).view(d,d)
            J += torch.abs(jac)

    return J / n