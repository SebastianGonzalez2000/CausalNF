import torch

def perm_2_mat(perm):
    """Converts a permutation to a torch matrix.

    Args:
        perm (list): A permutation.

    Returns:
        numpy.ndarray: A permutation matrix.
    """
    n = len(perm)
    P = torch.zeros((n,n))
    for i,j in enumerate(perm):
        P[i,j] = 1
    
    return P