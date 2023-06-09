import torch


def multiprod(A, B):
    # Added just to be parallel to manopt/pymanopt implemenetation
    return torch.matmul(A, B)


def multitransp(A):
    # First check if we have been given just one matrix
    if A.dim() == 2:
        return torch.transpose(A, 1, 0)
    return torch.transpose(A, 2, 1)


def multihconj(A):
    # Inspired by MATLAB multihconj function by Nicholas Boumal.
    return torch.conj(multitransp(A))


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))


def multiskew(A):
    # Inspired by MATLAB multiskew function by Nicholas Boumal.
    return 0.5 * (A - multitransp(A))

def multiherm(X):
    # Inspired by MATLAB multiherm function by Nicholas Boumal.
    return 0.5 * (X + multihconj(X))
