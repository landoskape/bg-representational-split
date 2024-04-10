from scipy.spatial.distance import pdist, squareform
import torch


# helper methods
def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))


def pca(input, centered=True, correction=True):
    """
    algorithm for pca

    input should either have shape (batch, dim, samples) or (dim, samples)
    will center data when centered=True
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input should be a matrix or batched matrices"
    assert isinstance(correction, bool), "correction should be a boolean"

    if input.ndim == 2:
        no_batch = True
        input = input.unsqueeze(0)  # create batch dimension for uniform code
    else:
        no_batch = False

    if centered:
        input = input - input.mean(dim=2, keepdim=True)

    _, _, S = input.size()
    v, w, _ = named_transpose([torch.linalg.svd(inp) for inp in input])

    # convert singular values to eigenvalues
    w = [ww**2 / (S - 1.0 * correction) for ww in w]

    # return to stacked tensor across batch dimension
    w = torch.stack(w)
    v = torch.stack(v)

    # if no batch originally provided, squeeze out batch dimension
    if no_batch:
        w = w.squeeze(0)
        v = v.squeeze(0)

    # return eigenvalues and eigenvectors
    return w, v


def spdist(X, square=True):
    """
    Wrapper for scipy pdist function to return squareform distance matrix
    """
    d = pdist(X)
    if square:
        d = squareform(d)
    return d
