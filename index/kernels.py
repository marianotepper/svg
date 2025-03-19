import numpy as np


def radial_basis_function(x: np.ndarray, y: np.ndarray, sigma: float = 1):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(y.shape) == 1:
        y = y[np.newaxis, :]
    return np.exp(-np.sum((x - y) ** 2, axis=1) / (sigma ** 2))