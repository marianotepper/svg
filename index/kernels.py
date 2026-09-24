from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from typing import Union


@dataclass
class Kernel:
    sigma: Union[str, float] = 'auto'
    similarity: str = 'euclidean'

    def build_kernel(self, X: np.ndarray):
        return self.build_kernel2(X, X)

    def build_kernel2(self, x: np.ndarray, y: np.ndarray):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if len(y.shape) == 1:
            y = y[np.newaxis, :]

        if self.similarity == 'euclidean':
            D = cdist(x, y, metric='sqeuclidean')
        elif self.similarity == 'dot_product':
            D = -x @ y.T
        else:
            raise ValueError("similarity must be 'euclidean' or 'dot_product'")

        if ((not isinstance(self.sigma, float))
                and isinstance(self.sigma, str) and self.sigma != 'auto'):
            raise ValueError("sigma must be either 'auto' or a float")

        elif self.sigma == 'auto':
            sigma = D.max() / 2
        else:
            sigma = self.sigma

        K = np.exp(-D / (sigma ** 2))
        return K

    def build_kernel_matrix(self, X: np.ndarray, build_full_kernel: bool):
        if build_full_kernel:
            return FullKernelMatrix(self, X)
        else:
            return KernelMatrix(self, X)


class KernelMatrix:
    def __init__(self, kernel: Kernel, X: np.ndarray):
        self.kernel = kernel
        self.X = X
        self.shape = (self.X.shape[0], self.X.shape[0])

    def get_submatrix(self, row_index, col_index=None):
        if col_index is None:
            K = self.kernel.build_kernel2(self.X, self.X[row_index])
        else:
            K = self.kernel.build_kernel2(self.X[row_index], self.X[col_index])

        return np.squeeze(K)

    def __len__(self):
        return self.shape[0]

class FullKernelMatrix:
    def __init__(self, kernel: Kernel, X: np.ndarray):
        self.kernel = kernel
        self.K = self.kernel.build_kernel(X)
        self.shape = self.K.shape

    def get_submatrix(self, row_index, col_index=None):
        if col_index is None:
            K = self.K[:, row_index]
        else:
            K = self.K[row_index]
            if len(K.shape) == 2:
                K = K[:, col_index]
            else:
                K = K[col_index]

        return np.squeeze(K)

    def __len__(self):
        return self.shape[0]