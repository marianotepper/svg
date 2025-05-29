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

        if self.sigma == 'auto':
            sigma = D.max() / 2
        else:
            sigma = self.sigma

        K = np.exp(-D / (sigma ** 2))
        return K