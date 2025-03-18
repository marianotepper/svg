from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

class SearchGraph(ABC):
    def __init__(self, kernel_fun):
        self.kernel_fun = kernel_fun
        self.graph = None
        self.X = None
        self.entrypoint = None

    @abstractmethod
    def fit(self, X: np.ndarray):
        ...

    def entrypoint_from_centroid(self):
        mu = np.mean(self.X, axis=0)
        K_mu = self.kernel_fun(mu, self.X)
        self.entrypoint = np.argmax(K_mu)

    def search(self, query: np.ndarray, entrypoint: Optional[int] = None):
        if entrypoint is None:
            current_node = self.entrypoint
        else:
            current_node = entrypoint

        K_current_node = self.kernel_fun(query, self.X[current_node])

        not_done = True
        while not_done:
            neighs = [e[1] for e in self.graph.edges(current_node)]
            K = self.kernel_fun(query, self.X[neighs])
            best_neigh = int(np.argmax(K))

            if K[best_neigh] > K_current_node:
                current_node = neighs[best_neigh]
                K_current_node = K[best_neigh]
            else:
                not_done = False

        return current_node
