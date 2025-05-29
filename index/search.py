from abc import ABC, abstractmethod
from heapq import nlargest
from typing import Optional

import numpy as np

from index.ann import SearchNeighbor
from index.kernels import Kernel


class SearchGraph(ABC):
    def __init__(self,  kernel: Kernel,
                 max_out_degree: Optional[int] = None):
        self.max_out_degree = max_out_degree

        self.kernel = kernel
        self.graph = None
        self.X = None
        self.entrypoint = None

    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def fit(self, X: np.ndarray):
        ...

    def _check(self, X: np.ndarray):
        if (self.max_out_degree is not None
                and not 0 < self.max_out_degree < len(X)):
            raise ValueError('We require 0 < max_out_degree < len(X)')

    def entrypoint_from_centroid(self):
        mu = np.mean(self.X, axis=0)
        K_mu = self.kernel.build_kernel2(mu, self.X)
        self.entrypoint = int(np.argmax(K_mu))

    def search(self, query: np.ndarray, k: int, overquery: float,
               entrypoint: Optional[int] = None, return_stats: bool = False):
        if overquery < 1:
            raise ValueError("Overquery must be greater than or equal to 1")

        queue_size = np.ceil(overquery * k).astype(int)
        queue_size = np.minimum(queue_size, self.graph.number_of_nodes())

        if entrypoint is None:
            init_node = self.entrypoint
        else:
            init_node = entrypoint

        K_current_node = self.kernel.build_kernel2(query, self.X[init_node])
        candidates = [SearchNeighbor(init_node, K_current_node[0, 0])]
        nearest_neighs = []
        visited = {init_node}
        expanded = []

        while candidates:
            current_sneigh = candidates[0]

            if (len(nearest_neighs) >= queue_size
                    and current_sneigh.score < nearest_neighs[-1].score):
                break

            expanded.append(int(current_sneigh.id))

            candidates.pop(0)
            nearest_neighs.append(current_sneigh)
            nearest_neighs = nlargest(queue_size, nearest_neighs,
                                      key=lambda x: x.score)

            neighs = [e[1] for e in self.graph.edges(current_sneigh.id)]
            neighs = [sn for sn in neighs if sn not in visited]

            visited.update(neighs)

            K = self.kernel.build_kernel2(query, self.X[neighs])

            new_candidates = [SearchNeighbor(neigh, K[0, i])
                              for i, neigh in enumerate(neighs)]

            candidates.extend(new_candidates)
            candidates = nlargest(queue_size, candidates, key=lambda x: x.score)

        nearest_neighs = nlargest(k, nearest_neighs, key=lambda x: x.score)
        if return_stats:
            return nearest_neighs, visited, expanded
        else:
            return nearest_neighs

    def greedy_search(self, query: np.ndarray,
                      entrypoint: Optional[int] = None):
        if entrypoint is None:
            current_node = self.entrypoint
        else:
            current_node = entrypoint

        K_current_node = self.kernel.build_kernel2(query, self.X[current_node])[0, 0]

        not_done = True
        while not_done:
            neighs = [e[1] for e in self.graph.edges(current_node)]
            K = self.kernel.build_kernel2(query, self.X[neighs])
            best_neigh = int(np.argmax(K[0]))

            if K[0, best_neigh] > K_current_node:
                current_node = neighs[best_neigh]
                K_current_node = K[0, best_neigh]
            else:
                not_done = False

        return current_node