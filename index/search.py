from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from index.ann import SearchNeighbor
from index.kernels import Kernel


class SearchGraph(ABC):
    def __init__(self, max_out_degree: Optional[int] = None):
        self.max_out_degree = max_out_degree

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

        K_current_node = self.score_neighbors(query, [init_node])
        candidates = [SearchNeighbor(init_node, K_current_node[0])]
        nearest_neighs = []
        visited = {init_node}
        expanded = []

        while candidates:
            current_sneigh = candidates[0]

            if self.stop_condition(queue_size, nearest_neighs, current_sneigh):
                break

            expanded.append(int(current_sneigh.id))

            candidates.pop(0)
            nearest_neighs.append(current_sneigh)
            nearest_neighs = self.keep_top_candidates(queue_size, nearest_neighs)

            neighs = [e[1] for e in self.graph.edges(current_sneigh.id)]
            neighs = [sn for sn in neighs if sn not in visited]

            visited.update(neighs)

            K = self.score_neighbors(query, neighs)

            new_candidates = [SearchNeighbor(neigh, K[i])
                              for i, neigh in enumerate(neighs)]

            candidates.extend(new_candidates)
            candidates = self.keep_top_candidates(queue_size, candidates)

        nearest_neighs = self.keep_top_candidates(k, nearest_neighs)
        if return_stats:
            return nearest_neighs, visited, expanded
        else:
            return nearest_neighs

    @abstractmethod
    def entrypoint_from_centroid(self):
        ...

    @abstractmethod
    def score_neighbors(self, query: np.ndarray, neighbors: list[int]):
        ...

    @abstractmethod
    def keep_top_candidates(self, size, candidates):
        ...

    @abstractmethod
    def stop_condition(self, queue_size, nearest_neighs, current_sneigh):
        ...

    def greedy_search(self, query: np.ndarray,
                      entrypoint: Optional[int] = None):
        return self.search(query, 1, 1, entrypoint=entrypoint,
                           return_stats=False)
