from typing import Optional, Callable

import networkx as nx
import numpy as np

from svg.search import SearchGraph


class MRNG(SearchGraph):
    def __init__(self, kernel_fun, n_candidates: Optional[int] = None):
        super().__init__(kernel_fun)
        self.n_candidates = n_candidates

    def fit(self, X: np.ndarray):
        if self.n_candidates is not None and self.n_candidates > len(X):
            raise ValueError("n_candidates must be less than to len(X)")

        self.X = X
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(X)))

        for idx in range(len(X)):
            neighbors = build_neighborhood(X, idx, self.kernel_fun,
                                           self.n_candidates)
            neighbors = [(idx, n) for n in neighbors]
            self.graph.add_edges_from(neighbors)

        self.entrypoint_from_centroid()


def build_neighborhood(X: np.ndarray, idx: int, kernel_fun,
                       n_candidates: Optional[int] = None) -> list[int]:
    if n_candidates is not None and n_candidates > len(X):
        raise ValueError("n_candidates must be less than to len(X)")

    K_idx = kernel_fun(X[idx], X)

    candidates = np.argsort(K_idx)
    if n_candidates is not None:
        candidates = candidates[-n_candidates - 1:-2]
    else:
        candidates = candidates[:-1]
    candidates = list(candidates)

    neighbors = []

    while candidates:
        i = candidates[-1]
        candidates.pop()
        neighbors.append(int(i))

        candidates_temp = list(candidates)
        for n in candidates:
            if kernel_fun(X[n], X[i]) >= K_idx[n]:
                candidates_temp.remove(n)

        candidates = candidates_temp

    return neighbors