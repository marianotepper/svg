from multiprocessing import Pool
import networkx as nx
import numpy as np
from typing import Optional, Union

from index.kernels import Kernel
from index.search import SearchGraph


class MRNG(SearchGraph):
    def __init__(self, kernel_fun,
                 n_candidates: Optional[int] = None,
                 max_out_degree: Optional[int] = None):
        super().__init__(kernel_fun, max_out_degree)
        self.n_candidates = n_candidates

    def name(self):
        return 'MRNG'

    def _check(self, X: np.ndarray):
        super()._check(X)
        if (self.n_candidates is not None
                and not 0 < self.n_candidates < len(X)):
            raise ValueError('We require 0 < n_candidates < len(X)')

    def fit(self, X: np.ndarray):
        self._check(X)

        self.X = X
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(X)))

        with Pool() as p:
            all_neighbors = p.map(
                NeighborhoodBuilder(X, self.kernel, self.n_candidates,
                                    self.max_out_degree),
                range(len(X))
            )

        for sublist in all_neighbors:
            self.graph.add_edges_from(sublist)

        self.entrypoint_from_centroid()


class NeighborhoodBuilder:
    def __init__(self, X: Kernel, kernel_fun, n_candidates, max_out_degree):
        self.X = X
        self.kernel_fun = kernel_fun
        self.n_candidates = n_candidates
        self.max_out_degree = max_out_degree

    def __call__(self, idx):
        return build_neighborhood(self.X, idx, self.kernel_fun,
                                  self.n_candidates,
                                  self.max_out_degree,
                                  return_edges=True)


def build_neighborhood(X: np.ndarray, idx: int, kernel: Kernel,
                       n_candidates: Optional[int] = None,
                       max_out_degree: Optional[int] = None,
                       return_edges: bool = False) -> Union[list[int], list[tuple]]:
    K_idx = kernel.build_kernel2(X[idx], X)

    candidates = np.argsort(K_idx[0])
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
            if kernel.build_kernel2(X[n], X[i])[0, 0] >= K_idx[0, n]:
                candidates_temp.remove(n)

        candidates = candidates_temp

    if max_out_degree is not None and max_out_degree < len(neighbors):
        neighbors = neighbors[:max_out_degree]

    if return_edges:
        neighbors = [(idx, n) for n in neighbors]

    return neighbors