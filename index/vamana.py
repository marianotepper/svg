from heapq import nsmallest
from multiprocessing import Pool
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Union

from index.search import SearchGraph


class Vamana(SearchGraph):
    def __init__(self,
                 n_candidates: Optional[int] = None,
                 max_out_degree: Optional[int] = None,
                 alpha_sequence: list[float] = [1, 1.2],):
        super().__init__(max_out_degree)
        self.n_candidates = n_candidates
        self.alpha_sequence = alpha_sequence

    def name(self):
        return f'Vamana'

    def _check(self, X: np.ndarray):
        super()._check(X)
        if (self.n_candidates is not None
                and not 0 < self.n_candidates < len(X)):
            raise ValueError('We require 0 < n_candidates < len(X)')
        for alpha in self.alpha_sequence:
            if alpha < 0:
                raise ValueError('We require alpha >= 0')

    def fit(self, X: np.ndarray):
        self._check(X)

        self.X = X
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(X)))

        with Pool() as p:
            all_neighbors = p.map(
                NeighborhoodBuilder(X, self.n_candidates,
                                    self.max_out_degree, self.alpha_sequence),
                range(len(X))
            )

        for sublist in all_neighbors:
            self.graph.add_edges_from(sublist)

        self.entrypoint_from_centroid()

    def entrypoint_from_centroid(self):
        mu = np.mean(self.X, axis=0, keepdims=True)
        K_mu = cdist(mu, self.X)[0]
        self.entrypoint = int(np.argmin(K_mu))

    def score_neighbors(self, query: np.ndarray, neighbors: list[int]):
        return cdist(query[np.newaxis, :], self.X[neighbors])[0]

    def keep_top_candidates(self, size, candidates):
        return nsmallest(size, candidates, key=lambda x: x.score)

    def stop_condition(self, queue_size, nearest_neighs, current_sneigh):
        return (len(nearest_neighs) >= queue_size
                and current_sneigh.score > nearest_neighs[-1].score)

class NeighborhoodBuilder:
    def __init__(self, X, n_candidates, max_out_degree, alpha_sequence):
        self.X = X
        self.n_candidates = n_candidates
        self.max_out_degree = max_out_degree
        self.alpha_sequence = alpha_sequence

    def __call__(self, idx):
        return build_neighborhood(self.X, idx,
                                  self.n_candidates,
                                  self.max_out_degree,
                                  self.alpha_sequence,
                                  return_edges=True)


def build_neighborhood(X: np.ndarray, idx: int,
                       n_candidates: Optional[int] = None,
                       max_out_degree: Optional[int] = None,
                       alpha_sequence: list[float] = [1, 1.2],
                       return_edges: bool = False) -> Union[list[int], list[tuple]]:
    K_idx = cdist(X[idx][np.newaxis, :], X)

    original_candidates = np.argsort(K_idx[0])
    if n_candidates is not None:
        original_candidates = original_candidates[1:n_candidates + 1]
    else:
        original_candidates = original_candidates[1:]

    neighbors = []

    for alpha_stages in alpha_sequence:
        candidates = list(original_candidates)

        while candidates:
            if max_out_degree is not None and len(neighbors) == max_out_degree:
                break

            i = candidates[0]
            candidates.pop(0)
            neighbors.append(int(i))

            candidates_temp = list(candidates)
            for n in candidates:
                if np.sum((X[n] - X[i]) ** 2) * alpha_stages <= np.sum((X[n] - X[idx]) ** 2):
                    candidates_temp.remove(n)

            candidates = candidates_temp

    if return_edges:
        neighbors = [(idx, n) for n in neighbors]

    return neighbors