from heapq import nlargest
from multiprocessing import Pool
import networkx as nx
import numpy as np
import timeit
from typing import Optional

from index.search import SearchGraph
from index.optimization import kernel_nnls, kernel_nnls_l0
from index.kernels import Kernel


class SVG(SearchGraph):
    def __init__(self, kernel: Kernel,
                 max_out_degree: Optional[int] = None,
                 outer_l0_iterations: Optional[int] = None):
        if max_out_degree is None and outer_l0_iterations is not None:
            raise ValueError('max_out_degree must be specified if outer_l0_iterations is specified')

        super().__init__(max_out_degree)
        self.kernel = kernel
        self.outer_l0_iterations = outer_l0_iterations
        self.stats_s_sum = None

    def name(self):
        if self.max_out_degree is None:
            return 'SVG'
        else:
            return 'SVG-L0'

    def fit(self, X: np.ndarray):
        self._check(X)

        self.X = X
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(X)))

        K = self.kernel.build_kernel(X)

        with Pool() as p:
            all_neighbors = p.map(
                NeighborhoodBuilder(K, self.max_out_degree,
                                    self.outer_l0_iterations),
                range(len(X))
            )

        self.stats_s_sum = []
        for sublist, s in all_neighbors:
            self.graph.add_edges_from(sublist)
            self.stats_s_sum.append(s.sum())

        self.entrypoint_from_centroid()

    def entrypoint_from_centroid(self):
        mu = np.mean(self.X, axis=0)
        K_mu = self.kernel.build_kernel2(mu, self.X)
        self.entrypoint = int(np.argmax(K_mu))

    def score_neighbors(self, query: np.ndarray, neighbors: list[int]):
        return self.kernel.build_kernel2(query, self.X[neighbors])[0]

    def keep_top_candidates(self, size, candidates):
        return nlargest(size, candidates, key=lambda x: x.score)

    def stop_condition(self, queue_size, nearest_neighs, current_sneigh):
        return (len(nearest_neighs) >= queue_size
                and current_sneigh.score < nearest_neighs[-1].score)


class NeighborhoodBuilder:
    def __init__(self, K, max_out_degree, outer_l0_iterations):
        self.K = K
        self.max_out_degree = max_out_degree
        self.outer_l0_iterations = outer_l0_iterations

    def __call__(self, idx):
        return build_neighborhood(self.K, idx, self.max_out_degree,
                                  self.outer_l0_iterations,
                                  return_edges=True)


def build_neighborhood(K: np.ndarray, idx: int,
                       max_out_degree: Optional[int] = None,
                       outer_l0_iterations: Optional[int] = None,
                       return_edges: bool = False) -> tuple[list[int], np.ndarray]:
    if max_out_degree is None:
        s = kernel_nnls(K, zero_dim=idx)
    else:
        s = kernel_nnls_l0(K, zero_dim=idx, nonzeros=max_out_degree,
                           outer_l0_iterations=outer_l0_iterations)

    s[s < s.max() * 1e-4] = 0

    neighbors = [i for i in np.argsort(s)[::-1] if s[i] > 0]

    if max_out_degree is not None and max_out_degree < len(neighbors):
        neighbors = neighbors[:max_out_degree]

    if return_edges:
        neighbors = [(idx, n) for n in neighbors]

    return neighbors, s
