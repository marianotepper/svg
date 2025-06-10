from multiprocessing import Pool
import networkx as nx
import numpy as np
from typing import Optional, Union

from index.search import SearchGraph
from index.optimization import kernel_nnls, kernel_nnls_l0
from index.kernels import Kernel


class SVG(SearchGraph):
    def __init__(self, kernel: Kernel,
                 max_out_degree: Optional[int] = None):
        super().__init__(kernel, max_out_degree)
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
                NeighborhoodBuilder(K, self.max_out_degree),
                range(len(X))
            )

        self.stats_s_sum = []
        for sublist, s in all_neighbors:
            self.graph.add_edges_from(sublist)
            self.stats_s_sum.append(s.sum())

        self.entrypoint_from_centroid()


class NeighborhoodBuilder:
    def __init__(self, K, max_out_degree):
        self.K = K
        self.max_out_degree = max_out_degree

    def __call__(self, idx):
        return build_neighborhood(self.K, idx, self.max_out_degree,
                                  return_edges=True)


def build_neighborhood(K: np.ndarray, idx: int,
                       max_out_degree: Optional[int] = None,
                       return_edges: bool = False) -> tuple[list[int], np.ndarray]:
    if max_out_degree is None:
        s = kernel_nnls(K, zero_dim=idx)
    else:
        s = kernel_nnls_l0(K, zero_dim=idx, nonzeros=max_out_degree)

    s[s < s.max() * 1e-4] = 0

    neighbors = [i for i in np.argsort(s)[::-1] if s[i] > 0]

    if max_out_degree is not None and max_out_degree < len(neighbors):
        neighbors = neighbors[:max_out_degree]

    if return_edges:
        neighbors = [(idx, n) for n in neighbors]

    return neighbors, s
