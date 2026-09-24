import numpy as np
import os
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import timeit

from index import IPNSW, SVG, Kernel
from plot_utils import write_image

"""
Non-Euclidean (max-inner-product) analogue of
exp_navigability_constrained_degree.py.

Reusing that script's design (i.i.d. uniform data, self-as-query-and-target,
Euclidean MRNG/Vamana baselines) verbatim under `similarity='dot_product'`
is unsound in ways that took two iterations to pin down. This version fixes
all of them:

1. GROUND TRUTH. The original script queries with X[i] and checks whether
   search returns index i, which is valid only because a point is always
   its own Euclidean nearest neighbor (distance 0 is the unique minimum).
   That is not true for inner product: x.x need not be the max of x.y over
   the dataset (this is the crux of why MIP is a genuinely non-metric
   problem, e.g. Morozov & Babenko 2018). Ground truth here is computed by
   brute force (argmax_{j!=i} <x_i, x_j>) and recall is measured against
   that, not against `i` itself.

2. DATA GENERATOR. Two separate failure modes were found empirically, not
   assumed:
   a) i.i.d. isotropic data (uniform cube or standard Gaussian) makes the
      true (self-excluded) top-1 match degenerate as `dims` grows: pairwise
      dot products of independent random vectors concentrate near 0 while
      self-similarity ||x||^2 grows with dims, so self dominates and the
      true top-1 match becomes statistically indistinguishable from noise.
      A first fix (clusters in the ambient space, fixed cluster/noise scale)
      addressed the low-dims case but reproduces the same failure at high
      dims for a different reason:
   b) with additive per-point noise of fixed scale in the ambient space,
      self-similarity gets a "self-energy" term ||noise_i||^2 that only
      ever appears in self-comparisons, never in cross-comparisons; that
      term grows like dims * noise_scale^2, so self again comes to dominate
      as dims grows. Measured on the fixed-scale cluster generator: median
      self-similarity vs. true-top-1-similarity go from 8.9 vs 22.4
      (dims=5, clearly separated) to 432.0 vs 432.9 (dims=100, statistically
      the same value) -- and this holds for ANY sigma (swept 20-100 at
      dims=100: recall pinned at 0.04-0.04, i.e. this is not a tuning
      problem, the ground truth itself has become degenerate).
   The fix used here: embed a fixed, LOW-dimensional (k=3) cluster signal
   isometrically into the ambient `dims` (so the signal's own scale never
   changes with dims), then add ambient per-point noise whose per-coordinate
   variance shrinks like 1/dims (so each point's total noise energy
   ||noise_i||^2 is dims-invariant, killing the self-energy inflation from
   (b)). Verified: frac(top-1 beats self) stays 0.95-0.99 across dims
   5-100 (vs. collapsing to 0.40-0.06 with the earlier generators), and a
   single sigma now works reasonably across the whole dims sweep (as
   expected, since neither the signal nor the noise scale changes with
   dims by construction) -- this also means any *remaining* recall
   degradation across the dims sweep reflects genuine curse-of-dimensionality
   difficulty for graph construction/search, not a vanishing-ground-truth
   artifact.

3. BASELINES. index/mrng.py and index/vamana.py call scipy.cdist directly
   and prune/search by squared Euclidean distance -- they do not look at
   `Kernel`/`similarity` at all, and their diversification/occlusion rule
   is specific to a metric (it reasons about a "closer" point standing in
   for a "farther" one along a route, which needs a triangle-inequality-like
   argument to justify skipping an edge). Two routes were considered:
   - The standard MIPS -> Euclidean reduction (Bachrach et al. 2014,
     padding database vectors with sqrt(M^2-||x||^2) and queries with 0)
     is EXACT for query-vs-database comparisons, but MRNG/Vamana's pruning
     also compares pairs of *database* points to each other during
     construction (every node acts as a "source" internally), and the
     reduction is not designed for that. Checked directly: for a node
     acting as its own source, the padded-space ranking of its top-12
     true inner-product neighbors overlaps the true top-12 in only 1 of 12
     entries. That's not a minor approximation, it corrupts the candidate
     pool outright, which is exactly what produced the collapsed
     MRNG(r=2)/(r=4)/Vamana recall (~0.02-0.05) in the first version of
     this script.
   - Instead, this version uses `index/ip_nsw.py`'s IPNSW class: ip-NSW
     (Morozov & Babenko, 2018) is already implemented natively for
     arbitrary (possibly non-metric) kernels, already handles the "self is
     not necessarily the most similar point" subtlety correctly (see its
     docstring), and is exactly the paper's own reference baseline for the
     non-Euclidean case -- the "undiversified top-M" rule that SVG-L0's
     diversified candidate selection is positioned against. No new code
     needed, and no reduction-approximation risk.

SVG needs none of this scaffolding: it takes `Kernel(similarity='dot_product')`
and runs on the raw vectors directly, exactly like ip-NSW.
"""

INTRINSIC_DIM = 3
N_CLUSTERS = 20
CLUSTER_SCALE = 2.0
NOISE_SCALE = 0.5


def make_clustered_data(rng, n, dims):
    """Cluster signal fixed at intrinsic dimension INTRINSIC_DIM, embedded
    isometrically into the ambient `dims` so its scale never changes with
    dims, plus ambient per-point noise scaled by 1/sqrt(dims) so each
    point's own noise energy (and hence the "self-similarity inflation"
    that would otherwise make a point trivially its own best match) stays
    dims-invariant too. See module docstring point 2 for why this matters.
    """
    A = rng.normal(size=(dims, INTRINSIC_DIM))
    Q, _ = np.linalg.qr(A)  # dims x INTRINSIC_DIM, orthonormal columns
    centers_latent = rng.normal(size=(N_CLUSTERS, INTRINSIC_DIM)) * CLUSTER_SCALE
    centers = centers_latent @ Q.T
    assign = rng.integers(0, N_CLUSTERS, size=n)
    noise = rng.normal(size=(n, dims)) * (NOISE_SCALE / np.sqrt(dims))
    return centers[assign] + noise


def true_top1_inner_product(X):
    """Brute-force ground truth: argmax_{j != i} <x_i, x_j>. See module
    docstring point 1 for why this (and not `i` itself) is the correct
    target under inner-product similarity."""
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    return np.argmax(G, axis=1)


def main():
    pio.templates.default = "plotly_white"

    # A single sigma per dims (not swept per baseline/candidate-pool size,
    # since neither IPNSW nor SVG-L0 need one calibrated differently) --
    # picked via a light plateau search (see the project's calib_mip3.py
    # scratch script). As with the Euclidean experiment, recall is fairly
    # flat over a broad sigma range; these values are close to that
    # plateau for every dims, consistent with the generator being
    # dims-invariant in scale by construction.
    configs = [
        dict(dims=5, sigma=12.0, max_out_degree=50),
        dict(dims=10, sigma=12.0, max_out_degree=60),
        dict(dims=20, sigma=10.0, max_out_degree=70),
        dict(dims=50, sigma=10.0, max_out_degree=85),
        dict(dims=100, sigma=12.0, max_out_degree=105),
    ]

    filename = 'exp_navigability_constrained_degree_mip.pickle'

    if os.path.exists(filename):
        df = pd.read_pickle(filename)
    else:
        records = []

        for i_config, config in enumerate(configs):
            for seed in range(10):
                rng = np.random.default_rng(seed)

                X = make_clustered_data(rng, 1_000, config['dims'])
                true_top1 = true_top1_inner_product(X)

                max_out_degree = config['max_out_degree']
                kernel = Kernel(sigma=config['sigma'], similarity='dot_product')

                indices = [
                    IPNSW(kernel, max_out_degree=max_out_degree),
                    SVG(kernel, max_out_degree=max_out_degree),
                ]

                for index in indices:
                    tic = timeit.default_timer()
                    index.fit(X)
                    toc = timeit.default_timer()
                    print(f'created in {toc - tic} seconds')

                    print(f'Graph with '
                          f'{index.graph.number_of_edges() / len(X)} edges')

                    tic = timeit.default_timer()

                    for overquery in [1, 2, 5]:
                        n_searches = 0
                        matches = 0

                        for entrypoint in range(0, len(X), 100):
                            for i in range(len(X)):
                                if i == entrypoint:
                                    continue

                                search_neighs = index.search(
                                    X[i], k=1, entrypoint=entrypoint,
                                    overquery=overquery
                                )
                                nneighs = [sn.id for sn in search_neighs]
                                matches += true_top1[i] == nneighs[0]
                                n_searches += 1

                        toc = timeit.default_timer()
                        print(f'searched in {toc - tic} seconds')
                        print(matches, matches / n_searches)

                        records.append(
                            dict(seed=seed,
                                 graph=index.name(),
                                 overquery=overquery,
                                 navigable_ratio=matches / n_searches,
                                 dims=config['dims'])
                        )
                        print(records[-1])

        df = pd.DataFrame.from_records(records)
        df.to_pickle(filename)

    groupby = ['dims', 'graph', 'overquery']
    avg_df = pd.DataFrame({
        'avg_navi': df.groupby(groupby)['navigable_ratio'].mean()
    }).reset_index()
    std_df = pd.DataFrame({
        'std_navi': df.groupby(groupby)['navigable_ratio'].std()
    }).reset_index()
    print(avg_df)

    unique_graph_names = df['graph'].unique()
    palette = plotly.colors.qualitative.Set1[:len(unique_graph_names)]
    line_types = [dict(dash='solid', color=c) for c in palette]

    overqueries = df['overquery'].unique()

    fig = make_subplots(
        rows=1, cols=len(overqueries),
        subplot_titles=[f'Backtracking={overquery}'
                        for overquery in overqueries],
    )

    for i_overquery, overquery in enumerate(overqueries):
        for graph, line in zip(unique_graph_names, line_types):
            avg_df_temp = avg_df[(avg_df['graph'] == graph)
                                 & (avg_df['overquery'] == overquery)]
            std_df_temp = std_df[(std_df['graph'] == graph)
                                 & (avg_df['overquery'] == overquery)]
            fig.add_trace(
                go.Scatter(name=graph,
                           x=avg_df_temp['dims'],
                           y=avg_df_temp['avg_navi'],
                           error_y=dict(
                               type='data',
                               array=std_df_temp['std_navi'],
                               thickness=3,
                               visible=True),
                           line=dict(color=line['color'], dash=line['dash'],
                                     width=3),
                           showlegend=i_overquery == 0,
                           mode='lines',),
                row=1, col=i_overquery + 1
            )

    fig.update_yaxes(title=dict(text='recall@1', standoff=30), row=1, col=1)
    for i in range(len(overqueries)):
        fig.update_xaxes(title='Dimensions', tickmode='array',
                         tickvals=df['dims'].unique(),
                         row=1, col=i+1)

    fig.update_annotations(font_size=25)
    fig.update_layout(
        height=400,
        width=1800,
        font=dict(size=25),
        boxmode="group",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    fig.show()
    write_image(fig, 'navigability_constrained_degree_mip.png', scale=3)


if __name__ == '__main__':
    main()
