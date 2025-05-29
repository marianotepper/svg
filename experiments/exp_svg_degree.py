import itertools
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import Delaunay

from index import SVG, Kernel
from plot_utils import write_image


def main():
    pio.templates.default = "plotly_white"
    pio.kaleido.scope.mathjax = None

    configs = [
        dict(dims=2, sigmas=[0.1]),#, 0.2, 0.3, 0.4, 0.5]),
        dict(dims=3, sigmas=[0.15]),  # , 0.2, 0.3, 0.4, 0.5]),
        dict(dims=4, sigmas=[0.20]),  # , 0.2, 0.3, 0.4, 0.5]),
        dict(dims=5, sigmas=[0.25]),#, 0.5, 0.6, 0.7, 0.8]),
        dict(dims=6, sigmas=[0.30]),  # , 0.5, 0.6, 0.7, 0.8]),
        dict(dims=7, sigmas=[0.35]),  # , 0.5, 0.6, 0.7, 0.8]),
        dict(dims=8, sigmas=[0.45]),  # , 0.5, 0.6, 0.7, 0.8]),
        dict(dims=9, sigmas=[0.50]),  # , 0.5, 0.6, 0.7, 0.8]),
        dict(dims=10, sigmas=[0.55]),#, 0.75, 0.80, 0.85, 0.90]),
    ]

    filename = 'exp_svg_degree.pickle'

    if os.path.exists(filename):
        df = pd.read_pickle(filename)
    else:
        records = []

        for i_config, config in enumerate(configs):
            for seed in range(10):
                rng = np.random.default_rng(seed)

                X = rng.random(size=(100, config['dims']))

                for sigma in config['sigmas']:
                    kernel = Kernel(sigma=sigma, similarity='euclidean')

                    # index = MRNG(kernel, max_out_degree=None)
                    index = SVG(kernel, max_out_degree=None)
                    index.fit(X)

                    svg_avg_degree = index.graph.number_of_edges() / len(X)
                    print(f'SVG AVG degree: {svg_avg_degree}')

                    triangulation = Delaunay(X)
                    delaunay_graph = nx.Graph()
                    delaunay_graph.add_nodes_from(range(len(X)))
                    edges = [(a, b) for triangle in triangulation.simplices
                             for a, b in itertools.combinations(triangle, 2)
                             if a != -1 or b != -1]
                    delaunay_graph.add_edges_from(edges)
                    delaunay_avg_degree = delaunay_graph.number_of_edges() / len(X)
                    print(f'Delaunay AVG degree: {delaunay_avg_degree}')

                    records.extend([
                        dict(seed=seed,
                             sigma=sigma,
                             avg_degree=delaunay_avg_degree,
                             graph='Delaunay',
                             dims=config['dims']),
                        dict(seed=seed,
                             sigma=sigma,
                             avg_degree=svg_avg_degree,
                             graph='SVG',
                             dims=config['dims'])
                    ])
                    print(records[-1])

        df = pd.DataFrame.from_records(records)
        df.to_pickle(filename)

    avg_df = pd.DataFrame({'avg_degree': df.groupby(['dims', 'graph'])[
        'avg_degree'].mean()}).reset_index()
    std_df = pd.DataFrame({'std_degree': df.groupby(['dims', 'graph'])[
        'avg_degree'].std()}).reset_index()

    traces = []
    for graph in ['Delaunay', 'SVG']:
        avg_df_temp = avg_df[avg_df['graph'] == graph]
        std_df_temp = std_df[std_df['graph'] == graph]
        traces.append(
            go.Scatter(name=graph,
                       x=avg_df_temp['dims'],
                       y=avg_df_temp['avg_degree'],
                       error_y=dict(
                           type='data',
                           array=std_df_temp['std_degree'],
                           visible=True),
                       mode='markers+lines',)
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=400,
        width=800,
        font=dict(size=30),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    fig.show()
    write_image(fig, 'svg_degree.pdf', scale=3)


if __name__ == '__main__':
    main()