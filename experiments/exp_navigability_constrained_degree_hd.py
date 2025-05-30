import numpy as np
import os
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys
import timeit

import datasets
from index import MRNG, SVG, Kernel
from plot_utils import write_image


def run_example(dirname, dataset_name):
    dataset = datasets.select_dataset(dirname, dataset_name)
    X = dataset.X_db[:1_000]

    pio.templates.default = "plotly_white"
    pio.kaleido.scope.mathjax = None

    sigmas = np.arange(0.5, 4, 0.25)
    max_out_degree = 16

    filename = f'exp_navigability_constrained_degree_{dataset_name}.pickle'

    if os.path.exists(filename):
        df = pd.read_pickle(filename)
    else:
        records = []

        for sigma in sigmas:
            kernel = Kernel(sigma=sigma, similarity='euclidean')

            for index in [
                MRNG(kernel,
                     n_candidates=max_out_degree * 2,
                     max_out_degree=max_out_degree),
                MRNG(kernel,
                     n_candidates=max_out_degree * 4,
                     max_out_degree=max_out_degree),
                MRNG(kernel,
                     n_candidates=max_out_degree * 8,
                     max_out_degree=max_out_degree),
                SVG(kernel, max_out_degree=max_out_degree)
            ]:

                tic = timeit.default_timer()
                index.fit(X)
                toc = timeit.default_timer()
                print(f'created in {toc - tic} seconds')

                print(f'Graph with '
                      f'{index.graph.number_of_edges() / len(X)} edges')

                tic = timeit.default_timer()

                for overquery in [1, 2]:
                    n_searches = 0
                    matches = 0

                    for i, query in enumerate(X):
                        search_neighs = index.search(
                            query, k=1, overquery=overquery
                        )
                        nneighs = [sn.id for sn in search_neighs]
                        matches += i == nneighs[0]
                        n_searches += 1

                    toc = timeit.default_timer()
                    print(f'searched in {toc - tic} seconds')
                    print(matches, matches / n_searches)

                    graph_name = index.name()
                    if (hasattr(index, 'n_candidates')
                            and index.n_candidates is not None):
                        r = index.n_candidates // index.max_out_degree
                        graph_name += f' (r={r})'

                    records.append(
                        dict(sigma=sigma,
                             graph=graph_name,
                             overquery=overquery,
                             navigable_ratio=matches / n_searches)
                    )
                    print(records[-1])

            df = pd.DataFrame.from_records(records)
            df.to_pickle(filename)


    unique_graph_names = df['graph'].unique()
    palette = plotly.colors.qualitative.Set1[:len(unique_graph_names)][::-1]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Backtracking=1', 'Backtracking=2'],
        shared_yaxes=True,
    )

    for i_overquery, overquery in enumerate([1, 2]):
        for graph, color in zip(unique_graph_names, palette):
            df_temp = df[(df['graph'] == graph)
                         & (df['overquery'] == overquery)]

            if 'SVG' in graph:
                mode = 'markers+lines'
            else:
                mode = 'lines'

            fig.add_trace(
                go.Scatter(name=graph,
                           x=df_temp['sigma'],
                           y=df_temp['navigable_ratio'],
                           line=dict(color=color, width=3),
                           showlegend=i_overquery == 0,
                           mode=mode,),
                row=1, col=i_overquery + 1
            )

    fig.update_yaxes(title='recall@1', row=1, col=1)
    fig.update_xaxes(title=u'\u03C3', row=1, col=1)
    fig.update_xaxes(title=u'\u03C3', row=1, col=2)

    fig.update_annotations(font_size=25)
    fig.update_layout(
        height=400,
        width=1200,
        font=dict(size=25),
        boxmode="group",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    fig.show()
    write_image(fig,
                f'exp_navigability_constrained_degree_{dataset_name}.png',
                scale=3)


def main():
    dirname = sys.argv[1]
    run_example(dirname, 'colbert-1M')
    run_example(dirname, 'cohere-english-v3-100k')


if __name__ == '__main__':
    main()
