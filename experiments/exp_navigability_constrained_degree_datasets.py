import itertools

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
from index import MRNG, SVG, Kernel, Vamana
from plot_utils import write_image

pio.templates.default = "plotly_white"


def run_example(datasets_dirname, dataset_name, max_out_degree):
    dataset = datasets.select_dataset(dataset_name, dirname=datasets_dirname)
    X = dataset.X_db[:10_000]

    X /= np.linalg.norm(X, axis=1).mean()

    sigmas = np.arange(0.5, 2.75, 0.25)

    filename = f'exp_navigability_degree{max_out_degree}_{dataset_name}.pickle'

    if os.path.exists(filename):
        df = pd.read_pickle(filename)
    else:
        records = []

        indices = [
            MRNG(n_candidates=max_out_degree * 2,
                 max_out_degree=max_out_degree),
            MRNG(n_candidates=max_out_degree * 4,
                 max_out_degree=max_out_degree),
            Vamana(n_candidates=max_out_degree * 4,
                   max_out_degree=max_out_degree,
                   alpha_sequence=[1, 1.2]),
        ]
        indices += [SVG(Kernel(sigma=float(sigma), similarity='euclidean'),
                        max_out_degree=max_out_degree, outer_l0_iterations=4)
                    for sigma in sigmas]

        for index in indices:
            graph_name = index.name()
            if (hasattr(index, 'n_candidates')
                    and index.n_candidates is not None):
                r = index.n_candidates // index.max_out_degree
                graph_name += f' r={r}'
            print(graph_name)

            tic = timeit.default_timer()
            index.fit(X)
            toc = timeit.default_timer()
            print(f'\tcreated in {toc - tic} seconds')

            print(f'\tgraph with '
                  f'{index.graph.number_of_edges() / len(X)} edges')

            tic = timeit.default_timer()

            for overquery in [1, 2, 5, 10]:
                n_searches = 0
                matches = 0

                for i, query in enumerate(X):
                    search_neighs = index.search(
                        query, k=1, overquery=overquery
                    )
                    nneighs = [int(sn.id) for sn in search_neighs]
                    matches += i == nneighs[0]
                    n_searches += 1

                toc = timeit.default_timer()
                print(f'\tsearched in {toc - tic} seconds')
                print(f'\tmatches={matches}, match ratio={matches / n_searches}')

                rec = dict(graph=graph_name,
                           overquery=overquery,
                           navigable_ratio=matches / n_searches)
                if hasattr(index, 'kernel'):
                    rec['sigma'] = index.kernel.sigma
                records.append(rec)
                print(f'\t{rec}')

        df = pd.DataFrame.from_records(records)
        df.to_pickle(filename)


    unique_graph_names = df['graph'].unique()
    palette = plotly.colors.qualitative.Set1[:len(unique_graph_names)][::-1]
    palette = [palette[i] for i in [1, 0, 2, 3]]

    min_sigma = np.nanmin(df['sigma'])
    max_sigma = np.nanmax(df['sigma'])
    overqueries = df['overquery'].unique()

    fig = make_subplots(
        rows=1, cols=overqueries.size,
        subplot_titles=[f'Backtracking={overquery}'
                        for overquery in overqueries],
    )

    for i_overquery, overquery in enumerate(overqueries):
        for graph, color in zip(unique_graph_names, palette):
            df_temp = df[(df['graph'] == graph)
                         & (df['overquery'] == overquery)]

            if 'Vamana' in graph:
                dash = 'dot'
            elif 'MRNG' in graph:
                dash = 'dash'
            else:
                dash = 'solid'

            if np.any(np.isnan(df_temp['sigma'])):
                fig.add_trace(
                    go.Scatter(name=graph,
                               x=[min_sigma, max_sigma],
                               y=df_temp['navigable_ratio'].tolist() * 2,
                               line=dict(color=color, dash=dash, width=3),
                               showlegend=i_overquery == 0,
                               mode='lines', ),
                    row=1, col=i_overquery + 1
                )
            else:
                fig.add_trace(
                    go.Scatter(name=graph,
                               x=df_temp['sigma'],
                               y=df_temp['navigable_ratio'],
                               line=dict(color=color, width=3),
                               showlegend=i_overquery == 0,
                               mode='markers+lines',),
                    row=1, col=i_overquery + 1
                )

    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    for i_overquery in range(overqueries.size):
        fig.update_yaxes(range=[0, 1.02], row=1, col=i_overquery + 1)
        fig.update_xaxes(title=u'\u03C3', row=1, col=i_overquery + 1)


    fig.update_annotations(font_size=18)
    fig.update_layout(
        legend=dict(
            orientation='h',  # Set to "h" for horizontal orientation
            yanchor='bottom',
            # Anchor the legend to the bottom of its container
            y=-0.7,  # Position the legend at the very bottom (y=0)
            xanchor='center',  # Anchor the legend horizontally in the center
            x=0.5,  # Position the legend horizontally in the middle (x=0.5)
            entrywidth=170,
        ),
        height=300,
        width=1200,
        font=dict(size=18),
        boxmode="group",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    fig.show()
    write_image(fig,
                f'exp_navigability_degree{max_out_degree}_{dataset_name}.png',
                scale=3)


def plot_examples(dataset_names, max_out_degrees):
    n_datasets = len(dataset_names)
    n_max_out_degrees = len(max_out_degrees)

    fig = make_subplots(
        rows=n_datasets, cols=n_max_out_degrees,
        # subplot_titles=[f'{dataset}  degree={max_degree}'
        #                 for dataset, max_degree in itertools.product(dataset_names, max_out_degrees)],
        row_titles=dataset_names,
        column_titles=[f'degree={d}' for d in max_out_degrees],
        vertical_spacing=0.1
    )

    for i_dataset, i_max_degree in itertools.product(
            range(n_datasets), range(n_max_out_degrees)
    ):
        filename = (f'exp_navigability'
                    f'_degree{max_out_degrees[i_max_degree]}'
                    f'_{dataset_names[i_dataset]}.pickle')
        df = pd.read_pickle(filename)

        df_temp = df.groupby(['graph', 'overquery'])['navigable_ratio'].max()
        df_temp = df_temp.reset_index()
        df_temp = df_temp.astype({'overquery': str})

        unique_graph_names = list(df_temp['graph'].unique())
        svg_graph_name = [name for name in unique_graph_names if 'SVG' in name][0]
        unique_graph_names.remove(svg_graph_name)
        unique_graph_names.append(svg_graph_name)

        palette = plotly.colors.qualitative.Set1
        palette = palette[:len(unique_graph_names)][::-1]

        for i_graph, graph_name in enumerate(unique_graph_names):
            fig.add_trace(
                go.Bar(name=graph_name,
                       x=df_temp[df_temp['graph'] == graph_name]['overquery'],
                       y=df_temp[df_temp['graph'] == graph_name]['navigable_ratio'],
                       marker_color=palette[i_graph],
                       showlegend=i_dataset == 0 and i_max_degree == 0,
                       ),
                row=i_dataset + 1, col=i_max_degree + 1
            )

    for i_dataset, i_max_degree in itertools.product(
            range(n_datasets),
            range(n_max_out_degrees)
    ):
        fig.update_yaxes(title='recall@1',
                         row=i_dataset + 1,
                         col=i_max_degree + 1)
        fig.update_xaxes(title='backtracking',
                         row=i_dataset + 1,
                         col=i_max_degree + 1)

    fig.update_annotations(font_size=20)
    fig.for_each_annotation(
        lambda a: a.update(y=-0.2) if a.text in max_out_degrees else a.update(
            x=-0.12, textangle=270) if a.text in dataset_names else ())

    fig.update_layout(
        barmode='group',
        # showlegend=False,
        height=1600,
        width=1600,
        font=dict(size=18),
        boxmode="group",
        margin={"l": 150, "r": 0, "t": 30, "b": 0},
    )

    fig.show()
    write_image(fig, f'exp_navigability_datasets_backtracking.png', scale=3)


def main():
    if len(sys.argv) != 2:
        datasets_dirname = './datasets'
    else:
        datasets_dirname = sys.argv[1]

    dataset_names = ['colbert-1M', 'cohere-english-v3-100k', 'e5-large-v2-100k', 'ada002-100k', 'openai-v3-small-100k', 'openai-v3-large-3072-100k']
    max_out_degrees = [8, 16, 32]

    for max_degree in max_out_degrees:
        for dataset_name in dataset_names:
            run_example(datasets_dirname, dataset_name, max_degree)

    plot_examples(dataset_names, max_out_degrees)

if __name__ == '__main__':
    main()
