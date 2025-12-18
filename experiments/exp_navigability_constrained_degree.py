import numpy as np
import os
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import timeit

from index import MRNG, SVG, Kernel, Vamana
from plot_utils import write_image


def main():
    pio.templates.default = "plotly_white"

    configs = [
        dict(dims=5, sigma=0.5, max_out_degree=5),
        dict(dims=10, sigma=0.8, max_out_degree=7),
        dict(dims=20, sigma=1.3, max_out_degree=8),
        dict(dims=50, sigma=1.9, max_out_degree=16),
    ]

    # sigmas = np.arange(0.1, 1.7, step=0.1, dtype=float)

    filename = 'exp_navigability_constrained_degree.pickle'

    if os.path.exists(filename):
        df = pd.read_pickle(filename)
    else:
        records = []

        for i_config, config in enumerate(configs):
            for seed in range(10):
                rng = np.random.default_rng(seed)

                X = rng.random(size=(1_000, config['dims']))

                max_out_degree = config['max_out_degree']
                sigma = config['sigma']

                indices = [
                    MRNG(n_candidates=None,
                         max_out_degree=max_out_degree),
                    MRNG(n_candidates=max_out_degree * 2,
                         max_out_degree=max_out_degree),
                    MRNG(n_candidates=max_out_degree * 4,
                         max_out_degree=max_out_degree),
                    Vamana(n_candidates=max_out_degree * 2,
                           max_out_degree=max_out_degree,
                           alpha_sequence=[1, 1.2]),
                    Vamana(n_candidates=max_out_degree * 4,
                           max_out_degree=max_out_degree,
                           alpha_sequence=[1, 1.2]),
                    SVG(Kernel(sigma=sigma, similarity='euclidean'),
                        max_out_degree=max_out_degree)
                ]
                # indices.extend([
                #     SVG(Kernel(sigma=sigma, similarity='euclidean'),
                #         max_out_degree=max_out_degree)
                #     for sigma in sigmas
                # ])

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

                        for entrypoint in range(0, len(X), 10):
                            for i, query in enumerate(X):
                                if i== entrypoint:
                                    continue

                                search_neighs = index.search(
                                    query, k=1, entrypoint=entrypoint,
                                    overquery=overquery
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
                        # if hasattr(index, 'kernel'):
                        #     graph_name += f' (sigma={index.kernel.sigma:.2f})'

                        records.append(
                            dict(seed=seed,
                                 graph=graph_name,
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
    palette = plotly.colors.qualitative.Set1[:4][::-1]
    line_types = [
        dict(dash='solid', color=palette[0]),
        dict(dash='solid', color=palette[1]),
        dict(dash='dash', color=palette[1]),
        dict(dash='solid', color=palette[2]),
        dict(dash='dash', color=palette[2]),
        dict(dash='solid', color=palette[3]),
    ]

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
    write_image(fig, 'navigability_constrained_degree.png', scale=3)


if __name__ == '__main__':
    main()