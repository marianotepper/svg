import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import timeit

from index import SVG, Kernel
from plot_utils import write_image


def main():
    pio.templates.default = "plotly_white"
    pio.kaleido.scope.mathjax = None

    configs = [
        dict(dims=2, sigmas=[0.1, 0.2, 0.3, 0.4, 0.5]),
        dict(dims=5, sigmas=[0.4, 0.5, 0.6, 0.7, 0.8]),
        dict(dims=10, sigmas=[0.70, 0.75, 0.80, 0.85, 0.90]),
        dict(dims=20, sigmas=[1.0, 1.25, 1.5, 1.75, 2.0]),
        dict(dims=50, sigmas=[2.0, 2.5, 3.0, 3.5, 4.0]),
        dict(dims=100, sigmas=[3.0, 4.0, 5.0, 6.0, 7.0]),
    ]

    filename = 'exp_svg_navigability.pickle'

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
                    tic = timeit.default_timer()
                    index.fit(X)
                    toc = timeit.default_timer()
                    print(f'created in {toc - tic} seconds')

                    print(f'Graph with {index.graph.number_of_edges() / len(X)} edges')

                    tic = timeit.default_timer()

                    for overquery in [1, 2]:
                        n_searches = 0
                        matches = 0

                        for entrypoint in range(len(X)):
                            for i, query in enumerate(X):
                                if i== entrypoint:
                                    continue

                                search_neighs = index.search(query, k=1,
                                                             entrypoint=entrypoint,
                                                             overquery=overquery)
                                nneighs = [sn.id for sn in search_neighs]
                                matches += i == nneighs[0]
                                n_searches += 1
                                # if i != nneighs[0]:
                                #     print(i, nneighs)

                        toc = timeit.default_timer()
                        print(f'searched in {toc - tic} seconds')
                        print(matches, matches / n_searches)

                        records.append(
                            dict(seed=seed,
                                 sigma=sigma,
                                 overquery=overquery,
                                 navigable_ratio=matches / n_searches,
                                 dims=config['dims'])
                        )
                        print(records[-1])

        df = pd.DataFrame.from_records(records)
        df.to_pickle(filename)

    cols = len(configs) // 2
    fig = make_subplots(rows=2, cols=cols,
                        subplot_titles=[f'd={config['dims']}'
                                        for config in configs])

    for i_config, config in enumerate(configs):
        df_temp = df[df['dims'] == config['dims']]

        groupby = ['dims', 'sigma', 'overquery']
        avg_df = pd.DataFrame({
            'avg_navi': df_temp.groupby(groupby)['navigable_ratio'].mean()
        }).reset_index()
        std_df = pd.DataFrame({
            'std_navi': df_temp.groupby(groupby)['navigable_ratio'].std()
        }).reset_index()

        for overquery, color in [(1, '#377eb8'), (2, '#e41a1c')]:
            avg_df_temp = avg_df[avg_df['overquery'] == overquery]
            std_df_temp = std_df[std_df['overquery'] == overquery]
            fig.add_trace(
                go.Scatter(name=overquery,
                           x=avg_df_temp['sigma'],
                           y=avg_df_temp['avg_navi'],
                           error_y=dict(
                               type='data',
                               array=std_df_temp['std_navi'],
                               visible=True),
                           line=dict(color=color, width=2),
                           showlegend=i_config == 0,
                           mode='markers+lines',),
                row=i_config // cols + 1, col=i_config % cols + 1
            )

    fig.update_annotations(font_size=25)
    fig.update_layout(
        height=400,
        width=1600,
        font=dict(size=25),
        boxmode="group",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend_title_text = 'Backtracking'
    )

    fig.show()
    write_image(fig, 'svg_navigability.pdf', scale=3)


if __name__ == '__main__':
    main()