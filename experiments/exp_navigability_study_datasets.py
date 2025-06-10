import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import sys
import timeit

import datasets
from index import SVG, Kernel
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def run_example(dirname, dataset_name):
    dataset = datasets.select_dataset(dirname, dataset_name)
    X = dataset.X_db[:1_000]

    sigma = 1

    filename = f'exp_quasi_navigability_study_{dataset_name}.pickle'

    if not os.path.exists(filename):


        kernel = Kernel(sigma=sigma, similarity='euclidean')

        index = SVG(kernel, max_out_degree=None)

        tic = timeit.default_timer()
        index.fit(X)
        toc = timeit.default_timer()
        print(f'created in {toc - tic} seconds')
        print(f'Graph with '
              f'{index.graph.number_of_edges() / len(X)} edges')

        records = []
        for s_sum in index.stats_s_sum:
            records.append(
                dict(s_sum=s_sum)
            )

        df = pd.DataFrame.from_records(records)
        print(df)
        df.to_pickle(filename)


def plot_examples(dataset_names):
    traces = []

    for i_dataset, dataset_name in enumerate(dataset_names):
        filename = (f'exp_quasi_navigability_study_{dataset_name}.pickle')
        df = pd.read_pickle(filename)

        traces.append(
            go.Box(name=dataset_name,
                   y=np.maximum(df['s_sum'], 1 ) - np.ones((len(df),)),
                   showlegend=False),
        )

    fig = go.Figure(data=traces)
    fig.update_yaxes(title=u'\u03F5<sub><i>i</i></sub>')
    fig.update_annotations(font_size=20)
    fig.update_layout(
        height=400,
        width=800,
        font=dict(size=20),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    fig.show()
    write_image(fig,
                f'exp_quasi_navigability_study_datasets.png',
                scale=3)


def main():
    dirname = sys.argv[1]
    dataset_names = ['colbert-1M', 'cohere-english-v3-100k', 'openai-v3-small-100k']

    for dataset_name in dataset_names:
        run_example(dirname, dataset_name)

    plot_examples(dataset_names)

if __name__ == '__main__':
    main()
