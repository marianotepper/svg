import pickle
import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.subplots
import timeit

# # This fix prevents a plotly error when saving figures
# import kaleido
# kaleido.get_chrome_sync()

from accuracy_metrics import compute_recall
from index import SVG, Kernel, IPNSW
from datasets import vecs_io
from plot_utils import write_image


def read_netflix_dataset():
    print('reading netflix dataset')
    # Netflix dataset downloaded from https://github.com/xinyandai/similarity-search/tree/mipsex/data/netflix
    X_db = vecs_io.fvecs_read('./datasets/netflix/netflix_base.fvecs')
    X_query = vecs_io.fvecs_read('./datasets/netflix/netflix_query.fvecs')
    return X_db, X_query

def read_yahoomusic_dataset():
    print('reading yahoomusic dataset')
    # YahooMusic dataset downloaded from https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/yahoomusic.tar.gz
    X_db = vecs_io.fvecs_read('./datasets/yahoomusic/yahoomusic_base.fvecs')
    X_query = vecs_io.fvecs_read('./datasets/yahoomusic/yahoomusic_query_all.fvecs')
    return X_db, X_query


def compute_measures(read_dataset_function, max_out_degrees, overqueries, result_file_prefix):
    X_db, X_query = read_dataset_function()

    if len(X_db) >  20_0:
        X_db = X_db[:20_000]
    if len(X_query) > 1_000:
        X_query = X_query[:1_000]

    print(X_query.shape)
    print(np.linalg.norm(X_query, axis=1).min(), np.linalg.norm(X_query, axis=1).max())
    print(X_db.shape)
    print(np.linalg.norm(X_db, axis=1).min(), np.linalg.norm(X_db, axis=1).max())

    gt = np.argsort(X_query @ X_db.T, axis=1)[:, ::-1]

    # Compute fraction of self-similar nodes
    self_similar_fraction = np.sum(np.argmax(X_db @ X_db.T, axis=1) == np.arange(len(X_db))) / len(X_db)
    print(self_similar_fraction)

    kernel = Kernel(sigma=10, similarity='dot_product')

    navigable_nodes = {}
    recalls = {}

    for degree in max_out_degrees:
        indices = [SVG(kernel, build_full_kernel=False, max_out_degree=degree),
                   IPNSW(kernel, n_candidates=None, max_out_degree=degree,
                         build_full_kernel=False)]

        for index in indices:
            if index.name() not in navigable_nodes:
                navigable_nodes[index.name()] = {}
            navigable_nodes[index.name()][degree] = dict([(overquery, 0) for overquery in overqueries])

            if index.name() not in recalls:
                recalls[index.name()] = {}
            recalls[index.name()][degree] = dict([(overquery, 0) for overquery in overqueries])

            tic = timeit.default_timer()
            index.fit(X_db)
            toc = timeit.default_timer()
            print("Time", toc - tic)

            for overquery in overqueries:
                for i_query, query in enumerate(X_db):
                    search_neighs = index.search(query, k=1, overquery=overquery, return_stats=False)
                    nneighs = [int(sn.id) for sn in search_neighs]

                    idx_gt = np.argmax(np.sum(query[np.newaxis, :] * X_db, axis=1))
                    navigable_nodes[index.name()][degree][overquery] += float(nneighs[0] == idx_gt) / len(X_db)

                neighbors = []
                for i_query, query in enumerate(X_query):
                    search_neighs = index.search(query, k=10, overquery=overquery, return_stats=False)
                    nneighs = [int(sn.id) for sn in search_neighs]

                    neighbors.append(nneighs)

                neighbors = np.array(neighbors)
                recalls[index.name()][degree][overquery] = compute_recall(gt, neighbors)

        print('navigability:', navigable_nodes)
        print('recalls:', recalls)

    with open(f'./{result_file_prefix}_results.pkl', 'wb') as f:
        pickle.dump(max_out_degrees, f)
        pickle.dump(overqueries, f)
        pickle.dump(recalls, f)
        pickle.dump(navigable_nodes, f)


def plot_measures(result_file_prefix):
    with open(f'./{result_file_prefix}_results.pkl', 'rb') as f:
        max_out_degrees = pickle.load(f)
        overqueries = pickle.load(f)
        recalls = pickle.load(f)
        navigable_nodes = pickle.load(f)
        indices = list(recalls.keys())
        print(indices)
        if indices != list(navigable_nodes.keys()):
            raise RuntimeError('indices and navigable_nodes do not match')

        palette = plotly.colors.qualitative.Plotly
        dashes = ['solid', 'dash', 'dot', 'dashdot']

        fig = plotly.subplots.make_subplots(rows=1, cols=2)

        fig.add_traces([
            go.Scatter(name=degree,
                       x=overqueries, y=[navigable_nodes[index][degree][oq] for oq in overqueries],
                       line=dict(color=palette[j], dash=dashes[i], width=3),
                       mode='lines', showlegend=False)
            for j, degree in enumerate(max_out_degrees)
            for i, index in enumerate(indices)
        ], rows=1, cols=1)

        fig.add_traces([
            go.Scatter(name=degree,
                       x=overqueries, y=[recalls[index][degree][oq] for oq in overqueries],
                       line=dict(color=palette[j], dash=dashes[i], width=3),
                       mode='lines', showlegend=False)
            for j, degree in enumerate(max_out_degrees)
            for i, index in enumerate(indices)
        ], rows=1, cols=2)

        fig.add_traces([
            go.Scatter(name=index,
                       x=[None], y=[None],
                       line=dict(color='black', dash=dashes[i], width=3),
                       mode='lines',
                       legendgroup='Index', legendgrouptitle_text='Index', showlegend=True)
            for i, index in enumerate(indices)
        ])
        fig.add_traces([
            go.Scatter(name=degree,
                       x=[None], y=[None],
                       line=dict(color=palette[j], width=3),
                       mode='lines',
                       legendgroup='degree', legendgrouptitle_text='Maximum out-degree M', showlegend=True)
            for j, degree in enumerate(max_out_degrees)
        ])
        fig.update_layout(
            template="plotly_white",
            height=500, width=1500,
            xaxis1_title='backtracking',
            yaxis1_title='Fraction of navigable nodes',
            xaxis2_title='backtracking',
            yaxis2_title='Recall',
            font=dict(size=18),
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        fig.show()
        write_image(fig, f'{result_file_prefix}_measures.png', scale=2)


if __name__ == '__main__':
    # compute_measures(read_netflix_dataset,
    #                  max_out_degrees=[8, 16, 32, 64],
    #                  overqueries=[1, 2, 5, 10],
    #                  result_file_prefix='netflix')
    # plot_measures(result_file_prefix='netflix')

    compute_measures(read_yahoomusic_dataset,
                     max_out_degrees=[8, 16, 32, 64],
                     overqueries=[1, 2, 5, 10],
                     result_file_prefix='yahooMusic')
    plot_measures(result_file_prefix='yahooMusic')
