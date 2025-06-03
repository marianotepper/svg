import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio

from index import MRNG, SVG, Kernel

from plot_utils import plot_graph, plot_lunes

if __name__ == '__main__':
    pio.templates.default = "plotly_white"
    pio.kaleido.scope.mathjax = None

    rng = np.random.default_rng(10)

    X = rng.random(size=(200, 2))

    kernel = Kernel(sigma=0.2, similarity='euclidean')

    # index = MRNG(kernel, max_out_degree=None)
    index = SVG(kernel, max_out_degree=None)

    index.fit(X)

    # query = np.array([0.5, 0.5])
    # query = X[55]
    # for idx in [55, 96]:
    for idx in [11]:
        query = X[idx]

        # search_neighs = index.search(query, k=1, overquery=3)
        search_neighs, visited, expanded = index.search(query, k=1, overquery=1,
                                                        return_stats=True)
        nneighs = [sn.id for sn in search_neighs]
        print(expanded)
        print(nneighs)

        from networkx.algorithms.shortest_paths import dijkstra_path
        print('dijkstra_path', dijkstra_path(index.graph, 34, 11))

        fig = plot_graph(index.graph, X, node_color='black')
        fig.add_trace(
            go.Scatter(name='Search path',
                       x=X[expanded, 0],
                       y=X[expanded, 1],
                       marker=dict(color=np.arange(len(expanded)),
                                   line=dict(color='black', width=2),
                                   colorscale=plotly.colors.sequential.Pinkyl,
                                   showscale=True,
                                   colorbar=dict(
                                       len=0.5,
                                       orientation='h',
                                       title=dict(text="Search progress")
                                   ),
                                   size=15),
                       mode='markers')
        )
        fig.add_trace(
            go.Scatter(name='Entrypoint',
                       x=[X[expanded[0], 0]],
                       y=[X[expanded[0], 1]],
                       marker=dict(color='#e7298a', symbol='circle-open-dot',
                                   size=30, line=dict(width=2)),
                       mode='markers')
        )
        fig.add_trace(
            go.Scatter(name='query',
                       x=[query[0]],
                       y=[query[1]],
                       marker=dict(color='#1b9e77', symbol='circle-open-dot',
                                   size=30, line=dict(width=2)),
                       mode='markers')
        )
        fig.show()