import itertools
import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots
from scipy.spatial import Delaunay
import shapely

from index import Kernel
from index.optimization import kernel_nnls
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None

def subplot_voronoi_cartoon(fig, subplot: int, even_multiplier: float):
    coords = np.arange(3) - 1
    X = np.stack(np.meshgrid(coords, coords)).T.reshape(-1, 2).astype(float)
    X_even = X[[0, 2, 8, 6]] * even_multiplier
    X_odd = X[[1, 3, 7, 5]]
    X = np.vstack((X_even, X_odd, X[4]))
    idx = len(X) - 1

    # add Voronoi trace
    voronoi = shapely.voronoi_polygons(
        shapely.MultiPoint(X),
        only_edges=True
    )

    for ii, line in enumerate(voronoi.geoms):
        fig.add_trace(
            go.Scatter(
                name='Voronoi cell',
                x=np.array(line.xy[0]),
                y=np.array(line.xy[1]),
                mode='lines',
                line=dict(color='#377eb8', width=5),
                showlegend=ii == 0 and subplot == 1),
            1, subplot
        )

    # add unit circle trace
    center = shapely.Point(X[idx, 0], X[idx, 1])
    circle = center.buffer(1, resolution=360)

    fig.add_trace(
        go.Scatter(
            name='Unit circle',
            x=np.array(circle.exterior.xy[0]),
            y=np.array(circle.exterior.xy[1]),
            mode='lines',
            line=dict(color='#4daf4a', dash='dot', width=5),
            showlegend=subplot == 1),
        1, subplot
    )

    # add edges traces
    kernel = Kernel(1.5)
    K = kernel.build_kernel(X)
    s = kernel_nnls(K, zero_dim=idx)
    s[s < 1e-9] = 0
    s /= s.sum()

    for ii, k in enumerate(np.nonzero(s)[0]):
        fig.add_trace(
            go.Scatter(
                name=f'Out-edges',
                x=[X[idx, 0], X[k, 0]],
                y=[X[idx, 1], X[k, 1]],
                line=dict(color='#e41a1c', width=5),
                mode='lines',
                showlegend=ii == 0 and subplot == 1,
            ),
            1, subplot
        )

    # add point traces
    fig.add_trace(
        go.Scatter(x=X_even[:, 0],
                   y=X_even[:, 1],
                   marker=dict(size=20, color='black'),
                   mode='markers',
                   showlegend=False, ),
        1, subplot
    )
    fig.add_trace(
        go.Scatter(x=X_odd[:, 0],
                   y=X_odd[:, 1],
                   marker=dict(size=20, color='black'),
                   mode='markers',
                   showlegend=False, ),
        1, subplot
    )
    fig.add_trace(
        go.Scatter(x=[X[idx, 0]],
                   y=[X[idx, 1]],
                   marker=dict(size=25, color='#e41a1c'),
                   mode='markers',
                   hoverinfo='text',
                   showlegend=False),
        1, subplot
    )


def delaunay_subset_cartoon():
    fig = plotly.subplots.make_subplots(rows=1, cols=3,
                                        shared_xaxes=True,
                                        shared_yaxes=True)
    subplot_voronoi_cartoon(fig, 1, 0.6)
    subplot_voronoi_cartoon(fig, 2, np.sqrt(2) / 2)
    subplot_voronoi_cartoon(fig, 3, 0.8)

    fig.update_xaxes(range=[-1.05, 1.05], row=1, col=1)
    fig.update_xaxes(range=[-1.05, 1.05], row=1, col=2)
    fig.update_xaxes(range=[-1.05, 1.05], row=1, col=3)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[-1.05, 1.05], row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, range=[-1.05, 1.05], row=1, col=2)
    fig.update_yaxes(scaleanchor="x3", scaleratio=1, range=[-1.05, 1.05], row=1, col=3)
    fig.update_layout(
        height=600,
        width=1500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            indentation=20,
            font=dict(size=40),
        ),
        font=dict(size=30),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, 'delaunay_subset_cartoon.png', scale=2)


def plot_voronoi_delaunay_example():
    rng = np.random.default_rng(0)
    X = rng.random((50, 2))
    # idx = -1

    # add point traces
    points_trace = go.Scatter(
        name='Indexed points',
        x=X[:, 0],
        y=X[:, 1],
           marker=dict(size=12, color='black'),
           mode='markers',
           showlegend=True
    )
    points_trace.text = [n for n in range(len(X))]

    # add Voronoi trace
    voronoi = shapely.voronoi_polygons(
        shapely.MultiPoint(X),
        only_edges=True
    )

    voronoi_traces = [
        go.Scatter(
            name='Voronoi cell',
            x=np.array(line.xy[0]),
            y=np.array(line.xy[1]),
            mode='lines',
            line=dict(color='#1b9e77', width=3),
            showlegend=ii == 0,
        )
        for ii, line in enumerate(voronoi.geoms)
    ]

    # add Delaunay graph trace
    triangulation = Delaunay(X)

    edge_traces = []
    for triangle in triangulation.simplices:
        for a, b in itertools.combinations(triangle, 2):
            if a == -1 or b == -1:
                continue

            edge_traces.append(go.Scatter(
                name='Delaunay graph',
                x=[X[a, 0], X[b, 0]],
                y=[X[a, 1], X[b, 1]],
                line=dict(color='#d95f02', width=3),
                mode='lines',
                showlegend=len(edge_traces) == 0,
            ))

    fig = go.Figure(data=voronoi_traces + edge_traces + [points_trace])
    fig.update_layout(
        height=600,
        width=800,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05], scaleanchor='x', scaleratio=1),
        font=dict(size=20),
    )
    fig.show()
    write_image(fig, 'voronoi_delaunay_example.png', scale=2)


if __name__ == '__main__':
    delaunay_subset_cartoon()
    plot_voronoi_delaunay_example()
