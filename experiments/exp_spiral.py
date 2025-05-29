import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots

from index import build_neighborhood_mrng, Kernel
from index.optimization import kernel_nnls, kernel_nnls_l0
from plot_utils import write_image

pio.templates.default = "plotly_white"
# pio.kaleido.scope.mathjax = None


def generate_spiral(n_points, n_turns):
    """
    Generates a 2D spiral dataset.

    Args:
      n_points: The number of points in the spiral.
      n_turns: The number of turns in the spiral.

    Returns:
      A tuple containing two NumPy arrays:
        - x: x-coordinates of the spiral points.
        - y: y-coordinates of the spiral points.
    """
    theta = np.linspace(0, n_turns * 2 * np.pi, n_points)
    r = np.linspace(1, 2, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack((x, y)).T


def generate_edge_traces(X, idx, neighbors, name, color, opacity, width, dash):
    return [
        go.Scatter(
            name=name,
            x=[X[idx, 0], X[neigh, 0]],
            y=[X[idx, 1], X[neigh, 1]],
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            mode='lines',
            # hoverinfo='text',
            showlegend=i == 0,
        )
        for i, neigh in enumerate(neighbors)
    ]


def main():
    X = generate_spiral(30, 1)
    X = np.vstack((np.zeros((1, 2)), X))
    idx = 0

    kernel = Kernel(2)

    mrng_neighbors = build_neighborhood_mrng(X, idx, kernel)
    mrng_edge_traces = generate_edge_traces(X, idx, mrng_neighbors,
                                            'MRNG', '#a6cee3',
                                            1, 7, 'solid')
    mrng_trunc_edge_traces = generate_edge_traces(X, idx, mrng_neighbors[:3],
                                                  'MRNG - truncated', '#1f78b4',
                                                  1, 7, 'dot')

    K = kernel.build_kernel(X)

    s = kernel_nnls(K, zero_dim=idx)
    s[s < 1e-6] = 0
    svg_neighbors = [i for i in np.argsort(s)[::-1] if s[i] > 0]
    svg_edge_traces = generate_edge_traces(X, 0, svg_neighbors, 'SVG',
                                           '#b2df8a', 1, 7, 'solid')

    s = kernel_nnls_l0(K, zero_dim=idx, nonzeros=3)
    s[s < 1e-6] = 0
    print(s)
    svg_neighbors = [i for i in np.argsort(s)[::-1] if s[i] > 0]
    svg_trunc_edge_traces = generate_edge_traces(X, 0, svg_neighbors,
                                                 'SVG-L0', '#33a02c',
                                                 1, 7, 'dot')

    fig = plotly.subplots.make_subplots(rows=1, cols=2,
                                        shared_xaxes=True,
                                        shared_yaxes=True,
                                        subplot_titles=['MRNG', 'SVG'])

    fig.add_traces(mrng_edge_traces + mrng_trunc_edge_traces + [
        go.Scatter(text=r'$\huge{\mathbf{x}_i}$',
                   textposition='bottom right',
                   x=[X[0, 0]], y=[X[0, 1]],
                   marker=dict(size=10, color='black'),
                   showlegend=False,
                   mode='markers+text',),
        go.Scatter(x=X[1:, 0], y=X[1:, 1],
                   marker=dict(size=10, color='black'),
                   showlegend=False,
                   mode='markers',),
    ], rows=1, cols=1)
    fig.add_traces(svg_edge_traces + svg_trunc_edge_traces + [
        go.Scatter(text=r'$\huge{\mathbf{x}_i}$',
                   textposition='bottom right',
                   x=[X[0, 0]], y=[X[0, 1]],
                   marker=dict(size=10, color='black'),
                   showlegend=False,
                   mode='markers+text', ),
        go.Scatter(name='Vectors',
                   x=X[:, 0], y=X[:, 1],
                   marker=dict(size=10, color='black'),
                   mode='markers', ),
    ], rows=1, cols=2)

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_annotations(font_size=30)
    fig.update_layout(
        height=600,
        width=1600,
        font=dict(size=25),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
    )
    fig.show()
    write_image(fig, 'spiral.png', scale=3)

if __name__ == '__main__':
    main()