import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.distance import cdist

from index.qp import qp_simplex
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def svg_cartoon_no_max_degree():
    idx = 4
    sigma = 15

    coord0 = np.array([0, 1, 6, 7])
    coord1 = np.arange(3)
    res = np.meshgrid(coord0, coord1)
    X = np.stack(res).T.reshape(-1, 2).astype(float)
    print(X.shape)

    D = cdist(X, X, metric='sqeuclidean')
    K = np.exp(-D / (sigma ** 2))
    K_idx = K[idx]

    s = qp_simplex(K, K_idx, zero_dim=idx)
    s[s < 1e-6] = 0

    for j in range(len(X)):
        print(j, s[j])
    print(s.sum())

    points_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        marker=dict(size=20, color='black'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    points_trace.text = [n for n in range(len(X))]

    i_trace = go.Scatter(
        x=[X[idx, 0]], y=[X[idx, 1]],
        marker=dict(size=20, color='#e41a1c'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    i_trace.text = [idx]

    edge_traces = []
    opacity = np.log(1 + s)
    opacity[s > 0] += 0.2 - opacity[s > 0].min()
    opacity /= opacity.max()
    print(opacity)
    for k in range(len(X)):
        edge_traces.append(go.Scatter(
            x=[X[idx, 0], X[k, 0]],
            y=[X[idx, 1], X[k, 1]],
            line=dict(color='#4daf4a', width=10),
            opacity=opacity[k],
            mode='lines',
            # hoverinfo='text',
            showlegend=False,
        ))

    traces = []
    traces.extend(edge_traces)
    traces.append(points_trace)
    traces.append(i_trace)
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=300,
        width=800,
        xaxis=dict(
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            tickfont=dict(size=20),
            tickvals=coord1,
            scaleanchor = 'x',
            scaleratio = 1,
            # showgrid = False,
            # zeroline = False,
        ),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, 'svg_cartoon_no_max_degree.pdf')


def svg_cartoon_atention():
    idx = 12
    sigma = 2

    coord0 = np.array([-1, 0, 1])
    coord1 = np.arange(5)
    coord_eval= (np.linspace(-3, 20, num=300),
                 np.linspace(coord1[0] - 2, coord1[-1] + 2, num=200))

    mesh = np.meshgrid(coord0, coord1)
    X = np.stack(mesh).T.reshape(-1, 2).astype(float)
    print(X.shape)

    D = cdist(X, X, metric='sqeuclidean')
    K = np.exp(-D / (sigma ** 2))
    K_idx = K[idx]

    s = qp_simplex(K, K_idx, zero_dim=idx)
    s[s < 1e-6] = 0

    mesh_eval = np.meshgrid(coord_eval[0], coord_eval[1])
    X_eval = np.stack(mesh_eval).T.reshape(-1, 2).astype(float)
    print(X_eval.shape)

    D1 = cdist(X[idx][np.newaxis, :], X_eval, metric='sqeuclidean')
    K1 = np.exp(-D1 / (sigma ** 2))
    print(K1.shape)
    D2 = cdist(X[s > 0], X_eval, metric='sqeuclidean')
    K2 = np.exp(-D2 / (sigma ** 2))
    print(K2.shape)
    landscape = K1 - s[s>0] @ K2
    landscape[landscape <= 0] = np.nan
    landscape = np.log10(landscape)
    landscape = landscape.reshape(coord_eval[0].shape[0],
                                  coord_eval[1].shape[0]).T

    points_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        marker=dict(size=20, color='black'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    points_trace.text = [n for n in range(len(X))]

    i_trace = go.Scatter(
        x=[X[idx, 0]], y=[X[idx, 1]],
        marker=dict(size=20, color='#e41a1c'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    i_trace.text = [idx]

    edge_traces = []
    opacity = np.log(1 + s)
    opacity[s > 0] += 0.2 - opacity[s > 0].min()
    opacity /= opacity.max()
    print(opacity)
    for k in range(len(X)):
        edge_traces.append(go.Scatter(
            x=[X[idx, 0], X[k, 0]],
            y=[X[idx, 1], X[k, 1]],
            line=dict(color='#4daf4a', width=10),
            opacity=opacity[k],
            mode='lines',
            # hoverinfo='text',
            showlegend=False,
        ))

    # max_index_flat = np.nanargmax(landscape)
    # max_index_2d = np.unravel_index(max_index_flat, landscape.shape)

    # palette = plotly.colors.diverging.RdBu_r
    # # colorscale_split = np.sum(landscape >= 0) / landscape.size
    # colorscale_split = -np.min(landscape) / (np.max(landscape) - np.min(landscape))
    # colorscale_lower = list(np.linspace(0, colorscale_split,
    #                                     num=len(palette) // 2, endpoint=False))
    # colorscale_upper = list(np.linspace(colorscale_split, 1,
    #                                     num=1 + len(palette) // 2))
    # colorscale = list(zip(colorscale_lower + colorscale_upper, palette))

    contour_traces = [
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1000,
            # colorscale=colorscale,
            contours=dict(showlines=False),
        ),
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=10,
            contours=dict(coloring='none', start=-10, end=0),
            line=dict(width=1, color='black'),
            showscale=False,
            showlegend=False,
        ),
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=5,
            contours=dict(coloring='none', start=landscape.min(), end=10),
            line=dict(width=1, color='black'),
            showscale=False,
            showlegend=False,
        ),
        # go.Scatter(
        #     x=[coord_eval[0][max_index_2d[1]]],
        #     y=[coord_eval[1][max_index_2d[0]]],
        #     marker=dict(size=20, color='#e41a1c', symbol='x'),
        #     mode='markers',
        #     showlegend=False,
        # )
    ]

    traces = []
    traces.extend(contour_traces)
    traces.extend(edge_traces)
    traces.append(points_trace)
    traces.append(i_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=300,
        width=1000,
        font=dict(size=20),
        xaxis=dict(
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            tickfont=dict(size=20),
            tickvals=coord1,
            scaleanchor = 'x',
            scaleratio = 1,
            # showgrid = False,
            # zeroline = False,
        ),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, 'svg_cartoon_attention.png', scale=3)


if __name__ == '__main__':
    # svg_cartoon_no_max_degree()
    svg_cartoon_non_uniqueness()
    # svg_cartoon_atention()
