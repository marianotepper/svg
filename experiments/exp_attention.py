import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from index import Kernel
from index.optimization import kernel_nnls
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def svg_cartoon_atention():
    idx = 12
    sigma = 2

    coord0 = np.array([-1, 0, 1])
    coord1 = np.arange(5)
    coord_eval= (np.linspace(-3, 20, num=300),
                 np.linspace(coord1[0] - 2, coord1[-1] + 2, num=200))

    mesh = np.meshgrid(coord0, coord1)
    X = np.stack(mesh).T.reshape(-1, 2).astype(float)

    additional_coord0 = np.array([12, 13, 14])
    additional_coord1 = np.arange(5)
    X_additional = np.stack(np.meshgrid(additional_coord0, additional_coord1))
    X_additional = X_additional.T.reshape(-1, 2).astype(float)


    kernel = Kernel(sigma)

    K = kernel.build_kernel(X)

    s = kernel_nnls(K, zero_dim=idx)
    s[s < 1e-6] = 0

    mesh_eval = np.meshgrid(coord_eval[0], coord_eval[1])
    X_eval = np.stack(mesh_eval).T.reshape(-1, 2).astype(float)

    K1 = kernel.build_kernel2(X[idx], X_eval)
    K2 = kernel.build_kernel2(X[s > 0], X_eval)
    landscape = K1 - s[s>0] @ K2
    landscape[landscape <= 0] = np.nan
    landscape = np.log10(landscape)
    landscape = landscape.reshape(coord_eval[0].shape[0],
                                  coord_eval[1].shape[0]).T

    points_traces = [
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            marker=dict(size=20, color='black'),
            mode='markers',
            hoverinfo='text',
            showlegend=False,
        ),
        go.Scatter(
            x=X_additional[:, 0], y=X_additional[:, 1],
            marker=dict(size=20, color='black'),
            mode='markers',
            hoverinfo='text',
            showlegend=False,
        ),
    ]
    points_traces[0].text = [n for n in range(len(X))]

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
            showlegend=False,
        ))

    contour_traces = [
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1000,
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
    ]

    traces = []
    traces.extend(contour_traces)
    traces.extend(edge_traces)
    traces.extend(points_traces)
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
    svg_cartoon_atention()
