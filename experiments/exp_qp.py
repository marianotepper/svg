import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio

from index import Kernel
from index.optimization import kernel_nnls, kernel_nnls_l0
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def svg_cartoon_no_max_degree():
    idx = 12
    sigma = 3
    print(sigma)

    coord0 = np.array([0, 1, 2, 12, 13, 14])
    coord1 = np.arange(5)
    res = np.meshgrid(coord0, coord1)
    X = np.stack(res).T.reshape(-1, 2).astype(float)
    print(X.shape)

    kernel = Kernel(sigma)

    K = kernel.build_kernel(X)

    s = kernel_nnls(K, zero_dim=idx)
    print(s, s.sum())
    s[s < 1e-9] = 0
    s /= s.sum()

    # print(s, s.sum())
    # print(np.nonzero(s)[0])

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
        marker=dict(size=22, color='#e41a1c'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    i_trace.text = [idx]

    edge_traces = []
    nonzeros = np.where(s > 0)[0]

    for ii in nonzeros:
        edge_traces.append(go.Scatter(
            name=f'{s[ii]:.3f}',
            x=[X[idx, 0], X[ii, 0]],
            y=[X[idx, 1], X[ii, 1]],
            line=dict(color='#377eb8', width=10),
            mode='lines',
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
            tickfont=dict(size=30),
        ),
        yaxis=dict(
            tickfont=dict(size=30),
            tickvals=coord1,
            scaleanchor = 'x',
            scaleratio = 1,
        ),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, 'svg_cartoon_no_max_degree.pdf')


if __name__ == '__main__':
    svg_cartoon_no_max_degree()
