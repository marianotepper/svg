import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots

from index.optimization import kernel_nnls
from plot_utils import write_image
from index import Kernel

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def subplot_hard_margin(fig, idx, subplot, step_x=3, step_y=3, sigma=2.,
                     colorbar_x_pos=0.5, n_grid=300):
    if not (0 <= idx < step_x * step_y):
        raise ValueError("idx out of range")

    X = np.array([[0, 0],
                  [6, 0],
                  [3, 1],
                  [3, -1]])
    coord_eval = (
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, num=n_grid),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, num=n_grid)
    )

    kernel = Kernel(sigma)

    K = kernel.build_kernel(X)

    s = kernel_nnls(K, zero_dim=idx)
    s[s < 1e-6] = 0
    print(s, s.sum())

    mesh_eval = np.meshgrid(coord_eval[0], coord_eval[1])
    X_eval = np.stack(mesh_eval).T.reshape(-1, 2).astype(float)

    K1 = kernel.build_kernel2(X[idx], X_eval)
    K2 = kernel.build_kernel2(X[s > 0], X_eval)
    j = np.nonzero(s)[0][0]
    b = -0.5 * (K[idx, idx] - s @ K[:, idx] + K[idx, j] - s @ K[:, j])
    landscape = K1 - s[s > 0] @ K2 + b
    landscape /= K[idx, idx] - s @ K[:, idx] + b
    landscape = landscape.reshape(coord_eval[0].shape[0],
                                  coord_eval[1].shape[0]).T

    points_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        marker=dict(size=12, color='black'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    points_trace.text = [n for n in range(len(X))]

    i_trace = go.Scatter(
        x=[X[idx, 0]], y=[X[idx, 1]],
        marker=dict(size=12, color='#e41a1c'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    i_trace.text = [idx]

    edge_traces = []
    for k in range(len(X)):
        if s[k] > 0:
            edge_traces.append(go.Scatter(
                x=[X[idx, 0], X[k, 0]],
                y=[X[idx, 1], X[k, 1]],
                line=dict(color='#4daf4a', width=5),
                mode='lines',
                showlegend=False,
            ))

    palette = plotly.colors.diverging.RdBu_r
    colorscale_split = -np.min(landscape) / (np.max(landscape) - np.min(landscape))
    colorscale_lower = list(np.linspace(0, colorscale_split,
                                        num=len(palette) // 2, endpoint=False))
    colorscale_upper = list(np.linspace(colorscale_split, 1,
                                        num=1 + len(palette) // 2))
    colorscale = list(zip(colorscale_lower + colorscale_upper, palette))


    contour_traces = [
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1000,
            colorscale=colorscale,
            colorbar=dict(orientation='v', len=0.7, x=colorbar_x_pos),
            contours=dict(showlines=False),
        ),
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1,
            contours=dict(coloring='none', start=0, end=0),
            line=dict(width=3, color='black', dash='dot'),
            showscale=False,
            showlegend=False,
        ),
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1,
            contours=dict(coloring='none', start=-1, end=-1),
            line=dict(width=3, color='#377eb8', dash='dot'),
            showscale=False,
            showlegend=False,
        ),
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=landscape,
            ncontours=1,
            contours=dict(coloring='none', start=1, end=1),
            line=dict(width=3, color='#e41a1c', dash='dot'),
            showscale=False,
            showlegend=False,
        ),
    ]

    traces = []
    traces.extend(contour_traces)
    traces.extend(edge_traces)
    traces.append(points_trace)
    traces.append(i_trace)

    for trace in traces:
        fig.add_trace(trace, subplot[0], subplot[1])

    rows, cols = fig._get_subplot_rows_columns()
    scaleanchor = f'x{(subplot[0] - 1) * len(cols) + subplot[1]}'
    fig.update_yaxes(scaleanchor=scaleanchor, scaleratio=1,
                     row=subplot[0], col=subplot[1])


def plot_hard_margin():
    rows = 1
    cols = 2
    fig = plotly.subplots.make_subplots(rows=rows, cols=cols,
                                        shared_xaxes=True,
                                        shared_yaxes=True,
                                        subplot_titles=[u'\u03C3 = 10',
                                                        u'\u03C3 = 3'])
    subplot_hard_margin(fig, 3, (1, 1), sigma=10, colorbar_x_pos=0.45)
    subplot_hard_margin(fig, 3, (1, 2), sigma=3, colorbar_x_pos=1)

    fig.update_annotations(font_size=25)
    fig.update_layout(
        height=300,
        width=1200,
        font=dict(size=25),
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
        ),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
    )
    fig.show()
    write_image(fig, 'quasi_navigability_example.png', scale=3)


if __name__ == '__main__':
    plot_hard_margin()
