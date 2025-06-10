import numpy as np
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries


from index import Kernel
from index.optimization import kernel_nnls
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None

def topographic_watershed(landscape_max):
    image = landscape_max
    image = (image - image.min()) / (image.max() - image.min())
    image = np.floor(image * 255).astype(int)

    coords = peak_local_max(image, footprint=np.ones((3, 3)), min_distance=5)
    mask = np.zeros(image.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-image, markers=markers, connectivity=2)

    return labels


def plot_all_boundaries(step_x=6, step_y=6, n_grid=100):
    sigma = np.sqrt((step_x - 1) ** 2 + (step_y - 1) ** 2) / 2

    coord0 = np.arange(step_x) - 0.5 * step_x + 0.5
    coord1 = np.arange(step_y) - 0.5 * step_y + 0.5
    coord_eval = (np.linspace(coord0.min() - 3, coord0.max() + 3, num=n_grid),
                  np.linspace(coord1.min() - 3, coord1.max() + 3, num=n_grid))

    mesh = np.meshgrid(coord0, coord1)
    X = np.stack(mesh).T.reshape(-1, 2).astype(float)

    mesh_eval = np.meshgrid(coord_eval[0], coord_eval[1])
    X_eval = np.stack(mesh_eval).T.reshape(-1, 2).astype(float)

    points_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        marker=dict(size=12, color='black'),
        mode='markers',
        hoverinfo='text',
        showlegend=False,
    )
    points_trace.text = [n for n in range(len(X))]

    kernel = Kernel(sigma)

    K = kernel.build_kernel(X)

    contour_traces = []

    landscape_max = None

    for idx in range(len(X)):
        s = kernel_nnls(K, zero_dim=idx)
        s[s < 1e-6] = 0

        K1 = kernel.build_kernel2(X[idx], X_eval)
        K2 = kernel.build_kernel2(X[s > 0], X_eval)
        j = np.nonzero(s)[0][0]
        b = -0.5 * (K[idx, idx] - s @ K[:, idx] + K[idx, j] - s @ K[:, j])
        landscape = K1 - s[s > 0] @ K2 + b
        landscape /= K[idx, idx] - s @ K[:, idx] + b
        landscape = landscape.reshape(coord_eval[0].shape[0],
                                      coord_eval[1].shape[0]).T

        if landscape_max is None:
            landscape_max = landscape
        else:
            landscape_max = np.maximum(landscape_max, landscape)

        contour_traces.append(
            go.Contour(
                x=coord_eval[0],
                y=coord_eval[1],
                z=landscape,
                ncontours=1,
                contours=dict(coloring='none', start=0, end=0),
                line=dict(width=2, color='#1b9e77'),
                showscale=False,
                showlegend=False,
            )
        )

    traces = []
    traces.extend(contour_traces)
    traces.append(points_trace)

    fig = plotly.subplots.make_subplots(rows=1, cols=3, shared_yaxes=True)

    for tr in traces:
        fig.add_trace(tr, row=1, col=1)
    fig.update_xaxes(range=(X_eval[:, 0].min(), X_eval[:, 0].max()),
                     row=1, col=1)
    fig.update_yaxes(scaleanchor='x', scaleratio=1,
                     range=(X_eval[:, 1].min(), X_eval[:, 1].max()),
                     row=1, col=1)

    fig.add_trace(
        go.Contour(
            x=coord_eval[0],
            y=coord_eval[1],
            z=np.log10(landscape_max + 1 - landscape_max.min()),
            ncontours=1000,
            contours=dict(showlines=False),
            colorscale=plotly.colors.diverging.RdBu_r,
            colorbar=dict(len=0.5),
            showlegend=False,
        ),
        row=1, col=2
    )
    fig.add_trace(points_trace, row=1, col=2)

    labels = topographic_watershed(landscape_max)
    fig.add_trace(
        go.Heatmap(
            x=np.interp(np.arange((len(coord_eval[0])), step=0.5),
                        np.arange((len(coord_eval[0]))),
                        coord_eval[0]),
            y=np.interp(np.arange((len(coord_eval[1])), step=0.5),
                        np.arange((len(coord_eval[1]))),
                        coord_eval[1]),
            z=find_boundaries(labels, mode='subpixel').astype(int),
            colorscale=['white', '#ff7f00'],
            showscale=False,
        ),
        row=1, col=3
    )
    fig.add_trace(points_trace, row=1, col=3)

    fig.update_xaxes(range=(X_eval[:, 0].min(), X_eval[:, 0].max()),
                     row=1, col=2)
    fig.update_yaxes(scaleanchor='x2', scaleratio=1,
                     range=(X_eval[:, 1].min(), X_eval[:, 1].max()),
                     row=1, col=2)
    fig.update_layout(
        height=400,
        width=1300,
        font=dict(size=20),
        xaxis=dict(
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            tickfont=dict(size=20),
            scaleanchor='x',
            scaleratio=1,
        ),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, 'all_boundaries_grid_example.png', scale=3)


if __name__ == '__main__':
    plot_all_boundaries()