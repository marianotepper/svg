import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from svg import MRNG, radial_basis_function

from plot_utils import plot_graph, plot_lunes

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None

rng = np.random.default_rng(0)

X = rng.random(size=(50, 2))

mrng = MRNG(radial_basis_function)
mrng.fit(X)

fig = plot_graph(mrng.graph, X)
# fig = plot_lunes(mrng.graph, X, 46, idx_additional_lunes=[48])
# fig = plot_lunes(mrng.graph, X, 22, idx_additional_lunes=[])
# fig = plot_lunes(mrng.graph, X, 16, idx_additional_lunes=[])
fig.show()