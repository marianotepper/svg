import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from index import MRNG, radial_basis_function

from plot_utils import plot_graph, plot_lunes

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None

rng = np.random.default_rng(0)

X = rng.random(size=(50, 2))

mrng = MRNG(radial_basis_function)
mrng.fit(X)

fig = plot_graph(mrng.graph, X)
# fig.show()
# fig = plot_lunes(mrng.graph, X, 46, idx_additional_lunes=[])
# fig.show()
# fig = plot_lunes(mrng.graph, X, 22, idx_additional_lunes=[])
# fig = plot_lunes(mrng.graph, X, 16, idx_additional_lunes=[])

query = np.array([0.1, 0.1])
nneigh = mrng.search(query)
print(nneigh)
fig.add_trace(go.Scatter(x=[query[0]], y=[query[1]], mode='markers',))
fig.add_trace(go.Scatter(x=[X[nneigh, 0]], y=[X[nneigh, 1]], mode='markers',))

fig.show()