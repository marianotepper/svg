import numpy as np
import plotly.graph_objects as go
import time
from shapely.geometry import Point

from plotly_dirgraph import add_edge

def plot_graph(graph, X, node_size=5):
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        start = X[edge[0]]
        end = X[edge[1]]
        edge_x, edge_y = add_edge(start, end, edge_x, edge_y, 1, 'end', .02,
                                 30, node_size)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
    )

    node_trace = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        marker=dict(size=8),
        mode='markers',
        hoverinfo='text',
    )

    node_trace.text = [n for n in graph.nodes()]

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_lunes(graph, X, idx, idx_additional_lunes: list[int] = []):
    neighs = [e[1] for e in graph.edges(idx)]
    neighs.extend(idx_additional_lunes)
    dists = np.sum((X[idx] - X[neighs]) ** 2, axis=1) ** 0.5

    p_idx = Point(X[idx, 0], X[idx, 1])
    circles = [
        p_idx.buffer(d).intersection(Point(X[j, 0], X[j, 1]).buffer(d))
              for d, j in zip(dists, neighs)]

    fig = plot_graph(graph, X)

    for circle in circles:
        x, y = circle.exterior.coords.xy
        fig.add_trace(go.Scatter(x=list(x), y=list(y)))
    return fig



def write_image(fig, filename, scale=None):
    fig.write_image(filename, scale=scale)
    time.sleep(2)
    fig.write_image(filename, scale=scale)