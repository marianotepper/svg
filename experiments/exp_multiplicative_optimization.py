from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots
import sys


import datasets
from index import Kernel
from index.optimization import qp
from plot_utils import write_image

pio.templates.default = "plotly_white"


def run_qp(K: np.ndarray, idx: int, solver: Callable):
    n = len(K)
    candidates = list(range(n))
    candidates.remove(idx)

    A = K[candidates, :][:, candidates]
    b = K[idx, :][candidates]

    x = solver(A, b)

    fun = lambda xx: 0.5 * (xx @ A @ xx) - b @ xx
    # print(f'{solver}, ID={idx}, F(x) = {fun(x)}')
    return fun(x)


def qp_multiplicative(A: np.ndarray, b: np.ndarray, n_iter: int = 1100,
                      plot_graphs: bool = False):
    objective_fun = lambda xx: 0.5 * (xx @ A @ xx) - b @ xx

    gamma_history = []
    delta_gamma_history = []
    objective_history = []

    n = len(A)
    x = np.ones(n) / n
    for it in range(n_iter):
        gamma = b / (A @ x)
        x = x * gamma

        if it > 0:
            max_gamma_diff = np.max(np.abs(gamma_history[-1] - gamma))
            delta_gamma_history.append(max_gamma_diff)

        gamma_history.append(gamma)
        objective_history.append(objective_fun(x))

    factor = n * (x ** 2).sum()
    if factor > 1:
        x /= factor ** 0.5

    gamma_history = np.array(gamma_history).T
    idx_gamma = np.argsort(np.abs(1 - gamma_history[:, -1]))#[:20]
    gamma_history = gamma_history[idx_gamma]

    if plot_graphs:
        import plotly.graph_objects as go

        mask_non_support = gamma_history[:, -1] < 0.95
        gamma_non_support = gamma_history[mask_non_support]

        dashes = ['solid', 'longdash', 'dot', 'dash']
        # fig = plotly.subplots.make_subplots(rows=1, cols=1, shared_yaxes=True,
        #                                     column_widths=[3, 1])
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        for each_gamma, dash in zip(gamma_history[:4], dashes):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(each_gamma)),
                    y=each_gamma,
                    line=dict(width=3, dash=dash),
                    # opacity=0.5,
                    mode='lines',
                    showlegend=False),
                row=1, col=1)

        for each_gamma in np.random.default_rng(0).choice(gamma_non_support, size=10, replace=False):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(each_gamma)),
                    y=each_gamma,
                    line=dict(color='silver', dash='solid'),
                    mode='lines',
                    showlegend=False),
                row=1, col=1)

        fig.update_xaxes(title='Iterations', type="log",
                         exponentformat="power", range=(0, 3.2), row=1, col=1)
        fig.update_yaxes(title=r'$\LARGE{\mathbf{\gamma}}$', row=1, col=1)

        # Scipy is not necessarily more accurate than the multiplicative
        # algorithm, so we cannot use it as the ground truth.
        # x_qp = qp(A, b)
        # mask_qp = x_qp[idx_gamma] > 1e-8
        # fig.add_trace(
        #     go.Box(
        #         name='zeros',
        #          # x=np.zeros(len(gamma_history)) + noise,
        #         y=gamma_history[~mask_qp, -1],  # gamma_history[:, -1],
        #         marker=dict(color='#66c2a5'),
        #         showlegend=True),
        #     row=1, col=2
        # )
        # fig.add_trace(
        #      go.Box(
        #          name='nonzeros',
        #          # x=np.zeros(len(gamma_history)) + noise,
        #          y=gamma_history[mask_qp, -1],
        #          marker=dict(color='#fc8d62'),
        #          showlegend=True),
        #     row=1, col=2
        # )

        # fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_layout(
            # legend=dict(
            #     x=1,
            #     y=0.9,
            #     xanchor="right",  # Anchor point within the legend box
            #     yanchor="top",  # Anchor point within the legend box
            #     orientation="v"
            # ),
            # height=400,
            # width=700,
            font=dict(size=25),
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        fig.show()
        write_image(fig, 'multiplicative_gammas.png', scale=3)

        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
             go.Scatter(
                 name=r'$\LARGE{\max(|\mathbf{\gamma}_{t+1} - \mathbf{\gamma}_{t}|)}$',
                 x=np.arange(len(delta_gamma_history)),
                 y=delta_gamma_history,
                 line=dict(width=3, color='#377eb8'),
                 # opacity=0.5,
                 mode='lines',
                 showlegend=False,
             ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                name='Objective function',
                x=np.arange(len(objective_history)),
                y=objective_history,
                line=dict(width=3, color='#e41a1c'),
                # opacity=0.5,
                mode='lines',
                showlegend=False,
            ),
            secondary_y=True
        )
        fig.update_xaxes(title='Iterations', type="log",
                         exponentformat="power")
        fig.update_yaxes(type="log", dtick=1, exponentformat="power",
                         secondary_y=False,
                         title=dict(text=r'$\LARGE{\max(|\mathbf{\gamma}_{t+1} - \mathbf{\gamma}_{t}|)}$',
                                    font=dict(color="#377eb8")))
        fig.update_yaxes(secondary_y=True,
                         title=dict(text='Objective function',
                                    font=dict(color="#e41a1c")))
        # fig.update_yaxes(type="log", dtick=1, exponentformat="power",
        #                  secondary_y=False)
        fig.update_layout(
            legend=dict(
                x=0.05,
                y=0.3,
                xanchor="left",  # Anchor point within the legend box
                yanchor="top",  # Anchor point within the legend box
                orientation="v"
            ),
            # height=400,
            # width=1800,
            font=dict(size=30),
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        fig.show()
        write_image(fig, 'multiplicative_convergence.png', scale=3)

    # print([np.count_nonzero(np.abs(gamma_history[:, -1] - 1) < tol)
    #       for tol in [1e-4, 1e-3, 1e-2, 1e-1]])

    return x


def run_example(dirname, dataset_name):
    dataset = datasets.select_dataset(dataset_name, dirname=dirname)
    X = dataset.X_db[:100]

    sigma = 1
    kernel = Kernel(sigma=sigma, similarity='euclidean')
    K = kernel.build_kernel(X)

    palette = plotly.colors.qualitative.Plotly

    n_iter_ls = [10, 20, 50, 100, 200, 500, 1000]
    fun_values = [
        [run_qp(K, i, qp)] +
        [run_qp(K, i, lambda A, b: qp_multiplicative(A, b, n_iter=n_iter,
                                                     plot_graphs=False))
         for n_iter in n_iter_ls]
        for i in range(10)
    ]
    fun_values = [np.array(e) for e in zip(*fun_values)]
    fun_values_scipy = fun_values[0]

    fig = go.Figure(data=[
        go.Box(name=f'{n_iter_ls[j]}',
               y= fun_values[j + 1] / fun_values_scipy,
               marker_color=palette[j],
               showlegend=False)
        for j in range(len(n_iter_ls))
    ])
    fig.update_xaxes(title='Iterations')
    fig.update_yaxes(title='Loss ratio wrt scipy')
    fig.update_layout(
        # height=400,
        # width=1800,
        font=dict(size=30),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    fig.show()
    write_image(fig, 'multiplicative_scipy_comparison.png', scale=3)


def plot_paths(dirname, dataset_name):
    dataset = datasets.select_dataset(dataset_name, dirname=dirname)
    X = dataset.X_db[:100]

    sigma = 1
    kernel = Kernel(sigma=sigma, similarity='euclidean')
    K = kernel.build_kernel(X)

    idx = 10
    # run_qp(K, idx, qp)
    run_qp(K, idx, lambda A, b: qp_multiplicative(A, b, plot_graphs=True))


def main():
    if len(sys.argv) != 2:
        datasets_dirname = './datasets'
    else:
        datasets_dirname = sys.argv[1]

    dataset_names = ['openai-v3-small-100k']

    for dataset_name in dataset_names:
        run_example(datasets_dirname, dataset_name)
        plot_paths(datasets_dirname, dataset_name)


if __name__ == '__main__':
    main()