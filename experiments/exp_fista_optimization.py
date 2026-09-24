"""
Generate figures/fista_solver.pdf: an empirical demonstration that the
projected-gradient (FISTA) solver in qp_budget_fixed.qp_budget solves the
budget-constrained SVG QP correctly and fast.

Two panels (plotly):
  Left  - convergence on a representative budget-active node: objective gap to
          the optimum F(s_t) - F* and the step residual ||s_{t+1}-s_t|| vs iter.
  Right - wall-clock time per node vs n for FISTA and SLSQP (scipy), both
          converged to the same optimum (equal-accuracy speed comparison).

Requires: numpy, scipy, plotly, kaleido, and qp_budget_fixed.py on the path.
Run:  python make_fista_figure.py
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize

from index.optimization import _proj_ball_orthant, _top_eig, qp_fista

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plot_utils import write_image


# ---------------------------------------------------------------------------
# solvers / helpers
# ---------------------------------------------------------------------------
def rbf(X, s):
    d2 = np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T
    return np.exp(-np.maximum(d2, 0) / (2 * s**2))


def obj(A, b, x):
    return 0.5 * x @ A @ x - b @ x


def fista_trace(A, b, iters):
    """FISTA with no early stop, recording objective and step residual per iter."""
    n = len(A); r = np.sqrt(1.0 / n); L = _top_eig(A)
    x = _proj_ball_orthant(b.copy(), r); y = x.copy(); t = 1.0
    F, R = [], []
    for _ in range(iters):
        xn = _proj_ball_orthant(y - (A @ y - b) / L, r)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        F.append(obj(A, b, xn)); R.append(np.linalg.norm(xn - x) / (np.linalg.norm(x) + 1e-30))
        x, t = xn, tn
    return np.array(F), np.array(R)


def slsqp(A, b):
    n = len(A); f = lambda x: 0.5 * x @ A @ x - b @ x; g = lambda x: A @ x - b
    cons = [{"type": "ineq", "fun": lambda x: 1 - n * (x @ x), "jac": lambda x: -2 * n * x}]
    return minimize(f, np.ones(n) / n, jac=g, bounds=[(0, None)] * n, constraints=cons,
                    method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14}).x


# ---------------------------------------------------------------------------
# data for the two panels
# ---------------------------------------------------------------------------
def convergence_panel():
    X = np.random.default_rng(1).standard_normal((100, 8)); K = rbf(X, 1.0); i = 0
    idx = [j for j in range(100) if j != i]; A = K[np.ix_(idx, idx)]; b = K[i, idx]
    F, R = fista_trace(A, b, 300)
    Fstar = min(F.min(), obj(A, b, slsqp(A, b)))          # reference optimum
    gap = np.maximum(F - Fstar, 1e-16)
    return np.arange(1, len(F) + 1), gap, R


def speed_panel(ns=(20, 40, 80, 160, 320, 640), nodes=12, d=8):
    tf, ts, maxgap = [], [], 0.0
    for n in ns:
        Xn = np.random.default_rng(0).standard_normal((n, d)); Kn = rbf(Xn, 1.0)
        af, asl = [], []
        for i in range(min(n, nodes)):
            idx = [j for j in range(n) if j != i]; A = Kn[np.ix_(idx, idx)]; b = Kn[i, idx]
            t0 = time.perf_counter(); xf = qp_fista(A, b); af.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); xs = slsqp(A, b); asl.append(time.perf_counter() - t0)
            maxgap = max(maxgap, abs(obj(A, b, xf) - obj(A, b, xs)))
        tf.append(1e3 * np.mean(af)); ts.append(1e3 * np.mean(asl))
    return np.array(ns), np.array(tf), np.array(ts), maxgap


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def main():
    it, gap, res = convergence_panel()
    ns, tf, ts, maxgap = speed_panel()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("FISTA convergence (budget active)",
                        f"Speed at equal optimum (max obj. gap <= {maxgap:.0e})"),
    )
    # left panel
    fig.add_trace(go.Scatter(x=it, y=gap, mode="lines", name="objective gap F(s_t) - F*"),
                             row=1, col=1)
    fig.add_trace(go.Scatter(x=it, y=res, mode="lines", name="step residual ||s_{t+1}-s_t||"),
                             row=1, col=1)
    fig.update_xaxes(title_text="iteration", row=1, col=1)
    fig.update_yaxes(title_text="value", type="log", exponentformat="power", row=1, col=1)
    # right panel
    fig.add_trace(go.Scatter(x=ns, y=ts, mode="lines+markers", name="SLSQP",
                             line=dict(color="gray"), marker=dict(symbol="circle", size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=ns, y=tf, mode="lines+markers", name="FISTA",
                             line=dict(color="seagreen"), marker=dict(symbol="square", size=8)), row=1, col=2)
    fig.update_xaxes(title_text="n", type="log", tickmode="array",
                     tickvals=ns, ticktext=[str(int(v)) for v in ns], row=1, col=2)
    fig.update_yaxes(title_text="time per node (ms)", type="log", exponentformat="power", row=1, col=2)

    fig.update_layout(template="simple_white", width=920, height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
                      margin=dict(l=60, r=20, t=70, b=50), font=dict(size=13))
    write_image(fig, 'fista_solver.pdf', scale=3)
    print("saved", 'fista_solver.pdf')
    print("speed ms/node:", list(zip(ns.tolist(), np.round(tf, 3).tolist(), np.round(ts, 3).tolist())))
    print("max |obj_FISTA - obj_SLSQP|:", maxgap)


if __name__ == "__main__":
    main()