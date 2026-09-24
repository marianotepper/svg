from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize

from .kernels import KernelMatrix


def kernel_nnls(K: np.ndarray, zero_dim: int, solver='multiplicative'):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ K @ x - K[zero_dim].T @ x
    subject to x >= 0, x[zero_dim] = 0.

    :param K: square ndarray representing a positive definite matrix
    :param zero_dim: integer
    :param solver: The optimization method to use, either 'scipy' or 'multiplicative'
    :return: The solution x
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("A must be a square ndarray")
    if not (0 <= zero_dim < K.shape[0]):
        raise ValueError("Only 0 <= zero_dim < A.shape[0] allowed")

    n = len(K)
    idx = list(range(n))
    idx.remove(zero_dim)

    A = K[idx, :][:, idx]
    b = K[zero_dim, :][idx]

    if solver == 'scipy':
        x_temp = qp(A, b)
    elif solver == 'multiplicative':
        x_temp = qp_fista(A, b, budget=1.0 / n)

    x = np.zeros((n,))
    x[idx] = x_temp
    return x


def qp(A: np.ndarray, b: np.ndarray):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ A @ x - b.T @ x
    subject to x >= 0.

    :param A: square ndarray representing a positive definite matrix
    :param b: one-dimensional ndarray
    :return: the solution x
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square ndarray")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and B must have the same number of dimensions")

    n = len(A)
    fun = lambda x: 0.5 * (x @ A @ x) - b @ x
    bounds = [(0, None)] * n
    constraints = [{'type': 'ineq', 'fun': lambda x: 1 - n * (x ** 2).sum()}]
    x0 = np.ones(n) / n
    res = minimize(fun, x0, bounds=bounds, constraints=constraints, tol=1e-50)

    return res.x


def _proj_ball_orthant(y, r):
    """Exact Euclidean projection onto {x >= 0, ||x|| <= r}: clip, then scale."""
    z = np.maximum(y, 0.0)
    nz = np.linalg.norm(z)
    return z if nz <= r else z * (r / nz)


def _top_eig(A, iters=64):
    """Largest eigenvalue of the symmetric PSD matrix A via power iteration,
    with a 1% safety margin so it upper-bounds lambda_max (valid FISTA step)."""
    n = len(A)
    v = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 1.0
        v = w / nw
        lam = v @ (A @ v)
    return lam * 1.01 + 1e-12


def qp_fista(A, b, budget=None, iters=5000, tol=1e-12):
    """min_x 0.5 x^T A x - b^T x  s.t. x >= 0, ||x||^2 <= budget.

    Accelerated projected gradient. `budget` defaults to 1/len(A) to match
    optimization.qp_multiplicative's semantics; pass budget=1.0/N explicitly to
    use the manuscript's 1/N with N = total number of points (recommended for
    consistency inside kernel_nnls_l0, which otherwise uses 1/|candidates|).
    """
    n = len(A)
    r = np.sqrt((1.0 / n) if budget is None else float(budget))
    L = _top_eig(A)                        # Lipschitz constant of the gradient
    x = _proj_ball_orthant(b.copy(), r)    # warm start
    y = x.copy()
    t = 1.0
    for _ in range(iters):
        xn = _proj_ball_orthant(y - (A @ y - b) / L, r)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        if np.linalg.norm(xn - x) / (np.linalg.norm(x) + 1e-30) < tol:
            x = xn
            break
        x, t = xn, tn
    return x

def kernel_nnls_l0(K: Union[np.ndarray, KernelMatrix], zero_dim: int, nonzeros: int,
                   outer_l0_iterations: Optional[int] = None):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ K @ x - K[zero_dim].T @ x
    subject to x >= 0, x[zero_dim] = 0, ||x||_0 <= nonzeros.

    :param K: square ndarray representing a positive definite
    and nonnegative matrix
    :param zero_dim: integer (see problem description)
    :param nonzeros: integer (see problem description)
    :param outer_l0_iterations: integer number of outer iterations
    :return: The solution x
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("A must be a square ndarray")
    if not (0 <= zero_dim < K.shape[0]):
        raise ValueError("Only 0 <= zero_dim < A.shape[0] allowed")
    if not (0 < nonzeros < K.shape[0]):
        raise ValueError("Only 0 < nonzeros < A.shape[0] allowed")

    if outer_l0_iterations is None:
        outer_l0_iterations = 10

    n = len(K)
    candidates_old = []
    y = K.get_submatrix(zero_dim)

    error_y = np.inf

    for it in range(outer_l0_iterations):
        if nonzeros + 1 <= len(K):
            largest = np.argpartition(-y, nonzeros + 1)
        else:
            largest = np.argsort(-y)

        if zero_dim in largest[:nonzeros]:
            candidates_t = list(largest[:nonzeros + 1])
            candidates_t.remove(zero_dim)
        else:
            candidates_t = list(largest[:nonzeros])
        candidates = list(set(candidates_old).union(candidates_t))

        idx_temp = list(candidates)
        idx_temp.append(zero_dim)
        x_prime = qp_fista(K.get_submatrix(candidates, candidates),
                           K.get_submatrix(zero_dim, candidates),
                           budget=1.0 / n)
        x_prime[x_prime < x_prime.max() * 1e-4] = 0

        keep_n_entries = np.minimum(nonzeros, np.count_nonzero(x_prime))
        idx = np.argsort(x_prime)[-keep_n_entries:]
        candidates = [candidates[i] for i in idx]
        x_prime = x_prime[idx]

        K_candidates = K.get_submatrix(candidates)
        if len(K_candidates.shape) == 1:
            K_candidates = K_candidates[:, np.newaxis]
        y_new = K.get_submatrix(zero_dim) - K_candidates @ x_prime
        error_y_new = np.linalg.norm(y_new - y)

        if sorted(candidates) == sorted(candidates_old) and np.abs(error_y - error_y_new) < 1e-6:
            break
        else:
            error_y = error_y_new
            y = y_new
            candidates_old = list(candidates)

    x = np.zeros(n)
    x[candidates] = x_prime

    return x
