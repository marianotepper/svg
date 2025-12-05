from typing import Optional

import numpy as np
from scipy.optimize import minimize


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
        x_temp = qp_multiplicative(A, b)

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


def qp_multiplicative(A: np.ndarray, b: np.ndarray):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ A @ x - b.T @ x
    subject to x >= 0.

    :param A: square ndarray representing a positive definite matrix with
              nonnegative entries
    :param b: one-dimensional ndarray
    :return: the solution x
    """
    n = len(A)
    x = np.ones(n) / n
    for it in range(1000):
        gamma = b / (A @ x)
        x_new = x * gamma

        factor = n * (x_new ** 2).sum()
        if factor > 1:
            x_new /= factor ** 0.5

        if np.linalg.norm(x_new - x) / np.linalg.norm(x) < 1e-6:
            return x_new
        else:
            x = x_new

    return x


def kernel_nnls_l0(K: np.ndarray, zero_dim: int, nonzeros: int,
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
    y = K[zero_dim]

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
        x_prime = qp_multiplicative(K[candidates, :][:, candidates],
                                    K[zero_dim, :][candidates])
        x_prime[x_prime < x_prime.max() * 1e-4] = 0

        keep_n_entries = np.minimum(nonzeros, np.count_nonzero(x_prime))
        idx = np.argsort(x_prime)[-keep_n_entries:]
        candidates = [candidates[i] for i in idx]
        x_prime = x_prime[idx]
        y_new = K[zero_dim] - x_prime.T @ K[candidates]
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
