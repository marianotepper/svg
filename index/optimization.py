import numpy as np
from scipy.optimize import minimize


def kernel_nnls(K: np.ndarray, zero_dim: int):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ K @ x - K[zero_dim].T @ x
    subject to x >= 0, x[zero_dim] = 0.

    :param K: square ndarray representing a positive definite matrix
    :param zero_dim: integer
    :return: The solution x
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("A must be a square ndarray")
    if not (0 <= zero_dim < K.shape[0]):
        raise ValueError("Only 0 <= zero_dim < A.shape[0] allowed")

    n = len(K)
    fun = lambda x: 0.5 * (x @ K @ x) - K[zero_dim].T @ x
    bounds = [(0, None)] * n
    bounds[zero_dim] = (0, 0)
    x0 = np.ones(n) / n
    res = minimize(fun, x0, bounds=bounds, tol=1e-50)
    return res.x


def qp(A: np.ndarray, b: np.ndarray):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ A @ x - b.T @ x
    subject to x >= 0.

    :param A: square ndarray representing a positive definite matrix
    :param b: one-dimensional ndarray
    :return: The solution x
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square ndarray")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Only 0 <= zero_dim < A.shape[0] allowed")

    n = len(A)
    fun = lambda x: 0.5 * (x @ A @ x) - b @ x
    bounds = [(0, None)] * n
    x0 = np.ones(n) / n
    res = minimize(fun, x0, bounds=bounds, tol=1e-50)
    return res.x


def kernel_nnls_l0(K: np.ndarray, zero_dim: int, nonzeros: int):
    """
    Solves the convex problem:
        min_{x} 0.5 x.T @ K @ x - K[zero_dim].T @ x
    subject to x >= 0, x[zero_dim] = 0, ||x||_0 <= nonzeros.

    :param K: square ndarray representing a positive definite
    and nonnegative matrix
    :param zero_dim: integer (see problem description)
    :param nonzeros: integer (see problem description)
    :return: The solution x
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("A must be a square ndarray")
    if not (0 <= zero_dim < K.shape[0]):
        raise ValueError("Only 0 <= zero_dim < A.shape[0] allowed")
    if not (0 < nonzeros < K.shape[0]):
        raise ValueError("Only 0 < nonzeros < A.shape[0] allowed")

    n = len(K)
    candidates_old = []
    y = K[zero_dim]

    for it in range(100):
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
        x_prime = qp(K[candidates, :][:, candidates],
                     K[zero_dim, :][candidates])
        x_prime[x_prime < x_prime.max() * 1e-6] = 0

        keep_n_entries = np.minimum(nonzeros, np.count_nonzero(x_prime))
        idx = np.argsort(x_prime)[-keep_n_entries:]
        candidates = [candidates[i] for i in idx]
        x_prime = x_prime[idx]
        y = K[zero_dim] - x_prime.T @ K[candidates]

        if sorted(candidates) == sorted(candidates_old):
            break
        else:
            candidates_old = list(candidates)

    x = np.zeros(n)
    x[candidates] = x_prime

    return x
