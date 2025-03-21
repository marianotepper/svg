import numpy as np
from scipy.optimize import minimize


def qp_simplex(A: np.ndarray, b: np.ndarray, zero_dim: int):
    """
    Solves he convex problem for vector x:
        min_{v} 0.5 x^T A x - b^T x
    subject to x >= 0 and x.sum() = 1.

    :param A: square ndarray representing a positive definite
    and nonnegative matrix
    :param b: ndarray representing a nonnegative vector
    :return: The solution x
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square ndarray")
    if A.shape[0] != b.shape[0]:
        raise ValueError("The dimensions of A and b must match")

    n = len(b)
    fun = lambda x: (0.5 * (x @ A @ x) - b.T @ x)
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
    bounds = [(0, None)] * n
    if zero_dim is not None:
        bounds[zero_dim] = (0, 0)
    x0 = np.zeros(n)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints)
    return res.x