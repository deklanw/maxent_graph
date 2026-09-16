"""
Small networks shared by the tests for the count models.
"""

import numpy as np

from maxent_graph.util import nx_get_A, nx_get_B


def random_bipartite(n_row=7, n_col=9, seed=0, density=0.6, lam=3.0):
    rng = np.random.default_rng(seed)
    B = rng.poisson(lam, size=(n_row, n_col)) * (rng.random((n_row, n_col)) < density)
    # no empty rows or columns
    B[np.arange(n_row), rng.integers(0, n_col, n_row)] += 1
    B[rng.integers(0, n_row, n_col), np.arange(n_col)] += 1
    return B.astype(float)


def random_undirected(n=8, seed=1, density=0.7, lam=3.0, self_loops=False):
    rng = np.random.default_rng(seed)
    A = rng.poisson(lam, size=(n, n)) * (rng.random((n, n)) < density)
    A = np.triu(A, 0 if self_loops else 1)
    A = A + np.triu(A, 1).T
    if not self_loops:
        np.fill_diagonal(A, 0)
    A[np.arange(n - 1), np.arange(1, n)] += 1
    A[np.arange(1, n), np.arange(n - 1)] += 1
    return A.astype(float)


def random_directed(n=8, seed=2, density=0.7, lam=3.0, self_loops=False):
    rng = np.random.default_rng(seed)
    A = rng.poisson(lam, size=(n, n)) * (rng.random((n, n)) < density)
    if not self_loops:
        np.fill_diagonal(A, 0)
    A[np.arange(n - 1), np.arange(1, n)] += 1
    A[np.arange(1, n), np.arange(n - 1)] += 1
    return A.astype(float)


def kato():
    return np.asarray(
        nx_get_B(
            "data/plant_pol_kato.graphml",
            weight_key="count",
            bipartite_key="pollinator",
        ).todense(),
        dtype=float,
    )


def kangaroo():
    return np.asarray(
        nx_get_A("data/kangaroo.graphml", weight_key="weight").todense(), dtype=float
    )


def residence_hall():
    return np.asarray(
        nx_get_A("data/residence_hall.graphml", weight_key="weight").todense(),
        dtype=float,
    )
