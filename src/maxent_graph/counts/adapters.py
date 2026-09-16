"""
Adapters presenting an existing BiCM or BiECM fit as a DyadModel.

The aggregation utility only needs a dyad's mean, variance and pmf, and both
of those models are dyad-independent with a product-form parameterisation, so
wrapping a solved fit costs nothing and lets the same block-level machinery
run on the models that were already here.
"""

import numpy as np
import scipy.stats

from .base import DyadModel
from .dists import Hurdle, ShiftedGeometric
from .layout import DyadLayout, dense


class DyadTable(DyadModel):
    """
    A dyad model whose parameters were fitted elsewhere.

    ``fit()`` is a no-op; everything else behaves like any other model here.
    """

    def __init__(self, W, layout, dyad_mean, dist_factory, name=None):
        super().__init__(W, layout)
        self._M = np.asarray(dyad_mean, dtype=np.float64) / layout.dyad_scale
        self._dist_factory = dist_factory
        self._name = name

    def __repr__(self):
        return f"DyadTable({self._name or 'custom'}, {self.layout!r})"

    def fit(self):
        return self

    def _dist(self, idx):
        return self._dist_factory(idx)


def _solution_vector(solution):
    return np.asarray(getattr(solution, "x", solution), dtype=np.float64)


def from_bicm(bicm, solution, B=None):
    """
    Wraps a solved :class:`~maxent_graph.bicm.BICM` as a Bernoulli dyad model.

    The BiCM compresses nodes by degree, so the parameters are expanded back
    out to one per node here.
    """
    z = np.asarray(bicm.transform_parameters(_solution_vector(solution)))
    x = z[: bicm.n_row_degrees][bicm.row_inverse]
    y = z[bicm.n_row_degrees :][bicm.col_inverse]

    xy = np.outer(x, y)
    p = xy / (1 + xy)

    B = bicm.B if B is None else B
    A = (dense(B) > 0).astype(np.float64)
    layout = DyadLayout.bipartite(*p.shape)

    def factory(idx):
        return scipy.stats.bernoulli(p if idx is None else p[idx])

    return DyadTable(A, layout, p, factory, name="BICM")


def from_biecm(biecm, solution, W):
    """
    Wraps a solved :class:`~maxent_graph.biecm.BIECM` as a hurdle dyad model.

    The BiECM gives a dyad presence probability ``p`` and, conditional on
    presence, a geometric weight on ``1, 2, ...`` with ratio ``y``, so its
    upper tail is ``p * y**(w - 1)`` -- exactly what ``get_pval_matrix``
    computes edge by edge.
    """
    z = np.asarray(biecm.transform_parameters(_solution_vector(solution)))

    n_rows, n_cols = biecm.num_rows, biecm.num_cols
    x_row = z[:n_rows][biecm.row_inverse]
    x_col = z[n_rows : n_rows + n_cols][biecm.col_inverse]
    y_row = z[n_rows + n_cols : 2 * n_rows + n_cols][biecm.row_inverse]
    y_col = z[2 * n_rows + n_cols :][biecm.col_inverse]

    xx = np.outer(x_row, x_col)
    yy = np.outer(y_row, y_col)
    p = xx * yy / (1 - yy + xx * yy)

    W = dense(W)
    layout = DyadLayout.bipartite(*p.shape)
    mean = p / (1 - yy)

    def factory(idx):
        if idx is None:
            return Hurdle(p, ShiftedGeometric(yy))
        return Hurdle(p[idx], ShiftedGeometric(yy[idx]))

    return DyadTable(W, layout, mean, factory, name="BIECM")
