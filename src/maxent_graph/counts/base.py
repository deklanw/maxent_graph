"""
Base class for dyad-independent count models with a product-form mean.

Every model here factorises over dyads and writes dyad (i, j)'s mean as a
product of a row effect and a column effect, so fitting is always "solve for
two effect vectors" and every distributional quantity is always "evaluate one
frozen distribution per dyad". Subclasses supply the fit and the distribution;
this class supplies the interface.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from .layout import DyadLayout, dense


def solve_product_form(
    update_row,
    update_col,
    x0,
    y0,
    tied,
    tol=1e-12,
    max_iter=5000,
    damping=1.0,
    converged=None,
    check_every=25,
    check_after=0,
    name="model",
):
    """
    Gauss-Seidel fixed-point iteration for a pair of effect vectors.

    ``update_row(x, y)`` returns the next row effects, ``update_col(x, y)`` the
    next column effects. When ``tied`` the two vectors are the same object and
    only ``update_row`` is used -- the layout's support is symmetric in that
    case, so a row update already sees every incidence.

    ``damping`` below one mixes each update with the previous iterate. A tied
    model updates every node against every other at once and can oscillate
    without it; alternating updates generally do not need it.

    ``converged(x, y)`` is an optional second stopping rule, checked every
    ``check_every`` iterations once ``check_after`` have passed. Use it when
    what matters is that the constraints are met rather than that the
    parameters have stopped moving -- the two part company when the maximum
    sits on a boundary the iteration can only creep towards. Holding it back
    for a while first lets the well-behaved cases converge properly instead of
    settling for the looser rule.
    """
    x = np.array(x0, dtype=np.float64)
    y = x if tied else np.array(y0, dtype=np.float64)

    for iteration in range(1, max_iter + 1):
        x_new = np.asarray(update_row(x, y), dtype=np.float64)
        if damping != 1.0:
            x_new = damping * x_new + (1 - damping) * x
        y_new = x_new if tied else np.asarray(update_col(x_new, y), dtype=np.float64)
        if damping != 1.0 and not tied:
            y_new = damping * y_new + (1 - damping) * y

        delta = max(
            np.max(np.abs(x_new - x) / (1 + np.abs(x))),
            np.max(np.abs(y_new - y) / (1 + np.abs(y))),
        )
        x, y = x_new, y_new

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise RuntimeError(f"{name}: fixed-point iteration diverged")

        if delta < tol:
            return x, y, {"iterations": iteration, "delta": delta}

        if (
            converged is not None
            and iteration > check_after
            and iteration % check_every == 0
            and converged(x, y)
        ):
            return (
                x,
                y,
                {
                    "iterations": iteration,
                    "delta": delta,
                    "stopped_on": "constraints",
                },
            )

    warnings.warn(
        f"{name}: fixed-point iteration hit max_iter={max_iter} with "
        f"relative change {delta:.3e}",
        RuntimeWarning,
    )
    return x, y, {"iterations": max_iter, "delta": delta}


class DyadModel(ABC):
    """
    ABC for a dyad-independent count model over an arbitrary dyad set.

    Parameters
    ----------
    W : array or sparse matrix
        Observed non-negative integer weights.
    layout : DyadLayout
        The dyad set. See :mod:`maxent_graph.counts.layout` for the support /
        dyad scale conventions used below.

    Notes
    -----
    Tail conventions: ``cdf(w)`` is ``P(X <= w)`` and ``sf(w)`` is
    ``P(X >= w)``, so the two are not complementary -- ``sf(w) == 1 -
    cdf(w - 1)``. ``sf`` is the one you want for a p-value on an observed
    weight.

    Dyad selection: pass ``dyads`` as a pair of index arrays, an ``(n, 2)``
    array of pairs, or a boolean mask. ``dyads=None`` evaluates everything and
    returns a full matrix. In that case the extensive quantities (``mean``,
    ``var``) are zeroed off the canonical dyads so they sum over the matrix to
    the model total, while the probability-valued ones are filled in over the
    whole support (both triangles for an undirected layout) and are NaN
    elsewhere.
    """

    def __init__(self, W, layout):
        if not isinstance(layout, DyadLayout):
            raise TypeError("layout must be a DyadLayout")

        self.layout = layout
        self.W = dense(W)
        self.support_weights = layout.to_support(self.W)
        self.weights = layout.to_dyads(self.support_weights)

        self.row_strengths = layout.row_totals(self.support_weights)
        self.col_strengths = layout.col_totals(self.support_weights)
        self.total_weight = layout.total(self.support_weights)

        # degrees count incident dyads, so unlike strengths a self-loop counts once
        self.adjacency = (self.weights > 0) & layout.support
        self.row_degrees = self.adjacency.sum(axis=1).astype(np.float64)
        self.col_degrees = self.adjacency.sum(axis=0).astype(np.float64)

        self._M = None
        self.fit_info = {}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self):
        """
        Fits the model's parameters. Returns self.
        """

    @property
    def is_fitted(self):
        return self._M is not None

    def _require_fit(self):
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet. call fit().")

    @property
    def M(self):
        """
        Support-scale mean matrix, whose row and column sums are the expected
        node constraints.
        """
        self._require_fit()
        return self._M

    @property
    def mu(self):
        """
        Dyad-scale mean matrix: entry (i, j) is that dyad's own mean.
        """
        return self.layout.to_dyads(self.M)

    def expected_row_strengths(self):
        return self.layout.row_totals(self.M)

    def expected_col_strengths(self):
        return self.layout.col_totals(self.M)

    def constraint_error(self):
        """
        Largest absolute deviation of an expected node constraint from the
        observed one.
        """
        return max(
            np.max(np.abs(self.expected_row_strengths() - self.row_strengths)),
            np.max(np.abs(self.expected_col_strengths() - self.col_strengths)),
        )

    # ------------------------------------------------------------------
    # distribution
    # ------------------------------------------------------------------

    @abstractmethod
    def _dist(self, idx):
        """
        Returns a frozen distribution over the selected dyads. ``idx`` is
        either None, meaning the whole matrix, or a pair of index arrays.
        """

    def _index(self, dyads):
        """
        Normalises a dyad selector to None or a pair of index arrays.
        """
        if dyads is None:
            return None
        if isinstance(dyads, tuple):
            rows, cols = dyads
            return (np.asarray(rows), np.asarray(cols))
        dyads = np.asarray(dyads)
        if dyads.dtype == bool:
            if dyads.shape != self.layout.shape:
                raise ValueError("boolean dyad mask has the wrong shape")
            return np.nonzero(dyads)
        if dyads.ndim == 2 and dyads.shape[1] == 2:
            return (dyads[:, 0], dyads[:, 1])
        raise ValueError(
            "dyads must be None, a (rows, cols) tuple, an (n, 2) array or a boolean mask"
        )

    def _take(self, mat, idx):
        """
        Selects parameters for the chosen dyads. Used by subclasses building
        their frozen distributions.
        """
        return mat if idx is None else np.asarray(mat)[idx]

    def _mask_extensive(self, values, idx):
        if idx is None:
            return np.where(self.layout.canonical, values, 0.0)
        return values

    def _mask_prob(self, values, idx):
        if idx is None:
            return np.where(self.layout.support, values, np.nan)
        return values

    def mean(self, dyads=None):
        """
        Expected weight of each selected dyad.
        """
        idx = self._index(dyads)
        values = np.asarray(self._dist(idx).mean(), dtype=np.float64)
        return self._mask_extensive(values, idx)

    def var(self, dyads=None):
        """
        Variance of the weight of each selected dyad.
        """
        idx = self._index(dyads)
        values = np.asarray(self._dist(idx).var(), dtype=np.float64)
        return self._mask_extensive(values, idx)

    def pmf(self, w, dyads=None):
        """
        P(X == w).
        """
        idx = self._index(dyads)
        return self._mask_prob(
            np.asarray(self._dist(idx).pmf(w), dtype=np.float64), idx
        )

    def cdf(self, w, dyads=None):
        """
        P(X <= w).
        """
        idx = self._index(dyads)
        return self._mask_prob(
            np.asarray(self._dist(idx).cdf(w), dtype=np.float64), idx
        )

    def sf(self, w, dyads=None):
        """
        Upper tail P(X >= w). Note the inclusive bound: scipy's own sf is
        P(X > w), so this is ``scipy_sf(w - 1)``.
        """
        idx = self._index(dyads)
        w = np.asarray(w)
        values = np.asarray(self._dist(idx).sf(w - 1), dtype=np.float64)
        return self._mask_prob(values, idx)

    def cell_distribution(self, dyads):
        """
        Exact distribution of the summed weight over ``dyads``, or None when
        the family has no closed form for it.

        Used by the partition aggregation utility to give a block its own tail
        probability instead of a normal approximation.
        """
        return

    def loglik(self):
        """
        Log-likelihood of the observed weights under the fitted model.
        """
        pairs = self.layout.canonical_pairs()
        observed = self.weights[pairs]
        return float(np.sum(np.log(self._dist(pairs).pmf(observed))))

    def sample(self, n=1, rng=None):
        """
        Draws ``n`` weight matrices from the fitted model.

        Returns an ``(n, n_row, n_col)`` array, symmetric for an undirected
        layout.
        """
        self._require_fit()
        rng = np.random.default_rng(rng)
        pairs = self.layout.canonical_pairs()
        values = self._dist(pairs).rvs(
            size=(int(n), self.layout.n_dyads), random_state=rng
        )
        return self.layout.expand(np.asarray(values, dtype=np.float64), pairs)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def summary(self):
        """
        A short dict describing the fit, handy in notebooks.
        """
        self._require_fit()
        return {
            "model": type(self).__name__,
            "layout": repr(self.layout),
            "n_dyads": self.layout.n_dyads,
            "total_weight": self.total_weight,
            "constraint_error": self.constraint_error(),
            "loglik": self.loglik(),
            **self.fit_info,
        }
