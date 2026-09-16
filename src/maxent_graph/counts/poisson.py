"""
Family A: Poisson (multi-edge) configuration models.

The count-aware counterpart of the BWCM: weights are Poisson rather than
geometric, so ``var == mean`` and the maximum-likelihood mean is the familiar
product of strengths over the total. Constraints are the expected strengths.
"""

import warnings

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import gammaln

from .base import DyadModel, solve_product_form
from .layout import DyadLayout, dense


class PoissonCM(DyadModel):
    """
    Poisson configuration model over an arbitrary dyad set.

    ``w_ij ~ Poisson(x_i y_j)``, with the expected node strengths constrained
    to the observed ones. When every index pair is in the support -- bipartite,
    or unipartite with self-loops -- the MLE is closed form. Dropping the
    diagonal breaks that, and the effect vectors are found by fixed-point
    iteration instead.

    Parameters
    ----------
    exact : bool
        Switch to the microcanonical stub-matching model, in which every
        strength is fixed exactly rather than in expectation. The dyad marginal
        becomes ``Hypergeometric(W, s_j, s_i)``: the mean is unchanged but the
        variance is deflated by the finite-population correction. Bipartite
        only.
    """

    def __init__(self, W, layout, exact=False):
        super().__init__(W, layout)

        if exact:
            if layout.kind != "bipartite":
                raise NotImplementedError(
                    "exact stub matching is only implemented for bipartite layouts"
                )
            for name, s in (
                ("row", self.row_strengths),
                ("column", self.col_strengths),
            ):
                if not np.allclose(s, np.round(s)):
                    raise ValueError(
                        f"exact stub matching needs integer weights, but the {name} "
                        "strengths are not integral"
                    )

        self.exact = bool(exact)
        self.row_effects = None
        self.col_effects = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    @property
    def has_closed_form(self):
        """
        True when the support is the full index grid, in which case the score
        equations are solved by the strength product.
        """
        return bool(self.layout.support.all())

    def _closed_form(self):
        # sum of the row strengths equals sum of the column strengths, and is
        # W for bipartite and directed layouts and 2W for undirected ones --
        # exactly the normalisation that makes x_i * sum(y) == s_i.
        total = self.row_strengths.sum()
        if total <= 0:
            raise ValueError("the network has no weight")
        scale = np.sqrt(total)
        return self.row_strengths / scale, self.col_strengths / scale

    def _solve_tied(self):
        """
        Solves the undirected score equations by a scalar root find.

        Every row effect satisfies ``x_i (X - c x_i) = s_i`` with ``X = sum(x)``
        and ``c`` one when the diagonal is out of the support, so given ``X``
        each ``x_i`` is available in closed form and only ``X`` is unknown.
        ``sum(x)(X) - X`` is strictly decreasing, which makes the bracketed
        solve unconditional -- and lets a failed bracket be reported honestly
        as "no interior maximum" rather than as a failure to converge.
        """
        s = self.row_strengths
        total = s.sum()

        if self.layout.self_loops:
            x_of_X = lambda X: s / X
            lo = np.sqrt(total) / 2
        else:
            # x_i is the smaller root of x^2 - X x + s_i; at most one node could
            # sit on the larger root, and then sum(x) > X, which is impossible.
            x_of_X = lambda X: (X - np.sqrt(np.maximum(X * X - 4 * s, 0.0))) / 2
            lo = max(2 * np.sqrt(s.max()), np.sqrt(total)) * (1 + 1e-12)

        g = lambda X: x_of_X(X).sum() - X

        if g(lo) < 0:
            raise RuntimeError(
                f"{type(self).__name__}: the expected strengths cannot be matched "
                "with the diagonal excluded -- the likelihood has no interior "
                "maximum for this network (a star is the standard example). "
                "Allow self-loops with self_loops=True."
            )

        hi = 2 * lo
        for _ in range(200):
            if g(hi) < 0:
                break
            hi *= 2
        else:
            raise RuntimeError(f"{type(self).__name__}: could not bracket sum(x)")

        X = scipy.optimize.brentq(g, lo, hi, xtol=1e-14, rtol=8.9e-16, maxiter=500)
        x = x_of_X(X)
        return x, {"method": "scalar root find", "sum_of_effects": X}

    def _solve_alternating(self, x0, y0, tol, max_iter):
        support = self.layout.support.astype(np.float64)

        def update_row(x, y):
            return self.row_strengths / (support @ y)

        def update_col(x, y):
            return self.col_strengths / (support.T @ x)

        x, y, info = solve_product_form(
            update_row,
            update_col,
            x0,
            y0,
            tied=False,
            tol=tol,
            max_iter=max_iter,
            name=type(self).__name__,
        )
        return x, y, {"method": "fixed point", **info}

    def fit(self, method="auto", tol=1e-12, max_iter=5000):
        """
        Fits the effect vectors.

        ``method="auto"`` uses the closed form where it exists and solves
        numerically otherwise; ``method="numerical"`` always solves numerically,
        which is what the closed form is tested against.
        """
        if method not in ("auto", "closed", "numerical"):
            raise ValueError("method must be one of 'auto', 'closed', 'numerical'")
        if method == "closed" and not self.has_closed_form:
            raise ValueError(
                "no closed form without the full index grid; pass self_loops=True "
                "or use method='numerical'"
            )

        x0, y0 = self._closed_form()

        if method == "closed" or (method == "auto" and self.has_closed_form):
            x, y = x0, y0
            info = {"method": "closed form"}
        elif self.layout.tied:
            x, info = self._solve_tied()
            y = x
        else:
            x, y, info = self._solve_alternating(x0, y0, tol, max_iter)

        self.row_effects, self.col_effects = x, y
        self.fit_info = info
        self._M = np.outer(x, y)

        error = self.constraint_error()
        if error > 1e-8 * max(1.0, self.total_weight):
            warnings.warn(
                f"{type(self).__name__}: expected strengths are off by up to {error:.3e}",
                RuntimeWarning,
            )
        self.fit_info["constraint_error"] = error
        return self

    # ------------------------------------------------------------------
    # distribution
    # ------------------------------------------------------------------

    def _dist(self, idx):
        self._require_fit()
        if not self.exact:
            return scipy.stats.poisson(self._take(self.mu, idx))

        shape = self.layout.shape
        row = np.broadcast_to(self.row_strengths[:, None], shape)
        col = np.broadcast_to(self.col_strengths[None, :], shape)
        return scipy.stats.hypergeom(
            M=round(self.total_weight),
            n=np.round(self._take(col, idx)).astype(np.int64),
            N=np.round(self._take(row, idx)).astype(np.int64),
        )

    def loglik(self):
        if not self.exact:
            return super().loglik()

        # under stub matching every pairing of the W row stubs with the W column
        # stubs is equally likely, and the number of pairings realising a given
        # table is prod_i s_i! prod_j s_j! / prod_ij w_ij!
        w = self.weights[self.layout.canonical_pairs()]
        return float(
            gammaln(self.row_strengths + 1).sum()
            + gammaln(self.col_strengths + 1).sum()
            - gammaln(self.total_weight + 1)
            - gammaln(w + 1).sum()
        )

    def sample(self, n=1, rng=None):
        if not self.exact:
            return super().sample(n=n, rng=rng)

        # the hypergeometric dyad marginals are not independent, so sample the
        # configuration itself: shuffle the column stubs against the row stubs
        rng = np.random.default_rng(rng)
        row_stubs = np.repeat(
            np.arange(self.layout.n_row), np.round(self.row_strengths).astype(np.int64)
        )
        col_stubs = np.repeat(
            np.arange(self.layout.n_col), np.round(self.col_strengths).astype(np.int64)
        )
        flat_size = self.layout.n_row * self.layout.n_col

        out = np.empty((int(n),) + self.layout.shape)
        for k in range(int(n)):
            shuffled = rng.permutation(col_stubs)
            flat = np.bincount(
                row_stubs * self.layout.n_col + shuffled, minlength=flat_size
            )
            out[k] = flat.reshape(self.layout.shape)
        return out


class BIPCM(PoissonCM):
    """
    Bipartite Poisson configuration model.

    ``w_ia ~ Poisson(x_i y_a)`` with expected row and column strengths
    constrained to the observed ones, so ``mean == s_i * s_a / W``.
    """

    def __init__(self, B, exact=False):
        B = dense(B)
        super().__init__(B, DyadLayout.bipartite(*B.shape), exact=exact)


class UPCM(PoissonCM):
    """
    Undirected Poisson configuration model.

    ``w_ij ~ Poisson(x_i x_j)`` over unordered pairs.
    """

    def __init__(self, A, self_loops=False):
        A = dense(A)
        super().__init__(A, DyadLayout.undirected(A.shape[0], self_loops=self_loops))


class DPCM(PoissonCM):
    """
    Directed Poisson configuration model.

    ``w_ij ~ Poisson(x_i^out y_j^in)`` over ordered pairs.
    """

    def __init__(self, A, self_loops=False):
        A = dense(A)
        super().__init__(A, DyadLayout.directed(A.shape[0], self_loops=self_loops))
