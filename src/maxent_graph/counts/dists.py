"""
Small frozen-distribution objects used by the dyad models.

These implement enough of scipy's frozen-distribution interface for
:class:`~maxent_graph.counts.base.DyadModel` to use them interchangeably with
the real thing: ``pmf``, ``cdf``, ``sf``, ``mean``, ``var`` and ``rvs``.
Parameters are arrays and everything broadcasts.

Note that ``sf`` follows scipy and is the *strict* upper tail ``P(X > k)``;
``DyadModel.sf`` is the inclusive one and calls ``sf(w - 1)``.
"""

import numpy as np
import scipy.stats

TINY = 1e-300


class ZeroTruncatedPoisson:
    """
    Poisson(lam) conditioned on being at least one.

    Stays accurate as ``lam -> 0``, where the distribution collapses onto a
    point mass at one -- which is where the hurdle models end up when every
    observed positive weight is one.
    """

    def __init__(self, lam):
        self.lam = np.maximum(np.asarray(lam, dtype=np.float64), TINY)
        # -expm1(-lam) is 1 - exp(-lam) without the cancellation
        self.denominator = -np.expm1(-self.lam)

    def mean(self):
        return self.lam / self.denominator

    def var(self):
        m = self.mean()
        return m * (self.lam + 1 - m)

    def pmf(self, w):
        w = np.asarray(w)
        out = scipy.stats.poisson.pmf(w, self.lam) / self.denominator
        return np.where(w >= 1, out, 0.0)

    def cdf(self, k):
        k = np.floor(np.asarray(k))
        out = (scipy.stats.poisson.cdf(k, self.lam) - np.exp(-self.lam)) / (
            self.denominator
        )
        return np.where(k < 1, 0.0, np.clip(out, 0.0, 1.0))

    def sf(self, k):
        k = np.floor(np.asarray(k))
        out = scipy.stats.poisson.sf(k, self.lam) / self.denominator
        return np.where(k < 1, 1.0, np.clip(out, 0.0, 1.0))

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        p0 = np.exp(-self.lam)
        u = rng.random(size if size is not None else np.shape(self.lam))
        # keep the quantile strictly inside (p0, 1): rounding onto p0 sends ppf
        # to zero and rounding onto 1 sends it to infinity, and the support
        # starts at one, so the low end can only have meant one
        q = np.minimum(p0 + u * (1 - p0), np.nextafter(1.0, 0.0))
        return np.maximum(scipy.stats.poisson.ppf(q, self.lam), 1.0)


class ShiftedGeometric:
    """
    ``P(X = w) = (1 - y) y**(w - 1)`` on ``w = 1, 2, ...``.

    The positive part of the BiECM and of the enhanced configuration models
    generally.
    """

    def __init__(self, y):
        self.y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1 - 1e-15)

    def mean(self):
        return 1.0 / (1.0 - self.y)

    def var(self):
        return self.y / (1.0 - self.y) ** 2

    def pmf(self, w):
        w = np.asarray(w)
        out = (1 - self.y) * self.y ** np.maximum(w - 1, 0)
        return np.where(w >= 1, out, 0.0)

    def sf(self, k):
        k = np.floor(np.asarray(k))
        return np.where(k < 0, 1.0, self.y ** np.maximum(k, 0))

    def cdf(self, k):
        return 1.0 - self.sf(k)

    def rvs(self, size=None, random_state=None):
        return scipy.stats.geom.rvs(
            1 - self.y, size=size, random_state=np.random.default_rng(random_state)
        )


class Hurdle:
    """
    Zero with probability ``1 - p``, otherwise a draw from ``positive``, which
    must be supported on the strictly positive integers.

    Equivalently a zero-inflated distribution: a ZIP with inflation ``1 - pi``
    and rate ``lam`` is this with ``p = pi * (1 - exp(-lam))`` and a
    zero-truncated Poisson positive part.
    """

    def __init__(self, p, positive):
        self.p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        self.positive = positive

    def mean(self):
        return self.p * self.positive.mean()

    def var(self):
        m = self.positive.mean()
        return self.p * self.positive.var() + self.p * (1 - self.p) * m**2

    def pmf(self, w):
        w = np.asarray(w)
        return np.where(w == 0, 1 - self.p, self.p * self.positive.pmf(w))

    def sf(self, k):
        k = np.floor(np.asarray(k))
        return np.where(k < 0, 1.0, self.p * self.positive.sf(k))

    def cdf(self, k):
        return 1.0 - self.sf(k)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        if size is None:
            size = np.shape(self.p)
        present = rng.random(size) < self.p
        return np.where(present, self.positive.rvs(size=size, random_state=rng), 0.0)
