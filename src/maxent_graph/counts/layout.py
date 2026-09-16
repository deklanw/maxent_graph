"""
Dyad-set layouts for the count-valued configuration models.

Every model in the Poisson, negative binomial and hurdle families gives dyad
(i, j) a mean of product form, so the only thing separating the bipartite,
directed and undirected cases is *which* index pairs are dyads and how a dyad
enters the node constraints. A ``DyadLayout`` is that bookkeeping and nothing
else.

Two matrix conventions are used throughout, and keeping them straight is the
whole point of this class:

support scale
    An ``(n_row, n_col)`` matrix whose row sums are the row constraints and
    whose column sums are the column constraints. The mean matrix
    ``M = outer(x, y)`` lives here, as does the observed weight matrix. An
    undirected self-loop appears here doubled, because it consumes two stubs.

dyad scale
    The same matrix multiplied by ``dyad_scale``: one entry per index pair
    holding that pair's own mean or observed weight. Every distribution lives
    on this scale. Restricted to ``canonical`` it has exactly one entry per
    dyad, so summing it gives the total weight.

The two coincide everywhere except the diagonal of an undirected layout with
self-loops, where ``dyad_scale`` is 1/2.
"""

import numpy as np
import scipy.sparse as sp

KINDS = ("bipartite", "undirected", "directed")


def dense(A, dtype=np.float64):
    """
    Coerces a sparse matrix, matrix or array to a plain dense ndarray.
    """
    if sp.issparse(A):
        A = A.todense() if hasattr(A, "todense") else A.toarray()
    return np.asarray(A, dtype=dtype)


class DyadLayout:
    """
    The dyad set of a dyad-independent model, plus the two scale conventions
    relating a dyad's own distribution to the node constraints.

    Attributes
    ----------
    support : (n_row, n_col) bool
        Index pairs that participate at all. Node constraints are row and
        column sums over this mask.
    canonical : (n_row, n_col) bool
        One entry per dyad: ``support`` minus the redundant lower triangle for
        an undirected layout.
    dyad_scale : (n_row, n_col) float
        Multiplier taking a support-scale matrix to dyad scale.
    tied : bool
        Row and column effects are the same vector (undirected).
    """

    def __init__(self, kind, n_row, n_col=None, self_loops=False):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

        if kind == "bipartite":
            if n_col is None:
                raise ValueError("a bipartite layout needs n_col")
            if self_loops:
                raise ValueError("self_loops is meaningless for a bipartite layout")
        else:
            if n_col is None:
                n_col = n_row
            if n_col != n_row:
                raise ValueError(f"a {kind} layout must be square")

        self.kind = kind
        self.n_row = int(n_row)
        self.n_col = int(n_col)
        self.self_loops = bool(self_loops)
        self.tied = kind == "undirected"

        support = np.ones((self.n_row, self.n_col), dtype=bool)
        if kind != "bipartite" and not self_loops:
            np.fill_diagonal(support, False)

        # for the undirected case each unordered pair sits in the support twice,
        # once as (i, j) and once as (j, i). only the upper triangle is a dyad.
        canonical = np.triu(support) if self.tied else support.copy()

        dyad_scale = np.ones((self.n_row, self.n_col))
        if self.tied and self_loops:
            np.fill_diagonal(dyad_scale, 0.5)

        self.support = support
        self.canonical = canonical
        self.dyad_scale = dyad_scale
        self.n_dyads = int(canonical.sum())

    @classmethod
    def bipartite(cls, n_row, n_col):
        return cls("bipartite", n_row, n_col)

    @classmethod
    def undirected(cls, n, self_loops=False):
        return cls("undirected", n, n, self_loops=self_loops)

    @classmethod
    def directed(cls, n, self_loops=False):
        return cls("directed", n, n, self_loops=self_loops)

    def __repr__(self):
        return (
            f"DyadLayout(kind={self.kind!r}, n_row={self.n_row}, "
            f"n_col={self.n_col}, self_loops={self.self_loops})"
        )

    @property
    def shape(self):
        return (self.n_row, self.n_col)

    def to_support(self, A):
        """
        Validates a user-supplied weight matrix and puts it on the support
        scale, so that its row and column sums are the node constraints.
        """
        A = dense(A)

        if A.shape != self.shape:
            raise ValueError(f"expected a {self.shape} matrix, got {A.shape}")
        if np.any(A < 0):
            raise ValueError("weights must be non-negative")

        if self.tied and not np.allclose(A, A.T):
            raise ValueError("an undirected layout needs a symmetric weight matrix")

        if self.kind != "bipartite" and not self.self_loops:
            if np.any(np.diag(A) != 0):
                raise ValueError(
                    "the diagonal carries weight but self_loops=False. "
                    "pass self_loops=True or zero the diagonal."
                )

        S = np.where(self.support, A, 0.0)
        if self.tied and self.self_loops:
            # a self-loop uses two of node i's stubs
            S[np.diag_indices(self.n_row)] *= 2
        return S

    def to_dyads(self, S):
        """
        Support scale -> dyad scale, zeroed off the support.
        """
        return np.where(self.support, np.asarray(S) * self.dyad_scale, 0.0)

    def row_totals(self, S):
        """
        Row constraints (strengths / out-strengths) of a support-scale matrix.
        """
        return np.where(self.support, S, 0.0).sum(axis=1)

    def col_totals(self, S):
        """
        Column constraints (strengths / in-strengths) of a support-scale matrix.
        """
        return np.where(self.support, S, 0.0).sum(axis=0)

    def total(self, S):
        """
        Total weight: the sum over dyads, each counted once.
        """
        return float(np.where(self.canonical, self.to_dyads(S), 0.0).sum())

    def canonical_pairs(self):
        """
        Row and column indices of every dyad, each listed once.
        """
        return np.nonzero(self.canonical)

    def expand(self, values, pairs=None):
        """
        Builds a full weight matrix from per-dyad values, mirroring the upper
        triangle for undirected layouts.

        ``values`` may have a leading sample dimension.
        """
        if pairs is None:
            pairs = self.canonical_pairs()
        rows, cols = pairs
        values = np.asarray(values)
        out = np.zeros(values.shape[:-1] + self.shape, dtype=values.dtype)
        out[..., rows, cols] = values
        if self.tied:
            # mirror, taking care not to double the diagonal
            off = rows != cols
            out[..., cols[off], rows[off]] = values[..., off]
        return out
