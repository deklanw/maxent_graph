"""
Utility U: partition aggregation.

A fitted dyad model answers node-level questions -- is this edge surprising?
This turns it into block-level ones: given a partition of the rows and a
partition of the columns, is the total weight flowing from block r to block s
surprising? Because every model here is dyad-independent, a cell's expectation
and variance are just sums over its dyads, and for the Poisson family the
cell total has an exact distribution of its own.

The singleton partition, where every node is its own block, recovers the
per-dyad answer exactly -- a one-dyad cell is answered by the dyad's own tail
rather than convolved -- which makes it a regression test against the existing
BiECM p-values rather than a separate code path.
"""

import numpy as np
import pandas as pd
import scipy.stats
from scipy.signal import fftconvolve

from .layout import dense

METHODS = ("auto", "exact", "fft", "normal")

COLUMNS = [
    "row_block",
    "col_block",
    "n_dyads",
    "observed",
    "expected",
    "variance",
    "enrichment",
    "z",
    "p_upper",
    "p_lower",
    "coverage_row",
    "coverage_col",
    "method",
]


def _cell_pmf(model, dyads, cap):
    """
    Convolves the dyad pmfs of one cell into the pmf of the cell total,
    truncated above ``cap``.

    Truncation costs nothing below ``cap``: dropped per-dyad mass can only
    ever land above it, since weights are non-negative. So the lower
    cumulative -- the only part the tail probabilities need -- is exact.
    """
    support = np.arange(cap + 1)
    pmfs = np.asarray(model.pmf(support[:, None], dyads), dtype=np.float64)

    total = np.zeros(cap + 1)
    total[0] = 1.0
    for j in range(pmfs.shape[1]):
        total = fftconvolve(total, pmfs[:, j])[: cap + 1]
    return np.clip(total, 0.0, 1.0)


def _cell_tails(
    model,
    dyads,
    observed,
    expected,
    variance,
    method,
    sigma,
    max_fft_dyads,
    distribution=None,
):
    """
    Returns ``(p_upper, p_lower, method_used)`` for one cell, where p_upper is
    ``P(total >= observed)`` and p_lower is ``P(total <= observed)``.

    ``distribution`` is the family's exact cell distribution when it has one.
    """
    observed = round(observed)

    if method in ("auto", "exact", "fft") and len(dyads[0]) == 1:
        # a cell holding one dyad is that dyad: no convolution, and no
        # floating-point drift away from the model's own tail
        return (
            float(np.clip(model.sf(observed, dyads)[0], 0.0, 1.0)),
            float(np.clip(model.cdf(observed, dyads)[0], 0.0, 1.0)),
            "exact",
        )

    if method in ("auto", "exact"):
        if distribution is not None:
            return (
                float(np.clip(distribution.sf(observed - 1), 0.0, 1.0)),
                float(np.clip(distribution.cdf(observed), 0.0, 1.0)),
                "exact",
            )
        if method == "exact":
            raise ValueError(
                f"{type(model).__name__} has no exact cell distribution; "
                "use method='fft' or method='normal'"
            )
        method = "fft" if len(dyads[0]) <= max_fft_dyads else "normal"

    if method == "fft":
        sd = np.sqrt(variance)
        cap = int(max(observed, np.ceil(expected + sigma * sd))) + 1
        pmf = _cell_pmf(model, dyads, cap)
        lower = pmf[: observed + 1].sum()
        return (
            float(np.clip(1.0 - pmf[:observed].sum(), 0.0, 1.0)),
            float(np.clip(lower, 0.0, 1.0)),
            "fft",
        )

    if variance <= 0:
        # a total that cannot vary is a point mass at its mean
        return (
            1.0 if observed <= expected else 0.0,
            1.0 if observed >= expected else 0.0,
            "normal",
        )
    sd = np.sqrt(variance)
    # continuity correction, since the cell total is integer valued
    upper = scipy.stats.norm.sf((observed - 0.5 - expected) / sd)
    lower = scipy.stats.norm.cdf((observed + 0.5 - expected) / sd)
    return float(upper), float(lower), "normal"


def aggregate_blocks(
    model,
    row_blocks,
    col_blocks=None,
    W=None,
    method="auto",
    max_fft_dyads=512,
    sigma=10.0,
):
    """
    Aggregates a fitted dyad model over a partition of the rows and columns.

    Parameters
    ----------
    model : DyadModel
        Any fitted model from families A, B or C, or an existing BiCM / BiECM
        fit wrapped by :mod:`maxent_graph.counts.adapters`.
    row_blocks, col_blocks : array-like
        Block label per row node and per column node. Labels can be anything
        pandas can factorize. ``col_blocks`` defaults to ``row_blocks`` for a
        unipartite layout and is required for a bipartite one. An undirected
        layout needs the two to be the same partition, and its cells are
        unordered block pairs.
    W : array or sparse matrix, optional
        Observed weights, defaulting to the matrix the model was built on.
    method : {"auto", "exact", "fft", "normal"}
        How to get each cell's tail probability. ``"exact"`` uses the family's
        own cell distribution (Poisson for family A, hypergeometric for
        ``exact=True`` stub matching) and errors if there isn't one.
        ``"fft"`` convolves the dyad pmfs, which is exact up to floating
        point. ``"normal"`` uses a normal approximation with a continuity
        correction; it drifts badly for the overdispersed families, where a
        cell total can be far from normal, so prefer the convolution when the
        cell is small enough to afford it. ``"auto"`` takes the exact
        distribution when it exists, else convolves cells of at most
        ``max_fft_dyads`` dyads, else approximates.
    sigma : float
        How many standard deviations above the mean to truncate a convolution.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty cell, with the observed and expected totals,
        the variance of the cell total -- from the family's own cell
        distribution where it has one, so dependent dyads are handled -- the enrichment ratio, a z-score, upper and lower tail
        probabilities, and the per-side coverage: the fraction of nodes in the
        row block with at least one edge into the column block, and vice
        versa.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    layout = model.layout
    row_blocks = np.asarray(row_blocks)

    if col_blocks is None:
        if layout.kind == "bipartite":
            raise ValueError("a bipartite model needs col_blocks")
        col_blocks = row_blocks
    col_blocks = np.asarray(col_blocks)

    if len(row_blocks) != layout.n_row or len(col_blocks) != layout.n_col:
        raise ValueError("block labels must be one per node")
    if layout.tied and not np.array_equal(row_blocks, col_blocks):
        raise ValueError("an undirected model needs one partition, not two")

    if layout.tied:
        codes, labels = pd.factorize(row_blocks)
        row_codes = col_codes = codes
        row_labels = col_labels = labels
    else:
        row_codes, row_labels = pd.factorize(row_blocks)
        col_codes, col_labels = pd.factorize(col_blocks)

    row_sizes = np.bincount(row_codes, minlength=len(row_labels))
    col_sizes = np.bincount(col_codes, minlength=len(col_labels))

    observed_matrix = layout.to_dyads(
        layout.to_support(model.W if W is None else dense(W))
    )

    rows, cols = layout.canonical_pairs()
    observed = observed_matrix[rows, cols]
    mean = np.asarray(model.mean((rows, cols)), dtype=np.float64)
    variance = np.asarray(model.var((rows, cols)), dtype=np.float64)

    cell_rows = row_codes[rows]
    cell_cols = col_codes[cols]
    if layout.tied:
        # an unordered pair of blocks: (r, s) and (s, r) are one cell
        cell_rows, cell_cols = (
            np.minimum(cell_rows, cell_cols),
            np.maximum(cell_rows, cell_cols),
        )

    key = cell_rows * len(col_labels) + cell_cols
    order = np.argsort(key, kind="stable")
    boundaries = np.flatnonzero(np.diff(key[order])) + 1

    records = []
    for cell in np.split(order, boundaries):
        r = int(cell_rows[cell[0]])
        s = int(cell_cols[cell[0]])
        dyads = (rows[cell], cols[cell])

        total = float(observed[cell].sum())
        distribution = model.cell_distribution(dyads)
        if distribution is not None:
            # the family knows the cell total's own law, which matters when
            # its dyads are dependent: under exact stub matching the dyad
            # variances do not add, and a fixed grand total has variance zero
            expected = float(distribution.mean())
            cell_variance = float(distribution.var())
        else:
            expected = float(mean[cell].sum())
            cell_variance = float(variance[cell].sum())

        p_upper, p_lower, used = _cell_tails(
            model,
            dyads,
            total,
            expected,
            cell_variance,
            method,
            sigma,
            max_fft_dyads,
            distribution=distribution,
        )

        present = cell[observed[cell] > 0]
        if layout.tied:
            endpoints = np.concatenate([rows[present], cols[present]])
            touched_rows = np.unique(endpoints[row_codes[endpoints] == r])
            touched_cols = np.unique(endpoints[col_codes[endpoints] == s])
        else:
            touched_rows = np.unique(rows[present])
            touched_cols = np.unique(cols[present])

        records.append(
            {
                "row_block": row_labels[r],
                "col_block": col_labels[s],
                "n_dyads": len(cell),
                "observed": total,
                "expected": expected,
                "variance": cell_variance,
                "enrichment": total / expected if expected > 0 else np.nan,
                "z": (total - expected) / np.sqrt(cell_variance)
                if cell_variance > 0
                else np.nan,
                "p_upper": p_upper,
                "p_lower": p_lower,
                "coverage_row": len(touched_rows) / row_sizes[r],
                "coverage_col": len(touched_cols) / col_sizes[s],
                "method": used,
            }
        )

    return pd.DataFrame.from_records(records, columns=COLUMNS)
