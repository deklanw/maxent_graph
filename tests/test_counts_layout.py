import numpy as np
import pytest
import scipy.sparse as sp

from maxent_graph.counts import DyadLayout


def symmetric(n, seed=0, loops=False):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 5, size=(n, n)).astype(float)
    A = A + A.T
    if not loops:
        np.fill_diagonal(A, 0)
    return A


def test_constructors_and_dyad_counts():
    assert DyadLayout.bipartite(3, 4).n_dyads == 12
    assert DyadLayout.undirected(5).n_dyads == 10
    assert DyadLayout.undirected(5, self_loops=True).n_dyads == 15
    assert DyadLayout.directed(5).n_dyads == 20
    assert DyadLayout.directed(5, self_loops=True).n_dyads == 25


def test_constructor_validation():
    with pytest.raises(ValueError):
        DyadLayout("bipartite", 3)
    with pytest.raises(ValueError):
        DyadLayout("bipartite", 3, 4, self_loops=True)
    with pytest.raises(ValueError):
        DyadLayout("undirected", 3, 4)
    with pytest.raises(ValueError):
        DyadLayout("nonsense", 3, 3)


def test_bipartite_scales_are_the_identity():
    rng = np.random.default_rng(1)
    B = rng.integers(0, 6, size=(4, 6)).astype(float)
    layout = DyadLayout.bipartite(4, 6)

    S = layout.to_support(B)
    np.testing.assert_allclose(S, B)
    np.testing.assert_allclose(layout.to_dyads(S), B)
    np.testing.assert_allclose(layout.row_totals(S), B.sum(axis=1))
    np.testing.assert_allclose(layout.col_totals(S), B.sum(axis=0))
    assert layout.total(S) == pytest.approx(B.sum())


def test_sparse_input_is_accepted():
    B = np.array([[0.0, 2.0], [3.0, 0.0]])
    layout = DyadLayout.bipartite(2, 2)
    np.testing.assert_allclose(
        layout.to_support(sp.csr_matrix(B)), layout.to_support(B)
    )


def test_undirected_totals_and_double_counted_loops():
    A = symmetric(6, loops=True)
    layout = DyadLayout.undirected(6, self_loops=True)
    S = layout.to_support(A)

    # strengths double count self-loops, and the total is half the strength sum
    expected = A.sum(axis=1) + np.diag(A)
    np.testing.assert_allclose(layout.row_totals(S), expected)
    assert layout.total(S) == pytest.approx(expected.sum() / 2)

    # the dyad itself is not doubled
    dyads = layout.to_dyads(S)
    np.testing.assert_allclose(np.diag(dyads), np.diag(A))
    np.testing.assert_allclose(layout.total(S), dyads[np.triu_indices(6)].sum())


def test_undirected_without_loops():
    A = symmetric(6)
    layout = DyadLayout.undirected(6)
    S = layout.to_support(A)

    np.testing.assert_allclose(layout.row_totals(S), A.sum(axis=1))
    assert layout.total(S) == pytest.approx(A.sum() / 2)
    assert not layout.support[np.diag_indices(6)].any()


def test_directed_totals():
    rng = np.random.default_rng(3)
    A = rng.integers(0, 4, size=(5, 5)).astype(float)
    np.fill_diagonal(A, 0)
    layout = DyadLayout.directed(5)
    S = layout.to_support(A)

    np.testing.assert_allclose(layout.row_totals(S), A.sum(axis=1))
    np.testing.assert_allclose(layout.col_totals(S), A.sum(axis=0))
    assert layout.total(S) == pytest.approx(A.sum())


def test_input_validation():
    layout = DyadLayout.undirected(3)
    with pytest.raises(ValueError, match="symmetric"):
        layout.to_support(np.array([[0.0, 1, 2], [0, 0, 1], [0, 0, 0]]))
    with pytest.raises(ValueError, match="diagonal"):
        layout.to_support(np.eye(3))
    with pytest.raises(ValueError, match="non-negative"):
        layout.to_support(-np.ones((3, 3)))
    with pytest.raises(ValueError, match="3, 3"):
        layout.to_support(np.zeros((2, 2)))


def test_expand_round_trips():
    A = symmetric(5, loops=True)
    layout = DyadLayout.undirected(5, self_loops=True)
    pairs = layout.canonical_pairs()
    values = layout.to_dyads(layout.to_support(A))[pairs]

    np.testing.assert_allclose(layout.expand(values, pairs), A)

    # a leading sample dimension is preserved
    stacked = layout.expand(np.stack([values, 2 * values]), pairs)
    assert stacked.shape == (2, 5, 5)
    np.testing.assert_allclose(stacked[1], 2 * A)
