import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats

from maxent_graph import BICM, BIECM
from maxent_graph.counts import (
    BIPCM,
    DPCM,
    UPCM,
    aggregate_blocks,
    from_bicm,
    from_biecm,
)
from maxent_graph.util import nx_get_B
from tests.counts_fixtures import random_bipartite, random_directed, random_undirected

ROW_BLOCKS = np.array([0, 0, 0, 1, 1, 2, 2])
COL_BLOCKS = np.array(list("aabbbcccc"))


@pytest.fixture(scope="module")
def bipartite_model():
    return BIPCM(random_bipartite()).fit()


def test_cells_partition_the_dyads(bipartite_model):
    model = bipartite_model
    table = aggregate_blocks(model, ROW_BLOCKS, COL_BLOCKS)

    assert len(table) == 3 * 3
    assert table.n_dyads.sum() == model.layout.n_dyads
    assert table.observed.sum() == pytest.approx(model.total_weight)
    assert table.expected.sum() == pytest.approx(model.total_weight)


def test_cell_moments_are_sums_over_the_cell(bipartite_model):
    model = bipartite_model
    table = aggregate_blocks(model, ROW_BLOCKS, COL_BLOCKS).set_index(
        ["row_block", "col_block"]
    )

    rows = np.flatnonzero(ROW_BLOCKS == 1)
    cols = np.flatnonzero(COL_BLOCKS == "c")
    cell = np.ix_(rows, cols)

    row = table.loc[(1, "c")]
    assert row.observed == pytest.approx(model.W[cell].sum())
    assert row.expected == pytest.approx(model.mean()[cell].sum())
    assert row.variance == pytest.approx(model.var()[cell].sum())
    assert row.enrichment == pytest.approx(row.observed / row.expected)
    assert row.z == pytest.approx((row.observed - row.expected) / np.sqrt(row.variance))


@pytest.mark.parametrize(
    "model",
    [
        BIPCM(random_bipartite()),
        UPCM(random_undirected()),
        DPCM(random_directed()),
    ],
)
def test_fft_agrees_with_the_exact_poisson_cell(model):
    model.fit()
    blocks = np.array([0, 0, 0, 1, 1, 2, 2, 2])[: model.layout.n_row]
    kwargs = (
        {"col_blocks": COL_BLOCKS}
        if model.layout.kind == "bipartite"
        else {"col_blocks": None}
    )
    blocks = ROW_BLOCKS if model.layout.kind == "bipartite" else blocks

    exact = aggregate_blocks(model, blocks, method="exact", **kwargs)
    fft = aggregate_blocks(model, blocks, method="fft", **kwargs)

    assert set(exact.method) == {"exact"}
    # a one-dyad cell short-circuits to the dyad's own tail under either name
    assert set(fft.method) <= {"fft", "exact"}
    assert (fft.method == "exact").sum() == (fft.n_dyads == 1).sum()
    np.testing.assert_allclose(exact.p_upper, fft.p_upper, atol=1e-10)
    np.testing.assert_allclose(exact.p_lower, fft.p_lower, atol=1e-10)


def test_normal_approximation_is_close_to_exact(bipartite_model):
    exact = aggregate_blocks(bipartite_model, ROW_BLOCKS, COL_BLOCKS, method="exact")
    normal = aggregate_blocks(bipartite_model, ROW_BLOCKS, COL_BLOCKS, method="normal")
    assert set(normal.method) == {"normal"}
    np.testing.assert_allclose(normal.p_upper, exact.p_upper, atol=0.03)


def test_auto_falls_back_to_normal_for_big_cells():
    model = BIPCM(random_bipartite(20, 20)).fit()
    blocks = np.zeros(20, dtype=int)
    table = aggregate_blocks(model, blocks, blocks, method="auto", max_fft_dyads=4)
    # family A always has an exact cell distribution, so size is irrelevant
    assert set(table.method) == {"exact"}


def test_singleton_partition_reproduces_the_per_dyad_tail():
    B = random_bipartite(4, 5)
    model = BIPCM(B).fit()
    table = aggregate_blocks(model, np.arange(4), np.arange(5))

    rows = table.row_block.to_numpy()
    cols = table.col_block.to_numpy()
    np.testing.assert_allclose(
        table.p_upper, model.sf(B[rows, cols], (rows, cols)), atol=1e-12
    )


def test_exact_stub_matching_cells_are_hypergeometric():
    B = random_bipartite()
    model = BIPCM(B, exact=True).fit()
    table = aggregate_blocks(model, ROW_BLOCKS, COL_BLOCKS).set_index(
        ["row_block", "col_block"]
    )
    assert set(table.method) == {"exact"}

    rows = np.flatnonzero(ROW_BLOCKS == 0)
    cols = np.flatnonzero(COL_BLOCKS == "b")
    kappa = model.row_strengths[rows].sum()
    lam = model.col_strengths[cols].sum()
    observed = B[np.ix_(rows, cols)].sum()

    cell = table.loc[(0, "b")]
    expected = scipy.stats.hypergeom(
        M=round(model.total_weight), n=round(lam), N=round(kappa)
    )
    assert cell.p_upper == pytest.approx(expected.sf(observed - 1))
    assert cell.p_lower == pytest.approx(expected.cdf(observed))
    assert (
        cell.variance
        < aggregate_blocks(BIPCM(B).fit(), ROW_BLOCKS, COL_BLOCKS)
        .set_index(["row_block", "col_block"])
        .loc[(0, "b")]
        .variance
    )


def test_coverage_counts_nodes_with_an_edge_into_the_block():
    B = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    B[3, 1] = 1.0  # keep every node connected
    model = BIPCM(B).fit()
    table = aggregate_blocks(
        model, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    ).set_index(["row_block", "col_block"])

    # rows 0 and 1 both reach column block 0, but only column 0 is reached
    assert table.loc[(0, 0)].coverage_row == pytest.approx(1.0)
    assert table.loc[(0, 0)].coverage_col == pytest.approx(0.5)
    # neither row of block 0 reaches column block 1
    assert table.loc[(0, 1)].coverage_row == pytest.approx(0.0)
    assert table.loc[(0, 1)].coverage_col == pytest.approx(0.0)
    # of rows 2 and 3 only row 2 reaches column block 1, and it reaches both
    assert table.loc[(1, 1)].coverage_row == pytest.approx(0.5)
    assert table.loc[(1, 1)].coverage_col == pytest.approx(1.0)


def test_undirected_cells_are_unordered_pairs():
    model = UPCM(random_undirected()).fit()
    blocks = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    table = aggregate_blocks(model, blocks)

    assert len(table) == 6  # 3 within-block plus 3 between-block cells
    assert table.n_dyads.sum() == model.layout.n_dyads
    assert table.observed.sum() == pytest.approx(model.total_weight)
    assert np.all(table.row_block <= table.col_block)


def test_undirected_coverage_uses_each_endpoint_side():
    A = np.zeros((4, 4))
    A[0, 2] = A[2, 0] = 5.0
    A[0, 1] = A[1, 0] = 1.0
    A[2, 3] = A[3, 2] = 1.0
    model = UPCM(A).fit()
    table = aggregate_blocks(model, np.array([0, 0, 1, 1])).set_index(
        ["row_block", "col_block"]
    )
    # only node 0 of block 0 and only node 2 of block 1 span the two blocks
    assert table.loc[(0, 1)].coverage_row == pytest.approx(0.5)
    assert table.loc[(0, 1)].coverage_col == pytest.approx(0.5)


def test_directed_cells_keep_their_orientation():
    A = random_directed()
    model = DPCM(A).fit()
    blocks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    table = aggregate_blocks(model, blocks).set_index(["row_block", "col_block"])

    assert len(table) == 4
    out_block = np.flatnonzero(blocks == 0)
    in_block = np.flatnonzero(blocks == 1)
    assert table.loc[(0, 1)].observed == pytest.approx(
        A[np.ix_(out_block, in_block)].sum()
    )
    assert table.loc[(1, 0)].observed == pytest.approx(
        A[np.ix_(in_block, out_block)].sum()
    )


def test_input_validation(bipartite_model):
    with pytest.raises(ValueError, match="col_blocks"):
        aggregate_blocks(bipartite_model, ROW_BLOCKS)
    with pytest.raises(ValueError, match="one per node"):
        aggregate_blocks(bipartite_model, ROW_BLOCKS, COL_BLOCKS[:3])
    with pytest.raises(ValueError, match="method"):
        aggregate_blocks(bipartite_model, ROW_BLOCKS, COL_BLOCKS, method="magic")

    model = UPCM(random_undirected()).fit()
    with pytest.raises(ValueError, match="one partition"):
        aggregate_blocks(model, np.zeros(8, int), np.ones(8, int))


# --------------------------------------------------------------------------
# the existing models, through the adapters
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def biecm_fit():
    W = nx_get_B(
        "data/plant_pol_vazquez_All_sites_pooled.graphml",
        weight_key="count",
        bipartite_key="pollinator",
    )
    model = BIECM(W)
    return model, model.solve(model.get_initial_guess()), W


def test_biecm_adapter_reproduces_the_edge_p_values(biecm_fit):
    model, solution, W = biecm_fit
    table = from_biecm(model, solution, W)

    dense_W = np.asarray(W.todense(), dtype=float)
    reference = model.get_pval_matrix(solution.x, sp.csr_matrix(dense_W))
    nonzero = reference.nonzero()

    # within one ulp: the remaining difference is the exponent's dtype, since
    # ShiftedGeometric floors it to a float where get_pval_matrix does not
    np.testing.assert_allclose(
        table.sf(dense_W[nonzero], nonzero),
        np.asarray(reference[nonzero]).ravel(),
        rtol=1e-15,
    )


def test_biecm_singleton_cells_reproduce_the_edge_p_values(biecm_fit):
    model, solution, W = biecm_fit
    dense_W = np.asarray(W.todense(), dtype=float)
    table = from_biecm(model, solution, W)

    n_row, n_col = dense_W.shape
    cells = aggregate_blocks(table, np.arange(n_row), np.arange(n_col), method="fft")
    rows = cells.row_block.to_numpy()
    cols = cells.col_block.to_numpy()

    reference = model.get_pval_matrix(solution.x, sp.csr_matrix(dense_W))
    edges = dense_W[rows, cols] > 0
    np.testing.assert_allclose(
        cells.p_upper.to_numpy()[edges],
        np.asarray(reference[rows[edges], cols[edges]]).ravel(),
        rtol=1e-9,
    )


def test_biecm_adapter_reproduces_the_fitted_constraints(biecm_fit):
    model, solution, W = biecm_fit
    table = from_biecm(model, solution, W)
    np.testing.assert_allclose(
        table.expected_row_strengths(), table.row_strengths, rtol=1e-4
    )
    assert table.mean().sum() == pytest.approx(table.total_weight, rel=1e-4)


def test_biecm_blocks_aggregate():
    W = nx_get_B(
        "data/plant_pol_vazquez_All_sites_pooled.graphml",
        weight_key="count",
        bipartite_key="pollinator",
    )
    model = BIECM(W)
    table = from_biecm(model, model.solve(model.get_initial_guess()), W)

    n_row, n_col = table.layout.shape
    rng = np.random.default_rng(0)
    row_blocks = rng.integers(0, 3, n_row)
    col_blocks = rng.integers(0, 3, n_col)

    # no closed form for a hurdle cell total, so auto convolves the small
    # cells and approximates once they get big
    cells = aggregate_blocks(table, row_blocks, col_blocks)
    assert set(cells.method) == {"fft"}
    assert cells.observed.sum() == pytest.approx(table.total_weight)
    assert cells.expected.sum() == pytest.approx(table.total_weight, rel=1e-3)

    approximate = aggregate_blocks(table, row_blocks, col_blocks, max_fft_dyads=1)
    assert set(approximate.method) == {"normal"}


def test_convolved_cell_tails_match_simulation(biecm_fit):
    """
    The convolution is the reference the normal approximation is judged
    against, so check it against the model's own sampler. These cells are
    badly overdispersed -- variance around fifty times the mean -- which is
    exactly where the normal approximation drifts.
    """
    model, solution, W = biecm_fit
    table = from_biecm(model, solution, W)

    rng = np.random.default_rng(0)
    row_blocks = rng.integers(0, 3, table.layout.n_row)
    col_blocks = rng.integers(0, 3, table.layout.n_col)
    cells = aggregate_blocks(table, row_blocks, col_blocks, method="fft")

    draws = table.sample(20000, rng=1)
    for cell in cells.itertuples():
        rows = np.flatnonzero(row_blocks == cell.row_block)
        cols = np.flatnonzero(col_blocks == cell.col_block)
        totals = draws[:, rows][:, :, cols].sum(axis=(1, 2))
        simulated = (totals >= cell.observed).mean()
        assert cell.p_upper == pytest.approx(simulated, abs=0.02)

    with pytest.raises(ValueError, match="no exact cell distribution"):
        aggregate_blocks(table, row_blocks, col_blocks, method="exact")


def test_bicm_adapter_matches_its_own_fit():
    B = nx_get_B("data/opsahl-southernwomen_bipartite.graphml")
    model = BICM(B)
    table = from_bicm(model, model.solve(model.get_initial_guess()))

    dense_B = (np.asarray(B.todense()) > 0).astype(float)
    np.testing.assert_allclose(
        table.expected_row_strengths(), dense_B.sum(axis=1), atol=1e-6
    )
    np.testing.assert_allclose(table.var(), table.mean() * (1 - table.mean()))

    cells = aggregate_blocks(
        table, np.arange(dense_B.shape[0]) // 6, np.arange(dense_B.shape[1]) // 5
    )
    assert cells.observed.sum() == pytest.approx(dense_B.sum())
    # every cell probability is a sum of Bernoullis, so fft is exact
    assert set(cells.method) == {"fft"}


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------


def test_exact_margins_fixed_grand_total_has_zero_variance():
    """
    Under stub matching the dyads are dependent, so their variances do not
    add. Summing them reported a variance for a grand total that is fixed.
    """
    B = random_bipartite()
    model = BIPCM(B, exact=True).fit()
    cell = aggregate_blocks(
        model, np.zeros(B.shape[0], int), np.zeros(B.shape[1], int)
    ).iloc[0]

    assert cell.variance == 0.0
    assert cell.observed == cell.expected == model.total_weight
    assert cell.p_upper == pytest.approx(1.0)
    assert cell.p_lower == pytest.approx(1.0)


def test_exact_margin_cell_variance_matches_simulation():
    B = random_bipartite()
    model = BIPCM(B, exact=True).fit()
    table = aggregate_blocks(model, ROW_BLOCKS, COL_BLOCKS)

    draws = model.sample(6000, rng=np.random.default_rng(4))
    for cell in table.itertuples():
        rows = np.flatnonzero(ROW_BLOCKS == cell.row_block)
        cols = np.flatnonzero(COL_BLOCKS == cell.col_block)
        totals = draws[:, rows][:, :, cols].sum(axis=(1, 2))
        assert totals.mean() == pytest.approx(cell.expected, rel=0.02)
        assert totals.var() == pytest.approx(cell.variance, rel=0.1)

    # and it is smaller than the independent-dyad sum would claim
    independent = model.var()
    rows = np.flatnonzero(ROW_BLOCKS == 0)
    cols = np.flatnonzero(COL_BLOCKS == "a")
    cell = table.set_index(["row_block", "col_block"]).loc[(0, "a")]
    assert cell.variance < independent[np.ix_(rows, cols)].sum()


def test_normal_approximation_of_a_fixed_total():
    from maxent_graph.counts.aggregate import _cell_tails

    model = BIPCM(random_bipartite()).fit()
    dyads = (np.array([0, 0]), np.array([0, 1]))
    below = _cell_tails(model, dyads, 3, 5.0, 0.0, "normal", 10.0, 512)
    above = _cell_tails(model, dyads, 7, 5.0, 0.0, "normal", 10.0, 512)
    assert below[:2] == (1.0, 0.0)
    assert above[:2] == (0.0, 1.0)
