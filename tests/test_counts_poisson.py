import numpy as np
import pytest
import scipy.stats

from maxent_graph.counts import BIPCM, DPCM, UPCM, DyadLayout, PoissonCM
from tests.counts_fixtures import (
    kangaroo,
    kato,
    random_bipartite,
    random_directed,
    random_undirected,
    residence_hall,
)

TOL = 1e-10


def bipartite_models():
    return [BIPCM(random_bipartite()), BIPCM(kato())]


def all_models():
    return [
        BIPCM(random_bipartite()),
        BIPCM(kato()),
        UPCM(random_undirected()),
        UPCM(random_undirected(self_loops=True), self_loops=True),
        UPCM(kangaroo()),
        DPCM(random_directed()),
        DPCM(random_directed(self_loops=True), self_loops=True),
        DPCM(residence_hall()),
    ]


@pytest.mark.parametrize("model", all_models())
def test_expected_strengths_reproduce_observed(model):
    model.fit()
    np.testing.assert_allclose(
        model.expected_row_strengths(), model.row_strengths, atol=TOL
    )
    np.testing.assert_allclose(
        model.expected_col_strengths(), model.col_strengths, atol=TOL
    )
    assert model.constraint_error() < TOL


@pytest.mark.parametrize("model", all_models())
def test_mean_sums_to_total_weight_and_var_equals_mean(model):
    model.fit()
    assert model.mean().sum() == pytest.approx(model.total_weight)
    np.testing.assert_allclose(model.mean(), model.var())


@pytest.mark.parametrize(
    "model",
    [
        UPCM(random_undirected(self_loops=True), self_loops=True),
        UPCM(kangaroo(), self_loops=True),
        DPCM(random_directed(self_loops=True), self_loops=True),
        DPCM(residence_hall(), self_loops=True),
        BIPCM(random_bipartite()),
    ],
)
def test_closed_form_matches_numerical_solver(model):
    closed = model.fit(method="closed").M.copy()
    numerical = model.fit(method="numerical").M
    np.testing.assert_allclose(closed, numerical, rtol=1e-9, atol=1e-12)


def test_bipartite_mean_is_the_strength_product():
    B = random_bipartite()
    model = BIPCM(B).fit()
    expected = np.outer(B.sum(axis=1), B.sum(axis=0)) / B.sum()
    np.testing.assert_allclose(model.mean(), expected)
    assert model.fit_info["method"] == "closed form"


def test_undirected_mean_with_loops_is_halved():
    A = random_undirected(self_loops=True)
    model = UPCM(A, self_loops=True).fit()
    s = model.row_strengths
    W = model.total_weight
    mean = model.mean((np.array([0, 1]), np.array([1, 1])))
    # off-diagonal s_i s_j / 2W, diagonal half that
    assert mean[0] == pytest.approx(s[0] * s[1] / (2 * W))
    assert mean[1] == pytest.approx(s[1] * s[1] / (4 * W))


def test_no_interior_maximum_is_reported():
    star = np.zeros((5, 5))
    star[0, 1:] = 1
    star[1:, 0] = 1
    with pytest.raises(RuntimeError, match="no interior maximum"):
        UPCM(star).fit()
    # allowing self-loops makes it well posed again
    UPCM(star, self_loops=True).fit()


@pytest.mark.parametrize("model", all_models())
def test_tail_conventions(model):
    model.fit()
    dyads = (np.array([0, 1, 0]), np.array([2, 2, 1]))
    mean = model.mean(dyads)
    for w in (1, 3, 7):
        np.testing.assert_allclose(
            model.sf(w, dyads), 1 - model.cdf(w - 1, dyads), atol=1e-12
        )
        np.testing.assert_allclose(
            model.pmf(w, dyads), scipy.stats.poisson.pmf(w, mean)
        )
    np.testing.assert_allclose(model.sf(0, dyads), 1.0)


@pytest.mark.parametrize("model", all_models())
def test_selectors_agree(model):
    model.fit()
    full = model.mean()
    rows, cols = model.layout.canonical_pairs()
    np.testing.assert_allclose(model.mean((rows, cols)), full[rows, cols])
    np.testing.assert_allclose(
        model.mean(np.stack([rows, cols], axis=1)), full[rows, cols]
    )
    np.testing.assert_allclose(model.mean(model.layout.canonical), full[rows, cols])


@pytest.mark.parametrize("model", all_models())
def test_loglik_matches_a_direct_sum(model):
    model.fit()
    rows, cols = model.layout.canonical_pairs()
    direct = scipy.stats.poisson.logpmf(
        model.weights[rows, cols], model.mean((rows, cols))
    ).sum()
    assert model.loglik() == pytest.approx(direct)


@pytest.mark.parametrize("model", all_models())
def test_samples_respect_the_layout(model):
    model.fit()
    draws = model.sample(3, rng=np.random.default_rng(0))
    assert draws.shape == (3,) + model.layout.shape
    if model.layout.tied:
        np.testing.assert_allclose(draws[0], draws[0].T)
    if not model.layout.self_loops and model.layout.kind != "bipartite":
        assert np.all(np.diagonal(draws, axis1=1, axis2=2) == 0)


@pytest.mark.parametrize(
    "model",
    [
        BIPCM(random_bipartite()),
        UPCM(random_undirected()),
        UPCM(random_undirected(self_loops=True), self_loops=True),
        DPCM(random_directed()),
    ],
)
def test_samples_have_the_right_mean(model):
    model.fit()
    n = 4000
    draws = model.sample(n, rng=np.random.default_rng(0))

    rows, cols = model.layout.canonical_pairs()
    empirical = draws[:, rows, cols].mean(axis=0)
    expected = model.mean((rows, cols))
    se = np.sqrt(expected / n)
    assert np.all(np.abs(empirical - expected) < 5 * se + 1e-9)


@pytest.mark.parametrize("model", all_models())
def test_pvalues_evaluate_sf_at_the_observed_weights(model):
    model.fit()
    dyads = np.nonzero(model.weights)
    np.testing.assert_allclose(
        model.pvalues(dyads), model.sf(model.weights[dyads], dyads)
    )

    full = model.pvalues()
    assert full.shape == model.layout.shape
    np.testing.assert_allclose(full[dyads], model.pvalues(dyads))
    assert np.all((full[model.layout.support] >= 0) & (full[model.layout.support] <= 1))


def test_unfitted_model_refuses_to_answer():
    model = BIPCM(random_bipartite())
    with pytest.raises(RuntimeError, match="not fitted"):
        model.mean()


# --------------------------------------------------------------------------
# exact stub matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("B", [random_bipartite(), kato()])
def test_exact_matches_the_hypergeometric_moments(B):
    canonical = BIPCM(B).fit()
    exact = BIPCM(B, exact=True).fit()

    np.testing.assert_allclose(exact.mean(), canonical.mean())

    W = exact.total_weight
    s_row = exact.row_strengths[:, None]
    s_col = exact.col_strengths[None, :]
    expected_var = canonical.mean() * (1 - s_col / W) * (W - s_row) / (W - 1)
    np.testing.assert_allclose(exact.var(), expected_var)

    # stub matching is the deflated version of the Poisson model
    assert np.all(exact.var() <= canonical.var() + 1e-12)


def test_exact_samples_match_the_margins_exactly():
    B = random_bipartite()
    model = BIPCM(B, exact=True).fit()
    draws = model.sample(5, rng=np.random.default_rng(1))
    assert np.all(draws.sum(axis=2) == model.row_strengths)
    assert np.all(draws.sum(axis=1) == model.col_strengths)


def test_exact_loglik_is_the_stub_matching_probability():
    B = np.array([[1.0, 2.0], [3.0, 0.0]])
    model = BIPCM(B, exact=True).fit()
    # enumerate every pairing of the 6 row stubs with the 6 column stubs
    from itertools import permutations

    rows = np.repeat([0, 1], [3, 3])
    cols = np.repeat([0, 1], [4, 2])
    counts = {}
    for perm in permutations(range(6)):
        table = np.zeros((2, 2))
        for stub, slot in enumerate(perm):
            table[rows[stub], cols[slot]] += 1
        counts[tuple(table.ravel())] = counts.get(tuple(table.ravel()), 0) + 1
    total = sum(counts.values())
    assert model.loglik() == pytest.approx(np.log(counts[(1, 2, 3, 0)] / total))


def test_exact_rejects_what_it_cannot_do():
    with pytest.raises(ValueError, match="integer weights"):
        BIPCM(np.array([[0.5, 1.0], [1.0, 2.0]]), exact=True)
    with pytest.raises(NotImplementedError):
        PoissonCM(random_undirected(), DyadLayout.undirected(8), exact=True)
