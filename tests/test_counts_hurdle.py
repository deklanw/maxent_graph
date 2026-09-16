from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse
import scipy.stats

from maxent_graph import BICM
from maxent_graph.counts import BIHPCM, DHPCM, UHPCM, DyadLayout, aggregate_blocks
from maxent_graph.counts.hurdle import HurdlePoissonCM
from tests.counts_fixtures import (
    kangaroo,
    kato,
    random_bipartite,
    random_directed,
    random_undirected,
    residence_hall,
)


def models(**kwargs):
    return [
        BIHPCM(random_bipartite(), **kwargs),
        UHPCM(random_undirected(), **kwargs),
        DHPCM(random_directed(), **kwargs),
    ]


def both_positive_parts():
    return models() + models(positive="ztp")


@pytest.fixture(scope="module")
def bipartite_model():
    return BIHPCM(random_bipartite()).fit()


def test_presence_part_is_the_bicm():
    """
    The presence half of the hurdle model is exactly the BiCM, so it reuses
    that fit rather than re-deriving it -- and the result is identical, not
    merely close.
    """
    B = random_bipartite()
    model = BIHPCM(B).fit()

    A = scipy.sparse.csr_matrix((B > 0).astype(np.float64))
    reference = BICM(A)
    solution = reference.solve(reference.get_initial_guess())
    z = np.asarray(reference.transform_parameters(solution.x))
    x = z[: reference.n_row_degrees][reference.row_inverse]
    y = z[reference.n_row_degrees :][reference.col_inverse]
    xy = np.outer(x, y)

    np.testing.assert_array_equal(model.presence, xy / (1 + xy))


@pytest.mark.parametrize("model", both_positive_parts())
def test_expected_strengths_over_positive_dyads(model):
    model.fit()
    np.testing.assert_allclose(
        model.expected_positive_row_strengths(), model.row_strengths, atol=1e-8
    )
    np.testing.assert_allclose(
        model.expected_positive_col_strengths(), model.col_strengths, atol=1e-8
    )


@pytest.mark.parametrize("model", both_positive_parts())
def test_expected_degrees_reproduce_observed(model):
    model.fit()
    np.testing.assert_allclose(
        model.expected_row_degrees(), model.row_degrees, atol=1e-4
    )
    np.testing.assert_allclose(
        model.expected_col_degrees(), model.col_degrees, atol=1e-4
    )


@pytest.mark.parametrize("positive", ["shifted", "ztp"])
def test_unit_weights_drive_the_rate_to_zero(positive):
    B = (random_bipartite() > 0).astype(float)
    model = BIHPCM(B, positive=positive).fit()

    assert np.all(model.rate == 0)
    # a zero rate is a point mass at one, so the model is the BiCM again
    np.testing.assert_allclose(model.mean(), model.presence)
    np.testing.assert_allclose(model.pmf(1), model.presence)
    assert model.fit_info["rate_zero_dyads"] == model.adjacency.sum()


def test_partially_unit_nodes_are_peeled_rather_than_chased():
    B = random_bipartite()
    B[0] = (B[0] > 0).astype(float)  # row 0 carries only unit weights
    model = BIHPCM(B, positive="ztp").fit()

    assert np.all(model.rate[0] == 0)
    assert model.fit_info["rate_zero_dyads"] >= (B[0] > 0).sum()
    assert model.positive_strength_error() < 1e-6


def positive_dyads(model, count=3):
    """
    Dyads with a strictly positive rate: the closed forms below are stated for
    those, and the rate-zero dyads have their own test.
    """
    chosen = np.argwhere(model.rate > 0)[:count]
    return (chosen[:, 0], chosen[:, 1])


@pytest.mark.parametrize("model", models())
def test_shifted_quantities_follow_the_stated_formulae(model):
    model.fit()
    assert model.positive == "shifted"

    dyads = positive_dyads(model)
    p = model.presence[dyads]
    lam = model.rate[dyads]

    # w - 1 is Poisson, so the conditional mean is 1 + lam and its variance lam
    np.testing.assert_allclose(model.mean(dyads), p * (1 + lam))
    np.testing.assert_allclose(model.var(dyads), p * lam + p * (1 - p) * (1 + lam) ** 2)

    np.testing.assert_allclose(model.pmf(0, dyads), 1 - p)
    for w in (1, 2, 5):
        np.testing.assert_allclose(
            model.pmf(w, dyads), p * scipy.stats.poisson.pmf(w - 1, lam)
        )
        # P(w >= m) = p * P_Pois(X >= m - 1)
        np.testing.assert_allclose(
            model.sf(w, dyads), p * scipy.stats.poisson.sf(w - 2, lam)
        )
    np.testing.assert_allclose(model.sf(0, dyads), 1.0)


@pytest.mark.parametrize("model", models(positive="ztp"))
def test_truncated_quantities_follow_the_stated_formulae(model):
    model.fit()
    dyads = positive_dyads(model)
    p = model.presence[dyads]
    lam = model.rate[dyads]
    conditional = lam / -np.expm1(-lam)

    np.testing.assert_allclose(model.mean(dyads), p * conditional)

    # law of total variance
    conditional_var = conditional * (lam + 1 - conditional)
    np.testing.assert_allclose(
        model.var(dyads), p * conditional_var + p * (1 - p) * conditional**2
    )

    np.testing.assert_allclose(model.pmf(0, dyads), 1 - p)
    for w in (1, 2, 5):
        expected = p * scipy.stats.poisson.sf(w - 1, lam) / -np.expm1(-lam)
        np.testing.assert_allclose(model.sf(w, dyads), expected)
    np.testing.assert_allclose(model.sf(0, dyads), 1.0)


@pytest.mark.parametrize("model", models())
def test_shifted_part_is_a_poisson_fit_on_the_weight_minus_one(model):
    """
    The whole point of shifting rather than truncating: the conditional mean
    is linear in the rate, so the strength constraint reduces to a Poisson
    configuration model on ``w - 1`` with targets ``s_i - k_i``.
    """
    model.fit()
    mask = model.adjacency
    rates = np.where(mask, model.rate, 0.0)

    np.testing.assert_allclose(
        model.layout.row_totals(rates),
        model.row_strengths - model.row_degrees,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        model.layout.col_totals(rates),
        model.col_strengths - model.col_degrees,
        atol=1e-9,
    )


def test_shifted_part_reaches_the_boundary_in_one_step():
    """
    A node carrying only unit weights has target zero and so rate exactly
    zero, with no peeling and no creeping -- unlike the truncated part, which
    needs both.
    """
    B = random_bipartite()
    B[0] = (B[0] > 0).astype(float)

    shifted = BIHPCM(B).fit()
    assert np.all(shifted.rate[0] == 0)
    assert shifted.positive_strength_error() < 1e-9

    truncated = BIHPCM(B, positive="ztp").fit()
    assert np.all(truncated.rate[0] == 0)
    # and the two are genuinely different fits elsewhere
    assert not np.allclose(shifted.rate, truncated.rate)


def test_shifted_part_converges_at_least_as_easily():
    for W, cls in [(kato(), BIHPCM), (kangaroo(), UHPCM)]:
        shifted = cls(W).fit()
        truncated = cls(W, positive="ztp").fit()
        assert shifted.fit_info["iterations"] <= truncated.fit_info["iterations"]


@pytest.mark.parametrize("model", both_positive_parts())
def test_moments_match_a_brute_force_sum(model):
    model.fit()
    dyads = (np.array([0, 1]), np.array([2, 3]))
    grid = np.arange(0, 400)[:, None]
    pmf = model.pmf(grid, dyads)

    np.testing.assert_allclose(pmf.sum(axis=0), 1.0)
    mean = (grid * pmf).sum(axis=0)
    np.testing.assert_allclose(mean, model.mean(dyads))
    np.testing.assert_allclose(((grid - mean) ** 2 * pmf).sum(axis=0), model.var(dyads))


@pytest.mark.parametrize("model", both_positive_parts())
def test_samples_respect_the_layout_and_the_mean(model):
    model.fit()
    draws = model.sample(4000, rng=np.random.default_rng(0))
    assert draws.shape == (4000,) + model.layout.shape
    if model.layout.tied:
        np.testing.assert_allclose(draws[0], draws[0].T)
        assert np.all(np.diagonal(draws, axis1=1, axis2=2) == 0)

    rows, cols = model.layout.canonical_pairs()
    expected = model.mean((rows, cols))
    variance = model.var((rows, cols))
    empirical = draws[:, rows, cols].mean(axis=0)
    assert np.all(np.abs(empirical - expected) < 5 * np.sqrt(variance / 4000) + 1e-9)


def test_presence_can_be_supplied():
    B = random_bipartite()
    fitted = BIHPCM(B).fit()
    reused = BIHPCM(B).fit(presence=fitted.presence)

    np.testing.assert_array_equal(reused.presence, fitted.presence)
    np.testing.assert_allclose(reused.rate, fitted.rate)
    assert reused.presence_model is None


def test_self_loops_are_refused():
    with pytest.raises(NotImplementedError, match="self-loops"):
        HurdlePoissonCM(
            random_undirected(self_loops=True), DyadLayout.undirected(8, True)
        )


def test_kind_validation():
    with pytest.raises(ValueError, match="kind"):
        BIHPCM(random_bipartite(), kind="something")
    with pytest.raises(ValueError, match="positive"):
        BIHPCM(random_bipartite(), positive="something")


def test_zero_inflation_implies_a_truncated_positive_part():
    assert BIHPCM(random_bipartite()).positive == "shifted"
    assert BIHPCM(random_bipartite(), kind="zip").positive == "ztp"
    with pytest.raises(ValueError, match="zero-truncated"):
        BIHPCM(random_bipartite(), kind="zip", positive="shifted")


@pytest.mark.parametrize(
    "model",
    [
        BIHPCM(kato()),
        UHPCM(kangaroo()),
        DHPCM(residence_hall()),
        BIHPCM(kato(), positive="ztp"),
        UHPCM(kangaroo(), positive="ztp"),
        DHPCM(residence_hall(), positive="ztp"),
    ],
)
def test_real_networks_fit(model):
    model.fit()
    assert model.degree_error() < 1e-3
    assert np.isfinite(model.loglik())
    # the strength constraint is met to within the tolerance the fit settles for
    assert model.positive_strength_error() < 1e-5 * model.total_weight


def test_aggregates_into_blocks(bipartite_model):
    model = bipartite_model
    table = aggregate_blocks(
        model,
        np.array([0, 0, 0, 1, 1, 2, 2]),
        np.array(list("aabbbcccc")),
        method="fft",
    )
    assert table.observed.sum() == pytest.approx(model.total_weight)
    assert table.expected.sum() == pytest.approx(model.mean().sum())


# --------------------------------------------------------------------------
# zero inflation
# --------------------------------------------------------------------------


def test_zip_is_the_same_distribution_family_reparameterised():
    B = random_bipartite()
    model = BIHPCM(B, kind="zip").fit()

    support = model.layout.support
    np.testing.assert_allclose(
        model.presence[support], (model.pi * -np.expm1(-model.rate))[support]
    )

    dyads = (np.array([0, 1]), np.array([2, 3]))
    pi = model.pi[dyads]
    lam = model.rate[dyads]
    np.testing.assert_allclose(
        model.pmf(0, dyads), 1 - pi + pi * np.exp(-lam), atol=1e-12
    )
    for w in (1, 3):
        np.testing.assert_allclose(
            model.pmf(w, dyads), pi * scipy.stats.poisson.pmf(w, lam)
        )


@pytest.mark.parametrize(
    "model",
    [
        BIHPCM(random_bipartite(), kind="zip"),
        UHPCM(random_undirected(), kind="zip"),
        DHPCM(random_directed(), kind="zip"),
    ],
)
def test_em_climbs(model):
    values = []
    original = model._zip_loglik

    def spy(pi, rate):
        values.append(original(pi, rate))
        return values[-1]

    model._zip_loglik = spy
    model.fit()

    assert len(values) > 1
    assert all(b >= a - 1e-8 * abs(a) for a, b in pairwise(values))
    assert model.loglik() == pytest.approx(values[-1])
    assert model.fit_info["zip_loglik"] == pytest.approx(values[-1])


def test_zip_and_hurdle_are_different_fits():
    B = random_bipartite()
    hurdle = BIHPCM(B).fit()
    zip_model = BIHPCM(B, kind="zip").fit()

    # neither parameterisation nests the other, so neither is guaranteed to win
    assert not np.allclose(hurdle.presence, zip_model.presence)
    assert np.isfinite(zip_model.loglik())
    assert zip_model.mean().sum() == pytest.approx(B.sum(), rel=0.05)
