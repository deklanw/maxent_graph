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


def both_positive_parts(**kwargs):
    return models(**kwargs) + models(positive="ztp", **kwargs)


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
def test_expected_strengths_reproduce_observed(model):
    """
    The default constrains the model's own expected strengths, summed over
    every dyad, like every other model in the library. Before, the positive
    part was fitted on the observed edges only and the model's expected total
    came up well short -- 65% of the observed weight on kato.
    """
    model.fit()
    assert model.strengths == "joint"
    np.testing.assert_allclose(
        model.expected_row_strengths(), model.row_strengths, atol=1e-5
    )
    np.testing.assert_allclose(
        model.expected_col_strengths(), model.col_strengths, atol=1e-5
    )
    assert model.mean().sum() == pytest.approx(model.total_weight, rel=1e-7)
    assert model.fit_info["strength_error"] == model.joint_strength_error()


@pytest.mark.parametrize("positive", ["shifted", "ztp"])
def test_joint_strengths_on_a_sparse_network(positive):
    model = BIHPCM(kato(), positive=positive).fit()
    relative = np.abs(model.expected_row_strengths() - model.row_strengths) / (
        model.row_strengths
    )
    assert relative.max() < 1e-8
    assert model.mean().sum() == pytest.approx(model.total_weight, rel=1e-10)
    # the joint constraint is spread over every dyad, so no boundary to creep
    # towards and no need for the constraint-residual stopping rule
    assert model.fit_info["iterations"] < 500


def test_joint_fit_gives_absent_dyads_a_rate():
    """
    Fitted on the observed edges only, every absent dyad had rate zero, so a
    weight of two or more there had probability zero. The joint fit gives
    every dyad whose endpoints both carry weight beyond one a positive rate.
    """
    B = random_bipartite()
    joint = BIHPCM(B).fit()
    conditional = BIHPCM(B, strengths="conditional").fit()

    absent = joint.layout.support & ~joint.adjacency
    excess_rows = joint.row_strengths > joint.row_degrees
    excess_cols = joint.col_strengths > joint.col_degrees
    reachable = absent & excess_rows[:, None] & excess_cols[None, :]
    assert reachable.any()

    assert np.all(joint.rate[reachable] > 0)
    assert np.all(joint.sf(2)[reachable] > 0)
    assert np.all(conditional.rate[absent] == 0)
    assert np.all(conditional.sf(2)[absent] == 0)


def test_strengths_validation():
    with pytest.raises(ValueError, match="strengths"):
        BIHPCM(random_bipartite(), strengths="something")
    with pytest.raises(ValueError, match="EM"):
        BIHPCM(random_bipartite(), kind="zip", strengths="joint")
    assert BIHPCM(random_bipartite(), kind="zip").strengths is None


@pytest.mark.parametrize("model", both_positive_parts(strengths="conditional"))
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
    assert model.fit_info["rate_zero_dyads"] >= model.adjacency.sum()


@pytest.mark.parametrize("strengths", ["joint", "conditional"])
@pytest.mark.parametrize("positive", ["shifted", "ztp"])
def test_unit_weight_nodes_get_rate_exactly_zero(positive, strengths):
    """
    Both positive parts are solved for the excess of the weight over one, so
    a node carrying only unit weights has excess zero and rate exactly zero
    after one update, with nothing to peel and nothing to creep towards.
    """
    B = random_bipartite()
    B[0] = (B[0] > 0).astype(float)  # row 0 carries only unit weights
    model = BIHPCM(B, positive=positive, strengths=strengths).fit()

    assert np.all(model.rate[0] == 0)
    assert model.fit_info["iterations"] < 100
    # a unit-weight node's expected strength is its expected degree, so it is
    # matched exactly as well as the presence fit matched the degree
    assert model.strength_error() <= model.degree_error() + 1e-9


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
def test_shifted_joint_fit_is_a_weighted_poisson_fit(model):
    """
    The conditional mean is linear in the rate, so the joint constraint
    reduces to a Poisson configuration model on ``w - 1`` weighted by the
    presence probabilities, with targets ``s_i - sum_j p_ij``.
    """
    model.fit()
    weighted = model.presence * model.rate

    np.testing.assert_allclose(
        model.layout.row_totals(weighted),
        model.row_strengths - model.expected_row_degrees(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        model.layout.col_totals(weighted),
        model.col_strengths - model.expected_col_degrees(),
        atol=1e-5,
    )


@pytest.mark.parametrize("model", models(strengths="conditional"))
def test_shifted_part_is_a_poisson_fit_on_the_weight_minus_one(model):
    """
    Over the observed edges only, the same linearity makes the constraint a
    Poisson configuration model on ``w - 1`` with targets ``s_i - k_i``.
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
    assert shifted.strength_error() <= shifted.degree_error() + 1e-9

    truncated = BIHPCM(B, positive="ztp").fit()
    assert np.all(truncated.rate[0] == 0)
    # and the two are genuinely different fits elsewhere
    assert not np.allclose(shifted.rate, truncated.rate)


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


@pytest.mark.parametrize("positive", ["shifted", "ztp"])
def test_loglik_survives_pmf_underflow(positive):
    B = random_bipartite()
    # a unit weight where a heavy row meets a heavy column: its rate runs to
    # thousands, and the probability of weight one underflows
    B[0, 0] = 1.0
    B[0, 1] = B[1, 0] = 5000.0
    model = BIHPCM(B, positive=positive).fit()

    rows, cols = model.layout.canonical_pairs()
    assert np.any(model.pmf(model.weights[rows, cols], (rows, cols)) == 0)
    assert np.isfinite(model.loglik())
    assert model.loglik() == pytest.approx(
        model.logpmf(model.weights[rows, cols], (rows, cols)).sum()
    )


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
    assert model.strength_error() < 1e-6 * model.total_weight


@pytest.mark.parametrize(
    "model",
    [
        BIHPCM(kato(), strengths="conditional"),
        UHPCM(kangaroo(), strengths="conditional"),
        DHPCM(residence_hall(), strengths="conditional"),
    ],
)
def test_real_networks_fit_conditionally(model):
    model.fit()
    assert model.fit_info["strength_error"] == model.positive_strength_error()
    # the tolerance the fit settles for near a boundary on sparse supports
    assert model.strength_error() < 1e-5 * model.total_weight


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
