import numpy as np
import pytest
import scipy.stats

from maxent_graph import BWCM
from maxent_graph.counts import BINBCM, BIPCM, DNBCM, DPCM, UNBCM, UPCM
from maxent_graph.util import nx_get_B
from tests.counts_fixtures import (
    kangaroo,
    kato,
    random_bipartite,
    random_directed,
    random_undirected,
)

POISSON_LIMIT = 1e8


def models(**kwargs):
    return [
        BINBCM(random_bipartite(), **kwargs),
        UNBCM(random_undirected(), **kwargs),
        UNBCM(random_undirected(self_loops=True), self_loops=True, **kwargs),
        DNBCM(random_directed(), **kwargs),
        DNBCM(random_directed(self_loops=True), self_loops=True, **kwargs),
    ]


@pytest.mark.parametrize("model", models())
def test_variance_follows_the_dispersion(model):
    model.fit()
    mean = model.mean()
    np.testing.assert_allclose(model.var(), mean + mean**2 / model.r, atol=1e-12)
    assert np.all(model.var() >= mean - 1e-12)


@pytest.mark.parametrize("model", models())
def test_default_fit_reproduces_the_strengths(model):
    model.fit()
    assert model.fit_info["constrain_strengths"] is True
    np.testing.assert_allclose(
        model.expected_row_strengths(), model.row_strengths, atol=1e-9
    )
    np.testing.assert_allclose(
        model.expected_col_strengths(), model.col_strengths, atol=1e-9
    )


@pytest.mark.parametrize("model", models())
def test_unconstrained_fit_solves_the_score_equations(model):
    model.fit(constrain_strengths=False)
    assert model.fit_info["score_norm"] < 1e-6
    assert model.r > 0
    # the weighted score equations are not the strength constraints
    assert model.constraint_error() > 1e-9


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize(
    "W, pair, kwargs",
    [
        (random_bipartite(), (BIPCM, BINBCM), {}),
        (random_undirected(), (UPCM, UNBCM), {}),
        (
            random_directed(self_loops=True),
            (DPCM, DNBCM),
            {"self_loops": True},
        ),
        (kangaroo(), (UPCM, UNBCM), {}),
    ],
)
def test_large_r_reproduces_the_poisson_model(W, pair, kwargs, constrained):
    poisson_class, negbinom_class = pair
    poisson = poisson_class(W, **kwargs).fit()
    negbinom = negbinom_class(W, r=POISSON_LIMIT, **kwargs)
    negbinom.fit(constrain_strengths=constrained)

    np.testing.assert_allclose(negbinom.mean(), poisson.mean(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(negbinom.var(), poisson.var(), rtol=1e-6, atol=1e-6)
    assert negbinom.loglik() == pytest.approx(poisson.loglik(), abs=1e-4)


def test_r_of_one_is_geometric():
    B = random_bipartite()
    model = BINBCM(B, r=1.0).fit()
    mean = model.mean()

    np.testing.assert_allclose(model.var(), mean * (1 + mean))

    # the dyad pmf is the geometric one with success probability mu / (1 + mu)
    dyads = (np.array([0, 1, 2]), np.array([3, 1, 0]))
    q = model.mean(dyads) / (1 + model.mean(dyads))
    for w in (0, 1, 5):
        np.testing.assert_allclose(model.pmf(w, dyads), (1 - q) * q**w)


def test_r_of_one_against_the_bwcm():
    """
    The BWCM is the other geometric model on the same constraints: it makes
    the geometric *ratio* a product rather than the mean, which is a different
    link and so a different fit. What they share is the variance function and,
    with the means pinned to the strengths, the strengths themselves.
    """
    W = nx_get_B(
        "data/plant_pol_vazquez_All_sites_pooled.graphml",
        weight_key="count",
        bipartite_key="pollinator",
    )
    dense_W = np.asarray(W.todense(), dtype=float)

    bwcm = BWCM(W)
    solution = bwcm.solve(bwcm.get_initial_guess())
    z = np.asarray(bwcm.transform_parameters(solution.x))
    x = z[: bwcm.n_row_strengths][bwcm.row_inverse]
    y = z[bwcm.n_row_strengths :][bwcm.col_inverse]
    xy = np.outer(x, y)
    bwcm_mean = xy / (1 - xy)
    bwcm_var = bwcm_mean * (1 + bwcm_mean)

    model = BINBCM(dense_W, r=1.0).fit(constrain_strengths=True)

    # same variance function
    np.testing.assert_allclose(model.var(), model.mean() * (1 + model.mean()))
    # same strengths, to the accuracy the BWCM was solved to
    np.testing.assert_allclose(bwcm_mean.sum(axis=1), dense_W.sum(axis=1), rtol=1e-4)
    np.testing.assert_allclose(
        model.expected_row_strengths(), dense_W.sum(axis=1), atol=1e-9
    )
    # but different means, because the link differs
    assert not np.allclose(bwcm_mean, model.mean(), rtol=1e-3)
    assert bwcm_var.sum() > 0


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize(
    "W, cls", [(random_bipartite(), BINBCM), (kato(), BINBCM), (kangaroo(), UNBCM)]
)
def test_profile_likelihood_is_unimodal(W, cls, constrained):
    model = cls(W).fit(constrain_strengths=constrained)
    grid = np.geomspace(model.r / 50, model.r * 50, 40)
    profile = model.profile_loglik(grid)

    increments = np.sign(np.diff(profile))
    assert np.count_nonzero(np.diff(increments)) == 1  # one turning point
    assert profile.max() <= model.loglik() + 1e-6
    assert grid[np.argmax(profile)] == pytest.approx(model.r, rel=0.3)


@pytest.mark.parametrize("model", models())
def test_constrained_fit_reproduces_the_strengths(model):
    model.fit()
    np.testing.assert_allclose(
        model.expected_row_strengths(), model.row_strengths, atol=1e-9
    )
    np.testing.assert_allclose(
        model.expected_col_strengths(), model.col_strengths, atol=1e-9
    )
    assert model.mean().sum() == pytest.approx(model.total_weight)


def test_maximum_likelihood_beats_the_constrained_fit():
    B = kato()
    free = BINBCM(B).fit(constrain_strengths=False).loglik()
    assert free > BINBCM(B).fit().loglik()


def test_standard_error_is_reported():
    model = BINBCM(kato()).fit()
    assert model.r_std_error > 0
    assert np.isfinite(model.r_std_error)
    assert model.log_r_std_error == pytest.approx(model.r_std_error / model.r)
    assert model.fit_info["r_std_error"] == model.r_std_error


def test_standard_error_covers_a_simulated_truth():
    truth = BINBCM(random_bipartite(25, 30, seed=3), r=4.0).fit()
    simulated = truth.sample(1, rng=np.random.default_rng(7))[0]

    refit = BINBCM(simulated).fit()
    assert abs(refit.r - 4.0) < 4 * refit.r_std_error


@pytest.mark.parametrize("constrained", [True, False])
def test_overdispersion_evidence_is_reported(constrained):
    B = kato()
    model = BINBCM(B).fit(constrain_strengths=constrained)
    info = model.fit_info

    assert info["loglik"] == pytest.approx(model.loglik())
    # the r -> inf limit is the Poisson configuration model on the same data
    assert info["loglik_poisson"] == pytest.approx(BIPCM(B).fit().loglik())
    assert info["overdispersion_lr"] == pytest.approx(
        2 * (info["loglik"] - info["loglik_poisson"])
    )
    assert info["overdispersion_lr"] > 0
    # the boundary correction halves the chi-square tail
    assert info["overdispersion_p"] == pytest.approx(
        0.5 * scipy.stats.chi2.sf(info["overdispersion_lr"], 1)
    )


@pytest.mark.parametrize("constrained", [True, False])
def test_row_dispersion_has_no_single_parameter_p_value(constrained):
    """
    The halved chi-square on one degree of freedom is the reference for one
    dispersion; with one per row it is miscalibrated, so it is not reported.
    """
    B = kato()
    model = BINBCM(B, dispersion="row").fit(constrain_strengths=constrained)
    info = model.fit_info

    assert np.isnan(info["overdispersion_p"])
    assert info["overdispersion_lr"] == pytest.approx(
        2 * (info["loglik"] - info["loglik_poisson"])
    )
    assert info["overdispersion_lr"] > 0
    # the global fit keeps its p-value
    assert np.isfinite(
        BINBCM(B).fit(constrain_strengths=constrained).fit_info["overdispersion_p"]
    )


def test_a_poisson_network_shows_no_overdispersion():
    simulated = (
        BIPCM(random_bipartite(25, 30, seed=11))
        .fit()
        .sample(1, rng=np.random.default_rng(3))[0]
    )
    info = BINBCM(simulated).fit().fit_info
    assert info["overdispersion_p"] > 0.05


@pytest.mark.parametrize("constrained", [True, False])
def test_row_dispersion(constrained):
    B = random_bipartite()
    model = BINBCM(B, dispersion="row").fit(constrain_strengths=constrained)

    assert model.r.shape == (B.shape[0],)
    mean = model.mean()
    np.testing.assert_allclose(
        model.var(), mean + mean**2 / model.r[:, None], atol=1e-10
    )
    if not constrained:
        assert model.fit_info["score_norm"] < 1e-6
    # a free r per row can only fit better than one shared r
    shared = BINBCM(B).fit(constrain_strengths=constrained).loglik()
    assert model.loglik() >= shared - 1e-8


def test_row_dispersion_is_refused_when_the_sides_are_tied():
    with pytest.raises(ValueError, match="undirected"):
        UNBCM(random_undirected(), dispersion="row")


@pytest.mark.parametrize("model", models())
def test_distribution_is_the_negative_binomial(model):
    model.fit()
    dyads = (np.array([0, 1, 2]), np.array([3, 2, 1]))
    mean = model.mean(dyads)
    r = model.r_matrix[dyads]
    reference = scipy.stats.nbinom(n=r, p=r / (r + mean))

    for w in (0, 1, 4):
        np.testing.assert_allclose(model.pmf(w, dyads), reference.pmf(w))
        np.testing.assert_allclose(model.cdf(w, dyads), reference.cdf(w))
        np.testing.assert_allclose(
            model.sf(w, dyads), 1 - model.cdf(w - 1, dyads), atol=1e-12
        )


@pytest.mark.parametrize("model", models())
def test_samples_are_overdispersed_and_respect_the_layout(model):
    model.fit()
    draws = model.sample(3000, rng=np.random.default_rng(0))
    assert draws.shape == (3000,) + model.layout.shape
    if model.layout.tied:
        np.testing.assert_allclose(draws[0], draws[0].T)

    rows, cols = model.layout.canonical_pairs()
    expected = model.mean((rows, cols))
    variance = model.var((rows, cols))
    empirical = draws[:, rows, cols].mean(axis=0)
    assert np.all(np.abs(empirical - expected) < 5 * np.sqrt(variance / 3000) + 1e-9)


def test_loglik_survives_pmf_underflow():
    B = random_bipartite()
    # a zero where a heavy row meets a heavy column: a huge mean, and with a
    # large r the pmf of zero underflows
    B[0, 0] = 0.0
    B[0, 1] = B[1, 0] = 1e5
    model = BINBCM(B, r=1000.0).fit()
    rows, cols = model.layout.canonical_pairs()
    assert np.any(model.pmf(model.weights[rows, cols], (rows, cols)) == 0)
    assert np.isfinite(model.loglik())


def test_validation():
    with pytest.raises(ValueError, match="dispersion"):
        BINBCM(random_bipartite(), dispersion="column")
    with pytest.raises(ValueError, match="r must be positive"):
        BINBCM(random_bipartite(), r=0.0)
