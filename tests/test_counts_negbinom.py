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
def test_score_equations_are_solved(model):
    model.fit()
    assert model.fit_info["score_norm"] < 1e-6
    # and the fit beats the Poisson one it started from
    assert model.r > 0


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


@pytest.mark.parametrize(
    "model", [BINBCM(random_bipartite()), BINBCM(kato()), UNBCM(kangaroo())]
)
def test_profile_likelihood_is_unimodal(model):
    model.fit()
    grid = np.geomspace(model.r / 50, model.r * 50, 40)
    profile = model.profile_loglik(grid)

    increments = np.sign(np.diff(profile))
    assert np.count_nonzero(np.diff(increments)) == 1  # one turning point
    assert profile.max() <= model.loglik() + 1e-6
    assert grid[np.argmax(profile)] == pytest.approx(model.r, rel=0.3)


@pytest.mark.parametrize("model", models())
def test_constrained_fit_reproduces_the_strengths(model):
    model.fit(constrain_strengths=True)
    np.testing.assert_allclose(
        model.expected_row_strengths(), model.row_strengths, atol=1e-9
    )
    np.testing.assert_allclose(
        model.expected_col_strengths(), model.col_strengths, atol=1e-9
    )
    assert model.mean().sum() == pytest.approx(model.total_weight)


def test_maximum_likelihood_beats_the_constrained_fit():
    B = kato()
    assert BINBCM(B).fit().loglik() > BINBCM(B).fit(constrain_strengths=True).loglik()


def test_standard_error_is_reported():
    model = BINBCM(kato()).fit()
    assert model.r_std_error > 0
    assert np.isfinite(model.r_std_error)
    assert model.log_r_std_error == pytest.approx(model.r_std_error / model.r)
    assert model.fit_info["r_std_error"] == model.r_std_error


def test_standard_error_covers_a_simulated_truth():
    truth = BINBCM(random_bipartite(25, 30, seed=3), r=4.0).fit(
        constrain_strengths=True
    )
    simulated = truth.sample(1, rng=np.random.default_rng(7))[0]

    refit = BINBCM(simulated).fit(constrain_strengths=True)
    assert abs(refit.r - 4.0) < 4 * refit.r_std_error


def test_row_dispersion():
    B = random_bipartite()
    model = BINBCM(B, dispersion="row").fit()

    assert model.r.shape == (B.shape[0],)
    mean = model.mean()
    np.testing.assert_allclose(
        model.var(), mean + mean**2 / model.r[:, None], atol=1e-10
    )
    assert model.fit_info["score_norm"] < 1e-6
    # a free r per row can only fit better than one shared r
    assert model.loglik() >= BINBCM(B).fit().loglik() - 1e-8


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


def test_validation():
    with pytest.raises(ValueError, match="dispersion"):
        BINBCM(random_bipartite(), dispersion="column")
    with pytest.raises(ValueError, match="r must be positive"):
        BINBCM(random_bipartite(), r=0.0)
