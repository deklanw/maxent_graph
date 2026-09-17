import numpy as np
import pytest
import scipy.stats

from maxent_graph.counts.dists import (
    Hurdle,
    ShiftedGeometric,
    ShiftedPoisson,
    ZeroTruncatedPoisson,
)

GRID = np.arange(0, 600)


def moments(dist):
    pmf = dist.pmf(GRID)
    mean = np.sum(GRID * pmf)
    return pmf.sum(), mean, np.sum((GRID - mean) ** 2 * pmf)


@pytest.mark.parametrize("lam", [1e-14, 1e-6, 0.5, 3.0, 20.0])
def test_zero_truncated_poisson_moments(lam):
    dist = ZeroTruncatedPoisson(lam)
    total, mean, var = moments(dist)
    assert total == pytest.approx(1.0)
    assert mean == pytest.approx(dist.mean())
    assert var == pytest.approx(dist.var(), abs=1e-12)
    assert dist.pmf(0) == 0
    assert dist.sf(0) == 1.0


def test_zero_truncated_poisson_collapses_onto_one():
    dist = ZeroTruncatedPoisson(1e-14)
    assert dist.mean() == pytest.approx(1.0)
    assert dist.pmf(1) == pytest.approx(1.0)
    assert dist.var() == pytest.approx(0.0, abs=1e-12)


def test_zero_truncated_poisson_is_poisson_conditioned_on_being_positive():
    lam = 2.5
    dist = ZeroTruncatedPoisson(lam)
    expected = scipy.stats.poisson.pmf(GRID, lam) / (1 - np.exp(-lam))
    np.testing.assert_allclose(dist.pmf(GRID[1:]), expected[1:])
    np.testing.assert_allclose(dist.cdf(GRID) + dist.sf(GRID), 1.0)


@pytest.mark.parametrize("y", [0.05, 0.4, 0.9])
def test_shifted_geometric_moments(y):
    dist = ShiftedGeometric(y)
    total, mean, var = moments(dist)
    assert total == pytest.approx(1.0)
    assert mean == pytest.approx(dist.mean())
    assert var == pytest.approx(dist.var())
    # the inclusive upper tail is the BiECM's edge p-value
    for w in (1, 2, 5):
        assert dist.sf(w - 1) == pytest.approx(y ** (w - 1))


@pytest.mark.parametrize("positive", [ZeroTruncatedPoisson(2.0), ShiftedGeometric(0.4)])
@pytest.mark.parametrize("p", [0.0, 0.25, 1.0])
def test_hurdle_moments(p, positive):
    dist = Hurdle(p, positive)
    total, mean, var = moments(dist)
    assert total == pytest.approx(1.0)
    assert mean == pytest.approx(dist.mean())
    assert var == pytest.approx(dist.var())
    assert dist.pmf(0) == pytest.approx(1 - p)
    np.testing.assert_allclose(dist.cdf(GRID) + dist.sf(GRID), 1.0)


def test_hurdle_is_the_zip_reparameterised():
    # a ZIP(pi, lam) is a hurdle with p = pi * (1 - exp(-lam)) and a ZTP body
    pi, lam = 0.6, 1.7
    dist = Hurdle(pi * (1 - np.exp(-lam)), ZeroTruncatedPoisson(lam))
    zip_pmf = np.where(
        GRID == 0,
        1 - pi + pi * np.exp(-lam),
        pi * scipy.stats.poisson.pmf(GRID, lam),
    )
    np.testing.assert_allclose(dist.pmf(GRID), zip_pmf, atol=1e-300)


def test_parameters_broadcast_and_rvs_follows_them():
    dist = Hurdle(np.array([0.0, 1.0]), ZeroTruncatedPoisson(np.array([1e-14, 1e-14])))
    draws = dist.rvs(size=(5000, 2), random_state=0)
    assert np.all(draws[:, 0] == 0)
    assert np.all(draws[:, 1] == 1)


@pytest.mark.parametrize("lam", [1e-14, 0.3, 4.0, 30.0])
def test_zero_truncated_poisson_rvs_stays_in_support(lam):
    draws = ZeroTruncatedPoisson(lam).rvs(size=20000, random_state=1)
    assert np.all(np.isfinite(draws))
    assert draws.min() >= 1
    dist = ZeroTruncatedPoisson(lam)
    assert draws.mean() == pytest.approx(
        dist.mean(), abs=5 * np.sqrt(dist.var() / 20000) + 1e-9
    )


@pytest.mark.parametrize(
    "dist",
    [
        ZeroTruncatedPoisson(np.array([1e-14, 0.5, 40.0])),
        ShiftedPoisson(np.array([0.0, 0.5, 40.0])),
        ShiftedGeometric(np.array([0.0, 0.3, 0.99])),
        Hurdle(np.array([0.0, 0.4, 1.0]), ShiftedPoisson(np.array([2.0, 0.0, 40.0]))),
        Hurdle(
            np.array([0.2, 0.4, 0.9]), ZeroTruncatedPoisson(np.array([2.0, 1e-9, 40.0]))
        ),
    ],
)
def test_logpmf_matches_log_pmf_and_survives_underflow(dist):
    grid = np.arange(0, 3000)[:, None]
    with np.errstate(divide="ignore"):
        from_pmf = np.log(dist.pmf(grid))
    logpmf = dist.logpmf(grid)

    # comparing only where the pmf is comfortably representable: near
    # underflow the pmf is subnormal and its log is the inaccurate one
    representable = dist.pmf(grid) > 1e-280
    np.testing.assert_allclose(
        logpmf[representable], from_pmf[representable], rtol=1e-9, atol=1e-12
    )
    representable = np.isfinite(from_pmf)
    # where the pmf underflows but the outcome is possible, the log stays finite
    assert not np.any(np.isnan(logpmf))
    assert np.isfinite(logpmf[~representable & (logpmf > -np.inf)]).all()


def test_logpmf_is_finite_far_in_a_poisson_tail():
    dist = ShiftedPoisson(3.0)
    assert dist.pmf(2000) == 0.0
    assert np.isfinite(dist.logpmf(2000))
    assert dist.logpmf(2000) == pytest.approx(scipy.stats.poisson.logpmf(1999, 3.0))


def test_hurdle_lower_tail_has_no_cancellation():
    """
    A hurdle cdf computed as 1 - sf rounds a lower tail near 1e-16 to the
    nearest ulp of one, which here was 2.2e-16 against a true 1.7e-16.
    """
    dist = Hurdle(1.0, ShiftedPoisson(40.0))
    tiny = dist.cdf(2)
    reference = scipy.stats.poisson.cdf(1, 40.0)
    assert 0 < tiny < 1e-15
    np.testing.assert_allclose(tiny, reference, rtol=1e-9, atol=0)
