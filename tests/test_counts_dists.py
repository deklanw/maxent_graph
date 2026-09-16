import numpy as np
import pytest
import scipy.stats

from maxent_graph.counts.dists import Hurdle, ShiftedGeometric, ZeroTruncatedPoisson

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
