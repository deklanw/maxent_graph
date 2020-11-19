import pytest
import numpy as np
import scipy.stats

from maxent_graph.poibin import dc_pb1, dc_pb2, dc_fft_pb


def test_dc_pb():
    # todo: handle empty list case
    for n in [10, 100, 1000, 10_000]:
        ps = np.random.rand(n)

        # both DC methods should give same results
        r1 = dc_pb1(ps)
        r2 = dc_pb2(ps)

        assert np.allclose(r1, r2)

        # sum of pmf values should be 1
        assert np.sum(r1) == pytest.approx(1)

        # fft method should be close to DC
        r3 = dc_fft_pb(ps)

        assert np.allclose(r1, r3)


def test_dc_fft():
    for n in [10, 100, 1000, 10_000, 50_000]:
        ps = np.random.rand(n)

        r = dc_fft_pb(ps)
        assert len(r) == len(ps) + 1

        # test against binomial distribution, within a given error
        ps = np.repeat(0.5, n)
        r = dc_fft_pb(ps)

        # less values to test, and smaller probs will have larger relative error
        larger_probs = np.argwhere(r > 1e-5)
        correct_probs = np.array(
            [scipy.stats.binom.pmf(i, n, 0.5).item() for i in larger_probs]
        )

        assert np.allclose(r[larger_probs].ravel(), correct_probs)

