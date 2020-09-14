import math
import numpy as np
import scipy.optimize
import scipy.signal

from numba import jit


# faster without numba.. not sure why
def dc_pb1(ps):
    """
    Computes the pmf of a Poisson Binomial distribution using the direct convolution method (via numpy's convolve).
    """
    result = np.array([1 - ps[0], ps[0]])

    for i in range(1, len(ps)):
        p_i = ps[i]
        signal = np.array([1 - p_i, p_i])
        result = np.convolve(result, signal)

    return result


# faster with numba by a lot
# Based on the implementation here https://github.com/andrew12678/ShiftConvolveFFTW/blob/master/src/dcpaired.c
# for the small sizes this is used for (< 1000 elements) it's faster than dc_pb1
@jit(nopython=True)
def dc_pb2(ps):
    """
    Computes the pmf of a Poisson Binomial distribution using the direct convolution method.
    """
    # initialize (old kernel)
    result = [1 - ps[0], ps[0]]

    for i in range(1, len(ps)):
        old_len = len(result)

        p_i = ps[i]
        signal = [1 - p_i, p_i]

        # initialize result and calculate the two edge cases
        # i.e., no successes and all success
        result.append(signal[1] * result[old_len - 1])

        last = result[0]
        result[0] = signal[0] * last

        # calculate the interior cases
        # conceptually, we're taking the old pmf and adding one more Bernoulli trial with prob p
        # the new probability of k successes is the old probability of k-1 successes times the new p
        # plus the old probability of k successes times (1-p)
        for j in range(1, old_len):
            # have to save the value before changing it so it can be used for the next iteration
            tmp = result[j]
            result[j] = signal[1] * last + signal[0] * result[j]
            last = tmp

    return result


def dc_fft_pb(ps):
    N = len(ps)

    assert N > 1

    # need k to be positive, so N has to be at least 750 for the log to be >= 1
    # < 1000 M=2 is best, according to paper
    # also, in the [1000, 1500] range M=2 is better than M=1
    if N <= 1500:
        M = 2
        k = 1
    else:
        # k chosen such that the splits are of about size 750
        k = math.floor(np.log2(N / 750))
        M = int(math.pow(2, k))

    assert M <= N

    # array_split ensures the split is as even as possible even if len(ps) doesn't divide M
    subsets = np.array_split(ps, M)

    convolved_subsets = [dc_pb2(s) for s in subsets]

    while len(convolved_subsets) > 1:
        it = iter(convolved_subsets)
        poibin = []
        for xs in it:
            ys = next(it)

            poibin.append(scipy.signal.convolve(xs, ys, method="fft"))

        convolved_subsets = poibin

    # need probabilities so clip the error
    return np.clip(np.real(convolved_subsets[0]), 0, 1)
