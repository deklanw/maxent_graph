import math
import numpy as np
import scipy.optimize
import scipy.signal

from numba import jit

EPS = np.finfo(float).eps


# faster without numba.. not sure why
# keeping this here although it's slower than dc_pb2 for the relevant sizes
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


def dc_fft_pb(ps, split_amount=200):
    """
    Basic intuition: we split up the probabilities into chunks and use direct convolution on the small chunks, and then combine them in pairs with FFT repeatedly.
    DC is faster for smaller sizes.

    Paper claimed that 750 was a good number for split_amount. I found 175-300 to be better.
    """
    N = len(ps)

    assert N > 1

    # need k to be positive, so N has to be at least 750 for the log to be >= 1
    # < 1000 M=2 is best, according to paper
    # also, in the [1000, 1500] range M=2 is better than M=1
    if N <= 2 * split_amount:
        M = 2
    else:
        # k chosen such that the splits are of about size split_amount
        k = math.floor(np.log2(N / split_amount))
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

    # need probabilities so take real part and clip the error
    return np.clip(np.real(convolved_subsets[0]), 0, 1)


def fft_pb(ps):
    """
    Computes the pmf of a Poisson Binomial distribution using only FFT. Slower than DC-FFT. Just keeping this here for comparison.
    """
    result = np.array([1 - ps[0], ps[0]])

    for i in range(1, len(ps)):
        p_i = ps[i]
        signal = np.array([1 - p_i, p_i])
        result = scipy.signal.convolve(result, signal, method="fft")

    return result


@jit(nopython=True)
def gaussian_pdf(x):
    """
    Computes standard Gaussian pdf.

    about 3000x faster than scipy.stats.norm(0, 1).pdf(x)
    around 6x faster with jit than without
    """
    # ** is slightly faster than np.power for some reason
    g = (2 * math.pi) ** -0.5 * np.exp(-0.5 * x ** 2)
    return g


SQRT2 = math.sqrt(2.0)


@jit(nopython=True)
def gaussian_cdf(x):
    """
    Computes standard Gaussian cdf.

    about 3000x faster than scipy.stats.norm(0, 1).cdf(x)
    about 5x faster than scipy.special.ndtr(x)
    """
    return math.erfc(-x / SQRT2) / 2.0


"""
comparison  of rna_cdf against dc_fft_cdf (dc_fft pmf then summing)

n
100       190x
1000      130x
10_000    127x
100_000   200x 
"""


@jit(nopython=True, parallel=True)
def rna_mean_std_gamma(ps):
    mean = np.sum(ps)
    std = np.sqrt(np.sum(ps * (1 - ps)))
    gamma = std ** -3 * np.sum(ps * (1 - ps) * (1 - 2 * ps))

    return mean, std, gamma


@jit(nopython=True, parallel=True)
def rna_mean_std_gamma_with_multiplicity(ps, mult):
    mean = np.sum(mult * ps)
    std = np.sqrt(np.sum(mult * (ps * (1 - ps))))
    gamma = std ** -3 * np.sum(mult * (ps * (1 - ps) * (1 - 2 * ps)))

    return mean, std, gamma


@jit(nopython=True)
def rna_cdf(k, mean, std, gamma):
    """
    Refined normal approximation.
    """
    x = (k + 0.5 - mean) / std
    phi_x = gaussian_pdf(x)
    big_phi_x = gaussian_cdf(x)

    # can return invalid probabilities. just clip.
    p = big_phi_x + (gamma * (1 - x ** 2) * phi_x) / 6

    # can't use clip in numba
    if p <= 0:
        return EPS
    elif p >= 1:
        return 1 - EPS
    else:
        return p


# https://stats.stackexchange.com/questions/35658/simple-approximation-of-poisson-cumulative-distribution-in-long-tail
@jit(nopython=True)
def poisson_upper(k, mu):
    return math.exp(
        k
        - mu
        + k * (math.log(mu) - math.log(k))
        + math.log(k + 1)
        - math.log(math.sqrt(2 * math.pi * k))
        - math.log(k + 1 - mu)
    )


@jit(nopython=True)
def binomial_cdf(x, n, p):
    """
    Copied from  https://stackoverflow.com/a/45869209/4749956
    Faster than scipy (the next fastest I could find). I suppose writing it up in C might be even faster.
    
    Falls apart for values close to 1.
    """
    cdf = 0
    b = 0
    for k in range(x + 1):
        if k > 0:
            b += math.log(n - k + 1) - math.log(k)
        log_pmf_k = b + k * math.log(p) + (n - k) * math.log(1 - p)
        cdf += math.exp(log_pmf_k)
    return cdf


# https://www.johndcook.com/blog/wilson_hilferty/
@jit(nopython=True)
def poisson_wh_cdf(k, lam):
    c = math.pow(lam / (1 + k), 1 / 3)
    mu = 1 - 1 / (9 * k + 9)
    sigma = 1 / (3 * math.sqrt(1 + k))

    x = (c - mu) / sigma

    return 1 - gaussian_cdf(x)


LOG_PI = math.log(math.pi)

# refinement of stirling approximation by ramanujan
@jit(nopython=True)
def log_fact_raman(n):
    return (
        n * math.log(n) - n + math.log(n * (1 + 4 * n * (1 + 2 * n))) / 6 + LOG_PI / 2
    )


@jit(nopython=True)
def poisson_pmf_approx(k, mean):
    return math.exp(k * math.log(mean) - mean - log_fact_raman(k))


@jit(nopython=True)
def make_cdf(ps):
    # slightly faster if we mutate instead
    cdf = ps.copy()
    for i in range(len(ps)):
        if i == 0:
            continue
        cdf[i] += cdf[i - 1]
    return cdf


@jit(nopython=True, parallel=True)
def mean_with_multiplicity(ps, mult):
    return np.sum(mult * ps)
