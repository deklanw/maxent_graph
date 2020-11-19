import numpy as np
import scipy.optimize
import pandas as pd
import math
import itertools
import numba
import time

from tqdm import tqdm

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit
from . import poibin


class BICM(MaxentGraph):
    def __init__(self, B):
        self.B = B
        num_rows, num_cols = B.shape

        self.num_edges = B.count_nonzero()

        # since B is a (sparse) matrix, the sums will be matrices
        # the sums above will give ints, so we need to convert to floats
        row_sums = np.asarray(np.sum(B, axis=1).astype(np.float64)).flatten()
        col_sums = np.asarray(np.sum(B, axis=0).astype(np.float64)).flatten()

        # assert not np.any(np.where(row_sums == 0))
        # assert not np.any(np.where(row_sums == len(col_sums)))
        # assert not np.any(np.where(col_sums == 0))
        # assert not np.any(np.where(col_sums == len(row_sums)))

        assert len(row_sums) == num_rows
        assert len(col_sums) == num_cols
        # since in empirical networks there will be many nodes with the same degree
        # we can count them and use that information to speed up solving the equations.
        # the bicm doesn't distinguish between nodes with the same degree.
        # we also want to keep track of which nodes have which degree (for later). for that we just use pd's groupby
        row_degrees, row_inverse, row_multiplicity = np.unique(
            row_sums, return_index=False, return_inverse=True, return_counts=True
        )
        row_df = pd.DataFrame(row_sums)
        self.row_groups = row_df.groupby(by=0).groups
        self.row_degrees = row_degrees
        self.row_inverse = row_inverse
        self.row_multiplicity = row_multiplicity

        col_degrees, col_inverse, col_multiplicity = np.unique(
            col_sums, return_index=False, return_inverse=True, return_counts=True
        )
        col_df = pd.DataFrame(col_sums)
        self.col_groups = col_df.groupby(by=0).groups
        self.col_degrees = col_degrees
        self.col_inverse = col_inverse
        self.col_multiplicity = col_multiplicity

        self.n_row_degrees = len(self.row_degrees)
        self.n_col_degrees = len(self.col_degrees)
        self.total_unique = self.n_row_degrees + self.n_col_degrees

    def bounds(self):
        lower_bounds = np.array([EPS] * self.total_unique)
        upper_bounds = np.array([np.inf] * self.total_unique)

        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.row_degrees, self.col_degrees])

    def get_initial_guess(self, option=1):
        if option == 1:
            x0_rows = self.row_degrees / np.max(self.row_degrees)
            x0_cols = self.col_degrees / np.max(self.col_degrees)
        elif option == 2:
            x0_rows = self.row_degrees / np.sqrt(np.sum(self.row_degrees) + 1)
            x0_cols = self.col_degrees / np.sqrt(np.sum(self.col_degrees) + 1)
        elif option == 3:
            denom = np.sqrt(np.sum(self.row_degrees) * np.sum(self.col_degrees))

            x0_rows = self.row_degrees / denom
            x0_cols = self.col_degrees / denom
        else:
            raise ValueError("Invalid option value. Choose from 1-3.")

        x0 = np.concatenate([x0_rows, x0_cols])

        return x0

    @jax_class_jit
    def expected_node_sequence(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]

        xy = jnp.outer(x, y)
        p = xy / (1 + xy)

        # row expected
        # multiply every row by col_multiplicity then sum across columns
        row_expected = (p * self.col_multiplicity).sum(axis=1)

        # multiply every column by row_multiplicity then sum across rows
        col_expected = (p.T * self.row_multiplicity).sum(axis=1)

        return jnp.concatenate((row_expected, col_expected))

    # painstakingly by hand
    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]

        xy = jnp.outer(x, y)
        t = jnp.power(1 / (1 + xy), 2)

        d_x = (t * y * self.col_multiplicity).sum(axis=1)
        d_x = jnp.diag(d_x)

        d_y = jnp.outer(self.col_multiplicity, x).T * t
        u_x = (jnp.outer(self.row_multiplicity, y) * t).T

        u_y = (t.T * x * self.row_multiplicity).sum(axis=1)
        u_y = jnp.diag(u_y)

        d = jnp.concatenate([d_x, d_y], axis=1)
        u = jnp.concatenate([u_x, u_y], axis=1)
        j = jnp.concatenate([d, u], axis=0)

        return j

    def expected_node_sequence_loops(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]

        row_expected = np.zeros(self.n_row_degrees)
        col_expected = np.zeros(self.n_col_degrees)

        for i in range(self.n_row_degrees):
            for j in range(self.n_col_degrees):
                x_ij = x[i] * y[j]
                v = x_ij / (1.0 + x_ij)
                row_expected[i] += self.col_multiplicity[j] * v
                col_expected[j] += self.row_multiplicity[i] * v

        return np.concatenate((row_expected, col_expected))

    def neg_log_likelihood_loops(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]
        llhood = 0

        for i in range(self.n_row_degrees):
            llhood += self.row_degrees[i] * self.row_multiplicity[i] * np.log(x[i])

        for i in range(self.n_col_degrees):
            llhood += self.col_degrees[i] * self.col_multiplicity[i] * np.log(y[i])

        for i in range(self.n_row_degrees):
            for j in range(self.n_col_degrees):
                llhood -= (
                    self.row_multiplicity[i]
                    * self.col_multiplicity[j]
                    * np.log(1 + x[i] * y[j])
                )

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]

        llhood = jnp.sum(self.row_degrees * self.row_multiplicity * jnp.log(x))
        llhood += jnp.sum(self.col_degrees * self.col_multiplicity * jnp.log(y))

        Q = jnp.log(1 + jnp.outer(x, y))
        Q = Q * self.col_multiplicity
        Q = Q.T * self.row_multiplicity
        # don't need to transpose back because we're summing anyways
        llhood -= jnp.sum(Q)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        x = v[: self.n_row_degrees]
        y = v[self.n_row_degrees :]

        xy = jnp.outer(x, y)

        denom = jnp.outer(self.row_multiplicity, self.col_multiplicity) / (1 + xy)

        x_1 = self.row_degrees * self.row_multiplicity / x
        # sum along columns
        x_2 = (denom * y).sum(axis=1)

        y_1 = self.col_degrees * self.col_multiplicity / y
        # sum along rows
        y_2 = (denom.T * x).sum(axis=1)

        return -jnp.concatenate((x_1 - x_2, y_1 - y_2))

    def get_fitness_model_solution(self):
        start = time.time()
        solution = scipy.optimize.root_scalar(
            self.fitness_zero,
            args=(
                self.num_edges,
                self.row_degrees,
                self.row_multiplicity,
                self.col_degrees,
                self.col_multiplicity,
            ),
            method=None,
            x0=1e-10,
            x1=0.01,
            fprime=self.fitness_zero_prime,
            fprime2=self.fitness_zero_prime_prime,
        )
        print(f"Fitness model solution took {time.time() - start}")

        z = solution.root

        return np.sqrt(z) * self.order_node_sequence()

    @staticmethod
    @numba.jit(nopython=True)
    def fitness_zero(x, E, row_degrees, row_mult, col_degrees, col_mult):
        s = -E
        for i, _ in enumerate(row_degrees):
            for j, _ in enumerate(col_degrees):
                kk = row_degrees[i] * col_degrees[j]
                s += row_mult[i] * col_mult[j] * x * kk / (1 + x * kk)
        return s

    @staticmethod
    @numba.jit(nopython=True)
    def fitness_zero_prime(x, E, row_degrees, row_mult, col_degrees, col_mult):
        s = 0
        for i, _ in enumerate(row_degrees):
            for j, _ in enumerate(col_degrees):
                kk = row_degrees[i] * col_degrees[j]
                s += row_mult[i] * col_mult[j] * kk / (1 + x * kk) ** 2
        return s

    @staticmethod
    @numba.jit(nopython=True)
    def fitness_zero_prime_prime(x, E, row_degrees, row_mult, col_degrees, col_mult):
        s = 0
        for i, _ in enumerate(row_degrees):
            for j, _ in enumerate(col_degrees):
                kk = row_degrees[i] * col_degrees[j]
                s += row_mult[i] * col_mult[j] * -2 * kk ** 2 / (1 + x * kk) ** 3
        return s

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def get_probs_with_multiplicity(i, j, x, y):
        v_i = x[i] * y
        v_j = x[j] * y

        ps_i = v_i / (1 + v_i)
        ps_j = v_j / (1 + v_j)

        expected_lambda_motif_probs = ps_i * ps_j

        return expected_lambda_motif_probs

    def get_projection(self, solution, p_val=0.05):
        B = self.B
        # faster indexing when dense. but, more memory. in most cases it won't be sparse so this is fine.
        # this will be symmetric
        observed_lambda_motif_counts = (B @ B.T).todense()
        nonzero_set = set(zip(*observed_lambda_motif_counts.nonzero()))

        print(f"Nonzero lambda-motif counts to check pval of {len(nonzero_set)}")
        print(f"Total possible pairs: {math.comb(B.shape[0], 2)}")

        x = solution[: self.n_row_degrees]
        y = solution[self.n_row_degrees :]

        print(f"Unique degrees {(self.n_row_degrees, self.n_col_degrees)}")

        edgelist = []
        print(
            f"Total unique row degree pairs to check {math.comb(self.n_row_degrees, 2)}"
        )
        for (i, j) in tqdm(itertools.combinations(range(self.n_row_degrees), 2)):
            degree_i = self.row_degrees[i]
            degree_j = self.row_degrees[j]

            expected_lambda_motif_probs_with_mult = self.get_probs_with_multiplicity(
                i, j, x, y
            )

            mean = poibin.mean_with_multiplicity(
                expected_lambda_motif_probs_with_mult, self.col_multiplicity
            )

            # filter zeros to speed up loop
            # this may slow it down if the number of nonzeros is very dense, but only slightly
            # if it's sparse then this can substantially speed up
            to_check = nonzero_set & set(
                itertools.product(self.row_groups[degree_i], self.row_groups[degree_j])
            )

            for orig_i, orig_j in to_check:
                observed = observed_lambda_motif_counts[orig_i, orig_j]
                assert observed != 0
                q = poibin.poisson_wh_cdf(observed, mean)
                p = 1 - q
                if p < p_val:
                    # the WH approx to the poisson isn't numerically stable this low, nor is the dc_fft
                    # breaks down in the 1-e14/1e-15 range
                    if p < 1e-13:
                        # technically an upper bound on poisson approx, but it keeps close enough for these very small values to get order of magnitude
                        # and is numerically stable/fast. seems to be stable at least until 1e-300
                        # certainly NOT low relative error this small, just order of magnitude is right
                        p = poibin.poisson_upper(observed, mean)
                    edgelist.append((orig_i, orig_j, -np.log(p)))

        return edgelist
