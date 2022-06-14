import warnings
import numpy as np
import scipy.optimize

import jax.numpy as jnp

import pandas as pd

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit, R_to_zero_to_one


class BWCM(MaxentGraph):
    """
    Very similar to the BICM. Just changed some names and flipped from 1+xy to 1-xy in a bunch of places.

    """

    def __init__(self, B, transform=0):
        self.B = B
        num_rows, num_cols = B.shape

        row_sums = B.sum(axis=1).getA1().astype(np.float64)
        col_sums = B.sum(axis=0).getA1().astype(np.float64)

        if np.any(np.where(row_sums == 0)) or np.any(np.where(col_sums == 0)):
            warnings.warn("Some nodes have 0 degree. Check on that")

        if np.any(np.where(row_sums == len(col_sums))) or np.any(
            np.where(col_sums == len(row_sums))
        ):
            warnings.warn(
                "Some nodes are connected to every other node in the opposite part. Check on that."
            )

        assert len(row_sums) == num_rows
        assert len(col_sums) == num_cols
        # since in empirical networks there will be many nodes with the same strength
        # we can count them and use that information to speed up solving the equations.
        # the bicm doesn't distinguish between nodes with the same strength.
        # we also want to keep track of which nodes have which strength (for later). for that we just use pd's groupby
        row_strengths, row_inverse, row_multiplicity = np.unique(
            row_sums, return_index=False, return_inverse=True, return_counts=True
        )
        row_df = pd.DataFrame(row_sums)
        self.row_groups = row_df.groupby(by=0).groups
        self.row_strengths = row_strengths
        self.row_inverse = row_inverse
        self.row_multiplicity = row_multiplicity

        col_strengths, col_inverse, col_multiplicity = np.unique(
            col_sums, return_index=False, return_inverse=True, return_counts=True
        )
        col_df = pd.DataFrame(col_sums)
        self.col_groups = col_df.groupby(by=0).groups
        self.col_strengths = col_strengths
        self.col_inverse = col_inverse
        self.col_multiplicity = col_multiplicity

        self.n_row_strengths = len(self.row_strengths)
        self.n_col_strengths = len(self.col_strengths)
        self.total_unique = self.n_row_strengths + self.n_col_strengths

        self.transform, self.inv_transform = R_to_zero_to_one[transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * self.total_unique)
        upper_bounds = np.array([1 - EPS] * self.total_unique)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.row_strengths, self.col_strengths])

    @jax_class_jit
    def transform_parameters(self, v):
        return self.transform(v)

    @jax_class_jit
    def transform_parameters_inv(self, v):
        return self.inv_transform(v)

    def get_initial_guess(self, option=1):
        if option == 1:
            x0_rows = self.row_strengths / np.max(self.row_strengths)
            x0_cols = self.col_strengths / np.max(self.col_strengths)
        elif option == 2:
            x0_rows = self.row_strengths / np.sqrt(np.sum(self.row_strengths) + 1)
            x0_cols = self.col_strengths / np.sqrt(np.sum(self.col_strengths) + 1)
        elif option == 3:
            denom = np.sqrt(np.sum(self.row_strengths) * np.sum(self.col_strengths))

            x0_rows = self.row_strengths / denom
            x0_cols = self.col_strengths / denom
        else:
            raise ValueError("Invalid option value. Choose from 1-3.")

        x0 = np.concatenate([x0_rows, x0_cols])

        return self.transform_parameters_inv(self.clip(x0))

    @jax_class_jit
    def expected_node_sequence(self, v):
        z = self.transform_parameters(v)
        x = z[: self.n_row_strengths]
        y = z[self.n_row_strengths :]

        xy = jnp.outer(x, y)
        p = xy / (1 - xy)

        # row expected
        # multiply every row by col_multiplicity then sum across columns
        row_expected = (p * self.col_multiplicity).sum(axis=1)

        # multiply every column by row_multiplicity then sum across rows
        col_expected = (p.T * self.row_multiplicity).sum(axis=1)

        return jnp.concatenate((row_expected, col_expected))

    def expected_node_sequence_loops(self, v):
        z = self.transform_parameters(v)
        x = z[: self.n_row_strengths]
        y = z[self.n_row_strengths :]

        row_expected = np.zeros(self.n_row_strengths)
        col_expected = np.zeros(self.n_col_strengths)

        for i in range(self.n_row_strengths):
            for j in range(self.n_col_strengths):
                x_ij = x[i] * y[j]
                v = x_ij / (1.0 - x_ij)
                row_expected[i] += self.col_multiplicity[j] * v
                col_expected[j] += self.row_multiplicity[i] * v

        return np.concatenate((row_expected, col_expected))

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)
        x = z[: self.n_row_strengths]
        y = z[self.n_row_strengths :]

        llhood = jnp.sum(self.row_strengths * self.row_multiplicity * jnp.log(x))
        llhood += jnp.sum(self.col_strengths * self.col_multiplicity * jnp.log(y))

        Q = jnp.log(1 - jnp.outer(x, y))
        Q = Q * self.col_multiplicity
        Q = Q.T * self.row_multiplicity
        # don't need to transpose back because we're summing anyways
        llhood += jnp.sum(Q)

        return -llhood

    def neg_log_likelihood_loops(self, v):
        z = self.transform_parameters(v)
        x = z[: self.n_row_strengths]
        y = z[self.n_row_strengths :]

        llhood = 0

        for i in range(self.n_row_strengths):
            llhood += self.row_strengths[i] * self.row_multiplicity[i] * np.log(x[i])

        for i in range(self.n_col_strengths):
            llhood += self.col_strengths[i] * self.col_multiplicity[i] * np.log(y[i])

        for i in range(self.n_row_strengths):
            for j in range(self.n_col_strengths):
                llhood += (
                    self.row_multiplicity[i]
                    * self.col_multiplicity[j]
                    * np.log(1 - x[i] * y[j])
                )

        return -llhood
