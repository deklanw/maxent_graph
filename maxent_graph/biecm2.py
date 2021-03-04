import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse as sp
import math
import numba

import jax.numpy as jnp
import jax
from jax import jit

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit


### R <=> (0, inf) homeomorphisms
@jit
def softplus_inv(x):
    return jnp.log(jnp.exp(x) - 1)


R_to_zero_to_inf = [(jit(jnp.exp), jit(jnp.log)), (jit(jax.nn.softplus), softplus_inv)]

### R <=> (0,1) homeomorphisms
@jit
def shift_scale_arctan(x):
    # scaled, shifted arctan
    return (1 / jnp.pi) * jnp.arctan(x) + 1 / 2


@jit
def shift_scale_arctan_inv(x):
    return jnp.tan(jnp.pi * (x - 1 / 2))


@jit
def sigmoid_inv(x):
    return -jnp.log(1 / x - 1)


R_to_zero_to_one = [
    (jit(jax.nn.sigmoid), sigmoid_inv),
    (shift_scale_arctan, shift_scale_arctan_inv),
]


class BIECM2(MaxentGraph):
    """
    Bipartite enhanced configuration model.
    """

    def __init__(self, W, x_transform=0, y_transform=0):
        # validate?
        row_strengths = W.sum(axis=1).getA1().astype(np.float64)
        row_degrees = (W > 0).sum(axis=1).getA1().astype(np.float64)

        col_strengths = W.sum(axis=0).getA1().astype(np.float64)
        col_degrees = (W > 0).sum(axis=0).getA1().astype(np.float64)

        row_combinations = np.stack((row_degrees, row_strengths), axis=1)
        unique_rows, self.row_inverse, self.row_multiplicity = np.unique(
            row_combinations,
            return_index=False,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        self.row_degrees, self.row_strengths = unique_rows[:, 0], unique_rows[:, 1]

        col_combinations = np.stack((col_degrees, col_strengths), axis=1)
        unique_cols, self.col_inverse, self.col_multiplicity = np.unique(
            col_combinations,
            return_index=False,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )
        self.col_degrees, self.col_strengths = unique_cols[:, 0], unique_cols[:, 1]

        self.num_rows, self.num_cols = len(unique_rows), len(unique_cols)
        self.num_nodes = self.num_rows + self.num_cols
        self.x_transform, self.x_inv_transform = R_to_zero_to_inf[x_transform]
        self.y_transform, self.y_inv_transform = R_to_zero_to_one[y_transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * 2 * self.num_nodes)
        upper_bounds = np.array([np.inf] * self.num_nodes + [1 - EPS] * self.num_nodes)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate(
            [self.row_degrees, self.col_degrees, self.row_strengths, self.col_strengths]
        )

    @jax_class_jit
    def transform_parameters(self, v):
        x = v[: self.num_nodes]
        y = v[self.num_nodes :]

        return jnp.concatenate((self.x_transform(x), self.y_transform(y)))

    @jax_class_jit
    def transform_parameters_inv(self, v):
        x = v[: self.num_nodes]
        y = v[self.num_nodes :]

        return jnp.concatenate((self.x_inv_transform(x), self.y_inv_transform(y)))

    def get_initial_guess(self, option=5):
        """
        Just some options for initial guesses.
        """
        num_edges = np.sum(self.row_degrees)

        if option == 1:
            initial_guess = np.repeat(0.01, 2 * self.num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.10, 2 * self.num_nodes)
        elif option == 3:
            initial_guess = np.random.sample(2 * self.num_nodes)
        elif option == 4:
            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / math.sqrt(num_edges),
                        self.col_degrees / math.sqrt(num_edges),
                        np.random.sample(self.num_nodes),
                    ]
                )
            )
        elif option == 5:
            row_strength_per_degree = self.row_strengths / (1 + self.row_degrees)
            col_strength_per_degree = self.col_strengths / (1 + self.col_degrees)

            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / math.sqrt(num_edges),
                        self.col_degrees / math.sqrt(num_edges),
                        row_strength_per_degree / (row_strength_per_degree.max() + 1),
                        col_strength_per_degree / (col_strength_per_degree.max() + 1),
                    ]
                )
            )
        elif option == 6:
            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / math.sqrt(num_edges),
                        self.col_degrees / math.sqrt(num_edges),
                        np.repeat(0.10, self.num_nodes),
                    ]
                )
            )
        elif option == 7:
            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / math.sqrt(num_edges),
                        self.col_degrees / math.sqrt(num_edges),
                        np.repeat(0.01, self.num_nodes),
                    ]
                )
            )
        else:
            raise ValueError("Invalid option value. Choose from 1-7.")

        return self.transform_parameters_inv(initial_guess)

    @jax_class_jit
    def expected_node_sequence(self, v):
        z = self.transform_parameters(v)

        x_row = z[: self.num_rows]
        x_col = z[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = z[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = z[(2 * self.num_rows + self.num_cols) :]

        avg_row_degree = jnp.zeros(self.num_rows)
        avg_col_degree = jnp.zeros(self.num_cols)

        avg_row_strength = jnp.zeros(self.num_rows)
        avg_col_strength = jnp.zeros(self.num_cols)

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        degree = xx * yy / (1 - yy + xx * yy)
        strength = degree / (1 - yy)

        avg_row_degree = (degree * self.col_multiplicity).sum(axis=1)
        avg_row_strength = (strength * self.col_multiplicity).sum(axis=1)

        avg_col_degree = (degree * self.row_multiplicity[:, None]).sum(axis=0)
        avg_col_strength = (strength * self.row_multiplicity[:, None]).sum(axis=0)

        return jnp.concatenate(
            [avg_row_degree, avg_col_degree, avg_row_strength, avg_col_strength]
        )

    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        ...

    def expected_node_sequence_loops(self, v):
        z = self.transform_parameters_inv(v)

        x_row = z[: self.num_rows]
        x_col = z[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = z[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = z[(2 * self.num_rows + self.num_cols) :]

        avg_row_degree = np.zeros(self.num_rows)
        avg_col_degree = np.zeros(self.num_cols)

        avg_row_strength = np.zeros(self.num_rows)
        avg_col_strength = np.zeros(self.num_cols)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                xx = x_row[i] * x_col[j]
                yy = y_row[i] * y_col[j]

                degree = xx * yy / (1 - yy + xx * yy)
                strength = degree / (1 - yy)

                avg_row_degree[i] += degree * self.col_multiplicity[j]
                avg_row_strength[i] += strength * self.col_multiplicity[j]

                avg_col_degree[j] += degree * self.row_multiplicity[i]
                avg_col_strength[j] += strength * self.row_multiplicity[i]

        return np.concatenate(
            [avg_row_degree, avg_col_degree, avg_row_strength, avg_col_strength]
        )

    def neg_log_likelihood_loops(self, v):
        z = self.transform_parameters(v)

        llhood = 0

        x_row = z[: self.num_rows]
        x_col = z[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = z[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = z[(2 * self.num_rows + self.num_cols) :]

        for i in range(self.num_rows):
            llhood += self.row_multiplicity[i] * self.row_degrees[i] * np.log(x_row[i])
            llhood += (
                self.row_multiplicity[i] * self.row_strengths[i] * np.log(y_row[i])
            )

        for j in range(self.num_cols):
            llhood += self.col_multiplicity[j] * self.col_degrees[j] * np.log(x_col[j])
            llhood += (
                self.col_multiplicity[j] * self.col_strengths[j] * np.log(y_col[j])
            )

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                xx = x_row[i] * x_col[j]
                yy = y_row[i] * y_col[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += (
                    self.row_multiplicity[i] * self.col_multiplicity[j] * np.log(t)
                )

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)

        llhood = 0

        x_row = z[: self.num_rows]
        x_col = z[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = z[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = z[(2 * self.num_rows + self.num_cols) :]

        llhood += jnp.sum(self.row_multiplicity * self.row_degrees * jnp.log(x_row))
        llhood += jnp.sum(self.row_multiplicity * self.row_strengths * jnp.log(y_row))

        llhood += jnp.sum(self.col_multiplicity * self.col_degrees * jnp.log(x_col))
        llhood += jnp.sum(self.col_multiplicity * self.col_strengths * jnp.log(y_col))

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        t = (1 - yy) / (1 - yy + xx * yy)
        Q = jnp.log(t)
        llhood += jnp.sum(self.col_multiplicity * Q * self.row_multiplicity[:, None])

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        ...

    def get_pval_matrix(self, v, W):
        z = self.transform_parameters(v)

        # numba doesn't understand xla devicearrays
        x_row = np.array(z[: self.num_rows])
        x_col = np.array(z[self.num_rows : (self.num_rows + self.num_cols)])
        y_row = np.array(
            z[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        )
        y_col = np.array(z[(2 * self.num_rows + self.num_cols) :])

        nonzero = W.nonzero()
        nonzero_values = W[nonzero].getA1()

        # do loops with numba for efficiency
        @numba.jit(nopython=True)
        def inner_loop(
            nonzero,
            nonzero_values,
            x_row,
            x_col,
            y_row,
            y_col,
            row_inverse,
            col_inverse,
        ):
            new_values = []

            for i, j, w in zip(*nonzero, nonzero_values):
                i_star = row_inverse[i]
                j_star = col_inverse[j]

                xx = x_row[i_star] * x_col[j_star]
                yy = y_row[i_star] * y_col[j_star]
                pij = xx * yy / (1 - yy + xx * yy)

                # probability this weight would be at least this large, given null model
                p_val = pij * np.power(y_row[i_star] * y_col[j_star], w - 1)
                new_values.append(p_val)

            return new_values

        new_values = inner_loop(
            nonzero,
            nonzero_values,
            x_row,
            x_col,
            y_row,
            y_col,
            self.row_inverse,
            self.col_inverse,
        )

        W_new = sp.coo_matrix((new_values, nonzero))

        return W_new.tocsr()
