import numpy as np
import scipy.optimize
import scipy.sparse
import math

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit


class BIECM(MaxentGraph):
    """
    Bipartite enhanced configuration model.
    """

    def __init__(self, W):
        # validate?
        self.row_strengths = W.sum(axis=1).getA1().astype(np.float64)
        self.row_degrees = (W > 0).sum(axis=1).getA1().astype(np.float64)

        self.col_strengths = W.sum(axis=0).getA1().astype(np.float64)
        self.col_degrees = (W > 0).sum(axis=0).getA1().astype(np.float64)

        self.num_rows, self.num_cols = W.shape
        self.num_nodes = self.num_rows + self.num_cols

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
            total_weight = self.row_strengths.sum()

            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / num_edges,
                        self.col_degrees / num_edges,
                        self.row_strengths / total_weight,
                        self.col_strengths / total_weight,
                    ]
                )
            )
        elif option == 6:
            total_weight = self.row_strengths.sum()
            initial_guess = self.clip(
                np.concatenate(
                    [
                        self.row_degrees / self.row_degrees.max(),
                        self.col_degrees / self.col_degrees.max(),
                        self.row_strengths / total_weight,
                        self.col_strengths / total_weight,
                    ]
                )
            )
        else:
            raise ValueError("Invalid option value. Choose from 1-6.")

        return initial_guess

    @jax_class_jit
    def expected_node_sequence(self, v):
        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        avg_row_degree = jnp.zeros(self.num_rows)
        avg_col_degree = jnp.zeros(self.num_cols)

        avg_row_strength = jnp.zeros(self.num_rows)
        avg_col_strength = jnp.zeros(self.num_cols)

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        degree = xx * yy / (1 - yy + xx * yy)
        strength = degree / (1 - yy)

        avg_row_degree = degree.sum(axis=1)
        avg_row_strength = strength.sum(axis=1)

        avg_col_degree = degree.sum(axis=0)
        avg_col_strength = strength.sum(axis=0)

        return jnp.concatenate(
            [avg_row_degree, avg_col_degree, avg_row_strength, avg_col_strength]
        )

    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        """
        Using this little trick to multiply a vector with a matrix column-wise:
        A * b[:, None]

        https://stackoverflow.com/a/45895371/4749956

        """

        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        inv_denom = jnp.power(1 / (yy * (xx - 1) + 1), 2)

        m1 = x_col * yy * (1 - yy) * inv_denom
        d_k_row_x_row = jnp.diag(jnp.sum(m1, axis=1))

        d_k_row_x_col = x_row[:, None] * yy * (1 - yy) * inv_denom

        m3 = xx * y_col * inv_denom
        d_k_row_y_row = jnp.diag(jnp.sum(m3, axis=1))

        d_k_row_y_col = xx * y_row[:, None] * inv_denom

        first_row_block = jnp.concatenate(
            (d_k_row_x_row, d_k_row_x_col, d_k_row_y_row, d_k_row_y_col), axis=1
        )

        d_k_col_x_row = m1.T
        d_k_col_x_col = jnp.diag(jnp.sum(d_k_row_x_col, axis=0))
        d_k_col_y_row = m3.T
        d_k_col_y_col = jnp.diag(jnp.sum(d_k_row_y_col, axis=0))

        second_row_block = jnp.concatenate(
            (d_k_col_x_row, d_k_col_x_col, d_k_col_y_row, d_k_col_y_col), axis=1
        )

        m5 = yy * x_col * inv_denom
        d_s_row_x_row = jnp.diag(jnp.sum(m5, axis=1))

        d_s_row_x_col = yy * x_row[:, None] * inv_denom

        inv_denom2 = jnp.power(1 / (yy - 1), 2) * inv_denom

        m7 = xx * (y_row[:, None] * yy * y_col * y_col * (xx - 1) + y_col) * inv_denom2
        d_s_row_y_row = jnp.diag(jnp.sum(m7, axis=1))

        d_s_row_y_col = (
            xx
            * (y_row[:, None] * y_row[:, None] * yy * y_col * (xx - 1) + y_row[:, None])
            * inv_denom2
        )

        third_row_block = jnp.concatenate(
            (d_s_row_x_row, d_s_row_x_col, d_s_row_y_row, d_s_row_y_col), axis=1
        )

        d_s_col_x_row = m5.T
        d_s_col_x_col = jnp.diag(jnp.sum(d_s_row_x_col, axis=0))
        d_s_col_y_row = m7.T
        d_s_col_y_col = jnp.diag(jnp.sum(d_s_row_y_col, axis=0))

        fourth_row_block = jnp.concatenate(
            (d_s_col_x_row, d_s_col_x_col, d_s_col_y_row, d_s_col_y_col), axis=1
        )

        return jnp.concatenate(
            (first_row_block, second_row_block, third_row_block, fourth_row_block),
            axis=0,
        )

    def expected_node_sequence_loops(self, v):
        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

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

                avg_row_degree[i] += degree
                avg_row_strength[i] += strength

                avg_col_degree[j] += degree
                avg_col_strength[j] += strength

        return np.concatenate(
            [avg_row_degree, avg_col_degree, avg_row_strength, avg_col_strength]
        )

    def neg_log_likelihood_loops(self, v):
        llhood = 0

        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        for i in range(self.num_rows):
            llhood += self.row_degrees[i] * np.log(x_row[i])
            llhood += self.row_strengths[i] * np.log(y_row[i])

        for j in range(self.num_cols):
            llhood += self.col_degrees[j] * np.log(x_col[j])
            llhood += self.col_strengths[j] * np.log(y_col[j])

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                xx = x_row[i] * x_col[j]
                yy = y_row[i] * y_col[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += np.log(t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        llhood = 0

        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        llhood += jnp.sum(self.row_degrees * jnp.log(x_row))
        llhood += jnp.sum(self.row_strengths * jnp.log(y_row))

        llhood += jnp.sum(self.col_degrees * jnp.log(x_col))
        llhood += jnp.sum(self.col_strengths * jnp.log(y_col))

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        t = (1 - yy) / (1 - yy + xx * yy)
        llhood += jnp.sum(jnp.log(t))

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        xx = jnp.outer(x_row, x_col)
        yy = jnp.outer(y_row, y_col)

        denom = 1 - yy + xx * yy

        m1 = (x_col * yy) / denom
        m2 = (x_row[:, None] * yy) / denom

        d_x_row = self.row_degrees / x_row - jnp.sum(m1, axis=1)
        d_x_col = self.col_degrees / x_col - jnp.sum(m2, axis=0)

        m3 = -y_col * (1 / (1 - yy))
        m4 = (-y_col + xx * y_col) / denom
        d_y_row = self.row_strengths / y_row + jnp.sum(m3, axis=1) - jnp.sum(m4, axis=1)

        m5 = -y_row[:, None] * (1 / (1 - yy))
        m6 = (-y_row[:, None] + xx * y_row[:, None]) / denom
        d_y_col = self.col_strengths / y_col + jnp.sum(m5, axis=0) - jnp.sum(m6, axis=0)

        return -jnp.concatenate((d_x_row, d_x_col, d_y_row, d_y_col))

    def get_pval_matrix(self, v, W):
        x_row = v[: self.num_rows]
        x_col = v[self.num_rows : (self.num_rows + self.num_cols)]
        y_row = v[(self.num_rows + self.num_cols) : (2 * self.num_rows + self.num_cols)]
        y_col = v[(2 * self.num_rows + self.num_cols) :]

        # only need one triangle since symmetric
        # convert to lil for fast index assignment
        # convert to float because may be int at this point
        W_new = W.copy().tolil().astype(np.float64)

        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx = x_row[i] * x_col[j]
            yy = y_row[i] * y_col[j]
            pij = xx * yy / (1 - yy + xx * yy)

            # probability this weight would be at least this large, given null model
            p_val = pij * np.power(y_row[i] * y_col[j], w - 1)
            W_new[i, j] = p_val

        return W_new

