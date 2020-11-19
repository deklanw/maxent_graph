import numpy as np
import scipy.optimize
import scipy.sparse

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit


class DECM(MaxentGraph):
    """
    Directed enhanced configuration model.
    """

    def __init__(self, W):
        """
        Wij = 2.0 means an edge from i -> j with weight 2.0

        Ensure you're following this convention. graph-tool, for instance, has this reversed.
        Just transpose before passing if that's the case.
        """
        # validate?

        # ignore self-loops
        W -= scipy.sparse.diags(W.diagonal())

        self.k_out = (W > 0).sum(axis=1).getA1().astype("float64")
        self.k_in = (W > 0).sum(axis=0).getA1().astype("float64")
        self.s_out = W.sum(axis=1).getA1()
        self.s_in = W.sum(axis=0).getA1()

        self.num_nodes = len(self.k_out)

    def bounds(self):
        lower_bounds = np.array([EPS] * 4 * self.num_nodes)
        upper_bounds = np.array(
            [np.inf] * 2 * self.num_nodes + [1 - EPS] * 2 * self.num_nodes
        )
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k_out, self.k_in, self.s_out, self.s_in])

    def get_initial_guess(self, option=6):
        """
        Just some options for initial guesses.
        """
        num_nodes = len(self.k_out)
        num_edges = np.sum(self.k_out)

        ks = np.concatenate([self.k_out, self.k_in])
        ss = np.concatenate([self.s_out, self.s_in])

        if option == 1:
            initial_guess = np.random.sample(4 * num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.01, 4 * num_nodes)
        elif option == 3:
            initial_guess = np.repeat(0.10, 4 * num_nodes)
        elif option == 4:
            initial_guess = self.clip(np.concatenate([ks / ks.max(), ss / ss.max()]))
        elif option == 5:
            initial_guess = self.clip(
                np.concatenate(
                    [ks / np.sqrt(num_edges), np.random.sample(2 * num_nodes)]
                )
            )
        elif option == 6:
            xs_guess = ks / np.sqrt(num_edges)
            s_per_k = ss / (ks + 1)
            ys_guess = s_per_k / s_per_k.max()
            initial_guess = self.clip(np.concatenate([xs_guess, ys_guess]))
        else:
            raise ValueError("Invalid option value. Choose from 1-6.")

        return initial_guess

    @jax_class_jit
    def expected_node_sequence(self, v):
        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        pij = xx * yy / (1 - yy + xx * yy)
        pij = pij - jnp.diag(jnp.diag(pij))
        avg_k_out = pij.sum(axis=1)
        avg_k_in = pij.sum(axis=0)

        sij = pij / (1 - yy)
        # don't need to zero out diagonals again because still 0 after division
        avg_s_out = sij.sum(axis=1)
        avg_s_in = sij.sum(axis=0)

        return jnp.concatenate((avg_k_out, avg_k_in, avg_s_out, avg_s_in))

    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        """
        Using this little trick to multiply a vector with a matrix column-wise:
        A * b[:, None]
        
        https://stackoverflow.com/a/45895371/4749956
        
        """
        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        inv_denom = jnp.power(1 / (yy * (xx - 1) + 1), 2)

        m1 = x_in * yy * (1 - yy) * inv_denom
        d_k_out_x_out = jnp.diag(jnp.sum(m1, axis=1) - jnp.diag(m1))

        m2 = x_out[:, None] * yy * (1 - yy) * inv_denom
        d_k_out_x_in = m2 - jnp.diag(jnp.diag(m2))

        m3 = xx * y_in * inv_denom
        d_k_out_y_out = jnp.diag(jnp.sum(m3, axis=1) - jnp.diag(m3))

        m4 = xx * y_out[:, None] * inv_denom
        d_k_out_y_in = m4 - jnp.diag(jnp.diag(m4))

        first_row_block = jnp.concatenate(
            (d_k_out_x_out, d_k_out_x_in, d_k_out_y_out, d_k_out_y_in), axis=1
        )

        d_k_in_x_out = m1.T - jnp.diag(jnp.diag(m1))
        d_k_in_x_in = jnp.diag(jnp.sum(m2, axis=0) - jnp.diag(m2))
        d_k_in_y_out = m3.T - jnp.diag(jnp.diag(m3))
        d_k_in_y_in = jnp.diag(jnp.sum(m4, axis=0) - jnp.diag(m4))

        second_row_block = jnp.concatenate(
            (d_k_in_x_out, d_k_in_x_in, d_k_in_y_out, d_k_in_y_in), axis=1
        )

        m5 = yy * x_in * inv_denom
        d_s_out_x_out = jnp.diag(jnp.sum(m5, axis=1) - jnp.diag(m5))

        m6 = yy * x_out[:, None] * inv_denom
        d_s_out_x_in = m6 - jnp.diag(jnp.diag(m6))

        inv_denom2 = jnp.power(1 / (yy - 1), 2) * inv_denom

        m7 = xx * (y_out[:, None] * yy * y_in * y_in * (xx - 1) + y_in) * inv_denom2
        d_s_out_y_out = jnp.diag(jnp.sum(m7, axis=1) - jnp.diag(m7))

        m8 = (
            xx
            * (y_out[:, None] * y_out[:, None] * yy * y_in * (xx - 1) + y_out[:, None])
            * inv_denom2
        )
        d_s_out_y_in = m8 - jnp.diag(jnp.diag(m8))

        third_row_block = jnp.concatenate(
            (d_s_out_x_out, d_s_out_x_in, d_s_out_y_out, d_s_out_y_in), axis=1
        )

        d_s_in_x_out = m5.T - jnp.diag(jnp.diag(m5))
        d_s_in_x_in = jnp.diag(jnp.sum(m6, axis=0) - jnp.diag(m6))
        d_s_in_y_out = m7.T - jnp.diag(jnp.diag(m7))
        d_s_in_y_in = jnp.diag(jnp.sum(m8, axis=0) - jnp.diag(m8))

        fourth_row_block = jnp.concatenate(
            (d_s_in_x_out, d_s_in_x_in, d_s_in_y_out, d_s_in_y_in), axis=1
        )

        return jnp.concatenate(
            (first_row_block, second_row_block, third_row_block, fourth_row_block),
            axis=0,
        )

    def expected_node_sequence_loops(self, v):
        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        # initialize the residuals
        avg_k_out_r = np.zeros(N)
        avg_k_in_r = np.zeros(N)
        avg_s_out_r = np.zeros(N)
        avg_s_in_r = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                xx_out = x_out[i] * x_in[j]
                yy_out = y_out[i] * y_in[j]

                xx_in = x_out[j] * x_in[i]
                yy_in = y_out[j] * y_in[i]

                pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)
                pji = xx_in * yy_in / (1 - yy_in + xx_in * yy_in)

                avg_k_out_r[i] += pij
                avg_k_in_r[i] += pji
                avg_s_out_r[i] += pij / (1 - yy_out)
                avg_s_in_r[i] += pji / (1 - yy_in)

        return np.concatenate([avg_k_out_r, avg_k_in_r, avg_s_out_r, avg_s_in_r])

    def neg_log_likelihood_loops(self, v):
        # nll not written out in paper afaict... worked it out myself
        # hope it's right :)

        llhood = 0

        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        for i in range(N):
            llhood += self.k_out[i] * np.log(x_out[i])
            llhood += self.k_in[i] * np.log(x_in[i])
            llhood += self.s_out[i] * np.log(y_out[i])
            llhood += self.s_in[i] * np.log(y_in[i])

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                xx = x_out[i] * x_in[j]
                yy = y_out[i] * y_in[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += np.log(t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        llhood = 0

        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        llhood += jnp.sum(self.k_out * jnp.log(x_out))
        llhood += jnp.sum(self.k_in * jnp.log(x_in))
        llhood += jnp.sum(self.s_out * jnp.log(y_out))
        llhood += jnp.sum(self.s_in * jnp.log(y_in))

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        t = (1 - yy) / (1 - yy + xx * yy)
        log_t = jnp.log(t)
        llhood += jnp.sum(log_t) - jnp.trace(log_t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        """

        TODO: Check if I can make this more efficient. It already scales better than AD.
        """
        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        denom = 1 - yy + xx * yy

        m1 = (x_in * yy) / denom
        d_x_out = self.k_out / x_out - jnp.sum(m1, axis=1) + jnp.diag(m1)

        m2 = (x_out * yy.T).T / denom
        d_x_in = self.k_in / x_in - jnp.sum(m2, axis=0) + jnp.diag(m2)

        m3 = -y_in * (1 / (1 - yy))
        m4 = (-y_in + xx * y_in) / denom
        d_y_out = (
            self.s_out / y_out
            + jnp.sum(m3, axis=1)
            - jnp.diag(m3)
            - jnp.sum(m4, axis=1)
            + jnp.diag(m4)
        )

        m5 = (-y_out * (1 / (1 - yy)).T).T
        m6 = (-y_out + xx.T * y_out).T / denom
        d_y_in = (
            self.s_in / y_in
            + jnp.sum(m5, axis=0)
            - jnp.diag(m5)
            - jnp.sum(m6, axis=0)
            + jnp.diag(m6)
        )

        return -jnp.concatenate((d_x_out, d_x_in, d_y_out, d_y_in))

    def get_pval_matrix(self, v, W):
        N = self.num_nodes

        x_out = v[:N]
        x_in = v[N : 2 * N]
        y_out = v[2 * N : 3 * N]
        y_in = v[3 * N :]

        W_new = W.copy().tolil()

        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx_out = x_out[i] * x_in[j]
            yy_out = y_out[i] * y_in[j]
            pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)

            # probability this weight would be at least this large, given null model
            p_val = pij * np.power(y_out[i] * y_in[j], w - 1)
            W_new[i, j] = p_val

        return W_new
