import numpy as np
import scipy.optimize
import scipy.sparse

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, R_to_zero_to_inf, R_to_zero_to_one, jax_class_jit


class DECM(MaxentGraph):
    """
    Directed enhanced configuration model.
    """

    def __init__(self, W, x_transform=0, y_transform=0):
        """
        Wij = 2.0 means an edge from i -> j with weight 2.0

        Ensure you're following this convention. graph-tool, for instance, has this reversed.
        Just transpose before passing if that's the case.
        """
        self.k_out = (W > 0).sum(axis=1).astype(np.float64)
        self.k_in = (W > 0).sum(axis=0).astype(np.float64)
        self.s_out = W.sum(axis=1).astype(np.float64)
        self.s_in = W.sum(axis=0).astype(np.float64)

        self.num_nodes = len(self.k_out)

        self.x_transform, self.x_inv_transform = R_to_zero_to_inf[x_transform]
        self.y_transform, self.y_inv_transform = R_to_zero_to_one[y_transform]

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

    @jax_class_jit
    def transform_parameters(self, v):
        x = v[: 2 * self.num_nodes]
        y = v[2 * self.num_nodes :]

        return jnp.concatenate((self.x_transform(x), self.y_transform(y)))

    @jax_class_jit
    def transform_parameters_inv(self, v):
        x = v[: 2 * self.num_nodes]
        y = v[2 * self.num_nodes :]

        return jnp.concatenate((self.x_inv_transform(x), self.y_inv_transform(y)))

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
            initial_guess = np.concatenate([ks / ks.max(), ss / ss.max()])
        elif option == 5:
            initial_guess = np.concatenate(
                [ks / np.sqrt(num_edges), np.random.sample(2 * num_nodes)]
            )
        elif option == 6:
            xs_guess = ks / np.sqrt(num_edges)
            s_per_k = ss / (ks + 1)
            ys_guess = s_per_k / s_per_k.max()
            initial_guess = np.concatenate([xs_guess, ys_guess])
        else:
            raise ValueError("Invalid option value. Choose from 1-6.")

        return self.transform_parameters_inv(self.clip(initial_guess))

    @jax_class_jit
    def expected_node_sequence(self, v):
        N = self.num_nodes
        z = self.transform_parameters(v)

        x_out = z[:N]
        x_in = z[N : 2 * N]
        y_out = z[2 * N : 3 * N]
        y_in = z[3 * N :]

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        pij = xx * yy / (1 - yy + xx * yy)

        avg_k_out = pij.sum(axis=1)
        avg_k_in = pij.sum(axis=0)

        sij = pij / (1 - yy)
        # don't need to zero out diagonals again because still 0 after division
        avg_s_out = sij.sum(axis=1)
        avg_s_in = sij.sum(axis=0)

        return jnp.concatenate((avg_k_out, avg_k_in, avg_s_out, avg_s_in))

    def expected_node_sequence_loops(self, v):
        N = self.num_nodes
        z = self.transform_parameters(v)

        x_out = z[:N]
        x_in = z[N : 2 * N]
        y_out = z[2 * N : 3 * N]
        y_in = z[3 * N :]

        # initialize the residuals
        avg_k_out_r = np.zeros(N)
        avg_k_in_r = np.zeros(N)
        avg_s_out_r = np.zeros(N)
        avg_s_in_r = np.zeros(N)

        for i in range(N):
            for j in range(N):

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

        z = self.transform_parameters(v)

        llhood = 0

        N = self.num_nodes

        x_out = z[:N]
        x_in = z[N : 2 * N]
        y_out = z[2 * N : 3 * N]
        y_in = z[3 * N :]

        for i in range(N):
            llhood += self.k_out[i] * np.log(x_out[i])
            llhood += self.k_in[i] * np.log(x_in[i])
            llhood += self.s_out[i] * np.log(y_out[i])
            llhood += self.s_in[i] * np.log(y_in[i])

        for i in range(N):
            for j in range(N):
                xx = x_out[i] * x_in[j]
                yy = y_out[i] * y_in[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += np.log(t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)

        llhood = 0

        N = self.num_nodes

        x_out = z[:N]
        x_in = z[N : 2 * N]
        y_out = z[2 * N : 3 * N]
        y_in = z[3 * N :]

        llhood += jnp.sum(self.k_out * jnp.log(x_out))
        llhood += jnp.sum(self.k_in * jnp.log(x_in))
        llhood += jnp.sum(self.s_out * jnp.log(y_out))
        llhood += jnp.sum(self.s_in * jnp.log(y_in))

        xx = jnp.outer(x_out, x_in)
        yy = jnp.outer(y_out, y_in)

        t = (1 - yy) / (1 - yy + xx * yy)
        llhood += jnp.sum(jnp.log(t))

        return -llhood

    def get_surprise_matrix(self, v, W):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x_out = z[:N]
        x_in = z[N : 2 * N]
        y_out = z[2 * N : 3 * N]
        y_in = z[3 * N :]

        W_new = W.copy().tolil()

        # p=1 for w=0. so, the surprise is 0 for those cases. just ignore.
        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx_out = x_out[i] * x_in[j]
            yy_out = y_out[i] * y_in[j]
            pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)

            # probability this weight would be at least this large, given null model
            p_val = pij * np.power(y_out[i] * y_in[j], w - 1)
            W_new[i, j] = -np.log(p_val)

        return W_new
