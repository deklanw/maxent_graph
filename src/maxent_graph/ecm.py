import numpy as np
import scipy.optimize
import scipy.sparse

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, R_to_zero_to_inf, R_to_zero_to_one, flatten, jax_class_jit


class ECM(MaxentGraph):
    """
    (Undirected) Enhanced configuration model.
    """

    def __init__(self, W, x_transform=0, y_transform=0):
        # validate?

        # ignore self-loops
        W -= scipy.sparse.diags(W.diagonal())

        self.k = flatten((W > 0).sum(axis=1)).astype(np.float64)
        self.s = flatten(W.sum(axis=1)).astype(np.float64)

        self.num_nodes = len(self.k)

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
        return np.concatenate([self.k, self.s])

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
        num_nodes = len(self.k)
        num_edges = np.sum(self.k) / 2

        ks = self.k
        ss = self.s

        if option == 1:
            initial_guess = np.random.sample(2 * num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.01, 2 * num_nodes)
        elif option == 3:
            initial_guess = np.repeat(0.10, 2 * num_nodes)
        elif option == 4:
            initial_guess = np.concatenate([ks / ks.max(), ss / ss.max()])
        elif option == 5:
            initial_guess = np.concatenate(
                [ks / np.sqrt(num_edges), np.random.sample(num_nodes)]
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
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        pij = xx * yy / (1 - yy + xx * yy)
        pij = pij - jnp.diag(jnp.diag(pij))
        avg_k = pij.sum(axis=1)

        sij = pij / (1 - yy)
        # don't need to zero out diagonal again, still 0
        avg_s = sij.sum(axis=1)

        return jnp.concatenate((avg_k, avg_s))

    def expected_node_sequence_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        avg_k = np.zeros(N)
        avg_s = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                xx = x[i] * x[j]
                yy = y[i] * y[j]

                pij = xx * yy / (1 - yy + xx * yy)

                avg_k[i] += pij
                avg_s[i] += pij / (1 - yy)

        return np.concatenate([avg_k, avg_s])

    def neg_log_likelihood_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = 0

        for i in range(N):
            llhood += self.k[i] * np.log(x[i])
            llhood += self.s[i] * np.log(y[i])

        for i in range(N):
            for j in range(i):
                xx = x[i] * x[j]
                yy = y[i] * y[j]
                t = (1 - yy) / (1 - yy + xx * yy)
                llhood += np.log(t)

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = 0

        llhood += jnp.sum(self.k * jnp.log(x))
        llhood += jnp.sum(self.s * jnp.log(y))

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        t = (1 - yy) / (1 - yy + xx * yy)
        log_t = jnp.log(t)
        llhood += jnp.sum(log_t) - jnp.sum(jnp.tril(log_t))

        return -llhood

    def get_pval_matrix(self, v, W):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        # only need one triangle since symmetric
        # convert to lil for fast index assignment
        # convert to float because may be int at this point
        W_new = scipy.sparse.tril(W.copy()).tolil().astype(np.float64)

        for i, j in zip(*W.nonzero()):
            w = W[i, j]
            xx_out = x[i] * x[j]
            yy_out = y[i] * y[j]
            pij = xx_out * yy_out / (1 - yy_out + xx_out * yy_out)

            # probability this weight would be at least this large, given null model
            p_val = pij * np.power(y[i] * y[j], w - 1)
            W_new[i, j] = p_val

        return W_new
