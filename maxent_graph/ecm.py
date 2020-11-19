import numpy as np
import scipy.optimize
import scipy.sparse

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit


class ECM(MaxentGraph):
    """
    (Undirected) Enhanced configuration model.
    """

    def __init__(self, W):
        # validate?

        # ignore self-loops
        W -= scipy.sparse.diags(W.diagonal())

        self.k = (W > 0).sum(axis=1).getA1().astype("float64")
        self.s = W.sum(axis=1).getA1()

        self.num_nodes = len(self.k)

    def bounds(self):
        lower_bounds = np.array([EPS] * 2 * self.num_nodes)
        upper_bounds = np.array([np.inf] * self.num_nodes + [1 - EPS] * self.num_nodes)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k, self.s])

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
            initial_guess = self.clip(np.concatenate([ks / ks.max(), ss / ss.max()]))
        elif option == 5:
            initial_guess = self.clip(
                np.concatenate([ks / np.sqrt(num_edges), np.random.sample(num_nodes)])
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

        x = v[:N]
        y = v[N:]

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        pij = xx * yy / (1 - yy + xx * yy)
        pij = pij - jnp.diag(jnp.diag(pij))
        avg_k = pij.sum(axis=1)

        sij = pij / (1 - yy)
        # don't need to zero out diagonal again, still 0
        avg_s = sij.sum(axis=1)

        return jnp.concatenate((avg_k, avg_s))

    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        """
        Using this little trick to multiply a vector with a matrix column-wise:
        A * b[:, None]

        https://stackoverflow.com/a/45895371/4749956

        """
        N = self.num_nodes

        x = v[:N]
        y = v[N:]

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        inv_denom = jnp.power(1 / (yy * (xx - 1) + 1), 2)

        m1 = x[:, None] * yy * (1 - yy) * inv_denom
        m1 -= jnp.diag(jnp.diag(m1))
        m2 = x * yy * (1 - yy) * inv_denom
        m2 -= jnp.diag(jnp.diag(m2))
        d_k_x = m1 + jnp.diag(jnp.sum(m2, axis=1))

        m3 = xx * y[:, None] * inv_denom
        m3 -= jnp.diag(jnp.diag(m3))
        m4 = xx * y * inv_denom
        m4 -= jnp.diag(jnp.diag(m4))
        d_k_y = m3 + jnp.diag(jnp.sum(m4, axis=1))

        first_row_block = jnp.concatenate((d_k_x, d_k_y), axis=1)

        m5 = yy * x[:, None] * inv_denom
        m5 -= jnp.diag(jnp.diag(m5))
        m6 = yy * x * inv_denom
        m6 -= jnp.diag(jnp.diag(m6))
        d_s_x = m5 + jnp.diag(jnp.sum(m6, axis=1))

        inv_denom2 = jnp.power(1 / (yy - 1), 2) * inv_denom

        m7 = (
            xx * (y[:, None] * y[:, None] * yy * y * (xx - 1) + y[:, None]) * inv_denom2
        )
        m7 -= jnp.diag(jnp.diag(m7))
        m8 = xx * (y[:, None] * yy * y * y * (xx - 1) + y) * inv_denom2
        m8 -= jnp.diag(jnp.diag(m8))
        d_s_y = m7 + jnp.diag(jnp.sum(m8, axis=1))

        second_row_block = jnp.concatenate((d_s_x, d_s_y), axis=1)

        return jnp.concatenate((first_row_block, second_row_block), axis=0)

    def expected_node_sequence_loops(self, v):
        N = self.num_nodes

        x = v[:N]
        y = v[N:]

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
        llhood = 0

        N = self.num_nodes

        x = v[:N]
        y = v[N:]

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
        llhood = 0

        N = self.num_nodes

        x = v[:N]
        y = v[N:]

        llhood += jnp.sum(self.k * jnp.log(x))
        llhood += jnp.sum(self.s * jnp.log(y))

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        t = (1 - yy) / (1 - yy + xx * yy)
        log_t = jnp.log(t)
        llhood += jnp.sum(log_t) - jnp.sum(jnp.tril(log_t))

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        """

        TODO: Check if I can make this more efficient. It already scales better than AD.
        """
        N = self.num_nodes

        x = v[:N]
        y = v[N:]

        xx = jnp.outer(x, x)
        yy = jnp.outer(y, y)

        denom = 1 - yy + xx * yy

        m1 = (x * yy) / denom
        d_x = self.k / x - jnp.sum(m1, axis=1) + jnp.diag(m1)

        m2 = -y * (1 / (1 - yy))
        m3 = (-y + xx * y) / denom
        d_y = (
            self.s / y
            + jnp.sum(m2, axis=1)
            - jnp.diag(m2)
            - jnp.sum(m3, axis=1)
            + jnp.diag(m3)
        )

        return -jnp.concatenate((d_x, d_y))

    def get_pval_matrix(self, v, W):
        N = self.num_nodes

        x = v[:N]
        y = v[N:]

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

