import numpy as np
import scipy.optimize


import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, jax_class_jit


class RCM(MaxentGraph):
    def __init__(self, A):
        A_dense = A.todense()
        A_t_dense = A.T.todense()
        unreciprocated = np.multiply(A_dense, 1 - A_t_dense)
        reciprocated = np.multiply(A_dense, A_t_dense)

        self.k_unr_out = unreciprocated.sum(axis=1).getA1()
        self.k_unr_in = unreciprocated.sum(axis=0).getA1()
        self.k_recip = reciprocated.sum(axis=1).getA1()

        self.num_nodes = len(self.k_unr_out)

    def bounds(self):
        lower_bounds = np.array([EPS] * 3 * self.num_nodes)
        upper_bounds = np.array([np.inf] * 3 * self.num_nodes)

        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k_unr_out, self.k_unr_in, self.k_recip])

    def get_initial_guess(self, option=1):
        if option == 1:
            return np.random.sample(3 * self.num_nodes)
        elif option == 2:
            return np.repeat(0.10, 3 * self.num_nodes)
        elif option == 3:
            return np.repeat(0.01, 3 * self.num_nodes)
        elif option == 4:
            return self.clip(self.order_node_sequence())
        else:
            assert False

    @jax_class_jit
    def expected_node_sequence(self, v):
        N = self.num_nodes

        x = v[:N]
        y = v[N : 2 * N]
        z = v[2 * N :]

        xy = jnp.outer(x, y)
        zz = jnp.outer(z, z)

        denom = 1 + xy + xy.T + zz
        unr_term_out = xy / denom
        unr_term_in = xy.T / denom
        recip_term = zz / denom

        avg_k_unr_out = unr_term_out.sum(axis=1) - jnp.diag(unr_term_out)
        avg_k_unr_in = unr_term_in.sum(axis=1) - jnp.diag(unr_term_in)
        avg_k_recip = recip_term.sum(axis=1) - jnp.diag(recip_term)

        return jnp.concatenate((avg_k_unr_out, avg_k_unr_in, avg_k_recip))

    @jax_class_jit
    def expected_node_sequence_jac(self, v):
        # todo
        assert False

    def expected_node_sequence_loops(self, v):
        N = self.num_nodes

        x = v[:N]
        y = v[N : 2 * N]
        z = v[2 * N :]

        k_unr_out_r = np.zeros(N)
        k_unr_in_r = np.zeros(N)
        k_recip_r = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                denom = 1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j]

                k_unr_out_r[i] += x[i] * y[j] / denom
                k_unr_in_r[i] += x[j] * y[i] / denom
                k_recip_r[i] += z[i] * z[j] / denom

        return np.concatenate((k_unr_out_r, k_unr_in_r, k_recip_r))

    def neg_log_likelihood_loops(self, v):
        N = self.num_nodes

        x = v[:N]
        y = v[N : 2 * N]
        z = v[2 * N :]

        llhood = 0

        for i in range(N):
            llhood += self.k_unr_out[i] * np.log(x[i])
            llhood += self.k_unr_in[i] * np.log(y[i])
            llhood += self.k_recip[i] * np.log(z[i])

        for i in range(N):
            for j in range(i + 1, N):
                llhood -= np.log(1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j])

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        N = self.num_nodes

        x = v[:N]
        y = v[N : 2 * N]
        z = v[2 * N :]

        llhood = 0

        llhood += jnp.sum(self.k_unr_out * jnp.log(x))
        llhood += jnp.sum(self.k_unr_in * jnp.log(y))
        llhood += jnp.sum(self.k_recip * jnp.log(z))

        xy = jnp.outer(x, y)
        zz = jnp.outer(z, z)

        Q = jnp.log(1 + xy + xy.T + zz)

        llhood -= jnp.sum(jnp.tril(Q))

        return -llhood

    @jax_class_jit
    def neg_log_likelihood_grad(self, v):
        # todo
        assert False
