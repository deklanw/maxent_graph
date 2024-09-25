import numpy as np
import scipy.optimize


import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, flatten, jax_class_jit, R_to_zero_to_inf


class RCM(MaxentGraph):
    def __init__(self, A, transform=0):
        A_dense = A.todense()
        A_t_dense = A_dense.T

        unreciprocated = np.multiply(A_dense, np.logical_xor(A_dense, A_t_dense))
        reciprocated = np.multiply(A_dense, A_t_dense)

        self.k_unr_out = flatten(unreciprocated.sum(axis=1)).astype(np.float64)
        self.k_unr_in = flatten(unreciprocated.sum(axis=0)).astype(np.float64)
        self.k_recip = flatten(reciprocated.sum(axis=1)).astype(np.float64)

        # sanity checking
        k_out = flatten(A_dense.sum(axis=1)).astype(np.float64)
        k_in = flatten(A_dense.sum(axis=0)).astype(np.float64)

        assert np.allclose(self.k_unr_out + self.k_recip, k_out)
        assert np.allclose(self.k_unr_in + self.k_recip, k_in)

        self.num_nodes = len(self.k_unr_out)
        self.transform, self.inv_transform = R_to_zero_to_inf[transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * 3 * self.num_nodes)
        upper_bounds = np.array([np.inf] * 3 * self.num_nodes)

        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k_unr_out, self.k_unr_in, self.k_recip])

    @jax_class_jit
    def transform_parameters(self, v):
        return self.transform(v)

    @jax_class_jit
    def transform_parameters_inv(self, v):
        return self.inv_transform(v)

    def get_initial_guess(self, option=4):
        if option == 1:
            g = np.random.sample(3 * self.num_nodes)
        elif option == 2:
            g = np.repeat(0.10, 3 * self.num_nodes)
        elif option == 3:
            g = np.repeat(0.01, 3 * self.num_nodes)
        elif option == 4:
            g = self.order_node_sequence()
        else:
            raise ValueError("Invalid option value. Choose from 1-4.")

        return self.transform_parameters_inv(self.clip(g))

    @jax_class_jit
    def expected_node_sequence(self, v):
        t = self.transform_parameters(v)
        N = self.num_nodes

        x = t[:N]
        y = t[N : 2 * N]
        z = t[2 * N :]

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

    def expected_node_sequence_loops(self, v):
        t = self.transform_parameters(v)
        N = self.num_nodes

        x = t[:N]
        y = t[N : 2 * N]
        z = t[2 * N :]

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
        t = self.transform_parameters(v)
        N = self.num_nodes

        x = t[:N]
        y = t[N : 2 * N]
        z = t[2 * N :]

        llhood = 0

        for i in range(N):
            llhood += self.k_unr_out[i] * np.log(x[i])
            llhood += self.k_unr_in[i] * np.log(y[i])
            llhood += self.k_recip[i] * np.log(z[i])

        for i in range(N):
            for j in range(i):
                llhood -= np.log(1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j])

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        t = self.transform_parameters(v)
        N = self.num_nodes

        x = t[:N]
        y = t[N : 2 * N]
        z = t[2 * N :]

        llhood = 0

        llhood += jnp.sum(self.k_unr_out * jnp.log(x))
        llhood += jnp.sum(self.k_unr_in * jnp.log(y))
        llhood += jnp.sum(self.k_recip * jnp.log(z))

        xy = jnp.outer(x, y)
        zz = jnp.outer(z, z)

        Q = jnp.log(1 + xy + xy.T + zz)

        llhood -= jnp.sum(jnp.tril(Q, -1))

        return -llhood
