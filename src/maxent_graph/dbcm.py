import numpy as np
import scipy.optimize

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, R_to_zero_to_inf, flatten, jax_class_jit


class DBCM(MaxentGraph):
    """
    Directed binary configuration model.

    The degree-only counterpart of the DECM. ``A[i, j] != 0`` means an edge
    from i to j; self-loops are ignored.
    """

    def __init__(self, A, transform=0):
        A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
        A = A.copy()
        np.fill_diagonal(A, 0)

        self.k_out = flatten((A > 0).sum(axis=1)).astype(np.float64)
        self.k_in = flatten((A > 0).sum(axis=0)).astype(np.float64)
        self.num_nodes = len(self.k_out)

        self.transform, self.inv_transform = R_to_zero_to_inf[transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * 2 * self.num_nodes)
        upper_bounds = np.array([np.inf] * 2 * self.num_nodes)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return np.concatenate([self.k_out, self.k_in])

    @jax_class_jit
    def transform_parameters(self, v):
        return self.transform(v)

    @jax_class_jit
    def transform_parameters_inv(self, v):
        return self.inv_transform(v)

    def get_initial_guess(self, option=4):
        """
        Just some options for initial guesses.
        """
        num_edges = np.sum(self.k_out)
        ks = np.concatenate([self.k_out, self.k_in])

        if option == 1:
            initial_guess = np.random.sample(2 * self.num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.01, 2 * self.num_nodes)
        elif option == 3:
            initial_guess = ks / ks.max()
        elif option == 4:
            initial_guess = ks / np.sqrt(num_edges + 1)
        else:
            raise ValueError("Invalid option value. Choose from 1-4.")

        return self.transform_parameters_inv(self.clip(initial_guess))

    @jax_class_jit
    def expected_node_sequence(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        xy = jnp.outer(x, y)
        pij = xy / (1 + xy)
        pij = pij - jnp.diag(jnp.diag(pij))

        return jnp.concatenate((pij.sum(axis=1), pij.sum(axis=0)))

    def expected_node_sequence_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        avg_k_out = np.zeros(N)
        avg_k_in = np.zeros(N)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                xy = x[i] * y[j]
                pij = xy / (1 + xy)
                avg_k_out[i] += pij
                avg_k_in[j] += pij

        return np.concatenate([avg_k_out, avg_k_in])

    def neg_log_likelihood_loops(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = 0

        for i in range(N):
            llhood += self.k_out[i] * np.log(x[i])
            llhood += self.k_in[i] * np.log(y[i])

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                llhood -= np.log(1 + x[i] * y[j])

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        z = self.transform_parameters(v)
        N = self.num_nodes

        x = z[:N]
        y = z[N:]

        llhood = jnp.sum(self.k_out * jnp.log(x))
        llhood += jnp.sum(self.k_in * jnp.log(y))

        log_t = jnp.log(1 + jnp.outer(x, y))
        llhood -= jnp.sum(log_t) - jnp.sum(jnp.diag(jnp.diag(log_t)))

        return -llhood
