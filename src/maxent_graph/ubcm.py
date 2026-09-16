import numpy as np
import scipy.optimize
import scipy.sparse

import jax.numpy as jnp

from .MaxentGraph import MaxentGraph
from .util import EPS, R_to_zero_to_inf, flatten, jax_class_jit


class UBCM(MaxentGraph):
    """
    (Undirected) binary configuration model.

    The degree-only counterpart of the ECM, and the unipartite counterpart of
    the BiCM. The ECM fit does not expose its presence part separately, so
    this is it as a model in its own right.
    """

    def __init__(self, A, transform=0):
        # ignore self-loops
        A = A - scipy.sparse.diags(A.diagonal())

        self.k = flatten((A > 0).sum(axis=1)).astype(np.float64)
        self.num_nodes = len(self.k)

        self.transform, self.inv_transform = R_to_zero_to_inf[transform]

    def bounds(self):
        lower_bounds = np.array([EPS] * self.num_nodes)
        upper_bounds = np.array([np.inf] * self.num_nodes)
        return (
            (lower_bounds, upper_bounds),
            scipy.optimize.Bounds(lower_bounds, upper_bounds),
        )

    def order_node_sequence(self):
        return self.k

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
        num_edges = np.sum(self.k) / 2

        if option == 1:
            initial_guess = np.random.sample(self.num_nodes)
        elif option == 2:
            initial_guess = np.repeat(0.01, self.num_nodes)
        elif option == 3:
            initial_guess = self.k / self.k.max()
        elif option == 4:
            initial_guess = self.k / np.sqrt(2 * num_edges + 1)
        else:
            raise ValueError("Invalid option value. Choose from 1-4.")

        return self.transform_parameters_inv(self.clip(initial_guess))

    @jax_class_jit
    def expected_node_sequence(self, v):
        x = self.transform_parameters(v)

        xx = jnp.outer(x, x)
        pij = xx / (1 + xx)
        pij = pij - jnp.diag(jnp.diag(pij))

        return pij.sum(axis=1)

    def expected_node_sequence_loops(self, v):
        x = self.transform_parameters(v)

        avg_k = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                xx = x[i] * x[j]
                avg_k[i] += xx / (1 + xx)

        return avg_k

    def neg_log_likelihood_loops(self, v):
        x = self.transform_parameters(v)

        llhood = 0

        for i in range(self.num_nodes):
            llhood += self.k[i] * np.log(x[i])

        for i in range(self.num_nodes):
            for j in range(i):
                llhood -= np.log(1 + x[i] * x[j])

        return -llhood

    @jax_class_jit
    def neg_log_likelihood(self, v):
        x = self.transform_parameters(v)

        llhood = jnp.sum(self.k * jnp.log(x))

        log_t = jnp.log(1 + jnp.outer(x, x))
        # strictly upper triangular sum, since every pair counts once
        llhood -= jnp.sum(log_t) - jnp.sum(jnp.tril(log_t))

        return -llhood
