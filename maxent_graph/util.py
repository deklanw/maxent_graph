"""
Just some utilities.
"""

import jax
import numpy as np
import jax.numpy as jnp
import networkx as nx
from networkx.algorithms import bipartite
from tabulate import tabulate
from functools import partial
from jax import jvp, grad


EPS = np.finfo(float).eps


def print_percentiles(v):
    """
    Prints the min, 25th percentile, median, 75th percentile, and max.
    """

    table = [
        ["Min", np.min(v)],
        ["25th", np.percentile(v, 25)],
        ["Median", np.median(v)],
        ["75th", np.percentile(v, 75)],
        ["Max", np.max(v)],
    ]
    table_str = tabulate(table, headers=["Percentile", "Relative error"])
    print(table_str)


def wrap_with_array(f):
    # asarray gives fortran contiguous error with L-BFGS-B
    # have to use np.array
    # https://github.com/google/jax/issues/1510#issuecomment-542419523
    return lambda v: np.array(f(v))


def jax_class_jit(f):
    """
    Lets you JIT a class method with JAX.
    """
    return partial(jax.jit, static_argnums=(0,))(f)


@jax.jit
def zero_diagonal(x):
    """
    Note that you could just subtract out the diagonal.
    https://github.com/google/jax/issues/2680
    """
    return jax.ops.index_update(x, jnp.diag_indices(x.shape[0]), 0)


def nx_get_A(fn, weight_key=None):
    g = nx.read_graphml(fn)
    return nx.adjacency_matrix(g, nodelist=None, weight=weight_key)


def nx_get_B(fn, weight_key=None, bipartite_key=None):
    g = nx.read_graphml(fn)

    assert nx.is_bipartite(g)

    if bipartite_key is None:
        bottom_nodes, top_nodes = bipartite.sets(g)
    else:
        bottom_nodes = [
            node
            for node, data in g.nodes(data=True)
            if data.get(bipartite_key) == False
        ]
        top_nodes = None

    B = bipartite.biadjacency_matrix(
        g, bottom_nodes, top_nodes, dtype=None, weight=weight_key, format="csr"
    )
    return B


def hvp(f):
    """
    Hessian-vector-product

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode
    """
    return lambda x, v: jvp(grad(f), (x,), (v,))[1]
