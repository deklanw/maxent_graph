import pytest
import numpy as np

import jax

from maxent_graph.bicm import BICM
from maxent_graph.decm import DECM
from maxent_graph.bwcm import BWCM
from maxent_graph.ecm import ECM
from maxent_graph.util import nx_get_A, nx_get_B, jax_class_jit

models = [
    BICM(nx_get_B("data/my_senate_116_bipartite.graphml")),
    BICM(nx_get_B("data/opsahl-southernwomen_bipartite.graphml")),
    DECM(nx_get_A("data/residence_hall.graphml", weight_key="weight")),
    DECM(nx_get_A("data/macaques.graphml", weight_key="weight")),
    BWCM(
        nx_get_B(
            "data/plant_pol_kato.graphml",
            weight_key="count",
            bipartite_key="pollinator",
        )
    ),
    BWCM(
        nx_get_B(
            "data/plant_pol_vazquez_All_sites_pooled.graphml",
            weight_key="count",
            bipartite_key="pollinator",
        )
    ),
    ECM(nx_get_A("data/kangaroo.graphml", weight_key="weight")),
    ECM(nx_get_A("data/train_terrorists.graphml", weight_key="weight")),
]


@pytest.mark.parametrize("model", models)
def test_model(model):
    initial_guess = model.get_initial_guess()
    solution = model.solve(initial_guess)

    assert solution is not None
    assert max(solution.relative_error) < 0.1

    nll_loops = model.neg_log_likelihood_loops(initial_guess)

    nll = model.neg_log_likelihood(initial_guess)

    assert np.allclose(nll_loops, nll)

    ens_loops = model.expected_node_sequence_loops(initial_guess)

    ens = model.expected_node_sequence(initial_guess)

    assert np.allclose(ens_loops, ens)

    # test nll gradient
    nll_grad = model.neg_log_likelihood_grad(initial_guess)

    nll_jit = jax_class_jit(model.neg_log_likelihood)

    nll_auto_grad = jax.grad(nll_jit)(initial_guess)

    assert np.allclose(nll_grad, nll_auto_grad)

    # test expected node sequence jacobian

    ens_jac = model.expected_node_sequence_jac(initial_guess)

    ens_jit = jax_class_jit(model.expected_node_sequence)

    ens_auto_jac = jax.jacfwd(ens_jit)(initial_guess)

    assert np.allclose(ens_jac, ens_auto_jac)

