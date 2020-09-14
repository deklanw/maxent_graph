import igraph as ig
import scipy.sparse

from maxent_graph.bicm import solve_equations, solve_equations_kitchen_sink


def test_fitness_equations():
    gns = ["senate_114_bipartite", "movielens100k_bipartite", "anime_tropes_bipartite"]

    for gn in gns:
        g = ig.read(f"data/{gn}.graphml", format="graphml")
        assert g.is_bipartite()
        B = scipy.sparse.csr_matrix(g.get_incidence()[0])

        fitnesses = solve_equations(B)

        assert fitnesses is not None


def test_kitchen_sink():
    # test with a tiny graph
    gn = "brunson_corporate-leadership_bipartite"
    g = ig.read(f"data/{gn}.graphml", format="graphml")
    assert g.is_bipartite()
    B = scipy.sparse.csr_matrix(g.get_incidence()[0])

    sol_bundle = solve_equations_kitchen_sink(B)

    assert sol_bundle.error < 0.001
