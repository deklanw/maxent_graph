import warnings
import itertools

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.optimize
import igraph as ig

from numba import jit

from . import poibin


SolutionBundle = namedtuple(
    "SolutionBundle", ["sol", "error", "unique_row_info", "unique_col_info"]
)


def initial_guess(option, row_degrees, col_degrees, dseq, multiplicity):
    if option == 1:
        x0_rows = row_degrees / np.sqrt(np.sum(row_degrees) + 1)
        x0_cols = col_degrees / np.sqrt(np.sum(col_degrees) + 1)
    elif option == 2:
        x0_rows = row_degrees / np.sqrt(np.sum(row_degrees) + np.sum(col_degrees))
        x0_cols = col_degrees / np.sqrt(np.sum(row_degrees) + np.sum(col_degrees))
    elif option == 3:
        x0_rows = row_degrees / np.sqrt(np.sum(row_degrees) * np.sum(col_degrees))
        x0_cols = col_degrees / np.sqrt(np.sum(row_degrees) * np.sum(col_degrees))
    elif option == 4:
        # ~10% chance
        x0_rows = np.repeat(1 / 20, len(row_degrees))
        x0_cols = np.repeat(1 / 20, len(col_degrees))
    else:
        raise ValueError(f"Invalid option value. Choose from 1-4.")

    x0 = np.concatenate([x0_rows, x0_cols])

    return x0


# much faster with jit
@jit(nopython=True)
def equations(xx, dseq, multiplicity, n_row_degrees):
    eq = -dseq
    for i in range(n_row_degrees):
        for j in range(n_row_degrees, len(dseq)):
            x_ij = xx[i] * xx[j]
            v = x_ij / (1.0 + x_ij)
            eq[i] += multiplicity[j] * v
            eq[j] += multiplicity[i] * v

    return eq


def solve_equations(B, method="lm", initial_guess_option=4):
    num_rows, num_cols = B.shape

    # since B is a (sparse) matrix, the sums will be matrices
    # the sums above will give ints, so we need to convert to floats
    row_sums = np.asarray(np.sum(B, axis=1).astype(np.float64)).flatten()
    col_sums = np.asarray(np.sum(B, axis=0).astype(np.float64)).flatten()

    assert not np.any(np.where(row_sums == 0))
    assert not np.any(np.where(row_sums == len(col_sums)))
    assert not np.any(np.where(col_sums == 0))
    assert not np.any(np.where(col_sums == len(row_sums)))

    assert len(row_sums) == num_rows
    assert len(col_sums) == num_cols

    # since in empirical networks there will be many nodes with the same degree
    # we can count them and use that information to speed up solving the equations.
    # the bicm doesn't distinguish between nodes with the same degree.
    # we also want to keep track of which nodes have which degree (for later). for that we just use pd's groupby
    unique_row_info = np.unique(
        row_sums, return_index=False, return_inverse=True, return_counts=True
    )
    row_df = pd.DataFrame(row_sums)
    row_groups = row_df.groupby(by=0).groups
    unique_row_info += (row_groups,)
    row_degrees, rows_inverse, rows_multiplicity, _row_groups = unique_row_info

    unique_col_info = np.unique(
        col_sums, return_index=False, return_inverse=True, return_counts=True
    )
    col_df = pd.DataFrame(col_sums)
    col_groups = col_df.groupby(by=0).groups
    unique_col_info += (col_groups,)
    col_degrees, cols_inverse, cols_multiplicity, _col_groups = unique_col_info

    dseq = np.concatenate([row_degrees, col_degrees])
    multiplicity = np.concatenate([rows_multiplicity, cols_multiplicity])

    x0 = initial_guess(
        initial_guess_option, row_degrees, col_degrees, dseq, multiplicity
    )

    n_row_degrees = len(row_degrees)

    sol = scipy.optimize.root(
        fun=equations, args=(dseq, multiplicity, n_row_degrees), x0=x0, method=method,
    )

    error = np.linalg.norm(equations(sol.x, dseq, multiplicity, n_row_degrees), ord=2)

    all_positive = np.all(sol.x > 0)

    # make sure everything makes sense
    if not (all_positive and sol.success and error < 0.001):
        raise RuntimeError(
            "Couldn't find solution with the chosen method and initial guess."
        )

    return SolutionBundle(sol, error, unique_row_info, unique_col_info)


def solve_equations_kitchen_sink(B):
    """
    Tries every initial guess method and every equation solving method (which is known to work well in these types of problems) to produce the solution with minimum error.
    Slow. Only recommended if you want the absolute best accuracy.
    """
    methods_to_try = ["hybr", "krylov", "broyden2", "anderson", "lm", "df-sane"]
    initial_guess_options = [1, 2, 3, 4]
    ComboInfo = namedtuple("ComboInfo", ["method", "initial_guess_option", "error"])
    best_combo_info = None

    # some of these will give warnings. we don't care for this kitchen sink
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in methods_to_try:
            for initial_guess_option in initial_guess_options:
                try:
                    sol_bundle = solve_equations(
                        B, method=method, initial_guess_option=initial_guess_option
                    )
                    combo_info = ComboInfo(
                        method, initial_guess_option, sol_bundle.error
                    )
                    if (
                        best_combo_info is None
                        or sol_bundle.error < best_combo_info.error
                    ):
                        best_combo_info = combo_info
                except:
                    pass

    if best_combo_info is None:
        raise RuntimeError(
            "No combination of initial guess option or equation-solving method worked. Yikes!"
        )

    print(
        f"The best initial_guess_option was {best_combo_info.initial_guess_option} and the best equation-solving method was {best_combo_info.method}. The error is {best_combo_info.error}"
    )

    return sol_bundle


def construct_projection(B, solution_bundle, p_val=0.05):
    observed_lambda_motif_counts = B @ B.T

    (
        row_degrees,
        _rows_inverse,
        _rows_multiplicity,
        row_groups,
    ) = solution_bundle.unique_row_info

    (
        _col_degrees,
        cols_inverse,
        _cols_multiplicity,
        _col_groups,
    ) = solution_bundle.unique_col_info

    x = solution_bundle.sol.x

    col_fitnesses = x[len(row_degrees) :][cols_inverse]

    new_A = scipy.sparse.lil_matrix((B.shape[0], B.shape[0]))

    for (i, j) in itertools.combinations(range(len(row_degrees)), 2):
        fitness_i = x[i]
        fitness_j = x[j]

        v_i = fitness_i * col_fitnesses
        v_j = fitness_j * col_fitnesses

        ps_i = v_i / (1 + v_i)
        ps_j = v_j / (1 + v_j)

        expected_lambda_motifs = ps_i * ps_j
        pmf = poibin.dc_fft_pb(expected_lambda_motifs)

        degree_i = row_degrees[i]
        degree_j = row_degrees[j]

        for orig_i in row_groups[degree_i]:
            for orig_j in row_groups[degree_j]:
                observed = observed_lambda_motif_counts[orig_i, orig_j]
                p = np.sum(pmf[observed:])
                if p < p_val:
                    new_A[orig_i, orig_j] = -np.log(p)

    return new_A
