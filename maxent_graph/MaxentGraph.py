"""
Contains ABC for Maximum Entropy graph null model.
"""

import warnings
import time
from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np
import scipy.optimize
from jax import jit, jacfwd, jacrev, grad


from .util import wrap_with_array, print_percentiles, jax_class_jit, hvp

Solution = namedtuple(
    "Solution", ["x", "nll", "residual_error_norm", "relative_error", "total_time"]
)


class MaxentGraph(ABC):
    """
    ABC for Maximum Entropy graph null model.
    """

    @abstractmethod
    def bounds(self):
        """
        Returns the bounds on the parameters vector.
        """

    def clip(self, v):
        """
        Clips the parameters vector according to bounds.
        """
        (lower, upper), _bounds_object = self.bounds()
        return np.clip(v, lower, upper)

    @abstractmethod
    def order_node_sequence(self):
        """
        Concatenates node constraint sequence in a canonical order.
        """

    @abstractmethod
    def get_initial_guess(self, option):
        """
        Gets initial guess.
        """

    @abstractmethod
    def expected_node_sequence(self, v):
        """
        Computes the expected node constraint using matrices.
        """

    @abstractmethod
    def expected_node_sequence_jac(self, v):
        """
        Computes the Jacobian of the expected node constraint using matrices.
        Since the actual node sequence is constant, we can ignore it for the Jacobian.
        """

    @abstractmethod
    def expected_node_sequence_loops(self, v):
        """
        Computes the expected node constraint using loops.
        """

    @jax_class_jit
    def node_sequence_residuals(self, v):
        """
        Computes the residuals of the expected node constraint sequence minus the actual sequence.
        """
        return self.expected_node_sequence(v) - self.order_node_sequence()

    @abstractmethod
    def neg_log_likelihood_loops(self, v):
        """
        Computes the negative log-likelihood using loops.
        """

    @abstractmethod
    def neg_log_likelihood(self, v):
        """
        Computes the negative log-likelihood using matrix operations.
        """

    @abstractmethod
    def neg_log_likelihood_grad(self, v):
        """
        Computes the gradient of the negative log-likelihood using matrix operations.
        """

    def compute_relative_error(self, expected):
        """
        Computes relative error for solution for every element of the sequence.
        """
        actual = self.order_node_sequence()

        # okay not actually relative error but close enough
        return np.abs(expected - actual) / (1 + np.abs(actual))

    def solve(self, x0, method="TNC", verbose=False):
        """
        Solves for the parameters of the null model using either bounded minimization of the
        negative log-likelihood or bounded least-squares minimization of the equation residuals.
        """

        args = {}

        # for some reason scipy prefers hess over hessp if the former is passed
        # but since the latter is more efficient, only pass hess when necessary
        if method in ["trust-exact", "dogleg"]:
            hess = jit(jacfwd(jacrev(self.neg_log_likelihood)))
            args["hess"] = hess

        if method in ["trf", "dogbox", "lm"]:
            f = self.node_sequence_residuals
            jac = jit(jacrev(self.expected_node_sequence))
            args["jac"] = jac

            solver = scipy.optimize.least_squares
        elif method in [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]:
            f = self.neg_log_likelihood
            jac = jit(grad(self.neg_log_likelihood))

            # lbfgsb is fussy. wont accept jax's devicearray
            # there may be others, though
            if method in ["L-BFGS-B"]:
                jac = wrap_with_array(jac)

            hessp = jit(hvp(self.neg_log_likelihood))
            args["hessp"] = hessp
            args["jac"] = jac

            solver = scipy.optimize.minimize
        else:
            assert False

        start = time.time()
        sol = solver(f, x0=x0, method=method, **args)
        end = time.time()

        total_time = end - start

        eq_r = self.node_sequence_residuals(sol.x)
        expected = self.expected_node_sequence(sol.x)
        residual_error_norm = np.linalg.norm(eq_r, ord=2)
        relative_error = self.compute_relative_error(expected)
        nll = self.neg_log_likelihood(sol.x)

        if not sol.success:
            if np.max(relative_error) < 0.5:
                warnings.warn(
                    "Didn't succeed according to algorithm, but max relative error is low.",
                    RuntimeWarning,
                )
            else:
                raise RuntimeError(
                    f"Didn't succeed in minimization. Message: {sol.message}"
                )

        if verbose:
            print(f"Took {total_time} seconds")
            print("Relative error for expected degree/strength sequence: ")
            print()
            print_percentiles(relative_error)

            print(f"\nResidual error: {residual_error_norm}")

        return Solution(
            x=sol.x,
            nll=float(nll),
            residual_error_norm=residual_error_norm,
            relative_error=relative_error,
            total_time=total_time,
        )
