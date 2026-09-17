"""
Family C: hurdle Poisson configuration models.

The count-aware counterpart of the BiECM. Presence is degree constrained and
the positive weight is strength constrained, but the positive part is Poisson
rather than geometric, which is a much lighter tail.

The BiECM's positive part is a *shifted* geometric -- ``w - 1`` is geometric,
which is why its p-values carry ``y**(w - 1)`` -- so the faithful count
analogue is a shifted Poisson, ``w - 1 ~ Poisson(u_i v_a)``, and that is the
default here. Zero truncation is available as an option, and is what zero
inflation implies, but it buys a transcendental link for nothing: see
``ShiftedPoisson`` for why the shifted form fits so much more easily.

The likelihood factorises exactly -- nothing in the presence part appears in
the positive part or vice versa -- so the two halves are fitted independently,
and the presence half simply *is* the binary configuration model. These
models reuse that existing fit rather than re-deriving it.
"""

import warnings

import numpy as np
import scipy.sparse
import scipy.special

from ..bicm import BICM
from ..dbcm import DBCM
from ..ubcm import UBCM
from .base import DyadModel, solve_product_form
from .dists import Hurdle, ShiftedPoisson, ZeroTruncatedPoisson
from .layout import DyadLayout, dense

KINDS = ("hurdle", "zip")
STRENGTHS = ("joint", "conditional")
POSITIVE_PARTS = {"shifted": ShiftedPoisson, "ztp": ZeroTruncatedPoisson}

PRESENCE_MODELS = {"bipartite": BICM, "undirected": UBCM, "directed": DBCM}


def solve_bernoulli(
    layout, row_targets, col_targets, mask, tol, max_iter, damping, start=None
):
    """
    Product-form Bernoulli fit: ``p_ij = a_i b_j / (1 + a_i b_j)`` with given
    expected row and column totals.

    This is the binary configuration model's own fixed point, written so that
    the targets can be fractional -- which is what a zero-inflation E-step
    hands it.
    """
    support = mask.astype(np.float64)
    total = max(row_targets.sum(), 1e-12)
    if start is None:
        a0 = row_targets / np.sqrt(total)
        b0 = col_targets / np.sqrt(total)
    else:
        a0, b0 = start

    def odds_weights(a, b):
        return support / (1.0 + np.outer(a, b))

    def update_row(a, b):
        return _safe_ratio(row_targets, odds_weights(a, b) @ b)

    def update_col(a, b):
        return _safe_ratio(col_targets, odds_weights(a, b).T @ a)

    a, b, info = solve_product_form(
        update_row,
        update_col,
        a0,
        b0,
        tied=layout.tied,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
        name="presence",
    )
    ab = np.outer(a, b)
    return np.where(mask, ab / (1 + ab), 0.0), (a, b), info


def _safe_ratio(numerator, denominator):
    return np.where(
        denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0
    )


class HurdlePoissonCM(DyadModel):
    """
    Hurdle Poisson configuration model over an arbitrary dyad set.

    ``a_ij ~ Bernoulli(x_i y_j / (1 + x_i y_j))`` and, given presence, a
    Poisson positive part with rate ``u_i v_j``. The presence half is the
    binary configuration model and is fitted by the existing BiCM / UBCM /
    DBCM code, so expected degrees match observed ones. The positive half is
    fitted so that expected strengths match too -- by default over the whole
    model, as for every other model in the library.

    Parameters
    ----------
    kind : {"hurdle", "zip"}
        The default hurdle model fits the two halves separately. ``"zip"``
        fits a zero-inflated Poisson instead: same dyad distribution family --
        a ZIP with inflation ``1 - pi`` and rate ``lam`` is a hurdle with
        ``p = pi (1 - exp(-lam))`` -- but a different parameterisation, whose
        likelihood no longer factorises and which is fitted by EM.
    positive : {"shifted", "ztp"}
        The positive part: ``w - 1 ~ Poisson(lam)``, matching the BiECM's
        shifted geometric, or a zero-truncated Poisson. Defaults to
        ``"shifted"``, except that ``kind="zip"`` implies ``"ztp"``, since
        that is what conditioning a zero-inflated Poisson on being positive
        gives.
    strengths : {"joint", "conditional"}
        What the positive half's strength constraint sums over. ``"joint"``,
        the default, matches the model's expected strengths,
        ``sum_j p_ij E[w_ij | present] = s_i`` over every dyad, which gives
        every dyad a rate and so lets an absent dyad carry any positive
        weight. ``"conditional"`` matches them over the observed edges only,
        which is what maximising the factorised likelihood gives, but leaves
        the model's own expected strengths short of the observed ones and pins
        the rate of every absent dyad at zero, making a weight of two or more
        impossible there. Not used by ``kind="zip"``, which is fitted by EM.

    Notes
    -----
    The binary configuration models ignore self-loops, so these do too.

    Under ``strengths="conditional"`` the positive half can sit on a boundary
    the product form cannot reach: a set of rows whose neighbours' excess
    weight is owed almost entirely to that set drives some rates towards zero
    or infinity. The fit then stops on the constraint residual rather than on
    the parameters, so ``fit_info["strength_error"]`` is the number to check
    -- small relative to the strengths means the fitted distribution has
    settled, whatever the rates are still doing. The joint constraint spreads
    over every dyad and does not have that problem.
    """

    def __init__(self, W, layout, kind="hurdle", positive=None, strengths=None):
        if layout.self_loops:
            raise NotImplementedError(
                "the binary presence models ignore self-loops, so the hurdle "
                "models do too; pass self_loops=False"
            )
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

        if positive is None:
            positive = "ztp" if kind == "zip" else "shifted"
        elif positive not in POSITIVE_PARTS:
            raise ValueError(f"positive must be one of {tuple(POSITIVE_PARTS)}")
        elif kind == "zip" and positive != "ztp":
            raise ValueError(
                "conditioning a zero-inflated Poisson on being positive gives a "
                "zero-truncated Poisson, so kind='zip' needs positive='ztp'"
            )

        if strengths is None:
            strengths = "joint" if kind == "hurdle" else None
        elif kind == "zip":
            raise ValueError(
                "kind='zip' is fitted by EM and has no strength constraint to choose"
            )
        elif strengths not in STRENGTHS:
            raise ValueError(f"strengths must be one of {STRENGTHS}")

        super().__init__(W, layout)
        self.kind = kind
        self.positive = positive
        self.strengths = strengths
        self.presence = None
        self.rate = None
        self.pi = None
        self.presence_model = None
        self.presence_solution = None

    def positive_dist(self, rate):
        """
        The positive part at the given rates.
        """
        return POSITIVE_PARTS[self.positive](rate)

    # ------------------------------------------------------------------
    # presence
    # ------------------------------------------------------------------

    def _fit_presence(self, **solve_kwargs):
        """
        Fits and expands the binary configuration model on the observed
        adjacency.
        """
        model_class = PRESENCE_MODELS[self.layout.kind]
        A = scipy.sparse.csr_matrix(self.adjacency.astype(np.float64))
        model = model_class(A)
        solution = model.solve(model.get_initial_guess(), **solve_kwargs)

        self.presence_model = model
        self.presence_solution = solution

        z = np.asarray(model.transform_parameters(solution.x))
        if isinstance(model, BICM):
            x = z[: model.n_row_degrees][model.row_inverse]
            y = z[model.n_row_degrees :][model.col_inverse]
        elif isinstance(model, UBCM):
            x = y = z
        else:
            x, y = z[: model.num_nodes], z[model.num_nodes :]

        xy = np.outer(x, y)
        return np.where(self.layout.support, xy / (1 + xy), 0.0)

    # ------------------------------------------------------------------
    # positive part
    # ------------------------------------------------------------------

    def _excess_ratio(self, rate):
        """
        ``(E[w | present] - 1) / rate``, the factor the rate solver weights
        each dyad by.

        One for the shifted part, whose excess over one is the rate itself.
        For the truncated part it is ``1 / (1 - exp(-lam)) - 1 / lam``, which
        cancels badly as the rate goes to zero, where it tends to one half.
        """
        if self.positive == "shifted":
            return np.ones_like(rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            direct = 1.0 / -np.expm1(-rate) - 1.0 / rate
        series = 0.5 - rate / 12.0 + rate**3 / 720.0
        return np.where(rate < 1e-4, series, direct)

    def _fit_rate(self, weights, tol, max_iter, damping, strength_tol=1e-6):
        """
        Solves the positive part's strength constraints.

        ``weights`` says how much each dyad counts towards a node's expected
        strength: the presence probabilities, for strengths matched over the
        whole model, or the observed adjacency, for strengths matched over the
        observed edges only. Either way the constraint is
        ``sum_j weights_ij * E[w_ij | present] = s_i``.

        It is solved for the *excess* over one, ``sum_j weights_ij *
        (E[w_ij | present] - 1) = s_i - sum_j weights_ij``. A node whose
        weights are all one has excess zero, so its rate is exactly zero after
        a single update -- no peeling, and no creeping towards the boundary
        through a transcendental link.
        """
        layout = self.layout
        weights = np.where(layout.support, weights, 0.0)
        scale = layout.dyad_scale

        unit_rows = np.isclose(self.row_strengths, self.row_degrees)
        unit_cols = np.isclose(self.col_strengths, self.col_degrees)
        row_target = np.where(
            unit_rows,
            0.0,
            np.maximum(self.row_strengths - layout.row_totals(weights), 0.0),
        )
        col_target = np.where(
            unit_cols,
            0.0,
            np.maximum(self.col_strengths - layout.col_totals(weights), 0.0),
        )

        excess = row_target.sum()
        if excess <= 0:
            return np.zeros(layout.shape), {
                "iterations": 0,
                "delta": 0.0,
                "rate_zero_dyads": int((weights > 0).sum()),
            }

        # with presence probabilities as weights the two sides' targets agree
        # only to the precision of the presence fit; make them agree exactly,
        # or the alternating updates chase a mismatch forever
        if not layout.tied and col_target.sum() > 0:
            col_target = col_target * (excess / col_target.sum())

        def weighted(u, v):
            rate = scale * np.outer(u, v)
            return weights * self._excess_ratio(rate)

        def update_row(u, v):
            return _safe_ratio(row_target, weighted(u, v) @ v)

        def update_col(u, v):
            return _safe_ratio(col_target, weighted(u, v).T @ u)

        threshold = strength_tol * max(1.0, self.total_weight)

        def constraints_met(u, v):
            achieved = weighted(u, v) * np.outer(u, v)
            return (
                max(
                    np.max(np.abs(layout.row_totals(achieved) - row_target)),
                    np.max(np.abs(layout.col_totals(achieved) - col_target)),
                )
                < threshold
            )

        root = np.sqrt(excess)
        u, v, info = solve_product_form(
            update_row,
            update_col,
            row_target / root,
            col_target / root,
            tied=layout.tied,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            converged=constraints_met,
            check_after=max(500, max_iter // 10),
            name=type(self).__name__,
        )
        rate = np.where(layout.support, scale * np.outer(u, v), 0.0)
        zeros = int(((weights > 0) & (rate == 0)).sum())
        if zeros:
            info["rate_zero_dyads"] = zeros
        return rate, info

    # ------------------------------------------------------------------
    # zero inflation
    # ------------------------------------------------------------------

    def _zip_loglik(self, pi, rate):
        canonical = self.layout.canonical
        present = self.adjacency & canonical
        absent = canonical & ~self.adjacency

        with np.errstate(divide="ignore", invalid="ignore"):
            zero_term = np.log(1 - pi + pi * np.exp(-rate))
            positive = (
                np.log(pi)
                - rate
                + self.weights * np.log(np.where(rate > 0, rate, 1.0))
                - scipy.special.gammaln(self.weights + 1)
            )
        return float(
            np.where(absent, zero_term, 0.0).sum()
            + np.where(present, positive, 0.0).sum()
        )

    def _fit_zip(self, presence, rate, tol, max_iter, damping, max_em, em_tol):
        """
        EM for the zero-inflated parameterisation.

        The E-step splits the observed zeros between the inflation component
        and a Poisson that happened to come up zero; both M-steps are then the
        same product-form solves used elsewhere, with soft counts.

        The M-steps are deliberately run to a modest iteration budget. EM only
        needs each step to improve its own objective, not to solve it, and the
        inflation step in particular drifts towards probability one wherever
        the Poisson can explain the zeros on its own, which is slow to reach
        and not worth reaching.
        """
        inner_iter = min(max_iter, 100)
        scale = self.layout.dyad_scale
        with np.errstate(divide="ignore", invalid="ignore"):
            pi = np.where(
                self.layout.support,
                np.clip(presence / -np.expm1(-rate), 1e-12, 1 - 1e-12),
                0.0,
            )
        pi = np.where(np.isfinite(pi), pi, 0.5)
        support = self.layout.support

        total = max(self.row_strengths.sum(), 1e-12)
        rate_start = (
            self.row_strengths / np.sqrt(total),
            self.col_strengths / np.sqrt(total),
        )
        presence_start = None

        previous = -np.inf
        info = {}
        for iteration in range(1, max_em + 1):
            weight = pi * np.exp(-rate)
            responsibility = np.where(self.adjacency, 1.0, weight / (1 - pi + weight))
            responsibility = np.where(support, responsibility, 0.0)

            def update_row(u, v, r=responsibility):
                return _safe_ratio(self.row_strengths, r @ v)

            def update_col(u, v, r=responsibility):
                return _safe_ratio(self.col_strengths, r.T @ u)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # warm starting is what keeps a partial M-step from undoing
                # the previous one, and so keeps EM climbing
                u, v, info = solve_product_form(
                    update_row,
                    update_col,
                    *rate_start,
                    tied=self.layout.tied,
                    tol=tol,
                    max_iter=inner_iter,
                    damping=damping,
                    name=type(self).__name__,
                )
                rate_start = (u, v)
                rate = np.where(support, scale * np.outer(u, v), 0.0)

                pi, presence_start, _ = solve_bernoulli(
                    self.layout,
                    responsibility.sum(axis=1),
                    responsibility.sum(axis=0),
                    support,
                    tol,
                    inner_iter,
                    damping,
                    start=presence_start,
                )
            pi = np.clip(pi, 1e-12, 1 - 1e-12)

            value = self._zip_loglik(pi, rate)
            if abs(value - previous) < em_tol * max(1.0, abs(value)):
                break
            previous = value

        info["em_iterations"] = iteration
        info["em_converged"] = iteration < max_em
        info["zip_loglik"] = value
        return pi, rate, info

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        presence=None,
        tol=1e-12,
        max_iter=5000,
        damping=None,
        max_em=200,
        em_tol=1e-10,
        strength_tol=1e-6,
        **solve_kwargs,
    ):
        """
        Fits the presence and positive halves.

        ``presence`` optionally supplies an already-fitted matrix of presence
        probabilities instead of solving the binary model again.
        ``strength_tol``, relative to the total weight, is the constraint
        residual the positive half settles for when the maximum is near a
        boundary the parameters can only creep towards. Remaining keyword arguments go to the binary
        model's solver.
        """
        if damping is None:
            damping = 0.5 if self.layout.tied else 1.0

        if presence is None:
            presence = self._fit_presence(**solve_kwargs)
        else:
            presence = np.where(self.layout.support, dense(presence), 0.0)
            if presence.shape != self.layout.shape:
                raise ValueError("presence must have the layout's shape")

        weights = (
            presence if self.strengths == "joint" else self.adjacency.astype(np.float64)
        )
        rate, info = self._fit_rate(weights, tol, max_iter, damping, strength_tol)
        if self.strengths == "conditional":
            # the conditional fit only identifies rates on the observed edges;
            # the product form off them follows directions the fit leaves free,
            # and on a sparse network runs to rates hundreds of times any
            # observed weight, so they are not extrapolated
            rate = np.where(self.adjacency, rate, 0.0)

        if self.kind == "zip":
            self.pi, rate, info = self._fit_zip(
                presence, rate, tol, max_iter, damping, max_em, em_tol
            )
            presence = self.pi * -np.expm1(-rate)

        self.presence = presence
        self.rate = rate
        self._M = self._dyad_mean() / self.layout.dyad_scale

        self.fit_info = {
            "kind": self.kind,
            "positive": self.positive,
            "strengths": self.strengths,
            "iterations": info.get("iterations"),
            "degree_error": self.degree_error(),
            "strength_error": self.strength_error(),
        }
        for key in (
            "rate_zero_dyads",
            "em_iterations",
            "em_converged",
            "zip_loglik",
            "fallback",
        ):
            if key in info:
                self.fit_info[key] = info[key]
        return self

    def _dyad_mean(self):
        return np.where(
            self.layout.support,
            self.presence * self.positive_dist(self.rate).mean(),
            0.0,
        )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def expected_row_degrees(self):
        return self.layout.row_totals(self.presence)

    def expected_col_degrees(self):
        return self.layout.col_totals(self.presence)

    def degree_error(self):
        self._require_fit()
        return max(
            np.max(np.abs(self.expected_row_degrees() - self.row_degrees)),
            np.max(np.abs(self.expected_col_degrees() - self.col_degrees)),
        )

    def expected_positive_row_strengths(self):
        """
        Expected strength restricted to the observed edges, which is what the
        positive half constrains.
        """
        self._require_fit()
        conditional = np.where(
            self.adjacency, self.positive_dist(self.rate).mean(), 0.0
        )
        return self.layout.row_totals(conditional / self.layout.dyad_scale)

    def expected_positive_col_strengths(self):
        self._require_fit()
        conditional = np.where(
            self.adjacency, self.positive_dist(self.rate).mean(), 0.0
        )
        return self.layout.col_totals(conditional / self.layout.dyad_scale)

    def positive_strength_error(self):
        """
        Error in the strengths over the observed edges, which
        ``strengths="conditional"`` constrains.
        """
        return max(
            np.max(np.abs(self.expected_positive_row_strengths() - self.row_strengths)),
            np.max(np.abs(self.expected_positive_col_strengths() - self.col_strengths)),
        )

    def joint_strength_error(self):
        """
        Error in the model's own expected strengths, which
        ``strengths="joint"`` constrains.

        It cannot fall below the presence fit's degree error: a node carrying
        only unit weights has rate zero, so its expected strength is its
        expected degree.
        """
        self._require_fit()
        return DyadModel.constraint_error(self)

    def strength_error(self):
        """
        Error in whichever strength constraint the fit imposed.
        """
        if self.strengths == "conditional":
            return self.positive_strength_error()
        return self.joint_strength_error()

    def constraint_error(self):
        """
        The two constraints the hurdle fit imposes: expected degrees, and
        expected strengths as chosen by ``strengths``.

        For ``kind="zip"`` neither is a constraint of the fit -- the
        zero-inflated likelihood does not factorise -- and the number measures
        how far the fit drifted from them rather than its convergence.
        """
        return max(self.degree_error(), self.strength_error())

    # ------------------------------------------------------------------
    # distribution
    # ------------------------------------------------------------------

    def _dist(self, idx):
        self._require_fit()
        return Hurdle(
            self._take(self.presence, idx),
            self.positive_dist(self._take(self.rate, idx)),
        )


class BIHPCM(HurdlePoissonCM):
    """
    Bipartite hurdle Poisson configuration model. Presence is the BiCM.
    """

    def __init__(self, B, kind="hurdle", positive=None, strengths=None):
        B = dense(B)
        super().__init__(
            B,
            DyadLayout.bipartite(*B.shape),
            kind=kind,
            positive=positive,
            strengths=strengths,
        )


class UHPCM(HurdlePoissonCM):
    """
    Undirected hurdle Poisson configuration model. Presence is the UBCM.
    """

    def __init__(self, A, kind="hurdle", positive=None, strengths=None):
        A = dense(A)
        super().__init__(
            A,
            DyadLayout.undirected(A.shape[0]),
            kind=kind,
            positive=positive,
            strengths=strengths,
        )


class DHPCM(HurdlePoissonCM):
    """
    Directed hurdle Poisson configuration model. Presence is the DBCM.
    """

    def __init__(self, A, kind="hurdle", positive=None, strengths=None):
        A = dense(A)
        super().__init__(
            A,
            DyadLayout.directed(A.shape[0]),
            kind=kind,
            positive=positive,
            strengths=strengths,
        )
