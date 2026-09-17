"""
Family B: negative binomial configuration models.

Same product-form mean as family A with one extra parameter: a dispersion r
giving ``var = mu + mu^2 / r``. The family nests both of its neighbours --
``r = 1`` is geometric and ``r -> inf`` is Poisson -- so it is the natural
thing to reach for when a network's weights are more variable than Poisson,
which empirically they nearly always are.

Note that this parameterises the *mean* as a product, where the BWCM
parameterises the geometric ratio as one. The two agree on the constraints
they impose and on the variance function at ``r = 1``, but they are different
link functions and so different fits.
"""

import warnings

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import gammaln

from .base import DyadModel, solve_product_form
from .layout import DyadLayout, dense

DISPERSIONS = ("global", "row")


class NegativeBinomialCM(DyadModel):
    """
    Negative binomial configuration model over an arbitrary dyad set.

    ``w_ij ~ NB(mean = x_i y_j, r)``. Fitting alternates between the effect
    vectors, which for fixed r solve weighted score equations, and the
    dispersion, which for fixed means is a bounded one-dimensional search on
    ``log r``.

    Parameters
    ----------
    dispersion : {"global", "row"}
        One r for the whole network, or one per row node. Per-row dispersion
        needs the two sides of a dyad to be distinguishable, so it is not
        available for undirected layouts.
    r : float, optional
        Fixes the dispersion instead of estimating it. ``r = 1`` gives a
        geometric model and a large r reproduces the Poisson one.

    Notes
    -----
    ``fit`` has two readings of "configuration model", and defaults to the one
    that earns the name.

    ``constrain_strengths=True``, the default, holds the means at the family A
    solution ``s_i s_j / W`` and estimates only the dispersion around them.
    This is the Gamma-Poisson mixture over the Poisson configuration model:
    the constraints are reproduced exactly, the Poisson model is an exact
    special case at every r rather than only in the limit, and r is an
    estimate of how far dyad intensities vary beyond Poisson noise.
    ``fit_info`` carries the log-likelihood at the fitted r and at ``r ->
    inf`` so that evidence can be read off directly.

    ``constrain_strengths=False`` is instead the unrestricted maximum
    likelihood fit, in which the means are free. Its score equations are
    weighted by ``1 / (1 + mu / r)``, downweighting the dyads whose means are
    large relative to r, so it does *not* reproduce the observed strengths
    except in the Poisson limit. For that fit ``fit_info["score_norm"]`` is
    the convergence diagnostic that matters, not ``constraint_error()``.
    """

    def __init__(self, W, layout, dispersion="global", r=None):
        super().__init__(W, layout)

        if dispersion not in DISPERSIONS:
            raise ValueError(f"dispersion must be one of {DISPERSIONS}")
        if dispersion == "row" and layout.tied:
            raise ValueError(
                "per-row dispersion needs distinguishable sides, which an "
                "undirected layout does not have"
            )
        if r is not None and np.any(np.asarray(r) <= 0):
            raise ValueError("r must be positive")

        self.dispersion = dispersion
        self.fixed_r = r
        self.r = None
        self.row_effects = None
        self.col_effects = None
        self.r_std_error = None
        self.log_r_std_error = None

    # ------------------------------------------------------------------
    # likelihood
    # ------------------------------------------------------------------

    def _as_dispersion(self, r):
        """
        Puts a dispersion in the shape this model's parameterisation expects.
        """
        r = np.asarray(r, dtype=np.float64)
        if self.dispersion == "row":
            return np.full(self.layout.n_row, float(r)) if r.ndim == 0 else r
        if r.ndim != 0:
            raise ValueError("a global dispersion must be a scalar")
        return r

    def _r_matrix(self, r):
        r = self._as_dispersion(r)
        if self.dispersion == "row":
            return np.broadcast_to(r[:, None], self.layout.shape)
        return np.broadcast_to(r, self.layout.shape)

    def _poisson_loglik_at(self, x, y):
        """
        Log-likelihood of the Poisson limit at the given effects, computed
        directly rather than by pushing r to a large number, which loses
        precision in the gamma terms.
        """
        canonical = self.layout.canonical
        mu = np.where(canonical, self.layout.dyad_scale * np.outer(x, y), 1.0)
        w = np.where(canonical, self.weights, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(w > 0, w * np.log(mu), 0.0) - mu - gammaln(w + 1)
        return float(np.where(canonical, terms, 0.0).sum())

    def _loglik_at(self, x, y, r_matrix):
        """
        Log-likelihood of the observed weights at the given effects and
        dispersion, summed over the canonical dyads.
        """
        canonical = self.layout.canonical
        mu = np.where(canonical, self.layout.dyad_scale * np.outer(x, y), 1.0)
        w = np.where(canonical, self.weights, 0.0)
        r = np.where(canonical, r_matrix, 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = (
                gammaln(w + r)
                - gammaln(r)
                - gammaln(w + 1)
                + r * (np.log(r) - np.log(r + mu))
                + np.where(w > 0, w * (np.log(mu) - np.log(r + mu)), 0.0)
            )
        return float(np.where(canonical, terms, 0.0).sum())

    def _irls_weights(self, x, y, r_matrix):
        """
        ``1 / (1 + mu / r)`` on the support: the weight the score equations
        put on each dyad, which is one in the Poisson limit and shrinks as a
        dyad's mean grows relative to r.
        """
        mu = self.layout.dyad_scale * np.outer(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1.0 / (1.0 + mu / r_matrix)
        return np.where(self.layout.support, weights, 0.0)

    def _score(self, x, y, r_matrix):
        """
        Gradient of the log-likelihood with respect to the log effects.
        """
        V = self._irls_weights(x, y, r_matrix)
        residual = (self.support_weights - np.outer(x, y)) * V
        return residual.sum(axis=1), residual.sum(axis=0)

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def _solve_effects(self, r_matrix, x0, y0, tol, max_iter, damping):
        """
        Solves the weighted score equations for fixed dispersion.

        The fixed point is IRLS written out: each effect is a weighted ratio
        of observed to expected. It converges quickly when the two sides are
        distinguishable; the undirected case updates every node at once and
        needs damping, and either can fall back on a direct maximisation,
        which is safe because the log-likelihood is concave in the log
        effects at fixed r.
        """
        observed = self.support_weights

        def update_row(x, y):
            V = self._irls_weights(x, y, r_matrix)
            return (observed * V).sum(axis=1) / (V @ y)

        def update_col(x, y):
            V = self._irls_weights(x, y, r_matrix)
            return (observed * V).sum(axis=0) / (V.T @ x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x, y, info = solve_product_form(
                update_row,
                update_col,
                x0,
                y0,
                tied=self.layout.tied,
                tol=tol,
                max_iter=max_iter,
                damping=damping,
                name=type(self).__name__,
            )

        if info["delta"] < tol:
            return x, y, info
        return self._maximise_effects(r_matrix, x, y)

    def _maximise_effects(self, r_matrix, x0, y0):
        """
        Direct maximisation of the concave profile in the log effects, used
        when the fixed point stalls.
        """
        n_row = self.layout.n_row
        tied = self.layout.tied
        floor = np.log(1e-300)

        def split(theta):
            if tied:
                return np.exp(theta), np.exp(theta)
            return np.exp(theta[:n_row]), np.exp(theta[n_row:])

        def objective(theta):
            x, y = split(np.maximum(theta, floor))
            value = self._loglik_at(x, y, r_matrix)
            row_score, col_score = self._score(x, y, r_matrix)
            gradient = (
                row_score + col_score
                if tied
                else np.concatenate([row_score, col_score])
            )
            return -value, -gradient

        theta0 = np.log(np.maximum(x0, 1e-300))
        if not tied:
            theta0 = np.concatenate([theta0, np.log(np.maximum(y0, 1e-300))])

        result = scipy.optimize.minimize(
            objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
        )
        x, y = split(result.x)
        return (
            x,
            y,
            {
                "iterations": int(result.nit),
                "delta": np.nan,
                "fallback": "L-BFGS-B",
                "message": str(result.message),
            },
        )

    def _initial_effects(self):
        total = self.row_strengths.sum()
        if total <= 0:
            raise ValueError("the network has no weight")
        scale = np.sqrt(total)
        return self.row_strengths / scale, self.col_strengths / scale

    def _poisson_effects(self, tol, max_iter):
        """
        The family A solution, which is both the constrained fit's means and
        the ``r -> inf`` limit of the unconstrained one. None when the Poisson
        model has no interior maximum for this network.
        """
        from .poisson import PoissonCM

        try:
            poisson = PoissonCM(self.W, self.layout).fit(tol=tol, max_iter=max_iter)
        except RuntimeError:
            return None
        return poisson.row_effects, poisson.col_effects

    def _initial_r(self, x, y):
        """
        Method of moments on the Poisson residuals.
        """
        canonical = self.layout.canonical
        mu = self.layout.dyad_scale * np.outer(x, y)
        residual = ((self.weights - mu) ** 2)[canonical].sum() - mu[canonical].sum()
        if residual <= 0:
            return 1e6
        return float(np.clip((mu[canonical] ** 2).sum() / residual, 1e-3, 1e6))

    def _profile_r(self, x, y, bounds):
        """
        Maximises the log-likelihood over the dispersion at fixed means.
        """
        if self.dispersion == "global":

            def negative(log_r):
                return -self._loglik_at(x, y, self._r_matrix(np.exp(log_r)))

            result = scipy.optimize.minimize_scalar(
                negative,
                bounds=(np.log(bounds[0]), np.log(bounds[1])),
                method="bounded",
                options={"xatol": 1e-10},
            )
            return float(np.exp(result.x))

        # per-row: the rows are separable at fixed means
        means = self.layout.dyad_scale * np.outer(x, y)
        r = np.empty(self.layout.n_row)
        for i in range(self.layout.n_row):
            mask = self.layout.canonical[i]
            w = self.weights[i][mask]
            mu = means[i][mask]

            def negative(log_r, w=w, mu=mu):
                r_i = np.exp(log_r)
                with np.errstate(divide="ignore", invalid="ignore"):
                    terms = (
                        gammaln(w + r_i)
                        - gammaln(r_i)
                        - gammaln(w + 1)
                        + r_i * (np.log(r_i) - np.log(r_i + mu))
                        + np.where(w > 0, w * (np.log(mu) - np.log(r_i + mu)), 0.0)
                    )
                return -float(terms.sum())

            result = scipy.optimize.minimize_scalar(
                negative,
                bounds=(np.log(bounds[0]), np.log(bounds[1])),
                method="bounded",
                options={"xatol": 1e-8},
            )
            r[i] = np.exp(result.x)
        return r

    def _dispersion_standard_error(self, r, bounds, profile, delta=0.1):
        """
        Standard error of r from the curvature of the profile log-likelihood
        in ``log r``, then the delta method back to r itself.
        """
        centre = np.log(r)
        if not (
            np.log(bounds[0]) < centre - delta and centre + delta < np.log(bounds[1])
        ):
            return np.nan, np.nan

        curvature = (
            profile(centre + delta) - 2 * profile(centre) + profile(centre - delta)
        ) / delta**2

        if curvature >= 0:
            return np.nan, np.nan
        log_se = float(1.0 / np.sqrt(-curvature))
        return float(r * log_se), log_se

    def fit(
        self,
        tol=1e-12,
        max_iter=5000,
        max_outer=100,
        outer_tol=1e-10,
        r_bounds=(1e-3, 1e8),
        damping=None,
        constrain_strengths=True,
        standard_error=True,
    ):
        """
        Fits the effect vectors and the dispersion.

        By default the means are pinned to the family A solution, which
        reproduces the observed strengths exactly, and only the dispersion is
        estimated around them. Pass ``constrain_strengths=False`` for the
        unrestricted maximum likelihood fit, which alternates between the
        effects at fixed dispersion and the dispersion at fixed means and does
        not reproduce the strengths. See the class notes.
        """
        if damping is None:
            damping = 0.5 if self.layout.tied else 1.0

        x, y = self._initial_effects()
        poisson_effects = self._poisson_effects(tol, max_iter)

        if constrain_strengths:
            if poisson_effects is None:
                raise RuntimeError(
                    "constrain_strengths=True needs the Poisson configuration "
                    "model, which has no interior maximum for this network; "
                    "pass constrain_strengths=False"
                )
            x, y = poisson_effects
            r = (
                self._as_dispersion(self.fixed_r)
                if self.fixed_r is not None
                else self._profile_r(x, y, r_bounds)
            )
            outer = 1
            info = {"method": "strength constrained"}
        elif self.fixed_r is not None:
            r = self._as_dispersion(self.fixed_r)
            x, y, info = self._solve_effects(
                self._r_matrix(r), x, y, tol, max_iter, damping
            )
            outer = 1
        else:
            r = self._as_dispersion(self._initial_r(x, y))
            previous = -np.inf
            info = {}
            for iteration in range(1, max_outer + 1):
                x, y, info = self._solve_effects(
                    self._r_matrix(r), x, y, tol, max_iter, damping
                )
                outer = iteration

                r = self._profile_r(x, y, r_bounds)
                value = self._loglik_at(x, y, self._r_matrix(r))
                if abs(value - previous) < outer_tol * max(1.0, abs(value)):
                    break
                previous = value

            # the effects above were solved at the previous r, so settle them
            # against the one that was finally chosen
            x, y, info = self._solve_effects(
                self._r_matrix(r), x, y, tol, max_iter, damping
            )

        self.row_effects, self.col_effects = x, y
        self.r = r
        self.constrained = bool(constrain_strengths)
        self._M = np.outer(x, y)
        self.r_matrix = np.asarray(self._r_matrix(r), dtype=np.float64)

        self.fit_info = {
            "r": r,
            "outer_iterations": outer,
            "inner_iterations": info.get("iterations"),
            "dispersion": self.dispersion,
            "constrain_strengths": self.constrained,
        }
        self.fit_info.update(self._overdispersion_evidence(poisson_effects))
        if "fallback" in info:
            self.fit_info["fallback"] = info["fallback"]

        if standard_error and self.fixed_r is None and self.dispersion == "global":
            if constrain_strengths:

                def profile(log_r):
                    return self._loglik_at(x, y, self._r_matrix(np.exp(log_r)))
            else:
                x0, y0 = self._initial_effects()

                def profile(log_r):
                    r_matrix = self._r_matrix(np.exp(log_r))
                    fitted_x, fitted_y, _ = self._solve_effects(
                        r_matrix, x0, y0, tol, max_iter, damping
                    )
                    return self._loglik_at(fitted_x, fitted_y, r_matrix)

            self.r_std_error, self.log_r_std_error = self._dispersion_standard_error(
                r, r_bounds, profile
            )
            self.fit_info["r_std_error"] = self.r_std_error

        if constrain_strengths:
            self.fit_info["constraint_error"] = self.constraint_error()
        else:
            row_score, col_score = self._score(x, y, self.r_matrix)
            score_norm = max(np.max(np.abs(row_score)), np.max(np.abs(col_score)))
            self.fit_info["score_norm"] = score_norm
            if score_norm > 1e-6 * max(1.0, self.total_weight):
                warnings.warn(
                    f"{type(self).__name__}: score equations are only satisfied to "
                    f"{score_norm:.3e}",
                    RuntimeWarning,
                )
        return self

    def _overdispersion_evidence(self, poisson_effects):
        """
        The log-likelihood at the fitted dispersion and at ``r -> inf``, plus
        the likelihood ratio between them.

        The Poisson model is the boundary of the negative binomial family, so
        the null distribution of the ratio is an even mixture of a point mass
        at zero and a chi-square on one degree of freedom, not a plain
        chi-square -- which is where the halved tail probability comes from.

        That reference is for one dispersion. With ``dispersion="row"`` the
        ratio is still reported but the p-value is NaN: it would need its own
        calibration, a parametric bootstrap for instance.
        """
        value = self.loglik()
        evidence = {"loglik": value}

        if poisson_effects is None:
            evidence["loglik_poisson"] = np.nan
            return evidence

        limit = self._poisson_loglik_at(*poisson_effects)
        ratio = 2 * (value - limit)
        evidence["loglik_poisson"] = limit
        evidence["overdispersion_lr"] = ratio

        if self.dispersion == "row":
            # the Poisson limit puts every row's r on its boundary at once, so
            # the reference is a mixture over how many of them sit there, not
            # the one-parameter mixture below -- and with the means free that
            # mixture's weights are not known. No p-value rather than a wrong one.
            evidence["overdispersion_p"] = np.nan
            return evidence

        evidence["overdispersion_p"] = (
            0.5 * scipy.stats.chi2.sf(ratio, 1) if ratio > 0 else 1.0
        )
        return evidence

    # ------------------------------------------------------------------
    # distribution
    # ------------------------------------------------------------------

    def _dist(self, idx):
        self._require_fit()
        mu = self._take(self.mu, idx)
        r = self._take(self.r_matrix, idx)
        return scipy.stats.nbinom(n=r, p=r / (r + mu))

    def profile_loglik(self, r_values, tol=1e-12, max_iter=5000, damping=None):
        """
        The profile log-likelihood at each of ``r_values``. Handy for checking
        unimodality.

        Follows however the model was fitted: with the means constrained they
        stay at the family A solution, and otherwise the effects are re-fitted
        at each r so the nuisance parameters are profiled out.
        """
        self._require_fit()
        if self.dispersion != "global":
            raise ValueError("the profile is only one dimensional for a global r")
        if damping is None:
            damping = 0.5 if self.layout.tied else 1.0

        out = np.empty(len(r_values))
        if self.constrained:
            for k, r in enumerate(r_values):
                out[k] = self._loglik_at(
                    self.row_effects, self.col_effects, self._r_matrix(r)
                )
            return out

        x0, y0 = self._initial_effects()
        for k, r in enumerate(r_values):
            r_matrix = self._r_matrix(r)
            x, y, _ = self._solve_effects(r_matrix, x0, y0, tol, max_iter, damping)
            out[k] = self._loglik_at(x, y, r_matrix)
        return out


class BINBCM(NegativeBinomialCM):
    """
    Bipartite negative binomial configuration model.
    """

    def __init__(self, B, dispersion="global", r=None):
        B = dense(B)
        super().__init__(B, DyadLayout.bipartite(*B.shape), dispersion=dispersion, r=r)


class UNBCM(NegativeBinomialCM):
    """
    Undirected negative binomial configuration model.
    """

    def __init__(self, A, self_loops=False, dispersion="global", r=None):
        A = dense(A)
        super().__init__(
            A,
            DyadLayout.undirected(A.shape[0], self_loops=self_loops),
            dispersion=dispersion,
            r=r,
        )


class DNBCM(NegativeBinomialCM):
    """
    Directed negative binomial configuration model.
    """

    def __init__(self, A, self_loops=False, dispersion="global", r=None):
        A = dense(A)
        super().__init__(
            A,
            DyadLayout.directed(A.shape[0], self_loops=self_loops),
            dispersion=dispersion,
            r=r,
        )
