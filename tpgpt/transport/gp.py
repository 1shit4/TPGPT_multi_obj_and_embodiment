"""Zero-mean multi-output Gaussian Process regression (paper Appendix A).

This replaces ``sklearn.gaussian_process.GaussianProcessRegressor`` for two
reasons that are load-bearing for the paper:

1. **Derivative uncertainty.** Sec. III-H propagates the uncertainty of the
   transportation map's *derivative* into the transported velocity labels
   (Eqs. 12, 16). sklearn exposes no derivative posterior at all.
2. **Zero-mean prior.** The out-of-distribution argument in Sec. III-E-b -- far
   from the keypoints the residual ``psi`` decays to zero, so the full map
   ``phi`` reduces to the affine ``gamma`` (Fig. 2d) -- requires a strict zero
   prior mean. sklearn's ``normalize_y=True`` centres and rescales the targets,
   which silently breaks that guarantee.

Following Appendix A, one kernel (and therefore one Cholesky factor of
``K + sigma_n^2 I``) is shared across all output dimensions, so the expensive
factorisation is computed once regardless of output dimensionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

logger = logging.getLogger(__name__)

_LOG2PI = float(np.log(2.0 * np.pi))


@dataclass
class KernelHyperparameters:
    """Squared-exponential hyperparameters (paper Appendix A).

    ``k_SE(x_i, x_j) = signal_variance * exp(-||x_i - x_j||^2 / (2 * length_scale^2))``
    """

    length_scale: float
    signal_variance: float
    noise_variance: float

    def to_log_theta(self) -> np.ndarray:
        return np.log(
            [self.length_scale, self.signal_variance, max(self.noise_variance, 1e-300)]
        )

    @staticmethod
    def from_log_theta(theta: np.ndarray) -> "KernelHyperparameters":
        ell, sig, noise = np.exp(np.asarray(theta, dtype=float))
        return KernelHyperparameters(ell, sig, noise)


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at zero."""
    return np.maximum(cdist(A, B, metric="sqeuclidean"), 0.0)


def geometric_length_scale_bounds(
    X: np.ndarray, lower_factor: float = 0.5, upper_factor: float = 2.0
) -> tuple[float, float]:
    """Length-scale bounds derived from the training-input geometry.

    Marginal-likelihood maximisation on the handful of keypoints available in a
    task parameterisation is badly under-determined, and it fails in *both*
    directions. Left unbounded it either

    * drives the length scale to zero, turning ``psi`` into a nearest-neighbour
      spike train that is exactly zero at every trajectory point -- the map then
      matches the keypoints (property (i)) while deforming nothing in between,
      defeating the purpose of the nonlinear stage; or
    * drives it to infinity together with the amplitude, the degenerate regime
      in which a squared-exponential kernel imitates a low-order polynomial.
      That fits the keypoints too, but extrapolates violently, which breaks the
      out-of-distribution behaviour the paper relies on in Sec. III-E-b: far
      from the source distribution ``psi`` must decay to zero so that ``phi``
      reduces to the affine ``gamma`` (Fig. 2d).

    The paper does not specify how the hyperparameters are bounded, so we tie
    them to the keypoint geometry: the length scale may not shrink below half
    the closest keypoint separation, nor exceed a small multiple of the diameter
    of the keypoint cloud. See also :func:`residual_signal_variance_bounds`,
    which bounds the amplitude for the same reason.

    Returns:
        ``(lower, upper)`` bounds on the length scale.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] < 2:
        return 1e-3, 1e3
    dists = pdist(X)
    positive = dists[dists > 1e-12]
    if positive.size == 0:
        return 1e-3, 1e3
    lower = lower_factor * float(positive.min())
    upper = upper_factor * float(dists.max())
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def residual_signal_variance_bounds(
    Y: np.ndarray, lower_factor: float = 1e-8, upper_factor: float = 10.0
) -> tuple[float, float]:
    """Amplitude bounds anchored on the observed residual magnitude.

    The residual field ``psi`` explains the keypoint mismatch left by the affine
    stage, so its prior amplitude should be of the order of that mismatch. An
    unconstrained ``sigma_p^2`` will happily grow by orders of magnitude to buy a
    slightly smoother interpolant, at the cost of predicting metre-scale
    deformations in regions with no keypoints at all.

    Returns:
        ``(lower, upper)`` bounds on ``sigma_p^2``.
    """
    Y = np.asarray(Y, dtype=float)
    scale = float(np.mean(np.sum(Y**2, axis=-1))) if Y.size else 0.0
    scale = max(scale, float(np.var(Y)) if Y.size else 0.0, 1e-12)
    return lower_factor * scale, upper_factor * scale


class GaussianProcessRegressor:
    """Exact zero-mean GP with a shared squared-exponential kernel.

    Args:
        length_scale: Initial length scale. ``None`` uses the geometric mean of
            the bounds derived from the training inputs.
        signal_variance: Initial kernel amplitude ``sigma_p^2``. ``None``
            initialises from the target variance.
        noise_variance: Likelihood noise ``sigma_n^2``. Kept small by default so
            that property (i) of Sec. III-C -- ``T = phi(S)`` -- holds to
            numerical precision.
        optimize: Maximise the log marginal likelihood over the hyperparameters.
            When ``False`` the supplied values are used as given.
        optimize_noise: Include the noise in the optimisation. Off by default;
            letting the noise grow trades away exact keypoint matching.
        length_scale_bounds: ``"auto"`` for :func:`geometric_length_scale_bounds`,
            or an explicit ``(lower, upper)`` pair.
        signal_variance_bounds: ``"auto"`` for
            :func:`residual_signal_variance_bounds`, or an explicit pair.
        n_restarts: Extra random restarts of the optimiser.
        jitter: Base diagonal jitter for numerical conditioning, *relative* to
            the signal variance so it stays scale invariant. Escalated
            automatically if the Cholesky factorisation fails. Keep it well
            below the smallest eigenvalue of the correlation matrix, otherwise
            it smooths away the low-energy interpolation modes and property (i)
            of Sec. III-C degrades.
    """

    def __init__(
        self,
        length_scale: float | None = None,
        signal_variance: float | None = None,
        noise_variance: float = 1e-10,
        optimize: bool = True,
        optimize_noise: bool = False,
        length_scale_bounds: str | tuple[float, float] = "auto",
        signal_variance_bounds: str | tuple[float, float] = "auto",
        noise_variance_bounds: tuple[float, float] = (1e-12, 1e-2),
        n_restarts: int = 3,
        jitter: float = 1e-12,
        random_state: int | None = 0,
    ):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.optimize = optimize
        self.optimize_noise = optimize_noise
        self.length_scale_bounds = length_scale_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.noise_variance_bounds = noise_variance_bounds
        self.n_restarts = int(n_restarts)
        self.jitter = float(jitter)
        self.random_state = random_state

        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
        self.params: KernelHyperparameters | None = None
        self.alpha_: np.ndarray | None = None  # K^-1 Y, shape (N, D)
        self._L: np.ndarray | None = None
        self._applied_jitter: float = 0.0

    # ------------------------------------------------------------- kernel
    def _kernel(self, A: np.ndarray, B: np.ndarray, params=None) -> np.ndarray:
        p = params or self.params
        return p.signal_variance * np.exp(-0.5 * _sqdist(A, B) / p.length_scale**2)

    def _train_gram(self, params: KernelHyperparameters, jitter: float) -> np.ndarray:
        K = self._kernel(self.X_train, self.X_train, params)
        K[np.diag_indices_from(K)] += params.noise_variance + jitter
        return K

    def _stable_cholesky(self, params: KernelHyperparameters) -> tuple[np.ndarray, float]:
        """Cholesky with escalating jitter; returns ``(L, applied_jitter)``."""
        jitter = self.jitter * max(params.signal_variance, 1e-300)
        for _ in range(10):
            try:
                return cholesky(self._train_gram(params, jitter), lower=True), jitter
            except np.linalg.LinAlgError:
                jitter *= 100.0
        raise np.linalg.LinAlgError(
            "Gram matrix is not positive definite even with escalated jitter; "
            "check for duplicate training inputs."
        )

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit the GP.

        Args:
            X: ``(N, d)`` training inputs.
            Y: ``(N, D)`` training targets. A zero prior mean is assumed, so
                targets are used exactly as supplied -- no centring or scaling.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y disagree on N: {X.shape[0]} vs {Y.shape[0]}")
        self.X_train, self.Y_train = X, Y

        if self.length_scale_bounds == "auto":
            ls_bounds = geometric_length_scale_bounds(X)
        else:
            ls_bounds = tuple(float(b) for b in self.length_scale_bounds)
        if self.signal_variance_bounds == "auto":
            sv_bounds = residual_signal_variance_bounds(Y)
        else:
            sv_bounds = tuple(float(b) for b in self.signal_variance_bounds)
        self._length_scale_bounds_used = ls_bounds
        self._signal_variance_bounds_used = sv_bounds

        ls0 = self.length_scale if self.length_scale is not None else float(
            np.sqrt(ls_bounds[0] * ls_bounds[1])
        )
        ls0 = float(np.clip(ls0, *ls_bounds))
        sv0 = self.signal_variance
        if sv0 is None:
            sv0 = float(np.sqrt(sv_bounds[0] * sv_bounds[1]))
        sv0 = float(np.clip(sv0, *sv_bounds))
        params = KernelHyperparameters(ls0, sv0, float(self.noise_variance))

        if self.optimize:
            params = self._optimize_hyperparameters(params, ls_bounds, sv_bounds)
        self.params = params

        self._L, self._applied_jitter = self._stable_cholesky(params)
        self.alpha_ = cho_solve((self._L, True), self.Y_train)
        logger.debug(
            "GP fitted: length_scale=%.5g signal_variance=%.5g noise=%.3g (bounds %s)",
            params.length_scale,
            params.signal_variance,
            params.noise_variance,
            ls_bounds,
        )
        return self

    # ------------------------------------------------ marginal likelihood
    def log_marginal_likelihood(
        self, params: KernelHyperparameters, eval_gradient: bool = False
    ):
        """Log marginal likelihood of the shared-kernel multi-output GP.

        With ``D`` conditionally independent outputs sharing one kernel,
        ``log p(Y|X) = -0.5 sum_o y_o^T K^-1 y_o - (D/2) log|K| - (N D/2) log 2pi``.
        """
        N, D = self.Y_train.shape
        try:
            L, jitter = self._stable_cholesky(params)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros(3)) if eval_gradient else -np.inf

        alpha = cho_solve((L, True), self.Y_train)  # (N, D)
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        lml = -0.5 * float(np.sum(self.Y_train * alpha)) - 0.5 * D * log_det
        lml -= 0.5 * N * D * _LOG2PI
        if not eval_gradient:
            return lml

        # dLML/dtheta = 0.5 tr[(alpha alpha^T - D K^-1) dK/dtheta]
        K_inv = cho_solve((L, True), np.eye(N))
        W = alpha @ alpha.T - D * K_inv

        d2 = _sqdist(self.X_train, self.X_train)
        Kf = params.signal_variance * np.exp(-0.5 * d2 / params.length_scale**2)
        grads = np.array(
            [
                0.5 * float(np.sum(W * (Kf * d2 / params.length_scale**2))),  # log ell
                0.5 * float(np.sum(W * Kf)),                                  # log sigma_p^2
                0.5 * float(np.trace(W)) * params.noise_variance,             # log sigma_n^2
            ]
        )
        return lml, grads

    def _optimize_hyperparameters(
        self,
        init: KernelHyperparameters,
        ls_bounds: tuple[float, float],
        sv_bounds: tuple[float, float],
    ) -> KernelHyperparameters:
        nv_bounds = self.noise_variance_bounds
        fixed_noise = not self.optimize_noise

        def negative_lml(theta_free: np.ndarray):
            theta = (
                np.array([theta_free[0], theta_free[1], np.log(init.noise_variance)])
                if fixed_noise
                else np.asarray(theta_free)
            )
            params = KernelHyperparameters.from_log_theta(theta)
            lml, grad = self.log_marginal_likelihood(params, eval_gradient=True)
            if not np.isfinite(lml):
                return 1e30, np.zeros_like(theta_free)
            return -lml, -(grad[:2] if fixed_noise else grad)

        bounds = [np.log(ls_bounds), np.log(sv_bounds)]
        if not fixed_noise:
            bounds.append(np.log(nv_bounds))

        theta0 = init.to_log_theta()
        starts = [theta0[:2] if fixed_noise else theta0]
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_restarts):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

        best_theta, best_val = starts[0], np.inf
        for theta_init in starts:
            try:
                res = minimize(
                    negative_lml,
                    np.clip(theta_init, [b[0] for b in bounds], [b[1] for b in bounds]),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                continue
            if res.fun < best_val:
                best_val, best_theta = res.fun, res.x

        theta = (
            np.array([best_theta[0], best_theta[1], np.log(init.noise_variance)])
            if fixed_noise
            else best_theta
        )
        fitted = KernelHyperparameters.from_log_theta(theta)
        if np.isclose(fitted.length_scale, ls_bounds[0], rtol=1e-3):
            logger.info(
                "Length scale settled on its lower bound (%.4g). The residual field is "
                "as rough as the keypoint spacing allows.",
                ls_bounds[0],
            )
        return fitted

    # ------------------------------------------------------------ predict
    def predict(
        self, X: np.ndarray, return_std: bool = False, return_cov: bool = False
    ):
        """Posterior mean (Eq. 14) and, optionally, variance (Eq. 15).

        Returns:
            ``mean`` of shape ``(n, D)``; plus ``std`` of shape ``(n,)`` when
            ``return_std`` (the kernel is shared, so every output dimension has
            the same predictive variance), or the full ``(n, n)`` covariance
            when ``return_cov``.
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K_s = self._kernel(X, self.X_train)          # (n, N)
        mean = K_s @ self.alpha_                     # Eq. (14)
        if not (return_std or return_cov):
            return mean

        v = solve_triangular(self._L, K_s.T, lower=True)  # (N, n)
        if return_cov:
            cov = self._kernel(X, X) - v.T @ v            # Eq. (15)
            return mean, cov
        var = self.params.signal_variance - np.einsum("ij,ij->j", v, v)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def posterior_covariance(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        """Posterior covariance between two sets of test inputs (Eq. 15)."""
        self._check_fitted()
        Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))
        Ka = solve_triangular(self._L, self._kernel(Xa, self.X_train).T, lower=True)
        Kb = solve_triangular(self._L, self._kernel(Xb, self.X_train).T, lower=True)
        return self._kernel(Xa, Xb) - Ka.T @ Kb

    def predict_gradient(self, X: np.ndarray, return_std: bool = False):
        """Posterior of the GP derivative (paper Appendix A, Eq. 16).

        A GP derivative is itself a GP. For the squared-exponential kernel,

        ``k10_d = dk(x*, x)/dx*_d = -k(x*, x) (x*_d - x_d) / ell^2``
        ``k11_de = d^2 k(x*, x*')/dx*_d dx*'_e |_{x*=x*'} = (sigma_p^2 / ell^2) delta_de``

        Returns:
            ``J`` of shape ``(n, D, d)`` -- the Jacobian of the posterior mean --
            and, when ``return_std``, its standard deviation with the same shape.
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        ell2 = self.params.length_scale**2

        K_s = self._kernel(X, self.X_train)                       # (n, N)
        diff = X[:, None, :] - self.X_train[None, :, :]           # (n, N, d)
        k10 = -K_s[:, :, None] * diff / ell2                      # (n, N, d)

        # Mean derivative: dK_s/dx* @ alpha  ->  (n, D, d)
        J = np.einsum("njd,jo->nod", k10, self.alpha_)
        if not return_std:
            return J

        # Var[df/dx*_d] = k11_dd - k10_d^T K^-1 k10_d, shared across outputs.
        k10_flat = k10.transpose(1, 0, 2).reshape(self.X_train.shape[0], n * d)
        v = solve_triangular(self._L, k10_flat, lower=True)       # (N, n*d)
        quad = np.einsum("ij,ij->j", v, v).reshape(n, d)
        var = np.maximum(self.params.signal_variance / ell2 - quad, 0.0)
        std = np.sqrt(var)                                        # (n, d)
        return J, np.broadcast_to(std[:, None, :], (n, J.shape[1], d)).copy()

    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state=None):
        """Draw ``n_samples`` joint function samples; shape ``(n_samples, n, D)``."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean, cov = self.predict(X, return_cov=True)
        cov = cov + 1e-12 * np.eye(cov.shape[0])
        rng = np.random.default_rng(random_state)
        L = cholesky(cov, lower=True)
        eps = rng.standard_normal((n_samples, X.shape[0], mean.shape[1]))
        return mean[None] + np.einsum("ij,sjo->sio", L, eps)

    def _check_fitted(self) -> None:
        if self.alpha_ is None:
            raise RuntimeError("GaussianProcessRegressor.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.params is None:
            return "GaussianProcessRegressor(unfitted)"
        return (
            f"GaussianProcessRegressor(N={self.X_train.shape[0]}, "
            f"length_scale={self.params.length_scale:.4g}, "
            f"signal_variance={self.params.signal_variance:.4g}, "
            f"noise={self.params.noise_variance:.3g})"
        )
