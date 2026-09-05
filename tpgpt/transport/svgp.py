"""Sparse variational Gaussian Process regression (paper Appendix A).

The surface-cleaning parameterisation in Sec. V-C uses a 20x20 = 400-point
source/target point cloud and approximates the transportation map with 100
inducing points ("SV-GPT"). Exact GP inference costs O(N^3); the collapsed
variational bound of Titsias (ref. [63], with the stochastic extension of
ref. [64]) reduces this to O(N M^2).

At the problem sizes this project reaches (N <= a few thousand) the collapsed
bound has a closed form, so no minibatching or stochastic optimisation is
needed -- the objective is maximised directly with L-BFGS-B. The public API
mirrors :class:`tpgpt.transport.gp.GaussianProcessRegressor` so the two are
interchangeable inside a :class:`~tpgpt.transport.maps.TransportMap`.

Note that with ``M < N`` the approximation no longer interpolates the keypoints
exactly, so property (i) of Sec. III-C (``T = phi(S)``) holds only up to the
approximation error. Use the exact GP when exact keypoint matching matters.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from tpgpt.transport.gp import (
    KernelHyperparameters,
    _sqdist,
    geometric_length_scale_bounds,
    residual_signal_variance_bounds,
)

logger = logging.getLogger(__name__)

_LOG2PI = float(np.log(2.0 * np.pi))


def farthest_point_sample(X: np.ndarray, n_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic farthest-point subsampling of ``X``.

    Preferred over random or k-means initialisation for inducing points because
    it is reproducible and spreads the points over the full support of the
    source distribution, which is what the transportation map needs.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n_points >= n:
        return X.copy()
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    dist = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(n_points - 1):
        nxt = int(np.argmax(dist))
        idx.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return X[np.array(idx)]


class SparseGaussianProcessRegressor:
    """Titsias-style sparse GP with a shared squared-exponential kernel.

    Args:
        n_inducing: Number of inducing points ``M``. If it meets or exceeds
            ``N`` every training point is used and the model reduces to the
            exact GP.
        noise_variance: Likelihood noise. Must be strictly positive: it appears
            as ``1/sigma_n^2`` in the collapsed bound.
        optimize: Maximise the collapsed bound over the kernel hyperparameters.
        optimize_inducing: Also optimise inducing-point locations. Off by
            default -- it multiplies the parameter count by ``M * d`` for a
            usually marginal gain once farthest-point initialisation is used.
    """

    def __init__(
        self,
        n_inducing: int = 100,
        length_scale: float | None = None,
        signal_variance: float | None = None,
        noise_variance: float = 1e-6,
        optimize: bool = True,
        optimize_inducing: bool = False,
        length_scale_bounds: str | tuple[float, float] = "auto",
        signal_variance_bounds: str | tuple[float, float] = "auto",
        jitter: float = 1e-10,
        random_state: int | None = 0,
    ):
        self.n_inducing = int(n_inducing)
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = float(noise_variance)
        self.optimize = optimize
        self.optimize_inducing = optimize_inducing
        self.length_scale_bounds = length_scale_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.jitter = float(jitter)
        self.random_state = random_state

        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.params: KernelHyperparameters | None = None
        self._L: np.ndarray | None = None   # chol(Kuu)
        self._LB: np.ndarray | None = None  # chol(I + A A^T)
        self._c: np.ndarray | None = None   # (M, D)
        self._w: np.ndarray | None = None   # (M, D) mean weights on k(., Z)

    # ------------------------------------------------------------- kernel
    def _kernel(self, A: np.ndarray, B: np.ndarray, params=None) -> np.ndarray:
        p = params or self.params
        return p.signal_variance * np.exp(-0.5 * _sqdist(A, B) / p.length_scale**2)

    def _chol_kuu(self, Z: np.ndarray, params: KernelHyperparameters) -> np.ndarray:
        Kuu = self._kernel(Z, Z, params)
        jitter = self.jitter * max(params.signal_variance, 1e-12)
        for _ in range(8):
            try:
                return cholesky(Kuu + jitter * np.eye(Z.shape[0]), lower=True)
            except np.linalg.LinAlgError:
                jitter *= 100.0
        raise np.linalg.LinAlgError("Kuu is not positive definite; inducing points may coincide")

    # --------------------------------------------------------------- bound
    def _collapsed_bound(self, Z: np.ndarray, params: KernelHyperparameters) -> float:
        """Titsias' collapsed evidence lower bound, summed over outputs."""
        X, Y = self.X_train, self.Y_train
        N, D = Y.shape
        sigma2 = params.noise_variance
        try:
            L = self._chol_kuu(Z, params)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return -np.inf

        Kuf = self._kernel(Z, X, params)
        A = solve_triangular(L, Kuf, lower=True) / np.sqrt(sigma2)   # (M, N)
        B = np.eye(Z.shape[0]) + A @ A.T
        LB = cholesky(B, lower=True)
        c = solve_triangular(LB, A @ Y, lower=True) / np.sqrt(sigma2)  # (M, D)

        bound = -0.5 * N * D * (_LOG2PI + np.log(sigma2))
        bound -= D * float(np.sum(np.log(np.diag(LB))))
        bound -= 0.5 * float(np.sum(Y**2)) / sigma2
        bound += 0.5 * float(np.sum(c**2))
        # Trace term: penalises the approximation error Kff - Qff.
        trace_kff = N * params.signal_variance
        trace_qff = sigma2 * float(np.sum(A**2))
        bound -= 0.5 * D * (trace_kff - trace_qff) / sigma2
        return float(bound)

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SparseGaussianProcessRegressor":
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
        self._signal_variance_bounds_used = sv_bounds

        ls0 = self.length_scale or float(np.sqrt(ls_bounds[0] * ls_bounds[1]))
        ls0 = float(np.clip(ls0, *ls_bounds))
        sv0 = self.signal_variance or float(np.sqrt(sv_bounds[0] * sv_bounds[1]))
        params = KernelHyperparameters(ls0, float(np.clip(sv0, *sv_bounds)), self.noise_variance)
        Z = farthest_point_sample(X, self.n_inducing, seed=self.random_state or 0)

        if self.optimize:
            Z, params = self._optimize(Z, params, ls_bounds, sv_bounds)

        self.Z, self.params = Z, params
        self._finalize()
        logger.debug(
            "SVGP fitted: M=%d N=%d length_scale=%.5g signal_variance=%.5g",
            Z.shape[0], X.shape[0], params.length_scale, params.signal_variance,
        )
        return self

    def _optimize(self, Z, params, ls_bounds, sv_bounds):
        M, d = Z.shape
        n_kernel = 2

        def unpack(v):
            p = KernelHyperparameters(np.exp(v[0]), np.exp(v[1]), self.noise_variance)
            Zc = v[n_kernel:].reshape(M, d) if self.optimize_inducing else Z
            return Zc, p

        def objective(v):
            Zc, p = unpack(v)
            val = self._collapsed_bound(Zc, p)
            return 1e30 if not np.isfinite(val) else -val

        v0 = [np.log(params.length_scale), np.log(params.signal_variance)]
        bounds = [np.log(ls_bounds), np.log(sv_bounds)]
        if self.optimize_inducing:
            v0 = np.concatenate([v0, Z.ravel()])
            bounds = bounds + [(None, None)] * (M * d)

        res = minimize(objective, np.asarray(v0, dtype=float), method="L-BFGS-B", bounds=bounds)
        return unpack(res.x)

    def _finalize(self) -> None:
        """Cache the factors reused by every prediction."""
        sigma2 = self.params.noise_variance
        self._L = self._chol_kuu(self.Z, self.params)
        Kuf = self._kernel(self.Z, self.X_train)
        A = solve_triangular(self._L, Kuf, lower=True) / np.sqrt(sigma2)
        B = np.eye(self.Z.shape[0]) + A @ A.T
        self._LB = cholesky(B, lower=True)
        self._c = solve_triangular(self._LB, A @ self.Y_train, lower=True) / np.sqrt(sigma2)
        # Mean is a plain kernel expansion on the inducing points:
        #   mu(x*) = k(x*, Z) @ w,  w = Kuu^-1/2 LB^-T c
        self._w = solve_triangular(
            self._L, solve_triangular(self._LB, self._c, lower=True, trans="T"),
            lower=True, trans="T",
        )

    # ------------------------------------------------------------ predict
    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False):
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Kus = self._kernel(self.Z, X)               # (M, n)
        mean = Kus.T @ self._w
        if not (return_std or return_cov):
            return mean

        tmp1 = solve_triangular(self._L, Kus, lower=True)     # (M, n)
        tmp2 = solve_triangular(self._LB, tmp1, lower=True)   # (M, n)
        if return_cov:
            cov = self._kernel(X, X) - tmp1.T @ tmp1 + tmp2.T @ tmp2
            return mean, cov
        var = (
            self.params.signal_variance
            - np.einsum("ij,ij->j", tmp1, tmp1)
            + np.einsum("ij,ij->j", tmp2, tmp2)
        )
        return mean, np.sqrt(np.maximum(var, 0.0))

    def posterior_covariance(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))
        a1 = solve_triangular(self._L, self._kernel(self.Z, Xa), lower=True)
        b1 = solve_triangular(self._L, self._kernel(self.Z, Xb), lower=True)
        a2 = solve_triangular(self._LB, a1, lower=True)
        b2 = solve_triangular(self._LB, b1, lower=True)
        return self._kernel(Xa, Xb) - a1.T @ b1 + a2.T @ b2

    def predict_gradient(self, X: np.ndarray, return_std: bool = False):
        """Derivative posterior of the sparse GP (same structure as Eq. 16)."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        M = self.Z.shape[0]
        ell2 = self.params.length_scale**2

        Kus = self._kernel(self.Z, X)                             # (M, n)
        diff = X[:, None, :] - self.Z[None, :, :]                 # (n, M, d)
        dKus = -Kus.T[:, :, None] * diff / ell2                   # (n, M, d)

        J = np.einsum("nmd,mo->nod", dKus, self._w)
        if not return_std:
            return J

        flat = dKus.transpose(1, 0, 2).reshape(M, n * d)
        t1 = solve_triangular(self._L, flat, lower=True)
        t2 = solve_triangular(self._LB, t1, lower=True)
        var = (
            self.params.signal_variance / ell2
            - np.einsum("ij,ij->j", t1, t1)
            + np.einsum("ij,ij->j", t2, t2)
        ).reshape(n, d)
        std = np.sqrt(np.maximum(var, 0.0))
        return J, np.broadcast_to(std[:, None, :], (n, J.shape[1], d)).copy()

    def _check_fitted(self) -> None:
        if self._w is None:
            raise RuntimeError("SparseGaussianProcessRegressor.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.params is None:
            return "SparseGaussianProcessRegressor(unfitted)"
        return (
            f"SparseGaussianProcessRegressor(N={self.X_train.shape[0]}, "
            f"M={self.Z.shape[0]}, length_scale={self.params.length_scale:.4g})"
        )
