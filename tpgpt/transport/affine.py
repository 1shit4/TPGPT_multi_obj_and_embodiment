"""Affine component ``gamma`` of the transportation map.

Implements paper Sec. III-E-a, Eqs. (4)-(7): the globally-acting rigid
transformation that aligns the source keypoint set ``S`` to the target set
``T`` before the nonlinear residual ``psi`` is fitted. Solving for ``gamma``
first is what keeps the nonlinear part small; the paper notes
``||T - gamma(S)|| <= ||T - S||`` as a direct consequence.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AffineMap:
    """Least-squares rigid alignment ``gamma(x) = A (x - S_bar) + T_bar``.

    ``A`` is a proper rotation obtained by SVD (Procrustes / Kabsch, ref. [46]
    in the paper). Reflections are rejected, since a mirrored transform cannot
    correspond to a physical rearrangement of the scene.

    Args:
        rank_tol: Relative tolerance used to decide that the cross-covariance
            is rank deficient. Paper Sec. III-E-a: when ``det(Sigma) = 0`` the
            rotation is not uniquely defined and is set to the identity.
    """

    def __init__(self, rank_tol: float = 1e-10):
        self.rank_tol = float(rank_tol)
        self.A: np.ndarray | None = None
        self.source_mean: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.rank_deficient: bool = False
        self.singular_values: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, S: np.ndarray, T: np.ndarray) -> "AffineMap":
        """Fit ``gamma`` from paired source/target keypoints.

        Args:
            S: ``(N, d)`` source keypoints.
            T: ``(N, d)`` target keypoints, paired elementwise with ``S``.
        """
        S = np.atleast_2d(np.asarray(S, dtype=float))
        T = np.atleast_2d(np.asarray(T, dtype=float))
        if S.shape != T.shape:
            raise ValueError(f"S and T must have equal shape, got {S.shape} vs {T.shape}")
        if S.ndim != 2:
            raise ValueError(f"expected 2-D keypoint arrays, got {S.ndim}-D")

        d = S.shape[1]
        self.source_mean = S.mean(axis=0)
        self.target_mean = T.mean(axis=0)

        S_c = S - self.source_mean
        T_c = T - self.target_mean

        # Eq. (5): U Sigma V^T = (S - S_bar)^T (T - T_bar)
        H = S_c.T @ T_c
        U, sigma, Vt = np.linalg.svd(H)
        self.singular_values = sigma

        # Paper Sec. III-E-a: det(Sigma) == 0 leaves the rotation undetermined,
        # in which case it defaults to the identity. This happens whenever the
        # keypoints span fewer than ``d`` dimensions, e.g. coplanar corners.
        scale = sigma[0] if sigma[0] > 0 else 1.0
        self.rank_deficient = bool(np.any(sigma < self.rank_tol * scale))
        if self.rank_deficient:
            logger.warning(
                "Cross-covariance is rank deficient (singular values %s); "
                "rotation is not uniquely defined, defaulting to identity.",
                np.array2string(sigma, precision=3),
            )
            self.A = np.eye(d)
            return self

        # Eq. (6): A = V U^T
        A = Vt.T @ U.T
        if np.linalg.det(A) < 0:
            # Reflection: flip the last column of V and recompute (ref. [46]).
            Vt = Vt.copy()
            Vt[-1, :] *= -1
            A = Vt.T @ U.T
        self.A = A
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply Eq. (7). Accepts ``(d,)`` or ``(N, d)``; preserves that shape."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        single = X_arr.ndim == 1
        X_2d = np.atleast_2d(X_arr)
        out = (X_2d - self.source_mean) @ self.A.T + self.target_mean
        return out[0] if single else out

    __call__ = predict

    def jacobian(self, X: np.ndarray | None = None) -> np.ndarray:
        """Jacobian of ``gamma``, which is the constant ``A`` (paper Sec. III-F).

        ``X`` is accepted and ignored so the API matches the nonlinear
        regressors, which do depend on the query location.
        """
        self._check_fitted()
        return self.A

    def inverse(self, X: np.ndarray) -> np.ndarray:
        """Inverse map ``A^T (x - T_bar) + S_bar``; exact since ``A`` is a rotation."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        single = X_arr.ndim == 1
        X_2d = np.atleast_2d(X_arr)
        out = (X_2d - self.target_mean) @ self.A + self.source_mean
        return out[0] if single else out

    def _check_fitted(self) -> None:
        if self.A is None:
            raise RuntimeError("AffineMap.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.A is None:
            return "AffineMap(unfitted)"
        return (
            f"AffineMap(d={self.A.shape[0]}, rank_deficient={self.rank_deficient}, "
            f"det={np.linalg.det(self.A):.6f})"
        )
