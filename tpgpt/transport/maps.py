"""The policy transportation map ``phi`` (paper Sec. III-D to III-H).

``phi(x) = gamma(x) + psi(gamma(x))``  -- Eq. (2)

A globally-acting affine ``gamma`` (Eq. 4-7) is composed with a nonlinear
residual ``psi`` (Eq. 8) that is fitted only on what ``gamma`` leaves over. The
same map, through its Jacobian, transports every other policy label: velocities
(Eq. 9), end-effector orientation (Eq. 10-11), and Cartesian stiffness and
damping (Sec. III-G).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tpgpt.transport.affine import AffineMap
from tpgpt.transport.gp import GaussianProcessRegressor
from tpgpt.transport.uncertainty import propagate_velocity_variance
from tpgpt.utils.rotations import project_to_so3

logger = logging.getLogger(__name__)


@dataclass
class DiffeomorphismReport:
    """Result of the property (ii) check of Sec. III-C."""

    determinants: np.ndarray
    fraction_positive: float
    min_determinant: float
    max_determinant: float
    consistent_sign: bool

    @property
    def satisfied(self) -> bool:
        """True when every sampled Jacobian determinant shares one sign.

        The paper's Proposition gives this as the necessary condition for
        ``phi`` to be a change of coordinates. It is sufficient only if checked
        over all possible keypoint sets, so this is a diagnostic, not a proof --
        which is exactly how the paper uses it ("checked experimentally a
        posteriori").
        """
        return self.consistent_sign

    def summary(self) -> str:
        return (
            f"det(J) in [{self.min_determinant:.4f}, {self.max_determinant:.4f}], "
            f"{100 * self.fraction_positive:.1f}% positive, "
            f"consistent sign: {self.consistent_sign}"
        )


class TransportMap:
    """Composite transportation map ``phi = gamma + psi . gamma``.

    Args:
        residual: Regressor for ``psi``. Any object exposing ``fit``,
            ``predict(X, return_std)`` and ``predict_gradient(X, return_std)``
            works -- :class:`~tpgpt.transport.gp.GaussianProcessRegressor` by
            default, or the sparse variant for large point clouds. Pass ``None``
            for an affine-only map, which is also what Sec. III-E-b describes
            when no keypoints are supplied: the zero-deformation prior applies
            and ``phi`` reduces to ``gamma``.
        affine: Custom :class:`~tpgpt.transport.affine.AffineMap`.
    """

    def __init__(
        self,
        residual: object | None = "default",
        affine: AffineMap | None = None,
    ):
        if residual == "default":
            residual = GaussianProcessRegressor()
        self.residual = residual
        self.affine = affine if affine is not None else AffineMap()
        self.source: np.ndarray | None = None
        self.target: np.ndarray | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, S: np.ndarray, T: np.ndarray) -> "TransportMap":
        """Fit ``gamma`` then ``psi`` on the affine residual.

        Args:
            S: ``(N, d)`` source keypoints.
            T: ``(N, d)`` target keypoints, paired elementwise with ``S``.
        """
        S = np.atleast_2d(np.asarray(S, dtype=float))
        T = np.atleast_2d(np.asarray(T, dtype=float))
        self.source, self.target = S, T

        self.affine.fit(S, T)
        if self.residual is not None:
            # Eq. (8): psi(gamma(S)) = T - gamma(S)
            S_affine = self.affine.predict(S)
            self.residual.fit(S_affine, T - S_affine)
        return self

    # ---------------------------------------------------- position labels
    def transport_positions(self, X: np.ndarray, return_std: bool = False):
        """``X_hat = phi(X)`` (paper Sec. III-F).

        Returns:
            ``(n, d)`` transported positions, and when ``return_std`` the
            ``(n,)`` positional standard deviation contributed by ``psi``.
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_affine = self.affine.predict(X)
        if self.residual is None:
            return (X_affine, np.zeros(X.shape[0])) if return_std else X_affine
        if return_std:
            delta, std = self.residual.predict(X_affine, return_std=True)
            return X_affine + delta, std
        return X_affine + self.residual.predict(X_affine)

    __call__ = transport_positions

    # --------------------------------------------------------- Jacobian
    def jacobian(self, X: np.ndarray, return_std: bool = False):
        """Jacobian of ``phi`` (paper Sec. III-F).

        ``J(x) = dgamma/dx + (dpsi/dgamma)(dgamma/dx) = (I + J_psi(gamma(x))) A``

        Note that ``psi`` and its derivative are evaluated at ``gamma(x)``, not
        at ``x``, per Eq. (2).

        Returns:
            ``(n, d, d)`` Jacobians, plus their ``(n, d, d)`` standard deviation
            when ``return_std``.
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        A = self.affine.jacobian()

        if self.residual is None:
            J = np.broadcast_to(A, (n, d, d)).copy()
            return (J, np.zeros((n, d, d))) if return_std else J

        X_affine = self.affine.predict(X)
        if return_std:
            J_psi, J_psi_std = self.residual.predict_gradient(X_affine, return_std=True)
        else:
            J_psi = self.residual.predict_gradient(X_affine)

        J = np.einsum("nik,kj->nij", J_psi, A) + A
        if not return_std:
            return J

        # A is deterministic, so variances propagate through it elementwise:
        # Var[J_ij] = sum_k Var[J_psi_ik] A_kj^2.
        J_var = np.einsum("nik,kj->nij", J_psi_std**2, A**2)
        return J, np.sqrt(np.maximum(J_var, 0.0))

    def orthogonal_jacobian(self, X: np.ndarray) -> np.ndarray:
        """``J_perp``: the Jacobian projected onto ``SO(3)`` (Sec. III-G)."""
        J = self.jacobian(X)
        return np.stack([project_to_so3(j) for j in J])

    # -------------------------------------------------- velocity labels
    def transport_velocities(
        self, X: np.ndarray, X_dot: np.ndarray, return_std: bool = False
    ):
        """Eq. (9): ``xdot_hat = J(x) xdot``.

        The paper deliberately does not differentiate the transported
        trajectory numerically -- labels are treated as independent
        state-action pairs, not as an ordered trajectory, so that interactively
        aggregated or edited labels transport correctly too (Sec. III-F).

        Returns:
            ``(n, d)`` transported velocities, plus their ``(n, d)`` standard
            deviation from Eq. (12) when ``return_std``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_dot = np.atleast_2d(np.asarray(X_dot, dtype=float))
        if X.shape != X_dot.shape:
            raise ValueError(f"X and X_dot must match, got {X.shape} vs {X_dot.shape}")

        if return_std:
            J, J_std = self.jacobian(X, return_std=True)
            V = np.einsum("nij,nj->ni", J, X_dot)
            var = propagate_velocity_variance(J_std, X_dot)
            return V, np.sqrt(np.maximum(var, 0.0))
        J = self.jacobian(X)
        return np.einsum("nij,nj->ni", J, X_dot)

    # ----------------------------------------------- orientation labels
    def transport_orientations(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Eq. (11): ``R_hat = J_perp R``.

        Derived in Eq. (10) by transporting the tip of an infinitesimal vector
        attached to the end effector and projecting the resulting Jacobian onto
        ``SO(3)`` so the result stays a proper rotation.

        Args:
            X: ``(n, 3)`` positions the orientations are attached to.
            R: ``(n, 3, 3)`` source rotation matrices.
        """
        R = np.asarray(R, dtype=float).reshape(-1, 3, 3)
        J_perp = self.orthogonal_jacobian(X)
        if J_perp.shape[0] != R.shape[0]:
            raise ValueError(f"batch mismatch: {J_perp.shape[0]} vs {R.shape[0]}")
        return np.einsum("nij,njk->nik", J_perp, R)

    # ---------------------------------- stiffness and damping labels
    def _congruence(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """``M_hat = J_perp M J_perp^T`` -- the congruence of Sec. III-G."""
        M = np.asarray(M, dtype=float).reshape(-1, 3, 3)
        J_perp = self.orthogonal_jacobian(X)
        if J_perp.shape[0] != M.shape[0]:
            raise ValueError(f"batch mismatch: {J_perp.shape[0]} vs {M.shape[0]}")
        return np.einsum("nij,njk,nlk->nil", J_perp, M, J_perp)

    def transport_stiffness(self, X: np.ndarray, K: np.ndarray) -> np.ndarray:
        """``K_hat = J_perp K J_perp^T`` (Sec. III-G).

        Follows from transporting the force the robot applies to the
        environment: ``F_hat = K_hat (J dx) = J (K dx)``. Because ``J_perp`` is
        orthogonal the transform is a rotation of the stiffness ellipsoid, so
        the eigenvalues -- the physical stiffness magnitudes -- are preserved.
        """
        return self._congruence(X, K)

    def transport_damping(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """``D_hat = J_perp D J_perp^T`` (Sec. III-G)."""
        return self._congruence(X, D)

    # ------------------------------------------------------ diagnostics
    def check_diffeomorphism(self, X: np.ndarray) -> DiffeomorphismReport:
        """Property (ii) of Sec. III-C: is ``det(J_phi)`` nonzero everywhere?

        The paper's sanity check on a transported policy is that the Jacobian
        determinant keeps one sign across all transported labels; where it does,
        the map is locally invertible and stability properties of the source
        dynamics carry over. Fig. 7 reports the fraction of labels with
        ``det(J) > 0`` as a quality measure of the generalisation.
        """
        J = self.jacobian(X)
        dets = np.linalg.det(J)
        positive = float(np.mean(dets > 0))
        return DiffeomorphismReport(
            determinants=dets,
            fraction_positive=positive,
            min_determinant=float(dets.min()),
            max_determinant=float(dets.max()),
            consistent_sign=bool(np.all(dets > 0) or np.all(dets < 0)),
        )

    def keypoint_residual(self) -> np.ndarray:
        """Per-keypoint error ``||phi(S) - T||`` -- property (i) of Sec. III-C."""
        self._check_fitted()
        return np.linalg.norm(self.transport_positions(self.source) - self.target, axis=1)

    def _check_fitted(self) -> None:
        if self.source is None:
            raise RuntimeError("TransportMap.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.source is None:
            return "TransportMap(unfitted)"
        return (
            f"TransportMap(N_keypoints={self.source.shape[0]}, "
            f"residual={type(self.residual).__name__ if self.residual else None})"
        )
