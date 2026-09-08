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

from tpgpt.transport.affine import AffineMap, IdentityAffine
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


class BaseTransportMap:
    """Everything a transportation map can do once it can be *evaluated*.

    Every method here is written purely in terms of :meth:`transport_positions`,
    :meth:`jacobian` and the fitted ``source`` / ``target`` keypoints, so a
    subclass supplying those three gets Eqs. 9, 11 and the Sec. III-G congruence
    for free. That matters because the alternative is two implementations of the
    paper's theory that can drift apart.

    Subclasses: :class:`TransportMap` (a single affine-plus-residual map) and
    :class:`ComposedMap` (one map applied after another).
    """

    source: np.ndarray | None = None
    target: np.ndarray | None = None

    def transport_positions(self, X, return_std: bool = False):  # pragma: no cover
        raise NotImplementedError

    def jacobian(self, X, return_std: bool = False):  # pragma: no cover
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if self.source is None:
            raise RuntimeError("the map must be fitted before use")

    __call__ = transport_positions

    def orthogonal_jacobian(self, X: np.ndarray) -> np.ndarray:
        """``J_perp``: the Jacobian projected onto ``SO(3)`` (Sec. III-G)."""
        J = self.jacobian(X)
        return np.stack([project_to_so3(j) for j in J])

    def transport_velocities(
        self, X: np.ndarray, X_dot: np.ndarray, return_std: bool = False
    ):
        """Eq. (9): ``xdot_hat = J(x) xdot``.

        The paper deliberately does not differentiate the transported trajectory
        numerically -- labels are treated as independent state-action pairs, not
        as an ordered trajectory, so that interactively aggregated or edited
        labels transport correctly too (Sec. III-F).
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
        return np.einsum("nij,nj->ni", self.jacobian(X), X_dot)

    def transport_orientations(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Eq. (11): ``R_hat = J_perp R``."""
        R = np.asarray(R, dtype=float).reshape(-1, 3, 3)
        J_perp = self.orthogonal_jacobian(X)
        if J_perp.shape[0] != R.shape[0]:
            raise ValueError(f"batch mismatch: {J_perp.shape[0]} vs {R.shape[0]}")
        return np.einsum("nij,njk->nik", J_perp, R)

    def _congruence(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """``M_hat = J_perp M J_perp^T`` -- the congruence of Sec. III-G."""
        M = np.asarray(M, dtype=float).reshape(-1, 3, 3)
        J_perp = self.orthogonal_jacobian(X)
        if J_perp.shape[0] != M.shape[0]:
            raise ValueError(f"batch mismatch: {J_perp.shape[0]} vs {M.shape[0]}")
        return np.einsum("nij,njk,nlk->nil", J_perp, M, J_perp)

    def transport_stiffness(self, X: np.ndarray, K: np.ndarray) -> np.ndarray:
        """``K_hat = J_perp K J_perp^T`` (Sec. III-G)."""
        return self._congruence(X, K)

    def transport_damping(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """``D_hat = J_perp D J_perp^T`` (Sec. III-G)."""
        return self._congruence(X, D)

    def check_diffeomorphism(self, X: np.ndarray) -> DiffeomorphismReport:
        """Property (ii) of Sec. III-C: is ``det(J_phi)`` nonzero everywhere?"""
        dets = np.linalg.det(self.jacobian(X))
        return DiffeomorphismReport(
            determinants=dets,
            fraction_positive=float(np.mean(dets > 0)),
            min_determinant=float(dets.min()),
            max_determinant=float(dets.max()),
            consistent_sign=bool(np.all(dets > 0) or np.all(dets < 0)),
        )

    def keypoint_residual(self) -> np.ndarray:
        """Per-keypoint error ``||phi(S) - T||`` -- property (i) of Sec. III-C."""
        self._check_fitted()
        return np.linalg.norm(self.transport_positions(self.source) - self.target, axis=1)


class TransportMap(BaseTransportMap):
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

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.source is None:
            return "TransportMap(unfitted)"
        return (
            f"TransportMap(N_keypoints={self.source.shape[0]}, "
            f"residual={type(self.residual).__name__ if self.residual else None})"
        )


class ComposedMap(BaseTransportMap):
    """Two transportation maps applied in sequence, ``phi(x) = phi_2(phi_1(x))``.

    Built for one specific job: letting a **global** stage carry the trajectory
    while a **local** stage pins the jaw contacts, without the local stage's
    deformation leaking into the rest of the path.

    **Why composition rather than a weighted blend.** A convex blend
    ``w phi_2 + (1 - w) phi_1`` has Jacobian
    ``w J_2 + (1 - w) J_1 + (phi_2 - phi_1) (grad w)^T`` -- a rank-one cross term
    whose size is the disagreement between the two maps divided by the distance
    over which the weight turns over. It can flip the determinant on its own, and
    the only way to find out is to sample. Under composition the Jacobians simply
    multiply, so

        ``det(J) = det(J_2) det(J_1)``

    and each stage can be checked **independently**: if both are diffeomorphisms
    the composite is one, with no interaction to reason about. That is the whole
    argument for this class, and :meth:`stage_reports` exposes the per-stage
    check so the guarantee is usable rather than merely true.

    Exactly two stages, because a composed map satisfies the interface it
    consumes: three is ``ComposedMap(ComposedMap(a, b), c)``.

    **The keypoint residual is honest about a real cost.** Stage 2 hits its own
    keypoints exactly, but stage 1's no longer land exactly on their targets --
    ``phi(S_1) = phi_2(T_1)``, which is ``T_1`` only if stage 2 is the identity
    there. :meth:`keypoint_residual` reports over *both* sets, so that
    disturbance is visible instead of silently absorbed. Keeping stage 2 local is
    exactly what keeps it small.

    Args:
        first: The global stage, already fitted.
        second: The local correction, already fitted **in the first stage's
            output space** -- see :func:`fit_local_correction`.
    """

    def __init__(self, first: BaseTransportMap, second: BaseTransportMap):
        self.first = first
        self.second = second
        self.source = np.vstack([first.source, second.first_source]) if getattr(
            second, "first_source", None
        ) is not None else first.source
        self.target = np.vstack([first.target, second.target]) if getattr(
            second, "first_source", None
        ) is not None else first.target

    def transport_positions(self, X: np.ndarray, return_std: bool = False):
        """``phi_2(phi_1(x))``.

        The standard deviations of the two stages are combined by linearising
        stage 2 about stage 1's output and treating the two GP posteriors as
        independent, then collapsing to the largest marginal:

            ``s = sqrt(s_1^2 sigma_max(J_2)^2 + s_2^2)``

        This is an **upper bound**, and it reduces to ``hypot(s_1, s_2)`` when
        stage 2 is near the identity -- which is the regime a local correction
        lives in. Documented as a bound rather than presented as exact, following
        the precedent of :func:`~tpgpt.transport.uncertainty.propagate_velocity_variance`.
        """
        if not return_std:
            return self.second.transport_positions(self.first.transport_positions(X))
        Y1, s1 = self.first.transport_positions(X, return_std=True)
        Y, s2 = self.second.transport_positions(Y1, return_std=True)
        gain = np.linalg.svd(self.second.jacobian(Y1), compute_uv=False)[:, 0]
        return Y, np.sqrt(np.maximum((s1 * gain) ** 2 + s2**2, 0.0))

    def jacobian(self, X: np.ndarray, return_std: bool = False):
        """Chain rule: ``J = J_2(phi_1(x)) J_1(x)``.

        For the variance, the two stages' Jacobian entries are treated as
        independent, which gives the exact variance of a product of independent
        matrices:

            ``Var[J_ij] = sum_k ( J_2ik^2 Var[J_1kj] + Var[J_2ik] J_1kj^2
                                  + Var[J_2ik] Var[J_1kj] )``

        With a deterministic second stage this collapses to
        ``sum_k J_2ik^2 Var[J_1kj]``, which is exactly what
        :meth:`TransportMap.jacobian` already does for its own affine stage --
        the reduction is the best available check on the rule. What it ignores is
        that ``J_2`` is itself evaluated at an uncertain ``phi_1(x)``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y1 = self.first.transport_positions(X)
        if not return_std:
            return np.einsum(
                "nik,nkj->nij", self.second.jacobian(Y1), self.first.jacobian(X)
            )
        J1, J1_std = self.first.jacobian(X, return_std=True)
        J2, J2_std = self.second.jacobian(Y1, return_std=True)
        J = np.einsum("nik,nkj->nij", J2, J1)
        var = (
            np.einsum("nik,nkj->nij", J2**2, J1_std**2)
            + np.einsum("nik,nkj->nij", J2_std**2, J1**2)
            + np.einsum("nik,nkj->nij", J2_std**2, J1_std**2)
        )
        return J, np.sqrt(np.maximum(var, 0.0))

    def stage_reports(self, X: np.ndarray) -> tuple[DiffeomorphismReport, DiffeomorphismReport]:
        """Property (ii) checked on each stage separately.

        Determinants multiply, but a *minimum over a path* does not distribute
        over a product -- the two minima occur at different points -- so the
        per-stage reports carry information the composite's cannot. They are also
        the thing that makes the composition verifiable in advance: if both
        stages are diffeomorphisms, so is the composite.
        """
        return (
            self.first.check_diffeomorphism(X),
            self.second.check_diffeomorphism(self.first.transport_positions(X)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ComposedMap({self.first!r} -> {self.second!r})"


def fit_local_correction(
    first: BaseTransportMap,
    source_points: np.ndarray,
    target_points: np.ndarray,
    locality: float = 0.03,
) -> ComposedMap:
    """Pin extra points on top of a fitted map, without disturbing the rest.

    Fits a second stage on the residual **after** the first: the source points
    are pushed through ``first`` and the second stage moves them the remaining
    distance to their targets. Because ``first`` has already done most of the
    work, that correction is small and the second stage is near the identity --
    the regime in which a local map is safe.

    Three things are pinned deliberately rather than left to defaults, and each
    one is a silent failure if it is not:

    **The affine stage is forced to the identity** and is not an argument, so a
    caller cannot reintroduce the failure it prevents. With a real
    :class:`~tpgpt.transport.affine.AffineMap`, a correction whose points all
    move the same way has that displacement absorbed into ``A`` and ``T_bar``,
    which act *globally*: measured, a 12 mm correction at four contacts moved
    points 12 mm at 80 cm away, and the residual's length scale made no
    difference at all because the residual had nothing left to fit.

    **The length scale is fixed at** ``locality`` **and optimisation is off.**
    Left to fit, ``length_scale_bounds="auto"`` derives its bounds from the
    correction points alone and the marginal likelihood puts the value wherever
    it lands -- so the locality radius would be an accident of the contact
    spacing rather than a design parameter. With the affine suppressed, this
    length scale *is* the radius: measured, a 30 mm scale leaks 0.41 mm at 10 cm
    and a 60 mm scale 0.09 mm at 20 cm.

    **The signal variance is set from the correction's own size.** With
    ``signal_variance=None`` the GP initialises from the geometric mean of its
    bounds, roughly ``3.2e-4 * mean(||Y||^2)``. For a half-millimetre correction
    that is about ``8e-11`` -- **below the default noise variance of 1e-10** -- so
    the GP would smooth the correction away entirely and the stage would silently
    do nothing.

    Raises:
        ValueError: if the correction is larger than ``locality``. A "local"
            stage asked to move points further than its own reach overshoots
            violently between them -- measured, asking a 30 mm-locality stage for
            a 411 mm correction produced a 336 mm excursion 5 cm away while still
            interpolating its four points to machine precision. That is the
            ``NOISE_VARIANCE`` failure of
            :class:`~tpgpt.policy.gp_policy.GPPolicy` in another guise: exact on
            the data, explosive just off it. The keypoint check below cannot
            catch it, which is exactly why this one exists.
        RuntimeError: if the fitted stage does not interpolate its own points to
            1 um. Returning a map that quietly fails property (i) would be worse
            than failing loudly.
    """
    source_points = np.atleast_2d(np.asarray(source_points, dtype=float))
    target_points = np.atleast_2d(np.asarray(target_points, dtype=float))
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"source {source_points.shape} and target {target_points.shape} must match"
        )
    moved = first.transport_positions(source_points)
    correction = target_points - moved
    reach = float(np.linalg.norm(correction, axis=1).max())
    if reach > locality:
        raise ValueError(
            f"the correction is {reach * 1000:.1f} mm but the locality radius is "
            f"{locality * 1000:.1f} mm, so it is not a local correction: a "
            "second stage asked to move points further than it is allowed to "
            "reach overshoots wildly between them. Either the first stage is "
            "not already carrying these points to roughly the right place, or "
            "the locality is too small for the disagreement."
        )
    signal = max(float(np.mean(np.sum(correction**2, axis=1))), 1e-6)

    second = TransportMap(
        residual=GaussianProcessRegressor(
            length_scale=float(locality),
            signal_variance=signal,
            optimize=False,
        ),
        affine=IdentityAffine(),
    ).fit(moved, target_points)
    residual = float(second.keypoint_residual().max())
    if residual > 1e-6:
        raise RuntimeError(
            f"the local correction does not interpolate its own points "
            f"({residual * 1000:.3f} mm); property (i) of Sec. III-C fails"
        )
    # Remembered so the composite can report the residual over *both* stages'
    # keypoints, which is where the cost of composing becomes visible.
    second.first_source = source_points
    return ComposedMap(first, second)
