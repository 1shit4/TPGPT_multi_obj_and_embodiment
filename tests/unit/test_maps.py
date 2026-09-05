"""The transportation map phi -- paper Sec. III-C to III-H."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import is_rotation


def _jacobian_by_finite_differences(tm, X, h=1e-6):
    J = np.zeros((X.shape[0], 3, 3))
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        J[:, :, d] = (
            tm.transport_positions(X + e) - tm.transport_positions(X - e)
        ) / (2 * h)
    return J


# ------------------------------------------------------------ property (i)
def test_property_i_keypoints_are_matched_rigid(rigid_keypoints):
    """Sec. III-C property (i): T = phi(S). Exact for a rigid target."""
    S, T, _, _ = rigid_keypoints
    tm = TransportMap().fit(S, T)
    assert tm.keypoint_residual().max() < 1e-12


def test_property_i_keypoints_are_matched_curved(curved_keypoints):
    """Property (i) under a genuinely nonlinear target.

    The residual is bounded by GP conditioning rather than machine precision;
    10 um is orders of magnitude below any real keypoint detector.
    """
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    assert tm.keypoint_residual().max() < 1e-4


# ----------------------------------------------------------- property (ii)
def test_property_ii_determinant_is_positive(curved_keypoints, rng):
    """Sec. III-C property (ii) and the Fig. 7 'Jac > 0' statistic."""
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    report = tm.check_diffeomorphism(rng.uniform(-0.2, 0.2, (40, 3)))
    assert report.fraction_positive == 1.0
    assert report.consistent_sign and report.satisfied
    assert report.min_determinant > 0
    assert "det(J)" in report.summary()


# ------------------------------------------------------ the rigid limit
def test_rigid_target_reduces_to_the_affine_map(rigid_keypoints, rng):
    """With nothing nonlinear to explain, phi must equal gamma everywhere."""
    S, T, A, t = rigid_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.3, 0.3, (20, 3))
    assert np.allclose(tm.transport_positions(X), X @ A.T + t, atol=1e-8)
    assert np.allclose(tm.jacobian(X), A, atol=1e-8)


def test_rigid_target_transports_velocity_by_the_rotation(rigid_keypoints, rng):
    S, T, A, _ = rigid_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.3, 0.3, (20, 3))
    V = rng.normal(0, 0.1, (20, 3))
    assert np.allclose(tm.transport_velocities(X, V), V @ A.T, atol=1e-8)


def test_rigid_target_transports_orientation_by_the_rotation(rigid_keypoints, rng):
    """Eq. (11) must include the affine rotation.

    The prototype projected only the residual Jacobian onto SO(3), silently
    dropping A, which is wrong for any target involving rotation.
    """
    S, T, A, _ = rigid_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.3, 0.3, (15, 3))
    R = Rotation.random(15, random_state=5).as_matrix()
    assert np.allclose(
        tm.transport_orientations(X, R), np.einsum("ij,njk->nik", A, R), atol=1e-7
    )


def test_rigid_target_transports_stiffness_by_congruence(rigid_keypoints, rng):
    S, T, A, _ = rigid_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.3, 0.3, (10, 3))
    K = np.stack([np.diag([600.0, 400.0, 200.0])] * 10)
    assert np.allclose(
        tm.transport_stiffness(X, K), np.einsum("ij,njk,lk->nil", A, K, A), atol=1e-6
    )


# ---------------------------------------------------- label transformations
def test_transported_orientations_are_proper_rotations(curved_keypoints, rng):
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.2, 0.2, (20, 3))
    R = Rotation.random(20, random_state=7).as_matrix()
    assert all(is_rotation(r, atol=1e-8) for r in tm.transport_orientations(X, R))


def test_transported_stiffness_is_symmetric_and_preserves_eigenvalues(
    curved_keypoints, rng
):
    """J_perp is orthogonal, so the congruence only rotates the ellipsoid."""
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.2, 0.2, (12, 3))
    K = np.stack([np.diag([800.0, 300.0, 150.0])] * 12)
    K_hat = tm.transport_stiffness(X, K)
    assert np.allclose(K_hat, K_hat.transpose(0, 2, 1), atol=1e-9)
    assert np.allclose(np.sort(np.linalg.eigvalsh(K_hat), axis=1),
                       [150.0, 300.0, 800.0], atol=1e-8)
    assert np.linalg.eigvalsh(K_hat).min() > 0  # stays positive definite


def test_damping_uses_the_same_congruence(curved_keypoints, rng):
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.2, 0.2, (6, 3))
    M = np.stack([np.diag([40.0, 30.0, 20.0])] * 6)
    assert np.allclose(tm.transport_stiffness(X, M), tm.transport_damping(X, M))


def test_jacobian_matches_finite_differences(curved_keypoints, rng):
    """Eq. (9): J = (I + J_psi(gamma(x))) A, with psi evaluated at gamma(x)."""
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.2, 0.2, (20, 3))
    assert np.allclose(tm.jacobian(X), _jacobian_by_finite_differences(tm, X), atol=1e-6)


def test_velocity_uncertainty_follows_equation_12(curved_keypoints, rng):
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    X = rng.uniform(-0.2, 0.2, (10, 3))
    V = rng.normal(0, 0.1, (10, 3))
    _, J_std = tm.jacobian(X, return_std=True)
    _, v_std = tm.transport_velocities(X, V, return_std=True)
    expected = np.sqrt(np.einsum("nod,nd->no", J_std**2, V**2))
    assert np.allclose(v_std, expected)


# ------------------------------------------------------------- uncertainty
def test_uncertainty_grows_away_from_the_keypoints(curved_keypoints):
    """Paper Fig. 5 left: transport uncertainty grows with distance."""
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    distances = [0.0, 0.3, 0.6, 1.0]
    stds = [
        tm.transport_positions(np.array([[d, d, d]]), return_std=True)[1][0]
        for d in distances
    ]
    assert all(b > a for a, b in zip(stds, stds[1:])), stds


def test_map_reduces_to_affine_far_from_the_keypoints(curved_keypoints):
    """Sec. III-E-b / Fig. 2d: the zero-mean prior makes phi -> gamma o.o.d."""
    S, T = curved_keypoints
    tm = TransportMap().fit(S, T)
    far = np.array([[20.0, 20.0, 20.0]])
    assert np.allclose(tm.transport_positions(far), tm.affine.predict(far), atol=1e-9)
    assert np.allclose(tm.jacobian(far)[0], tm.affine.jacobian(), atol=1e-9)


# ----------------------------------------------------------- affine-only map
def test_affine_only_map_is_the_zero_deformation_prior(rigid_keypoints, rng):
    """Sec. III-E-b: with no keypoint deformation, phi is exactly gamma."""
    S, T, _, _ = rigid_keypoints
    tm = TransportMap(residual=None).fit(S, T)
    X = rng.uniform(-0.3, 0.3, (10, 3))
    assert np.allclose(tm.transport_positions(X), tm.affine.predict(X))
    assert np.allclose(tm.jacobian(X, return_std=True)[1], 0.0)
    assert np.allclose(tm.transport_positions(X, return_std=True)[1], 0.0)


def test_rejects_mismatched_velocity_shapes(rigid_keypoints):
    S, T, _, _ = rigid_keypoints
    tm = TransportMap().fit(S, T)
    with pytest.raises(ValueError, match="must match"):
        tm.transport_velocities(np.zeros((3, 3)), np.zeros((4, 3)))


def test_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        TransportMap().transport_positions(np.zeros((1, 3)))
