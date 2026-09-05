"""Affine component gamma -- paper Sec. III-E-a, Eqs. (4)-(7)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.transport.affine import AffineMap


def test_recovers_exact_rigid_transform(rigid_keypoints):
    S, T, A, t = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert np.allclose(m.A, A, atol=1e-9)
    assert np.allclose(m.predict(S), T, atol=1e-9)


def test_maps_source_centroid_to_target_centroid(rigid_keypoints):
    """Eq. (7) is built around the centroids, so this must hold exactly."""
    S, T, _, _ = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert np.allclose(m.predict(S.mean(axis=0)), T.mean(axis=0), atol=1e-12)


def test_rejects_reflections(rigid_keypoints):
    """A mirrored target must still yield a proper rotation (det = +1)."""
    S, _, _, t = rigid_keypoints
    T_mirrored = S @ np.diag([1.0, 1.0, -1.0]) + t
    m = AffineMap().fit(S, T_mirrored)
    assert np.linalg.det(m.A) == pytest.approx(1.0, abs=1e-9)


def test_rank_deficient_covariance_falls_back_to_identity():
    """Sec. III-E-a: det(Sigma) == 0 leaves the rotation undetermined."""
    rng = np.random.default_rng(0)
    S = np.c_[rng.uniform(-1, 1, (8, 2)), np.zeros(8)]  # coplanar
    m = AffineMap().fit(S, S + np.array([0.1, 0.2, 0.3]))
    assert m.rank_deficient
    assert np.allclose(m.A, np.eye(3))


def test_reduces_distance_to_target(rigid_keypoints):
    """Paper Sec. III-E-a: ||T - gamma(S)|| <= ||T - S||."""
    S, T, _, _ = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert np.linalg.norm(T - m.predict(S)) <= np.linalg.norm(T - S) + 1e-12


def test_jacobian_is_the_constant_rotation(rigid_keypoints):
    S, T, A, _ = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert np.allclose(m.jacobian(), A, atol=1e-9)
    assert np.allclose(m.jacobian(np.zeros(3)), m.jacobian(np.ones(3)))


def test_inverse_roundtrip(rigid_keypoints):
    S, T, _, _ = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert np.allclose(m.inverse(m.predict(S)), S, atol=1e-10)


def test_shape_preservation(rigid_keypoints):
    S, T, _, _ = rigid_keypoints
    m = AffineMap().fit(S, T)
    assert m.predict(S[0]).shape == (3,)
    assert m.predict(S).shape == S.shape


def test_works_in_two_dimensions():
    rng = np.random.default_rng(0)
    S = rng.uniform(-1, 1, (10, 2))
    theta = 0.6
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = AffineMap().fit(S, S @ A.T + np.array([1.0, -2.0]))
    assert np.allclose(m.A, A, atol=1e-9)


def test_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        AffineMap().predict(np.zeros(3))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shape"):
        AffineMap().fit(np.zeros((4, 3)), np.zeros((5, 3)))
