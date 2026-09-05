"""Rotation utilities used by the orientation and stiffness transport."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.utils.rotations import (
    is_rotation,
    matrix_to_quat,
    project_batch_to_so3,
    project_to_so3,
    quat_to_matrix,
    rotation_geodesic,
)


def test_projection_is_identity_on_rotations():
    R = Rotation.random(random_state=0).as_matrix()
    assert np.allclose(project_to_so3(R), R, atol=1e-12)


def test_projection_of_a_scaled_rotation_recovers_it():
    """Polar decomposition strips the symmetric stretch factor."""
    R = Rotation.random(random_state=1).as_matrix()
    assert np.allclose(project_to_so3(R @ np.diag([1.4, 0.7, 1.1])), R, atol=1e-9)


def test_projection_enforces_positive_determinant():
    """Sec. III-G requires det(J_perp) = +1, so reflections must be corrected."""
    R = Rotation.random(random_state=2).as_matrix()
    P = project_to_so3(R @ np.diag([1.0, 1.0, -1.0]))
    assert np.linalg.det(P) == pytest.approx(1.0, abs=1e-9)
    assert is_rotation(P)


def test_projection_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        project_to_so3(np.zeros((3, 2)))


def test_batch_projection_matches_the_scalar_version():
    J = np.stack([Rotation.random(4, random_state=3).as_matrix()[i] @ np.diag([1.2, 0.9, 1.0])
                  for i in range(4)])
    assert np.allclose(project_batch_to_so3(J), np.stack([project_to_so3(j) for j in J]))


@pytest.mark.parametrize("scalar_first", [False, True])
def test_quaternion_roundtrip(scalar_first):
    R = Rotation.random(5, random_state=4).as_matrix()
    q = matrix_to_quat(R, scalar_first=scalar_first)
    assert np.allclose(quat_to_matrix(q, scalar_first=scalar_first), R, atol=1e-12)


def test_quaternion_conventions_differ():
    """xyzw vs wxyz is a real trap: robosuite reports xyzw, MuJoCo stores wxyz."""
    R = Rotation.random(random_state=5).as_matrix()
    assert not np.allclose(matrix_to_quat(R), matrix_to_quat(R, scalar_first=True))


def test_geodesic_is_zero_for_identical_rotations():
    R = Rotation.random(4, random_state=6).as_matrix()
    assert np.allclose(rotation_geodesic(R, R), 0.0, atol=1e-6)


def test_geodesic_recovers_a_known_angle():
    R_a = np.eye(3)[None]
    R_b = Rotation.from_euler("z", 30, degrees=True).as_matrix()[None]
    assert rotation_geodesic(R_a, R_b)[0] == pytest.approx(np.deg2rad(30), abs=1e-9)


def test_is_rotation_rejects_scaled_matrices():
    assert not is_rotation(2.0 * np.eye(3))
    assert not is_rotation(np.diag([1.0, 1.0, -1.0]))
