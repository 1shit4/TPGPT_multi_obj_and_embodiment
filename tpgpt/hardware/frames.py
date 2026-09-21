"""Rigid transforms and hand conventions for the real robot.

Three coordinate conventions meet on the hardware path, and every one of them
has already cost this project a silent error in simulation (``docs/sim/SIM_CLAUDE.md``,
"Keypoints live in the grasp convention"). They are made explicit here:

* **Frames.** Everything the planner and executor touch is expressed in the
  robot's ``base`` frame -- the frame CRISP's ``/target_pose`` and
  ``/current_pose`` use. The source demonstration is converted from the
  robosuite world into the *source* robot's base frame once, at export.
* **Grasp convention.** GraspGen-X and every keypoint cube use ``+X`` =
  closing direction, ``+Z`` = approach (:class:`tpgpt.sim.keypoints.GraspFrame`).
* **FR3 TCP convention.** ``fr3_hand_tcp`` sits 0.1034 m along ``fr3_hand``'s
  ``+Z`` with no rotation, and both fingers are prismatic along ``fr3_hand``'s
  ``+Y`` (``franka_description/end_effectors/common/franka_hand.xacro``). So
  its ``+Z`` is the approach and its ``+Y`` is the closing axis.

All transforms are 4x4 homogeneous matrices ``T_a_b`` taking coordinates in
frame ``b`` to frame ``a``: ``p_a = T_a_b @ p_b``.
"""

from __future__ import annotations

import numpy as np

from tpgpt.utils.rotations import matrix_to_quat, quat_to_matrix

#: Rotation taking grasp-convention axes to ``fr3_hand_tcp`` axes:
#: ``R_tcp = R_grasp @ GRASP_TO_FR3_TCP``.
#:
#: Columns are the TCP axes written in grasp coordinates. TCP ``+Y`` (fingers)
#: is grasp ``+X`` (closing); TCP ``+Z`` is grasp ``+Z`` (approach); TCP ``+X``
#: completes a right-handed frame, ``Y x Z = X_grasp x Z_grasp = -Y_grasp``.
#: That is a rotation of -90 degrees about the shared approach axis.
#:
#: A parallel jaw grips identically after a half turn about the approach, so a
#: sign error here would not drop the object -- it would command the wrist
#: 180 degrees away from the plan, which is what ``scene_keypoints`` resolves the
#: roll to avoid. Verify with the TCP axes marker in RViz before execution.
GRASP_TO_FR3_TCP = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def rotation_about_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(rotation=None, translation=None) -> np.ndarray:
    """Homogeneous transform from a ``(3, 3)`` rotation and ``(3,)`` translation."""
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def transform_from_xyz_quat(xyz, quat_xyzw) -> np.ndarray:
    """Transform from a translation and a ROS-ordered ``[x, y, z, w]`` quaternion."""
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quaternion has zero length")
    return make_transform(quat_to_matrix(q / norm, scalar_first=False)[0], xyz)


def transform_to_xyz_quat(T) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz, [x, y, z, w])`` of a transform."""
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), np.asarray(matrix_to_quat(T[:3, :3], scalar_first=False)).reshape(4)


def invert_transform(T) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def apply_to_points(T, points) -> np.ndarray:
    """Apply ``T`` to ``(N, 3)`` points."""
    T = np.asarray(T, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ T[:3, :3].T + T[:3, 3]


def apply_to_vectors(T, vectors) -> np.ndarray:
    """Rotate ``(N, 3)`` free vectors (velocities, axes) by ``T``'s rotation."""
    T = np.asarray(T, dtype=float)
    return np.atleast_2d(np.asarray(vectors, dtype=float)) @ T[:3, :3].T


def apply_to_rotations(T, rotations) -> np.ndarray:
    """Left-multiply ``(N, 3, 3)`` orientations by ``T``'s rotation."""
    R = np.asarray(T, dtype=float)[:3, :3]
    return np.einsum("ij,njk->nik", R, np.asarray(rotations, dtype=float).reshape(-1, 3, 3))


def apply_to_tensors(T, tensors) -> np.ndarray:
    """Congruence ``R M R^T`` for ``(N, 3, 3)`` stiffness or damping matrices.

    A stiffness is a bilinear form on displacements, so a change of frame
    rotates it on both sides. Rotating only one side would make it asymmetric,
    which ``log_cholesky`` would then silently symmetrise into a different gain.
    """
    R = np.asarray(T, dtype=float)[:3, :3]
    M = np.asarray(tensors, dtype=float).reshape(-1, 3, 3)
    return np.einsum("ij,njk,lk->nil", R, M, R)


def grasp_to_fr3_tcp(rotations) -> np.ndarray:
    """Grasp-convention orientations to ``fr3_hand_tcp`` orientations."""
    R = np.asarray(rotations, dtype=float)
    return np.einsum("...ij,jk->...ik", R, GRASP_TO_FR3_TCP)


def fr3_tcp_to_grasp(rotations) -> np.ndarray:
    """Inverse of :func:`grasp_to_fr3_tcp`."""
    R = np.asarray(rotations, dtype=float)
    return np.einsum("...ij,jk->...ik", R, GRASP_TO_FR3_TCP.T)


def slerp_rotation(R_a, R_b, s: float) -> np.ndarray:
    """Geodesic interpolation between two rotations, ``s`` in ``[0, 1]``."""
    from scipy.spatial.transform import Rotation, Slerp

    key = Rotation.from_matrix(np.stack([np.asarray(R_a), np.asarray(R_b)]))
    return Slerp([0.0, 1.0], key)([float(np.clip(s, 0.0, 1.0))]).as_matrix()[0]


def rotation_angle(R_a, R_b) -> float:
    """Geodesic angle between two rotations, in radians."""
    rel = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    return float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
