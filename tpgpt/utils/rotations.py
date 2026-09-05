"""Rotation helpers shared across the transportation and simulation code.

The paper works with orientation labels in ``SO(3)`` (Sec. III-A) but the
simulator reports quaternions, so conversions live here rather than being
re-derived at every call site.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation


def project_to_so3(J: np.ndarray) -> np.ndarray:
    """Project a square matrix onto ``SO(3)`` (paper Sec. III-G).

    The paper defines ``J_perp`` as the orthogonal factor of the polar
    decomposition of ``J``, with ``det(J_perp) = +1`` enforced so the result is
    a proper rotation rather than a reflection.

    Args:
        J: ``(d, d)`` matrix, typically a transportation-map Jacobian.

    Returns:
        ``(d, d)`` orthogonal matrix with determinant ``+1``.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {J.shape}")

    U, _ = polar(J, side="right")

    # ``polar`` returns the closest orthogonal matrix, which is a reflection
    # when det(J) < 0. Flipping the singular vector with the smallest singular
    # value gives the closest *proper* rotation (Kabsch/Umeyama correction).
    if np.linalg.det(U) < 0:
        Uu, _, Vt = np.linalg.svd(J)
        Vt[-1, :] *= -1
        U = Uu @ Vt
    return U


def project_batch_to_so3(J: np.ndarray) -> np.ndarray:
    """Vectorised :func:`project_to_so3` over a leading batch axis."""
    J = np.asarray(J, dtype=float)
    return np.stack([project_to_so3(j) for j in J.reshape(-1, *J.shape[-2:])]).reshape(
        J.shape
    )


def quat_to_matrix(quat: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """Convert quaternions to rotation matrices.

    Args:
        quat: ``(4,)`` or ``(N, 4)`` quaternions.
        scalar_first: ``True`` for ``wxyz`` (MuJoCo native), ``False`` for
            ``xyzw`` (scipy / robosuite observation convention).
    """
    quat = np.atleast_2d(np.asarray(quat, dtype=float))
    if scalar_first:
        quat = quat[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat).as_matrix()


def matrix_to_quat(mat: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """Inverse of :func:`quat_to_matrix`; returns ``(N, 4)``."""
    mat = np.asarray(mat, dtype=float).reshape(-1, 3, 3)
    quat = Rotation.from_matrix(mat).as_quat()
    if scalar_first:
        quat = quat[:, [3, 0, 1, 2]]
    return quat


def rotation_geodesic(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between batches of rotation matrices."""
    R_a = np.asarray(R_a, dtype=float).reshape(-1, 3, 3)
    R_b = np.asarray(R_b, dtype=float).reshape(-1, 3, 3)
    rel = np.einsum("nij,nkj->nik", R_a, R_b)
    trace = np.trace(rel, axis1=1, axis2=2)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def is_rotation(R: np.ndarray, atol: float = 1e-8) -> bool:
    """True if ``R`` is orthonormal with determinant ``+1``."""
    R = np.asarray(R, dtype=float)
    d = R.shape[-1]
    orthonormal = np.allclose(R.T @ R, np.eye(d), atol=atol)
    return bool(orthonormal and abs(np.linalg.det(R) - 1.0) < atol)
