"""Continuous representation of rotations for regression.

Rotation matrices cannot be averaged or interpolated entrywise and stay in
``SO(3)``, and quaternions are double-covering, so a regressor trained on them
can be pulled towards the wrong hemisphere. The six-dimensional representation
of Zhou et al. -- keep the first two columns of ``R``, recover the third by
Gram-Schmidt -- is continuous and surjective onto ``SO(3)``, so any regressed
value maps back to a valid rotation.
"""

from __future__ import annotations

import numpy as np


def rotation_to_6d(R: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` rotations to their ``(n, 6)`` continuous representation."""
    R = np.asarray(R, dtype=float).reshape(-1, 3, 3)
    return R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def rotation_from_6d(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rotation_to_6d` via Gram-Schmidt; always in ``SO(3)``."""
    v = np.atleast_2d(np.asarray(v, dtype=float)).reshape(-1, 2, 3)
    a1, a2 = v[:, 0], v[:, 1]

    n1 = np.linalg.norm(a1, axis=1, keepdims=True)
    # Degenerate input (e.g. a zero prediction) falls back to a valid basis.
    b1 = np.where(n1 > 1e-12, a1 / np.maximum(n1, 1e-12), np.array([1.0, 0.0, 0.0]))

    proj = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    n2 = np.linalg.norm(proj, axis=1, keepdims=True)
    fallback = np.cross(b1, np.array([0.0, 0.0, 1.0]))
    fb_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = np.where(
        fb_norm > 1e-6,
        fallback / np.maximum(fb_norm, 1e-12),
        np.array([0.0, 1.0, 0.0]),
    )
    b2 = np.where(n2 > 1e-12, proj / np.maximum(n2, 1e-12), fallback)

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=2)
