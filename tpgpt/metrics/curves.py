"""Curve-similarity metrics (paper Sec. IV-B, ref. [55]).

The paper compares a rollout against a reference demonstration with five
measures: Frechet distance, area between the curves, dynamic time warping, final
position error, and the final approach angle. The first three come from
Jekel et al., ref. [55]; ``similaritymeasures`` is that paper's own
implementation, so it is used rather than re-derived.

These are provided now because the deferred Sec. IV baseline comparison needs
exactly this set, and because they are useful for reporting a transported
rollout against the demonstration it came from.
"""

from __future__ import annotations

import numpy as np
import similaritymeasures


def _as_curve(x: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(x, dtype=float))


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Frechet distance.

    Time-agnostic: the largest of the distances between the closest pairs along
    the two curves, so it penalises the worst local mismatch.
    """
    return float(similaritymeasures.frechet_dist(_as_curve(a), _as_curve(b)))


def area_between_curves(a: np.ndarray, b: np.ndarray) -> float:
    """Area enclosed between two planar curves.

    Only defined in 2-D; project first if the curves are spatial.
    """
    a, b = _as_curve(a), _as_curve(b)
    if a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("area between curves is only defined for 2-D curves")
    return float(similaritymeasures.area_between_two_curves(a, b))


def dynamic_time_warping(a: np.ndarray, b: np.ndarray) -> float:
    """Cumulative DTW distance between two curves."""
    return float(similaritymeasures.dtw(_as_curve(a), _as_curve(b))[0])


def final_position_error(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between the two curves' endpoints."""
    return float(np.linalg.norm(_as_curve(a)[-1] - _as_curve(b)[-1]))


def final_approach_angle(a: np.ndarray, b: np.ndarray, n_points: int = 10) -> float:
    """Angle (radians) between the two curves' final approach directions.

    The paper's "docking" angle: a low value means the rollout reaches the goal
    from the same direction as the demonstration, not merely at the same point.
    """
    def direction(curve):
        curve = _as_curve(curve)
        k = min(n_points, len(curve) - 1)
        if k < 1:
            return np.zeros(curve.shape[1])
        vector = curve[-1] - curve[-1 - k]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-12 else np.zeros_like(vector)

    da, db = direction(a), direction(b)
    if not np.any(da) or not np.any(db):
        return float("nan")
    return float(np.arccos(np.clip(np.dot(da, db), -1.0, 1.0)))


def compare_trajectories(rollout: np.ndarray, reference: np.ndarray) -> dict:
    """All of the paper's Sec. IV-B measures at once.

    ``area_between_curves`` is skipped for spatial curves, where it is undefined.
    """
    metrics = {
        "frechet": frechet_distance(rollout, reference),
        "dtw": dynamic_time_warping(rollout, reference),
        "final_position_error": final_position_error(rollout, reference),
        "final_approach_angle": final_approach_angle(rollout, reference),
    }
    if _as_curve(rollout).shape[1] == 2:
        metrics["area"] = area_between_curves(rollout, reference)
    return metrics
