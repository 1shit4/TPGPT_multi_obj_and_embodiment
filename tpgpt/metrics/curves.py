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


def point_to_curve_distance(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest place on a polyline.

    The distance is to the **polyline**, not to its vertices: each segment is
    treated as a line segment and the query point is projected onto it, clamped
    to the segment's ends. Measuring to vertices instead over-reports by up to
    half the vertex spacing, which on a 200-label demonstration sampled at 20 Hz
    is a few millimetres -- the same size as the quantity being measured.

    Args:
        points: ``(n, d)`` query points.
        curve: ``(m, d)`` polyline vertices, in order.

    Returns:
        ``(n,)`` distances.
    """
    P = _as_curve(points)
    C = _as_curve(curve)
    if C.shape[0] == 1:
        return np.linalg.norm(P - C[0], axis=1)

    a, b = C[:-1], C[1:]                       # (m-1, d) segment ends
    ab = b - a
    denom = np.einsum("md,md->m", ab, ab)
    denom = np.where(denom > 1e-18, denom, 1.0)

    delta = P[:, None, :] - a[None, :, :]      # (n, m-1, d)
    t = np.clip(np.einsum("nmd,md->nm", delta, ab) / denom, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(P[:, None, :] - closest, axis=2).min(axis=1)


def path_deviation(points: np.ndarray, curve: np.ndarray) -> dict:
    """How far a sampled trajectory strays from a reference path.

    **This is the measurement to use for integration drift.** The natural
    temptation -- comparing how close each curve gets to some third point, such
    as the grasp -- is not a drift measurement at all: it is the difference of
    two separately minimised distances, it can come out negative, and the two
    minima need not occur at the same moment. This compares the curves to each
    other, pointwise, and cannot go negative.

    Timing is deliberately excluded. A trajectory that follows the reference
    path exactly but arrives late has zero deviation here, which is correct --
    that is a schedule problem, and mixing it in is what made the earlier
    numbers unreadable. Pair this with a phase-aware measure when the question
    is *when* rather than *where*.

    Args:
        points: ``(n, d)`` sampled trajectory, e.g. the integrated attractor.
        curve: ``(m, d)`` reference path, e.g. the transported labels.

    Returns:
        ``max``, ``median`` and ``mean`` deviation, and ``argmax``, the index of
        the sample that strayed furthest.
    """
    d = point_to_curve_distance(points, curve)
    return {
        "max": float(d.max()),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "argmax": int(d.argmax()),
    }
