"""What a transportation map does to *orientation*, as opposed to position.

Every other diagnostic in this project measures where the warped path goes. None
of them measures which way the hand is pointing when it gets there, and that gap
is why two keypoint designs that behave very differently could not be told apart.

The quantity behind both metrics here is ``J_perp``, the **orthogonal Jacobian**
-- the rotation part of the map's local linearisation
(:meth:`~tpgpt.transport.maps.TransportMap.orthogonal_jacobian`). Eq. 11 of the
paper transports the end-effector orientation with it, ``R_hat = J_perp R``, so
``J_perp`` is simultaneously:

* the rotation applied to the **commanded hand orientation**, and
* the rotation applied to **every other direction at that point**, including the
  world's vertical.

Those are the same matrix, which is the whole point. A map cannot be made to
rotate the gripper without also rotating the space around it, so "did the hand
arrive at the right angle" and "did the map tilt the world" are two readings of
one number. Measuring only the first hides the cost; measuring only the second
hides the benefit. Measured on a target grasp tilted 30 degrees out of plane, the
exchange rate is one-for-one: a scheme that recovers 30 degrees of gripper
orientation tilts the vertical by 30 degrees, and one that recovers 7.3 degrees
tilts it by 22.7.

:func:`orientation_transport_error` reads the first.
:func:`vertical_tilt`, :func:`tilt_profile` and :func:`lift_deviation` read the
second.

**Why the tilt is a cost and not an inconsistency.** ``J_perp`` does not know that
gravity is special. If it rotates the hand by 30 degrees at the grasp it also
rotates the demonstration's straight-up lift by 30 degrees, turning a vertical
lift into a diagonal one. Physics still applies gravity in the world frame -- the
simulator is not being lied to -- but the *path* now asks the arm to travel
somewhere the demonstration never went, which costs clearance and reachability.
That is what :func:`lift_deviation` puts a number on.

**The jaw symmetry is not optional.** A parallel jaw closing along ``+c`` and
along ``-c`` is one physical grasp, so two gripper poses 180 degrees apart about
the approach axis are the same grasp and must not be scored as a 180-degree
error. :func:`orientation_transport_error` minimises over that symmetry by
default. ``task_frame`` already carries the same lesson for the keypoint frame:
the sign of the closing axis is not a free choice, and getting it wrong plants a
half turn in the middle of the map.
"""

from __future__ import annotations

import numpy as np

from tpgpt.utils.rotations import rotation_geodesic

#: World "up" -- the support normal used throughout ``tpgpt.sim.keypoints``.
UP = np.array([0.0, 0.0, 1.0])

#: A half turn about the approach axis, as a matrix.
#:
#: **Named, but never forgiven.** ``orientation_transport_error`` used to
#: minimise over this and no longer offers the option, because a half turn is
#: *not* the same grasp:
#:
#: * the wrist is somewhere else, which is a different joint configuration and
#:   may not be reachable;
#: * GraspGen-X's own config declares ``symmetric`` **per gripper**, and it is
#:   ``False`` for both three-finger hands in this registry -- rolling those over
#:   puts the fingers somewhere else entirely
#:   (:func:`~tpgpt.grasp.grippers.declared_symmetric`);
#: * and forgiving it concealed the half turn of 7.33 for the life of the
#:   project, reporting 0.2 to 4.5 degrees for a plan that was 179 wrong.
#:
#: Measured after the frames were fixed, the minimisation changed the answer on
#: **0 of 20** cells when compared against the grasp actually executed. So it
#: was inert as well as dangerous.
#:
#: The matrix itself stays, because the *rotation* is a real object: it is what
#: ``scene_keypoints`` applies when it chooses the other roll, and what a test
#: needs to construct a rolled case.
#:
#: In the ``Grasp6D`` convention a grasp rotation's columns are
#: ``(closing, jaw, approach)``, so the approach is the local ``z`` and the
#: symmetry is ``R @ diag(-1, -1, 1)`` -- closing and jaw both flipped, approach
#: preserved. That is exactly "the same grasp approached with the hand rolled
#: over".
JAW_SYMMETRY = np.diag([-1.0, -1.0, 1.0])


def orientation_transport_error(
    transport_map,
    X: np.ndarray,
    R_source: np.ndarray,
    R_target: np.ndarray,
) -> np.ndarray:
    """How far Eq. 11 leaves the transported hand from the orientation it needs.

    The map is asked to carry the source hand orientation onto the target's.
    Eq. 11 delivers ``J_perp(X) R_source``; this returns the angle between what
    it delivered and what was wanted.

    A grasp-centred cube built in the **task frame** carries only a yaw about the
    vertical, so an out-of-plane approach tilt survives untouched and this
    returns the full tilt. A cube built in the **full grasp pose** returns zero,
    and pays for it in :func:`vertical_tilt`.

    Args:
        transport_map: A fitted map exposing ``orthogonal_jacobian``. Duck-typed,
            so a composed map works unchanged.
        X: ``(n, 3)`` source-space positions to evaluate at -- typically the
            source grasp and release points.
        R_source: ``(n, 3, 3)`` or ``(3, 3)`` source hand orientations.
        R_target: ``(n, 3, 3)`` or ``(3, 3)`` orientations the hand must reach.
    Returns:
        ``(n,)`` angles in **degrees**.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    J_perp = transport_map.orthogonal_jacobian(X)
    R_source = np.broadcast_to(
        np.asarray(R_source, dtype=float).reshape(-1, 3, 3), (n, 3, 3)
    )
    R_target = np.broadcast_to(
        np.asarray(R_target, dtype=float).reshape(-1, 3, 3), (n, 3, 3)
    )
    transported = np.einsum("nij,njk->nik", J_perp, R_source)

    return np.degrees(rotation_geodesic(transported, R_target))


def carry_orientation_error(
    transport_map,
    x_grasp: np.ndarray,
    x_release: np.ndarray,
    R_grasp: np.ndarray,
    R_release: np.ndarray,
    R_grasp_target: np.ndarray,
    R_release_target: np.ndarray,
) -> float:
    """How far the plan turns the carried object from how far it should turn.

    Between the jaws closing and opening the object is rigidly attached to the
    hand, so its placed orientation is fixed by the hand's *relative* rotation
    over that span and by nothing else. This compares the relative rotation Eq.
    11 delivers,

    ``(J_perp(x_release) R_release) (J_perp(x_grasp) R_grasp)^T``

    against the one the target configurations ask for,
    ``R_release_target R_grasp_target^T``.

    **Why this exists rather than two calls to**
    :func:`orientation_transport_error`. That function minimises over
    :data:`JAW_SYMMETRY` at each point independently, which is right for a
    single grasp and destroys the pairing. A half turn about the approach taken
    at *both* ends cancels here -- it is the same task with the wrist rolled
    over -- while one taken at a *single* end does not, and that is exactly what
    has to be caught: it reflects the object through the grasp point and lands
    it twice its lateral grasp offset away.

    Nothing else sees that. The map stays a valid diffeomorphism, the keypoint
    residual stays at 1e-7, and **the aim stays at 0.00 mm** -- because a half
    turn about the approach leaves the grasp point fixed, so it is the one point
    a reflection cannot move. Measured over 40 cells before the frames were made
    to agree, this read **178.5 to 180.0 degrees** while
    ``orientation_transport_error`` read 0.2 to 4.5 at both ends. Section 7.33.

    Args:
        transport_map: A fitted map exposing ``orthogonal_jacobian``.
        x_grasp: ``(3,)`` source-space position where the jaws close.
        x_release: ``(3,)`` source-space position where they open.
        R_grasp: ``(3, 3)`` source hand orientation at the grasp.
        R_release: ``(3, 3)`` source hand orientation at the release.
        R_grasp_target: ``(3, 3)`` orientation wanted at the target grasp.
        R_release_target: ``(3, 3)`` orientation wanted at the target release.

    Returns:
        The angle in **degrees**.
    """
    X = np.vstack([
        np.asarray(x_grasp, dtype=float).reshape(3),
        np.asarray(x_release, dtype=float).reshape(3),
    ])
    J = transport_map.orthogonal_jacobian(X)
    delivered = (J[1] @ np.asarray(R_release, dtype=float).reshape(3, 3)) @ (
        J[0] @ np.asarray(R_grasp, dtype=float).reshape(3, 3)
    ).T
    wanted = np.asarray(R_release_target, dtype=float).reshape(3, 3) @ np.asarray(
        R_grasp_target, dtype=float
    ).reshape(3, 3).T
    return float(np.degrees(rotation_geodesic(delivered[None], wanted[None])[0]))


def vertical_tilt(transport_map, X: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """How far the map tilts "up" at each queried point, in degrees.

    ``J_perp`` rotates every direction at a point, so whatever rotation it
    applies to the hand it also applies to the vertical. This is the cost side of
    :func:`orientation_transport_error`, and the two are equal whenever the map's
    rotation is what supplies the hand's orientation change.

    Zero means the map leaves vertical alone: a lift that was straight up in the
    demonstration is still straight up after transport.

    Args:
        transport_map: A fitted map exposing ``orthogonal_jacobian``.
        X: ``(n, 3)`` source-space positions.
        up: The direction to test; world "up" by default.

    Returns:
        ``(n,)`` angles in **degrees**.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    up = np.asarray(up, dtype=float).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    rotated = np.einsum("nij,j->ni", transport_map.orthogonal_jacobian(X), up)
    return np.degrees(np.arccos(np.clip(rotated @ up, -1.0, 1.0)))


def tilt_profile(
    transport_map,
    path: np.ndarray,
    up: np.ndarray = UP,
    grasp_index: int | None = None,
    release_index: int | None = None,
) -> dict:
    """The tilt along a whole trajectory, summarised.

    A single tilt reading at the grasp cannot distinguish "the hand was rotated
    where it needed to be" from "the entire trajectory was rotated". That
    distinction is the whole case for confining a contact correction to a local
    second stage -- 22.7 degrees at the grasp with 0.1 degrees mid-path is a
    different result from 22.7 degrees everywhere, and one number cannot say
    which you have. So this returns a profile.

    ``tilt_mid_path`` is the median over the middle half of the trajectory: the
    transit, where the object is merely being carried and there is no reason for
    the map to be rotating anything.

    Key names mirror :func:`tpgpt.metrics.curves.path_deviation` so the two read
    alike in a results row.

    Args:
        transport_map: A fitted map exposing ``orthogonal_jacobian``.
        path: ``(n, 3)`` source-space trajectory.
        up: Direction to test.
        grasp_index: Label index where the jaws close, if known.
        release_index: Label index where they open, if known.

    Returns:
        ``{"tilt_max", "tilt_median", "tilt_mid_path", "tilt_argmax"}`` in
        degrees, plus ``"tilt_at_grasp"`` / ``"tilt_at_release"`` when the
        corresponding index is given.
    """
    path = np.atleast_2d(np.asarray(path, dtype=float))
    tilt = vertical_tilt(transport_map, path, up=up)
    n = len(tilt)
    lo, hi = n // 4, max(n // 4 + 1, (3 * n) // 4)
    out = {
        "tilt_max": float(tilt.max()),
        "tilt_median": float(np.median(tilt)),
        "tilt_mid_path": float(np.median(tilt[lo:hi])),
        "tilt_argmax": int(tilt.argmax()),
    }
    if grasp_index is not None:
        out["tilt_at_grasp"] = float(tilt[int(np.clip(grasp_index, 0, n - 1))])
    if release_index is not None:
        out["tilt_at_release"] = float(tilt[int(np.clip(release_index, 0, n - 1))])
    return out


def lift_deviation(
    transport_map, path: np.ndarray, start: int, end: int, up: np.ndarray = UP
) -> float:
    """How far a demonstrated vertical lift ends up off vertical after transport.

    The demonstration lifts straight up off the table to clear the shelf. If the
    map carries a rotation, that lift becomes diagonal, which costs clearance and
    reachability even when nothing about the map is invalid. Measured on a
    30-degree tilted grasp, a straight-up lift came out **24.2 degrees** off
    vertical.

    Taken from the transported segment's own end-to-end direction rather than
    from ``J_perp`` at a point, because what matters is where the hand actually
    travels, not the rotation anywhere in particular.

    Args:
        transport_map: A fitted map exposing ``transport_positions``.
        path: ``(n, 3)`` source-space trajectory.
        start: Label index where the lift begins.
        end: Label index where it ends.
        up: Direction the lift should be along.

    Returns:
        Degrees between the transported lift direction and ``up``; ``nan`` when
        the transported segment has no length.
    """
    path = np.atleast_2d(np.asarray(path, dtype=float))
    warped = transport_map.transport_positions(path[[start, end]])
    direction = warped[1] - warped[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return float("nan")
    up = np.asarray(up, dtype=float).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    return float(np.degrees(np.arccos(np.clip((direction / norm) @ up, -1.0, 1.0))))
