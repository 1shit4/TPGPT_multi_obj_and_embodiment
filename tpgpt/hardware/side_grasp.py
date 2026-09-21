"""A scripted side grasp for an upright round object, for a hand that wraps.

:func:`tpgpt.hardware.planner.default_grasp` comes straight down, which suits a
parallel jaw. A five-finger hand holds a can, cup or bottle best the way a
person does: the palm against its side, the fingers curling round it and the
thumb opposing them. That grasp needs no generator -- for an upright cylinder
there is one natural choice per approach direction -- so there is no GraspGen-X
here (``grasp_source: side``):

1. **Where the axis is.** A circle is fitted to the cloud's horizontal
   projection (algebraic least squares, :func:`fit_circle`). One camera sees an
   arc of the surface, never the centre, so the cloud's *centroid* sits in front
   of the true axis by up to ``2r/pi`` (a 33 mm can: 21 mm); the circle does not.
   When the fit is implausible (radius outside ``[min_radius, max_radius]`` or
   centre more than ``max_centre_shift`` from the centroid, which happens on a
   flat or tiny cloud), the centroid and half the horizontal extent are used and
   the fallback is recorded.
2. **From which side.** The approach is horizontal and **radial from the robot
   base**: the palm travels away from the robot, towards the axis. That is the
   direction the arm reaches in most naturally, and it keeps the wrist from
   having to wrap around the object.
3. **At what height.** ``height_fraction`` of the object's height above the
   support, but never lower than ``min_height_above_support`` (the lowest finger
   must clear the table: the palm is ~80 mm wide and hangs half of it below the
   grasp point when the hand is on its side) and never higher than
   ``max_below_top`` under the top.
4. **How close.** The grasp point is the palm point, placed ``standoff`` in
   front of the fitted surface. The executor drives the hand there and the
   fingers close; the force threshold of each finger stops it on contact.
5. **Which way up.** The fingers (grasp ``+X``, the closing axis) run
   horizontally across the object's face and the palm width (grasp ``Y``) is
   vertical. With ``thumb_up`` the thumb and index finger are on top and the
   little finger is nearest the table
   (:func:`tpgpt.hardware.end_effector.thumb_side_sign`).

numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.hardware.end_effector import thumb_side_sign
from tpgpt.sim.keypoints import GraspFrame


@dataclass
class SideGraspSettings:
    """Side grasp choices. SI units."""

    standoff: float = 0.015
    height_fraction: float = 0.5
    min_height_above_support: float = 0.05
    max_below_top: float = 0.02
    thumb_up: bool = True
    base_xy: tuple = (0.0, 0.0)
    min_radius: float = 0.01
    max_radius: float = 0.12
    max_centre_shift: float = 0.08


@dataclass
class SideGraspInfo:
    """What the side grasp measured, for the plan record."""

    centre: np.ndarray
    radius: float
    fit_rms: float
    fit_used: bool
    height_above_support: float
    object_height: float
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "axis_xy_m": np.asarray(self.centre)[:2].tolist(),
            "radius_m": self.radius,
            "circle_fit_rms_m": self.fit_rms,
            "circle_fit_used": self.fit_used,
            "grasp_height_above_support_m": self.height_above_support,
            "object_height_m": self.object_height,
            "notes": list(self.notes),
        }


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares circle through ``(N, 2)`` points (Kasa's algebraic fit).

    Solves ``x^2 + y^2 = 2 a x + 2 b y + c`` for the centre ``(a, b)`` and
    ``r = sqrt(c + a^2 + b^2)``. Returns ``(centre, radius, rms)``, where
    ``rms`` is the RMS distance of the points from the fitted circle.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    if len(xy) < 3:
        raise ValueError("a circle needs at least three points")
    A = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    (a0, b0, c0), *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c0 + a0 ** 2 + b0 ** 2
    centre = np.array([a0, b0])
    radius = float(np.sqrt(max(r2, 0.0)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - centre, axis=1) - radius) ** 2)))
    return centre, radius, rms


def side_grasp(
    points: np.ndarray,
    support_height: float,
    handedness: str = "left",
    settings: SideGraspSettings | None = None,
) -> tuple[GraspFrame, SideGraspInfo]:
    """The palm-first side grasp on an upright object. See the module docstring."""
    s = settings or SideGraspSettings()
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) < 10:
        raise ValueError(f"{len(points)} points is too few to place a side grasp")
    notes: list[str] = []
    xy = points[:, :2]
    centroid = xy.mean(axis=0)
    base = np.asarray(s.base_xy, dtype=float).reshape(2)

    centre, radius, rms = fit_circle(xy)
    fit_used = (s.min_radius <= radius <= s.max_radius
                and float(np.linalg.norm(centre - centroid)) <= s.max_centre_shift)
    if not fit_used:
        notes.append(f"circle fit implausible (r {radius * 1000:.0f} mm, centre "
                     f"{np.linalg.norm(centre - centroid) * 1000:.0f} mm from the centroid); "
                     "used the centroid and half the horizontal extent")
        radial = centroid - base
        radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
        across = np.array([-radial[1], radial[0]])
        centre = centroid
        radius = float(np.clip(0.5 * np.ptp(xy @ across), s.min_radius, s.max_radius))

    radial = centre - base
    dist = float(np.linalg.norm(radial))
    if dist < 1e-6:
        raise ValueError("the object is on the robot's base axis; no radial approach exists")
    approach = np.array([radial[0] / dist, radial[1] / dist, 0.0])

    support = float(support_height)
    top = float(points[:, 2].max())
    height = max(top - support, 0.0)
    wanted = s.height_fraction * height
    ceiling = height - s.max_below_top
    z_rel = min(wanted, ceiling)
    if z_rel < s.min_height_above_support:
        notes.append(f"object {height * 1000:.0f} mm tall: grasp raised to the "
                     f"{s.min_height_above_support * 1000:.0f} mm floor that keeps the lowest "
                     "finger off the table")
        z_rel = s.min_height_above_support

    axis_point = np.array([centre[0], centre[1], support + z_rel])
    tcp = axis_point - approach * (radius + s.standoff)
    # Thumb (and index) on grasp -sign * Y (end_effector.thumb_side_sign). With the
    # closing axis c = up x approach, Y = approach x c = up; so for the thumb on top
    # (-sign * Y = up) take c = -sign * (up x approach).
    thumb = thumb_side_sign(handedness)
    up = np.array([0.0, 0.0, 1.0])
    across = np.cross(up, approach)
    closing = (-thumb * across) if s.thumb_up else (thumb * across)
    grasp = GraspFrame(tcp=tcp, approach=approach, closing=closing)
    info = SideGraspInfo(centre=np.array([centre[0], centre[1], support]), radius=float(radius),
                         fit_rms=float(rms), fit_used=bool(fit_used),
                         height_above_support=float(z_rel), object_height=float(height),
                         notes=notes)
    return grasp, info


__all__ = ["SideGraspInfo", "SideGraspSettings", "fit_circle", "side_grasp"]
