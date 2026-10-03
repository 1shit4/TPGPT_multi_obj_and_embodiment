"""Geometry of the Franka Hand on the FR3, for judging a grasp against a cloud.

Two questions decide whether a grasp on a *measured* cloud is physically
possible, and neither is answered by the object's overall size:

1. **Does what lies between the pads fit?** Only the material the finger pads
   sweep matters. A cup is 90 mm across and cannot be closed on as a whole by an
   80 mm hand, but its rim is a 3 mm wall, and a grasp that puts one pad inside
   the cup and one outside closes on 3 mm. Measuring the whole cloud along the
   closing axis -- what :func:`tpgpt.hardware.planner.plan` used to do -- refuses
   that grasp for a width the jaws never have to span.
2. **Does the rest of the hand stay out of the object and out of the table?**
   A pad that would land *on* the object when the hand arrives open, a palm that
   would push into the rim, or fingertips below the support plane.

Every dimension below is read from ``franka_description``
(``end_effectors/common/franka_hand.xacro`` and the hand's collision mesh
``meshes/robot_ee/franka_hand_white/collision/hand.stl``), not estimated:

* ``fr3_hand_tcp`` sits **0.1034 m** along the hand's ``+Z``;
* both fingers are prismatic along the hand's ``+Y``, mounted **0.0584 m** along
  ``+Z``, each travelling 0 to **0.04 m** -- an 80 mm maximum opening;
* each rubber pad is a box **17.5 x 15.2 x 18.5 mm** centred at
  ``(0, 7.58, 45.25) mm`` in its finger frame, so its inner face sits exactly at
  the finger's joint position and its centre at 0.10365 m, 0.25 mm past the TCP;
* the collision mesh of the hand body spans x ``[-31.6, 31.6]`` mm,
  y ``[-104.0, 100.4]`` mm, z ``[-25.9, 66.0]`` mm.

Everything is expressed in the **grasp convention** (``+X`` closing, ``+Z``
approach, origin at the TCP), the frame :class:`~tpgpt.sim.keypoints.GraspFrame`
describes. :data:`tpgpt.hardware.frames.GRASP_TO_FR3_TCP` relates it to the
hand: hand ``+Y`` is grasp ``+X``, hand ``+X`` is grasp ``-Y``, and hand ``+Z``
is grasp ``+Z`` shifted by the TCP offset.

numpy only; no ROS, no GraspGen-X assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: ``fr3_hand`` to ``fr3_hand_tcp`` along ``+Z`` (``tcp_xyz`` in the xacro).
TCP_OFFSET = 0.1034
#: Finger joints' origin along the hand's ``+Z``.
FINGER_MOUNT_Z = 0.0584
#: Upper limit of each finger joint; the opening is twice this.
FINGER_TRAVEL = 0.04
MAX_OPENING = 2.0 * FINGER_TRAVEL

#: Collision boxes of one (left) finger, in its own frame:
#: ``(name, centre, roll about x in radians, size)``.
_FINGER_BOXES = (
    ("screw_mount", (0.0, 18.5e-3, 11e-3), 0.0, (22e-3, 15e-3, 20e-3)),
    ("sledge", (0.0, 6.8e-3, 2.2e-3), 0.0, (22e-3, 8.8e-3, 3.8e-3)),
    ("diagonal", (0.0, 15.9e-3, 28.35e-3), np.pi / 6.0, (17.5e-3, 7e-3, 23.5e-3)),
    ("pad", (0.0, 7.58e-3, 45.25e-3), 0.0, (17.5e-3, 15.2e-3, 18.5e-3)),
)

#: Axis-aligned bound of ``hand.stl`` in the hand frame, ``(min, max)``.
_BODY_MIN = np.array([-0.0316, -0.1040, -0.0259])
_BODY_MAX = np.array([0.0316, 0.1004, 0.0660])

#: Hand frame to grasp frame (origin at the TCP): x_g = y_h, y_g = -x_h, z_g = z_h - TCP.
_R_GRASP_HAND = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _roll(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass(frozen=True)
class HandBox:
    """One oriented collision box, in the grasp frame."""

    name: str
    centre: np.ndarray
    rotation: np.ndarray      # box axes as columns, in grasp coordinates
    half: np.ndarray

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        local = (np.atleast_2d(points) - self.centre) @ self.rotation
        return np.all(np.abs(local) <= self.half + margin, axis=1)

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return self.centre + (signs * self.half) @ self.rotation.T

    def edges(self) -> list:
        """Index pairs into :meth:`corners` of the box's 12 edges, for drawing."""
        return [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def hand_boxes(opening: float = MAX_OPENING) -> list[HandBox]:
    """The hand's collision boxes with the fingers ``opening`` apart, grasp frame.

    The right finger is the left one turned a half turn about the hand's ``+Z``
    (``rpy="0 0 ${pi}"`` on ``finger_joint2``), so its boxes mirror in x and y.
    """
    q = float(np.clip(opening, 0.0, MAX_OPENING)) / 2.0
    boxes = [HandBox(
        "body",
        centre=_R_GRASP_HAND @ ((_BODY_MIN + _BODY_MAX) / 2.0 - [0.0, 0.0, TCP_OFFSET]),
        rotation=_R_GRASP_HAND.copy(),
        half=(_BODY_MAX - _BODY_MIN) / 2.0,   # in the box's own (hand) axes
    )]
    for side, mirror in (("left", np.diag([1.0, 1.0, 1.0])), ("right", np.diag([-1.0, -1.0, 1.0]))):
        for name, centre, roll, size in _FINGER_BOXES:
            in_finger = np.asarray(centre) + [0.0, q, 0.0]
            in_hand = mirror @ in_finger + [0.0, 0.0, FINGER_MOUNT_Z]
            boxes.append(HandBox(
                f"{side}_{name}",
                centre=_R_GRASP_HAND @ (in_hand - [0.0, 0.0, TCP_OFFSET]),
                rotation=_R_GRASP_HAND @ mirror @ _roll(roll),
                half=np.asarray(size) / 2.0,
            ))
    return boxes


def pad_box(opening: float = MAX_OPENING) -> tuple[np.ndarray, np.ndarray]:
    """The region *between* the pads, grasp frame, as ``(min, max)`` corners.

    Its closing extent is the opening; its other two extents are the pad's own
    face (17.5 mm across, 18.5 mm along the approach), which is the only part of
    the finger that is meant to touch the object.
    """
    pad = next(b for b in hand_boxes(opening) if b.name == "left_pad")
    half_y, half_z = pad.half[0], pad.half[2]
    lo = np.array([-opening / 2.0, -half_y, pad.centre[2] - half_z])
    hi = np.array([opening / 2.0, half_y, pad.centre[2] + half_z])
    return lo, hi


@dataclass
class JawCheck:
    """What a grasp's jaws would meet, measured on the cloud.

    Attributes:
        verdict: ``"fits"``, ``"too_wide"`` (material any part of an open
            finger would land on, or a width the hand cannot span),
            ``"collides"`` (the palm inside the object), ``"empty"`` (too
            little cloud between the pads to claim the jaws close on anything)
            or ``"shallow"`` (the object fills less than
            :data:`MIN_ENGAGEMENT` of the pads' 18.5 mm height -- a pinch on
            the very edge of a rim, which slips).
        width: Extent along the closing axis of the cloud between the pads, m.
            A **lower bound** on the true width, because a depth cloud is a
            partial view.
        n_between: Cloud points between the pads.
        engagement: Extent along the approach of the cloud between the pads, m:
            how much of the pad height has material to press on.
        blocked_by: Box names that contained cloud points.
    """

    verdict: str
    width: float
    n_between: int
    blocked_by: list = field(default_factory=list)
    engagement: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict == "fits"


#: Clearance the object must leave inside the open jaws, each side, m. 0: an object
#: up to the full opening (80 mm) is accepted; a 73 mm apple was refused at 4 mm.
JAW_MARGIN = 0.0
#: Cloud points that must lie between the pads. At the planner's 3 mm voxel a
#: pad face (17.5 x 18.5 mm) pressed on a surface covers about 36 points.
MIN_POINTS_BETWEEN = 8
#: Pad height, along the approach, that must have object behind it. The pads are
#: 18.5 mm tall; with less than 10 mm engaged, a grasp on a cup's rim holds the
#: top few millimetres of wall and the cup pivots out.
MIN_ENGAGEMENT = 0.010
#: Cloud points inside a hand box before it counts as a collision, so a single
#: stray depth pixel does not veto a grasp.
MIN_COLLISION_POINTS = 3


def to_grasp_frame(points: np.ndarray, tcp, rotation) -> np.ndarray:
    """World points in the grasp frame whose origin is ``tcp``."""
    return (np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(tcp)) @ np.asarray(rotation)


def check_jaws(
    points: np.ndarray,
    tcp,
    rotation,
    opening: float = MAX_OPENING,
    margin: float = JAW_MARGIN,
    min_points: int = MIN_POINTS_BETWEEN,
    min_collision_points: int = MIN_COLLISION_POINTS,
    min_engagement: float = MIN_ENGAGEMENT,
) -> JawCheck:
    """Judge one grasp (TCP position and grasp-convention rotation) on a cloud.

    The hand is taken **fully open**, because that is how it arrives: anything a
    pad or finger link would occupy at full opening has to be empty, and what
    lies between the pads has to leave ``margin`` on each side.
    """
    local = to_grasp_frame(points, tcp, rotation)
    lo, hi = pad_box(opening)
    between = np.all((local >= lo) & (local <= hi), axis=1)
    n_between = int(between.sum())
    width = float(np.ptp(local[between, 0])) if n_between >= 2 else 0.0
    engagement = float(np.ptp(local[between, 2])) if n_between >= 2 else 0.0

    blocked = []
    for box in hand_boxes(opening):
        inside = box.contains(local, margin=margin if box.name.endswith("pad") else 0.0)
        if int(inside.sum()) >= min_collision_points:
            blocked.append(box.name)
    # Any part of an open *finger* on the object means the object does not fit
    # between the fingers -- a surface-only cloud of a solid box shows this as
    # the finger links meeting its top face, with nothing between the pads.
    fingers = [b for b in blocked if b != "body"]
    if fingers or width + 2.0 * margin > opening:
        return JawCheck("too_wide", width, n_between, blocked, engagement)
    if blocked:
        return JawCheck("collides", width, n_between, blocked, engagement)
    if n_between < min_points:
        return JawCheck("empty", width, n_between, blocked, engagement)
    if engagement < min_engagement:
        return JawCheck("shallow", width, n_between, blocked, engagement)
    return JawCheck("fits", width, n_between, blocked, engagement)


def lowest_point(tcp, rotation, opening: float = MAX_OPENING) -> float:
    """World ``z`` of the lowest corner of any hand box at this pose."""
    corners = np.vstack([b.corners() for b in hand_boxes(opening)])
    return float((corners @ np.asarray(rotation).T + np.asarray(tcp))[:, 2].min())


__all__ = [
    "JAW_MARGIN",
    "MIN_ENGAGEMENT",
    "MAX_OPENING",
    "TCP_OFFSET",
    "HandBox",
    "JawCheck",
    "check_jaws",
    "hand_boxes",
    "lowest_point",
    "pad_box",
    "to_grasp_frame",
]
