"""Which hand is on the FR3's flange, and how its grasp point relates to the controller frame.

The planner, the teaching converter and the executor all work with two things
per label: a **grasp point** (where the object is held) and an **orientation**.
The transportation map moves both in the *grasp convention*
(:class:`~tpgpt.sim.keypoints.GraspFrame`: ``+X`` closing, ``+Z`` approach,
middle column ``approach x closing``). The robot, however, is commanded in the
frame CRISP's controllers and ``/current_pose`` use -- the *controller frame*.
An :class:`EndEffector` is the fixed relation between the two:

* ``grasp_to_ee`` -- ``R_ee = R_grasp @ grasp_to_ee`` (columns: the controller
  frame's axes written in grasp coordinates);
* ``tool_offset`` -- the grasp point in controller-frame coordinates, which is
  what :class:`tpgpt.hardware.executor.HardwareExecutor` subtracts to get the
  pose it commands (``tcp = grasp_point - R_ee @ tool_offset``).

Two hands are described:

* :data:`FRANKA_HAND` -- the Franka Hand, whose ``fr3_hand_tcp`` *is* the
  fingertip centre: the previous, implicit behaviour, unchanged.
* :func:`inspire_end_effector` -- an Inspire RH56 hand bolted coaxially onto the
  flange through an adapter plate. There is no GraspGen-X in this path: the
  grasp is the scripted side grasp of :mod:`tpgpt.hardware.side_grasp`, so the
  "grasp point" is a point just in front of the **palm**, and the grasp frame is
  the palm frame:

  ============  ==========================================  ==========
  grasp axis    physical meaning on the hand                symbol
  ============  ==========================================  ==========
  ``+Z``        palm normal, out of the palm (approach)     ``n``
  ``+X``        wrist -> fingertips, along the hand's axis  ``f``
  ``+Y``        ``n x f`` = ``-w``, across the palm         ``-w``
  ============  ==========================================  ==========

  with ``w = f x n``. "Closing" is the fingers' direction: in a side grasp of an
  upright object it is horizontal, which the keypoint task frame needs (it is
  built from the support normal and the closing axis, and refuses a vertical
  one). On a **left** hand the thumb and index finger are on ``-w`` (= grasp
  ``+Y``); on a right hand on ``+w`` (see :func:`thumb_side_sign`).

The mount is described in the flange frame ``fr3_link8`` (``+Z`` out of the
flange face, the joint-7 axis; its axes are joint 7's, translated 0.107 m along
``+Z``), which is the only frame a physical adapter plate can be measured
against. ``fr3_hand_tcp`` is ``fr3_link8`` turned **-45 degrees about +Z** and
shifted **0.1034 m along +Z** (``franka_description`` ``fr3.urdf.xacro``:
``rpy_ee="0 0 -pi/4"``, ``tcp_xyz="0 0 0.1034"``), and it exists in the URDF
only while the robot is launched with the Franka Hand description
(``load_gripper:=true``); it is used here as a fixed virtual frame.

numpy only; no ROS, no GraspGen-X.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.hardware import hand as franka_hand
from tpgpt.hardware.frames import GRASP_TO_FR3_TCP, rotation_about_z

#: ``fr3_link8`` -> ``fr3_hand_tcp`` (franka_description fr3.urdf.xacro).
FLANGE_TO_HAND_TCP_ROTATION = rotation_about_z(-np.pi / 4.0)
FLANGE_TO_HAND_TCP_TRANSLATION = np.array([0.0, 0.0, franka_hand.TCP_OFFSET])

#: Controller frames this module knows the flange transform of.
CONTROLLER_FRAMES = {
    "fr3_hand_tcp": (FLANGE_TO_HAND_TCP_ROTATION, FLANGE_TO_HAND_TCP_TRANSLATION),
    "fr3_link8": (np.eye(3), np.zeros(3)),
}


@dataclass(frozen=True)
class EnvelopeBox:
    """An axis-aligned box in the grasp frame, duck-typed like :class:`hand.HandBox`
    (``name``, ``corners()``, ``contains()``) so drawing code accepts either."""

    name: str
    lo: np.ndarray
    hi: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return (np.asarray(self.lo) + np.asarray(self.hi)) / 2.0

    @property
    def half(self) -> np.ndarray:
        return (np.asarray(self.hi) - np.asarray(self.lo)) / 2.0

    @property
    def rotation(self) -> np.ndarray:
        return np.eye(3)

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return self.centre + signs * self.half

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        local = np.atleast_2d(points) - self.centre
        return np.all(np.abs(local) <= self.half + margin, axis=1)


@dataclass(frozen=True)
class EndEffector:
    """One hand on the flange, as the planner, teacher and executor need it.

    Attributes:
        name: ``"panda"`` or ``"inspire"``.
        kind: ``"parallel_jaw"`` (Franka Hand jaw and collision checks apply) or
            ``"inspire_side"`` (scripted side grasp, envelope-box clearance).
        controller_frame: The frame CRISP commands and ``/current_pose`` reports.
        grasp_to_ee: ``R_ee = R_grasp @ grasp_to_ee``.
        tool_offset: Grasp point in controller-frame coordinates, metres.
        envelope: Boxes bounding the hand in the grasp frame, for clearance.
        handedness: ``"left"``, ``"right"`` or ``""`` (a parallel jaw has none).
        details: Everything the description was built from, for the record.
    """

    name: str
    kind: str
    controller_frame: str
    grasp_to_ee: np.ndarray
    tool_offset: np.ndarray
    envelope: tuple = ()
    handedness: str = ""
    details: dict = field(default_factory=dict)

    @property
    def is_parallel_jaw(self) -> bool:
        return self.kind == "parallel_jaw"

    # -------------------------------------------------------------- rotations
    def ee_from_grasp(self, rotations) -> np.ndarray:
        """Grasp-convention orientations -> controller-frame orientations."""
        return np.einsum("...ij,jk->...ik", np.asarray(rotations, dtype=float), self.grasp_to_ee)

    def grasp_from_ee(self, rotations) -> np.ndarray:
        """Controller-frame orientations -> grasp-convention orientations."""
        return np.einsum("...ij,jk->...ik", np.asarray(rotations, dtype=float), self.grasp_to_ee.T)

    # -------------------------------------------------------------- positions
    def grasp_point(self, ee_positions, ee_rotations) -> np.ndarray:
        """Where the hand holds, from measured controller-frame poses (``(N, 3)``)."""
        p = np.atleast_2d(np.asarray(ee_positions, dtype=float))
        R = np.asarray(ee_rotations, dtype=float).reshape(-1, 3, 3)
        return p + np.einsum("nij,j->ni", R, self.tool_offset)

    # -------------------------------------------------------------- geometry
    def boxes(self, opening: float = franka_hand.MAX_OPENING) -> list:
        """The hand's boxes in the grasp frame (Franka Hand collision boxes, or the
        Inspire envelope)."""
        if self.is_parallel_jaw:
            return franka_hand.hand_boxes(opening)
        return list(self.envelope)

    def lowest_point(self, grasp_point, grasp_rotation,
                     opening: float = franka_hand.MAX_OPENING) -> float:
        """World ``z`` of the lowest corner of any of :meth:`boxes` at this pose."""
        if self.is_parallel_jaw:
            return franka_hand.lowest_point(grasp_point, grasp_rotation, opening)
        corners = np.vstack([b.corners() for b in self.envelope])
        return float((corners @ np.asarray(grasp_rotation).T + np.asarray(grasp_point))[:, 2].min())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "controller_frame": self.controller_frame,
            "grasp_to_ee": np.asarray(self.grasp_to_ee).tolist(),
            "tool_offset": np.asarray(self.tool_offset).tolist(),
            "handedness": self.handedness,
            "envelope": [{"name": b.name, "lo": np.asarray(b.lo).tolist(),
                          "hi": np.asarray(b.hi).tolist()} for b in self.envelope],
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EndEffector":
        """Rebuild the description a plan recorded (``plan.metadata['end_effector']``)."""
        if data.get("kind", "parallel_jaw") == "parallel_jaw":
            return FRANKA_HAND
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            controller_frame=str(data["controller_frame"]),
            grasp_to_ee=np.asarray(data["grasp_to_ee"], dtype=float).reshape(3, 3),
            tool_offset=np.asarray(data["tool_offset"], dtype=float).reshape(3),
            envelope=tuple(EnvelopeBox(b["name"], np.asarray(b["lo"], dtype=float),
                                       np.asarray(b["hi"], dtype=float))
                           for b in data.get("envelope", [])),
            handedness=str(data.get("handedness", "")),
            details=dict(data.get("details", {})),
        )


#: The Franka Hand: fr3_hand_tcp is the fingertip centre, closing along its +Y.
FRANKA_HAND = EndEffector(
    name="panda",
    kind="parallel_jaw",
    controller_frame="fr3_hand_tcp",
    grasp_to_ee=GRASP_TO_FR3_TCP,
    tool_offset=np.zeros(3),
)


def thumb_side_sign(handedness: str) -> float:
    """Sign of ``w = f x n`` on whose side the thumb and index finger sit.

    Palm frame: ``n`` palm normal, ``f`` wrist -> fingertips, ``w = f x n``.
    Look at a **right** hand's palm with the fingers up: the thumb is on your
    right, which is ``+w``. A **left** hand is its mirror image: thumb on ``-w``.
    Grasp ``+Y`` is ``-w``, so the thumb is on grasp ``-sign * Y``.
    """
    side = str(handedness).strip().lower()
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise ValueError(f"handedness must be 'left' or 'right', not {handedness!r}")


def inspire_end_effector(
    handedness: str = "left",
    controller_frame: str = "fr3_hand_tcp",
    mount_yaw_deg: float = -45.0,
    adapter_thickness: float = 0.012,
    palm_along_axis: float = 0.095,
    palm_normal_offset: float = 0.020,
    palm_lateral_offset: float = 0.0,
    envelope_lo=(-0.115, -0.045, -0.045),
    envelope_hi=(0.125, 0.045, 0.080),
) -> EndEffector:
    """An Inspire RH56 hand mounted coaxially on the FR3 flange.

    Args:
        handedness: ``"left"`` or ``"right"``.
        controller_frame: ``"fr3_hand_tcp"`` (robot launched with the Franka
            Hand description, unchanged controller configs) or ``"fr3_link8"``
            (``load_gripper:=false`` and every controller's
            ``end_effector_frame`` set to ``fr3_link8``).
        mount_yaw_deg: Rotation of the hand about the flange ``+Z`` (joint-7
            axis): the palm normal is ``Rz(mount_yaw_deg) * fr3_link8 +X``. The
            default -45 makes the palm face ``fr3_hand_tcp``'s ``+X`` (the
            direction the Franka Hand's broad face pointed) and the fingers run
            along ``fr3_hand_tcp``'s ``+Z``.
        adapter_thickness: Flange face -> the hand's mounting face, along ``+Z``.
        palm_along_axis: Hand mounting face -> the grasp point, along the hand's
            axis (wrist -> fingertips).
        palm_normal_offset: Hand axis -> the grasp point, along the palm normal.
            Measure it to the palm *surface*; the side grasp adds its own
            stand-off in front of it.
        palm_lateral_offset: Along ``w = f x n`` (grasp ``-Y``), if the palm
            centre is not on the axis.
        envelope_lo, envelope_hi: One box bounding the whole hand (open fingers,
            opposed thumb) in the grasp frame, for table and shelf clearance.

    The lengths default to estimates from the RH56 manual drawing (217.8 mm
    mount face -> fingertip, 80.7 mm palm width). **Measure them with a caliper**:
    every millimetre of error lands directly on where the palm stops.
    """
    if controller_frame not in CONTROLLER_FRAMES:
        raise ValueError(f"controller_frame must be one of {sorted(CONTROLLER_FRAMES)}, "
                         f"not {controller_frame!r}")
    thumb_side_sign(handedness)          # validates
    yaw = np.radians(float(mount_yaw_deg))
    f = np.array([0.0, 0.0, 1.0])                        # fingers along the flange axis
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])       # palm normal
    w = np.cross(f, n)
    # Grasp axes written in the flange frame: columns (closing, jaw, approach) = (f, -w, n).
    R_flange_grasp = np.column_stack([f, -w, n])
    p_flange_grasp = (f * (float(adapter_thickness) + float(palm_along_axis))
                      + n * float(palm_normal_offset) + w * float(palm_lateral_offset))
    R_flange_ee, p_flange_ee = CONTROLLER_FRAMES[controller_frame]
    # R_base_ee = R_base_grasp @ R_grasp_flange @ R_flange_ee
    grasp_to_ee = R_flange_grasp.T @ R_flange_ee
    tool_offset = R_flange_ee.T @ (p_flange_grasp - p_flange_ee)
    return EndEffector(
        name="inspire",
        kind="inspire_side",
        controller_frame=controller_frame,
        grasp_to_ee=grasp_to_ee,
        tool_offset=tool_offset,
        envelope=(EnvelopeBox("hand_envelope", np.asarray(envelope_lo, dtype=float),
                              np.asarray(envelope_hi, dtype=float)),),
        handedness=str(handedness).strip().lower(),
        details={
            "mount_yaw_deg": float(mount_yaw_deg),
            "adapter_thickness_m": float(adapter_thickness),
            "palm_along_axis_m": float(palm_along_axis),
            "palm_normal_offset_m": float(palm_normal_offset),
            "palm_lateral_offset_m": float(palm_lateral_offset),
            "grasp_point_in_flange_m": p_flange_grasp.tolist(),
            "grasp_axes_in_flange": R_flange_grasp.tolist(),
        },
    )


def inspire_from_config(config: dict) -> EndEffector:
    """:func:`inspire_end_effector` from the ``mount`` / ``envelope`` blocks of
    ``ycb_pointcloud_pipeline/config/inspire_hand.yaml`` (``inspire_hand:`` key)."""
    cfg = config.get("inspire_hand", config)
    mount = cfg.get("mount", {})
    env = cfg.get("envelope", {})
    kwargs = {
        "handedness": cfg.get("handedness", "left"),
        "controller_frame": mount.get("controller_frame", "fr3_hand_tcp"),
        "mount_yaw_deg": mount.get("yaw_deg", -45.0),
        "adapter_thickness": mount.get("adapter_thickness_m", 0.012),
        "palm_along_axis": mount.get("palm_along_axis_m", 0.095),
        "palm_normal_offset": mount.get("palm_normal_offset_m", 0.020),
        "palm_lateral_offset": mount.get("palm_lateral_offset_m", 0.0),
    }
    if env:
        kwargs["envelope_lo"] = env["lo_m"]
        kwargs["envelope_hi"] = env["hi_m"]
    return inspire_end_effector(**kwargs)


def end_effector_for_plan(metadata: dict | None) -> EndEffector:
    """The hand a saved plan was made for; plans from before this module are the Franka Hand."""
    data = (metadata or {}).get("end_effector")
    return FRANKA_HAND if not data else EndEffector.from_dict(data)


__all__ = [
    "CONTROLLER_FRAMES",
    "FLANGE_TO_HAND_TCP_ROTATION",
    "FLANGE_TO_HAND_TCP_TRANSLATION",
    "FRANKA_HAND",
    "EndEffector",
    "EnvelopeBox",
    "end_effector_for_plan",
    "inspire_end_effector",
    "inspire_from_config",
    "thumb_side_sign",
]
