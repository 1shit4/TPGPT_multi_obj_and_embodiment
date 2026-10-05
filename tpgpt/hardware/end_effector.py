"""Which hand is on the FR3's flange, and how its grasp point relates to the controller frame.

The planner, the teaching converter and the executor all work with two things
per label: a **grasp point** (where the object is held) and an **orientation**.
The transportation map moves both in the *grasp convention*
(:class:`~tpgpt.sim.keypoints.GraspFrame`: ``+X`` closing, ``+Z`` approach,
middle column ``approach x closing``) -- GraspGen-X's, uniform across hands. The
robot, however, is commanded in the frame CRISP's controllers and
``/current_pose`` use -- the *controller frame*. An :class:`EndEffector` is the
fixed relation between the two:

* ``grasp_to_ee`` -- ``R_ee = R_grasp @ grasp_to_ee`` (columns: the controller
  frame's axes written in grasp coordinates);
* ``tool_offset`` -- the grasp point in controller-frame coordinates, which is
  what :class:`tpgpt.hardware.executor.HardwareExecutor` subtracts to get the
  pose it commands (``tcp = grasp_point - R_ee @ tool_offset``).

Two hands are described:

* :data:`FRANKA_HAND` -- the Franka Hand, whose ``fr3_hand_tcp`` *is* the
  fingertip centre: the previous, implicit behaviour, unchanged.
* :func:`inspire_end_effector` -- the Inspire RH56DFX-2R (a **right** hand) bolted
  coaxially onto the flange through an adapter plate. Its grasps come from
  GraspGen-X (``inspire_hand``), so its grasp frame is **GraspGen-X's frame of
  this hand** (:mod:`tpgpt.hardware.inspire_model`): ``+Z`` the approach (the
  bisector of the palm normal and the finger direction), ``+X`` closing (thumb on
  ``+X``), the grasp point ``fingertip`` (0.15 m) along ``+Z`` from GraspGen-X's
  pose origin, moved ``calibrated_depth`` further along the approach once
  measured (plan M6). Its geometry is the model's convex pieces, open or closed
  onto the object.

**The mount.** It is described in the flange frame ``fr3_link8`` (``+Z`` out of
the flange face, the joint-7 axis), the only frame an adapter plate can be
measured against. The hand's wrist face -- ``hand_base_link``'s origin; the RH56
manual's 217.8 mm to the fingertip is measured from it -- sits
``adapter_thickness`` out along ``+Z`` and ``lateral`` off the axis, the hand's
axis along the flange's (fingers along ``+Z``), its palm normal
``Rz(mount_yaw_deg) * fr3_link8 +X``. In ``hand_base_link`` the fingers run along
``-Y`` and the palm faces ``-X``, so::

    X_H = -n,  Y_H = -z_F,  Z_H = X_H x Y_H,   p_FH = (lateral_x, lateral_y, adapter)
    T_F<-W = T_F<-H . T_W<-H^-1                (T_W<-H: the URDF's world_joint)
    grasp_to_ee = R_FW^T R_FE
    tool_offset = R_FE^T (T_F<-W (0, 0, fingertip - delta) - p_FE)

``fr3_hand_tcp`` is ``fr3_link8`` turned **-45 degrees about +Z** and shifted
**0.1034 m along +Z** (``franka_description`` ``fr3.urdf.xacro``), and it exists
in the URDF only while the robot is launched with the Franka Hand description
(``load_gripper:=true``); it is used here as a fixed virtual frame.

numpy only; no ROS, no GraspGen-X.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

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

#: The adapter plate's radius when none is given: the FR3 flange's (ISO 9409-1-31.5,
#: 63 mm across), which a plate bolted to it is unlikely to undercut.
ADAPTER_RADIUS = 0.0315

#: The Franka Hand: the destination surface counts as under the hand within this of
#: the object's footprint -- half its width across the pads (planner ``_hand_clearance``).
FRANKA_FOOTPRINT_MARGIN = 0.05


@dataclass(frozen=True)
class EndEffector:
    """One hand on the flange, as the planner, teacher and executor need it.

    Attributes:
        name: ``"panda"`` or ``"inspire"``.
        kind: ``"parallel_jaw"`` (Franka Hand jaw and collision checks apply) or
            ``"multi_finger"`` (a hand model, :mod:`tpgpt.hardware.inspire_model`).
        controller_frame: The frame CRISP commands and ``/current_pose`` reports.
        grasp_to_ee: ``R_ee = R_grasp @ grasp_to_ee``.
        tool_offset: Grasp point in controller-frame coordinates, metres.
        handedness: ``"right"`` for the Inspire hand, ``""`` for a parallel jaw.
        graspgen: The GraspGen-X description its grasps are generated with.
        fingertip_depth: That description's grasp-point depth (``fingertip[-1]``),
            m: where :func:`~tpgpt.hardware.grasp_selection.pose_to_grasp_frame`
            puts the grasp point.
        calibrated_depth: How much further along the approach the hand goes than
            the description says, m (measured, plan M6); 0 until measured.
        model_spec: Everything the hand model and the mount were built from --
            :meth:`from_dict` rebuilds the hand from it.
        details: Derived numbers, for the record.
    """

    name: str
    kind: str
    controller_frame: str
    grasp_to_ee: np.ndarray
    tool_offset: np.ndarray
    handedness: str = ""
    graspgen: str = "franka_panda"
    fingertip_depth: float = franka_hand.TCP_OFFSET
    calibrated_depth: float = 0.0
    model_spec: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def is_parallel_jaw(self) -> bool:
        return self.kind == "parallel_jaw"

    @property
    def model(self):
        """The hand model (:class:`tpgpt.hardware.inspire_model.InspireHandModel`);
        ``None`` for a parallel jaw."""
        if self.is_parallel_jaw:
            return None
        return _model_for(json.dumps(self.model_spec, sort_keys=True))

    @property
    def symmetric(self) -> bool:
        """Whether a half turn about the approach is the same grasp (GraspGen-X's
        ``symmetric``): true for a parallel jaw, false for this five-finger hand."""
        return True if self.is_parallel_jaw else self.model.description.symmetric

    @property
    def open_state(self):
        """The hand's state with nothing held: an opening (m) for the jaw, ``"open"``
        for a modelled hand."""
        return franka_hand.MAX_OPENING if self.is_parallel_jaw else "open"

    @property
    def footprint_margin(self) -> float:
        """How far past the object the hand reaches across the approach, m: the open
        hand's largest distance from the grasp point perpendicular to the approach."""
        if self.is_parallel_jaw:
            return FRANKA_FOOTPRINT_MARGIN
        verts = np.vstack([p.vertices for p in self.model.pieces("open")])
        return float(np.linalg.norm(verts[:, :2], axis=1).max())

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
    def boxes(self, state=None) -> list:
        """The hand's pieces in the grasp frame, each with ``name``, ``corners()``,
        ``contains(points, margin)`` and ``edges()``.

        ``state``: for the Franka Hand an opening in metres (default fully open);
        for a modelled hand ``"open"``, ``"close"`` or six driven joint angles (DOF
        order, rad; default open).
        """
        if self.is_parallel_jaw:
            opening = franka_hand.MAX_OPENING if state is None else float(state)
            return franka_hand.hand_boxes(opening)
        return (self.model.pieces("open" if state is None else _hand_state(state))
                + [_adapter_piece(json.dumps(self.model_spec, sort_keys=True))])

    def lowest_point(self, grasp_point, grasp_rotation, state=None) -> float:
        """World ``z`` of the lowest corner of any of :meth:`boxes` at this pose."""
        if self.is_parallel_jaw:
            opening = franka_hand.MAX_OPENING if state is None else float(state)
            return franka_hand.lowest_point(grasp_point, grasp_rotation, opening)
        corners = np.vstack([b.corners() for b in self.boxes(state)])
        return float((corners @ np.asarray(grasp_rotation).T + np.asarray(grasp_point))[:, 2].min())

    def closed_state(self, points, grasp):
        """Where each finger stops closing onto ``points`` (base frame) with the hand
        at ``grasp`` (a :class:`~tpgpt.sim.keypoints.GraspFrame`):
        :class:`tpgpt.hardware.inspire_model.ClosedState`. Modelled hands only."""
        if self.is_parallel_jaw:
            raise ValueError("closed_state is for a modelled hand; a jaw closes to a width")
        local = (np.atleast_2d(np.asarray(points, dtype=float)) - grasp.tcp) @ grasp.rotation
        return self.model.closed_onto(local)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "controller_frame": self.controller_frame,
            "grasp_to_ee": np.asarray(self.grasp_to_ee).tolist(),
            "tool_offset": np.asarray(self.tool_offset).tolist(),
            "handedness": self.handedness,
            "graspgen": self.graspgen,
            "fingertip_depth": float(self.fingertip_depth),
            "calibrated_depth": float(self.calibrated_depth),
            "model_spec": self.model_spec,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EndEffector":
        """Rebuild the hand a plan recorded (``plan.metadata['end_effector']``).

        A modelled hand is rebuilt from its ``model_spec``; if that no longer gives the
        recorded ``grasp_to_ee`` and ``tool_offset`` (the description or the mount code
        changed since the plan was made) it raises rather than execute the plan with a
        different hand.
        """
        kind = data.get("kind", "parallel_jaw")
        if kind == "parallel_jaw":
            return FRANKA_HAND
        if kind != "multi_finger":
            raise ValueError(f"a {kind!r} plan comes from the 2026-09-15 side-grasp integration, "
                             "which is gone: re-plan")
        ee = inspire_end_effector(**data["model_spec"])
        recorded_R = np.asarray(data["grasp_to_ee"], dtype=float).reshape(3, 3)
        recorded_t = np.asarray(data["tool_offset"], dtype=float).reshape(3)
        if not (np.allclose(ee.grasp_to_ee, recorded_R, atol=1e-9)
                and np.allclose(ee.tool_offset, recorded_t, atol=1e-9)):
            raise ValueError(
                "the plan's hand no longer rebuilds to the same mount: tool_offset "
                f"{recorded_t.round(4).tolist()} recorded, {ee.tool_offset.round(4).tolist()} "
                "now -- the description or the mount code changed since planning; re-plan")
        return ee


@lru_cache(maxsize=16)
def _model_for(spec_json: str):
    from tpgpt.hardware.inspire_model import InspireHandModel, load_description

    spec = json.loads(spec_json)
    desc = load_description(spec.get("description", "inspire_hand"))
    return InspireHandModel(desc, tool_depth=desc.fingertip_depth
                            - float(spec.get("calibrated_depth", 0.0)),
                            joint_map={k: tuple(v) for k, v in spec["joint_map"].items()})


@lru_cache(maxsize=16)
def _adapter_piece(spec_json: str):
    """The adapter plate between the flange face and the hand's wrist face, as a body
    piece in the grasp frame: a prism of ``adapter_radius`` around the flange axis."""
    from tpgpt.hardware.inspire_model import HandPiece, _polytope, _prism_corners

    spec = json.loads(spec_json)
    ee = inspire_end_effector(**spec)
    a, r = float(spec["adapter_thickness"]), float(spec.get("adapter_radius", ADAPTER_RADIUS))
    corners_F = _prism_corners(r, max(a, 1e-4)) + [0.0, 0.0, a / 2.0]
    R_FG = np.asarray(ee.details["grasp_axes_in_flange"], dtype=float)
    p_FG = np.asarray(ee.details["grasp_point_in_flange_m"], dtype=float)
    verts, planes, _ = _polytope((corners_F - p_FG) @ R_FG)
    return HandPiece("body_adapter", "body", "body", verts, planes)


def _hand_state(state):
    if isinstance(state, str):
        return state
    return np.asarray(state, dtype=float).reshape(6)


#: The Franka Hand: fr3_hand_tcp is the fingertip centre, closing along its +Y.
FRANKA_HAND = EndEffector(
    name="panda",
    kind="parallel_jaw",
    controller_frame="fr3_hand_tcp",
    grasp_to_ee=GRASP_TO_FR3_TCP,
    tool_offset=np.zeros(3),
)


def flange_from_hand_base(mount_yaw_deg: float, adapter_thickness: float,
                          lateral=(0.0, 0.0)) -> np.ndarray:
    """``T_F<-H``: the hand's wrist face in the flange frame (module docstring)."""
    yaw = np.radians(float(mount_yaw_deg))
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])         # palm normal
    x_h, y_h = -n, np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_h, y_h, np.cross(x_h, y_h)])
    T[:3, 3] = [float(lateral[0]), float(lateral[1]), float(adapter_thickness)]
    return T


def inspire_end_effector(
    handedness: str = "right",
    controller_frame: str = "fr3_hand_tcp",
    mount_yaw_deg: float = 135.0,
    adapter_thickness: float = 0.016,
    lateral=(0.0, 0.0),
    calibrated_depth: float = 0.0,
    description: str = "inspire_hand",
    joint_map: dict | None = None,
    mount_measured: bool = False,
    adapter_radius: float = ADAPTER_RADIUS,
) -> EndEffector:
    """The Inspire RH56DFX-2R on the FR3 flange, in GraspGen-X's frame of the hand.

    Args:
        handedness: ``"right"`` -- the only hand this was written and measured for
            (GraspGen-X's model is the right hand); anything else is refused.
        controller_frame: ``"fr3_hand_tcp"`` (robot launched with the Franka Hand
            description, unchanged controllers). ``"fr3_link8"`` is arithmetic only:
            the FK, IK and path check work at ``fr3_hand_tcp``.
        mount_yaw_deg: Palm normal = ``Rz(mount_yaw_deg) * fr3_link8 +X``. -45 makes
            the palm face ``fr3_hand_tcp``'s ``+X``; 135, as mounted (seen 2026-10-05), its ``-X``.
        adapter_thickness: Flange face -> the hand's wrist face, along ``+Z``, m.
        lateral: The wrist face's centre off the flange axis, ``(x, y)`` in
            ``fr3_link8``, m.
        calibrated_depth: The grasp-depth correction ``delta``, m (plan M6).
        description: The GraspGen-X description the grasps come from (shipped
            ``inspire_hand`` or a measured one); the geometry is always the
            ``inspire_hand`` URDF.
        joint_map: ``{dof: (q_at_0, q_at_1000[, verified])}``; ``None`` = the
            defaults of :mod:`tpgpt.hardware.inspire_model`.
        mount_measured: Whether the mount numbers were measured (recorded).
        adapter_radius: The adapter plate's radius about the flange axis, m (a body
            piece of the hand, ``body_adapter``).
    """
    from tpgpt.hardware.inspire_model import (DEFAULT_JOINT_MAP, InspireHandModel,
                                              load_description)

    if str(handedness).strip().lower() != "right":
        raise ValueError(f"handedness {handedness!r}: the hand on the robot is an RH56DFX-2R, "
                         "a right hand, and GraspGen-X's inspire_hand model is the right hand")
    if controller_frame not in CONTROLLER_FRAMES:
        raise ValueError(f"controller_frame must be one of {sorted(CONTROLLER_FRAMES)}, "
                         f"not {controller_frame!r}")
    desc = load_description(description)
    jm = {k: (float(v[0]), float(v[1]), bool(v[2]) if len(v) > 2 else False)
          for k, v in (joint_map or {n: (*DEFAULT_JOINT_MAP[n], False)
                                     for n in DEFAULT_JOINT_MAP}).items()}
    tool_depth = desc.fingertip_depth - float(calibrated_depth)
    T_WH = InspireHandModel(desc, tool_depth=0.0).link_poses(desc.open_q)["hand_base_link"]
    T_FH = flange_from_hand_base(mount_yaw_deg, adapter_thickness, lateral)
    T_FW = T_FH @ np.linalg.inv(T_WH)
    R_FE, p_FE = CONTROLLER_FRAMES[controller_frame]
    p_F_grasp = T_FW[:3, :3] @ np.array([0.0, 0.0, tool_depth]) + T_FW[:3, 3]
    grasp_to_ee = T_FW[:3, :3].T @ R_FE
    tool_offset = R_FE.T @ (p_F_grasp - p_FE)
    spec = {
        "handedness": "right",
        "controller_frame": controller_frame,
        "mount_yaw_deg": float(mount_yaw_deg),
        "adapter_thickness": float(adapter_thickness),
        "lateral": [float(lateral[0]), float(lateral[1])],
        "calibrated_depth": float(calibrated_depth),
        "description": description,
        "joint_map": {k: list(v) for k, v in jm.items()},
        "mount_measured": bool(mount_measured),
        "adapter_radius": float(adapter_radius),
    }
    return EndEffector(
        name="inspire",
        kind="multi_finger",
        controller_frame=controller_frame,
        grasp_to_ee=grasp_to_ee,
        tool_offset=tool_offset,
        handedness="right",
        graspgen=description,
        fingertip_depth=desc.fingertip_depth,
        calibrated_depth=float(calibrated_depth),
        model_spec=spec,
        details={
            "grasp_point_in_flange_m": p_F_grasp.tolist(),
            "grasp_axes_in_flange": T_FW[:3, :3].tolist(),
            "hand_base_in_flange": T_FH.tolist(),
            "tool_depth_m": tool_depth,
            "description_hashes": desc.hashes,
            "joint_map_verified": all(v[2] for v in jm.values()),
        },
    )


def inspire_from_config(config: dict) -> EndEffector:
    """:func:`inspire_end_effector` from ``config/inspire_hand.yaml`` (``inspire_hand:``):
    its ``mount``, ``graspgenx`` and ``joint_map`` blocks."""
    from tpgpt.hardware.inspire_model import joint_map_from_config

    cfg = config.get("inspire_hand", config)
    mount, gg = cfg.get("mount", {}), cfg.get("graspgenx", {})
    return inspire_end_effector(
        handedness=cfg.get("handedness", "right"),
        controller_frame=mount.get("controller_frame", "fr3_hand_tcp"),
        mount_yaw_deg=mount.get("yaw_deg", 135.0),
        adapter_thickness=mount.get("adapter_thickness_m", 0.016),
        lateral=tuple(mount.get("lateral_m", (0.0, 0.0))),
        calibrated_depth=gg.get("calibrated_depth_m", 0.0),
        description=gg.get("gripper", "inspire_hand"),
        joint_map=joint_map_from_config(cfg),
        mount_measured=bool(mount.get("measured", False)),
        adapter_radius=mount.get("adapter_radius_m", ADAPTER_RADIUS),
    )


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
    "end_effector_for_plan",
    "flange_from_hand_base",
    "inspire_end_effector",
    "inspire_from_config",
]
