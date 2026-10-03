"""From a real object point cloud to a transported policy label set.

This is the keypoint-and-transport half of ``tpgpt.experiments.pipeline.run``
with the simulated perception replaced by a measured cloud:

1. a **grasp** on the target cloud -- :func:`~tpgpt.sim.keypoints.top_down_grasp`
   by default, or any externally supplied grasp in the grasp convention
   (:mod:`tpgpt.hardware.grasp_selection` chooses one from GraspGen-X);
2. :func:`~tpgpt.sim.keypoints.scene_keypoints` pairing the source and target
   configurations, with Experiment R's grasp-pose cube (``FINDINGS.md`` §8p),
   the pipeline's own default;
3. a :class:`~tpgpt.transport.maps.TransportMap` fitted on those pairs and
   checked for property (ii) of Sec. III-C on the path it will carry;
4. :func:`~tpgpt.transport.labels.transport_labels` on the source fingertip
   labels, then the grasp convention is converted to the controller frame of
   the hand on the flange (:mod:`tpgpt.hardware.end_effector`; ``fr3_hand_tcp``
   for the Franka Hand).

The Franka Hand's jaw-width and collision checks apply only to a parallel jaw.
A modelled hand (the Inspire RH56DFX-2R, :mod:`tpgpt.hardware.inspire_model`) takes
its grasp from GraspGen-X like the jaw; the plan records where its fingers stop on
the object (``hand_closed_q``), which is its carried state in the path check.

**The roll of an asymmetric hand** (plan of 2026-10-04, user). The sim's
``scene_keypoints`` turns the target grasp a half turn about its approach when its
closing axis opposes the demonstration's -- the same grip for a parallel jaw. For the
Inspire hand that is a different grasp, one GraspGen-X never proposed (thumb and
fingers swapped, the wrist on the far side). The turn is taken instead on the side
where it means nothing: the demonstrating Franka jaw, whose closing sign is only a
convention (``frames.GRASP_TO_FR3_TCP`` already differs from GraspGen-X's own Panda
frame by it). Describing the whole demonstration -- every label, pick and place
alike -- with its jaw frame turned (``R J``, ``J = diag(-1, -1, 1)``) pairs the same
keypoints, so the map, ``min det`` and every transported position are the sim's,
and the transported orientations are the sim's times ``J``: the Inspire hand runs
the generator's grasp. The object, rigid in the hand, moves identically
(``H(t) J (H(g) J)^-1 = H(t) H(g)^-1``).

Nothing here knows about ROS or cameras. The cloud must already be in the robot
**base** frame, and that transform is the single most important input: every
millimetre of extrinsic error lands directly on the grasp, because ``phi``
interpolates the keypoints built from this cloud exactly. The ROS side provides
the tools to verify it.

Also computed, because ``docs/sim/SIM_CLAUDE.md`` asks for geometric checks before any
end-to-end run: the aim of the transported path at the grasp and at the
release, the orientation error at the grasp, the minimum Jacobian determinant,
and the keypoint residual. All are milliseconds, deterministic, and need no
robot motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.hardware.end_effector import FRANKA_HAND, EndEffector
from tpgpt.hardware.frames import fr3_tcp_to_grasp
from tpgpt.hardware.hand import MAX_OPENING, check_jaws
from tpgpt.hardware.source_bundle import SourceBundle
from tpgpt.sim.keypoints import (
    GRASP_CUBE_HALF_EXTENT,
    GraspFrame,
    KeypointSet,
    ObjectPlacement,
    carry_indices,
    scene_keypoints,
    top_down_grasp,
)
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import rotation_geodesic

#: Keypoint construction. Identical to ``pipeline.DEFAULT_KEYPOINT_BOX`` and
#: ``DEFAULT_KEYPOINT_ORIENTATION`` -- restated rather than imported because
#: importing ``tpgpt.experiments.pipeline`` pulls in robosuite, which the robot
#: computer should not need. ``tests/unit/test_hardware_planner.py`` asserts they
#: stay equal.
KEYPOINT_BOX = "grasp_cube"
KEYPOINT_ORIENTATION = "grasp"

#: How far past the object's own footprint the destination surface is taken to
#: reach, when judging whether the hand dips into it: half the Franka Hand's
#: width across the pads. Beyond it the hand is off the board (`_hand_clearance`).
PLACE_FOOTPRINT_MARGIN = 0.05

#: Below this many points an object is too thinly seen to describe
#: (``pipeline.MIN_CLOUD_POINTS``).
MIN_CLOUD_POINTS = 40

#: Refuse a map whose smallest Jacobian determinant is below this. A merely
#: sign-consistent map can still be nearly folded somewhere along the path;
#: ``docs/sim/SIM_CLAUDE.md`` records ``min det`` as the useful diagnostic, with healthy maps
#: at a median of 0.9 and a collapsed one at 0.008.
MIN_DETERMINANT = 0.1

#: Franka Hand maximum opening, metres.
FRANKA_HAND_MAX_WIDTH = 0.08

#: Stay this far from the top of an object with the fingertip centre, so the
#: pads close on material rather than on the top edge.
GRASP_DEPTH_BELOW_TOP = 0.03


class PlanningError(RuntimeError):
    """A plan was refused. The message names the stage and the measurement."""


@dataclass
class TargetScene:
    """What the real scene contributes. All in the robot base frame.

    Attributes:
        points: ``(N, 3)`` object cloud.
        destination: ``(3,)`` where the object should end up; only x and y are
            used, as in ``ObjectPlacement``.
        destination_height: Height of the surface it will be set down on.
        support_height: Height of the surface it stands on now. ``None``
            estimates it from the cloud's lowest points, which truncation makes
            a little too high (``fit_aligned_box`` docstring: 0.5-24.5 mm).
        grasp: A grasp in the grasp convention. ``None`` uses a top-down grasp.
    """

    points: np.ndarray
    destination: np.ndarray
    destination_height: float
    support_height: float | None = None
    grasp: GraspFrame | None = None


@dataclass
class Plan:
    """Everything one planning call produced."""

    labels: PolicyLabels                 # grasp-point positions, controller-frame orientations
    source_keypoints: KeypointSet
    target_keypoints: KeypointSet
    grasp: GraspFrame                    # the grasp actually executed (roll resolved)
    target_points: np.ndarray
    support_height: float
    destination: np.ndarray
    destination_height: float
    metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    #: With labels that are a learned attractor (a bundle with an ``arm_path``): the path
    #: the arm really goes, transported like the labels (orientations in the controller
    #: frame), and its margin per label (m). The path check and the geometric metrics use
    #: it; the executor still runs ``labels``.
    arm_path: PolicyLabels | None = None
    arm_margin: np.ndarray | None = None
    #: The margin kept from the table (m, per label): the vertical part of ``arm_margin``.
    arm_table_margin: np.ndarray | None = None
    #: With learned labels: where the hand should be when the gripper acts -- the taught
    #: path transported at the grasp and the release (``positions`` (2, 3) grasp point,
    #: ``orientations`` (2, 3, 3) controller frame). The executor's arrival hold aims here;
    #: the learned command at those labels is offset from it by the lead.
    marks: dict | None = None
    #: The joint posture the whole-path check posed the arm in at each tested label
    #: (``labels`` (n,) int, ``q`` (n, 7) rad; :func:`tpgpt.hardware.path_check.path_feasibility`
    #: ``postures``). The executor sends it to CRISP as its posture target, so the elbow is
    #: where the check put it (2026-10-02).
    checked_postures: dict | None = None

    @property
    def dt(self) -> float:
        return 1.0 / float(self.metadata.get("control_freq", 20))

    # ------------------------------------------------------------------ io
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.labels.save(directory / "transported_labels.npz")
        if self.arm_path is not None:
            self.arm_path.save(directory / "transported_arm_path.npz")
            np.save(directory / "arm_path_margin.npy", np.asarray(self.arm_margin, dtype=float))
            if self.arm_table_margin is not None:
                np.save(directory / "arm_path_table_margin.npy",
                        np.asarray(self.arm_table_margin, dtype=float))
        if self.marks is not None:
            np.savez(directory / "transported_marks.npz",
                     **{k: np.asarray(v, dtype=float) for k, v in self.marks.items()})
        if self.checked_postures is not None and len(self.checked_postures["labels"]):
            np.savez(directory / "checked_postures.npz",
                     labels=np.asarray(self.checked_postures["labels"], dtype=int),
                     q=np.asarray(self.checked_postures["q"], dtype=float))
        self.source_keypoints.save(directory / "source_keypoints.json")
        self.target_keypoints.save(directory / "target_keypoints.json")
        np.savez_compressed(
            directory / "target_scene.npz",
            points=self.target_points,
            grasp_tcp=self.grasp.tcp,
            grasp_approach=self.grasp.approach,
            grasp_closing=self.grasp.closing,
            support_height=np.array(self.support_height),
            destination=np.asarray(self.destination, dtype=float),
            destination_height=np.array(self.destination_height),
            target_place_points=np.asarray(
                self.diagnostics.get("target_place_points", np.zeros((0, 3)))
            ),
        )
        (directory / "plan.json").write_text(
            json.dumps({"metrics": self.metrics, "metadata": self.metadata},
                       indent=2, default=_json_default)
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "Plan":
        directory = Path(directory)
        info = json.loads((directory / "plan.json").read_text())
        with np.load(directory / "target_scene.npz", allow_pickle=False) as d:
            grasp = GraspFrame(
                tcp=d["grasp_tcp"], approach=d["grasp_approach"], closing=d["grasp_closing"]
            )
            plan = cls(
                labels=PolicyLabels.load(directory / "transported_labels.npz"),
                source_keypoints=KeypointSet.load(directory / "source_keypoints.json"),
                target_keypoints=KeypointSet.load(directory / "target_keypoints.json"),
                grasp=grasp,
                target_points=d["points"],
                support_height=float(d["support_height"]),
                destination=d["destination"],
                destination_height=float(d["destination_height"]),
                metrics=info.get("metrics", {}),
                diagnostics={"target_place_points": d["target_place_points"]},
                metadata=info.get("metadata", {}),
            )
        if (directory / "transported_arm_path.npz").exists():
            plan.arm_path = PolicyLabels.load(directory / "transported_arm_path.npz")
            plan.arm_margin = np.load(directory / "arm_path_margin.npy")
            if (directory / "arm_path_table_margin.npy").exists():
                plan.arm_table_margin = np.load(directory / "arm_path_table_margin.npy")
        if (directory / "transported_marks.npz").exists():
            with np.load(directory / "transported_marks.npz") as d:
                plan.marks = {k: d[k] for k in d.files}
        if (directory / "checked_postures.npz").exists():
            with np.load(directory / "checked_postures.npz") as d:
                plan.checked_postures = {"labels": d["labels"], "q": d["q"]}
        return plan


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


def estimate_support_height(points: np.ndarray, percentile: float = 1.0) -> float:
    """Height of the surface under an object, from the object's own cloud.

    A low percentile rather than the minimum, so one stray depth pixel below
    the table cannot drag the box through it. Biased *high* by segmentation
    truncation; prefer a measured table height when you have one.
    """
    return float(np.percentile(np.asarray(points)[:, 2], percentile))


def default_grasp(
    points: np.ndarray,
    support_height: float,
    depth_below_top: float = GRASP_DEPTH_BELOW_TOP,
    closing_axis: np.ndarray | None = None,
) -> GraspFrame:
    """A vertical grasp across the object's narrow horizontal direction.

    :func:`~tpgpt.sim.keypoints.top_down_grasp` with the height chosen from a
    fingertip depth instead of a fraction: the source demonstration grasps at
    ``SOURCE_GRASP_FRACTION = 1.0``, the very top of a box, which suits a
    simulated contact offset and not a real pad that is ~2 cm long.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    top = float(points[:, 2].max())
    height = max(top - float(support_height), 1e-3)
    tcp_height = max(top - depth_below_top, float(support_height) + 0.5 * height)
    # top_down_grasp measures its fraction between the *cloud's* lowest and
    # highest points, so convert against those.
    low = float(points[:, 2].min())
    fraction = float(np.clip((tcp_height - low) / max(top - low, 1e-6), 0.0, 1.0))
    return top_down_grasp(points, height_fraction=fraction, closing_axis=closing_axis)


def plan(
    source: SourceBundle,
    scene: TargetScene,
    *,
    box: str = KEYPOINT_BOX,
    orientation: str = KEYPOINT_ORIENTATION,
    cube_half_extent: float = GRASP_CUBE_HALF_EXTENT,
    min_determinant: float = MIN_DETERMINANT,
    grasp_depth_below_top: float = GRASP_DEPTH_BELOW_TOP,
    max_object_width: float = FRANKA_HAND_MAX_WIDTH,
    end_effector: EndEffector | None = None,
) -> Plan:
    """Transport the source demonstration onto a measured scene.

    ``end_effector`` is the hand on the flange (default: the Franka Hand). It
    decides the orientation convention of the output labels and whether the
    parallel-jaw checks apply.

    Raises:
        PlanningError: when the cloud is too sparse, the object is provably too
            wide for the hand, the keypoints are degenerate, or the map folds.
    """
    points = np.atleast_2d(np.asarray(scene.points, dtype=float))
    if len(points) < MIN_CLOUD_POINTS:
        raise PlanningError(
            f"object_not_seen: only {len(points)} points in the target cloud "
            f"(need {MIN_CLOUD_POINTS})"
        )
    support_height = (
        estimate_support_height(points) if scene.support_height is None
        else float(scene.support_height)
    )
    ee = end_effector or FRANKA_HAND
    if scene.grasp is None and not ee.is_parallel_jaw:
        raise PlanningError(f"no_grasp: the {ee.name} hand needs an explicit grasp (GraspGen-X's, "
                            "through grasp_selection); the top-down default is a parallel jaw's")
    grasp = scene.grasp or default_grasp(points, support_height, grasp_depth_below_top)

    jaw = None
    width = 0.0
    if ee.is_parallel_jaw:
        # Width is measured where the pads close, not over the whole cloud: a cup is
        # wider than the hand and its rim is not (tpgpt.hardware.hand). A cloud's
        # extent is still a *lower bound* on width (docs/sim/SIM_CLAUDE.md, 7.36), so material
        # under an open pad proves "too wide" and nothing proves "fits".
        jaw = check_jaws(points, grasp.tcp, grasp.rotation, opening=max_object_width)
        if jaw.verdict == "too_wide":
            raise PlanningError(
                f"object_too_wide: at the grasp the cloud measures {jaw.width * 1000:.0f} mm "
                f"between the pads and reaches under an open pad "
                f"({', '.join(jaw.blocked_by) or 'no margin left'}); the hand opens "
                f"{max_object_width * 1000:.0f} mm"
            )
        if jaw.verdict == "collides":
            raise PlanningError(
                f"hand_collides: at the grasp the palm is inside the object cloud "
                f"({', '.join(jaw.blocked_by)})"
            )
        width = jaw.width

    target = ObjectPlacement(
        points=points,
        grasp=grasp,
        support_height=support_height,
        destination=np.asarray(scene.destination, dtype=float).reshape(3),
        destination_height=float(scene.destination_height),
    )
    tool_labels = source.tool_labels
    # The roll (module docstring): an asymmetric target hand executes its own grasp. With
    # a symmetric source the half turn goes to the source's jaw after the keypoints; with
    # an asymmetric one too, no turn is taken at all and the map's own gate decides.
    source_symmetric = _source_symmetric(source)
    keep_target_roll = not ee.symmetric and not source_symmetric
    flip_target = keep_target_roll and float(
        np.asarray(grasp.closing) @ np.asarray(source.placement.grasp.closing)) < 0.0
    # What the object did is read off the labels at the grasp and the release; for learned
    # labels those lead the hand, so the taught ones say it (SourceBundle.demonstration_labels).
    source_set, target_set, diagnostics = scene_keypoints(
        source.placement, target, source.demonstration_labels,
        include_contacts=False, box=box, orientation=orientation,
        cube_half_extent=cube_half_extent, flip_target=flip_target,
    )
    rolled = bool(diagnostics["target_grasp_rolled"])
    jaw_turned = rolled and not ee.symmetric and source_symmetric

    def unrolled(rotations):
        """Grasp-convention orientations of the demonstration with its jaw turned."""
        R = np.asarray(rotations, dtype=float)
        return R @ JAW_TURN if jaw_turned else R
    if source_set.is_degenerate() or target_set.is_degenerate():
        raise PlanningError(
            f"keypoints_degenerate: {len(source_set)} keypoints do not span three dimensions"
        )

    transport_map = TransportMap().fit(source_set.points, target_set.points)
    report = transport_map.check_diffeomorphism(tool_labels.positions)
    residual = float(transport_map.keypoint_residual().max())
    if not report.consistent_sign or report.min_determinant < min_determinant:
        raise PlanningError(
            f"map_not_a_diffeomorphism: min det(J) {report.min_determinant:.3f} "
            f"(need >= {min_determinant}), fraction positive {report.fraction_positive:.2f}"
        )

    transported = transport_labels(transport_map, tool_labels)
    grasp_orientations = unrolled(transported.orientations)
    transported = transported.replace(orientations=ee.ee_from_grasp(grasp_orientations))
    # Learned labels (audit B1, step 6): the labels are an attractor that leads the arm, so they
    # are not where the arm goes. The path the arm traced when they were learned, carried by the
    # same map, is what the geometry below and the path check measure. That the arm lags the
    # transported attractor as it lagged the learned one is an assumption (the transfer
    # allowance, not yet measured).
    arm = marks = None
    if source.arm_path is not None:
        arm_report = transport_map.check_diffeomorphism(source.arm_path["positions"])
        if not arm_report.consistent_sign or arm_report.min_determinant < min_determinant:
            raise PlanningError(
                f"map_not_a_diffeomorphism: along the arm's path min det(J) "
                f"{arm_report.min_determinant:.3f} (need >= {min_determinant})")
        arm = transport_labels(transport_map, PolicyLabels(
            positions=np.asarray(source.arm_path["positions"], dtype=float),
            orientations=np.asarray(source.arm_path["rotations"], dtype=float),
            gripper=tool_labels.gripper))
        grasp_orientations = unrolled(arm.orientations)
        arm = arm.replace(orientations=ee.ee_from_grasp(grasp_orientations))
        # Where the hand should be when the jaws act: the taught path, carried by the same map.
        gi, ri = carry_indices(tool_labels)
        taught = transport_labels(transport_map, source.demonstration_labels)
        marks = {"positions": np.asarray(taught.positions)[[gi, ri]],
                 "orientations": ee.ee_from_grasp(unrolled(
                     np.asarray(taught.orientations)[[gi, ri]]))}
    geometry = arm if arm is not None else transported

    # --- geometric checks -----------------------------------------------------
    executed = diagnostics["target_grasp"]
    place_grasp = diagnostics["target_place_grasp"]
    if jaw_turned:
        executed, place_grasp = _turned(executed), _turned(place_grasp)
    # A modelled hand: where its fingers stop on the object, from the executed grasp.
    closed = None if ee.is_parallel_jaw else ee.closed_state(points, executed)
    grasp_index, release_index = carry_indices(tool_labels)
    aim_grasp = float(np.linalg.norm(geometry.positions[grasp_index] - executed.tcp))
    aim_release = float(np.linalg.norm(geometry.positions[release_index] - place_grasp.tcp))
    orient_grasp = float(np.degrees(rotation_geodesic(
        grasp_orientations[grasp_index], executed.rotation
    )[0]))
    orient_release = float(np.degrees(rotation_geodesic(
        grasp_orientations[release_index], place_grasp.rotation
    )[0]))
    path = geometry.positions
    # The destination surface reaches as far as the object being set down on it,
    # plus the hand around it; beyond that the hand is past the board (docstring).
    horizontal = np.linalg.norm(points[:, :2] - np.median(points[:, :2], axis=0), axis=1).max()
    clearance_pick, clearance_place = _hand_clearance(
        geometry, grasp_index, release_index, width if closed is None else closed.q,
        support_height, float(target.destination_height), ee,
        destination_xy=target.destination[:2],
        footprint_radius=float(horizontal) + ee.footprint_margin,
    )
    metrics = {
        "n_cloud_points": int(len(points)),
        # Between the pads at the grasp; the executor's Grasp goal width.
        "object_width_closing_m": width,
        "object_extent_closing_m": float(np.ptp(points @ executed.closing)),
        "jaw_verdict": jaw.verdict if jaw else "not_applicable",
        "jaw_points_between_pads": jaw.n_between if jaw else 0,
        "jaw_engagement_m": jaw.engagement if jaw else 0.0,
        "end_effector": ee.name,
        "grasp_tilt_deg": float(np.degrees(np.arccos(np.clip(-executed.approach[2], -1, 1)))),
        "hand_clearance_pick_m": clearance_pick,
        "hand_clearance_place_m": clearance_place,
        "object_height_m": float(points[:, 2].max() - support_height),
        "support_height_m": support_height,
        "support_height_estimated": scene.support_height is None,
        "min_det": float(report.min_determinant),
        "fraction_positive": float(report.fraction_positive),
        "keypoint_residual_m": residual,
        "aim_grasp_m": aim_grasp,
        "aim_release_m": aim_release,
        "orientation_error_grasp_deg": orient_grasp,
        "orientation_error_release_deg": orient_release,
        "grasp_index": int(grasp_index),
        "release_index": int(release_index),
        "target_grasp_rolled": rolled,
        # which half turn the plan took (module docstring): "target" (the sim's, a
        # symmetric target), "source_jaw_turned" (an asymmetric target keeps its grasp),
        # or "none"
        "roll": ("source_jaw_turned" if jaw_turned else "target" if rolled else "none"),
        # a modelled hand: its fingers stopped on the object (DOF order, rad), which
        # touch it, and whether the thumb opposes a finger
        "hand_closed_q": None if closed is None else closed.q.tolist(),
        "hand_contacts_predicted": None if closed is None else closed.contact,
        "hand_opposed_predicted": None if closed is None else bool(closed.opposed),
        "path_min_xyz": path.min(axis=0).tolist(),
        "path_max_xyz": path.max(axis=0).tolist(),
        "path_min_height_above_support_m": float(path[:, 2].min() - support_height),
        "max_label_speed_mps": float(np.linalg.norm(transported.velocities, axis=1).max()),
        "keypoint_box": box,
        "keypoint_orientation": orientation,
        "cube_half_extent": cube_half_extent,
        # what the geometric metrics and the path check are measured on
        "checked_path": "arm path (learned labels)" if arm is not None else "labels",
    }
    return Plan(
        labels=transported,
        source_keypoints=source_set,
        target_keypoints=target_set,
        grasp=executed,
        target_points=points,
        support_height=support_height,
        destination=target.destination,
        destination_height=target.destination_height,
        metrics=metrics,
        diagnostics={
            "target_place_points": diagnostics["target_place_points"],
            "target_place_grasp": place_grasp,
            "transport_map": transport_map,
        },
        metadata={
            "control_freq": int(source.metadata.get("control_freq", 20)),
            "frame": "base",
            "position_reference": ("fingertip centre (grasp point)" if ee.is_parallel_jaw
                                   else "the hand's grasp point: GraspGen-X's fingertip depth "
                                        "less calibrated_depth (end_effector.tool_offset from "
                                        "the controller frame)"),
            "orientation_convention": ee.controller_frame,
            "grasp_to_fr3_tcp": ee.grasp_to_ee.tolist(),
            "end_effector": ee.to_dict(),
            "source_bundle": source.metadata.get("provenance", {}),
            # "static" for a demonstration taught by hand (tpgpt.hardware.teach):
            # one stiffness throughout, so the executor must not read firm and
            # compliant segments into it.
            "source": source.metadata.get("source", "simulation"),
            "source_stiffness": source.metadata.get("stiffness", "demonstrated"),
            # the settings a learned attractor was learned under; the executor runs with them
            "learned_execution": source.metadata.get("learned", {}).get("execution"),
        },
        arm_path=arm,
        arm_margin=(None if arm is None else
                    np.asarray(source.arm_path["margin"], dtype=float).copy()),
        marks=marks,
        arm_table_margin=(None if arm is None or "margin_vertical" not in source.arm_path else
                          np.asarray(source.arm_path["margin_vertical"], dtype=float).copy()),
    )


def _source_symmetric(source) -> bool:
    """Whether the demonstrating hand is symmetric under a half turn about its approach:
    a parallel jaw (every demonstration so far, the Franka Hand)."""
    ee = (source.metadata or {}).get("end_effector") or {}
    return ee.get("kind", "parallel_jaw") == "parallel_jaw"


#: The half turn about the approach, in the grasp convention.
JAW_TURN = np.diag([-1.0, -1.0, 1.0])


def _turned(grasp: GraspFrame) -> GraspFrame:
    """``grasp`` half-turned about its approach (``R J``)."""
    return GraspFrame(tcp=grasp.tcp, approach=grasp.approach, closing=-np.asarray(grasp.closing))


def _hand_clearance(labels: PolicyLabels, grasp_index: int, release_index: int,
                    closed_width, support_height: float,
                    destination_height: float,
                    end_effector: EndEffector | None = None,
                    destination_xy: np.ndarray | None = None,
                    footprint_radius: float | None = None) -> tuple[float, float]:
    """Lowest corner of the hand along the path, above the surface **under it**.

    Labels up to the grasp are measured against the support, labels from the
    release on against the destination surface. The hand is open where the labels
    are open and closed where they are closed: the Franka Hand to the measured jaw
    width (``closed_width``, m), a modelled hand to its fingers stopped on the object
    (``closed_width`` is then its six joint angles). Negative means the hand dips
    below that surface.

    The support is a table and effectively unbounded, so the pick side is every
    label up to the grasp. **The destination surface is not**: it is a shelf
    board, and once the retreat has carried the hand off it there is nothing
    underneath to hit. Measuring the whole retreat against an infinite plane at
    the destination's height refused every candidate for the upper boards --
    25 of 25 on each of four banana captures, 2026-09-24 -- because the taught
    retreat comes back down in *front* of the shelf, 215-314 mm away from the
    slot, 35-53 mm below the board it had just placed on. The exact path check
    passed those same plans (`path_check`, which knows where the boards end).
    So the place side is measured only while the hand is within
    ``footprint_radius`` of ``destination_xy``: where the surface really is.
    Without those two arguments the old, unbounded behaviour is kept.
    """
    ee = end_effector or FRANKA_HAND
    rotations = ee.grasp_from_ee(labels.orientations)

    def lowest(indices):
        return min(ee.lowest_point(labels.positions[k], rotations[k],
                                   closed_width if labels.gripper[k] > 0 else ee.open_state)
                   for k in indices)

    place_labels = range(release_index, len(labels.positions))
    if destination_xy is not None and footprint_radius is not None:
        over = [k for k in place_labels
                if np.linalg.norm(labels.positions[k][:2] - np.asarray(destination_xy)[:2])
                <= float(footprint_radius)]
        # The release itself is always over the destination, whatever the radius.
        place_labels = over or [release_index]
    pick = lowest(range(0, grasp_index + 1)) - support_height
    place = lowest(place_labels) - destination_height
    return float(pick), float(place)


def tcp_axes_at_grasp(plan: Plan) -> np.ndarray:
    """The controller-frame rotation the plan commands at the grasp."""
    return plan.labels.orientations[int(plan.metrics["grasp_index"])]


__all__ = [
    "Plan",
    "PlanningError",
    "TargetScene",
    "default_grasp",
    "estimate_support_height",
    "fr3_tcp_to_grasp",
    "plan",
    "tcp_axes_at_grasp",
]
