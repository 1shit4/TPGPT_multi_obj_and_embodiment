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
For the Inspire hand the grasp is a scripted side grasp
(:mod:`tpgpt.hardware.side_grasp`), and the hand is judged by its envelope box
against the table and the destination along the path.

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

    @property
    def dt(self) -> float:
        return 1.0 / float(self.metadata.get("control_freq", 20))

    # ------------------------------------------------------------------ io
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.labels.save(directory / "transported_labels.npz")
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
        raise PlanningError(f"no_grasp: the {ee.name} hand needs an explicit grasp "
                            "(tpgpt.hardware.side_grasp); the top-down default is a parallel jaw's")
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
    source_set, target_set, diagnostics = scene_keypoints(
        source.placement, target, tool_labels,
        include_contacts=False, box=box, orientation=orientation,
        cube_half_extent=cube_half_extent,
    )
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
    grasp_orientations = transported.orientations
    transported = transported.replace(orientations=ee.ee_from_grasp(grasp_orientations))

    # --- geometric checks -----------------------------------------------------
    executed = diagnostics["target_grasp"]
    place_grasp = diagnostics["target_place_grasp"]
    grasp_index, release_index = carry_indices(tool_labels)
    aim_grasp = float(np.linalg.norm(transported.positions[grasp_index] - executed.tcp))
    aim_release = float(np.linalg.norm(transported.positions[release_index] - place_grasp.tcp))
    orient_grasp = float(np.degrees(rotation_geodesic(
        grasp_orientations[grasp_index], executed.rotation
    )[0]))
    orient_release = float(np.degrees(rotation_geodesic(
        grasp_orientations[release_index], place_grasp.rotation
    )[0]))
    path = transported.positions
    clearance_pick, clearance_place = _hand_clearance(
        transported, grasp_index, release_index, width, support_height,
        float(target.destination_height), ee,
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
        "target_grasp_rolled": bool(diagnostics["target_grasp_rolled"]),
        "path_min_xyz": path.min(axis=0).tolist(),
        "path_max_xyz": path.max(axis=0).tolist(),
        "path_min_height_above_support_m": float(path[:, 2].min() - support_height),
        "max_label_speed_mps": float(np.linalg.norm(transported.velocities, axis=1).max()),
        "keypoint_box": box,
        "keypoint_orientation": orientation,
        "cube_half_extent": cube_half_extent,
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
                                   else "palm grasp point (end_effector.tool_offset from the "
                                        "controller frame)"),
            "orientation_convention": ee.controller_frame,
            "grasp_to_fr3_tcp": ee.grasp_to_ee.tolist(),
            "end_effector": ee.to_dict(),
            "source_bundle": source.metadata.get("provenance", {}),
            # "static" for a demonstration taught by hand (tpgpt.hardware.teach):
            # one stiffness throughout, so the executor must not read firm and
            # compliant segments into it.
            "source": source.metadata.get("source", "simulation"),
            "source_stiffness": source.metadata.get("stiffness", "demonstrated"),
        },
    )


def _hand_clearance(labels: PolicyLabels, grasp_index: int, release_index: int,
                    closed_width: float, support_height: float,
                    destination_height: float,
                    end_effector: EndEffector | None = None) -> tuple[float, float]:
    """Lowest corner of the hand along the path, above the surface under it.

    Labels up to the grasp are measured against the support, labels from the
    release on against the destination surface. A Franka Hand is open where the
    labels are open and closed to the measured jaw width where they are closed;
    an Inspire hand is its fixed envelope box. Negative means the hand dips
    below that surface.
    """
    ee = end_effector or FRANKA_HAND
    rotations = ee.grasp_from_ee(labels.orientations)

    def lowest(indices):
        return min(ee.lowest_point(labels.positions[k], rotations[k],
                                   closed_width if labels.gripper[k] > 0 else MAX_OPENING)
                   for k in indices)

    pick = lowest(range(0, grasp_index + 1)) - support_height
    place = lowest(range(release_index, len(labels.positions))) - destination_height
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
