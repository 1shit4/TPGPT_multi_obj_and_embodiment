"""The single source demonstration, packaged for a robot that has no simulator.

``tpgpt.experiments.pipeline.run`` obtains its source from
:func:`~tpgpt.experiments.reshelving_pipeline.record_source_placement` and then,
before any keypoint is built, re-expresses the labels at the fingertip
(``_to_tool_frame``) and in the grasp convention (``to_grasp_convention``). A
:class:`SourceBundle` is exactly that result, saved, so the hardware planner can
run the rest of the pipeline without importing MuJoCo.

**The one step that is new is the change of frame.** The simulation records in
robosuite's *world* frame, whose origin is under the table and ~0.9 m below the
robot. The real robot's targets are in its *base* frame. Everything in the
bundle is therefore moved into the **source robot's base frame** at export,
using the robot base pose read from the simulator.

That is not needed for correctness in principle -- the affine stage of ``phi``
(Sec. III-E-a) absorbs any rigid offset between source and target -- but it
keeps that affine stage close to identity, so the nonlinear residual the GP
has to explain is the scene difference and not a 0.9 m translation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.hardware.frames import (
    apply_to_points,
    apply_to_rotations,
    apply_to_tensors,
    apply_to_vectors,
)
from tpgpt.sim.keypoints import GraspFrame, ObjectPlacement
from tpgpt.transport.labels import PolicyLabels

LABELS_FILE = "tool_labels.npz"
ARM_PATH_FILE = "arm_path.npz"
TAUGHT_LABELS_FILE = "taught_labels.npz"
PLACEMENT_FILE = "placement.npz"
MANIFEST_FILE = "manifest.json"


def transform_labels(labels: PolicyLabels, T) -> PolicyLabels:
    """Re-express every label family in another frame.

    Positions move rigidly, velocities and orientations rotate, stiffness and
    damping rotate by congruence. Phase, gripper and uncertainty magnitudes are
    frame independent. Uses :meth:`PolicyLabels.replace` so no family is
    dropped -- rebuilding field by field is how ``pipeline.run`` once lost the
    stiffness channel.
    """
    changes = {"positions": apply_to_points(T, labels.positions)}
    if labels.velocities is not None:
        changes["velocities"] = apply_to_vectors(T, labels.velocities)
    if labels.orientations is not None:
        changes["orientations"] = apply_to_rotations(T, labels.orientations)
    if labels.stiffness is not None:
        changes["stiffness"] = apply_to_tensors(T, labels.stiffness)
    if labels.damping is not None:
        changes["damping"] = apply_to_tensors(T, labels.damping)
    if labels.velocity_std is not None:
        # A per-axis standard deviation does not rotate exactly; keep its norm
        # per label on every axis rather than inventing a covariance.
        norm = np.linalg.norm(labels.velocity_std, axis=1, keepdims=True) / np.sqrt(3.0)
        changes["velocity_std"] = np.repeat(norm, 3, axis=1)
    return labels.replace(**changes)


def transform_placement(placement: ObjectPlacement, T) -> ObjectPlacement:
    """Re-express an :class:`ObjectPlacement` in another frame.

    ``support_height`` and ``destination_height`` are heights *along the support
    normal*, which ``scene_keypoints`` takes as world ``+Z``. That is only
    preserved by a transform whose rotation keeps ``+Z`` fixed, so anything else
    is refused rather than producing heights measured along a tilted axis.
    """
    T = np.asarray(T, dtype=float)
    if abs(float(T[2, 2]) - 1.0) > 1e-6:
        raise ValueError(
            "this transform tilts the vertical, so support heights would change "
            "meaning; the source robot base must be upright in the simulator"
        )
    dz = float(T[2, 3])
    grasp = placement.grasp
    return ObjectPlacement(
        points=apply_to_points(T, placement.points),
        grasp=GraspFrame(
            tcp=apply_to_points(T, grasp.tcp)[0],
            approach=apply_to_vectors(T, grasp.approach)[0],
            closing=apply_to_vectors(T, grasp.closing)[0],
        ),
        support_height=float(placement.support_height) + dz,
        destination=apply_to_points(T, placement.destination)[0],
        destination_height=float(placement.destination_height) + dz,
        metadata=dict(placement.metadata),
    )


@dataclass
class SourceBundle:
    """Fingertip labels and source scene, both in the source robot's base frame.

    Attributes:
        tool_labels: The demonstration at the **fingertip**, orientations in the
            **grasp convention** -- the exact input ``scene_keypoints`` and
            ``transport_labels`` receive in ``pipeline.run``.
        placement: Source object cloud, grasp, support and destination.
        metadata: Provenance: git state, seed, base pose, offsets used.
        arm_path: Where the arm really went while commanded with these labels, when the
            labels are a learned attractor (``tpgpt.hardware.replay.export_bundle``):
            ``positions`` (N, 3) grasp point, ``rotations`` (N, 3, 3) grasp convention and
            ``margin`` (N,) m, its run-to-run spread (``margin_vertical``: its vertical part,
            the margin kept from the table). The planner checks this path, carried
            through the same map, instead of the labels (the arm does not go where the spring
            pulls). None for a bundle whose labels are the path itself.
        taught_labels: With learned labels, the demonstration they were learned from. What
            the object did is read off these (``carry_transform``: the hand's pose at the grasp
            and the release), since the learned labels lead the hand and did not hold it.
    """

    tool_labels: PolicyLabels
    placement: ObjectPlacement
    metadata: dict = field(default_factory=dict)
    arm_path: dict | None = None
    taught_labels: PolicyLabels | None = None

    @property
    def demonstration_labels(self) -> PolicyLabels:
        """The labels that say where the hand, and so the object, really went: the taught
        ones for learned labels, else the labels themselves."""
        return self.taught_labels if self.taught_labels is not None else self.tool_labels

    @property
    def dt(self) -> float:
        return 1.0 / float(self.metadata.get("control_freq", 20))

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tool_labels.save(directory / LABELS_FILE)
        p = self.placement
        np.savez_compressed(
            directory / PLACEMENT_FILE,
            points=p.points,
            grasp_tcp=p.grasp.tcp,
            grasp_approach=p.grasp.approach,
            grasp_closing=p.grasp.closing,
            support_height=np.array(p.support_height),
            destination=np.asarray(p.destination, dtype=float),
            destination_height=np.array(p.destination_height),
        )
        (directory / MANIFEST_FILE).write_text(json.dumps(self.metadata, indent=2, default=str))
        if self.arm_path is not None:
            np.savez_compressed(directory / ARM_PATH_FILE,
                                **{k: np.asarray(v) for k, v in self.arm_path.items()})
        if self.taught_labels is not None:
            self.taught_labels.save(directory / TAUGHT_LABELS_FILE)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "SourceBundle":
        directory = Path(directory)
        if not (directory / LABELS_FILE).exists():
            raise FileNotFoundError(
                f"{directory} has no {LABELS_FILE}; create it with "
                "`python -m tpgpt.hardware.export_source --out <dir>`"
            )
        labels = PolicyLabels.load(directory / LABELS_FILE)
        with np.load(directory / PLACEMENT_FILE, allow_pickle=False) as d:
            placement = ObjectPlacement(
                points=d["points"],
                grasp=GraspFrame(
                    tcp=d["grasp_tcp"], approach=d["grasp_approach"], closing=d["grasp_closing"]
                ),
                support_height=float(d["support_height"]),
                destination=d["destination"],
                destination_height=float(d["destination_height"]),
            )
        manifest = directory / MANIFEST_FILE
        metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
        arm_path = None
        if (directory / ARM_PATH_FILE).exists():
            with np.load(directory / ARM_PATH_FILE, allow_pickle=False) as d:
                arm_path = {k: d[k] for k in d.files}
        taught = (PolicyLabels.load(directory / TAUGHT_LABELS_FILE)
                  if (directory / TAUGHT_LABELS_FILE).exists() else None)
        return cls(tool_labels=labels, placement=placement, metadata=metadata, arm_path=arm_path,
                   taught_labels=taught)
