"""Grasp records and the GraspGen-X to robosuite frame contract.

GraspGen-X emits a 4x4 pose anchored at the **gripper base**, with ``+Z`` the
approach axis and ``+X`` the closing direction, in the frame of the point cloud
it was given. robosuite is commanded at the ``grip_site``, which sits at the
fingertips. Converting between them is two steps, both of which were measured
rather than assumed (ROBOTICS_NOTES.md section 6):

1. **Translate along the approach axis** by the gripper's base-to-fingertip
   depth, read from its own ``config.json``. This is 103 mm for a Panda and
   195 mm for a Robotiq 2F-140 -- nearly 2x -- which is precisely why a grasp
   cannot be replayed across embodiments as an end-effector pose.
2. **Rotate about the approach axis** so the closing directions agree. Measured
   in simulation, ``grip_site +Z`` is the approach axis for *every* gripper
   tested, so the only difference is whether the jaws close along ``grip_site``
   X or Y.

No filtering, ranking policy or selection happens here: candidates are returned
in full, sorted by the discriminator score the server reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.grasp.grippers import GripperPair, gripper_geometry, resolve_pair

#: Rotation about the approach axis mapping GraspGen-X's ``+X`` closing
#: direction onto a ``grip_site`` whose jaws close along Y.
_ROT_Z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Grasp6D:
    """One grasp candidate, with everything needed to act on it.

    Attributes:
        pose: ``(4, 4)`` SE(3) at the **gripper base**, in the frame of the
            point cloud that produced it (world frame, in this project).
        score: Discriminator confidence in ``[0, 1]``.
        gripper: GraspGen-X gripper name.
        width: Jaw aperture of that gripper, metres. GraspGen-X does not return
            a per-grasp width; this is the hand's maximum opening, read from its
            config.
    """

    pose: np.ndarray
    score: float
    gripper: str
    width: float

    @property
    def position(self) -> np.ndarray:
        """Gripper-base position."""
        return self.pose[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def approach(self) -> np.ndarray:
        """Unit approach axis: the direction the hand advances along."""
        return self.pose[:3, 2]

    @property
    def closing(self) -> np.ndarray:
        """Unit closing axis: the direction the jaws travel."""
        return self.pose[:3, 0]

    def tcp_position(self) -> np.ndarray:
        """Fingertip (TCP) position, base offset along the approach axis."""
        return self.position + self.approach * gripper_geometry(self.gripper).tcp_depth

    def as_dict(self) -> dict:
        return {
            "pose": self.pose.tolist(),
            "score": float(self.score),
            "gripper": self.gripper,
            "width": float(self.width),
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "closing": self.closing.tolist(),
        }


@dataclass
class GraspSet:
    """Ranked grasp candidates for one object and one gripper."""

    grasps: list[Grasp6D]
    gripper: str
    instance: str
    n_cloud_points: int = 0
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.grasps)

    def __iter__(self):
        return iter(self.grasps)

    def __getitem__(self, idx):
        return self.grasps[idx]

    @property
    def best(self) -> Grasp6D | None:
        return self.grasps[0] if self.grasps else None

    @property
    def scores(self) -> np.ndarray:
        return np.array([g.score for g in self.grasps])

    def describe(self) -> str:
        if not self.grasps:
            return f"GraspSet({self.instance}, {self.gripper}): no grasps"
        return (
            f"GraspSet({self.instance}, {self.gripper}): {len(self)} candidates, "
            f"score {self.scores.min():.3f}-{self.scores.max():.3f}"
        )


def build_grasp_set(
    poses: np.ndarray,
    scores: np.ndarray,
    gripper: str,
    instance: str = "",
    n_cloud_points: int = 0,
    metadata: dict | None = None,
) -> GraspSet:
    """Wrap raw server output, sorted best first.

    Sorting is done here because the server does not do it: it concatenates
    per-iteration and per-branch results without a global sort.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    width = gripper_geometry(gripper).aperture
    order = np.argsort(-scores)
    return GraspSet(
        grasps=[
            Grasp6D(pose=poses[i].copy(), score=float(scores[i]), gripper=gripper, width=width)
            for i in order
        ],
        gripper=gripper,
        instance=instance,
        n_cloud_points=int(n_cloud_points),
        metadata=dict(metadata or {}),
    )


def alignment_rotation(pair: GripperPair) -> np.ndarray:
    """Rotation taking the GraspGen-X grasp frame to robosuite's ``grip_site``.

    Raises:
        ValueError: if the pair's frame convention has not been measured. This
            is deliberate. A wrong rotation about the approach axis turns a
            side grasp into one that closes along the object's long dimension,
            and it fails silently -- the pose still looks plausible. Guessing an
            identity here would be worse than refusing.
    """
    if not pair.frame_verified:
        raise ValueError(
            f"the grip_site frame convention for {pair.robosuite} has not been "
            "measured, so a grasp cannot be converted into an end-effector "
            "command for it. Measure its closing axis and set closing_axis in "
            "tpgpt/grasp/grippers.py. "
            f"Verified grippers: see VERIFIED_PAIRS."
        )
    return np.eye(3) if pair.closing_axis == "x" else _ROT_Z_90


def grasp_to_eef_pose(grasp: Grasp6D, gripper: str | GripperPair | None = None):
    """Convert a grasp into a robosuite end-effector target.

    Args:
        grasp: A candidate, whose pose is at the gripper base.
        gripper: The pair to convert for. Defaults to the grasp's own gripper,
            which is the normal case; pass another to ask what pose a
            *different* hand would need for the same contact.

    Returns:
        ``(position, rotation)`` for the ``grip_site``: a ``(3,)`` world
        position and a ``(3, 3)`` rotation, ready for
        :meth:`~tpgpt.sim.controllers.cartesian_impedance.CartesianImpedanceController.action`.
    """
    pair = gripper if isinstance(gripper, GripperPair) else resolve_pair(gripper or grasp.gripper)
    depth = gripper_geometry(pair.graspgen).tcp_depth
    position = grasp.position + grasp.approach * depth
    rotation = grasp.rotation @ alignment_rotation(pair)
    return position, rotation


def approach_waypoint(grasp: Grasp6D, standoff: float = 0.10, gripper=None) -> np.ndarray:
    """Pre-grasp position, backed off along the approach axis.

    The hand must arrive along its own approach direction rather than dropping
    straight down, or the fingers sweep through the object on the way in.
    """
    position, _ = grasp_to_eef_pose(grasp, gripper)
    return position - grasp.approach * standoff
