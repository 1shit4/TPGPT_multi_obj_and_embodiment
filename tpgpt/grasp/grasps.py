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

from tpgpt.grasp.grippers import (
    GripperPair,
    gripper_frame,
    gripper_geometry,
    resolve_pair,
)

def _rotation_about_z(degrees: float) -> np.ndarray:
    """Rotation about the approach axis by ``degrees``."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def alignment_rotation(pair: GripperPair | str) -> np.ndarray:
    """Rotation taking the GraspGen-X grasp frame to robosuite's ``grip_site``.

    GraspGen-X emits ``+Z`` = approach and ``+X`` = closing, and the approach
    axis is ``grip_site +Z`` for every gripper measured, so the whole frame
    difference is one rotation about the approach axis by the hand's measured
    ``closing_angle``.

    This used to be a choice between identity and 90 degrees, which was enough
    for parallel jaws aligned to an axis and refused everything else. Measuring
    an angle instead is what admits the three-finger, five-finger and off-axis
    hands: the Inspire hand's fingers travel 7.4 degrees off ``+X``, which
    neither of the two old options could express.

    Raises:
        ValueError: if the pair's frame convention has not been measured. This
            is deliberate. A wrong rotation about the approach axis turns a
            side grasp into one that closes along the object's long dimension,
            and it fails silently -- the pose still looks plausible. Guessing an
            identity here would be worse than refusing.
    """
    pair = _as_pair(pair)
    if not pair.frame_verified:
        raise ValueError(
            f"the grip_site frame convention for {pair.robosuite} has not been "
            "measured, so a grasp cannot be converted into an end-effector "
            "command for it. Measure it with "
            "tpgpt.grasp.grippers.measure_closing_angle and set closing_angle "
            f"in tpgpt/grasp/grippers.py. Verified grippers: see VERIFIED_PAIRS."
        )
    frame = gripper_frame(_short_name(pair))
    if frame is not None:
        return np.asarray(frame["alignment"], dtype=float)
    return _rotation_about_z(pair.closing_angle)


def _short_name(pair: GripperPair) -> str:
    """Registry key for a pair, for looking up its measured frame."""
    from tpgpt.grasp.grippers import GRIPPER_PAIRS

    return next((k for k, v in GRIPPER_PAIRS.items() if v is pair), "")


def _as_pair(gripper) -> GripperPair:
    """Accept either a registry key or the pair itself.

    Named hands travel through this codebase as strings far more often than as
    objects, and the lookup below is by *identity*, so a string used to fall
    through to "this hand has never been measured" and return a zero offset.
    That is silent and wrong in the worst way: a zero contact offset is a
    perfectly plausible number, so the resulting 41 mm error looked like the
    arm missing rather than like a lookup that never happened.
    """
    return gripper if isinstance(gripper, GripperPair) else resolve_pair(gripper)


def contact_offset(pair: GripperPair | str) -> np.ndarray:
    """Where a grasped object sits, in ``grip_site`` coordinates.

    Starts from the geometric measurement -- the centroid of the distal third
    of the fingers once closed -- and applies the depth **calibrated in
    physics**, because geometry alone could not settle it. Three different
    geometric definitions were tried across the nine hands and each worked for
    six and failed three: long finger links, a flipped frame and an
    anthropomorphic thumb do not share one rule. So the depth is found by
    sweeping it and keeping the middle of the widest band that actually lifts
    the reference object.

    That sweep also measures how forgiving each hand is, and they differ far
    more than expected: the parallel jaws tolerate 120 to 135 mm of depth error,
    the UMI only 15 mm.

    Zero when the gripper has not been measured at all, which reduces to
    commanding ``grip_site`` straight to the grasp's contact point.

    Args:
        pair: A :class:`GripperPair` or its registry key.
    """
    pair = _as_pair(pair)
    frame = gripper_frame(_short_name(pair))
    if frame is None:
        return np.zeros(3)
    offset = np.asarray(frame["contact_offset"], dtype=float)
    depth = frame.get("calibrated_depth")
    if depth is not None:
        # An extra push of ``depth`` along the approach is the same as moving
        # the contact back along it in grip_site coordinates.
        offset = offset - float(depth) * np.asarray(frame["approach_in_site"], dtype=float)
    return offset


def to_grasp_convention(rotations: np.ndarray, gripper) -> np.ndarray:
    """Re-express wrist (``grip_site``) orientations in the grasp convention.

    Two coordinate systems are in play and they are **not** the same, by
    0.2 degrees on a Robotiq 2F-85 and by 90 on an XArm:

    * the **grasp convention** -- GraspGen-X's, ``+Z`` the approach and ``+X``
      the closing direction. It is uniform across every hand, which is what
      makes it the right place to do geometry that has to hold across
      embodiments;
    * the **wrist convention** -- each gripper model's own ``grip_site``, whose
      orientation relative to the fingers was chosen by whoever authored that
      model. Inverse kinematics aims this one, and the controller commands it.

    :func:`alignment_rotation` is the measured, constant, per-hand rotation
    between them, and :func:`grasp_to_eef_pose` already applies it for a single
    grasp. These two helpers do the same for a whole trajectory, which is what a
    transported demonstration is.

    **Why this has to be explicit.** A demonstration is recorded in the
    demonstrating hand's wrist convention, while a keypoint cube built from a
    planner grasp is in the grasp convention. Transporting one into the other
    without saying so silently attaches the *source* hand's alignment to the
    *target* hand's command. Measured on the Tier 2 replay, the commanded wrist
    orientation came out 0.6 to 91 degrees from the one ``grasp_to_eef_pose``
    would have given, depending on the hand -- and neither
    ``alignment_rotation`` nor ``grasp_to_eef_pose`` appeared anywhere in that
    code path. Section 7.34.

    Args:
        rotations: ``(3, 3)`` or ``(n, 3, 3)`` wrist orientations.
        gripper: Registry short name, robosuite name, or :class:`GripperPair`.

    Returns:
        The same shape, in the grasp convention.
    """
    R = np.asarray(rotations, dtype=float)
    return np.einsum("...ij,jk->...ik", R, alignment_rotation(gripper).T)


def to_wrist_convention(rotations: np.ndarray, gripper) -> np.ndarray:
    """Re-express grasp-convention orientations as ``grip_site`` orientations.

    The inverse of :func:`to_grasp_convention`; this is the direction a plan has
    to travel before it can be commanded, because inverse kinematics aims the
    ``grip_site``. See that function for why the two conventions differ and what
    it cost to conflate them.
    """
    R = np.asarray(rotations, dtype=float)
    return np.einsum("...ij,jk->...ik", R, alignment_rotation(gripper))


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
    rotation = grasp.rotation @ alignment_rotation(pair)

    # Where the object should end up: the grasp's own contact point, from
    # GraspGen-X's fingertip depth in its own frame.
    contact = grasp.position + grasp.approach * gripper_geometry(pair.graspgen).tcp_depth
    # Then back out to wherever this hand's grip_site has to be for the object
    # to land there. The offset is a full 3-vector, not a depth: on the Inspire
    # hand the contact sits 100 mm off the approach axis, because the thumb
    # opposes the fingers from one side.
    position = contact - rotation @ contact_offset(pair)
    return position, rotation


def approach_waypoint(grasp: Grasp6D, standoff: float = 0.10, gripper=None) -> np.ndarray:
    """Pre-grasp position, backed off along the approach axis.

    The hand must arrive along its own approach direction rather than dropping
    straight down, or the fingers sweep through the object on the way in.
    """
    position, _ = grasp_to_eef_pose(grasp, gripper)
    return position - grasp.approach * standoff
