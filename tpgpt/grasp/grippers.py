"""Bridge between robosuite grippers and GraspGen-X gripper models.

GraspGen-X conditions on a gripper's **swept volume**, not on trained-in weights,
so one checkpoint generates grasps for any hand whose ``config.json`` it can
read. Nine of the hands it ships are also simulated by robosuite, which is what
makes cross-embodiment transport testable here rather than hypothetical.

The pairs span all three of GraspGen-X's kinematic families and vary widely in
the two numbers that matter geometrically -- jaw aperture (50 to 125 mm) and the
base-to-fingertip depth (103 to 195 mm, nearly 2x). That depth is why a grasp
cannot simply be replayed across embodiments: the same contact on the same
object puts the *end effector* in a very different place for each hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Where the gripper descriptions checkout lives.
#:
#: ``GRASPGENX_GRIPPER_CFG_DIR`` points at the checkout *root*; the configs sit
#: under ``gripper_descriptions/assets/x_grippers/<name>/config.json`` beneath
#: it. This mirrors how GraspGen-X's own ``resolve_gripper_info`` searches.
DEFAULT_DESCRIPTIONS_ROOT = Path(
    os.environ.get(
        "GRASPGENX_GRIPPER_CFG_DIR",
        "/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/assets/gripper_descriptions",
    )
)
_CONFIG_SUBPATH = Path("gripper_descriptions/assets/x_grippers")


@dataclass(frozen=True)
class GripperPair:
    """A robosuite gripper and the GraspGen-X model that represents it.

    Attributes:
        closing_axis: Which axis of robosuite's ``grip_site`` frame the jaws
            close along, ``"x"`` or ``"y"``, or ``None`` when it has not been
            measured. GraspGen-X always emits ``+X`` = closing and ``+Z`` =
            approach, and the approach axis was measured to be ``grip_site +Z``
            for **every** gripper, so this single letter is the whole of the
            frame difference.

            ``None`` is not a default to fall back on: it means a grasp pose
            cannot be converted into an end-effector command for this hand
            without measuring it first, and
            :func:`~tpgpt.grasp.grasps.grasp_to_eef_pose` refuses rather than
            guessing. The three-finger hands have no single closing axis, and
            the Yumi's fingers measured 65 degrees off axis.
    """

    robosuite: str
    graspgen: str
    #: robosuite robots this gripper can be mounted on, for scene construction.
    robots: tuple[str, ...] = ("Panda",)
    closing_axis: str | None = None

    @property
    def frame_verified(self) -> bool:
        """Whether the grip_site frame convention has been measured."""
        return self.closing_axis is not None

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.robosuite} <-> {self.graspgen}"


#: Gripper pairs, keyed by a short name used in configs and on the CLI.
#:
#: ``closing_axis`` was measured in simulation, not assumed: the two opposing
#: finger bodies were located at reset and their separation expressed in the
#: ``grip_site`` frame. See ROBOTICS_NOTES.md section 6.
GRIPPER_PAIRS: dict[str, GripperPair] = {
    "panda": GripperPair("PandaGripper", "franka_panda", ("Panda",), "x"),
    "robotiq85": GripperPair(
        "Robotiq85Gripper", "robotiq_2f_85", ("Panda", "UR5e", "Kinova3", "IIWA"), "x"
    ),
    "robotiq140": GripperPair(
        "Robotiq140Gripper", "robotiq_2f_140", ("Panda", "UR5e", "IIWA"), "x"
    ),
    "rethink": GripperPair("RethinkGripper", "sawyer_hand", ("Sawyer", "Panda"), "x"),
    "xarm": GripperPair("XArm7Gripper", "xarm_hand", ("XArm7", "Panda"), "y"),
    "umi": GripperPair("UMIGripper", "franka_umi", ("Panda",), "y"),
    # Frame convention not established. Three-finger hands have no single
    # closing axis; the Yumi's fingers measured 65 degrees off any axis.
    "robotiq3f": GripperPair(
        "RobotiqThreeFingerGripper", "robotiq_3f", ("Panda", "UR5e"), None
    ),
    "yumi": GripperPair("YumiRightGripper", "abb_yumi", ("Yumi",), None),
    "inspire": GripperPair("InspireRightHand", "inspire_hand", ("Panda",), None),
}

#: Pairs whose GraspGen-X side is already loaded on the running server.
DEFAULT_PAIRS = ("panda", "robotiq85", "robotiq140")

#: Every pair whose grip_site frame has been measured.
VERIFIED_PAIRS = tuple(k for k, v in GRIPPER_PAIRS.items() if v.frame_verified)


@dataclass(frozen=True)
class GripperGeometry:
    """The geometry GraspGen-X publishes for a hand, in metres."""

    name: str
    #: Jaw aperture when open: the X extent of the open swept volume.
    aperture: float
    #: Distance from the pose origin (the gripper base) to the fingertip frame.
    #:
    #: Note this is the **TCP**, not the lowest point of the hand: the Panda's
    #: finger pads reach about 9 mm past it. Anything doing clearance checks
    #: should use :attr:`bbox`, not this.
    tcp_depth: float
    #: ``parallel_2f`` | ``revolute_2f`` | ``revolute_3f``.
    family: str
    standoff: tuple[float, ...]
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    @property
    def lowest_point(self) -> float:
        """Furthest extent of the hand along its approach axis."""
        return float(self.bbox[1][2]) if self.bbox else self.tcp_depth


def descriptions_root() -> Path:
    """Root of the gripper descriptions checkout."""
    return DEFAULT_DESCRIPTIONS_ROOT


def gripper_config_path(graspgen_name: str) -> Path:
    """Path to a GraspGen-X gripper's ``config.json``."""
    return descriptions_root() / _CONFIG_SUBPATH / graspgen_name / "config.json"


@lru_cache(maxsize=None)
def gripper_geometry(graspgen_name: str) -> GripperGeometry:
    """Read a gripper's published geometry.

    Raises:
        FileNotFoundError: with the searched path, since the usual cause is the
            sibling GraspGen-X checkout being absent or
            ``GRASPGENX_GRIPPER_CFG_DIR`` being unset.
    """
    path = gripper_config_path(graspgen_name)
    if not path.is_file():
        available = sorted(p.name for p in path.parent.parent.glob("*")) if path.parent.parent.is_dir() else []
        raise FileNotFoundError(
            f"No config.json for gripper {graspgen_name!r} at {path}. "
            f"Set GRASPGENX_GRIPPER_CFG_DIR to the gripper_descriptions checkout root. "
            + (f"Available: {available}" if available else "")
        )
    config = json.loads(path.read_text())
    sweep = config.get("sweep_volume", {})
    bbox = config.get("bbox")
    return GripperGeometry(
        name=graspgen_name,
        aperture=float(sweep["extents"][0]),
        tcp_depth=float(config["fingertip"][-1]),
        family=str(config.get("type", "unknown")),
        standoff=tuple(float(s) for s in config.get("standoff", ())),
        bbox=(tuple(bbox[0]), tuple(bbox[1])) if bbox else None,
    )


def resolve_pair(name: str) -> GripperPair:
    """Look up a pair by short name, robosuite name, or GraspGen-X name."""
    if name in GRIPPER_PAIRS:
        return GRIPPER_PAIRS[name]
    for pair in GRIPPER_PAIRS.values():
        if name in (pair.robosuite, pair.graspgen):
            return pair
    raise KeyError(
        f"unknown gripper {name!r}; known short names: {sorted(GRIPPER_PAIRS)}"
    )


def summary_table() -> str:
    """Human-readable table of every pair and its geometry."""
    lines = [
        f"{'short':11s} {'robosuite':26s} {'graspgen':16s} {'family':13s} "
        f"{'aperture':>9s} {'tcp depth':>10s}  {'closing':7s}"
    ]
    for short, pair in GRIPPER_PAIRS.items():
        axis = pair.closing_axis or "unverified"
        try:
            geom = gripper_geometry(pair.graspgen)
            lines.append(
                f"{short:11s} {pair.robosuite:26s} {pair.graspgen:16s} "
                f"{geom.family:13s} {geom.aperture * 1000:7.1f}mm {geom.tcp_depth * 1000:8.1f}mm  {axis:7s}"
            )
        except FileNotFoundError:
            lines.append(
                f"{short:11s} {pair.robosuite:26s} {pair.graspgen:16s} <config missing>"
            )
    return "\n".join(lines)
