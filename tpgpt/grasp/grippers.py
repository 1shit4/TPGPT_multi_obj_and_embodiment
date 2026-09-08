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
        closing_angle: Rotation about the approach axis, in **degrees**, taking
            GraspGen-X's ``+X`` closing direction onto the direction this hand's
            fingers actually travel in robosuite's ``grip_site`` frame.
            ``None`` means it has not been measured.

            GraspGen-X always emits ``+X`` = closing and ``+Z`` = approach, and
            the approach axis is ``grip_site +Z`` for **every** gripper, so this
            single angle is the whole of the frame difference.

            ``None`` is not a default to fall back on: it means a grasp pose
            cannot be converted into an end-effector command for this hand
            without measuring it first, and
            :func:`~tpgpt.grasp.grasps.grasp_to_eef_pose` refuses rather than
            guessing. A wrong angle fails *silently* -- the pose still looks
            plausible while the jaws close along the object's long side.

            Angles are normalised to ``[-90, 90)`` because a jaw axis is an
            axis: closing along ``+X`` and along ``-X`` are the same grasp.
        eef_depth: Distance, in metres, from the gripper's root body to
            robosuite's ``grip_site`` along the approach axis -- **measured in
            simulation**, not read from the GraspGen-X config.

            The two disagree, sometimes wildly, because ``grip_site`` is placed
            by whoever wrote each robosuite gripper model and is not required to
            sit at the fingertips. Measured against the config's ``fingertip``
            value: Panda -6 mm, Rethink -1 mm, Robotiq 2F-85 +9 mm, XArm +11 mm,
            Yumi -28 mm, Robotiq 3F -40 mm, Robotiq 2F-140 **+75 mm**, and for
            the UMI and Inspire hands ``grip_site`` sits exactly *at the base*,
            so the config value is out by the whole 177 mm and 150 mm.

            Using the config value for those drove the hand that far past the
            object -- through the table for the Inspire hand, which never
            touched the can at all. Since it is robosuite's ``grip_site`` that
            is being commanded, it is robosuite's offset that has to be used.
        anisotropy: How well a *single* closing axis describes the hand, from
            the same measurement -- the ratio of the first to the second
            singular value of the finger displacements. A parallel jaw scores in
            the hundreds or higher because every finger moves along one line. A
            multi-finger hand scores single digits, and for those the angle is
            the dominant opposition direction rather than an exact description.
    """

    robosuite: str
    graspgen: str
    #: robosuite robots this gripper can be mounted on, for scene construction.
    robots: tuple[str, ...] = ("Panda",)
    closing_angle: float | None = None
    anisotropy: float | None = None
    eef_depth: float | None = None

    @property
    def frame_verified(self) -> bool:
        """Whether the grip_site frame convention has been measured."""
        return self.closing_angle is not None

    @property
    def single_axis(self) -> bool:
        """Whether one closing axis describes this hand well.

        False for the multi-finger hands, whose fingers do not travel along a
        common line. They are still usable -- the dominant opposition direction
        is what a pinch grasp uses -- but the approximation is worth surfacing.
        """
        return self.anisotropy is not None and self.anisotropy >= SINGLE_AXIS_ANISOTROPY

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.robosuite} <-> {self.graspgen}"


#: Gripper pairs, keyed by a short name used in configs and on the CLI.
#:
#: ``closing_angle`` and ``anisotropy`` are **measured**, by
#: :func:`measure_closing_angle`, not assumed. See ROBOTICS_NOTES.md section 7.
GRIPPER_PAIRS: dict[str, GripperPair] = {
    "panda": GripperPair(
        "PandaGripper", "franka_panda", ("Panda",), 0.0, 6.9e11, 0.097
    ),
    "robotiq85": GripperPair(
        "Robotiq85Gripper", "robotiq_2f_85",
        ("Panda", "UR5e", "Kinova3", "IIWA"), 0.0, 553.0, 0.145,
    ),
    "robotiq140": GripperPair(
        "Robotiq140Gripper", "robotiq_2f_140",
        ("Panda", "UR5e", "IIWA"), 0.0, 1291.2, 0.270,
    ),
    "rethink": GripperPair(
        "RethinkGripper", "sawyer_hand", ("Sawyer", "Panda"), 0.0, 1.2e12, 0.109
    ),
    "xarm": GripperPair(
        "XArm7Gripper", "xarm_hand", ("XArm7", "Panda"), -90.0, 37608.6, 0.147
    ),
    "umi": GripperPair(
        "UMIGripper", "franka_umi", ("Panda",), -90.0, 732277.2, 0.000
    ),
    # Multi-finger and off-axis hands. Their fingers do not travel along one
    # line, so ``anisotropy`` is single digits and the angle is the dominant
    # opposition direction. The physics test in tests/integration decides
    # whether that is good enough.
    "robotiq3f": GripperPair(
        "RobotiqThreeFingerGripper", "robotiq_3f", ("Panda", "UR5e"), 0.2, 4.3, 0.150
    ),
    "yumi": GripperPair(
        "YumiRightGripper", "abb_yumi", ("Yumi", "Panda"), 0.0, 4.6e8, 0.097
    ),
    "inspire": GripperPair(
        "InspireRightHand", "inspire_hand", ("Panda",), 7.4, 6.3, 0.000
    ),
}

#: Pairs whose GraspGen-X side is already loaded on the running server.
DEFAULT_PAIRS = ("panda", "robotiq85", "robotiq140")

#: Every pair whose grip_site frame has been measured. All nine.
MEASURED_PAIRS = tuple(k for k, v in GRIPPER_PAIRS.items() if v.frame_verified)


def _physics_verified() -> tuple[str, ...]:
    """Pairs that actually pick an object up, not merely convert cleanly.

    Measuring a hand's frame says where to send it. It does not say the hand
    can execute the grasp. The Inspire hand is the case that separates the two:
    its frame measures cleanly, its conversion is well formed, and it lifted
    nothing across 24 combinations of object, grasp yaw and contact offset.
    A five-fingered hand driven by one open/close command does not pinch a can
    from above, and no amount of frame correction changes that.

    So campaigns run over this list, and the difference between the two lists
    is itself a result worth reporting.
    """
    frames = _frames()
    if not frames:
        return MEASURED_PAIRS
    return tuple(
        k for k in MEASURED_PAIRS
        if frames.get(k, {}).get("calibrated_depth") is not None
    )


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
        f"{'aperture':>9s} {'tcp depth':>10s}  {'closing':10s} {'kind':6s}"
    ]
    for short, pair in GRIPPER_PAIRS.items():
        axis = (
            f"{pair.closing_angle:+.1f}d" if pair.frame_verified else "unverified"
        )
        try:
            geom = gripper_geometry(pair.graspgen)
            lines.append(
                f"{short:11s} {pair.robosuite:26s} {pair.graspgen:16s} "
                f"{geom.family:13s} {geom.aperture * 1000:7.1f}mm "
                f"{geom.tcp_depth * 1000:8.1f}mm  {axis:10s} "
                f"{'1-axis' if pair.single_axis else 'multi '}"
            )
        except FileNotFoundError:
            lines.append(
                f"{short:11s} {pair.robosuite:26s} {pair.graspgen:16s} <config missing>"
            )
    return "\n".join(lines)


#: Anisotropy above which one closing axis describes a hand well.
#:
#: Every parallel jaw measured here scores at least 553; every multi-finger hand
#: scores under 7. Nothing lands between, so the threshold is not a tuned value.
SINGLE_AXIS_ANISOTROPY = 50.0


def measure_closing_angle(
    robosuite_name: str,
    robot: str = "Panda",
    steps: int = 40,
    min_travel: float = 2e-4,
) -> tuple[float, float, int]:
    """Measure which way a gripper's fingers travel, by closing it.

    Builds the hand in simulation, drives it fully open and then fully closed,
    and takes the principal direction of the finger displacements expressed in
    the ``grip_site`` frame, with the approach component removed.

    Measuring the *motion* rather than the finger geometry is what makes this
    work for every hand. Two naming-based attempts failed first: robosuite's
    ``important_geoms`` lists names that do not exist in the compiled model for
    some grippers (the XArm's pads are ``gripper0_right_left_finger_pad_1``
    against a listed ``gripper0_finger1_pad_collision``), the Yumi's lists are
    empty entirely, and taking the separation between named pad groups put the
    UMI's fingers 3.17 m apart. Displacement needs no names and no special case
    for three fingers or five.

    Args:
        robosuite_name: Gripper class name, e.g. ``"PandaGripper"``.
        robot: Arm to mount it on. Only the gripper's own motion is measured, so
            the arm does not matter beyond compatibility.
        steps: Control steps to hold each command for.
        min_travel: Geoms moving less than this are treated as structure rather
            than fingers.

    Returns:
        ``(angle_degrees, anisotropy, n_moving_geoms)``. The angle is normalised
        to ``[-90, 90)``: a jaw axis is an axis, so ``+X`` and ``-X`` closing are
        the same grasp. Anisotropy is the ratio of the first to the second
        singular value of the displacements -- how well one axis describes the
        hand.

    Raises:
        RuntimeError: if too few geoms move, which means the gripper never
            actuated and no angle can be claimed.
    """
    import robosuite as suite

    env = suite.make(
        "Lift",
        robots=robot,
        gripper_types=robosuite_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        sim = env.sim
        gripper = env.robots[0].gripper
        gripper = gripper["right"] if isinstance(gripper, dict) else gripper
        geom_ids = [
            i for i, name in enumerate(sim.model.geom_names)
            if name and gripper.naming_prefix.split("_")[0] in name
        ]
        site = sim.model.site_name2id(gripper.important_sites["grip_site"])

        def positions():
            return np.array([sim.data.geom_xpos[i].copy() for i in geom_ids])

        action = np.zeros(env.action_dim)
        action[-1] = -1.0                       # fully open
        for _ in range(steps):
            env.step(action)
        before = positions()
        rotation = np.array(sim.data.site_xmat[site]).reshape(3, 3)

        action[-1] = 1.0                        # fully closed
        for _ in range(steps):
            env.step(action)

        # Displacements in the grip_site frame, approach component dropped.
        travel = (positions() - before) @ rotation
        moving = travel[np.linalg.norm(travel, axis=1) > min_travel][:, :2]
        if len(moving) < 2:
            raise RuntimeError(
                f"{robosuite_name}: only {len(moving)} geoms moved when closing; "
                "the gripper did not actuate, so its closing axis is unmeasured"
            )
        _, singular, right = np.linalg.svd(moving, full_matrices=False)
        angle = np.degrees(np.arctan2(right[0][1], right[0][0]))
        anisotropy = float(singular[0] / max(singular[1], 1e-12))
        return float(((angle + 90.0) % 180.0) - 90.0), anisotropy, int(len(moving))
    finally:
        env.close()


#: Measured frames, written by ``python -m tpgpt.grasp.measure_frames``.
FRAMES_PATH = Path(__file__).with_name("gripper_frames.json")


@lru_cache(maxsize=1)
def _frames() -> dict:
    if not FRAMES_PATH.is_file():
        return {}
    return json.loads(FRAMES_PATH.read_text())


def gripper_frame(short: str) -> dict | None:
    """Measured ``grip_site`` frame for a registered gripper, if available.

    Returns a dict with ``alignment`` (3x3, GraspGen-X rotation -> grip_site
    rotation) and ``contact_offset`` (3-vector, where a grasped object sits in
    grip_site coordinates), or ``None`` if the gripper has not been measured.

    Both are needed because robosuite's gripper models place ``grip_site``
    wherever their author chose. Measured across the nine registered hands:
    the UMI's fingers lie along ``-Z``, the Inspire hand's approach is along
    ``-Y`` with its contact 100 mm off the axis, and two hands put the site at
    the gripper base rather than between the fingers.
    """
    return _frames().get(short)


#: Pairs that lift a reference object in physics through the frame contract.
#:
#: Eight of the nine. See :func:`_physics_verified` for the one that does not.
VERIFIED_PAIRS = _physics_verified()


#: Gripper surface points kept for collision checking.
#:
#: The sample spacing has to stay below the collision threshold or scene points
#: slip between them and objects pass through the hand. Measured in the sibling
#: project on a Panda: 384 points give 12.3 mm spacing, 1024 give 7.5 mm, 2048
#: give 5.5 mm. Against a 10 mm threshold, 384 is not enough and 1024 is.
DEFAULT_GRIPPER_POINTS = 1024


@lru_cache(maxsize=32)
def gripper_points(
    graspgen_name: str, n: int = DEFAULT_GRIPPER_POINTS, state: str = "open"
) -> np.ndarray:
    """Surface points of a gripper, in its own base frame.

    Read from each description's ``points.json``, which holds a 10 500-point
    sample in both the open and closed configurations. The **open** hand is
    what matters for collision: it is the widest the gripper gets, and it is the
    shape that has to fit into a shelf slot.

    Args:
        graspgen_name: GraspGen-X gripper name, e.g. ``"franka_panda"``.
        n: Points to keep, subsampled evenly.
        state: ``"open"`` or ``"close"``.

    Raises:
        FileNotFoundError: if the description has no point sample.
    """
    path = gripper_config_path(graspgen_name).with_name("points.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"no points.json for {graspgen_name!r} at {path}; collision "
            "checking needs the gripper's surface sample"
        )
    data = json.loads(path.read_text())
    points = np.asarray(data[state] if isinstance(data, dict) else data, dtype=float)
    if len(points) <= n:
        return points
    step = len(points) / float(n)
    return points[(np.arange(n) * step).astype(int)]
