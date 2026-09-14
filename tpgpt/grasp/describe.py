"""Author a GraspGen-X gripper description from the robosuite model we simulate.

GraspGen-X plans grasps for a hand it has never seen by **conditioning on the
hand's shape** rather than on weights trained for it, so one checkpoint serves
any gripper it can be told about. What it has to be told is a ``config.json``
with nine keys. Its own onboarding route for a new hand is an interactive
browser wizard: a person rotates the hand until its axes agree with the
convention, drives sliders to the open and closed poses, and **drags a box
around the empty space between the fingers** at two closure states.

Every one of those steps is a measurement, and the hand is already in a
simulator. This module makes them instead of asking for them.

**Why this is the direction worth taking.** robosuite can build 26 hands and
GraspGen-X ships descriptions for 26, but they are not the same 26, and the
overlap is what has limited this project to two-finger jaws
(``FINDINGS.md`` 8z, item 1h). Both halves can be brought together from either
side. Importing GraspGen-X's URDFs into MuJoCo is possible -- MuJoCo compiles
them directly -- but it means authoring actuators, a grip site and contact
parameters for a hand nobody has tuned. Going the other way needs only this
file, because the hand robosuite already simulates is the hand we want
described.

**Why no URDF is needed.** The shipped gripper directories also carry a point
cloud, a TSDF grid and a pointnet-VAE vector, and it would be reasonable to
assume the model needs them. It does not: the released checkpoint declares
``gripper_backbone: sweep_volume_v2`` in both ``gen/config.yaml`` and
``dis/config.yaml``, and GraspGen-X's own ``make_sweep_volume_gripper_info``
exists to fill exactly those fields with placeholders. What the conditioning
consumes is the **swept volume**, the fingertip depth and the kinematic family.

What each key means, and how it is obtained here:

``open`` / ``close``
    Joint positions at each extreme, read after driving every actuator to
    ``-1`` and ``+1``. Not consumed at inference; recorded because a
    description that does not say what pose it describes cannot be checked.
``fingertip``
    ``[0, 0, depth]``: how far along the approach axis the grasp point sits
    from the gripper's base. Becomes ``XGripperInfo.depth``.
``sweep_volume``
    **The region the fingers traverse while closing**, as an axis-aligned box,
    for two states: sweeping from fully open (``extents``/``offset``) and from
    half closed (``extents2``/``offset2``). Measured by driving the hand
    through its closing motion and taking the box that contains every position
    its fingers occupied.

    This is the paper's own definition -- *"the region traversed by the robot
    fingers during its grasping motion"* -- and it is **not** the free space
    between the fingers, which is the reading an earlier version of this module
    took. The two nearly coincide for a parallel jaw and are unrelated for a
    hand whose fingers curl inward. The Panda's shipped config is the check:
    its finger joint travels 0.04 m per side and ``extents[0]`` is **0.08**,
    the total travel of the two fingers, with ``extents2[0]`` = 0.04.
``bbox``
    The hand's own extent, from MuJoCo's exact per-geom ``geom_aabb`` rather
    than bounding spheres. Feeds the control points GraspGen-X scores with.
``standoff``
    How far the graspable volume extends beyond the sweep box along the
    approach axis, near end and far end.
``links``
    Body names under the gripper root. Documentation at inference time.
``type``
    ``parallel_2f``, ``revolute_2f`` or ``revolute_3f`` -- the trained
    kinematic family, **not the finger count**. GraspGen-X labels its own
    five-finger ``sharpa_wave`` ``revolute_3f``, so anything with three or more
    fingers takes that family here. Two fingers split on joint type: a sliding
    joint is ``parallel_2f``, a hinge is ``revolute_2f``.
``symmetric``
    Whether a half turn about the approach axis is the same physical grasp.
    True for a two-finger jaw, where the fingers merely swap. False otherwise:
    for three fingers the half turn lands them somewhere else entirely.

**The acceptance test is reproduction, not inspection.** Eleven of the
registered hands have a description written by someone else, so this module can
be asked to describe those eleven and the answers compared. ``--validate`` does
that. A generator that recovers the shipped numbers for hands it did not write
can be believed about hands nobody has written.

**An earlier version of this module measured the wrong quantity** -- the gap
between the fingers rather than the region they sweep -- and reported the
Inspire hand opening by 28 mm against a declared 80. It reproduced the shipped
apertures for the Panda, the Yumi and the Robotiq 2F-140 while doing so, which
is exactly how the error survived: for a parallel jaw the gap and the sweep are
nearly the same box. Any figure quoted from that version is withdrawn.

So: usable for onboarding a two-finger jaw, and an open problem for anything
else. ``docs/gripper_diversity.md``.

Usage::

    python -m tpgpt.grasp.describe --validate
    python -m tpgpt.grasp.describe --gripper JacoThreeFingerGripper --name jaco_3f
    python -m tpgpt.grasp.describe --gripper JacoThreeFingerGripper --name jaco_3f --install

Descriptions are written into this repository, under ``assets/x_grippers/``, so
the repo says what it is using. ``--install`` copies one into the GraspGen-X
asset root where its server resolves gripper names, and **refuses to overwrite a
gripper GraspGen-X ships**: those 26 directories are not ours to edit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tpgpt.grasp.grippers import (
    DEFAULT_DESCRIPTIONS_ROOT,
    GRIPPER_PAIRS,
    gripper_action,
    gripper_config_path,
)
from tpgpt.grasp.measure_frames import DISTAL_FRACTION, SETTLE_STEPS, finger_axes

#: Where authored descriptions live inside this repository.
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "x_grippers"

#: Sub-path under the GraspGen-X descriptions root where it resolves gripper
#: names. Mirrors ``grippers._CONFIG_SUBPATH``.
INSTALL_SUBPATH = "gripper_descriptions/assets/x_grippers"

#: Closure fractions sampled across a sweep. The box is the union over these,
#: so more samples can only grow it; nine is where the Panda's aperture stops
#: moving.
SWEEP_SAMPLES = 9

#: Control steps held at each sampled fraction, to let the jaws arrive.
SWEEP_HOLD = 12

#: Closure the second sweep box starts from. GraspGen-X calls it the
#: "half-open" box.
MID_CLOSURE = 0.5

#: Number of surface points written to ``points.json`` per closure state.
#: ``grippers.gripper_points`` subsamples to 1024 by default.
POINTS_PER_STATE = 10500


def _root_local(sim, geom_ids, root_id):
    """Gripper geom positions **in the root body's own frame**.

    Captured this way rather than in world coordinates because the arm does not
    hold still. Commanding a zero arm action is not the same as commanding the
    arm to stay: measured on the Ability hand, the wrist wandered **800 mm**
    over three forty-step settles, which put the whole of that excursion into
    ``closed - opened`` and flipped the derived approach axis end for end.
    Subtracting the root body's own pose at the moment of capture removes the
    arm exactly, whatever it did.
    """
    root_p = np.array(sim.data.body_xpos[root_id])
    root_R = np.array(sim.data.body_xmat[root_id]).reshape(3, 3)
    points = np.array([sim.data.geom_xpos[i].copy() for i in geom_ids])
    return (points - root_p) @ root_R


def _local_frame(opened_local, closed_local):
    """The rotation taking root-frame points into the grasp convention.

    The convention is GraspGen-X's: ``+X`` closing, ``+Z`` approach, origin at
    the gripper's root body. Returns ``(root_to_grasp, moving)``, applied as
    ``p_local @ root_to_grasp``.
    """
    zero = np.zeros(3)
    _, _, basis, moving = finger_axes(
        opened_local, closed_local, zero, np.eye(3), zero
    )
    return basis, moving


def _swept_volume(env, gripper, geom_ids, root_id, root_to_grasp, moving,
                  start_fraction, freeze):
    """The region the fingers **traverse** while closing, as an axis-aligned box.

    This is GraspGen-X's own definition (arXiv:2606.00998): *"the region
    traversed by the robot fingers during its grasping motion"*, recorded for
    two states -- from fully open, and from halfway closed -- and fed to the
    model as 3 extents plus 3 offsets per state.

    **Not the free space between the fingers**, which is what an earlier
    version of this module measured. The two nearly coincide for a parallel
    jaw, which is why that version reproduced the shipped apertures for the
    Panda, Yumi and Robotiq 2F-140 and why the agreement looked like
    validation. For a hand whose fingers curl inward they are unrelated
    quantities, and the error read as a five-finger hand that could not open
    more than 28 mm.

    The Panda's shipped config is the check: its finger joint travels 0.04 m
    per side, and ``extents[0]`` is **0.08** -- the total travel of the two
    fingers, not the gap between them -- with ``extents2[0]`` = 0.04, the
    travel from half-closed.

    Args:
        start_fraction: Closure to sweep from. 0 gives the open-state box,
            0.5 the half-closed one.
        freeze: Callable holding the arm still between steps.

    Returns:
        ``(extents, offset)`` in the grasp frame.
    """
    from tpgpt.sim.replay import closing_direction, set_closure

    sim = env.sim
    direction = closing_direction(gripper)
    action = gripper_action(env, gripper, -1.0)
    for _ in range(SETTLE_STEPS):
        env.step(action)
        freeze()

    traversed = []
    for fraction in np.linspace(float(start_fraction), 1.0, SWEEP_SAMPLES):
        set_closure(gripper, direction, float(fraction))
        hold = np.zeros(env.action_dim)
        for _ in range(SWEEP_HOLD):
            env.step(hold)
            freeze()
        traversed.append(_root_local(sim, geom_ids, root_id)[moving] @ root_to_grasp)

    points = np.vstack(traversed)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return hi - lo, (hi + lo) / 2.0


def _bbox(sim, geom_ids, root_p, root_R, root_to_grasp):
    """Exact axis-aligned bounds of the hand in the local frame.

    Uses MuJoCo's per-geom ``geom_aabb`` -- centre and half-extents in the
    geom's own frame -- rather than ``geom_rbound``, which is a bounding sphere
    and overestimates a long finger link by its whole length.
    """
    corners = []
    aabb = np.array(sim.model.geom_aabb).reshape(-1, 6)
    for gid in geom_ids:
        centre, half = aabb[gid][:3], aabb[gid][3:]
        geom_p = np.array(sim.data.geom_xpos[gid])
        geom_R = np.array(sim.data.geom_xmat[gid]).reshape(3, 3)
        signs = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)
        world = geom_p + (centre + signs * half) @ geom_R.T
        corners.append(((world - root_p) @ root_R) @ root_to_grasp)
    corners = np.vstack(corners)
    return corners.min(axis=0), corners.max(axis=0)


def _surface_points(sim, geom_ids, root_p, root_R, root_to_grasp, rng, n=POINTS_PER_STATE):
    """Points on the hand's surface, in the local frame.

    Sampled on each geom's exact bounding box faces, in proportion to face area,
    which is coarse for a mesh and exactly right for the use this project puts
    them to: ``grippers.gripper_points`` subsamples them to about a thousand and
    uses them for clearance tests at a 10 mm threshold.
    """
    aabb = np.array(sim.model.geom_aabb).reshape(-1, 6)
    boxes, areas = [], []
    for gid in geom_ids:
        centre, half = aabb[gid][:3], aabb[gid][3:]
        if np.all(half <= 0):
            continue
        boxes.append((gid, centre, half))
        areas.append(8 * (half[0] * half[1] + half[1] * half[2] + half[0] * half[2]))
    if not boxes:
        return np.zeros((0, 3))
    areas = np.array(areas, dtype=float)
    counts = rng.multinomial(n, areas / areas.sum())

    out = []
    for (gid, centre, half), count in zip(boxes, counts):
        if count == 0:
            continue
        pts = rng.uniform(-1.0, 1.0, size=(count, 3))
        # Push each point onto the face it is nearest, so points land on the
        # surface rather than filling the volume.
        axis = rng.integers(0, 3, size=count)
        pts[np.arange(count), axis] = np.sign(pts[np.arange(count), axis])
        local = centre + pts * half
        geom_p = np.array(sim.data.geom_xpos[gid])
        geom_R = np.array(sim.data.geom_xmat[gid]).reshape(3, 3)
        out.append((((geom_p + local @ geom_R.T) - root_p) @ root_R) @ root_to_grasp)
    return np.vstack(out) if out else np.zeros((0, 3))


def _family(sim, gripper, n_fingers):
    """The trained kinematic family this hand belongs to.

    Three or more fingers is ``revolute_3f``: that is the family GraspGen-X
    trained on multi-finger hands and the one it labels its own five-finger
    ``sharpa_wave`` with, so the field names a family and not a count. Two
    fingers split on how they travel -- a sliding joint gives ``parallel_2f``,
    a hinge ``revolute_2f``.
    """
    import mujoco

    if n_fingers >= 3:
        return "revolute_3f"
    types = {
        int(sim.model.jnt_type[sim.model.joint_name2id(j)]) for j in gripper.joints
    }
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    return "parallel_2f" if types == {slide} else "revolute_2f"


def describe_gripper(
    robosuite_name: str,
    *,
    robot: str = "Panda",
    n_fingers: int | None = None,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Measure a robosuite hand and return ``(config, points)``.

    Args:
        robosuite_name: Gripper class name, e.g. ``"SchunkSvhRightHand"``.
        robot: Arm to mount it on. Only the hand's own geometry is measured.
        n_fingers: How many fingers the hand has, which decides ``type`` and
            ``symmetric``. Required, because it cannot be derived reliably --
            ``diagnose.finger_groups`` documents three derivations and each is
            wrong on at least one hand, which is why GraspGen-X's own config is
            normally the arbiter. There is no config to read when authoring one.
        seed: Seeded before the environment is built and before ``reset``.

    Returns:
        ``(config, points)``: the ``config.json`` contents, and
        ``{"open": [...], "close": [...]}`` surface point clouds.
    """
    import robosuite as suite

    from tpgpt.sim.replay import closing_direction, set_closure

    if n_fingers is None:
        raise ValueError(
            f"{robosuite_name}: n_fingers must be given. It decides the "
            "kinematic family and the symmetry flag, and deriving it from the "
            "model is measurably unreliable -- see diagnose.finger_groups, "
            "which tries two derivations and refuses when they disagree."
        )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    env = suite.make(
        "Lift", robots=robot, gripper_types=robosuite_name,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
    )
    try:
        env.reset()
        sim = env.sim
        gripper = env.robots[0].gripper
        gripper = gripper["right"] if isinstance(gripper, dict) else gripper
        geom_ids = [
            i for i, n in enumerate(sim.model.geom_names) if n and "gripper0" in n
        ]
        geom_set = set(geom_ids)
        root_id = sim.model.body_name2id(gripper.root_body)
        joint_ids = [sim.model.joint_name2id(j) for j in gripper.joints]

        # Hold the arm where it started. A zero arm action is not a command to
        # stay put -- with a heavy hand the wrist wandered 800 mm over three
        # settles -- and every quantity measured here is a property of the hand
        # alone, so the arm is simply frozen rather than controlled. Only the
        # arm's own degrees of freedom are touched; the gripper's are left to
        # the physics, which is the distinction section 7.32 is about.
        arm_dofs = np.asarray(env.robots[0]._ref_joint_pos_indexes)
        arm_vels = np.asarray(env.robots[0]._ref_joint_vel_indexes)
        arm_home = np.array(sim.data.qpos[arm_dofs])

        def freeze_arm():
            sim.data.qpos[arm_dofs] = arm_home
            sim.data.qvel[arm_vels] = 0.0
            sim.forward()

        def positions():
            return _root_local(sim, geom_ids, root_id)

        def root_pose():
            return (np.array(sim.data.body_xpos[root_id]),
                    np.array(sim.data.body_xmat[root_id]).reshape(3, 3))

        def qpos():
            return {
                j: float(sim.data.qpos[sim.model.jnt_qposadr[i]])
                for j, i in zip(gripper.joints, joint_ids)
            }

        def settle(command):
            action = gripper_action(env, gripper, command)
            for _ in range(SETTLE_STEPS):
                env.step(action)
                freeze_arm()

        settle(-1.0)
        opened, open_qpos = positions(), qpos()
        settle(1.0)
        closed, close_qpos = positions(), qpos()

        # Re-open before measuring: the pocket has to be the pocket of the
        # *open* hand, and the hand was left shut by the travel measurement
        # above. ``set_closure`` further down writes the gripper's commanded
        # action directly, because robosuite's own action interface takes the
        # *sign* of a command and discards its size.
        settle(-1.0)
        root_to_grasp, moving = _local_frame(opened, closed)
        root_p, root_R = root_pose()
        local_open = positions() @ root_to_grasp
        bbox_lo, bbox_hi = _bbox(sim, geom_ids, root_p, root_R, root_to_grasp)
        open_points = _surface_points(sim, geom_ids, root_p, root_R,
                                      root_to_grasp, rng)

        # The two boxes the model conditions on: the region the fingers cover
        # sweeping from fully open to shut, and from half-closed to shut.
        extents, offset = _swept_volume(
            env, gripper, geom_ids, root_id, root_to_grasp, moving, 0.0, freeze_arm)
        extents2, offset2 = _swept_volume(
            env, gripper, geom_ids, root_id, root_to_grasp, moving,
            MID_CLOSURE, freeze_arm)

        settle(1.0)
        close_points = _surface_points(sim, geom_ids, root_p, root_R,
                                       root_to_grasp, rng)

        # The grasp point: the far face of the swept box along the approach
        # axis, which is where the fingertips are when the hand is open and so
        # where an object first meets them.
        fingertip_z = float(offset[2] + extents[2] / 2.0)

        config = {
            "open": open_qpos,
            "close": close_qpos,
            "fingertip": [0.0, 0.0, fingertip_z],
            "sweep_volume": {
                "extents": [round(float(v), 5) for v in extents],
                "offset": [round(float(v), 5) for v in offset],
                "extents2": [round(float(v), 5) for v in extents2],
                "offset2": [round(float(v), 5) for v in offset2],
            },
            "links": [
                sim.model.body_id2name(i)
                for i in range(sim.model.nbody)
                if (sim.model.body_id2name(i) or "").startswith("gripper0")
            ],
            "standoff": [0.0, 0.0],
            "bbox": [bbox_lo.tolist(), bbox_hi.tolist()],
            "symmetric": n_fingers == 2,
            "type": _family(sim, gripper, n_fingers),
            # Not part of GraspGen-X's schema. Recorded so a description can be
            # told from one written by hand, and so the measurement that
            # produced it can be judged.
            "tpgpt_measured": {
                "robosuite": robosuite_name,
                "robot": robot,
                "n_fingers": int(n_fingers),
                "seed": int(seed),
                "sweep_samples": SWEEP_SAMPLES,
                "moving_geoms": int(moving.sum()),
            },
        }
        points = {
            "open": open_points.round(5).tolist(),
            "close": close_points.round(5).tolist(),
        }
        return config, points
    finally:
        env.close()


def write_description(config: dict, points: dict, name: str,
                      root: Path = ASSETS_ROOT) -> Path:
    """Write ``config.json`` and ``points.json`` for one hand."""
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=4))
    (out / "points.json").write_text(json.dumps(points))
    return out


def install(name: str, root: Path = ASSETS_ROOT) -> Path:
    """Copy an authored description where GraspGen-X's server resolves it.

    Raises:
        FileExistsError: if GraspGen-X already ships a gripper by this name.
            Its own descriptions are not ours to overwrite, and a silent
            replacement would change what every existing result meant.
    """
    source = Path(root) / name
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"no authored description at {source}")
    target = Path(DEFAULT_DESCRIPTIONS_ROOT) / INSTALL_SUBPATH / name
    if target.exists() and not (target / ".tpgpt_authored").exists():
        raise FileExistsError(
            f"{target} already exists and was not authored here. GraspGen-X "
            "ships 26 gripper descriptions and they are not ours to overwrite; "
            "choose a different --name."
        )
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        shutil.copy2(item, target / item.name)
    (target / ".tpgpt_authored").write_text(
        "Authored by tpgpt.grasp.describe from the robosuite model.\n"
    )
    return target


def validate(shorts=None, robot: str = "Panda") -> list[dict]:
    """Describe hands GraspGen-X has already described, and compare.

    The acceptance test for this module. Reports, per hand, the shipped and the
    measured aperture, pocket depth and fingertip depth, so a reader can judge
    whether a measured description is the same kind of object as a written one.
    """
    shorts = shorts or [
        s for s, p in GRIPPER_PAIRS.items() if gripper_config_path(p.graspgen).is_file()
    ]
    rows = []
    for short in shorts:
        pair = GRIPPER_PAIRS[short]
        shipped = json.loads(gripper_config_path(pair.graspgen).read_text())
        fingers = int(shipped["type"][-2]) if shipped["type"][-2].isdigit() else 2
        try:
            config, _ = describe_gripper(
                pair.robosuite, robot=robot, n_fingers=fingers
            )
        except Exception as exc:
            rows.append({"short": short, "error": f"{type(exc).__name__}: {exc}"})
            continue
        sv_s, sv_m = shipped["sweep_volume"], config["sweep_volume"]
        rows.append({
            "short": short,
            "graspgen": pair.graspgen,
            "type_shipped": shipped["type"],
            "type_measured": config["type"],
            "aperture_shipped_mm": sv_s["extents"][0] * 1000,
            "aperture_measured_mm": sv_m["extents"][0] * 1000,
            "depth_shipped_mm": sv_s["extents"][2] * 1000,
            "depth_measured_mm": sv_m["extents"][2] * 1000,
            "offset_z_shipped_mm": sv_s["offset"][2] * 1000,
            "offset_z_measured_mm": sv_m["offset"][2] * 1000,
            "fingertip_shipped_mm": shipped["fingertip"][2] * 1000,
            "fingertip_measured_mm": config["fingertip"][2] * 1000,
            "mid_aperture_shipped_mm": sv_s["extents2"][0] * 1000,
            "mid_aperture_measured_mm": sv_m["extents2"][0] * 1000,
            "sweep_samples": config["tpgpt_measured"]["sweep_samples"],
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gripper", help="robosuite gripper class name")
    parser.add_argument("--name", help="GraspGen-X gripper name to write")
    parser.add_argument("--fingers", type=int, help="how many fingers it has")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--out", default=str(ASSETS_ROOT))
    parser.add_argument("--install", action="store_true",
                        help="also copy into the GraspGen-X asset root")
    parser.add_argument("--validate", action="store_true",
                        help="describe the hands GraspGen-X already describes "
                             "and compare, instead of writing anything")
    args = parser.parse_args(argv)

    if args.validate:
        rows = validate()
        head = (f"{'hand':<11}{'family':>26}{'aperture mm':>22}"
                f"{'pocket depth mm':>22}{'fingertip mm':>20}")
        print(head)
        print(f"{'':<11}{'shipped / measured':>26}{'shipped / measured':>22}"
              f"{'shipped / measured':>22}{'shipped / measured':>20}")
        for r in rows:
            if "error" in r:
                print(f"{r['short']:<11}  {r['error']}")
                continue
            print(
                f"{r['short']:<11}"
                f"{r['type_shipped']:>13} {r['type_measured']:>12}"
                f"{r['aperture_shipped_mm']:>11.1f} {r['aperture_measured_mm']:>10.1f}"
                f"{r['depth_shipped_mm']:>11.1f} {r['depth_measured_mm']:>10.1f}"
                f"{r['fingertip_shipped_mm']:>10.1f} {r['fingertip_measured_mm']:>9.1f}"
            )
        return rows

    if not args.gripper or not args.name or not args.fingers:
        parser.error("--gripper, --name and --fingers are required")

    config, points = describe_gripper(
        args.gripper, robot=args.robot, n_fingers=args.fingers
    )
    out = write_description(config, points, args.name, Path(args.out))
    sv = config["sweep_volume"]
    print(f"{args.name}: type={config['type']} symmetric={config['symmetric']}")
    print(f"  aperture      {sv['extents'][0] * 1000:.1f} mm open, "
          f"{sv['extents2'][0] * 1000:.1f} mm at half closure")
    print(f"  pocket        {np.round(np.array(sv['extents']) * 1000, 1)} mm "
          f"centred {np.round(np.array(sv['offset']) * 1000, 1)} mm")
    print(f"  fingertip     {config['fingertip'][2] * 1000:.1f} mm")
    print(f"  wrote         {out}")
    if args.install:
        print(f"  installed     {install(args.name, Path(args.out))}")
    return out


if __name__ == "__main__":
    main()
