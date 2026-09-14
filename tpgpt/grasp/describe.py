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
    The box enclosing the free space **between** the fingers -- the pocket an
    object has to fit into -- at the open state (``extents``/``offset``) and at
    half closure (``extents2``/``offset2``). This is the box the wizard asks a
    human to drag, and it is measured here by casting rays outward from the
    hand's mid-plane along the closing axis: where a ray going one way and a ray
    going the other both strike a finger, that point is inside the pocket, and
    the two hit distances add up to the local aperture.

    Ray casting rather than arithmetic on geom positions, because a finger is a
    mesh with no analytic half-width, and ``geom_rbound`` -- a bounding
    *sphere* -- overestimates a long link badly.
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

**Measured, it passes for two-finger jaws and fails for anthropomorphic hands.**
Against the shipped apertures: panda **78.2 mm** against 80.0, yumi **50.0**
against 50.0, robotiq140 126.3 against 125.0, xarm 86.0 against 85.0 -- within
3%. Less good but the same order: rethink 58.7 against 66.0, robotiq85 104.0
against 85.0. The first three of those were each confirmed by a second,
independent run with the arm frozen, agreeing to 0.3 mm.

For the multi-finger hands it does not work, and the reason is structural rather
than a tuning problem. Repeating the measurement along **twelve axes**
perpendicular to the approach finds, for a Panda, **no pocket on any axis but
the closing one** -- which is what a parallel jaw should look like. For the
Inspire hand it finds 10.4 mm along the axis its fingers travel and a widest gap
of **28.0 mm at 75 degrees away from it**, against a declared 80. For
``g1three`` it finds **no pocket at all** on the travel axis and 26.1 mm at 90
degrees, against a declared 100.

A hand whose five fingers curl into a palm does not hold things between two
opposed fingertips, so a box fitted between two opposed fingertips is not its
graspable volume, and no choice of axis rescues that. Do not use this on a hand
with more than two fingers without checking the result against something.

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

#: Rays longer than this are treated as missing the hand entirely. No registered
#: hand is wider than 200 mm across its jaw axis.
MAX_REACH = 0.20

#: Samples per axis across the search window when mapping the pocket. 41 gives
#: 4 mm resolution over the window, which resolves a 18 mm finger pad.
GRID = 41

#: Half-width, in metres, of the window searched for the pocket in the two axes
#: perpendicular to the closing direction.
WINDOW = 0.16

#: Closure fraction the second sweep box is measured at. GraspGen-X calls it the
#: "half-open" box and its own wizard sets the closed pose for it; half of the
#: commanded travel is the reading of that this module uses.
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


def _pocket(sim, gripper_geoms, root_p, root_R, root_to_grasp, z_lo, z_hi):
    """Map the free space between the fingers, by casting rays across it.

    The window is bounded to the **distal band of the fingers** rather than the
    whole hand. Searching from the base upward measures the wrong thing: on a
    Panda the rays pass either side of the palm and report a 168 mm "aperture"
    for a hand whose jaws open to 80, because the widest gap between two
    gripper surfaces is not the gap an object goes into. The band is the same
    one :func:`measure_frames.measure_frame` uses for the contact point --
    ``DISTAL_FRACTION`` of the fingers, measured back from their tips.

    Returns ``(extents, offset, cells)`` in the local frame, or ``None`` if no
    point in the window has a finger on both sides of it.
    """
    import mujoco

    ys = np.linspace(-WINDOW / 2, WINDOW / 2, GRID)
    zs = np.linspace(z_lo, z_hi, GRID)
    hit_geom = np.zeros(1, dtype=np.int32)
    # Local (grasp convention) -> world, via the root body's current pose.
    local_to_world = root_R @ root_to_grasp

    aperture = np.full((GRID, GRID), np.nan)
    centre_x = np.full((GRID, GRID), np.nan)
    for i, y in enumerate(ys):
        for j, z in enumerate(zs):
            start = root_p + local_to_world @ np.array([0.0, y, z])
            hits = {}
            for sign in (+1.0, -1.0):
                direction = local_to_world @ np.array([sign, 0.0, 0.0])
                distance = mujoco.mj_ray(
                    sim.model._model, sim.data._data, start, direction,
                    None, 1, -1, hit_geom,
                )
                if 0.0 <= distance <= MAX_REACH and int(hit_geom[0]) in gripper_geoms:
                    hits[sign] = float(distance)
            if len(hits) == 2:
                aperture[i, j] = hits[+1.0] + hits[-1.0]
                # Midpoint of the gap, as an x offset from the mid-plane.
                centre_x[i, j] = (hits[+1.0] - hits[-1.0]) / 2.0

    inside = ~np.isnan(aperture)
    if not inside.any():
        return None

    yi, zi = np.where(inside)
    y_low, y_high = ys[yi.min()], ys[yi.max()]
    z_low, z_high = zs[zi.min()], zs[zi.max()]
    # The aperture varies across the pocket -- a curved finger is closer at its
    # tip than at its knuckle -- so one number has to be chosen. The median is
    # used rather than the maximum, because the box is meant to be the space an
    # object fits into and the widest single ray is usually one that has found
    # a way past the fingers rather than between them.
    extents = np.array([
        float(np.nanmedian(aperture)),
        float(y_high - y_low),
        float(z_high - z_low),
    ])
    offset = np.array([
        float(np.nanmedian(centre_x)),
        float((y_high + y_low) / 2.0),
        float((z_high + z_low) / 2.0),
    ])
    return extents, offset, int(inside.sum())


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
        # The distal band of the fingers: where the pads are, and where an
        # object is actually held. Measured on the *open* hand, and reused for
        # the half-closed measurement so both boxes describe the same part of
        # the hand rather than two different parts of it.
        along = local_open[moving][:, 2]
        z_hi = float(along.max()) + 0.01
        z_lo = float(along.max() - DISTAL_FRACTION * (np.ptp(along) or 0.01))
        open_sweep = _pocket(sim, geom_set, root_p, root_R, root_to_grasp, z_lo, z_hi)
        bbox_lo, bbox_hi = _bbox(sim, geom_ids, root_p, root_R, root_to_grasp)
        open_points = _surface_points(sim, geom_ids, root_p, root_R, root_to_grasp, rng)

        direction = closing_direction(gripper)
        set_closure(gripper, direction, MID_CLOSURE)
        hold = np.zeros(env.action_dim)
        for _ in range(SETTLE_STEPS):
            env.step(hold)
            freeze_arm()
        root_p, root_R = root_pose()
        mid_sweep = _pocket(sim, geom_set, root_p, root_R, root_to_grasp, z_lo, z_hi)

        settle(1.0)
        close_points = _surface_points(sim, geom_ids, root_p, root_R, root_to_grasp, rng)

        if open_sweep is None:
            raise RuntimeError(
                f"{robosuite_name}: no point in the search window has a finger "
                "on both sides of it, so this hand has no measurable pocket "
                "between its fingers and cannot be described this way."
            )
        extents, offset, cells = open_sweep
        if mid_sweep is None:
            # A hand whose pocket closes completely at half travel. Keep the
            # open box's shape and collapse its aperture, rather than emitting
            # a box measured at a different state than it claims.
            extents2, offset2 = extents * np.array([0.5, 1.0, 1.0]), offset
            mid_cells = 0
        else:
            extents2, offset2, mid_cells = mid_sweep

        config = {
            "open": open_qpos,
            "close": close_qpos,
            "fingertip": [0.0, 0.0, float(offset[2] + extents[2] / 2.0)],
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
                "pocket_cells_open": cells,
                "pocket_cells_mid": mid_cells,
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
            "pocket_cells": config["tpgpt_measured"]["pocket_cells_open"],
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
