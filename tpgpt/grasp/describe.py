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
    commands_position,
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

#: MuJoCo's geom type code for a mesh.
MESH_GEOM = 7


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


def _pad_geoms(geom_ids, local_open, moving, fraction=None):
    """Which moving geoms are the contact pads.

    The distal band of the fingers, measured back from their tips along the
    approach axis. Everything proximal to that is knuckle and linkage: it moves
    when the hand closes, so it is "moving", and it sweeps a far wider arc than
    the pads do. Including it put the Panda's aperture at **104.6 mm** against a
    declared 80.
    """
    from tpgpt.grasp.measure_frames import DISTAL_FRACTION

    band = DISTAL_FRACTION if fraction is None else fraction
    along = local_open[moving][:, 2]
    cut = along.max() - band * (float(np.ptp(along)) or 0.01)
    # ``moving`` masks the *rows* of the geom array, so its indices address
    # ``geom_ids`` and not MuJoCo's global geom table. Using them directly as
    # geom ids reads the AABBs of whatever else is in the scene, and the Panda's
    # aperture came out as 3472 mm -- the width of the room.
    rows = np.flatnonzero(moving)[along >= cut]
    return [int(geom_ids[r]) for r in rows]


def _solid_corners(sim, geom_ids, root_p, root_R, root_to_grasp):
    """Corners of each geom's exact box, in the grasp frame.

    Solid rather than centres. The swept volume's extent **across** the closing
    axis is the pads' own cross-section -- the Panda's 18 x 18 mm -- and its two
    fingers sit at the same height and the same depth, so a box fitted to their
    *centres* is 0 mm wide in that direction. A degenerate box is not a smaller
    error than a wrong one: it is what the model conditions on.
    """
    aabb = np.array(sim.model.geom_aabb).reshape(-1, 6)
    signs = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)
    out = []
    for gid in geom_ids:
        centre, half = aabb[gid][:3], aabb[gid][3:]
        pos = np.array(sim.data.geom_xpos[gid])
        rot = np.array(sim.data.geom_xmat[gid]).reshape(3, 3)
        world = pos + (centre + signs * half) @ rot.T
        out.append(((world - root_p) @ root_R) @ root_to_grasp)
    return np.vstack(out)


def _largest_gap(coordinate):
    """The widest empty interval along one axis of a set of points.

    This is the aperture: the space between the two groups of fingers, which is
    what an object has to fit into.

    Splitting the pads by the *sign* of their coordinate is the obvious way and
    it is wrong for anything but a symmetric jaw. It assumes the two groups
    straddle the origin, and a hand whose fingers curl does not oblige -- at the
    distal band its pads can sit wholly on one side, so the split puts every pad
    in one group, leaves the other empty, and reports an aperture of nothing.
    Measured that way, the Inspire hand read **6.8 mm** against a declared 80,
    the SchunkSvh **0.8** and the Ability hand **0.1**.

    The largest gap needs no such assumption: sort the coordinates and take the
    biggest step between neighbours. For a parallel jaw that is exactly the
    space between the two fingers; for a thumb opposing four fingers it is the
    space between the thumb and the rest, wherever they sit.
    """
    values = np.sort(np.asarray(coordinate, dtype=float))
    if len(values) < 2:
        return 0.0
    return float(np.max(np.diff(values)))


def _swept_volume(env, gripper, geom_ids, pad_ids, root_id, root_to_grasp,
                  start_fraction, freeze):
    """The region the finger **pads** traverse while closing, as a box.

    GraspGen-X's own definition (arXiv:2606.00998): *"the region traversed by
    the robot fingers during its grasping motion"*, recorded from fully open
    and from half closed, and fed to the model as three extents plus three
    offsets per state.

    Three things make this match what the shipped descriptions contain, and
    each was got wrong first:

    * the **pads**, not every moving geom -- see :func:`_pad_geoms`;
    * the pads' **solid** volume, not their centres -- see
      :func:`_solid_corners`;
    * the offset's x and y forced to **zero**. Every one of GraspGen-X's own
      26 descriptions has ``offset = [0, 0, z]``: the box is centred on the
      approach axis by convention, and a hand whose pads are a millimetre
      off-centre in the model should not be described as gripping off-axis.

    The Panda is the check, because its declared numbers are unambiguous: its
    finger joint travels 0.04 m per side and ``extents[0]`` is 0.08, the two
    fingers' total travel, with ``extents[1]`` and ``extents[2]`` at 0.018, the
    pad's own cross-section.
    """
    from tpgpt.sim.replay import closing_direction, set_closure

    sim = env.sim
    passthrough = commands_position(gripper)
    direction = None if passthrough else closing_direction(gripper)

    action = gripper_action(env, gripper, -1.0)
    for _ in range(SETTLE_STEPS):
        env.step(action)
        freeze()

    root_p = np.array(sim.data.body_xpos[root_id])
    root_R = np.array(sim.data.body_xmat[root_id]).reshape(3, 3)

    traversed = []
    for fraction in np.linspace(float(start_fraction), 1.0, SWEEP_SAMPLES):
        if passthrough:
            command = gripper_action(env, gripper, 2.0 * float(fraction) - 1.0)
        else:
            set_closure(gripper, direction, float(fraction))
            command = np.zeros(env.action_dim)
        for _ in range(SWEEP_HOLD):
            env.step(command)
            freeze()
        traversed.append(
            _solid_corners(sim, pad_ids, root_p, root_R, root_to_grasp)
        )

    # Across the closing axis and along the approach, the box is the region the
    # pads **sweep**: a revolute finger swings through a much taller volume than
    # it occupies at any one moment, which is why the Robotiq 2F-140 declares
    # 62 mm of depth for a pad a fraction of that.
    points = np.vstack(traversed)
    lo, hi = points.min(axis=0), points.max(axis=0)

    # Along the closing axis it is the **gap between the jaws**, not the solid
    # union. A swept union there would span the fingers *and* the space between
    # them plus their own thickness; the Panda came out at 94.9 mm against a
    # declared 80, which is exactly its 2 x 0.04 m of travel. Closing only
    # narrows the gap, so the union of the gap over the sweep is the gap at the
    # state the sweep starts from.
    first = traversed[0]
    aperture = _largest_gap(first[:, 0])

    extents = np.array([aperture, float(hi[1] - lo[1]), float(hi[2] - lo[2])])
    offset = np.array([0.0, 0.0, float((hi[2] + lo[2]) / 2.0)])
    return extents, offset


def _finger_boxes(sim, geom_ids, root_p, root_R, root_to_grasp):
    """One axis-aligned box per finger geom, in the grasp frame.

    GraspGen-X's estimator works **per geom** -- it needs to know which finger
    each box belongs to so it can pick the outermost two. Pooling every corner
    into one cloud, which is what :func:`_largest_gap` did, throws that away.
    """
    corners = _solid_corners(sim, geom_ids, root_p, root_R, root_to_grasp)
    boxes = corners.reshape(len(geom_ids), 8, 3)
    return boxes.min(axis=1), boxes.max(axis=1)


def _inner_sweep_volume(lo, hi):
    """The pocket between the fingers, by GraspGen-X's own rule.

    A port of ``estimate_inner_sweep_volume`` in GraspGenX's
    ``scripts/gripper_config_wizard.py`` -- the function that seeds the box the
    wizard then asks a person to drag, and therefore the closest thing to an
    authoritative definition that exists. Its docstring: *"closing extent = gap
    between innermost finger surfaces"*.

    The rule, in three steps:

    1. the **closing axis** is whichever axis the finger centroids are most
       spread along (here it should come out 0, because the frame is already
       the grasp frame, and if it does not the frame measurement is wrong);
    2. sort the fingers by centroid along that axis and take only the **two
       extreme** ones -- the gap is from the far face of the most-negative
       finger to the near face of the most-positive one. If they overlap, so
       that there is no gap to enclose, fall back to the centroid separation;
    3. across the other two axes the box is the **union** of every finger box.

    Taking the two extremes is what my ``_largest_gap`` got wrong, and it is
    the whole of the five-finger failure. Sorting *all* the pad coordinates and
    taking the biggest step finds the space between two **adjacent** fingers on
    an anthropomorphic hand -- about 8 mm -- where the pocket the object goes
    into is thumb-to-little-finger, about 110 mm. Measured that way the Ability
    hand declared 8.8 mm against a real 112, the SchunkSvh 7.1 against 190 and
    the Fourier hand 10.5 against 30, and none of the three held an object at
    any depth. The two hands where the rules agree are parallel jaws, where
    "the widest gap" and "between the outermost two fingers" are the same
    sentence -- which is why the Panda and the Yumi validated a broken rule.
    """
    centroids = (lo + hi) / 2.0
    spread = centroids.max(axis=0) - centroids.min(axis=0)
    axis = int(np.argmax(spread))

    order = np.argsort(centroids[:, axis])
    inner_lo = float(hi[order[0], axis])
    inner_hi = float(lo[order[-1], axis])
    if inner_hi <= inner_lo:
        inner_lo = float(centroids[order[0], axis])
        inner_hi = float(centroids[order[-1], axis])

    sv_lo, sv_hi = lo.min(axis=0), hi.max(axis=0)
    sv_lo[axis], sv_hi[axis] = inner_lo, inner_hi
    return (sv_hi - sv_lo), (sv_lo + sv_hi) / 2.0, axis


def _pocket(env, gripper, moving_ids, root_id, root_to_grasp, fraction, freeze,
            rng):
    """The inner box at one closure fraction.

    Drives the hand to ``fraction`` (0 = fully open, 1 = shut), lets it settle,
    and fits :func:`tpgpt.grasp.authoring.estimate_sweep_box` to the finger
    links. Two calls -- at 0 and at 0.5 -- give the ``extents``/``offset`` and
    ``extents2``/``offset2`` the model conditions on.

    The estimator is scored against GraspGen-X's own 26 curated descriptions by
    :mod:`tpgpt.grasp.bench_authoring`; see that module for what the numbers
    mean. The closing axis is **forced** to 0 rather than derived, because the
    frame here is already the grasp frame that ``measure_frames`` produced --
    deriving it again would let a noisy hand overrule a measurement. The
    derived value is returned anyway, as a free check that the two agree.
    """
    from tpgpt.grasp.authoring import estimate_sweep_box
    from tpgpt.sim.replay import closing_direction, set_closure

    sim = env.sim
    if commands_position(gripper):
        command = gripper_action(env, gripper, 2.0 * float(fraction) - 1.0)
    else:
        set_closure(gripper, closing_direction(gripper), float(fraction))
        command = np.zeros(env.action_dim)
    for _ in range(SETTLE_STEPS):
        env.step(command)
        freeze()

    root_p = np.array(sim.data.body_xpos[root_id])
    root_R = np.array(sim.data.body_xmat[root_id]).reshape(3, 3)
    fingers = _finger_points(sim, moving_ids, root_p, root_R, root_to_grasp, rng)
    box = estimate_sweep_box(fingers, closing_axis=0)
    return box.extents, box.offset, box.diagnostics


def _geom_surface(sim, gid, rng, n):
    """Points on one geom's actual surface, in the world frame.

    A mesh geom gives up its **vertices**; everything else is sampled on its
    bounding box. The distinction is not cosmetic. The pocket's width is a gap
    between two surfaces, and a bounding box bulges inward wherever the shape
    inside it tapers: the Panda's finger pads face each other at +-39.4 mm,
    close to the 80 mm aperture its curated description declares, while the
    AABBs of the finger hulls behind them register material at +-30 mm -- a
    solid 10 mm inside the pads, through which no object could pass. Sampling
    boxes put the Panda's aperture at 70.8 mm; its own mesh gives 79.8 mm from
    the same rule.
    """
    gtype = int(sim.model.geom_type[gid])
    mesh_id = int(sim.model.geom_dataid[gid])
    geom_p = np.array(sim.data.geom_xpos[gid])
    geom_R = np.array(sim.data.geom_xmat[gid]).reshape(3, 3)

    if gtype == MESH_GEOM and mesh_id >= 0:
        start = int(sim.model.mesh_vertadr[mesh_id])
        count = int(sim.model.mesh_vertnum[mesh_id])
        verts = np.array(sim.model.mesh_vert[start:start + count]).reshape(-1, 3)
        if len(verts) > n:
            verts = verts[rng.choice(len(verts), n, replace=False)]
        return geom_p + verts @ geom_R.T

    aabb = np.array(sim.model.geom_aabb).reshape(-1, 6)
    centre, half = aabb[gid][:3], aabb[gid][3:]
    if np.all(half <= 0):
        return np.zeros((0, 3))
    local = rng.uniform(-1.0, 1.0, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    local[np.arange(n), axis] = np.sign(local[np.arange(n), axis])
    return geom_p + (centre + local * half) @ geom_R.T


def _finger_points(sim, moving_ids, root_p, root_R, root_to_grasp, rng,
                   per_geom=800):
    """Surface points of each finger **link**, in the grasp frame.

    Grouped by MuJoCo body, not by geom. GraspGen-X's estimator works on URDF
    link meshes -- one geometry per finger segment -- and robosuite's models
    split a single finger across several geoms (a Panda finger carries both a
    collision hull and a separate pad). Feeding those in as if they were
    separate fingers makes a two-finger hand look like a four-finger one and
    changes which pair counts as the outermost.
    """
    by_body = {}
    for gid in moving_ids:
        by_body.setdefault(int(sim.model.geom_bodyid[gid]), []).append(int(gid))

    fingers = []
    for _, gids in sorted(by_body.items()):
        pts = [_geom_surface(sim, gid, rng, per_geom) for gid in gids]
        pts = [q for q in pts if len(q)]
        if pts:
            world = np.vstack(pts)
            fingers.append(((world - root_p) @ root_R) @ root_to_grasp)
    return fingers


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

        # The two boxes the model conditions on: the pocket between the fingers
        # with the hand open, and the same pocket half closed. Not a swept
        # union over the closing motion -- GraspGen-X's wizard annotates "the
        # inner volume between the fingertips" at two static poses, and its own
        # estimator calls the closing extent "the gap between innermost finger
        # surfaces". See :func:`_inner_sweep_volume`.
        moving_ids = [int(geom_ids[r]) for r in np.flatnonzero(moving)]
        extents, offset, pocket_info = _pocket(
            env, gripper, moving_ids, root_id, root_to_grasp, 0.0, freeze_arm, rng)
        extents2, offset2, _ = _pocket(
            env, gripper, moving_ids, root_id, root_to_grasp,
            MID_CLOSURE, freeze_arm, rng)

        settle(1.0)
        close_points = _surface_points(sim, geom_ids, root_p, root_R,
                                       root_to_grasp, rng)

        # The grasp point. GraspGen-X's wizard derives it, and this is the
        # derivation, verbatim from ``gripper_config_wizard.py``:
        #
        #     # Derive fingertip from sweep volume offset
        #     self.fingertip = list(self.sv_offset)
        #
        # So it is the centre of the open pocket and nothing else has to be
        # measured -- which matters for a hand that exists only as CAD, where
        # there is no physics sweep to calibrate against.
        #
        # It is not the whole story, and the curated descriptions say so: five
        # of the 26 use exactly the box centre and the rest push the point
        # **forward** along the approach axis by a round 5, 10, 15, 20, 25 or
        # 30 mm, median 10, maximum 37 (the UMI). Those are a person's
        # judgement about where along the fingers the object should sit, not a
        # derivation. GraspGen-X's own fallback for a config-less gripper,
        # ``make_sweep_volume_gripper_info``, uses a *third* rule -- the top
        # plane of the box, ``offset[2] + extents[2] / 2`` -- which for the
        # Panda gives 112.4 mm against the 113.4 mm this module's physics
        # calibration found independently.
        #
        # The box centre is what the authoring tool writes, so it is the
        # default; ``--calibrate`` refines it against physics.
        fingertip_z = float(offset[2])

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
            # The rotation that takes the gripper's **native** frame -- whatever
            # frame its URDF or CAD was exported in -- to the canonical one,
            # +Z along the approach and +X along the closing direction. It is
            # step 1 of GraspGen-X's wizard and "applied to every downstream
            # computation", and omitting it silently asserts the model is
            # already aligned.
            #
            # Identity here, and that is a statement rather than a default:
            # every number above is measured in the grasp frame that
            # ``measure_frames`` derives, so the description is *already*
            # canonical and no further rotation is needed. A hand arriving as
            # CAD has no such measurement, and for it this field is the
            # difference between a working description and one rotated 90 or
            # 180 degrees -- the same spread ``alignment_rotation`` measures
            # across this registry's own hands.
            #
            # None of GraspGen-X's own 26 curated descriptions carries the key,
            # because their URDFs were exported aligned; the loader therefore
            # treats a missing one as identity.
            "base_rotation": np.eye(4).tolist(),
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
                # 0 means the pocket's own widest spread is along the frame's
                # closing axis, which is what the frame measurement claims.
                # Anything else means the two disagree about the hand.
                "closing_axis": int(pocket_info["derived_closing_axis"]),
                "pocket_slices": int(pocket_info["pocket_slices"]),
                "n_finger_links": int(pocket_info["n_fingers"]),
            },
        }
        points = {
            "open": open_points.round(5).tolist(),
            "close": close_points.round(5).tolist(),
        }
        return config, points
    finally:
        env.close()


def calibrate_fingertip(name, gripper_short, candidates=None, n_grasps=20,
                        cone_deg=75.0, seed=0):
    """Choose ``fingertip`` by grasping with it, rather than by deriving it.

    Every other field in a description is geometry and can be measured off the
    model. ``fingertip`` cannot, and the reason is worth stating because it
    looks like it should be derivable.

    It is the tool-centre depth: how far along the approach axis the grasp point
    sits from the gripper's base. The hand's own geometry gives a candidate --
    the centre of the region its pads sweep -- and for the Panda that is
    **93.4 mm** while the description GraspGen-X ships says **103.4**. Neither
    is wrong about the hand. They disagree because "the grasp point" is a
    convention: the model is trained to place the base so that
    ``base + depth * approach`` lands on the object, and which point of the
    object that is -- its surface, its centre, the middle of the pads -- is not
    fixed by the hand.

    What physics settles is the **total** depth. Executed on a 40 mm cube, the
    same grasps held 3 of 12 at 93.4 mm and **8 of 12** at 103.4, and a sweep
    peaks around 110 -- which is the shipped 103.4 plus the Panda's own stored
    ``calibrated_depth`` of 7.5. So the two numbers trade off exactly, and only
    their sum is observable.

    Inside this project that means a wrong ``fingertip`` is absorbed by
    ``verify.calibrate_depth`` and costs nothing once that has been run -- which
    is the onboarding step that was skipped for the four hands that then held
    **0 of 41** grasps. On hardware there is no such second chance: the
    description has to be right on its own. So it is calibrated here.

    Args:
        name: GraspGen-X description to calibrate, already installed.
        gripper_short: The registry hand it describes.
        candidates: Depths in metres to try. Defaults to a spread around the
            description's current value.
        n_grasps: Grasps executed per candidate. They are the **same** grasps at
            every depth -- the planner never sees ``fingertip``, so this is an
            exact one-variable sweep rather than a comparison of draws.

    Returns:
        ``{"fingertip", "tried", "held", "rate"}``, or ``None`` if no candidate
        held anything.
    """
    import json as _json

    from tpgpt.grasp import validate_config as V
    from tpgpt.grasp.client import GraspGenClient

    current = _json.loads(gripper_config_path(name).read_text())["fingertip"][2]
    if candidates is None:
        # Spread either side of the geometric value, not just beyond it. The
        # Yumi's optimum came out at the **lowest** depth a one-sided range
        # offered, which is a sweep reporting its own edge rather than a
        # maximum; a peak at an endpoint means the range was wrong.
        candidates = [current + d / 1000.0
                      for d in (-30, -20, -10, 0, 10, 20, 30)]

    env = V._lift_scene(gripper_short, seed=seed)
    try:
        centre, half = V._cube_state(env)
        cloud = V.surface_cloud(half, centre, seed=seed)
        client = GraspGenClient()
        try:
            poses, scores = V.propose(cloud, name, client=client,
                                      num_grasps=400, topk=100)
        finally:
            client.close()
        down = np.array([0.0, 0.0, -1.0])
        above = [i for i in range(len(poses))
                 if float(poses[i][:3, 2] @ down) >= np.cos(np.radians(cone_deg))]
        order = sorted(above, key=lambda i: -scores[i])[:n_grasps]
        if not order:
            return None

        best, tried = None, []
        for depth in candidates:
            rows = [V.teleport_grasp(env, poses[i], gripper_short, depth, seed=seed)
                    for i in order]
            got = [r for r in rows if r["reachable"]]
            held = sum(1 for r in got if r["held"])
            rate = held / len(got) if got else 0.0
            tried.append({"fingertip": float(depth), "reachable": len(got),
                          "held": held, "rate": rate})
            if best is None or rate > best["rate"]:
                best = tried[-1]
        # A maximum at either end is the range's edge, not the hand's optimum.
        if best is not None and tried and best["rate"] > 0 and \
                best["fingertip"] in (tried[0]["fingertip"], tried[-1]["fingertip"]):
            best = dict(best, at_edge=True)
        if best is None or best["held"] == 0:
            return {"fingertip": None, "tried": tried, "held": 0, "rate": 0.0}
        return {"fingertip": best["fingertip"], "tried": tried,
                "held": best["held"], "rate": best["rate"]}
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
    parser.add_argument(
        "--calibrate", action="store_true",
        help="after writing and installing, choose `fingertip` by grasping "
             "with it -- see calibrate_fingertip() for why it cannot be "
             "derived. Requires --install.",
    )
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
    if args.calibrate:
        if not args.install:
            parser.error("--calibrate needs --install: it grasps with the "
                         "installed description")
        short = next((k for k, v in GRIPPER_PAIRS.items()
                      if v.robosuite == args.gripper), None)
        if short is None:
            parser.error(f"{args.gripper} is not in GRIPPER_PAIRS, so there is "
                         "no registry hand to grasp with")
        result = calibrate_fingertip(args.name, short)
        if result is None:
            print("  calibration   no top-down grasps to try")
        elif result["fingertip"] is None:
            print("  calibration   nothing held at any depth; fingertip left "
                  f"at {config['fingertip'][2] * 1000:.1f} mm")
            for row in result["tried"]:
                print(f"      {row['fingertip'] * 1000:7.1f} mm  "
                      f"{row['held']}/{row['reachable']}")
        else:
            for row in result["tried"]:
                mark = "  <<<" if row["fingertip"] == result["fingertip"] else ""
                print(f"      {row['fingertip'] * 1000:7.1f} mm  "
                      f"held {row['held']:2}/{row['reachable']:<3}"
                      f"  {100 * row['rate']:3.0f}%{mark}")
            config["fingertip"] = [0.0, 0.0, float(result["fingertip"])]
            config.setdefault("tpgpt_measured", {})["fingertip_calibrated"] = {
                "rate": result["rate"], "held": result["held"],
                "tried": result["tried"],
            }
            write_description(config, points, args.name, Path(args.out))
            install(args.name, Path(args.out))
            print(f"  calibration   fingertip {result['fingertip'] * 1000:.1f} mm "
                  f"({100 * result['rate']:.0f}% of grasps held)")
    return out


if __name__ == "__main__":
    main()
