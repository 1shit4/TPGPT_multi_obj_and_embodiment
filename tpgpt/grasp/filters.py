"""Deciding which generated grasps are worth executing.

GraspGen-X proposes on geometry alone. Measured here previously: only about a
quarter of its candidates approach from above, executing the top-scoring ones
lifted nothing, and two knocked the object over. The discriminator score does
not order real success -- in the sibling project a 0.797 grasp held and a 0.625
did not, then a 0.604 held and a 0.557 did not.

So candidates are put through a funnel, and **every stage keeps a record**: how
many entered, how many left, and whether it had to give up. Giving up matters.
A filter that empties the set and returns nothing turns "the hand would have
collided" into "there is no grasp", which is strictly less useful, so each
stage falls back to passing its input through and raises a flag instead. The
flag reaches the report; a silent fallback would be worse than no filter.

numpy and scipy only. No torch, no trimesh, no mesh loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.grasp.grasps import Grasp6D, contact_offset, grasp_to_eef_pose
from tpgpt.grasp.grippers import (
    GripperPair,
    gripper_geometry,
    gripper_points,
    resolve_pair,
)

#: How far a grasp's approach may point away from the camera that saw the
#: object, in degrees.
#:
#: A single depth image sees one surface, and the generator reconstructs a
#: plausible whole object from it -- including the far side, which it then
#: proposes grasps on. Deliberately generous: side approaches around 90 degrees
#: are legitimate and are much of the point of going 6-DoF.
MAX_BACKWARD_DEG = 100.0

#: How far the held-object point may sit from the target's cloud, in metres.
MAX_TARGET_DISTANCE = 0.04

#: Gripper surface within this of a scene point counts as touching it.
COLLISION_THRESHOLD = 0.010

#: Scene points inside the hand before it counts as a collision.
#:
#: Counts **distinct scene points**, not gripper points. Querying the other way
#: looks equivalent and is not: with the hand sampled every 7 mm and a 10 mm
#: threshold, one stray depth point lands within reach of dozens of gripper
#: samples and trips any threshold immediately.
MIN_COLLISION_HITS = 3

#: Distance the hand is backed off along its approach and re-checked.
APPROACH_CORRIDOR = 0.10
APPROACH_SAMPLES = 4

#: Clearance the object's width must leave inside the jaws, in metres.
JAW_MARGIN = 0.005

#: How far a candidate's approach may differ from the demonstration's.
#:
#: The transported policy executes the *demonstrated* motion warped into the new
#: scene, not the candidate pose. So a candidate approaching from the side will
#: never actually be executed from the side -- the hand still comes down from
#: above, and the only thing the candidate contributed was its closing axis,
#: which for a side grasp points somewhere the jaws will never travel.
#:
#: Measured: with this filter off, the chosen grasp on a cereal box approached
#: 93.7 degrees from vertical, the arm reached the right point to within 2.3 mm,
#: closed its jaws across the box's 9.7 cm face with an 8 cm opening, and held
#: nothing.
MAX_APPROACH_MISMATCH_DEG = 45.0

#: How close IK must get for a pose to count as reachable, in metres.
#:
#: Matched to what the impedance controller achieves, not to solver precision.
#: The controller tracks a commanded pose to about 2 cm, so a stricter test here
#: rejects poses the robot would in fact have reached. Measured on the top
#: shelf, where the arm is near its limit: IK converges to 4-8 mm, and a 5 mm
#: tolerance rejected every candidate for a slot the arm reaches comfortably.
REACH_TOLERANCE = 0.015

#: How far the hand backs away after placing. Short on purpose: for a top-down
#: placement, retreating means lifting, and lifting from the top shelf is the
#: furthest the arm ever has to reach.
RETREAT_DISTANCE = 0.06

#: Two grasps closer than this in position and approach angle are duplicates.
DUPLICATE_POSITION = 0.02
DUPLICATE_ANGLE_DEG = 20.0


@dataclass
class FilterStage:
    """What one stage of the funnel did."""

    name: str
    checks: str
    entered: int
    survived: int
    fallback: bool = False

    def describe(self) -> str:
        line = f"{self.name:<14} {self.entered:4d} -> {self.survived:4d}   {self.checks}"
        return line + "   [FELL BACK: constraint dropped]" if self.fallback else line


@dataclass
class FilterFunnel:
    """The whole funnel: what survived, and what each stage cost."""

    stages: list[FilterStage] = field(default_factory=list)
    survivors: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    flags: dict = field(default_factory=dict)

    def add(self, name, checks, entered, survived_indices, fallback=False):
        self.stages.append(
            FilterStage(name, checks, len(entered), len(survived_indices), fallback)
        )
        if fallback:
            self.flags[f"{name}_fell_back"] = True
        return survived_indices

    @property
    def rejected_by(self) -> str | None:
        """The stage that removed the most candidates, for explaining a failure."""
        if not self.stages:
            return None
        worst = max(self.stages, key=lambda s: s.entered - s.survived)
        return worst.name if worst.entered > worst.survived else None

    def describe(self) -> str:
        lines = [stage.describe() for stage in self.stages]
        if self.flags:
            lines.append(f"flags: {sorted(self.flags)}")
        return "\n".join(lines)


def _keep(indices: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(indices)[np.asarray(mask, dtype=bool)]


def _held_point(grasp: Grasp6D, pair: GripperPair) -> np.ndarray:
    """Where this hand would hold the object, in world coordinates."""
    position, rotation = grasp_to_eef_pose(grasp, pair)
    return position + rotation @ contact_offset(pair)


# --------------------------------------------------------------- visibility
def by_visibility(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    camera_positions: dict | np.ndarray,
    max_backward_deg: float = MAX_BACKWARD_DEG,
) -> np.ndarray:
    """Drop grasps approaching from a direction nothing ever saw.

    Compares the approach axis against the ray from the camera to the grasp --
    the viewing ray, not the optical axis, since an object off to one side is
    seen along a different ray than one dead ahead.
    """
    if isinstance(camera_positions, dict):
        if not camera_positions:
            return np.asarray(indices)
        cameras = np.asarray(list(camera_positions.values()), dtype=float)
    else:
        cameras = np.atleast_2d(np.asarray(camera_positions, dtype=float))

    limit = np.cos(np.radians(max_backward_deg))
    keep = []
    for i in indices:
        grasp = grasps[i]
        rays = grasp.position - cameras
        norms = np.linalg.norm(rays, axis=1)
        valid = norms > 1e-9
        if not np.any(valid):
            keep.append(True)
            continue
        cosines = (rays[valid] / norms[valid, None]) @ grasp.approach
        # Seen from *any* camera is enough.
        keep.append(bool(np.max(cosines) >= limit))
    return _keep(indices, keep)


# ------------------------------------------------------------------ target
def by_target(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    target_points: np.ndarray,
    pair: GripperPair,
    max_distance: float = MAX_TARGET_DISTANCE,
) -> np.ndarray:
    """Keep grasps that would actually hold the requested object.

    Tests the point where the hand *holds* something, not the gripper base. The
    base stands 103 to 195 mm back depending on the hand, and testing it instead
    rejected about 90% of valid grasps in the sibling project -- 6 of 58 kept on
    a cup, against 58 of 58 once the fingertips were used.
    """
    from scipy.spatial import cKDTree

    if len(target_points) == 0:
        return np.asarray(indices)
    tree = cKDTree(np.asarray(target_points, dtype=float))
    held = np.array([_held_point(grasps[i], pair) for i in indices])
    distances, _ = tree.query(held)
    return _keep(indices, distances <= max_distance)


# ---------------------------------------------------------------- jaw width
def by_jaw_width(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    target_points: np.ndarray,
    pair: GripperPair,
    margin: float = JAW_MARGIN,
    slab: float = 0.012,
) -> np.ndarray:
    """Drop grasps where the object is wider than the hand can open.

    Measures the object's width along that grasp's own closing axis, in the
    slab of cloud the jaws would travel through, rather than using a global
    bounding box: a bottle is narrow at the neck and wide at the base, and which
    one matters depends on where the grasp sits.
    """
    aperture = gripper_geometry(pair.graspgen).aperture
    points = np.asarray(target_points, dtype=float)
    if len(points) == 0:
        return np.asarray(indices)

    keep = []
    for i in indices:
        grasp = grasps[i]
        held = _held_point(grasp, pair)
        along = (points - held) @ grasp.approach
        slice_points = points[np.abs(along) <= slab]
        if len(slice_points) < 4:
            keep.append(True)  # too little cloud to judge; not the same as too wide
            continue
        projected = slice_points @ grasp.closing
        keep.append(bool(float(np.ptp(projected)) + margin <= aperture))
    return _keep(indices, keep)


# ---------------------------------------------------------------- collision
def _points_for_threshold(graspgen_name: str, threshold: float) -> np.ndarray:
    """Gripper samples fine enough that scene points cannot slip between them.

    The Robotiq 3-Finger samples at 11 mm with 1024 points, coarser than the
    10 mm collision threshold, so it needs more.
    """
    from scipy.spatial import cKDTree

    for n in (1024, 2048, 4096):
        points = gripper_points(graspgen_name, n=n)
        spacing, _ = cKDTree(points).query(points, k=2)
        if float(np.percentile(spacing[:, 1], 95)) < threshold:
            return points
    return gripper_points(graspgen_name, n=4096)


def by_collision(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    scene_points: np.ndarray,
    pair: GripperPair,
    threshold: float = COLLISION_THRESHOLD,
    min_hits: int = MIN_COLLISION_HITS,
    corridor: float = APPROACH_CORRIDOR,
    samples: int = APPROACH_SAMPLES,
    extra_points: np.ndarray | None = None,
) -> np.ndarray:
    """Drop grasps where the hand would hit something.

    Checks the final pose *and* the approach corridor, stepping the hand back
    along its own approach axis. Checking only the final pose misses the case
    where the hand has to pass through a neighbour to reach an otherwise clean
    grasp.

    Args:
        scene_points: Everything the hand must avoid. The target object must be
            excluded -- the gripper closes around it, so leaving it in makes
            every grasp collide with the thing it is meant to pick up.
        extra_points: Carried along with the hand, in world coordinates at the
            grasp pose. Used to check that the hand *plus the object it is
            holding* fits, which is what the place side needs.
    """
    from scipy.spatial import cKDTree

    scene = np.asarray(scene_points, dtype=float)
    if len(scene) == 0:
        return np.asarray(indices)
    tree = cKDTree(scene)
    hand = _points_for_threshold(pair.graspgen, threshold)
    reach = float(np.linalg.norm(hand, axis=1).max()) + threshold

    keep = []
    for i in indices:
        grasp = grasps[i]
        clear = True
        for step in np.linspace(0.0, corridor, samples):
            origin = grasp.position - grasp.approach * step
            world = origin + hand @ grasp.rotation.T
            if extra_points is not None:
                world = np.vstack([world, np.asarray(extra_points) - grasp.approach * step])
            # Prune to a ball about the pose before the expensive query.
            near = tree.query_ball_point(origin, reach + step)
            if not near:
                continue
            candidates = scene[near]
            hits = cKDTree(world).query_ball_point(candidates, threshold)
            if sum(1 for h in hits if h) >= min_hits:
                clear = False
                break
        keep.append(clear)
    return _keep(indices, keep)


# ------------------------------------------------------------- reachability
#: Fractions of the standoff at which the approach corridor is checked.
#:
#: Checking only the endpoints -- standoff and grasp -- passes candidates whose
#: *descent* is infeasible in the middle, and a hand that cannot be held at the
#: commanded angle 4 cm above the object never reaches the object. Measured by
#: replaying a warped path pose by pose, about a quarter of it is unreachable
#: with the commanded orientation while the endpoints alone look fine.
APPROACH_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.0)

#: How far the hand lifts straight up after closing, and at what fractions that
#: lift is checked. The object has to leave the table before anything else can
#: happen, and it leaves it vertically.
LIFT_HEIGHT = 0.08
LIFT_FRACTIONS = (0.5, 1.0)


def arm_collides(env, qpos, arm: str = "right", ignore: tuple = ()) -> str | None:
    """What the arm or hand is inside, at this joint configuration.

    **Why this exists.** ``solve_ik`` is joint angles and a Jacobian: it has no
    collision model at all. A pose can solve to 3 mm and be physically
    unreachable because a shelf is in the way, and nothing in the funnel used to
    notice. Measured on the tabletop scene's top cubby, whose slot is a 78 mm
    gap in ``x`` between the lower cubby's back panel and the upper one's
    180 mm back wall: **15 of 20 transported plans command the hand inside that
    wall**, by 4.8 to 79.9 mm. In physics the arm jams against it, stalls at
    1 mm per waypoint against the 5.3 commanded, and opens its jaws with the
    object still 4 to 15 cm above the board. `ROBOTICS_NOTES.md` 7.35.

    Uses MuJoCo's own narrowphase on the real geometry rather than a point-cloud
    proxy, because the shelf *is* real geometry here and a cloud of it would be
    both slower and less exact.

    Args:
        env: The live environment. Its state is saved and restored, so this is a
            query.
        qpos: Arm joint configuration to test.
        arm: Which arm.
        ignore: Substrings naming geoms whose contacts do not count -- the
            target object above all, since the fingers are *meant* to close
            around it.

    Returns:
        ``"<robot geom> <-> <scene geom>"`` for the deepest offending contact,
        or ``None`` when the configuration is clear.
    """
    import mujoco

    model, data = env.sim.model, env.sim.data
    controller = env.robots[0].composite_controller.part_controllers[arm]
    qpos_index = np.asarray(controller.qpos_index)
    saved = np.array(data.qpos)
    try:
        data.qpos[qpos_index] = np.asarray(qpos, dtype=float)
        mujoco.mj_forward(model._model, data._data)
        worst, worst_depth = None, 0.0
        for c in range(data._data.ncon):
            contact = data._data.contact[c]
            a = model.geom_id2name(contact.geom1) or ""
            b = model.geom_id2name(contact.geom2) or ""
            robot = [n for n in (a, b) if n.startswith(("robot0", "gripper0"))]
            other = [n for n in (a, b) if not n.startswith(("robot0", "gripper0"))]
            # Both robot, or neither: self-collision and scene-on-scene are not
            # this function's business.
            if len(robot) != 1 or len(other) != 1:
                continue
            if any(token and token in other[0] for token in ignore):
                continue
            if contact.dist < worst_depth:
                worst_depth = float(contact.dist)
                worst = f"{robot[0]} <-> {other[0]}"
        return worst
    finally:
        data.qpos[:] = saved
        mujoco.mj_forward(model._model, data._data)


def by_reachability(
    env,
    grasps: list[Grasp6D],
    indices: np.ndarray,
    pair: GripperPair,
    place_pose: tuple[np.ndarray, np.ndarray] | None = None,
    standoff: float = 0.10,
    retreat: float = RETREAT_DISTANCE,
    tolerance: float = REACH_TOLERANCE,
    arm: str = "right",
    check_collision: bool = True,
    ignore_collisions_with: tuple = (),
) -> tuple[np.ndarray, dict]:
    """Keep grasps the arm can reach, at the pick and the place *and around them*.

    A grasp is not a pose, it is a short piece of trajectory: come down the
    approach axis, close, lift, carry, come down again, release, retreat. This
    checks that whole neighbourhood rather than its endpoints --
    :data:`APPROACH_FRACTIONS` down the corridor at both ends, plus the lift
    after closing -- because a candidate can be perfectly reachable at the
    standoff and at the grasp while the descent between them is not.

    The poses are ``grip_site`` poses, which is the frame the IK solves in and
    the frame the controller is commanded in.

    **Reachable is not the same as clear, and until 7.35 only the first was
    checked.** ``solve_ik`` is joint angles and a Jacobian with no collision
    model, so a pose inside a shelf solves happily. With ``check_collision`` the
    IK solution for every corridor pose is additionally put into the model and
    tested with MuJoCo's own narrowphase (:func:`arm_collides`). Measured cost
    of not doing it: 15 of 20 transported plans command the hand inside the top
    cubby's back wall by up to 79.9 mm, the arm jams, and every placement
    becomes a drop from up to 15 cm.

    Args:
        check_collision: Also reject a pose whose IK solution puts the arm or
            hand inside scene geometry.
        ignore_collisions_with: Substrings naming geoms whose contacts do not
            count. **The target object belongs here**: the fingers are meant to
            close around it, so its contacts are the grasp, not a fault.

    Returns the survivors and a per-stage tally of where the candidate failed,
    which is what distinguishes "the object is out of reach" from "the shelf
    is". A pose rejected for collision rather than for reach is tallied under
    ``"<segment>_collision"``, so the two causes stay separable -- they need
    completely different fixes.
    """
    from tpgpt.sim.kinematics import reachable

    failures = {"pre_grasp": 0, "grasp": 0, "lift": 0, "place": 0, "retreat": 0}
    failures.update({f"{k}_collision": 0 for k in list(failures)})
    keep = []
    for i in indices:
        grasp = grasps[i]
        position, rotation = grasp_to_eef_pose(grasp, pair)
        up = np.array([0.0, 0.0, 1.0])

        poses, names = [], []
        # Down the approach corridor onto the object.
        for f in APPROACH_FRACTIONS:
            poses.append((position - grasp.approach * (standoff * f), rotation))
            names.append("pre_grasp" if f > 0 else "grasp")
        # Straight up, still holding the grasp orientation.
        for f in LIFT_FRACTIONS:
            poses.append((position + up * (LIFT_HEIGHT * f), rotation))
            names.append("lift")

        if place_pose is not None:
            place_position, place_rotation = place_pose
            # Down onto the shelf, then back out the way it came.
            for f in APPROACH_FRACTIONS:
                poses.append(
                    (place_position - grasp.approach * (retreat * f), place_rotation)
                )
                names.append("place" if f == 0 else "retreat")

        results = reachable(env, poses, arm=arm, position_tolerance=tolerance)
        ok = True
        for name, result in zip(names, results):
            if not result.reachable:
                failures[name] += 1
                ok = False
                break
            if check_collision and arm_collides(
                env, result.qpos, arm=arm, ignore=ignore_collisions_with
            ):
                failures[f"{name}_collision"] += 1
                ok = False
                break
        keep.append(ok)
    return _keep(indices, keep), failures


def by_demonstration_consistency(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    reference_approach: np.ndarray,
    max_angle_deg: float = MAX_APPROACH_MISMATCH_DEG,
) -> np.ndarray:
    """Keep grasps the transportation map can carry, which is not the same as good grasps.

    **This is a constraint of the method, not a property of a good grasp**, and
    the distinction matters because everything else in this funnel is the
    latter. Whether a grasp holds depends on the object, the gripper and the
    scene; the demonstration has nothing to do with it. What the demonstration
    constrains is the *map*.

    ``phi`` must carry the source keypoint cube onto the target one, so the
    angle between the two grasp frames is a rotation the warp has to realise,
    and it degrades with that angle. Measured across 20 cells:

    ========================  ==============
    source-to-target rotation ``min det(J)``
    ========================  ==============
    3 - 19 deg                0.94 - 0.99
    74 - 135 deg              0.33 - 0.79
    149 - 179 deg             **-0.06 - 0.16**
    ========================  ==============

    Past about 145 degrees the map turns inside out. Section 8g measured the
    consequence of dropping this filter: the grasp-pose cube produced a valid
    map in only **7 of 16** cells, at a median ``min det`` of **-0.031**, with a
    median transported orientation error of 116.5 degrees.

    So the filter earns its place, but as a *feasibility bound on the warp*. It
    must not be read as a quality criterion and must not be used to rank: a
    candidate 6 degrees off the demonstration is not a better **grasp** than one
    40 degrees off, it is merely one the map can transport. Ranking is
    :func:`filter_grasps`' job and it uses the discriminator's score.

    The cheap answer to the tension is this filter; the expensive one is a
    demonstration per approach direction.
    """
    reference = np.asarray(reference_approach, dtype=float)
    reference = reference / np.linalg.norm(reference)
    limit = np.cos(np.radians(max_angle_deg))
    return _keep(indices, [float(grasps[i].approach @ reference) >= limit for i in indices])


# --------------------------------------------------------------- duplicates
def suppress_duplicates(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    position_tolerance: float = DUPLICATE_POSITION,
    angle_tolerance_deg: float = DUPLICATE_ANGLE_DEG,
    key=None,
) -> np.ndarray:
    """Keep the best of each cluster of near-identical grasps.

    The generator returns hundreds of poses whose scores sit within 0.02 to 0.05
    of each other, so an argmax picks an arbitrary member of a cluster. Thinning
    them keeps the surviving set *diverse*, which is what lets a later filter
    reject one candidate and still leave somewhere to go.

    **"Best" means the discriminator's score, and that is deliberate.** A
    cluster is anything within :data:`DUPLICATE_POSITION` (20 mm) and
    :data:`DUPLICATE_ANGLE_DEG` (20 degrees), so two members can differ by
    20 degrees of approach and the choice between them is not cosmetic. It is
    still a *grasp quality* choice, which is a property of the object, the
    gripper and the scene -- so the score is the right key and the
    demonstration is not.

    A previous version keyed this on agreement with the demonstration, on the
    grounds that it discarded the best-aligned candidate on 4 of 20 cells. That
    was a mistake: keeping the better-quality member of a cluster is this
    function's job, and "it dropped the one that best matched the source" is not
    evidence of a fault.

    Args:
        key: ``key(index) -> float``, smaller is better, deciding which member
            of a cluster survives. Defaults to the planner's score. Exposed so a
            caller with a genuinely better quality estimate can supply one --
            not so that a *task* criterion can be smuggled in here.
    """
    key = key if key is not None else (lambda i: -grasps[i].score)
    order = sorted(indices, key=key)
    limit = np.cos(np.radians(angle_tolerance_deg))
    kept: list[int] = []
    for i in order:
        grasp = grasps[i]
        duplicate = any(
            np.linalg.norm(grasp.position - grasps[j].position) < position_tolerance
            and float(grasp.approach @ grasps[j].approach) > limit
            for j in kept
        )
        if not duplicate:
            kept.append(i)
    return np.array(sorted(kept), dtype=int)


# ------------------------------------------------------------------- funnel
def filter_grasps(
    grasps: list[Grasp6D],
    gripper: str | GripperPair,
    target_points: np.ndarray,
    scene_points: np.ndarray | None = None,
    camera_positions: dict | None = None,
    env=None,
    place_pose: tuple[np.ndarray, np.ndarray] | None = None,
    held_points: np.ndarray | None = None,
    reference_approach: np.ndarray | None = None,
    target_name: str | None = None,
) -> FilterFunnel:
    """Run the whole funnel, recording what each stage cost.

    Stages run cheapest-first so the expensive ones see fewer candidates:
    visibility and target containment are vector arithmetic, jaw width touches
    the cloud, collision builds trees, and reachability solves IK.

    Every stage that would empty the set instead passes its input through and
    raises a flag. That keeps a hard scene answering "here is the least bad
    grasp, and here is what is wrong with it" rather than "no grasp exists".

    Args:
        target_name: The object being picked, so the reachability stage can
            tell the fingers closing around it -- which is the grasp -- from
            the hand fouling the shelf, which is not. Without it a legitimate
            grasp is rejected for touching its own target.
    """
    pair = gripper if isinstance(gripper, GripperPair) else resolve_pair(gripper)
    funnel = FilterFunnel()
    indices = np.arange(len(grasps))
    funnel.stages.append(FilterStage("generated", "candidates from GraspGen-X",
                                     len(grasps), len(grasps)))

    def stage(name, checks, survivors, current):
        if len(survivors) == 0 and len(current) > 0:
            return funnel.add(name, checks, current, current, fallback=True)
        return funnel.add(name, checks, current, survivors)

    if camera_positions:
        indices = stage(
            "visibility", f"approaches within {MAX_BACKWARD_DEG:.0f} deg of a camera's view",
            by_visibility(grasps, indices, camera_positions), indices,
        )
    if reference_approach is not None:
        indices = stage(
            "demonstrated",
            f"approaches within {MAX_APPROACH_MISMATCH_DEG:.0f} deg of the demonstrated one",
            by_demonstration_consistency(grasps, indices, reference_approach), indices,
        )
    indices = stage(
        "on target", f"the held object lands within {MAX_TARGET_DISTANCE * 100:.0f} cm of the cloud",
        by_target(grasps, indices, target_points, pair), indices,
    )
    indices = stage(
        "jaw width", "the object fits between the jaws at the grasp height",
        by_jaw_width(grasps, indices, target_points, pair), indices,
    )
    if scene_points is not None and len(scene_points):
        indices = stage(
            "collision", "the hand and its approach are clear of the rest of the scene",
            by_collision(grasps, indices, scene_points, pair, extra_points=held_points),
            indices,
        )
    if env is not None:
        survivors, failures = by_reachability(
            env, grasps, indices, pair, place_pose,
            # The fingers are *meant* to close around the target, so its
            # contacts are the grasp rather than a fault. Everything else --
            # the shelf above all -- counts.
            ignore_collisions_with=tuple(t for t in (target_name,) if t),
        )
        funnel.flags["reach_failures"] = failures
        indices = stage(
            "reachable",
            "the arm can hold the grasp, the lift and the placement, and the "
            "approach corridors into both",
            survivors, indices,
        )
    # **Rank by the discriminator's score.** Whether a grasp is good is a
    # property of the object, the gripper and the scene -- nothing about the
    # demonstration enters it -- and the score is the one estimate of it we
    # have: GraspGenX's discriminator is trained on a simulated grasp dataset
    # of some two billion grasps across 32 grippers and 8000+ objects, so it is
    # a learned grasp-quality signal by construction, not an arbitrary number.
    #
    # A previous version of this ranked by agreement with the demonstration
    # instead, and that was wrong in principle: it subordinates grasp quality to
    # an artefact of the transport method. Where the demonstration legitimately
    # enters is ``by_demonstration_consistency`` above, as a hard **constraint**
    # -- see that function for why the map, not the grasp, needs it.
    #
    # The two attempts to validate the score in this project both lacked the
    # power to say anything. 8h had 19 of 20 grasps lift, so there was no
    # variance to correlate, and it reported exactly that; quoting it as "the
    # score predicts nothing" was a misuse of an underpowered null. The sibling
    # project's measurement is real but small (4 of 6 and 3 of 8 held, and the
    # score did not order them) and was taken in a different simulator on
    # cluttered mesh scenes.
    rank = lambda i: -grasps[i].score  # noqa: E731

    indices = stage(
        "distinct", f"duplicates within {DUPLICATE_POSITION * 100:.0f} cm thinned out",
        suppress_duplicates(grasps, indices, key=rank), indices,
    )

    funnel.survivors = np.array(sorted(indices, key=rank), dtype=int)
    funnel.flags["ranked_by"] = "discriminator score"
    return funnel
