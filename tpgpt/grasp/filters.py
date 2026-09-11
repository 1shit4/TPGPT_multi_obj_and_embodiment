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

#: Cloud points the jaws' slab needs before the width it measures means anything.
#:
#: An observed cloud is a **lower bound** on an object's width: the camera sees
#: the near surface and the object continues behind it. So "measured width fits"
#: is never a sound conclusion, while "measured width already exceeds the
#: aperture" is. This threshold marks the cases where the measurement is too
#: thin to act on at all, so they can be reported as a perception failure rather
#: than silently accepted as a fit.
#:
#: Measured on the tabletop scene at the default 256 px, three fused cameras,
#: one pixel of mask erosion, comparing the slab's extent along each grasp's own
#: closing axis against the object's true width from its mesh vertices:
#:
#: =============  =========  ==============  ============  ============
#: cell           slab pts   slab measured   true width    under-read
#: =============  =========  ==============  ============  ============
#: ``yumi/can``   11          5.9 mm         52.8 mm       **46.9 mm**
#: robotiq85/can  13         27.0 mm         55.4 mm       **28.4 mm**
#: yumi/bread     100        38.4 mm         41.8 mm       3.4 mm
#: xarm/bread     97         44.2 mm         53.5 mm       9.3 mm
#: panda/bread    89         44.2 mm         53.0 mm       8.8 mm
#: =============  =========  ==============  ============  ============
#:
#: The can is 57 cloud points in total, so no grasp on it can populate a slab;
#: the bread is 167 and every grasp on it can. Set to match the pipeline's own
#: ``MIN_CLOUD_POINTS`` of 40, which draws the same line for the same reason.
#:
#: **This does not make the check sound**, and the limit is not sparsity. At
#: 512 px ``yumi/can`` has 108 points in its slab and still measures 21.3 mm
#: against 52.8 mm, because the cloud is a partial view and no density fixes
#: that. See ``ROBOTICS_NOTES.md`` for the open item.
MIN_JAW_WIDTH_POINTS = 40

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

#: How far a grasp's approach may lean away from straight down, in degrees.
#:
#: **This is the table talking, not the demonstration.** The support surface has
#: an outward normal -- straight up on a table -- and a hand cannot arrive from
#: the far side of it, because the far side is solid. So an approach direction
#: with an *upward* component describes a hand rising through the table top,
#: which is not a grasp at all.
#:
#: The angle is measured from straight down, so 0 degrees is a purely top-down
#: descent, 90 is exactly horizontal, and anything past 90 points upward. The
#: limit is set slightly under the geometric bound (85 rather than 90) because
#: the fingers hang *below* the tool centre point: a hand held exactly level
#: still has finger tips grazing the surface the object stands on.
#:
#: Measured on the default tabletop scene, seed 0, Panda, over the 100
#: candidates GraspGen-X returns per object:
#:
#: ========  ==============  ================  ===============
#: object    median approach within 85 deg     top-scoring one
#: ========  ==============  ================  ===============
#: cereal    89.3 deg        42 of 100         19.0 deg
#: milk      92.8 deg        27 of 100         **101.8 deg**
#: can       88.5 deg        39 of 100         **100.7 deg**
#: bread     88.4 deg        45 of 100         6.9 deg
#: ========  ==============  ================  ===============
#:
#: So the candidate set is dominated by side grasps, and on two of the four
#: objects the planner's own best candidate approaches from **underneath**.
#: That is the run-i failure of `FINDINGS.md` 8m -- ranking by score with no
#: constraint placed nothing in twenty cells -- and this rejects those two
#: without consulting the source demonstration at all.
SUPPORT_APPROACH_MAX_DEG = 85.0

#: How much of the arrival corridor must be clear, as a multiple of the hand's
#: own length behind the grasp point.
#:
#: The hand does not arrive as a point. It arrives as a body roughly
#: ``tcp_depth`` long trailing behind the fingertips -- 103 to 195 mm across the
#: registry -- so the corridor it sweeps to reach a release pose is at least
#: that long. Taking the full length is the honest figure; taking a fraction of
#: it would be asserting that the wrist may end up inside a wall.
PLACE_CORRIDOR_FRACTION = 1.0

#: Rays cast across the hand's cross-section when testing an arrival corridor.
#:
#: One central ray is not enough: it passes through the gap between two fingers
#: while the fingers themselves are inside a panel. These are spread over a disc
#: of the hand's own lateral radius, so the bundle covers the swept cross
#: section rather than its centre line.
PLACE_CORRIDOR_RAYS = 9

#: Penetration beyond which a waypoint counts as inside scene geometry, metres.
#:
#: Not zero, and the reason is 7.38: **contact is normal in this scene**. A
#: placement ends with the object resting on the board and the fingers within a
#: millimetre of it; a pick begins with the fingers straddling an object that is
#: itself touching the table. A rule that fired on literal overlap would fire on
#: every candidate. 2 mm is below the thickness of anything that matters and
#: above the incidental contact of a set-down, whose measured depth on cells
#: that succeed is 0.05 to 0.17 mm.
PATH_PENETRATION_TOLERANCE = 0.002

#: Fraction of a path's waypoints that may sit inside scene geometry.
#:
#: **Sustained penetration, not the deepest**, and the threshold is calibrated
#: on this instrument rather than inherited. 7.38 reached the same conclusion
#: with a different one -- MuJoCo's narrowphase applied to inverse-kinematics
#: solutions -- and a number carried between instruments is an assumption.
#:
#: Measured with *this* one (``outputs/path_study_iii``) over the twenty cells
#: of Experiment O run iii, whose outcomes are already known:
#:
#: =========  ====  =============================  ===================
#: outcome    n     inside fraction, min-med-max   max depth, med
#: =========  ====  =============================  ===================
#: placed     11    0.00 - **0.165** - **0.340**   5.9 mm
#: failed      9    0.00 - 0.335 - **0.855**       6.0 mm
#: =========  ====  =============================  ===================
#:
#: The separation is **one-sided and has a wide gap in it**: nothing was
#: observed between 0.34 and 0.70, every cell above 0.40 failed, and no cell
#: that placed went past 0.34. Any threshold in [0.35, 0.70] gives the identical
#: answer on these cells -- it rejects **4 of the 9 failures and 0 of the 11
#: successes** -- so 0.40 is the conservative end of a flat region rather than a
#: number fitted to a boundary.
#:
#: **Depth is not the statistic**, and this data shows why more sharply than
#: 7.38 did: ``max_depth`` saturates at 6.0 mm on fourteen of the twenty cells,
#: because ``shelf_top_back`` is a 12 mm slab and 6 mm is as deep inside a 12 mm
#: slab as a point can get. A bounded quantity cannot order unbounded severity.
#: How *long* the hand stays inside is unbounded and does order it.
#:
#: It is a partial criterion, deliberately: it reaches 4 of the 9 failures and
#: two of the rest involve no penetration at all.
PATH_PENETRATION_FRACTION = 0.40

#: How far from an object's centre of mass a grip may be taken, in metres.
#:
#: **Measured, and deliberately loose.** Experiment P (`FINDINGS.md` 8n) ran
#: five hand/object pairs at four grasp poses each -- twenty grasps -- and
#: recorded the horizontal distance from the object's centre of mass to the
#: grasp point alongside how far the object was actually carried:
#:
#: =========================  ===========  ===============
#: outcome                    n            median offset
#: =========================  ===========  ===============
#: carried past 200 mm        11           **6.3 mm**
#: did not                     9           **19.1 mm**
#: =========================  ===========  ===============
#:
#: and every grasp under 10 mm carried, bar one. The mechanism is a lever arm:
#: the object's weight acting this far from the grip applies a moment that
#: rotates it in the jaws, and a hand that pinches near an edge lets it pivot
#: out.
#:
#: 15 mm rejects **6 of the 9** grasps that failed to carry and costs **1 of
#: the 11** that succeeded -- ``panda/cereal``, which carried 418 mm from
#: 23.4 mm off centre. That one cell, and ``panda/cereal`` again *failing* at
#: 6.2 mm, are why this is a loose filter that falls back rather than a gate:
#: the criterion is real and it is not sufficient. Whether the object slips
#: also depends on friction, on finger area and on how much of the body is
#: between the jaws, none of which this sees.
#:
#: One study of twenty grasps set this number, so treat it as an operating
#: point rather than a constant of nature.
MAX_CENTRE_OFFSET = 0.015

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
def jaw_width_verdicts(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    target_points: np.ndarray,
    pair: GripperPair,
    margin: float = JAW_MARGIN,
    slab: float = 0.012,
    min_points: int = MIN_JAW_WIDTH_POINTS,
) -> dict[int, str]:
    """Per-grasp verdict on whether the object fits between the jaws.

    Three outcomes, because two are not enough:

    ``"too_wide"``
        The cloud *already* spans more than the hand can open. Sound whatever
        the point count, because the observed extent is a lower bound on the
        object: if the lower bound does not fit, neither does the object.
    ``"unverified"``
        Fewer than ``min_points`` in the slab the jaws sweep. The extent such a
        slab reports is not a measurement of the object -- on ``yumi/can`` it
        read 5.9 mm across a 52.8 mm can -- so nothing may be concluded from it
        in either direction.
    ``"fits"``
        Enough cloud to measure, and the measurement leaves ``margin``.

    Note what ``"fits"`` does **not** mean. The cloud is a partial view, so its
    extent under-reads the object and this verdict stays optimistic; it says the
    evidence available does not forbid the grasp, not that the grasp is possible.
    Making it sound needs an *upper* bound on the object, which a depth cloud
    cannot provide -- see ``ROBOTICS_NOTES.md`` for the visual-hull open item.
    """
    aperture = gripper_geometry(pair.graspgen).aperture
    points = np.asarray(target_points, dtype=float)
    verdicts: dict[int, str] = {}
    for i in indices:
        key = int(i)
        if len(points) == 0:
            verdicts[key] = "unverified"
            continue
        grasp = grasps[key]
        held = _held_point(grasp, pair)
        along = (points - held) @ grasp.approach
        slice_points = points[np.abs(along) <= slab]
        if len(slice_points) >= 2:
            width = float(np.ptp(slice_points @ grasp.closing))
            # A lower bound that already exceeds the aperture settles it, and
            # needs no minimum point count to do so.
            if width + margin > aperture:
                verdicts[key] = "too_wide"
                continue
        verdicts[key] = "unverified" if len(slice_points) < min_points else "fits"
    return verdicts


def by_jaw_width(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    target_points: np.ndarray,
    pair: GripperPair,
    margin: float = JAW_MARGIN,
    slab: float = 0.012,
    min_points: int = MIN_JAW_WIDTH_POINTS,
) -> np.ndarray:
    """Keep only grasps whose fit between the jaws was actually verified.

    Measures the object's width along that grasp's own closing axis, in the slab
    of cloud the jaws would travel through, rather than using a global bounding
    box: a bottle is narrow at the neck and wide at the base, and which one
    matters depends on where the grasp sits.

    A grasp whose width could not be measured is dropped rather than waved
    through. It used to be waved through, on the reasoning that absence of
    evidence is not evidence of a wide object -- which is true, and was the
    wrong action to take on it. ``yumi/can`` is the case that settled it: an
    11-point slab reported 5.9 mm across a 50.0 mm can, the check
    ``5.9 + 5 <= 50`` passed, and the yumi's jaws open to exactly 50.0 mm, so
    the hand could only ever touch the can tangentially. In physics the jaws
    then travelled 39.6 mm *through* it, wedging it 16.1 mm sideways while the
    grip force bled from 20.3 N to 0.6 N.

    Dropping them is safe because the funnel never returns an empty set: if this
    stage removes everything, ``filter_grasps`` restores its input and raises
    ``jaw width_fell_back``, and the unverified count reaches the report either
    way. So the effect is to *prefer* a grasp whose fit is known whenever one
    exists, and to say so plainly when none does.
    """
    verdicts = jaw_width_verdicts(
        grasps, indices, target_points, pair, margin, slab, min_points
    )
    return _keep(indices, [verdicts[int(i)] == "fits" for i in indices])


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


# ------------------------------------------------------- scene-derived zones
def by_support_approach(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    support_normal: np.ndarray = (0.0, 0.0, 1.0),
    max_angle_deg: float = SUPPORT_APPROACH_MAX_DEG,
) -> np.ndarray:
    """Drop grasps that would have the hand come up through the table.

    **The pick-side no-approach zone, derived from the scene rather than from
    the demonstration.** Every object in this task stands on a surface, and that
    surface is solid: a hand cannot reach the object from below it. The
    constraint is therefore a half space, fixed by the support's own outward
    normal, and it holds whatever was demonstrated and whatever the object is.

    Contrast :func:`by_demonstration_consistency`, which this is meant to
    replace. That one asks whether a candidate resembles the *source*, which
    ties every target scene to one recorded motion and rejects perfectly good
    grasps for being unlike it. This asks whether the hand could physically be
    where the candidate says it is.

    The test is one dot product: the angle between the candidate's approach axis
    and straight down, capped at :data:`SUPPORT_APPROACH_MAX_DEG`. See that
    constant for the measured candidate distributions and for why the cap sits
    at 85 rather than at the geometric 90.

    Args:
        support_normal: Outward normal of the surface the object rests on --
            ``+z`` for a table. Taken as an argument rather than assumed,
            because the same rule is what a sloped or vertical support would
            need and the only thing that changes is this vector.

    Returns:
        The surviving indices.
    """
    normal = np.asarray(support_normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    limit = float(np.cos(np.radians(max_angle_deg)))
    return _keep(
        indices, [float(grasps[i].approach @ (-normal)) >= limit for i in indices]
    )


def hand_envelope(pair: GripperPair, n: int = 1024) -> tuple[float, float]:
    """How far a hand reaches behind its grasp point, and how wide it is.

    Both read off the gripper's own published surface sample rather than
    assumed, so they are per-hand facts: the registry spans 103 to 195 mm of
    fingertip depth and the bodies differ far more than that.

    Returns:
        ``(length, radius)`` in metres -- how far the body extends back along
        the approach axis from the fingertip point, and the largest distance of
        any part of it from that axis.
    """
    points = gripper_points(pair.graspgen, n=n)
    depth = float(gripper_geometry(pair.graspgen).tcp_depth)
    # The sample is in the gripper base frame: +Z the approach, origin at the
    # base. The fingertip point sits ``depth`` along +Z, so measuring from it
    # means shifting the sample back by that much.
    behind = float(np.max(depth - points[:, 2]))
    radius = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    return max(behind, 0.0), radius


def free_corridor(
    env,
    point: np.ndarray,
    arrival: np.ndarray,
    length: float,
    radius: float,
    exclude: tuple[str, ...] = (),
    rays: int = PLACE_CORRIDOR_RAYS,
) -> str | None:
    """What, if anything, blocks a hand arriving at ``point`` along ``arrival``.

    Casts a bundle of rays **backwards** from the destination along the
    direction the hand came from, spread over a disc of the hand's own lateral
    radius so the bundle covers the cross section the body sweeps rather than
    its centre line. One central ray is not enough: it threads the gap between
    two fingers while the fingers are inside a panel.

    Uses :func:`~tpgpt.perception.obstacles.first_obstruction`, which skips
    geoms that are drawn but not solid -- this scene puts a translucent marker
    box at every shelf slot, and an unguarded cast reports it as an obstacle.

    Args:
        point: Where the hand ends up, world coordinates.
        arrival: Unit direction the hand travels along to get there, i.e. the
            grasp's approach axis.
        length: How far back the corridor must be clear.
        radius: Half width of the bundle.
        exclude: Substrings naming geoms that do not count -- the object being
            carried belongs here.

    Returns:
        The name of the first blocking geom, or ``None`` when the corridor is
        clear.
    """
    from tpgpt.perception.obstacles import first_obstruction

    point = np.asarray(point, dtype=float).reshape(3)
    arrival = np.asarray(arrival, dtype=float).reshape(3)
    arrival = arrival / np.linalg.norm(arrival)

    # Any two axes perpendicular to the arrival direction will do; what matters
    # is that the offsets span the disc rather than lie along one line.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(arrival @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    u = np.cross(arrival, reference)
    u /= np.linalg.norm(u)
    v = np.cross(arrival, u)

    offsets = [np.zeros(3)]
    for k in range(max(rays - 1, 0)):
        angle = 2.0 * np.pi * k / max(rays - 1, 1)
        offsets.append(radius * (np.cos(angle) * u + np.sin(angle) * v))

    for offset in offsets:
        _, name = first_obstruction(
            env, point + offset, -arrival, max_distance=length, exclude=exclude
        )
        if name is not None:
            return name
    return None


def by_place_approach(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    env,
    pair: GripperPair,
    release_positions,
    exclude: tuple[str, ...] = (),
    corridor_fraction: float = PLACE_CORRIDOR_FRACTION,
) -> tuple[np.ndarray, dict]:
    """Drop grasps whose release would have the hand arrive through a wall.

    **The place-side no-approach zone.** The pick side has one blocked
    direction, downwards, and it never changes. The place side has as many as
    the destination has neighbours, and which ones they are is a property of the
    shelf: the default ``cubby`` variant walls a slot on three sides, the
    ``open`` variant on none, and the ``enclosed`` variant adds a roof that
    makes a top-down placement impossible outright. Hand-listing them would
    encode one of those three into the code.

    So the blocked set is **not listed anywhere**. It is measured, per
    candidate, by asking the model whether the corridor the hand would sweep to
    reach its release pose is clear (:func:`free_corridor`). The corridor's
    length and width come from the hand's own published geometry
    (:func:`hand_envelope`), so a 270 mm-deep Robotiq 2F-140 is held to a longer
    clear run than a 97 mm Panda, which is the physical truth.

    Measured on the default scene at ``top_middle``, casting along the six world
    axes from 60 mm above the board: ``+x`` is blocked by ``shelf_top_back`` at
    **38 mm**, ``+/-y`` by the two side walls at **198 mm**, ``-z`` by the board
    itself at 60 mm, and ``+z`` and ``-x`` are open. That is the cubby's own
    geometry, read out rather than written down; on the ``open`` variant the
    same probe finds only the board.

    **The grasp does not change orientation between pick and place.** Once the
    jaws shut the object is rigid with the hand, so the release pose inherits
    the pick's rotation and the arrival direction is the grasp's own approach
    axis. That is the same assumption :func:`by_reachability` already makes
    through :func:`~tpgpt.experiments.pipeline.place_pose_for`, so the two
    stages agree about where the hand ends up.

    Args:
        release_positions: Where the hand's fingertip point must be at release,
            one per entry of ``indices`` or one shared position. Derived from
            the slot and the object's height, not chosen.
        exclude: Substrings naming geoms that do not count. The object being
            carried belongs here: it arrives with the hand.

    Returns:
        ``(survivors, blocked_by)`` -- the surviving indices, and a tally of
        which geom blocked how many candidates, so a cell rejected here can be
        attributed to a named panel rather than to "the shelf".
    """
    indices = np.asarray(indices)
    length, radius = hand_envelope(pair)
    length *= float(corridor_fraction)
    positions = np.atleast_2d(np.asarray(release_positions, dtype=float))
    if len(positions) == 1:
        positions = np.repeat(positions, len(indices), axis=0)
    if len(positions) != len(indices):
        raise ValueError(
            f"{len(positions)} release positions for {len(indices)} candidates; "
            "pass one per candidate or exactly one shared position"
        )

    blocked_by: dict = {}
    keep = []
    for position, i in zip(positions, indices):
        name = free_corridor(
            env, position, grasps[i].approach, length, radius, exclude=exclude
        )
        if name is not None:
            blocked_by[name] = blocked_by.get(name, 0) + 1
        keep.append(name is None)
    return _keep(indices, keep), blocked_by


# --------------------------------------------------- the executed trajectory
def path_clearance(
    env,
    positions: np.ndarray,
    rotations: np.ndarray,
    pair: GripperPair,
    exclude: tuple[str, ...] = (),
    carried_points: np.ndarray | None = None,
    carry_span: tuple[int, int] | None = None,
    obstacles=None,
    n_points: int = 512,
) -> dict:
    """How far inside scene geometry a transported path puts the hand, waypoint by waypoint.

    **Why the funnel needs this and why nothing already did it.**
    :func:`by_reachability` validates **13 poses** per candidate -- five down the
    approach, two on the lift, five at the placement and retreat -- while the
    replay executes **200**. Measured in 7.38, a candidate passes that sample and
    then collides at 79 to 162 of the other 187 waypoints. The thing checked and
    the thing executed are not the same trajectory.

    This checks the whole thing, and it is cheap because it needs no inverse
    kinematics: the hand's pose at every waypoint *is* the transported path.
    The hand is its own published surface sample, placed at each pose; the
    obstacles are the scene's solid primitives
    (:func:`~tpgpt.perception.obstacles.scene_obstacles`), which give an exact
    signed depth rather than a nearest-neighbour yes/no. Depth is the quantity
    the criterion turns on -- see :data:`PATH_PENETRATION_FRACTION`.

    **Frames.** ``positions`` are fingertip (TCP) positions and ``rotations``
    are in the *grasp* convention, ``+Z`` the approach and ``+X`` the closing
    direction -- which is what a transported label set is, and what the gripper's
    surface sample is expressed in. The sample is in the gripper *base* frame,
    so it is shifted back along the approach by the hand's own ``tcp_depth``
    before being placed. That shift is exactly the one
    :func:`~tpgpt.grasp.grasps.grasp_to_eef_pose` makes and unmakes, so the
    hand lands where the replay actually drives it.

    Args:
        positions: ``(m, 3)`` fingertip positions along the path.
        rotations: ``(m, 3, 3)`` grasp-convention orientations along the path.
        pair: The hand being checked.
        exclude: Substrings naming geoms that are not obstacles -- the object
            being carried, above all.
        carried_points: ``(k, 3)`` world points of the held object, given in the
            pose it has at waypoint ``carry_span[0]`` -- which is where the jaws
            shut. From there it is rigid with the hand, so its pose at every
            later waypoint follows from the hand's. Passing it asks whether the
            hand *plus its load* fits, which is what the place side needs.
        carry_span: ``(grasp, release)`` waypoint indices bounding the interval
            the object is actually held over. Outside it the object is standing
            on a surface and is not attached to the hand at all, so checking it
            there would report the object colliding with the table it is resting
            on. Comes from
            :func:`~tpgpt.sim.keypoints.carry_indices`, which reads the
            demonstration's own gripper channel.
        obstacles: Pre-built obstacle list, to avoid rebuilding it per candidate.
        n_points: Hand surface samples. 512 puts the samples about 10 mm apart
            on a Panda, fine enough that a 12 mm shelf panel cannot pass between
            them.

    Returns:
        A dict with ``depth`` (per waypoint, metres), ``depth_object`` (likewise
        for the carried load, or ``None``), ``inside_fraction`` (share of
        waypoints deeper than :data:`PATH_PENETRATION_TOLERANCE`),
        ``max_depth``, ``median_inside_depth`` (median depth over the waypoints
        that are inside, which is 7.38's discriminating statistic), and
        ``culprits`` (a tally of which geom was deepest, by waypoint).
    """
    from tpgpt.perception.obstacles import deepest_penetration, scene_obstacles

    if obstacles is None:
        obstacles = scene_obstacles(env, exclude=exclude)
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    if len(positions) != len(rotations):
        raise ValueError(
            f"{len(positions)} positions against {len(rotations)} rotations"
        )

    hand = gripper_points(pair.graspgen, n=n_points)
    depth_tcp = float(gripper_geometry(pair.graspgen).tcp_depth)

    load_local, carry_from, carry_to = None, 0, len(positions)
    if carried_points is not None and len(carried_points):
        carry_from, carry_to = carry_span if carry_span else (0, len(positions))
        carry_from = int(np.clip(carry_from, 0, len(positions) - 1))
        carry_to = int(np.clip(carry_to, carry_from, len(positions) - 1))
        # Held rigidly, so its pose in the hand's frame is fixed from the moment
        # the jaws shut. Expressed there once rather than re-derived per step.
        load_local = (
            np.asarray(carried_points, dtype=float) - positions[carry_from]
        ) @ rotations[carry_from]

    depths = np.zeros(len(positions))
    load_depths = np.zeros(len(positions)) if load_local is not None else None
    culprits: dict = {}
    for k, (p, R) in enumerate(zip(positions, rotations)):
        base = p - R[:, 2] * depth_tcp
        world = base + hand @ R.T
        depth, name = deepest_penetration(world, obstacles)
        depths[k] = depth
        if name is not None and depth > PATH_PENETRATION_TOLERANCE:
            culprits[name] = culprits.get(name, 0) + 1
        if load_local is not None and carry_from <= k <= carry_to:
            load_world = p + load_local @ R.T
            load_depths[k], _ = deepest_penetration(load_world, obstacles)

    inside = depths > PATH_PENETRATION_TOLERANCE
    return {
        "depth": depths,
        "depth_object": load_depths,
        "waypoints": int(len(depths)),
        "inside_waypoints": int(inside.sum()),
        "inside_fraction": float(inside.mean()) if len(depths) else 0.0,
        "max_depth": float(depths.max()) if len(depths) else 0.0,
        "median_inside_depth": float(np.median(depths[inside])) if inside.any() else 0.0,
        "culprits": culprits,
    }


def path_kinematics(
    env,
    positions: np.ndarray,
    rotations: np.ndarray,
    pair: GripperPair,
    arm: str = "right",
    tolerance: float = REACH_TOLERANCE,
    stride: int = 1,
) -> dict:
    """Can the arm hold every pose of a transported path, solving down the path?

    The kinematic half of the same question :func:`path_clearance` asks
    geometrically, and it runs **after** it for a reason: an isolated inverse
    kinematics solve costs hundreds of milliseconds because it starts from the
    rest configuration and has to walk all the way to the answer, while a solve
    seeded from the previous waypoint's answer starts a few millimetres away and
    converges almost immediately. Down a 200-pose trajectory that is the
    difference between a minute and a second or two per candidate.

    Seeding from the previous solution also makes this the *right* question.
    ``reachable_fraction`` is a property of the whole path rather than of any
    pose -- it answers "can the arm move **between** these poses" -- and the
    replay itself solves the same way, so this is the same computation the
    execution will perform.

    **It is blind to collision**, deliberately and unavoidably: ``solve_ik`` is
    joint angles and a Jacobian with no collision model, which is why a pose
    inside a shelf solves happily to 3 mm (7.35). That is what
    :func:`path_clearance` is for, and why both run.

    Args:
        positions: ``(m, 3)`` fingertip positions.
        rotations: ``(m, 3, 3)`` grasp-convention orientations.
        stride: Take every ``stride``-th waypoint. The path is smooth and
            consecutive poses are about 5 mm apart, so a stride of 2 or 4 costs
            little fidelity and the corresponding fraction of the time.

    Returns:
        ``reachable_fraction``, ``unreachable`` (count), ``tested``, and
        ``worst_residual`` in metres.
    """
    from tpgpt.grasp.grasps import alignment_rotation
    from tpgpt.sim.kinematics import solve_ik

    positions = np.atleast_2d(np.asarray(positions, dtype=float))[::stride]
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)[::stride]
    # The path is a fingertip path in the grasp convention; IK aims this hand's
    # ``grip_site``, which is a different frame by the measured, per-hand
    # ``alignment_rotation`` -- 0.2 degrees on a Robotiq, 180 on a Panda -- and
    # sits ``contact_offset`` behind the fingertip point. Both conversions are
    # the ones the replay makes; omitting either was 7.34.
    align = alignment_rotation(pair)
    offset = contact_offset(pair)

    seed, unreachable, worst = None, 0, 0.0
    for p, R in zip(positions, rotations):
        wrist = R @ align
        target = p - wrist @ offset
        result = solve_ik(
            env, target, wrist, arm=arm, seed_qpos=seed,
            position_tolerance=tolerance,
        )
        if result.reachable:
            seed = result.qpos
        else:
            unreachable += 1
        worst = max(worst, float(result.position_error))
    tested = int(len(positions))
    return {
        "tested": tested,
        "unreachable": int(unreachable),
        "reachable_fraction": float(1.0 - unreachable / tested) if tested else 0.0,
        "worst_residual": worst,
    }


def by_centre_offset(
    grasps: list[Grasp6D],
    indices: np.ndarray,
    centre_of_mass: np.ndarray,
    pair: GripperPair | None = None,
    max_offset: float = MAX_CENTRE_OFFSET,
) -> np.ndarray:
    """Drop grips taken too far out from an object's centre of mass.

    A thin wrapper on :func:`offset_from_centre`, which carries the measurement
    this threshold rests on and the two cells that contradict it. Read that
    first: this is a **loose** criterion and it is meant to run late, on
    candidates that are already admissible, as a tie-break between them rather
    than as a gate.

    Args:
        centre_of_mass: World position of the object's centre of mass. In
            simulation that is exact; from perception alone the best available
            proxy is the fitted box's centre, which is biased by the cloud being
            one-sided -- so a real deployment would apply this with a wider
            margin, not the same one.
        pair: Measure from where *this hand* would hold the object rather than
            from the grasp's own fingertip point. They differ by
            ``contact_offset``, up to 60 mm across the registry.
    """
    return _keep(
        indices,
        [offset_from_centre(grasps[i], centre_of_mass, pair) <= max_offset
         for i in indices],
    )


def offset_from_centre(grasp: Grasp6D, centre_of_mass: np.ndarray, pair=None) -> float:
    """Horizontal distance from an object's centre of mass to where it is gripped.

    **Measured, not asserted** (`FINDINGS.md` 8n, Experiment P). Twenty grasps
    across five hand/object pairs, each pair tried at four different poses: the
    eleven that carried the object past 200 mm sit a median **6.3 mm** from the
    centre of mass, against **19.1 mm** for the nine that did not, and every
    grasp under 10 mm carried bar one.

    **And it is not decisive on its own**, which is why this returns a number
    rather than a verdict. The same study found ``panda/cereal`` *failing* at
    6.2 mm and carrying 418 mm at 23.4 mm. An off-centre grip makes the object
    rotate in the jaws under its own weight -- the moment is the weight times
    this lever arm -- but whether it slips also depends on the contact friction,
    the finger area and how much of the object is between the jaws, none of
    which this sees. So it is a tie-break and a diagnostic, not a gate.

    Only the horizontal component counts: gravity acts vertically, so a grip
    taken higher or lower on the object changes nothing about the moment.
    """
    tcp = grasp.tcp_position() if pair is None else _held_point(grasp, pair)
    return float(np.linalg.norm((np.asarray(tcp, dtype=float) - np.asarray(
        centre_of_mass, dtype=float))[:2]))


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
    support_normal: np.ndarray | None = (0.0, 0.0, 1.0),
    check_place_approach: bool = True,
    centre_of_mass: np.ndarray | None = None,
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
        support_normal: Outward normal of the surface the object stands on,
            driving :func:`by_support_approach`. ``None`` switches that stage
            off. This is the **scene-derived** replacement for
            ``reference_approach``: it rejects a hand rising through the table
            rather than a hand unlike the demonstration, so it constrains the
            same pathology without tying the target scene to one recording.
        check_place_approach: Run :func:`by_place_approach`, which needs ``env``
            and ``place_pose``. Off only to isolate what it is worth.
        centre_of_mass: World centre of mass of the object being picked. Given,
            :func:`by_centre_offset` runs as a late, loose stage; omitted, it
            does not run at all. Late because it is a tie-break among grasps
            that are already admissible, not a reason to reject one outright.
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
    if support_normal is not None:
        indices = stage(
            "support",
            f"approaches within {SUPPORT_APPROACH_MAX_DEG:.0f} deg of straight "
            "down, so the hand does not come up through the support",
            by_support_approach(grasps, indices, support_normal), indices,
        )
    if check_place_approach and env is not None and place_pose is not None:
        # Where the fingertips must be at release. ``place_pose`` is a
        # ``grip_site`` pose -- what IK is aimed at -- and the fingertip point
        # sits ``contact_offset`` in front of it along this hand's own axes, so
        # the corridor is anchored where the hand actually ends up rather than
        # where its wrist does.
        release = np.array([
            np.asarray(place_pose[0], dtype=float)
            + grasp_to_eef_pose(grasps[int(i)], pair)[1] @ contact_offset(pair)
            for i in indices
        ]) if len(indices) else np.empty((0, 3))
        survivors, blocked = by_place_approach(
            grasps, indices, env, pair, release,
            exclude=tuple(t for t in (target_name,) if t),
        )
        if blocked:
            funnel.flags["place_approach_blocked_by"] = blocked
        indices = stage(
            "place zone",
            "the corridor the hand sweeps into the destination is clear of the "
            "shelf's own geometry",
            survivors, indices,
        )
    indices = stage(
        "on target", f"the held object lands within {MAX_TARGET_DISTANCE * 100:.0f} cm of the cloud",
        by_target(grasps, indices, target_points, pair), indices,
    )
    # Recorded before the stage runs, because the stage may fall back and
    # restore exactly the candidates whose width could not be measured. A cell
    # that ends up executing one of those has a *perception* fault, not a
    # grasping one, and the report must be able to tell them apart.
    verdicts = jaw_width_verdicts(grasps, indices, target_points, pair)
    tally = {v: sum(1 for x in verdicts.values() if x == v)
             for v in ("fits", "too_wide", "unverified")}
    funnel.flags["jaw_width"] = tally
    if tally["unverified"]:
        funnel.flags["jaw_width_unverified"] = tally["unverified"]
    indices = stage(
        "jaw width", "the object fits between the jaws at the grasp height",
        by_jaw_width(grasps, indices, target_points, pair), indices,
    )
    if tally["fits"] == 0 and tally["unverified"]:
        # Nothing was verifiable, so whatever survives here was chosen without a
        # width check. Named for the cause, which is the cloud.
        funnel.flags["cloud_too_sparse_for_jaw_width"] = True
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

    if centre_of_mass is not None:
        indices = stage(
            "centred",
            f"the grip is within {MAX_CENTRE_OFFSET * 1000:.0f} mm of the "
            "object's centre of mass, horizontally",
            by_centre_offset(grasps, indices, centre_of_mass, pair), indices,
        )
    indices = stage(
        "distinct", f"duplicates within {DUPLICATE_POSITION * 100:.0f} cm thinned out",
        suppress_duplicates(grasps, indices, key=rank), indices,
    )

    funnel.survivors = np.array(sorted(indices, key=rank), dtype=int)
    funnel.flags["ranked_by"] = "discriminator score"
    return funnel
