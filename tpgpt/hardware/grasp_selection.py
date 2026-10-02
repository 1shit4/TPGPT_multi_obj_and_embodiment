"""Choosing the grasp for a real object from GraspGen-X candidates.

This is ``tpgpt.experiments.pipeline.run``'s grasp choice -- ``filter_grasps``
followed by ``_choose_grasp`` -- on a measured cloud, a measured shelf and the
real arm. Three steps:

1. **Generate.** GraspGen-X samples 6-DoF grasps for the Franka Hand
   (``franka_panda``, the same hand: identical pad geometry and 0.1034 m
   fingertip depth, see :mod:`tpgpt.hardware.hand`), each with its
   discriminator score.
2. **Screen** (:func:`screen_candidates`), cheapest first, every rejection
   named. Two kinds of stage, and the difference is deliberate:

   * **physical stages, hard**: ``from_below`` (the approach leans more than
     ``max_tilt_deg`` from straight down -- ``filters.by_support_approach``'s
     rule); the jaws, :func:`tpgpt.hardware.hand.check_jaws`, after
     :func:`refine_grasp` (``too_wide``, ``collides``, ``empty``, ``shallow``);
     ``hits_support``; ``duplicate``. These were built on real one-camera
     clouds, where GraspGen-X stops short of the object (its training meshes
     are complete): measured on the real banana and a rendered partial cup,
     the pads engaged a median 2-3 mm before refinement against 11 mm on a
     complete box. The simulation's ``by_jaw_width`` assumes a complete scan.
   * **the simulation's funnel stages, soft** -- the functions of
     :mod:`tpgpt.grasp.filters` themselves, and with its rule that a stage which
     would empty the set passes it through and says so (``FilterFunnel``):
     ``visibility`` (``by_visibility``), ``on_target`` (``by_target``),
     ``grasp_collision`` (``by_collision`` -- the funnel's ``collision`` stage,
     renamed so it is not confused with the whole-path verdict -- against the
     shelf, at the grasp and along its approach), ``place_zone``
     (``by_place_approach``, against the shelf model) and ``centred``
     (``by_centre_offset``, on the cloud's box centre with a wider margin, as
     that function's own docstring prescribes for perception).

3. **Choose** (:func:`select_and_plan`), ``_choose_grasp``'s rule: walk the
   survivors in score order (up to ``max_plans``, 100 as in simulation), plan
   each through :func:`tpgpt.hardware.planner.plan`, and take the **first whose
   whole transported path is admissible** -- no collision at any waypoint, and
   (``kinematic_criterion="strict"``, the default) every waypoint reachable in
   position with the pick and place reachable with their orientation, or
   (``"sim"``) at least 60 % of it within the arm's reach
   (:func:`tpgpt.hardware.path_check.path_feasibility`). Ranking is the
   discriminator score and nothing else.

Where this still differs from the simulation, and why:

* **The fallback.** When nothing is admissible the simulation executes the
  candidate whose first fault comes latest. On the robot a fault is a collision
  with the shelf, so only a candidate that failed on *kinematics alone* may be
  executed that way (``allow_kinematic_fallback``); a colliding one never is.
  The scripted top-down grasp (:func:`tpgpt.hardware.planner.default_grasp`),
  a grasp the generator never proposed, is off by default and, when enabled,
  passes the same path check.
* **Extra acceptance gates** kept from the earlier hardware selection and not
  in the simulation: ``min det(J) >= 0.1``, the transported path within 10 mm
  and 15 degrees of the grasp, and the hand's lowest corner above the table and
  destination planes (:func:`_verify`). ``docs/hardware/FAITHFULNESS_AUDIT.md``
  item A4.

No ROS here. The node supplies the candidates (or this module fetches them over
ZMQ with :func:`request_candidates`), the shelf world and the joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from tpgpt.hardware import hand
from tpgpt.hardware.planner import Plan, PlanningError, TargetScene, default_grasp, plan
from tpgpt.sim.keypoints import GraspFrame

#: GraspGen-X gripper model of the FR3's Franka Hand.
GRASPGEN_GRIPPER = "franka_panda"


@dataclass
class SelectionSettings:
    """Thresholds for screening and for accepting a plan. SI units, degrees."""

    #: Largest angle between the approach and straight down. 85 rather than 90
    #: because the pads hang below the TCP (``tpgpt.grasp.filters.
    #: SUPPORT_APPROACH_MAX_DEG``). Lower it to prefer top-down grasps.
    max_tilt_deg: float = 85.0
    #: Lowest hand corner must clear the support plane by this at the grasp.
    min_support_clearance: float = 0.003
    #: Near-duplicate suppression.
    duplicate_position: float = 0.01
    duplicate_angle_deg: float = 15.0
    #: Plans attempted, best score first, before giving up
    #: (``pipeline.run``'s ``max_candidates``).
    max_plans: int = 100
    #: Plan acceptance.
    min_determinant: float = 0.1
    max_aim_grasp: float = 0.01
    max_orientation_error_deg: float = 15.0
    #: The hand's lowest corner along the path may dip this far below a surface
    #: (negative = allowed penetration). Set-down contact is normal; digging in
    #: is not.
    min_path_clearance: float = -0.002
    #: Try :func:`default_grasp` when no generated candidate passes. Off: the
    #: simulation has no scripted fallback, and the X6 brief forbids overriding
    #: the generator's choice.
    fallback_top_down: bool = False
    #: :func:`refine_grasp`: slide each candidate deeper and centre it.
    refine: bool = True
    refine_margin: float = 0.003
    refine_max_depth: float = 0.025
    refine_max_centre: float = 0.010
    #: Base-frame shift (m) added to the TCP of every grasp just before it is
    #: planned, generated or top-down alike. Screening still uses the unshifted grasp.
    grasp_offset: tuple = (0.0, 0.0, 0.0)
    #: The simulation's funnel stages (``tpgpt.grasp.filters``); each needs its
    #: input and is skipped without it.
    check_visibility: bool = True
    check_on_target: bool = True
    check_collision: bool = True
    check_place_zone: bool = True
    check_centred: bool = True
    #: ``by_centre_offset``'s 15 mm, widened by 10 mm because the centre comes
    #: from a one-sided cloud, which under-reports an object's extent by 11-21 mm
    #: (``ROBOTICS_NOTES`` 7.36) and so biases its box centre toward the camera.
    max_centre_offset: float = 0.025
    #: The box the executor will refuse to leave (its ``workspace_min`` /
    #: ``workspace_max``), so a path that leaves it is rejected here rather than
    #: at the first executed step. ``None`` on either: not checked.
    workspace_min: tuple | None = None
    workspace_max: tuple | None = None
    #: The whole-path check (``tpgpt.hardware.path_check``).
    check_path: bool = True
    #: ``"strict"``: every waypoint reachable in position, the pick and place
    #: reachable with their orientation (user's choice, 2026-09-22). ``"sim"``:
    #: ``min_reachable_fraction`` of the waypoints reachable as planned.
    kinematic_criterion: str = "strict"
    #: Whether a blocked move from the arm's current pose onto the path rejects the
    #: grasp. False (user, 2026-09-29): the move is still checked and reported, and
    #: the planner asks for a re-home instead -- the parked pose is not a property of
    #: the grasp, and on 2026-09-26 it refused every grasp to top_left.
    approach_rejects: bool = False
    #: Refuse a path that puts joints 1-6 inside CRISP's joint-limit push-back zone
    #: (``path_check.JOINT_MARGIN``, rad; None = off). See path_check.JOINT_MARGIN.
    joint_margin: float | None = None
    #: Refuse a path that brings the wrist this close to full stretch (m; None = off), where
    #: the arm cannot be pulled (``path_check.REACH_MARGIN``).
    reach_margin: float | None = None
    #: Once clear of the placed object by this much in the retreat, the hand must stay that
    #: far from it (m; None = off; ``path_check.path_feasibility``'s ``retreat_clearance``).
    retreat_clearance: float | None = None
    min_reachable_fraction: float = 0.60
    path_stride: int = 4
    #: With nothing admissible, execute the best candidate that failed on
    #: kinematics alone (the simulation's least-bad rule, restricted to the ones
    #: that do not collide). Off: under the strict criterion it would execute a
    #: plan the criterion refused.
    allow_kinematic_fallback: bool = False


@dataclass
class Candidate:
    """One grasp and everything decided about it."""

    index: int                      # position in the generator's output, -1 for top-down
    score: float
    grasp: GraspFrame
    tilt_deg: float
    jaw: hand.JawCheck | None = None
    support_clearance: float = float("nan")
    rejected: str | None = None     # reason, or None while still admissible
    plan_metrics: dict | None = None
    refined_depth: float = 0.0      # m moved along the approach by refine_grasp
    refined_centre: float = 0.0     # m moved along the closing axis
    path_check: dict | None = None  # tpgpt.hardware.path_check result, summarised

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "score": self.score,
            "tcp": self.grasp.tcp.tolist(),
            "approach": self.grasp.approach.tolist(),
            "closing": self.grasp.closing.tolist(),
            "tilt_deg": self.tilt_deg,
            "jaw_verdict": self.jaw.verdict if self.jaw else None,
            "jaw_width_m": self.jaw.width if self.jaw else None,
            "jaw_points": self.jaw.n_between if self.jaw else None,
            "jaw_engagement_m": self.jaw.engagement if self.jaw else None,
            "support_clearance_m": self.support_clearance,
            "refined_depth_m": self.refined_depth,
            "refined_centre_m": self.refined_centre,
            "rejected": self.rejected,
            "path_check": self.path_check,
        }


@dataclass
class Selection:
    """The chosen plan, and the record of how it was chosen."""

    plan: Plan
    chosen: Candidate
    candidates: list[Candidate]
    source: str                     # "graspgenx", "top_down" or "side"
    notes: list[str] = field(default_factory=list)
    #: The screening funnel: ``[name, entered, survived, fell_back]`` per stage.
    stages: list = field(default_factory=list)
    #: Candidates planned and path-checked before one was chosen (1 = the first).
    rank_examined: int | None = None
    #: Nothing was admissible and the least-bad kinematic failure was taken.
    fell_back: bool = False
    #: The shelf world the paths were checked against, for the record.
    world: dict | None = None

    def rejection_counts(self) -> dict:
        return rejection_counts(self.candidates)

    def describe(self) -> str:
        c = self.chosen
        if c.jaw is None:
            side = self.plan.metadata.get("side_grasp", {})
            return (f"grasp from {self.source}: palm point {np.round(c.grasp.tcp, 3).tolist()}, "
                    f"approach {np.round(c.grasp.approach, 2).tolist()}, object radius "
                    f"{side.get('radius_m', float('nan')) * 1000:.0f} mm, "
                    f"{side.get('grasp_height_above_support_m', float('nan')) * 1000:.0f} mm "
                    "above the support")
        head = (f"grasp from {self.source}: score {c.score:.3f}, tilt {c.tilt_deg:.0f} deg from "
                f"vertical, jaw width {c.jaw.width * 1000:.0f} mm ({c.jaw.n_between} pts between "
                f"pads, {c.jaw.engagement * 1000:.0f} mm of pad engaged), refined "
                f"{c.refined_depth * 1000:.0f} mm deeper / {c.refined_centre * 1000:+.0f} mm "
                f"to centre")
        tally = ", ".join(f"{k} {v}" for k, v in sorted(self.rejection_counts().items()))
        return head + (f"; {len(self.candidates)} candidates: {tally}" if self.candidates else "")


def rejection_counts(candidates: list[Candidate]) -> dict:
    """How many candidates ended with each outcome, keyed by the reason's tag."""
    counts: dict[str, int] = {}
    for c in candidates:
        key = (c.rejected or "admissible").split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


# ------------------------------------------------------------------ generation
def pose_to_grasp_frame(pose: np.ndarray, tcp_depth: float = hand.TCP_OFFSET) -> GraspFrame:
    """A GraspGen-X pose (gripper base, ``+Z`` approach, ``+X`` closing) as a
    :class:`GraspFrame` at the fingertips."""
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    approach = pose[:3, 2]
    return GraspFrame(tcp=pose[:3, 3] + approach * float(tcp_depth), approach=approach,
                      closing=pose[:3, 0])


def request_candidates(
    points: np.ndarray,
    host: str = "127.0.0.1",
    port: int = 5556,
    gripper: str = GRASPGEN_GRIPPER,
    num_grasps: int = 200,
    topk: int = 100,
    planner: str = "diffusion",
    timeout_ms: int = 60_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Ask a running GraspGen-X server for grasps on ``points`` (base frame).

    Returns ``(poses (K, 4, 4), scores (K,))`` in the cloud's frame. Raises
    :class:`tpgpt.grasp.client.GraspGenUnavailable` if the server does not answer.
    """
    from tpgpt.grasp.client import GraspGenClient

    client = GraspGenClient(host=host, port=port, timeout_ms=timeout_ms)
    try:
        return client.infer(np.asarray(points, dtype=np.float32), gripper_name=gripper,
                            num_grasps=num_grasps, topk_num_grasps=topk, planner=planner)
    finally:
        client.close()


# ------------------------------------------------------------------- refinement
def refine_grasp(
    points: np.ndarray,
    grasp: GraspFrame,
    support_height: float,
    settings: SelectionSettings | None = None,
) -> tuple[GraspFrame, float, float]:
    """Slide a grasp along its approach and centre the object between the pads.

    It moves the hand *in* to poses where :func:`tpgpt.hardware.hand.check_jaws`
    still says ``fits`` and the hand clears the support by
    ``min_support_clearance``. If the generator's own pose does not clear the
    support, it also tries moving *out*, up to ``refine_max_depth``: on a low
    object the generator aims at the object's middle, which puts the fingertips
    through the table, while the same grasp a centimetre higher holds it. On a
    YCB banana (37 mm tall) 44 of 100 candidates were refused for this, several
    by a single millimetre. The orientation is untouched.

    Returns:
        ``(grasp, depth, centre)`` -- the refined grasp and how far it moved
        along the approach and the closing axis, in metres. The input grasp and
        zeros when nothing better fits.
    """
    s = settings or SelectionSettings()
    points = np.atleast_2d(np.asarray(points, dtype=float))
    R = grasp.rotation
    lo, hi = hand.pad_box()
    local = hand.to_grasp_frame(points, grasp.tcp, R)
    # The column the pads sweep, at any depth: where the object starts along the approach.
    column = (np.abs(local[:, 1]) <= hi[1]) & (np.abs(local[:, 0]) <= hi[0])
    if int(column.sum()) < hand.MIN_POINTS_BETWEEN:
        return grasp, 0.0, 0.0
    entry = float(local[column, 2].min())
    want = float(np.clip(entry - (lo[2] - s.refine_margin), 0.0, s.refine_max_depth))

    def fits(tcp):
        return (hand.check_jaws(points, tcp, R).ok
                and hand.lowest_point(tcp, R) - float(support_height) >= s.min_support_clearance)

    clears = hand.lowest_point(grasp.tcp, R) - float(support_height) >= s.min_support_clearance
    # Out of the object (negative depth) only when the pose as generated does not
    # clear the support; in that case a shallower grasp is the only one available.
    span = np.arange(want, -1e-9, -0.002) if clears else \
        np.r_[np.arange(want, -1e-9, -0.002), -np.arange(0.002, s.refine_max_depth + 1e-9, 0.002)]
    for depth in span:
        tcp = grasp.tcp + grasp.approach * depth
        loc = hand.to_grasp_frame(points, tcp, R)
        between = np.all((loc >= lo) & (loc <= hi), axis=1)
        centre = 0.0
        if int(between.sum()) >= 2:
            xs = loc[between, 0]
            centre = float(np.clip((xs.max() + xs.min()) / 2.0, -s.refine_max_centre,
                                   s.refine_max_centre))
        for shift in ((centre, 0.0) if abs(centre) > 1e-4 else (0.0,)):
            moved = tcp + grasp.closing * shift
            if (depth > 0.0 or shift != 0.0) and fits(moved):
                return GraspFrame(tcp=moved, approach=grasp.approach,
                                  closing=grasp.closing), float(depth), float(shift)
    return grasp, 0.0, 0.0


# ------------------------------------------------------------------- screening
def _tilt_deg(approach: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(-float(approach[2]), -1.0, 1.0))))


@dataclass
class ScreenContext:
    """What the simulation's funnel stages need beyond the cloud. Each is optional;
    a stage whose input is missing is skipped, and the funnel says so."""

    #: Camera position in the base frame (``by_visibility``).
    camera_position: np.ndarray | None = None
    #: The shelf world (``by_collision``, ``by_place_approach``).
    world: object | None = None
    #: Where the object goes: ``(x, y)`` and the surface height there.
    destination: np.ndarray | None = None
    destination_height: float | None = None
    #: The demonstration's rotation of the hand from closing to opening
    #: (``tpgpt.sim.keypoints.carry_transform``), for the place zone.
    carry_rotation: np.ndarray | None = None


class Screened(list):
    """The screened candidates (best score first), plus the funnel's stage tallies.

    ``stages`` rows are ``[name, entered, survived, fell_back]`` -- the
    simulation's ``FilterFunnel.flags["stages"]`` format.
    """

    stages: list

    def __init__(self, items=(), stages=None):
        super().__init__(items)
        self.stages = list(stages or [])


def horizontal_centre(points: np.ndarray) -> np.ndarray:
    """Centre of the cloud's horizontal bounding box, along its own principal axes.

    The perception proxy ``filters.by_centre_offset`` names for a centre of
    mass: the fitted box's centre. Horizontal only, as that function measures.
    """
    xy = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    mean = xy.mean(axis=0)
    _, _, axes = np.linalg.svd(xy - mean, full_matrices=False)
    local = (xy - mean) @ axes.T
    mid = (local.min(axis=0) + local.max(axis=0)) / 2.0
    centre_xy = mean + mid @ axes
    z = float(np.median(np.asarray(points)[:, 2]))
    return np.array([centre_xy[0], centre_xy[1], z])


def _as_grasp6d(c: Candidate, tcp_depth: float):
    """A candidate as the simulation's ``Grasp6D`` (a pose at the gripper base)."""
    from tpgpt.grasp.grasps import Grasp6D

    pose = np.eye(4)
    pose[:3, :3] = c.grasp.rotation
    pose[:3, 3] = c.grasp.tcp - c.grasp.approach * float(tcp_depth)
    return Grasp6D(pose=pose, score=float(c.score) if np.isfinite(c.score) else 0.0,
                   gripper=GRASPGEN_GRIPPER, width=hand.MAX_OPENING)


def _soft_stage(stages: list, alive: list, name: str, checks: str, keep) -> list:
    """One of the simulation's funnel stages, with its rule: a stage that would
    reject every remaining candidate passes them all through and is flagged."""
    kept = set(int(i) for i in keep)
    survivors = [c for k, c in enumerate(alive) if k in kept]
    if alive and not survivors:
        stages.append([name, len(alive), len(alive), True])
        return alive
    for k, c in enumerate(alive):
        if k not in kept:
            c.rejected = f"{name}: {checks}"
    stages.append([name, len(alive), len(survivors), False])
    return survivors


def _release_positions(alive: list, points: np.ndarray, support: float, ctx: ScreenContext):
    """Where each candidate's fingertip point is when the object is set down.

    The object is rigid in the hand from the close, so the grasp point keeps
    its offset from the object's base, turned by the demonstration's carry
    rotation, and the base lands on the destination. (The simulation uses one
    provisional pose for all candidates -- the hand at the object's top above
    the slot; this is the same question asked per candidate.)
    """
    base = horizontal_centre(points)
    base[2] = float(support)
    R = (np.eye(3) if ctx.carry_rotation is None
         else np.asarray(ctx.carry_rotation, dtype=float).reshape(3, 3))
    dest = np.array([float(ctx.destination[0]), float(ctx.destination[1]),
                     float(ctx.destination_height)])
    return np.array([dest + R @ (c.grasp.tcp - base) for c in alive])


def screen_candidates(
    points: np.ndarray,
    poses: np.ndarray,
    scores: np.ndarray,
    support_height: float,
    settings: SelectionSettings | None = None,
    tcp_depth: float = hand.TCP_OFFSET,
    context: ScreenContext | None = None,
) -> Screened:
    """Judge every generated grasp before any is planned.

    Returns every candidate, best score first; admissible ones have
    ``rejected is None``. ``.stages`` tallies each stage. Stage order: the
    simulation's funnel order (visibility, support, place zone, on target, jaw
    width, collision, centred, distinct), with the hardware's physical checks
    standing where ``by_support_approach`` and ``by_jaw_width`` stand, and the
    place zone after the jaws because it is the costliest stage here.
    """
    s = settings or SelectionSettings()
    ctx = context or ScreenContext()
    points = np.atleast_2d(np.asarray(points, dtype=float))
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    candidates = [Candidate(index=int(i), score=float(scores[i]),
                            grasp=pose_to_grasp_frame(poses[i], tcp_depth),
                            tilt_deg=0.0) for i in order]
    stages: list = []

    # --- support (hard): nothing approaches from below the table -------------
    alive = []
    for c in candidates:
        c.tilt_deg = _tilt_deg(c.grasp.approach)
        if c.tilt_deg > s.max_tilt_deg:
            c.rejected = f"from_below: approach {c.tilt_deg:.0f} deg from vertical"
        else:
            alive.append(c)
    stages.append(["from_below", len(candidates), len(alive), False])

    need_filters = alive and (s.check_visibility or s.check_on_target or s.check_collision
                              or s.check_place_zone or s.check_centred)
    if need_filters:
        from tpgpt.grasp import filters
        from tpgpt.grasp.grippers import resolve_pair

        pair = resolve_pair("panda")

    def g6(group):
        return [_as_grasp6d(c, tcp_depth) for c in group], np.arange(len(group))

    # --- visibility (soft) -----------------------------------------------------
    if alive and s.check_visibility and ctx.camera_position is not None:
        grasps, idx = g6(alive)
        keep = filters.by_visibility(grasps, idx, {"camera": np.asarray(ctx.camera_position)})
        alive = _soft_stage(stages, alive, "visibility",
                            f"approaches more than {filters.MAX_BACKWARD_DEG:.0f} deg away from "
                            "the camera's view", keep)

    # --- on target (soft) ------------------------------------------------------
    if alive and s.check_on_target:
        grasps, idx = g6(alive)
        keep = filters.by_target(grasps, idx, points, pair)
        alive = _soft_stage(stages, alive, "on_target",
                            f"the held point is more than {filters.MAX_TARGET_DISTANCE * 100:.0f} "
                            "cm from the cloud", keep)

    # --- the jaws and the table (hard) ------------------------------------------
    entered, passed = len(alive), []
    for c in alive:
        if s.refine:
            c.grasp, c.refined_depth, c.refined_centre = refine_grasp(points, c.grasp,
                                                                      support_height, s)
        c.jaw = hand.check_jaws(points, c.grasp.tcp, c.grasp.rotation)
        if not c.jaw.ok:
            detail = f" ({', '.join(c.jaw.blocked_by)})" if c.jaw.blocked_by else ""
            c.rejected = (f"{c.jaw.verdict}: width {c.jaw.width * 1000:.0f} mm, "
                          f"{c.jaw.engagement * 1000:.0f} mm of pad engaged{detail}")
            continue
        c.support_clearance = (hand.lowest_point(c.grasp.tcp, c.grasp.rotation)
                               - float(support_height))
        if c.support_clearance < s.min_support_clearance:
            c.rejected = f"hits_support: hand {c.support_clearance * 1000:.0f} mm above the table"
            continue
        passed.append(c)
    alive = passed
    stages.append(["jaws", entered, len(alive), False])

    world = ctx.world
    # --- collision with the shelf, at the grasp and along its approach (soft) ----
    if alive and s.check_collision and world is not None:
        scene = np.vstack([world.surface()[0], world.cloud]) if world.model is not None \
            else world.cloud
        if len(scene):
            grasps, idx = g6(alive)
            keep = filters.by_collision(grasps, idx, scene, pair)
            alive = _soft_stage(stages, alive, "grasp_collision",
                                "the hand or its approach meets the shelf", keep)

    # --- the place zone, against the shelf model (soft) --------------------------
    if (alive and s.check_place_zone and world is not None and world.model is not None
            and ctx.destination is not None and ctx.destination_height is not None):
        grasps, idx = g6(alive)
        keep, blocked = filters.by_place_approach(
            grasps, idx, None, pair, _release_positions(alive, points, support_height, ctx),
            obstacles=list(world.model.boxes), carry_rotation=ctx.carry_rotation)
        tally = ", ".join(f"{k} {v}" for k, v in sorted(blocked.items()))
        alive = _soft_stage(stages, alive, "place_zone",
                            f"the hand arrives at the release through the shelf ({tally})", keep)

    # --- centred (soft) ---------------------------------------------------------
    if alive and s.check_centred:
        grasps, idx = g6(alive)
        keep = filters.by_centre_offset(grasps, idx, horizontal_centre(points), pair,
                                        max_offset=s.max_centre_offset)
        alive = _soft_stage(stages, alive, "centred",
                            f"gripped more than {s.max_centre_offset * 1000:.0f} mm from the "
                            "cloud's box centre", keep)

    # --- distinct (hard) --------------------------------------------------------
    entered, kept = len(alive), []
    cos_dup = np.cos(np.radians(s.duplicate_angle_deg))
    for c in alive:
        grasp = c.grasp
        if any(np.linalg.norm(grasp.tcp - k.grasp.tcp) < s.duplicate_position
               and float(grasp.approach @ k.grasp.approach) > cos_dup
               and abs(float(grasp.closing @ k.grasp.closing)) > cos_dup for k in kept):
            c.rejected = "duplicate"
            continue
        kept.append(c)
    stages.append(["distinct", entered, len(kept), False])
    return Screened(candidates, stages)


# ------------------------------------------------------------ plan and verify
def _verify(result: Plan, s: SelectionSettings) -> str | None:
    m = result.metrics
    if m["min_det"] < s.min_determinant:
        return f"map_folds: min det(J) {m['min_det']:.2f}"
    if m["aim_grasp_m"] > s.max_aim_grasp:
        return f"misses_grasp: transported path {m['aim_grasp_m'] * 1000:.0f} mm from the grasp"
    if m["orientation_error_grasp_deg"] > s.max_orientation_error_deg:
        return (f"wrong_orientation: {m['orientation_error_grasp_deg']:.0f} deg from the grasp "
                "at closing")
    if s.workspace_min is not None and s.workspace_max is not None:
        lo = np.asarray(s.workspace_min, dtype=float)
        hi = np.asarray(s.workspace_max, dtype=float)
        p = np.asarray(result.labels.positions, dtype=float)
        outside = np.any((p < lo) | (p > hi), axis=1)
        if outside.any():
            k = int(np.flatnonzero(outside)[0])
            return (f"outside_workspace: label {k} {np.round(p[k], 3).tolist()} is outside the "
                    f"executor's box {lo.tolist()} .. {hi.tolist()} "
                    f"({int(outside.sum())} of {len(p)} labels)")
    pick, place = m["hand_clearance_pick_m"], m["hand_clearance_place_m"]
    if min(pick, place) < s.min_path_clearance:
        return (f"path_hits_surface: hand {pick * 1000:.0f} mm above the table on the pick side, "
                f"{place * 1000:.0f} mm above the destination on the place side")
    return None


def _try_plan(source, scene: TargetScene, candidate: Candidate, s: SelectionSettings,
              plan_kwargs: dict) -> Plan | None:
    offset = np.asarray(s.grasp_offset, dtype=float).reshape(3)
    if np.any(offset):
        candidate.grasp = replace(candidate.grasp, tcp=candidate.grasp.tcp + offset)
    try:
        result = plan(source, replace(scene, grasp=candidate.grasp),
                      min_determinant=s.min_determinant, **plan_kwargs)
    except PlanningError as exc:
        candidate.rejected = f"plan_refused: {exc}"
        return None
    candidate.plan_metrics = result.metrics
    reason = _verify(result, s)
    if reason is not None:
        candidate.rejected = reason
        return None
    return result


@dataclass
class PathContext:
    """What the whole-path check needs: the world, the hand and where the arm is now."""

    world: object | None = None
    end_effector: object | None = None
    #: Measured joint angles and ``fr3_hand_tcp`` position at planning time.
    start_q: np.ndarray | None = None
    start_tcp: np.ndarray | None = None


def _path_verdict(result: Plan, candidate: Candidate, s: SelectionSettings,
                  pc: PathContext) -> tuple[str | None, bool]:
    """Run the whole-path check on a plan. Returns ``(reason, kinematic_only)``."""
    import time

    from tpgpt.hardware.end_effector import FRANKA_HAND
    from tpgpt.hardware.path_check import check_plan, verdict
    from tpgpt.hardware.shelf import ShelfWorld

    world = pc.world if pc.world is not None else ShelfWorld.build(None)
    t0 = time.perf_counter()
    check = check_plan(result, world, pc.end_effector or FRANKA_HAND, start_q=pc.start_q,
                       start_tcp=pc.start_tcp, stop_early=True, stride=s.path_stride,
                       min_reachable_fraction=s.min_reachable_fraction,
                       criterion=s.kinematic_criterion, approach_rejects=s.approach_rejects,
                       joint_margin=s.joint_margin, reach_margin=s.reach_margin,
                       retreat_clearance=s.retreat_clearance)
    reason = verdict(check, s.min_reachable_fraction)
    candidate.path_check = {
        "criterion": check["criterion"],
        "reachable_fraction": check["reachable_fraction"],
        "orientation_relaxed": check["orientation_relaxed"],
        "relaxed_in_pick_or_place": check["relaxed_in_pick_or_place"],
        "skipped": check["skipped"],
        "unreachable": check["unreachable"],
        "tested": check["tested"],
        "violations": check["violations"],
        "first_violation": check["first_violation"],
        "faults": dict(list(check["faults"].items())[:6]),
        "approach_in_checked": check["approach_in"] is not None,
        "approach_in": check["approach_in"],
        "decided_early": check["decided_early"],
        "seconds": time.perf_counter() - t0,
        "verdict": reason,
    }
    result.metrics["path_check"] = candidate.path_check
    # kept on the plan, not in the summary (which goes to every candidate's JSON record)
    result.checked_postures = check.get("postures")
    return reason, bool(reason) and not check["violations"]


def select_and_plan(
    source,
    scene: TargetScene,
    poses: np.ndarray | None,
    scores: np.ndarray | None,
    settings: SelectionSettings | None = None,
    tcp_depth: float = hand.TCP_OFFSET,
    unavailable_reason: str | None = None,
    context: ScreenContext | None = None,
    path: PathContext | None = None,
    **plan_kwargs,
) -> Selection:
    """Screen the candidates, then take the first whose whole path is admissible.

    ``pipeline._choose_grasp``'s rule (module docstring): survivors in score
    order, each planned and path-checked; the first with no collision and at
    least ``min_reachable_fraction`` of its path reachable is chosen. With none,
    the first that failed on kinematics alone (``allow_kinematic_fallback``),
    then the scripted top-down grasp if ``fallback_top_down`` -- which passes
    the same checks.

    Args:
        poses, scores: GraspGen-X output in the base frame. ``None`` (server
            unreachable, or a top-down run asked for) goes straight to the
            top-down grasp, if allowed.
        context: Inputs of the screening funnel's stages.
        path: The shelf world, the hand and the measured joint angles for the
            whole-path check. ``None`` checks kinematics only, from home, and
            says so in the notes.
        plan_kwargs: Forwarded to :func:`tpgpt.hardware.planner.plan`.

    Raises:
        PlanningError: nothing executable, with the tally of why. It carries
            ``candidates``, ``fallback`` and ``stages`` for the record.
    """
    s = settings or SelectionSettings()
    ctx = context or ScreenContext()
    pc = path or PathContext()
    points = np.atleast_2d(np.asarray(scene.points, dtype=float))
    support = (float(scene.support_height) if scene.support_height is not None
               else float(np.percentile(points[:, 2], 1.0)))
    notes: list[str] = []
    candidates: list[Candidate] = []
    stages: list = []
    world_record = pc.world.to_dict() if pc.world is not None else None
    if s.check_path and pc.world is None:
        notes.append("no shelf world: the path was checked for kinematics only")
    if s.check_path and pc.start_q is None:
        notes.append("no joint angles: the move onto the path was not checked, IK seeded from home")
    if s.check_path and pc.world is not None and pc.world.model is not None \
            and not pc.world.model.measured:
        notes.append("the shelf model contains ESTIMATED dimensions (shelf.yaml measured: false)")

    def selection(result, chosen, source_name, rank, fell_back=False):
        return Selection(result, chosen, candidates, source_name, notes, stages=stages,
                         rank_examined=rank, fell_back=fell_back, world=world_record)

    fallback = None
    if poses is not None and len(poses):
        candidates = screen_candidates(points, poses, scores, support, s, tcp_depth, ctx)
        stages = candidates.stages
        admissible = [c for c in candidates if c.rejected is None]
        limit = max(int(s.max_plans), 0)
        for rank, c in enumerate(admissible[:limit], start=1):
            result = _try_plan(source, scene, c, s, plan_kwargs)
            if result is None:
                continue
            if not s.check_path:
                return selection(result, c, "graspgenx", rank)
            reason, kinematic_only = _path_verdict(result, c, s, pc)
            if reason is None:
                return selection(result, c, "graspgenx", rank)
            c.rejected = reason
            if kinematic_only and fallback is None:
                fallback = (c, result, rank)
        for c in admissible[limit:]:
            c.rejected = "not_tried: max_plans reached"
        notes.append("no GraspGen-X candidate was admissible along its whole path")
    else:
        notes.append(f"no GraspGen-X candidates ({unavailable_reason or 'none generated'})")

    if fallback is not None and s.allow_kinematic_fallback:
        c, result, rank = fallback
        notes.append(f"fell back to candidate {c.index} (rank {rank}), which clears the shelf "
                     f"but the arm reaches only {c.path_check['reachable_fraction'] * 100:.0f} % "
                     "of its path")
        return selection(result, c, "graspgenx", rank, fell_back=True)

    tally = rejection_counts(candidates)
    top_down = None
    if s.fallback_top_down:
        grasp = default_grasp(points, support, plan_kwargs.get("grasp_depth_below_top", 0.03))
        top_down = Candidate(index=-1, score=float("nan"), grasp=grasp,
                             tilt_deg=_tilt_deg(grasp.approach),
                             jaw=hand.check_jaws(points, grasp.tcp, grasp.rotation))
        result = _try_plan(source, scene, top_down, s,
                           {k: v for k, v in plan_kwargs.items() if k != "grasp_depth_below_top"})
        if result is not None and s.check_path:
            reason, kinematic_only = _path_verdict(result, top_down, s, pc)
            if reason is not None and not (kinematic_only and s.allow_kinematic_fallback):
                top_down.rejected = reason
                result = None
        if result is not None:
            notes.append("used the scripted top-down grasp")
            return selection(result, top_down, "top_down", None)
    error = PlanningError(
        f"no_executable_grasp: {len(candidates)} generated candidates ({tally})"
        + (f", and the top-down grasp was rejected: {top_down.rejected}" if top_down else "")
        + ". " + "; ".join(notes))
    error.candidates, error.fallback, error.stages = candidates, top_down, stages
    raise error


def plan_side_grasp(
    source,
    scene: TargetScene,
    end_effector,
    settings: SelectionSettings | None = None,
    side_settings=None,
    path: PathContext | None = None,
    **plan_kwargs,
) -> Selection:
    """Plan the scripted side grasp for a wrapping hand; no generator, one candidate.

    The grasp comes from :func:`tpgpt.hardware.side_grasp.side_grasp` on the
    cloud, then goes through the same :func:`plan`, the same acceptance checks
    and the same whole-path check as a GraspGen-X candidate, with this hand's
    envelope as the hand. ``grasp_offset`` is applied as for every other grasp.

    Raises:
        PlanningError: the side grasp could not be placed or its plan was refused.
    """
    from tpgpt.hardware.side_grasp import side_grasp

    s = settings or SelectionSettings()
    pc = replace(path or PathContext(), end_effector=end_effector)
    points = np.atleast_2d(np.asarray(scene.points, dtype=float))
    support = (float(scene.support_height) if scene.support_height is not None
               else float(np.percentile(points[:, 2], 1.0)))
    try:
        grasp, info = side_grasp(points, support, end_effector.handedness or "left", side_settings)
    except ValueError as exc:
        error = PlanningError(f"no_executable_grasp: side grasp: {exc}")
        error.candidates, error.fallback, error.stages = [], None, []
        raise error from exc
    candidate = Candidate(index=-1, score=float("nan"), grasp=grasp,
                          tilt_deg=_tilt_deg(grasp.approach))
    candidate.support_clearance = end_effector.lowest_point(grasp.tcp, grasp.rotation) - support
    kwargs = {k: v for k, v in plan_kwargs.items() if k != "grasp_depth_below_top"}
    kwargs["end_effector"] = end_effector
    result = _try_plan(source, scene, candidate, s, kwargs)
    notes = list(info.notes)
    fell_back = False
    if result is not None and s.check_path:
        reason, kinematic_only = _path_verdict(result, candidate, s, pc)
        if reason is not None and kinematic_only and s.allow_kinematic_fallback:
            notes.append(f"the side grasp clears the shelf but the arm reaches only "
                         f"{candidate.path_check['reachable_fraction'] * 100:.0f} % of its path")
            fell_back = True
        elif reason is not None:
            candidate.rejected = reason
            result = None
    if result is None:
        error = PlanningError(f"no_executable_grasp: the side grasp was rejected: "
                              f"{candidate.rejected}")
        error.candidates, error.fallback, error.stages = [], candidate, []
        raise error
    result.metadata["side_grasp"] = info.as_dict()
    return Selection(result, candidate, [], "side", notes, rank_examined=1, fell_back=fell_back,
                     world=pc.world.to_dict() if pc.world is not None else None)


__all__ = [
    "GRASPGEN_GRIPPER",
    "Candidate",
    "PathContext",
    "ScreenContext",
    "Screened",
    "Selection",
    "SelectionSettings",
    "horizontal_centre",
    "plan_side_grasp",
    "pose_to_grasp_frame",
    "refine_grasp",
    "request_candidates",
    "screen_candidates",
    "select_and_plan",
]
