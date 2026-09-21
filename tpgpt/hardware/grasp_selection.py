"""Choosing the grasp for a real object from GraspGen-X candidates.

:func:`tpgpt.hardware.planner.default_grasp` is one scripted pose: straight
down, across the narrowest horizontal direction, a fixed depth below the top.
It cannot pinch a cup by its rim, take a tall bottle from the side, or notice
that its one pose puts a fingertip through the table. This module replaces it
with the generator the simulation pipeline already uses
(:mod:`tpgpt.grasp`), and decides among its candidates in three steps:

1. **Generate.** GraspGen-X samples 6-DoF grasps for the Franka Hand
   (``franka_panda``, the same hand: identical pad geometry and 0.1034 m
   fingertip depth, see :mod:`tpgpt.hardware.hand`) on the measured cloud, each
   with its discriminator score -- a learned estimate of whether that grasp
   holds, trained on simulated grasps across many grippers and objects.
2. **Refine, then screen on geometry** (:func:`refine_grasp`,
   :func:`screen_candidates`). GraspGen-X is trained on complete object meshes;
   on a one-camera cloud it stops short. Measured on the real banana cloud and a
   rendered partial cup, the top 30 % of its grasps (by score) had the object
   entering the pads a median 2-3 mm before the TCP, against 11 mm on a complete
   box -- 12 mm of the 18.5 mm pads engaged, and 40 % under 10 mm. Each grasp is
   therefore slid deeper along its own approach until the object enters 3 mm
   before the pad top (at most 25 mm), and centred between the pads along the
   closing axis (at most 10 mm), backing off wherever the hand would no longer
   fit. Then, cheapest first, every rejection named:

   * ``from_below`` -- the approach leans more than ``max_tilt_deg`` from
     straight down, so the hand would come up through the support
     (:func:`tpgpt.grasp.filters.by_support_approach`'s rule);
   * the jaws, :func:`tpgpt.hardware.hand.check_jaws`, which measures only
     the cloud **between the pads**: ``too_wide`` (material under an open
     pad, or wider than 80 mm there), ``collides`` (palm or finger link inside
     the object), ``empty`` (the pads close on air) or ``shallow`` (less than
     10 mm of the pad height has object behind it). A cup rim passes this
     although the cup is wider than the hand;
   * ``hits_support`` -- a corner of the hand below the support plane;
   * ``duplicate`` -- within 1 cm and 15 degrees of a better-scored survivor.

3. **Plan and verify** (:func:`select_and_plan`). Survivors are tried in score
   order through :func:`tpgpt.hardware.planner.plan`. Transport is what decides
   whether a grasp is *executable from this demonstration*: the map has to
   rotate the demonstrated grasp onto the candidate, and it degrades as that
   rotation grows (``min det(J)`` 0.94-0.99 at 3-19 degrees, 0.33-0.79 at
   74-135, folding past about 145 -- ``tpgpt.grasp.filters.
   by_demonstration_consistency``). So each plan is gated on its own measured
   ``min det(J)``, on how closely the transported path reaches the grasp and
   matches its orientation, and on the hand staying above the table and the
   destination along the whole path. The **highest-scoring candidate that
   passes** is chosen.

Ranking is by the discriminator score and nothing else; the demonstration only
enters as the feasibility gate of step 3. That is the same split
:func:`tpgpt.grasp.filters.filter_grasps` argues for.

No ROS here. The node supplies the candidates (or this module fetches them over
ZMQ with :func:`request_candidates`).
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
    #: Plans attempted, best score first, before giving up.
    max_plans: int = 12
    #: Plan acceptance.
    min_determinant: float = 0.1
    max_aim_grasp: float = 0.01
    max_orientation_error_deg: float = 15.0
    #: The hand's lowest corner along the path may dip this far below a surface
    #: (negative = allowed penetration). Set-down contact is normal; digging in
    #: is not.
    min_path_clearance: float = -0.002
    #: Try :func:`default_grasp` when no generated candidate passes.
    fallback_top_down: bool = True
    #: :func:`refine_grasp`: slide each candidate deeper and centre it.
    refine: bool = True
    refine_margin: float = 0.003
    refine_max_depth: float = 0.025
    refine_max_centre: float = 0.010
    #: Base-frame shift (m) added to the TCP of every grasp just before it is
    #: planned, generated or top-down alike. Screening still uses the unshifted grasp.
    grasp_offset: tuple = (0.0, 0.0, 0.0)


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
        }


@dataclass
class Selection:
    """The chosen plan, and the record of how it was chosen."""

    plan: Plan
    chosen: Candidate
    candidates: list[Candidate]
    source: str                     # "graspgenx", "top_down" or "side"
    notes: list[str] = field(default_factory=list)

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
    """Slide a grasp deeper along its approach and centre the object between the pads.

    Only ever moves the hand *in* (never shallower than the generator put it),
    and only to poses where :func:`tpgpt.hardware.hand.check_jaws` still says
    ``fits`` and the hand clears the support by ``min_support_clearance``. The
    orientation is untouched.

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

    for depth in np.arange(want, -1e-9, -0.002):
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


def screen_candidates(
    points: np.ndarray,
    poses: np.ndarray,
    scores: np.ndarray,
    support_height: float,
    settings: SelectionSettings | None = None,
    tcp_depth: float = hand.TCP_OFFSET,
) -> list[Candidate]:
    """Judge every generated grasp on the cloud alone. Sorted best score first;
    admissible ones have ``rejected is None``."""
    s = settings or SelectionSettings()
    points = np.atleast_2d(np.asarray(points, dtype=float))
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    candidates: list[Candidate] = []
    kept: list[Candidate] = []
    cos_dup = np.cos(np.radians(s.duplicate_angle_deg))
    for i in order:
        grasp = pose_to_grasp_frame(poses[i], tcp_depth)
        c = Candidate(index=int(i), score=float(scores[i]), grasp=grasp,
                      tilt_deg=_tilt_deg(grasp.approach))
        candidates.append(c)
        if c.tilt_deg > s.max_tilt_deg:
            c.rejected = f"from_below: approach {c.tilt_deg:.0f} deg from vertical"
            continue
        if s.refine:
            grasp, c.refined_depth, c.refined_centre = refine_grasp(points, grasp,
                                                                    support_height, s)
            c.grasp = grasp
        c.jaw = hand.check_jaws(points, grasp.tcp, grasp.rotation)
        if not c.jaw.ok:
            detail = f" ({', '.join(c.jaw.blocked_by)})" if c.jaw.blocked_by else ""
            c.rejected = (f"{c.jaw.verdict}: width {c.jaw.width * 1000:.0f} mm, "
                          f"{c.jaw.engagement * 1000:.0f} mm of pad engaged{detail}")
            continue
        c.support_clearance = hand.lowest_point(grasp.tcp, grasp.rotation) - float(support_height)
        if c.support_clearance < s.min_support_clearance:
            c.rejected = f"hits_support: hand {c.support_clearance * 1000:.0f} mm above the table"
            continue
        if any(np.linalg.norm(grasp.tcp - k.grasp.tcp) < s.duplicate_position
               and float(grasp.approach @ k.grasp.approach) > cos_dup
               and abs(float(grasp.closing @ k.grasp.closing)) > cos_dup for k in kept):
            c.rejected = "duplicate"
            continue
        kept.append(c)
    return candidates


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


def select_and_plan(
    source,
    scene: TargetScene,
    poses: np.ndarray | None,
    scores: np.ndarray | None,
    settings: SelectionSettings | None = None,
    tcp_depth: float = hand.TCP_OFFSET,
    unavailable_reason: str | None = None,
    **plan_kwargs,
) -> Selection:
    """Screen the generated candidates, plan the best ones, return the first that passes.

    Args:
        poses, scores: GraspGen-X output in the base frame. ``None`` (server
            unreachable) goes straight to the top-down fallback, if allowed.
        unavailable_reason: Why there are no candidates, for the record.
        plan_kwargs: Forwarded to :func:`tpgpt.hardware.planner.plan`.

    Raises:
        PlanningError: nothing passed, with the tally of why.
    """
    s = settings or SelectionSettings()
    points = np.atleast_2d(np.asarray(scene.points, dtype=float))
    support = (float(scene.support_height) if scene.support_height is not None
               else float(np.percentile(points[:, 2], 1.0)))
    notes: list[str] = []
    candidates: list[Candidate] = []

    if poses is not None and len(poses):
        candidates = screen_candidates(points, poses, scores, support, s, tcp_depth)
        admissible = [c for c in candidates if c.rejected is None]
        for c in admissible[: max(int(s.max_plans), 0)]:
            result = _try_plan(source, scene, c, s, plan_kwargs)
            if result is not None:
                return Selection(result, c, candidates, "graspgenx", notes)
        for c in admissible[max(int(s.max_plans), 0):]:
            c.rejected = "not_tried: max_plans reached"
        notes.append("no GraspGen-X candidate passed")
    else:
        notes.append(f"no GraspGen-X candidates ({unavailable_reason or 'none generated'})")

    tally = rejection_counts(candidates)
    if not s.fallback_top_down:
        error = PlanningError(f"no_executable_grasp: {len(candidates)} candidates; {tally}; "
                              + "; ".join(notes))
        # Every verdict, for a record of why nothing was executable (X6 trial record).
        error.candidates, error.fallback = candidates, None
        raise error

    grasp = default_grasp(points, support, plan_kwargs.get("grasp_depth_below_top",
                                                           0.03))
    fallback = Candidate(index=-1, score=float("nan"), grasp=grasp,
                         tilt_deg=_tilt_deg(grasp.approach),
                         jaw=hand.check_jaws(points, grasp.tcp, grasp.rotation))
    result = _try_plan(source, scene, fallback, s, {k: v for k, v in plan_kwargs.items()
                                                    if k != "grasp_depth_below_top"})
    if result is None:
        error = PlanningError(
            f"no_executable_grasp: {len(candidates)} generated candidates ({tally}), and the "
            f"top-down fallback was rejected too: {fallback.rejected}. " + "; ".join(notes))
        error.candidates, error.fallback = candidates, fallback
        raise error
    notes.append("fell back to the scripted top-down grasp")
    return Selection(result, fallback, candidates, "top_down", notes)


def plan_side_grasp(
    source,
    scene: TargetScene,
    end_effector,
    settings: SelectionSettings | None = None,
    side_settings=None,
    **plan_kwargs,
) -> Selection:
    """Plan the scripted side grasp for a wrapping hand; no generator, one candidate.

    The grasp comes from :func:`tpgpt.hardware.side_grasp.side_grasp` on the
    cloud, then goes through the same :func:`plan` and the same acceptance
    checks as a GraspGen-X candidate (``min det(J)``, aim, orientation error at
    the grasp, hand clearance along the path). ``grasp_offset`` is applied as
    for every other grasp.

    Raises:
        PlanningError: the side grasp could not be placed or its plan was refused.
    """
    from tpgpt.hardware.side_grasp import side_grasp

    s = settings or SelectionSettings()
    points = np.atleast_2d(np.asarray(scene.points, dtype=float))
    support = (float(scene.support_height) if scene.support_height is not None
               else float(np.percentile(points[:, 2], 1.0)))
    try:
        grasp, info = side_grasp(points, support, end_effector.handedness or "left", side_settings)
    except ValueError as exc:
        error = PlanningError(f"no_executable_grasp: side grasp: {exc}")
        error.candidates, error.fallback = [], None
        raise error from exc
    candidate = Candidate(index=-1, score=float("nan"), grasp=grasp,
                          tilt_deg=_tilt_deg(grasp.approach))
    candidate.support_clearance = end_effector.lowest_point(grasp.tcp, grasp.rotation) - support
    kwargs = {k: v for k, v in plan_kwargs.items() if k != "grasp_depth_below_top"}
    kwargs["end_effector"] = end_effector
    result = _try_plan(source, scene, candidate, s, kwargs)
    if result is None:
        error = PlanningError(f"no_executable_grasp: the side grasp was rejected: "
                              f"{candidate.rejected}")
        error.candidates, error.fallback = [], candidate
        raise error
    result.metadata["side_grasp"] = info.as_dict()
    return Selection(result, candidate, [], "side", list(info.notes))


__all__ = [
    "GRASPGEN_GRIPPER",
    "Candidate",
    "Selection",
    "SelectionSettings",
    "plan_side_grasp",
    "pose_to_grasp_frame",
    "refine_grasp",
    "request_candidates",
    "screen_candidates",
    "select_and_plan",
]
