"""Can the real arm fly this whole transported path without hitting the shelf?

:func:`path_feasibility` is ``tpgpt.grasp.filters.path_feasibility_observed``
-- the check the simulation's grasp selection (``pipeline._choose_grasp``) runs
on every candidate -- with the simulator replaced by :mod:`tpgpt.hardware.fr3`
and the observed scene replaced by :class:`tpgpt.hardware.shelf.ShelfWorld`.
What carries over unchanged, because it is the method rather than the simulator:

* **one pass for both questions.** Each tested waypoint is solved by inverse
  kinematics seeded from the previous waypoint's answer; an unreachable
  waypoint is counted and checked no further ("a configuration IK did not
  reach is not a configuration"); a reachable one places every robot piece and
  tests it. ``reachable_fraction`` is over the tested waypoints.
* **the phases and what each allows.** ``approach`` (before the close):
  nothing may touch, except the fingers meeting the target within
  ``grasp_window`` waypoints of the close. ``carry`` and ``place`` (the last
  ``place_window`` waypoints of the carry): the robot touches nothing; the
  carried object may only rest on the top face of a support board or the
  table. ``retreat``: the robot touches nothing.
* **the carried object travels with the hand** from the close onwards, as the
  target's own cloud rigidly attached to the grasp.
* the thresholds: IK tolerance ``REACH_TOLERANCE`` (15 mm), cloud contact radius
  ``CLOUD_CONTACT_RADIUS`` (5 mm), the grasp window (3), the place window (12).
* **measured points get the same slack as everything else**: a robot piece is in
  the cloud only when at least :data:`MIN_CLOUD_CONTACT_POINTS` points are
  :data:`CLOUD_CONTACT_SLACK` inside it.

What is new, and why:

* **Exact geometry where it exists.** The shelf model and the table plane are
  treated as ``path_feasibility`` treats MuJoCo's geometry -- penetration
  deeper than ``CONTACT_DEPTH_TOLERANCE`` (1 mm) is a fault. The shelf is
  tested both ways (model surface samples inside a robot piece, robot vertices
  inside a panel), since a thin board can pass through a link with no vertex in
  it. The camera's points on the table and the shelf are removed
  (``ShelfWorld``); the rest are tested as the simulation tests its cloud.
* **Where the carried object may rest is decided by faces, not directions.**
  The simulation's cloud rule calls a contact "support" when it lies within 60
  degrees of straight down from the object's centre, which misreads the bottom
  corners of a wide flat object as side hits. With exact geometry the question
  is answerable directly: resting is allowed only on the top face of a support
  board or of the table, up to the contact radius; everything else, including
  any contact with the remaining seen points, is a fault.
* **The arm against itself.** The simulation's check skips robot-robot contacts.
  On the robot a folded arm is as real a collision as the shelf, so every pair
  of robot pieces is tested -- exactly, as convex bodies (:func:`penetration`) --
  except the pairs that overlap by construction: links joined at a joint
  (``link k``/``link k+1``, ``link7``/hand) and the hand's own pieces. Measured on
  40 at-rest poses of the real arm (``tests/data/fr3_still_samples.npz``) plus
  home: only those pairs ever intersect (by up to 34 mm), no other pair touches.
* **The move onto the path.** The simulated arm starts every run from the same
  home, which the scene was built around. The real arm starts wherever it was
  left, and the executor drives it in a straight line to the path's first pose
  (``executor.approach_trajectory``). Given the measured joint angles, that
  line is checked too, as phase ``approach_in``, and it seeds the path's IK the
  way the real arm will arrive -- which is also the configuration branch the
  impedance controller will actually be on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpgpt.hardware.fr3 import HOME_Q, TCP_LINK, FR3, default_model

#: ``tpgpt.grasp.filters`` constants, restated so this module does not import
#: robosuite; ``tests/unit/test_path_check.py`` asserts they stay equal.
REACH_TOLERANCE = 0.015
CONTACT_DEPTH_TOLERANCE = 0.001
CLOUD_CONTACT_RADIUS = 0.005

#: How far a *measured* point must be inside a robot piece before it counts as
#: a collision, metres, and how many such points are needed.
#:
#: Every other contact test here has a tolerance -- 1 mm into exact geometry,
#: 5 mm for an object resting on a surface -- while the robot-against-cloud
#: test had none: one point a tenth of a millimetre inside a piece was a
#: collision. The points it tests are measured, so they carry the camera's
#: error (5-10 mm at the shelf) and, for a carried object, the scan fit's
#: (3-5 mm). ``tpgpt.hardware.hand`` already ignores fewer than three points
#: for the same reason ("so a single stray depth pixel does not veto a grasp").
#: Measured case: an obvious top-down grasp of a YCB can was refused because,
#: 114 mm above the board and sliding sideways away, a finger pad clipped four
#: points of the can's top rim by 1-2 mm.
CLOUD_CONTACT_SLACK = 0.002
MIN_CLOUD_CONTACT_POINTS = 3
GRASP_CONTACT_WINDOW = 3
#: ``path_feasibility_observed``'s default.
PLACE_WINDOW = 12
#: ``pipeline._choose_grasp``'s stride.
STRIDE = 4
#: ``pipeline.MIN_REACHABLE_FRACTION``: a candidate whose path the arm can hold
#: at fewer than this fraction of the tested waypoints is rejected.
MIN_REACHABLE_FRACTION = 0.60
#: Spacing of the checked approach-in line, metres.
APPROACH_IN_STEP = 0.02
#: How close a joint may come to its limit (rad) when ``joint_margin`` is on: CRISP's
#: ``joint_limit_repulsion.safe_range`` (config/tpgpt_variable_stiffness_controller.yaml).
#: Inside it the controller pushes the joint back (up to 5 N m), a torque the arm's joint
#: torque sensors and the executor cannot tell from friction, and it holds the hand tens of
#: millimetres off its command: the 2026-10-01 22:20 mug plan's path, posed by this IK, ran
#: the elbow 0.04 rad from its limit over labels 0-28 and joint 1 to its limit at 37, where
#: the run stopped (the arm 33-40 mm off its path from the first label).
JOINT_MARGIN = 0.1
#: The joints held to it. Joint 7 turns the hand about its own axis, so pushed back it turns
#: the hand without moving it: at its limit through the replay's whole approach and both
#: mug runs that worked (2026-10-01), harmless.
MARGIN_JOINTS = (0, 1, 2, 3, 4, 5)

#: How far inside full stretch every checked posture must keep the wrist, m, when
#: ``reach_margin`` is on (``stretch_room``). **Why**: near full stretch the arm cannot be
#: *pulled* the last way out. The controller pulls the hand, and the joints turn only by what
#: that pull makes them feel; with the arm nearly straight, turning the elbow a lot moves the
#: hand outward very little, so a pull outward turns the elbow just as little -- a lever of a few
#: centimetres instead of the forearm's ~40 -- and joint friction wins. Measured 2026-10-02
#: (mug into top_right, 23:28): the plan's start was 1.8 mm inside full stretch (reachable, so
#: the IK check passed); the arm stopped 12 mm short of full stretch, 54 mm from its command,
#: and a 31 N pull gave the elbow about 1.4 N m; the run never got going. Across 12 runs the
#: gap at the start grew as the starts got more stretched (correlation -0.8). 20 mm (the user,
#: 2026-10-03), from its cost on the fifteen recorded captures (GRASP_LIBRARY.md item 11): 315 ->
#: 246 executable grasps, no cell emptied, bottom_left unchanged (10/30/50 mm: 264/231/200). It
#: keeps room for the 10-15 mm the arm stops short. The planner's default (`grasp_reach_margin`);
#: path_feasibility itself leaves it off unless asked. The shoulder-to-wrist distance depends
#: only on the elbow (joint 4); full stretch is 719.4 mm at -26.7 deg.
REACH_MARGIN = 0.02

#: Once the open hand has got this far from the placed object in the retreat, it must stay
#: that far (``retreat_clearance``), m. **Why**: the taught retreat swings the open fingers
#: back past the placed object, to 0.5-8 mm on most of 33 recent plans, and the arm strays
#: about 3 mm across the jaws after a release; on 2026-10-03 the fingers dragged a set-down
#: mug. 10 mm (the user, 2026-10-03, once bottom_left was no longer run), from its cost on the
#: ten top-slot captures (GRASP_LIBRARY.md item 11): on its own 307 -> 218 executable grasps;
#: with the 20 mm reach margin, bounded from the two measured separately, 105-149 of the
#: margin's 187 (pudding box excluded, measured on its old scan), no cell emptied (mug
#: top_right 6-8, foam brick top_left 4-13). It emptied bottom_left cells. The planner's
#: default (`grasp_retreat_clearance`); path_feasibility itself leaves it off unless asked.
RETREAT_CLEARANCE = 0.010


def stretch_room(fr3: FR3, q) -> float:
    """How far the wrist is inside full stretch, m: the longest shoulder-to-wrist distance
    the elbow allows minus this posture's (fr3_link2 to fr3_link6 origins; it depends only on
    joint 4)."""
    return _full_stretch(fr3) - _shoulder_to_wrist(fr3, q)


def _shoulder_to_wrist(fr3: FR3, q) -> float:
    poses = fr3.link_poses(np.asarray(q, dtype=float))
    return float(np.linalg.norm(poses["fr3_link6"][:3, 3] - poses["fr3_link2"][:3, 3]))


_FULL_STRETCH: dict = {}


def _full_stretch(fr3: FR3) -> float:
    key = id(fr3)
    if key not in _FULL_STRETCH:
        q = np.asarray(HOME_Q, dtype=float).copy()
        best = 0.0
        for q4 in np.linspace(fr3.lower[3], fr3.upper[3], 2001):
            q[3] = q4
            best = max(best, _shoulder_to_wrist(fr3, q))
        _FULL_STRETCH[key] = best
    return _FULL_STRETCH[key]

#: Robot pieces that never move, so a clearance for straying does not apply to
#: them: the base sits 13 mm above the touched table (2026-09-26), and a larger
#: clearance would call the bolted-down base a collision.
FIXED_PIECES = frozenset({"fr3_link0"})


@dataclass
class _Piece:
    """One posed robot piece: an inside test in world coordinates and its vertices."""

    name: str
    contains: object          # callable(points (n, 3), shrink=0.0) -> (n,) bool
    vertices: np.ndarray      # (v, 3) world
    centre: np.ndarray
    radius: float


def robot_pieces(fr3: FR3, q, end_effector, opening) -> tuple[list[_Piece], np.ndarray]:
    """Every robot piece at configuration ``q``: the arm's hulls and the hand's pieces.

    ``opening`` is the hand's state, as :meth:`EndEffector.boxes` takes it: an opening
    in metres for the Franka Hand, ``"open"`` or six joint angles for a modelled hand.
    Returns ``(pieces, T_tcp)``.
    """
    poses = fr3.link_poses(q)
    pieces = []
    for hull, T in fr3.posed_hulls(poses=poses):
        pieces.append(_Piece(
            name=hull.name,
            contains=lambda p, shrink=0.0, h=hull, T=T: h.contains(p, T, tolerance=shrink),
            vertices=hull.world_vertices(T),
            centre=T[:3, 3].copy(),
            radius=hull.radius,
        ))
    T = poses[TCP_LINK]
    R_g = end_effector.grasp_from_ee(T[:3, :3])
    p_g = T[:3, 3] + T[:3, :3] @ np.asarray(end_effector.tool_offset, dtype=float)
    for box in end_effector.boxes(opening):
        corners = box.corners() @ R_g.T + p_g
        centre = corners.mean(axis=0)
        pieces.append(_Piece(
            name=f"hand_{box.name}",
            contains=lambda p, shrink=0.0, b=box: b.contains((np.atleast_2d(p) - p_g) @ R_g,
                                                             margin=-shrink),
            vertices=corners,
            centre=centre,
            radius=float(np.linalg.norm(corners - centre, axis=1).max()),
        ))
    return pieces, T


def _touches(piece: "_Piece", points: np.ndarray, clearance: float = 0.0) -> int:
    """How many measured points are inside ``piece`` deeper than the slack.

    Returns 0 unless at least :data:`MIN_CLOUD_CONTACT_POINTS` are, so a stray
    depth pixel or a millimetre of fit error is not a collision. With a
    ``clearance`` the piece is grown by it first (see :func:`path_feasibility`).
    """
    if points is None or not len(points):
        return 0
    n = int(piece.contains(points, CLOUD_CONTACT_SLACK - clearance).sum())
    return n if n >= MIN_CLOUD_CONTACT_POINTS else 0


def penetration(A: np.ndarray, B: np.ndarray) -> float:
    """How deep two convex bodies, given by their vertices, interpenetrate, metres.

    Exact: they intersect exactly when their Minkowski difference ``A - B``
    contains the origin, and the origin's distance to that hull's nearest face
    is then the penetration depth. Negative when they are apart (a lower bound
    on the gap, not the gap).
    """
    from scipy.spatial import ConvexHull

    D = (np.asarray(A)[:, None, :] - np.asarray(B)[None, :, :]).reshape(-1, 3)
    return float(-ConvexHull(D).equations[:, 3].max())


def _joined(a: str, b: str) -> bool:
    """Pieces that overlap by construction: joint neighbours and the hand's own parts."""
    if a.startswith("hand_") and b.startswith("hand_"):
        return True
    pair = {a, b}
    if "fr3_link7" in pair and any(n.startswith("hand_") for n in pair):
        return True
    if a.startswith("fr3_link") and b.startswith("fr3_link"):
        return abs(int(a[-1]) - int(b[-1])) == 1
    return False


def _self_faults(pieces, phase: str, depth_tolerance: float) -> dict:
    """The arm against itself: every pair not joined by construction."""
    faults: dict = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            a, b = pieces[i], pieces[j]
            if _joined(a.name, b.name):
                continue
            ca, cb = a.vertices.mean(axis=0), b.vertices.mean(axis=0)
            d = cb - ca
            ra = float(np.linalg.norm(a.vertices - ca, axis=1).max())
            rb = float(np.linalg.norm(b.vertices - cb, axis=1).max())
            if float(np.linalg.norm(d)) > ra + rb:
                continue
            # Cheap exact reject: separated along the line between centroids.
            if float((a.vertices @ d).max()) + depth_tolerance * float(np.linalg.norm(d)) \
                    < float((b.vertices @ d).min()):
                continue
            if penetration(a.vertices, b.vertices) > depth_tolerance:
                key = f"{phase}: self {a.name} <-> {b.name}"
                faults[key] = faults.get(key, 0) + 1
    return faults


def piece_gaps(pieces, clearance: float = 0.0, tilt: float = 0.0, pivot=None) -> dict:
    """The gap each robot piece must keep: ``clearance`` plus how far a ``tilt`` swings it.

    The controller holds the grasp point's position and the hand's orientation with
    separate springs, so an orientation error turns the hand about the grasp point
    (``pivot``), and a tilt of ``tilt`` radians moves a point ``r`` from it by at most
    ``tilt * r``. The hand and link 7 are rigid with it: ``r`` is their farthest
    vertex (the finger pads 57 mm, the hand's top corners 169 mm). The other links
    follow the wrist, whose centre -- link 7's origin, ~0.21 m up -- moves by ``tilt``
    times its own distance; they are given that, which assumes the arm re-poses to
    carry the wrist and does not magnify its motion (approximately true away from a
    singularity). The base (:data:`FIXED_PIECES`) gets nothing.
    """
    tilt = float(tilt)
    if tilt <= 0.0 or pivot is None:
        return {p.name: 0.0 if p.name in FIXED_PIECES else float(clearance) for p in pieces}
    pivot = np.asarray(pivot, dtype=float)
    wrist = next((float(np.linalg.norm(p.centre - pivot)) for p in pieces
                  if p.name == "fr3_link7"), 0.21)
    gaps = {}
    for p in pieces:
        if p.name in FIXED_PIECES:
            gaps[p.name] = 0.0
            continue
        rigid = p.name.startswith("hand_") or p.name == "fr3_link7"
        reach = float(np.linalg.norm(p.vertices - pivot, axis=1).max()) if rigid else wrist
        gaps[p.name] = float(clearance) + tilt * reach
    return gaps


def _robot_faults(pieces, world, phase: str, depth_tolerance: float,
                  check_self: bool = True, clearance: float = 0.0, tilt: float = 0.0,
                  pivot=None, table_clearance=None) -> dict:
    """Robot pieces against the shelf model, the observed cloud and themselves.

    ``clearance`` (m) and ``tilt`` (rad, about ``pivot``) set the gap each piece
    must keep from the table, the shelf model and the seen points
    (:func:`piece_gaps`); the arm against itself is tested without it, and so is
    the base (:data:`FIXED_PIECES`), which is bolted down and cannot stray.
    ``table_clearance`` (m), when given, replaces ``clearance`` against the table.
    """
    faults: dict = _self_faults(pieces, phase, depth_tolerance) if check_self else {}
    gap = piece_gaps(pieces, clearance, tilt, pivot)
    table = getattr(world, "table_height", None)
    if table is not None:
        table_gap = (gap if table_clearance is None else
                     piece_gaps(pieces, table_clearance, tilt, pivot))
        for piece in pieces:
            if float(piece.vertices[:, 2].min()) < table - depth_tolerance + table_gap[piece.name]:
                key = f"{phase}: {piece.name} in the table"
                faults[key] = faults.get(key, 0) + 1
    model = world.model
    if model is not None:
        surface, owner = world.surface()
        tree = world.tree("surface")
        for piece in pieces:
            c = gap[piece.name]
            if not model.near(piece.vertices, c):
                continue
            depth, names = model.penetration(piece.vertices)
            deep = depth > depth_tolerance - c
            hit = {names[i] for i in np.flatnonzero(deep)}
            if tree is not None:
                near = tree.query_ball_point(piece.centre, piece.radius + c + 1e-6)
                if near:
                    near = np.asarray(near, dtype=int)
                    inside = piece.contains(surface[near], -c)
                    hit |= {model.boxes[k].name for k in np.unique(owner[near[inside]])}
            for panel in hit:
                key = f"{phase}: {piece.name} in shelf {panel}"
                faults[key] = faults.get(key, 0) + 1
    tree = world.tree("cloud")
    if tree is not None:
        for piece in pieces:
            c = gap[piece.name]
            near = tree.query_ball_point(piece.centre, piece.radius + c + 1e-6)
            if not near:
                continue
            n = _touches(piece, world.cloud[np.asarray(near, dtype=int)], c)
            if n:
                key = f"{phase}: {piece.name} in the seen points"
                faults[key] = faults.get(key, 0) + n
    return faults


#: The face the plan sets the object down on: its cloud points within this of
#: its lowest point **in the planned release pose**, metres. Not the face it
#: stood on at the pick -- a plan may turn the object (a box picked lying flat
#: and shelved upright); the release pose is the map's transport of the
#: demonstration's release, so the face down in it is the one the method means
#: to set down.
BASE_BAND = 0.010
#: A support surface is the pick or the destination support when its top lies
#: within this of the object's lowest point / the destination height, metres.
SUPPORT_MATCH = 0.02
#: IK retries before a waypoint is called unreachable: more iterations, then a
#: few starting postures near the previous answer (never a jump to another
#: configuration branch, which the impedance controller could not follow).
RETRY_ITERATIONS = 400
RETRY_SEEDS = 3
RETRY_PERTURBATION = 0.10       # rad, per joint
MAX_JOINT_JUMP = 0.50           # rad: a retry further than this from the previous answer is refused
#: Where the arm is posed for the collision tests: the waypoint itself, to 1 mm and
#: 0.6 deg. Reachability is decided at the simulation's tolerances (15 mm,
#: ``fr3.solve_ik``'s 0.15 rad), but the solver returns as soon as it is inside them,
#: often with the previous waypoint's joints unchanged: on the 18:39 demonstration's
#: path the arm the check posed sat a median 10.8 mm (max 14.8) and up to 8.6 deg from
#: its waypoint (2026-09-30). The collision tests were then about a different arm from
#: the one the controller drives there, by more than any tolerance they apply. From an
#: answer that close, refining costs a few iterations; where it does not converge the
#: loose answer is kept and counted (``unrefined``).
POSE_TOLERANCE = 0.001          # m
POSE_ROTATION_TOLERANCE = 0.01  # rad
POSE_ITERATIONS = 100


def _is_finger(name: str) -> bool:
    """Pieces allowed to touch the object at the pick, the place and early in the retreat.

    Every hand piece except the body: the Franka Hand's finger boxes (its body box
    is ``hand_body``); the Inspire hand's finger links (its wrist, palm, knuckles,
    thumb base and adapter are ``hand_body_*``).
    """
    return name.startswith("hand_") and not name.startswith("hand_body")


def path_feasibility(
    positions: np.ndarray,
    rotations: np.ndarray,
    gripper: np.ndarray,
    end_effector,
    world,
    object_points: np.ndarray,
    grasp_index: int,
    release_index: int,
    closed_width: float,
    start_q=None,
    start_tcp: np.ndarray | None = None,
    fr3: FR3 | None = None,
    stride: int = STRIDE,
    closed_state=None,
    tolerance: float = REACH_TOLERANCE,
    depth_tolerance: float = CONTACT_DEPTH_TOLERANCE,
    contact_radius: float = CLOUD_CONTACT_RADIUS,
    grasp_window: int = GRASP_CONTACT_WINDOW,
    place_window: int = PLACE_WINDOW,
    stop_early: bool = False,
    min_reachable_fraction: float | None = MIN_REACHABLE_FRACTION,
    check_self: bool = True,
    destination_height: float | None = None,
    criterion: str = "strict",
    approach_rejects: bool = True,
    clearance=None,
    tilt=None,
    held_shift=None,
    table_clearance=None,
    joint_margin=None,
    margin_joints=MARGIN_JOINTS,
    reach_margin=None,
    retreat_clearance=None,
) -> dict:
    """Solve and check a transported path, waypoint by waypoint.

    **Stages** (from the gripper schedule; ``g`` the close, ``r`` the release):

    ``approach`` (before ``g - grasp_window``)
        Nothing may touch anything; the object to be picked is an obstacle.
    ``pick`` (``g - grasp_window`` .. ``g``)
        The fingers may touch the object; the rest of the robot may not.
    ``carry`` (after ``g``, before ``r - place_window``)
        The object may rest on the surface it was picked from **until it first
        clears it**, never again. Otherwise it touches nothing, and no arm link
        touches it.
    ``place`` (``r - place_window`` .. ``r``)
        The object may touch the destination surface -- the top face of the
        destination board, or the table -- with its **place face** only (the
        points lowest in the planned release pose), at most ``contact_radius``
        deep. Any other contact, or another face touching, is a fault.
    ``retreat`` (after ``r``)
        The placed object stays where it was set down and is an obstacle. The
        fingers may touch it **until they first clear it**; the rest of the
        robot never.

    In every stage the robot touches neither the table, the shelf, the seen
    points, nor itself (``_robot_faults``).

    **Clearance** (``clearance``: None, metres, or one value per waypoint).
    Without it (the planner's check, as the simulation's), "touches" means
    more than ``depth_tolerance`` (1 mm) *into* the table or a shelf panel.
    That tolerance absorbs the arithmetic of surfaces that just meet; it says
    nothing about the arm not being exactly on the path. A clearance ``c`` asks
    for a gap instead: the robot pieces must stay ``c`` from the table, the
    shelf model and the seen points (the test becomes ``depth > tolerance -
    c``; outside a box its depth is minus the distance, slightly less near a
    corner, which errs towards less room), and the carried object ``c`` from
    the side panels, the boards' edges and undersides and the seen points. It
    is how a path the arm will follow only approximately -- a taught path the
    arm strays from by a measured amount (``tpgpt.hardware.replay``) -- is
    checked. Not given the clearance, on purpose: the arm against itself (a
    stray moves the whole arm, not one link into another), the robot against
    the object at the pick and the placed object in the retreat (contact there
    is the task), and the carried object coming down onto a support's top face
    (it is set down or lifted off there; a clearance would call the set-down
    itself a collision), and the fixed base.

    ``tilt`` (None, radians, or one per waypoint) adds the hand turning about
    the grasp point by up to that angle: each piece's gap grows by the angle
    times how far that piece is from the grasp point (:func:`piece_gaps`), and
    the carried object's by the angle times its own farthest point.

    ``table_clearance`` (None, metres, or one per waypoint) is the gap kept from the
    table instead of ``clearance`` (None: the same). The table is a horizontal
    plane, so only how far the arm may be *lower* matters there; :func:`check_plan`
    passes an arm path's vertical margin (the user, 2026-10-01: relax the table's
    margin, do not drop it).

    ``joint_margin`` (None, or rad: :data:`JOINT_MARGIN`) makes a posture with any of
    ``margin_joints`` closer than that to its limit a fault at that waypoint (``"<stage>:
    joint N within ... rad of its limit"``): reachable, but inside the controller's push-back.

    ``reach_margin`` (None, or m: :data:`REACH_MARGIN`) makes a posture whose wrist is closer
    than that to full stretch a fault (``"<stage>: the wrist within N mm of full stretch"``):
    reachable, but the arm cannot be pulled there.

    ``retreat_clearance`` (None, or m): once the hand has got that far from the placed object
    in the retreat, no robot piece may come back within it (``"retreat: <piece> back within N
    mm of the placed object"``), and every retreat label is checked, not every ``stride``-th.
    The placed object is the fitted scan, so one point is enough. Why: the taught retreat swings
    the open fingers back past the placed object (labels ~413-424), on 33 plans of 2026-10-01..03
    to within 0.5-8 mm of it and once 1.5 mm into it; the check above lets 2 mm through and
    looks every 4th label, and on 2026-10-03 the fingers dragged a set-down mug.

    ``held_shift`` (None or a base-frame vector, m) moves the object by it when the
    jaws close, before it is carried: a parallel jaw closing off-centre pushes the
    object to its centre (its fingers are coupled). :func:`check_plan` passes it
    for a path the arm really traced, whose hand is not at the planned grasp.

    **Reachability** (``criterion``):

    ``"strict"`` (default)
        Every tested waypoint, and every step of the move onto the path, must be
        reachable in position. The orientation may be given up (the executor
        relaxes it; checked at that posture) in transit only; in the pick and
        place stages it must be reached. IK is retried (:data:`RETRY_ITERATIONS`,
        :data:`RETRY_SEEDS` near the previous answer) before anything is called
        unreachable.
    ``"sim"``
        The simulation's rule: at least ``min_reachable_fraction`` of the tested
        waypoints reachable as planned; the rest are checked where the executor
        will put the arm (relaxed orientation, or the straight line it steers
        along past positions it cannot reach).

    Either way a reachable waypoint is then solved again, to :data:`POSE_TOLERANCE`,
    and the collision tests pose the arm there -- where the controller will drive it
    -- not where the solver first came within the reachability tolerance.

    Returns:
        ``tested``, ``unreachable`` (not reachable as planned), ``reachable_fraction``,
        ``orientation_relaxed``, ``relaxed_in_pick_or_place``, ``skipped``
        (position out of reach), ``steered_unreachable``, ``violations``,
        ``violation_fraction``, ``first_violation`` (``-1``: the move onto the
        path), ``faults``, ``approach_in``, ``criterion``, ``first_kinematic``,
        ``decided_early`` and ``unrefined`` (postures checked where the solver
        stopped, not at the waypoint).
    """
    # The hand open, and closed on the object: the Franka Hand's jaws at ``closed_width``;
    # a modelled hand at ``closed_state`` (its fingers stopped on the object, the plan's
    # ``hand_closed_q``), else fully closed.
    open_state = end_effector.open_state
    if end_effector.is_parallel_jaw:
        closed = float(closed_width)
    else:
        closed = "close" if closed_state is None else np.asarray(closed_state, dtype=float)

    if criterion not in ("strict", "sim"):
        raise ValueError(f"criterion must be 'strict' or 'sim', not {criterion!r}")
    fr3 = fr3 or default_model()
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    gripper = np.asarray(gripper, dtype=float).reshape(-1)
    offset = np.asarray(end_effector.tool_offset, dtype=float)
    held = np.atleast_2d(np.asarray(object_points, dtype=float)) if len(
        np.asarray(object_points)) else np.zeros((0, 3))
    g, r = int(grasp_index), int(release_index)
    pick_from = g - int(grasp_window)
    place_from = max(r - int(place_window), g + 1)
    cloud_tree = world.tree("cloud")
    table = getattr(world, "table_height", None)
    model = world.model
    gap = (np.zeros(len(positions)) if clearance is None else
           np.broadcast_to(np.asarray(clearance, dtype=float), (len(positions),)).copy())
    swing = (np.zeros(len(positions)) if tilt is None else
             np.broadcast_to(np.asarray(tilt, dtype=float), (len(positions),)).copy())
    table_gap = (None if table_clearance is None else
                 np.broadcast_to(np.asarray(table_clearance, dtype=float), (len(positions),)).copy())
    if np.any(gap < 0.0) or np.any(swing < 0.0) or (table_gap is not None and np.any(table_gap < 0.0)):
        raise ValueError("clearance and tilt must not be negative")

    def grasp_point(T):
        return T[:3, 3] + T[:3, :3] @ offset

    local = base = None
    if len(held):
        moved = held if held_shift is None else held + np.asarray(held_shift, dtype=float)
        local = (moved - positions[g]) @ rotations[g]
        at_release = positions[r] + local @ rotations[r].T
        base = at_release[:, 2] <= float(at_release[:, 2].min()) + BASE_BAND
    # Which surface the object is picked from, and which it is set down on.
    pick_support = _support_at(model, table, float(held[:, 2].min()) if len(held) else None)
    place_support = _support_at(model, table, destination_height)

    faults: dict = {}
    counts = {"relaxed": 0, "relaxed_windows": 0, "skipped": 0, "gap": 0, "unrefined": 0}
    violations, first, first_kin = 0, None, None

    def stage_of(k):
        if k < pick_from:
            return "approach"
        if k <= g:
            return "pick"
        if k < place_from:
            return "carry"
        return "place" if k <= r else "retreat"

    def add(hits, key, n=1):
        hits[key] = hits.get(key, 0) + int(n)

    def solve(target_p, target_R, seed, k):
        """IK with the retries the strict criterion allows, then posed at the waypoint."""
        result = _reach(target_p, target_R, seed, k)
        if not result.reachable:
            return result
        fine = fr3.solve_ik(target_p, target_R, seed=result.qpos, max_iterations=POSE_ITERATIONS,
                            position_tolerance=POSE_TOLERANCE,
                            rotation_tolerance=POSE_ROTATION_TOLERANCE)
        if fine.reachable:
            return fine
        counts["unrefined"] += 1
        return result

    def _reach(target_p, target_R, seed, k):
        """Reachable at the simulation's tolerances? (the retries of the strict criterion)."""
        result = fr3.solve_ik(target_p, target_R, seed=seed, position_tolerance=tolerance)
        if result.reachable or criterion != "strict":
            return result
        base_q = np.asarray(HOME_Q if seed is None else seed, dtype=float)
        rng = np.random.default_rng(int(k) + 1)
        for attempt in range(RETRY_SEEDS + 1):
            start = base_q if attempt == 0 else np.clip(
                base_q + rng.uniform(-RETRY_PERTURBATION, RETRY_PERTURBATION, 7),
                fr3.lower, fr3.upper)
            retry = fr3.solve_ik(target_p, target_R, seed=start, max_iterations=RETRY_ITERATIONS,
                                 position_tolerance=tolerance)
            if retry.reachable and (seed is None or np.max(np.abs(retry.qpos - base_q))
                                    <= MAX_JOINT_JUMP):
                return retry
        return result

    def actual_pose(q):
        T = fr3.tcp(q)
        return T[:3, 3] + T[:3, :3] @ offset, T[:3, :3]

    # --- the move onto the path -----------------------------------------------
    seed, approach_in = None, None
    if start_q is not None:
        seed = np.asarray(start_q, dtype=float).reshape(7)
        p_now = (np.asarray(start_tcp, dtype=float).reshape(3) if start_tcp is not None
                 else fr3.tcp(seed)[:3, 3])
        p_goal = positions[0] - rotations[0] @ offset
        n = max(int(np.ceil(np.linalg.norm(p_goal - p_now) / APPROACH_IN_STEP)), 1)
        R_now = fr3.tcp(seed)[:3, :3]
        from tpgpt.hardware.frames import slerp_rotation

        bad, unreached = {}, 0
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            R_s = slerp_rotation(R_now, rotations[0], s)
            result = solve(p_now + s * (p_goal - p_now), R_s, seed, -1)
            if not result.reachable:
                position_only = solve(p_now + s * (p_goal - p_now), None, seed, -1)
                if not position_only.reachable:
                    unreached += 1
                    continue
                result = position_only
            seed = result.qpos
            pieces, T = robot_pieces(fr3, seed, end_effector, open_state)
            for key, count in _robot_faults(pieces, world, "approach_in", depth_tolerance,
                                            check_self, gap[0], swing[0], grasp_point(T),
                                            None if table_gap is None else table_gap[0]).items():
                add(bad, key, count)
            for piece in pieces:
                if _touches(piece, held):
                    add(bad, f"approach_in: {piece.name} in the object to be picked")
        approach_in = {"steps": int(n), "unreachable": int(unreached), "faults": bad,
                       "rejects": bool(approach_rejects)}
        # With approach_rejects False the move onto the path is still checked and
        # reported, but it does not decide the path: where the arm happens to be
        # parked says nothing about the grasp, and the operator re-homes instead
        # (the planner's approach_blocked, user 2026-09-29).
        if bad and approach_rejects:
            faults.update(bad)
            violations, first = 1, -1
        if unreached and criterion == "strict" and approach_rejects:
            first_kin = -1
        if stop_early and approach_rejects and (bad or first_kin is not None):
            return _result(1, 0, violations, first, faults, approach_in, criterion=criterion,
                           first_kinematic=first_kin, **counts)

    # --- the path -----------------------------------------------------------------
    indices = np.arange(0, len(positions), max(int(stride), 1))
    if retreat_clearance is not None:          # the retreat at every label
        indices = np.union1d(indices, np.arange(min(r + 1, len(positions)), len(positions)))
    max_unreachable = (int(np.floor((1.0 - float(min_reachable_fraction)) * len(indices)))
                       if min_reachable_fraction is not None else len(indices))
    tested, unreachable = 0, 0
    lifted = pick_support is None        # nothing to lift off from
    fingers_cleared = False
    clear_by_margin = False              # the hand has been retreat_clearance from the placed object
    placed = None                        # the object where it was set down
    last_good, last_good_q = None, None
    #: The posture the arm was checked in at each tested waypoint (the result's ``postures``).
    posed_labels, posed_q = [], []

    def check_posture(k, q, stage, point, R_ee, tag=""):
        nonlocal lifted, fingers_cleared, clear_by_margin
        hits: dict = {}
        pieces, T = robot_pieces(fr3, q, end_effector, closed if gripper[k] > 0 else open_state)
        label = stage + tag
        for key, count in _robot_faults(pieces, world, label, depth_tolerance, check_self,
                                        gap[k], swing[k], grasp_point(T),
                                        None if table_gap is None else table_gap[k]).items():
            add(hits, key, count)
        if joint_margin is not None:
            qq = np.asarray(q, dtype=float)
            room = np.minimum(qq - fr3.lower, fr3.upper - qq)
            for j in margin_joints:
                if room[j] < float(joint_margin):
                    add(hits, f"{label}: joint {j + 1} within {float(joint_margin):.2f} rad "
                              "of its limit")
        if reach_margin is not None and stretch_room(fr3, q) < float(reach_margin):
            add(hits, f"{label}: the wrist within {float(reach_margin) * 1000:.0f} mm of full "
                      "stretch")

        if len(held) and stage in ("approach", "pick"):
            # The object stands where it is; only the fingers, and only at the pick.
            for piece in pieces:
                if (stage == "approach" or not _is_finger(piece.name)) and _touches(piece, held):
                    add(hits, f"{label}: {piece.name} in the object")

        if local is not None and stage in ("carry", "place"):
            obj = point + local @ R_ee.T
            for piece in pieces:
                if piece.name.startswith("fr3_link") and _touches(piece, obj):
                    add(hits, f"{label}: {piece.name} in the carried object")
            reach = float(np.linalg.norm(obj - point, axis=1).max()) if swing[k] > 0 else 0.0
            contact = _object_contacts(obj, base, world, model, table, depth_tolerance,
                                       contact_radius, gap[k] + swing[k] * reach)
            if not lifted and _clear_of(obj, pick_support, model, table, contact_radius):
                lifted = True
            allowed = set()
            if not lifted:
                allowed.add(("rest", pick_support))           # still standing on its support
            if stage == "place":
                allowed.add(("base_rest", place_support))     # set down, on its place face
            for kind, where, n in contact:
                ok = ((kind in ("rest", "base_rest") and ("rest", where) in allowed)
                      or (kind == "base_rest" and ("base_rest", where) in allowed))
                if not ok:
                    add(hits, f"{label}: the carried object {_describe(kind, where)}", n)

        if stage == "retreat" and placed is not None:
            touching = [p for p in pieces if _touches(p, placed)]
            fingers_touch = [p for p in touching if _is_finger(p.name)]
            if not fingers_touch:
                fingers_cleared = True
            for piece in touching:
                if not _is_finger(piece.name) or fingers_cleared:
                    add(hits, f"{label}: {piece.name} in the placed object")
            if retreat_clearance is not None:
                near = [p for p in pieces if p.contains(placed, -float(retreat_clearance)).any()]
                if not near:
                    clear_by_margin = True
                elif clear_by_margin:
                    for piece in near:
                        add(hits, f"{label}: {piece.name} back within "
                                  f"{float(retreat_clearance) * 1000:.0f} mm of the placed object")
        return hits

    for k in indices:
        tested += 1
        stage = stage_of(int(k))
        if stage == "retreat" and placed is None and local is not None:
            placed = positions[r] + local @ rotations[r].T
        target_p = positions[k] - rotations[k] @ offset
        result = solve(target_p, rotations[k], seed, k)
        kind = "full"
        if not result.reachable:
            unreachable += 1
            position_only = solve(target_p, None, seed, k)
            if position_only.reachable:
                result, kind = position_only, "relaxed"
                counts["relaxed"] += 1
                if stage in ("pick", "place"):
                    counts["relaxed_windows"] += 1
                    if criterion == "strict" and first_kin is None:
                        first_kin = int(k)
            else:
                counts["skipped"] += 1
                if criterion == "strict" and first_kin is None:
                    first_kin = int(k)
            if stop_early and (first_kin is not None or (
                    criterion == "sim" and unreachable > max_unreachable)):
                return _result(len(indices) if criterion == "sim" else tested, unreachable,
                               violations, first, faults, approach_in, decided_early=True,
                               criterion=criterion, first_kinematic=first_kin,
                               postures=(posed_labels, posed_q), **counts)
            if kind != "relaxed":
                continue

        hits: dict = {}
        if criterion == "sim" and last_good is not None and last_good[2] < k - int(stride):
            # The simulation's rule lets the arm skip out-of-reach stretches: check
            # the straight line the executor steers along, from the last posture
            # reached to this one.
            p0 = last_good[0]
            p1 = positions[k]
            n = max(int(np.ceil(np.linalg.norm(p1 - p0) / APPROACH_IN_STEP)), 1)
            q_line = last_good_q
            for s_ in np.linspace(0.0, 1.0, n + 1)[1:-1]:
                point = p0 + s_ * (p1 - p0)
                step = fr3.solve_ik(point - actual_pose(q_line)[1] @ offset, None, seed=q_line,
                                    position_tolerance=tolerance)
                if not step.reachable:
                    counts["gap"] += 1
                    continue
                q_line = step.qpos
                gp, gR = actual_pose(q_line)
                for key, count in check_posture(k, q_line, stage, gp, gR, " (steered)").items():
                    add(hits, key, count)
        seed = result.qpos
        if kind == "full":
            point, R_ee = positions[k], rotations[k]          # as the simulation carries it
        else:
            point, R_ee = actual_pose(seed)                    # the wrist is where IK put it
        for key, count in check_posture(k, seed, stage, point, R_ee,
                                        "" if kind == "full" else " (relaxed)").items():
            add(hits, key, count)
        last_good, last_good_q = (point, R_ee, int(k)), seed
        posed_labels.append(int(k))
        posed_q.append(np.asarray(seed, dtype=float).copy())
        if hits:
            for key, count in hits.items():
                add(faults, key, count)
            violations += 1
            first = int(k) if first is None else first
            if stop_early:
                break
    return _result(tested, unreachable, violations, first, faults, approach_in,
                   criterion=criterion, first_kinematic=first_kin,
                   postures=(posed_labels, posed_q), **counts)


def _support_at(model, table, height):
    """The support surface whose top is at ``height``: ``"table"``, a board's name, or None."""
    if height is None:
        return None
    if table is not None and abs(float(height) - float(table)) <= SUPPORT_MATCH:
        return "table"
    if model is not None:
        for box in model.boxes:
            if box.name in model.supports and abs(
                    float(box.position[2] + box.size[2]) - float(height)) <= SUPPORT_MATCH:
                return box.name
    return None


def _object_contacts(obj, base, world, model, table, depth_tolerance, contact_radius,
                     clearance: float = 0.0):
    """Every contact of the carried object with the world, classified.

    Returns ``[(kind, where, count)]``: ``kind`` is ``"base_rest"`` (points of
    ``base`` -- the place face -- on a support's top face, at most
    ``contact_radius`` into it),
    ``"rest"`` (other points there), ``"deep"`` (into a support deeper than
    that), ``"into"`` (into a non-support face: a side panel, a board's
    underside or edge) or ``"seen"`` (within ``contact_radius`` of a seen
    point). ``clearance`` grows the ``deep``, ``into`` and ``seen`` tests by
    that much and leaves the support tops alone (:func:`path_feasibility`).
    """
    out = []
    c = float(clearance)
    if table is not None:
        touching = obj[:, 2] < float(table) + depth_tolerance
        if touching.any():
            deep = obj[:, 2] < float(table) - contact_radius
            on_top = touching & ~deep
            if (on_top & base).any():
                out.append(("base_rest", "table", int((on_top & base).sum())))
            if (on_top & ~base).any():
                out.append(("rest", "table", int((on_top & ~base).sum())))
            if deep.any():
                out.append(("deep", "table", int(deep.sum())))
    if model is not None and model.near(obj, c):
        for box in model.boxes:
            depth = box.penetration(obj)
            inside, within = depth > depth_tolerance, depth > depth_tolerance - c
            if not within.any():
                continue
            if box.name in model.supports:
                z_local = (obj - box.position) @ box.rotation[:, 2]
                top = z_local >= float(box.size[2]) - contact_radius
                on_top = inside & top
                if (on_top & base).any():
                    out.append(("base_rest", box.name, int((on_top & base).sum())))
                if (on_top & ~base).any():
                    out.append(("rest", box.name, int((on_top & ~base).sum())))
                if (within & ~top).any():
                    out.append(("deep", box.name, int((within & ~top).sum())))
            else:
                out.append(("into", box.name, int(within.sum())))
    tree = world.tree("cloud")
    if tree is not None:
        near = tree.query_ball_point(obj, contact_radius + c)
        touched = {int(i) for group in near for i in group}
        if touched:
            out.append(("seen", "seen points", len(touched)))
    return out


def _clear_of(obj, support, model, table, contact_radius) -> bool:
    """Whether the carried object has lifted off ``support``."""
    if support == "table":
        return float(obj[:, 2].min()) > float(table) + contact_radius
    box = next((b for b in model.boxes if b.name == support), None) if model is not None else None
    if box is None:
        return True
    return float(obj[:, 2].min()) > float(box.position[2] + box.size[2]) + contact_radius


def _describe(kind: str, where: str) -> str:
    return {
        "base_rest": f"touches {where}",
        "rest": f"touches {where} with a face other than the one it is set down on",
        "deep": f"pressed into {where}",
        "into": f"in shelf {where}",
        "seen": "touches seen points",
    }[kind]


def _result(tested, unreachable, violations, first, faults, approach_in,
            decided_early: bool = False, relaxed: int = 0, relaxed_windows: int = 0,
            skipped: int = 0, gap: int = 0, criterion: str = "strict",
            first_kinematic: int | None = None, unrefined: int = 0, postures=None) -> dict:
    labels, q = postures if postures is not None else ([], [])
    return {
        "criterion": criterion,
        "decided_early": bool(decided_early),
        "tested": int(tested),
        "unreachable": int(unreachable),
        "reachable_fraction": float(1.0 - unreachable / tested) if tested else 0.0,
        # Of the waypoints not reachable as planned: held with the orientation
        # relaxed (checked at that posture), how many of those fell in the pick
        # or place stage, position out of reach, and unreachable points on the
        # lines steered past them (criterion "sim" only).
        "orientation_relaxed": int(relaxed),
        "relaxed_in_pick_or_place": int(relaxed_windows),
        "skipped": int(skipped),
        "steered_unreachable": int(gap),
        # Postures checked where the solver stopped, not at the waypoint (POSE_TOLERANCE).
        "unrefined": int(unrefined),
        "first_kinematic": first_kinematic,
        "violations": int(violations),
        "violation_fraction": float(violations / tested) if tested else 0.0,
        "first_violation": None if first is None else int(first),
        "faults": faults,
        "approach_in": approach_in,
        # The joint posture each tested waypoint was checked in (each IK seeded from the one
        # before, from ``start_q`` through the move onto the path): what the executor sends
        # CRISP as its posture target, so the arm holds the posture that was checked.
        "postures": {"labels": np.asarray(labels, dtype=int),
                     "q": np.asarray(q, dtype=float).reshape(-1, 7)},
    }


def check_plan(plan, world, end_effector, start_q=None, start_tcp=None, fr3: FR3 | None = None,
               stop_early: bool = False, **kwargs) -> dict:
    """:func:`path_feasibility` on a :class:`tpgpt.hardware.planner.Plan`: its labels, or, for
    learned labels, the arm's own path with its margin (``Plan.arm_path``)."""
    m = plan.metrics
    kwargs.setdefault("destination_height", float(plan.destination_height))
    path = plan.labels
    arm = getattr(plan, "arm_path", None)
    if arm is not None:
        # Learned labels: the arm's own path is checked, with its run-to-run margin on top of
        # any clearance asked for (the attractor is not where the arm goes).
        path = arm
        extra = kwargs.pop("clearance", None)
        kwargs["clearance"] = np.asarray(plan.arm_margin, dtype=float) + (
            0.0 if extra is None else np.asarray(extra, dtype=float))
        # Against the table only the vertical part of the margin (the user, 2026-10-01: relax
        # it, do not drop it); a bundle without it keeps the full margin there.
        table = getattr(plan, "arm_table_margin", None)
        if table is not None and "table_clearance" not in kwargs:
            kwargs["table_clearance"] = np.asarray(table, dtype=float) + (
                0.0 if extra is None else np.asarray(extra, dtype=float))
        if end_effector.is_parallel_jaw and "held_shift" not in kwargs:
            # The hand closes where the arm is, off the planned grasp; the coupled jaws
            # centre the object across them (along the approach and the jaws' width it
            # stays where it stood).
            c = np.asarray(plan.grasp.closing, dtype=float)
            c = c / np.linalg.norm(c)
            miss = np.asarray(arm.positions[int(m["grasp_index"])]) - np.asarray(plan.grasp.tcp)
            kwargs["held_shift"] = float(miss @ c) * c
    return path_feasibility(
        path.positions, path.orientations, path.gripper, end_effector,
        world, plan.target_points, int(m["grasp_index"]), int(m["release_index"]),
        closed_width=float(m.get("object_width_closing_m", 0.0)), start_q=start_q,
        start_tcp=start_tcp, fr3=fr3, stop_early=stop_early,
        closed_state=None if end_effector.is_parallel_jaw else m.get("hand_closed_q"), **kwargs)


def verdict(check: dict, min_reachable_fraction: float = MIN_REACHABLE_FRACTION) -> str | None:
    """Whether a checked path is admissible: ``None`` if so, else why not.

    A collision anywhere rejects it. Then reachability, by ``check["criterion"]``:
    ``"strict"`` needs every waypoint (and the move onto the path) reachable in
    position and the pick and place stages reachable with their orientation;
    ``"sim"`` needs ``min_reachable_fraction`` of the waypoints reachable as
    planned (``pipeline._choose_grasp``'s rule).
    """
    if check["violations"]:
        faults = ", ".join(f"{k} x{v}" for k, v in list(check["faults"].items())[:3])
        where = ("on the way onto the path" if check["first_violation"] == -1
                 else f"first at waypoint {check['first_violation']}")
        return f"collision: {where} ({faults})"
    if check.get("criterion", "sim") == "strict":
        onto_check = check.get("approach_in") or {}
        onto = onto_check.get("unreachable", 0) if onto_check.get("rejects", True) else 0
        if onto:
            return f"kinematics: {onto} points of the move onto the path are out of reach"
        if check.get("skipped"):
            return (f"kinematics: {check['skipped']} waypoints out of reach in position (first at "
                    f"waypoint {check.get('first_kinematic')})")
        if check.get("relaxed_in_pick_or_place"):
            return (f"kinematics: the orientation cannot be held at {check['relaxed_in_pick_or_place']}"
                    f" waypoints of the pick or place (first at waypoint "
                    f"{check.get('first_kinematic')})")
        return None
    if check["reachable_fraction"] < min_reachable_fraction:
        return (f"kinematics: the arm holds {check['reachable_fraction'] * 100:.0f} % of the "
                f"tested waypoints (need {min_reachable_fraction * 100:.0f} %)")
    return None


__all__ = [
    "JOINT_MARGIN", "REACH_MARGIN", "stretch_room", "BASE_BAND", "penetration", "CLOUD_CONTACT_RADIUS", "CLOUD_CONTACT_SLACK",
           "MIN_CLOUD_CONTACT_POINTS", "CONTACT_DEPTH_TOLERANCE", "GRASP_CONTACT_WINDOW",
           "MIN_REACHABLE_FRACTION", "PLACE_WINDOW", "REACH_TOLERANCE", "STRIDE",
           "check_plan", "path_feasibility", "robot_pieces", "verdict"]
