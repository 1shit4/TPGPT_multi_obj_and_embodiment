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

from tpgpt.hardware.fr3 import TCP_LINK, FR3, default_model

#: ``tpgpt.grasp.filters`` constants, restated so this module does not import
#: robosuite; ``tests/unit/test_path_check.py`` asserts they stay equal.
REACH_TOLERANCE = 0.015
CONTACT_DEPTH_TOLERANCE = 0.001
CLOUD_CONTACT_RADIUS = 0.005
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


@dataclass
class _Piece:
    """One posed robot piece: an inside test in world coordinates and its vertices."""

    name: str
    contains: object          # callable(points (n, 3)) -> (n,) bool
    vertices: np.ndarray      # (v, 3) world
    centre: np.ndarray
    radius: float


def robot_pieces(fr3: FR3, q, end_effector, opening: float) -> tuple[list[_Piece], np.ndarray]:
    """Every robot piece at configuration ``q``: the arm's hulls and the hand's boxes.

    Returns ``(pieces, T_tcp)``.
    """
    poses = fr3.link_poses(q)
    pieces = []
    for hull, T in fr3.posed_hulls(poses=poses):
        pieces.append(_Piece(
            name=hull.name,
            contains=lambda p, h=hull, T=T: h.contains(p, T),
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
            contains=lambda p, b=box: b.contains((np.atleast_2d(p) - p_g) @ R_g),
            vertices=corners,
            centre=centre,
            radius=float(np.linalg.norm(corners - centre, axis=1).max()),
        ))
    return pieces, T


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


def _robot_faults(pieces, world, phase: str, depth_tolerance: float,
                  check_self: bool = True) -> dict:
    """Robot pieces against the shelf model, the observed cloud and themselves."""
    faults: dict = _self_faults(pieces, phase, depth_tolerance) if check_self else {}
    table = getattr(world, "table_height", None)
    if table is not None:
        for piece in pieces:
            if float(piece.vertices[:, 2].min()) < table - depth_tolerance:
                key = f"{phase}: {piece.name} in the table"
                faults[key] = faults.get(key, 0) + 1
    model = world.model
    if model is not None:
        surface, owner = world.surface()
        tree = world.tree("surface")
        for piece in pieces:
            if not model.near(piece.vertices, 0.0):
                continue
            depth, names = model.penetration(piece.vertices)
            deep = depth > depth_tolerance
            hit = {names[i] for i in np.flatnonzero(deep)}
            if tree is not None:
                near = tree.query_ball_point(piece.centre, piece.radius + 1e-6)
                if near:
                    near = np.asarray(near, dtype=int)
                    inside = piece.contains(surface[near])
                    hit |= {model.boxes[k].name for k in np.unique(owner[near[inside]])}
            for panel in hit:
                key = f"{phase}: {piece.name} in shelf {panel}"
                faults[key] = faults.get(key, 0) + 1
    tree = world.tree("cloud")
    if tree is not None:
        for piece in pieces:
            near = tree.query_ball_point(piece.centre, piece.radius + 1e-6)
            if not near:
                continue
            inside = piece.contains(world.cloud[np.asarray(near, dtype=int)])
            if inside.any():
                key = f"{phase}: {piece.name} in the seen points"
                faults[key] = faults.get(key, 0) + int(inside.sum())
    return faults


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
    tolerance: float = REACH_TOLERANCE,
    depth_tolerance: float = CONTACT_DEPTH_TOLERANCE,
    contact_radius: float = CLOUD_CONTACT_RADIUS,
    grasp_window: int = GRASP_CONTACT_WINDOW,
    place_window: int = PLACE_WINDOW,
    stop_early: bool = False,
    min_reachable_fraction: float | None = MIN_REACHABLE_FRACTION,
    check_self: bool = True,
) -> dict:
    """Solve and check a transported path, waypoint by waypoint.

    Args:
        positions: ``(m, 3)`` grasp-point positions of the plan (``Plan.labels``).
        rotations: ``(m, 3, 3)`` controller-frame orientations of the plan.
        gripper: ``(m,)`` gripper labels, ``> 0`` closed.
        end_effector: The hand on the flange (``tpgpt.hardware.end_effector``).
        world: The :class:`~tpgpt.hardware.shelf.ShelfWorld`.
        object_points: The target's cloud at the pick; carried from the close on.
        closed_width: Jaw opening while holding (the plan's width between pads).
        start_q: Measured joint angles now. Given, the line from the current TCP
            (``start_tcp``, else ``FK(start_q)``) to the path's first pose is
            checked first and seeds the path. ``None`` seeds from home and
            skips it, which the result says.
        stop_early: Return at the first fault, or once more waypoints are
            unreachable than ``min_reachable_fraction`` allows -- for
            *selecting*; leave it off to *diagnose*, where the whole tally is
            the point.

    Returns:
        ``tested``, ``unreachable``, ``reachable_fraction``, ``violations``,
        ``violation_fraction``, ``first_violation`` (waypoint index; ``-1`` for
        the approach-in line), ``faults`` keyed ``"<phase>: <what>"``, and
        ``approach_in`` (``None`` when not checked).
    """
    from scipy.spatial import cKDTree

    from tpgpt.hardware.hand import MAX_OPENING

    fr3 = fr3 or default_model()
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    gripper = np.asarray(gripper, dtype=float).reshape(-1)
    offset = np.asarray(end_effector.tool_offset, dtype=float)
    held = np.atleast_2d(np.asarray(object_points, dtype=float)) if len(
        np.asarray(object_points)) else np.zeros((0, 3))
    cloud_tree = world.tree("cloud")
    place_from = max(int(release_index) - int(place_window), int(grasp_index))

    local = None
    if len(held):
        p0 = positions[int(grasp_index)]
        R0 = rotations[int(grasp_index)]
        local = (held - p0) @ R0

    faults: dict = {}
    violations, first = 0, None

    def tcp_target(k):
        return positions[k] - rotations[k] @ offset, rotations[k]

    # --- the move onto the path -----------------------------------------------
    seed = None
    approach_in = None
    if start_q is not None:
        seed = np.asarray(start_q, dtype=float).reshape(7)
        p_now = (np.asarray(start_tcp, dtype=float).reshape(3) if start_tcp is not None
                 else fr3.tcp(seed)[:3, 3])
        p_goal, R_goal = tcp_target(0)
        n = max(int(np.ceil(np.linalg.norm(p_goal - p_now) / APPROACH_IN_STEP)), 1)
        R_now = fr3.tcp(seed)[:3, :3]
        from tpgpt.hardware.frames import slerp_rotation

        bad, unreached = {}, 0
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            result = fr3.solve_ik(p_now + s * (p_goal - p_now), slerp_rotation(R_now, R_goal, s),
                                  seed=seed, position_tolerance=tolerance)
            if not result.reachable:
                unreached += 1
                continue
            seed = result.qpos
            pieces, _ = robot_pieces(fr3, seed, end_effector, MAX_OPENING)
            for key, count in _robot_faults(pieces, world, "approach_in", depth_tolerance,
                                            check_self).items():
                bad[key] = bad.get(key, 0) + count
            # The object to be picked is an obstacle on the way onto the path.
            for piece in pieces:
                if len(held) and piece.contains(held).any():
                    key = f"approach_in: {piece.name} in the target's cloud"
                    bad[key] = bad.get(key, 0) + 1
        approach_in = {"steps": int(n), "unreachable": int(unreached), "faults": bad}
        if bad:
            faults.update(bad)
            violations, first = 1, -1
            if stop_early:
                return _result(1, 0, violations, first, faults, approach_in)

    # --- the path -----------------------------------------------------------------
    indices = np.arange(0, len(positions), max(int(stride), 1))
    # Selecting, a candidate is also decided once so many waypoints are
    # unreachable that the fraction cannot reach the admission threshold -- the
    # verdict is then fixed, and each failed solve costs 120 iterations.
    max_unreachable = (int(np.floor((1.0 - float(min_reachable_fraction)) * len(indices)))
                       if min_reachable_fraction is not None else len(indices))
    tested, unreachable = 0, 0
    for k in indices:
        tested += 1
        target_p, target_R = tcp_target(k)
        result = fr3.solve_ik(target_p, target_R, seed=seed, position_tolerance=tolerance)
        if not result.reachable:
            unreachable += 1
            if stop_early and unreachable > max_unreachable:
                # Report the fraction as if the rest were reachable: still below
                # the threshold, which is all a selection needs to know.
                return _result(len(indices), unreachable, violations, first, faults, approach_in,
                               decided_early=True)
            continue
        seed = result.qpos

        phase = ("approach" if k < grasp_index
                 else "carry" if k <= release_index else "retreat")
        if phase == "carry" and k >= place_from:
            phase = "place"
        opening = closed_width if gripper[k] > 0 else MAX_OPENING
        pieces, _ = robot_pieces(fr3, seed, end_effector, opening)

        hits = _robot_faults(pieces, world, phase, depth_tolerance, check_self)

        # --- the robot against the object it has not picked up yet ---------------
        # Outside the grasp window only, as in the simulation: within it the jaws
        # are closing on the target, and the jaw check at the grasp pose
        # (``hand.check_jaws``) has already judged the palm and fingers there.
        if len(held) and phase == "approach" and abs(int(k) - int(grasp_index)) > int(grasp_window):
            for piece in pieces:
                inside = piece.contains(held)
                if inside.any():
                    key = f"approach: {piece.name} in the target's cloud"
                    hits[key] = hits.get(key, 0) + 1

        # --- the object it is carrying, against everything else ---------------
        # Exact faces, not directions, so the object's rotation does not matter:
        # it may rest on the TOP face of a support board or of the table (at most
        # the contact radius into it); anything else it enters is a fault, and so
        # is any contact with the remaining seen points (other objects).
        if local is not None and phase in ("carry", "place"):
            world_obj = positions[k] + local @ rotations[k].T
            if world.model is not None and world.model.near(world_obj, 0.0):
                for panel, count in world.model.object_faults(
                        world_obj, rest=contact_radius, tolerance=depth_tolerance).items():
                    key = f"{phase}: the carried object in shelf {panel}"
                    hits[key] = hits.get(key, 0) + count
            table = getattr(world, "table_height", None)
            if table is not None:
                sunk = int(np.sum(world_obj[:, 2] < table - contact_radius))
                if sunk:
                    key = f"{phase}: the carried object in the table"
                    hits[key] = hits.get(key, 0) + sunk
            if cloud_tree is not None:
                near = cloud_tree.query_ball_point(world_obj, contact_radius)
                touched = {int(i) for group in near for i in group}
                if touched:
                    key = f"{phase}: the carried object touches seen points"
                    hits[key] = hits.get(key, 0) + len(touched)
        if hits:
            for key, count in hits.items():
                faults[key] = faults.get(key, 0) + count
            violations += 1
            first = int(k) if first is None else first
            if stop_early:
                break
    return _result(tested, unreachable, violations, first, faults, approach_in)


def _result(tested, unreachable, violations, first, faults, approach_in,
            decided_early: bool = False) -> dict:
    return {
        "decided_early": bool(decided_early),
        "tested": int(tested),
        "unreachable": int(unreachable),
        "reachable_fraction": float(1.0 - unreachable / tested) if tested else 0.0,
        "violations": int(violations),
        "violation_fraction": float(violations / tested) if tested else 0.0,
        "first_violation": None if first is None else int(first),
        "faults": faults,
        "approach_in": approach_in,
    }


def check_plan(plan, world, end_effector, start_q=None, start_tcp=None, fr3: FR3 | None = None,
               stop_early: bool = False, **kwargs) -> dict:
    """:func:`path_feasibility` on a :class:`tpgpt.hardware.planner.Plan`."""
    m = plan.metrics
    return path_feasibility(
        plan.labels.positions, plan.labels.orientations, plan.labels.gripper, end_effector,
        world, plan.target_points, int(m["grasp_index"]), int(m["release_index"]),
        closed_width=float(m.get("object_width_closing_m", 0.0)), start_q=start_q,
        start_tcp=start_tcp, fr3=fr3, stop_early=stop_early, **kwargs)


def verdict(check: dict, min_reachable_fraction: float = MIN_REACHABLE_FRACTION) -> str | None:
    """``pipeline._choose_grasp``'s admission rule: ``None`` if admissible, else why not."""
    if check["violations"]:
        faults = ", ".join(f"{k} x{v}" for k, v in list(check["faults"].items())[:3])
        where = ("on the way onto the path" if check["first_violation"] == -1
                 else f"first at waypoint {check['first_violation']}")
        return f"collision: {where} ({faults})"
    if check["reachable_fraction"] < min_reachable_fraction:
        return (f"kinematics: the arm holds {check['reachable_fraction'] * 100:.0f} % of the "
                f"tested waypoints (need {min_reachable_fraction * 100:.0f} %)")
    return None


__all__ = ["penetration", "CLOUD_CONTACT_RADIUS", "CONTACT_DEPTH_TOLERANCE", "GRASP_CONTACT_WINDOW",
           "MIN_REACHABLE_FRACTION", "PLACE_WINDOW", "REACH_TOLERANCE", "STRIDE",
           "check_plan", "path_feasibility", "robot_pieces", "verdict"]
