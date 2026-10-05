"""A grasp on its own: approach, close, lift, set down, release, retreat (plan stage 6).

Before a GraspGen-X grasp is carried through a transported demonstration, it has to be
shown to hold. This builds the smallest plan that tests one grasp and nothing else, as
an ordinary :class:`tpgpt.hardware.planner.Plan` that the existing executor runs --
its start onto the path, posture target, holds, aborts, gripper commands and full-rate
logs -- and that the whole-path check judges like any other:

1. the hand ``approach_distance`` back from the grasp along its approach, open;
2. straight in along the approach to the grasp, open, and a dwell;
3. **close** (the gripper label switches), a dwell;
4. **lift** ``lift`` straight up, and hold it there (``hold_lifted``) -- held or not
   is decided here;
5. back down to the grasp, **release**, a dwell;
6. back out along the approach.

The orientation is the grasp's throughout. No transport: the keypoints are the grasp
cube on both sides (the identity map), ``min det`` 1, and the object is set down where
it stood. Labels are at ``dt`` (20 Hz) and the executor stretches them by its
``speed_scale`` like every plan.

:func:`select_grasp_test` picks the grasp the way selection does -- GraspGen-X's
candidates screened (the hand's own hard stage, the funnel; no place zone), then the
first whose test path passes the whole-path check -- or a pinned one
(``library_index``) for a depth sweep, where the same grasp is tested at several
depths (``depth_offset``: the hand that much further along its approach).
:func:`constructed_grasp` is a grasp built by hand, not by the generator -- straight
down, closing across the object's narrow side -- for checking the frame chain: the
hand must arrive as drawn (``docs/sim/gripper_configs.md`` section 15).

No ROS; the planner node's ``task:=grasp_test`` calls it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.planner import Plan, PlanningError, _hand_clearance
from tpgpt.sim.keypoints import GRASP_CUBE_HALF_EXTENT, GraspFrame, KeypointSet, carry_indices
from tpgpt.transport.labels import PolicyLabels

#: Label period, s (the demonstrations' 20 Hz).
DT = 0.05
APPROACH_DISTANCE = 0.10      # m back from the grasp along the approach
LIFT = 0.10                   # m straight up once closed
SPEED = 0.06                  # m/s at the labels (the executor's speed_scale slows it)
DWELL = 1.0                   # s at the grasp, open and closed, and after the release
HOLD_LIFTED = 2.0             # s held up
#: The labels' stiffness, N/m, the same in every direction, critically damped
#: (2 sqrt(K)): the static profile of a taught demonstration
#: (``teach --stiffness-profile static``), which the executor's gate reads.
STIFFNESS = 400.0


def grasp_test_labels(grasp: GraspFrame, dt: float = DT,
                      approach_distance: float = APPROACH_DISTANCE, lift: float = LIFT,
                      speed: float = SPEED, dwell: float = DWELL,
                      hold_lifted: float = HOLD_LIFTED,
                      stiffness: float = STIFFNESS) -> PolicyLabels:
    """The test's labels in the grasp convention (grasp point, grasp-frame orientations)."""
    a = np.asarray(grasp.approach, dtype=float)
    tcp = np.asarray(grasp.tcp, dtype=float)
    pre = tcp - a * float(approach_distance)
    up = tcp + np.array([0.0, 0.0, float(lift)])
    step = float(speed) * float(dt)
    positions, gripper = [], []

    def stay(p, seconds, g):
        n = max(int(round(float(seconds) / dt)), 1)
        positions.extend([p] * n)
        gripper.extend([g] * n)

    def move(p0, p1, g):
        n = max(int(np.ceil(np.linalg.norm(p1 - p0) / step)), 1)
        for s in np.arange(1, n + 1) / n:
            positions.append(p0 + (p1 - p0) * s)
            gripper.append(g)

    stay(pre, dwell, -1.0)
    move(pre, tcp, -1.0)
    stay(tcp, dwell, -1.0)
    stay(tcp, dwell, 1.0)                      # closed here
    move(tcp, up, 1.0)
    stay(up, hold_lifted, 1.0)
    move(up, tcp, 1.0)
    stay(tcp, dwell / 2.0, 1.0)                # set down, still closed
    stay(tcp, dwell, -1.0)                     # released
    move(tcp, pre, -1.0)
    P = np.asarray(positions, dtype=float)
    eye = np.repeat(np.eye(3)[None], len(P), axis=0)
    return PolicyLabels(
        positions=P,
        velocities=np.gradient(P, float(dt), axis=0),
        orientations=np.repeat(grasp.rotation[None], len(P), axis=0),
        stiffness=eye * float(stiffness),
        damping=eye * 2.0 * np.sqrt(float(stiffness)),
        gripper=np.asarray(gripper, dtype=float),
        # The phase, and its rate per second at the labels' own pace: the executor advances
        # the phase by it (without it the phase runs one whole test per second).
        time_belief=np.linspace(0.0, 1.0, len(P)),
        time_rate=np.full(len(P), 1.0 / ((len(P) - 1) * float(dt))),
        metadata={"control_freq": int(round(1.0 / dt)), "task": "grasp_test"},
    )


def _grasp_cube(grasp: GraspFrame, half: float = GRASP_CUBE_HALF_EXTENT) -> KeypointSet:
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    pts = np.vstack([grasp.tcp, grasp.tcp + corners @ grasp.rotation.T])
    return KeypointSet(points=pts, labels=["pick_centre"] + [f"pick_c{k}" for k in range(8)],
                       frame="base")


def grasp_test_plan(grasp: GraspFrame, points: np.ndarray, support_height: float,
                    end_effector=None, object_name: str = "", depth_offset: float = 0.0,
                    **label_kwargs) -> Plan:
    """The test of one grasp as a :class:`Plan` (module docstring).

    ``depth_offset``: the hand goes that much further along the approach than ``grasp``
    (m; a depth sweep), recorded in the metadata.
    """
    from tpgpt.hardware import hand as franka_hand

    ee = end_effector or FRANKA_HAND
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if depth_offset:
        grasp = replace(grasp, tcp=np.asarray(grasp.tcp) + np.asarray(grasp.approach)
                        * float(depth_offset))
    g = grasp_test_labels(grasp, **label_kwargs)
    labels = g.replace(orientations=ee.ee_from_grasp(g.orientations))
    gi, ri = carry_indices(labels)
    closed = width = None
    if ee.is_parallel_jaw:
        jaw = franka_hand.check_jaws(points, grasp.tcp, grasp.rotation)
        width = float(jaw.width)
    else:
        closed = ee.closed_state(points, grasp)
    support = float(support_height)
    centre = np.median(points[:, :2], axis=0)
    destination = np.array([centre[0], centre[1], support])
    horizontal = float(np.linalg.norm(points[:, :2] - centre, axis=1).max())
    pick, place = _hand_clearance(labels, gi, ri, width if closed is None else closed.q,
                                  support, support, ee, destination_xy=destination[:2],
                                  footprint_radius=horizontal + ee.footprint_margin)
    keypoints = _grasp_cube(grasp)
    dt = 1.0 / g.metadata["control_freq"]
    metrics = {
        "task": "grasp_test",
        "n_cloud_points": int(len(points)),
        "object_width_closing_m": 0.0 if width is None else width,
        "object_extent_closing_m": float(np.ptp(points @ np.asarray(grasp.closing))),
        "jaw_verdict": "not_applicable" if width is None else "checked",
        "jaw_points_between_pads": 0,
        "end_effector": ee.name,
        "grasp_tilt_deg": float(np.degrees(np.arccos(np.clip(-grasp.approach[2], -1.0, 1.0)))),
        "hand_clearance_pick_m": float(pick),
        "hand_clearance_place_m": float(place),
        "hand_closed_q": None if closed is None else closed.q.tolist(),
        "hand_contacts_predicted": None if closed is None else closed.contact,
        "hand_opposed_predicted": None if closed is None else bool(closed.opposed),
        "object_height_m": float(points[:, 2].max() - support),
        "support_height_m": support,
        "support_height_estimated": False,
        "min_det": 1.0,
        "fraction_positive": 1.0,
        "keypoint_residual_m": 0.0,
        "aim_grasp_m": 0.0,
        "aim_release_m": 0.0,
        "orientation_error_grasp_deg": 0.0,
        "orientation_error_release_deg": 0.0,
        "grasp_index": int(gi),
        "release_index": int(ri),
        "target_grasp_rolled": False,
        "roll": "none",
        "path_min_xyz": labels.positions.min(axis=0).tolist(),
        "path_max_xyz": labels.positions.max(axis=0).tolist(),
        "path_min_height_above_support_m": float(labels.positions[:, 2].min() - support),
        "max_label_speed_mps": float(np.linalg.norm(labels.velocities, axis=1).max()),
        "checked_path": "labels",
    }
    return Plan(
        labels=labels,
        source_keypoints=keypoints,
        target_keypoints=keypoints,
        grasp=grasp,
        target_points=points,
        support_height=support,
        destination=destination,
        destination_height=support,
        metrics=metrics,
        diagnostics={"target_place_points": points},
        metadata={
            "control_freq": int(round(1.0 / dt)),
            "frame": "base",
            "task": "grasp_test",
            "object_name": object_name,
            "position_reference": "the hand's grasp point",
            "orientation_convention": ee.controller_frame,
            "grasp_to_fr3_tcp": ee.grasp_to_ee.tolist(),
            "end_effector": ee.to_dict(),
            "source": "grasp_test",
            "source_stiffness": "static",
            "learned_execution": None,
            "grasp_test": {"depth_offset_m": float(depth_offset),
                           "approach_distance_m": label_kwargs.get("approach_distance",
                                                                   APPROACH_DISTANCE),
                           "lift_m": label_kwargs.get("lift", LIFT),
                           "speed_mps": label_kwargs.get("speed", SPEED)},
        },
    )


def select_grasp_test(points, poses, scores, support_height, settings=None, context=None,
                      path=None, end_effector=None, object_name: str = "",
                      depth_offset: float = 0.0, library_index: int | None = None):
    """The first screened GraspGen-X candidate whose test passes the whole-path check.

    ``library_index``: test only the candidate at this index of the generator's output
    (a depth sweep pins one grasp; never in an experiment). Returns a
    :class:`~tpgpt.hardware.grasp_selection.Selection`.

    Raises:
        PlanningError: nothing testable, with the tally of why.
    """
    import dataclasses

    from tpgpt.hardware.grasp_selection import (PathContext, ScreenContext, Selection,
                                                SelectionSettings, _path_verdict,
                                                rejection_counts, screen_candidates)

    ee = end_effector or FRANKA_HAND
    s = dataclasses.replace(settings or SelectionSettings(), check_place_zone=False)
    ctx = dataclasses.replace(context or ScreenContext(), destination=None,
                              destination_height=None)
    pc = dataclasses.replace(path or PathContext(), end_effector=ee)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    candidates = screen_candidates(points, poses, scores, float(support_height), s,
                                   context=ctx, end_effector=ee)
    notes = []
    if library_index is not None:
        pinned = [c for c in candidates if c.index == int(library_index)]
        if not pinned:
            raise PlanningError(f"no_executable_grasp: no candidate {library_index} in "
                                f"{len(candidates)}")
        if pinned[0].rejected is not None:
            notes.append(f"candidate {library_index} was screened out ({pinned[0].rejected}); "
                         "tested anyway, as pinned")
            pinned[0].rejected = None
        pool = pinned
    else:
        pool = [c for c in candidates if c.rejected is None][:max(int(s.max_plans), 0)]
    for rank, c in enumerate(pool, start=1):
        result = grasp_test_plan(c.grasp, points, support_height, ee, object_name, depth_offset)
        result.metrics["grasp_score"] = float(c.score)
        result.metrics["library_index"] = int(c.index)
        if not s.check_path:
            return Selection(result, c, candidates, "graspgenx", notes, candidates.stages, rank)
        reason, _ = _path_verdict(result, c, s, pc)
        if reason is None:
            return Selection(result, c, candidates, "graspgenx", notes, candidates.stages, rank)
        c.rejected = reason
    error = PlanningError(f"no_executable_grasp: {len(candidates)} candidates "
                          f"({rejection_counts(candidates)}); none testable. " + "; ".join(notes))
    error.candidates, error.fallback, error.stages = candidates, None, candidates.stages
    raise error


def constructed_grasp_test(points, support_height, settings=None, path=None, end_effector=None,
                           object_name: str = "", depth_offset: float = 0.0):
    """:func:`constructed_grasp`'s test, path-checked: a
    :class:`~tpgpt.hardware.grasp_selection.Selection` with ``source`` ``"constructed"``.

    The hand closes across the object's narrow side, then its long side, each both ways
    (:func:`constructed_closings`): for a jaw the two ways are one grasp, for the Inspire hand
    a half turn apart, and with it mounted as it is (yaw 137 deg) some need more turn than
    joint 7 has -- the arm then contorts (2026-10-06, the foam brick at 294 deg: closing +x
    needed joint 7 at its limit and the shoulder at +1.7 rad, links 5 and 7 touched; closing
    -x passed). Of those that pass the whole-path check, the one whose start posture is
    nearest the arm's joints (``path.start_q``) is taken: the executor moves the joints there
    and refuses a move over ``start_joint_max_move`` (0.5 rad).

    Raises:
        PlanningError: the whole-path check refused every one.
    """
    import dataclasses

    from tpgpt.hardware.grasp_selection import (Candidate, PathContext, Selection,
                                                SelectionSettings, _path_verdict, _tilt_deg)

    ee = end_effector or FRANKA_HAND
    s = settings or SelectionSettings()
    pc = dataclasses.replace(path or PathContext(), end_effector=ee)
    closings = constructed_closings(points)
    if ee.symmetric:
        closings = closings[::2]                   # a jaw: +-closing is the same grasp
    tried, best = [], None
    for name, closing in closings:
        grasp = constructed_grasp(points, support_height, ee, closing=closing)
        c = Candidate(index=-1, score=float("nan"), grasp=grasp, tilt_deg=_tilt_deg(grasp.approach))
        result = grasp_test_plan(grasp, points, support_height, ee, object_name, depth_offset)
        if not s.check_path:
            best = (0.0, name, result, c)
            break
        reason, _ = _path_verdict(result, c, s, pc)
        if reason is not None:
            c.rejected = reason
            tried.append(f"{name}: {reason}")
            continue
        move = 0.0
        postures = getattr(result, "checked_postures", None)
        if pc.start_q is not None and postures is not None and len(postures["q"]):
            first = np.asarray(postures["q"])[int(np.argmin(np.asarray(postures["labels"])))]
            move = float(np.max(np.abs(first - np.asarray(pc.start_q, dtype=float))))
        tried.append(f"{name}: passes, its start {move:.2f} rad (largest joint) from the arm")
        if best is None or move < best[0]:
            best = (move, name, result, c)
    if best is None:
        error = PlanningError("no_executable_grasp: the constructed grasp, every way: "
                              + "; ".join(tried))
        error.candidates, error.fallback, error.stages = [], None, []
        raise error
    move, name, result, c = best
    result.metrics["constructed_closing"] = name
    note = ("a grasp built by hand, for checking the frame chain; not the generator's. "
            f"Closing {name} (tried: " + "; ".join(tried) + ")")
    return Selection(result, c, [], "constructed", [note], [], 1)


def constructed_closings(points) -> list:
    """The horizontal directions a hand can close across an object lying on a support,
    from its footprint's principal axes:
    ``[("narrow +", c), ("narrow -", -c), ("long +", d), ("long -", -d)]``."""
    xy = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    _, _, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    narrow = np.array([axes[1][0], axes[1][1], 0.0])
    long_ = np.array([axes[0][0], axes[0][1], 0.0])
    narrow, long_ = narrow / np.linalg.norm(narrow), long_ / np.linalg.norm(long_)
    return [("narrow +", narrow), ("narrow -", -narrow), ("long +", long_), ("long -", -long_)]


def constructed_grasp(points: np.ndarray, support_height: float, end_effector=None,
                      clearance: float = 0.035, closing=None) -> GraspFrame:
    """A grasp built by hand: straight down onto the object's horizontal centre, closing
    across its narrow horizontal side (or along ``closing``, horizontal), the grasp point at
    half its height but at least
    ``clearance`` above the support (top-down, the closed Inspire hand reaches ~26 mm below
    its grasp point). For checking the frame chain: the hand must arrive as drawn."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    xy = points[:, :2]
    centre = xy.mean(axis=0)
    _, _, axes = np.linalg.svd(xy - centre, full_matrices=False)
    narrow = np.array([axes[1][0], axes[1][1], 0.0])
    local = (xy - centre) @ axes.T
    mid = (local.min(axis=0) + local.max(axis=0)) / 2.0
    c = centre + mid @ axes
    top = float(points[:, 2].max())
    z = max(float(support_height) + 0.5 * (top - float(support_height)),
            float(support_height) + float(clearance))
    if closing is not None:
        narrow = np.array([float(closing[0]), float(closing[1]), 0.0])
    return GraspFrame(tcp=np.array([c[0], c[1], z]), approach=np.array([0.0, 0.0, -1.0]),
                      closing=narrow / np.linalg.norm(narrow))


__all__ = ["constructed_grasp", "constructed_closings", "grasp_test_labels", "grasp_test_plan",
           "select_grasp_test", "constructed_grasp_test"]
