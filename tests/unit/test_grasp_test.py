"""The grasp-only test: one grasp, approached, closed, lifted, set down, released.

Built as an ordinary plan, so the executor runs it and the whole-path check judges it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.hardware.end_effector import end_effector_for_plan, inspire_end_effector
from tpgpt.hardware.executor import ExecutorSettings, HardwareExecutor
from tpgpt.hardware.grasp_selection import PathContext, SelectionSettings
from tpgpt.hardware.grasp_test import (APPROACH_DISTANCE, LIFT, constructed_grasp,
                                       grasp_test_labels, grasp_test_plan, select_grasp_test)
from tpgpt.hardware.planner import Plan, PlanningError
from tpgpt.hardware.shelf import ShelfWorld
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.sim.keypoints import GraspFrame, carry_indices

TABLE = -0.0132
RADIUS = 0.035
CENTRE = np.array([0.50, 0.0, TABLE + RADIUS])


@pytest.fixture(scope="module")
def inspire():
    try:
        return inspire_end_effector()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def ball(n=900, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return CENTRE + RADIUS * v


TOP_DOWN = GraspFrame(tcp=CENTRE.copy(), approach=np.array([0.0, 0.0, -1.0]),
                      closing=np.array([1.0, 0.0, 0.0]))


def _world():
    return ShelfWorld(model=None, cloud=np.zeros((0, 3)), table_height=TABLE)


def test_the_labels_approach_close_lift_set_down_release_and_retreat():
    L = grasp_test_labels(TOP_DOWN)
    gi, ri = carry_indices(L)
    assert L.positions[0] == pytest.approx(CENTRE + [0, 0, APPROACH_DISTANCE])   # above, open
    assert L.gripper[0] < 0 and L.gripper[-1] < 0
    assert L.positions[gi] == pytest.approx(CENTRE)                               # closes there
    assert L.positions[:, 2].max() == pytest.approx(CENTRE[2] + max(LIFT, APPROACH_DISTANCE))
    assert L.positions[ri] == pytest.approx(CENTRE)                               # set down there
    assert np.all(L.gripper[gi:ri + 1] > 0)
    assert np.allclose(L.orientations, TOP_DOWN.rotation)
    assert np.all(np.diff(L.time_belief) > 0)
    speeds = np.linalg.norm(np.diff(L.positions, axis=0), axis=1) / 0.05
    assert speeds.max() <= 0.06 + 1e-9


def test_a_ball_on_the_table_is_a_testable_grasp(inspire):
    plan = grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball")
    assert plan.metrics["hand_opposed_predicted"]
    assert plan.metrics["min_det"] == 1.0 and plan.metadata["task"] == "grasp_test"
    from tpgpt.hardware.path_check import check_plan, verdict
    check = check_plan(plan, _world(), inspire)
    assert verdict(check) is None, check["faults"]


def test_it_round_trips_and_the_executor_puts_the_grasp_point_on_the_labels(inspire, tmp_path):
    plan = grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball", depth_offset=0.004)
    plan.save(tmp_path / "p")
    loaded = Plan.load(tmp_path / "p")
    assert loaded.metadata["grasp_test"]["depth_offset_m"] == 0.004
    assert loaded.grasp.tcp == pytest.approx(CENTRE + [0.0, 0.0, -0.004])          # 4 mm deeper
    hand = end_effector_for_plan(loaded.metadata)
    ex = HardwareExecutor(GPPolicy().fit(loaded.labels), loaded.dt,
                          ExecutorSettings(tool_offset=tuple(hand.tool_offset)))
    tcp, R = ex.start_pose
    assert hand.grasp_point(tcp, R)[0] == pytest.approx(loaded.labels.positions[0], abs=1e-9)


def _pose(grasp, depth=0.15):
    T = np.eye(4)
    T[:3, :3] = grasp.rotation
    T[:3, 3] = grasp.tcp - grasp.approach * depth
    return T


def test_selection_takes_the_first_testable_candidate(inspire):
    # Candidate 0 (best score) puts the palm into the ball; candidate 1 is the good one.
    deep = GraspFrame(tcp=CENTRE - [0.0, 0.0, 0.06], approach=TOP_DOWN.approach,
                      closing=TOP_DOWN.closing)
    poses = np.stack([_pose(deep), _pose(TOP_DOWN)])
    sel = select_grasp_test(ball(), poses, np.array([0.9, 0.8]), TABLE,
                            SelectionSettings(max_tilt_deg=90.0, check_visibility=False),
                            path=PathContext(world=_world()), end_effector=inspire,
                            object_name="ball")
    assert sel.chosen.index == 1
    assert sel.plan.metrics["library_index"] == 1
    assert sel.candidates[0].rejected.startswith(("collides", "hits_support", "lands_on"))


def test_a_pinned_candidate_is_tested_even_when_screened_out(inspire):
    deep = GraspFrame(tcp=CENTRE - [0.0, 0.0, 0.06], approach=TOP_DOWN.approach,
                      closing=TOP_DOWN.closing)
    poses = np.stack([_pose(deep), _pose(TOP_DOWN)])
    settings = SelectionSettings(max_tilt_deg=90.0, check_visibility=False, check_path=False)
    sel = select_grasp_test(ball(), poses, np.array([0.9, 0.8]), TABLE, settings,
                            end_effector=inspire, library_index=0)
    assert sel.chosen.index == 0 and any("tested anyway" in n for n in sel.notes)
    with pytest.raises(PlanningError, match="no candidate 7"):
        select_grasp_test(ball(), poses, np.array([0.9, 0.8]), TABLE, settings,
                          end_effector=inspire, library_index=7)


def test_the_constructed_grasp_closes_across_the_narrow_side():
    rng = np.random.default_rng(1)
    box = rng.uniform([-0.044, -0.0175, 0.0], [0.044, 0.0175, 0.07], size=(2000, 3))
    box = box @ np.array([[np.cos(0.5), -np.sin(0.5), 0], [np.sin(0.5), np.cos(0.5), 0],
                          [0, 0, 1]]).T + [0.5, 0.1, TABLE]
    g = constructed_grasp(box, TABLE)
    assert g.approach == pytest.approx([0.0, 0.0, -1.0])
    narrow = np.array([-np.sin(0.5), np.cos(0.5), 0.0])
    assert abs(float(g.closing @ narrow)) == pytest.approx(1.0, abs=0.02)
    assert g.tcp[:2] == pytest.approx([0.5, 0.1], abs=0.003)
    assert g.tcp[2] == pytest.approx(TABLE + 0.035, abs=1e-9)      # 35 mm tall half > clearance


def test_the_constructed_test_is_path_checked(inspire):
    from tpgpt.hardware.grasp_test import constructed_grasp_test

    sel = constructed_grasp_test(ball(), TABLE, SelectionSettings(), PathContext(world=_world()),
                                 inspire, "ball")
    assert sel.source == "constructed" and sel.plan.metadata["task"] == "grasp_test"
    assert sel.chosen.path_check["verdict"] is None


def test_the_constructed_closings_are_both_sides_both_ways():
    from tpgpt.hardware.grasp_test import constructed_closings

    rng = np.random.default_rng(2)
    box = rng.uniform([-0.044, -0.0175, 0.0], [0.044, 0.0175, 0.07], size=(2000, 3)) + [0.5, 0.1, TABLE]
    names = [n for n, _ in constructed_closings(box)]
    dirs = np.array([d for _, d in constructed_closings(box)])
    assert names == ["narrow +", "narrow -", "long +", "long -"]
    assert abs(dirs[0] @ [0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=0.02)       # across 35 mm
    assert dirs[1] == pytest.approx(-dirs[0]) and dirs[3] == pytest.approx(-dirs[2])


def test_a_refused_way_falls_through_to_the_next(inspire, monkeypatch):
    """The live brick of 2026-10-06: closing one way needed joint 7 past its limit (the arm
    contorted, links 5 and 7 touched), the other way passed. Every way is tried."""
    from tpgpt.hardware import grasp_selection
    from tpgpt.hardware.grasp_test import constructed_grasp_test

    real = grasp_selection._path_verdict
    calls = []

    def refuse_first(result, c, s, pc):
        calls.append(c.grasp.closing.copy())
        if len(calls) == 1:
            return "collision: first at waypoint 0 (approach: self fr3_link5 <-> fr3_link7 x1)", False
        return real(result, c, s, pc)

    monkeypatch.setattr(grasp_selection, "_path_verdict", refuse_first)
    sel = constructed_grasp_test(ball(), TABLE, SelectionSettings(), PathContext(world=_world()),
                                 inspire, "ball")
    assert sel.plan.metrics["constructed_closing"] != "narrow +"
    assert "narrow +: collision" in sel.notes[0]
    assert len(calls) == 4                                          # every way judged
    monkeypatch.setattr(grasp_selection, "_path_verdict", lambda *a: ("refused", False))
    with pytest.raises(PlanningError, match="every way"):
        constructed_grasp_test(ball(), TABLE, SelectionSettings(), PathContext(world=_world()),
                               inspire, "ball")


def test_the_labels_carry_a_taught_demonstrations_stiffness():
    """As the Franka Hand's transport runs: 600 N/m in every direction while still, 350
    along the motion and 600 across it while moving (teach.directional_stiffness)."""
    L = grasp_test_labels(TOP_DOWN)
    gi, _ = carry_indices(L)
    assert L.stiffness[0] == pytest.approx(600.0 * np.eye(3), abs=1.0)       # still, above
    assert L.stiffness[gi] == pytest.approx(600.0 * np.eye(3), abs=1.0)      # still, closing
    speed = np.linalg.norm(L.velocities, axis=1)
    moving = np.flatnonzero(speed[:gi] > 0.5 * speed[:gi].max())
    k = int(moving[len(moving) // 2])                                       # mid-descent
    assert L.stiffness[k][2, 2] == pytest.approx(350.0, abs=5.0)
    assert L.stiffness[k][0, 0] == pytest.approx(600.0, abs=5.0)
    w, v = np.linalg.eigh(L.stiffness[k])
    assert L.damping[k] == pytest.approx(v @ np.diag(2.0 * np.sqrt(w)) @ v.T)


def test_the_plan_carries_the_transport_execution_settings(inspire):
    execution = {"speed_scale": 0.35, "rotational_stiffness": 30.0,
                 "transit_rotational_scale": 1.0, "use_transported_stiffness": True,
                 "offset_compensation": "none", "arrival_hold": False}
    plan = grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball", execution=execution)
    assert plan.metadata["learned_execution"] == execution
    assert grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball").metadata[
        "learned_execution"] is None


def test_the_phase_advances_at_the_labels_pace():
    L = grasp_test_labels(TOP_DOWN)
    assert L.time_rate * 0.05 * (len(L) - 1) == pytest.approx(np.ones(len(L)))


def test_the_plan_is_followed_against_its_own_path(inspire, tmp_path):
    """No lead, no margin: the arm path is the labels, the marks the grasp and the set-down."""
    plan = grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball")
    plan.save(tmp_path / "p")
    loaded = Plan.load(tmp_path / "p")
    gi, ri = loaded.metrics["grasp_index"], loaded.metrics["release_index"]
    assert loaded.arm_path.positions == pytest.approx(loaded.labels.positions)
    assert loaded.arm_path.orientations == pytest.approx(loaded.labels.orientations)
    assert np.all(loaded.arm_margin == 0.0) and np.all(loaded.arm_table_margin == 0.0)
    assert loaded.marks["positions"] == pytest.approx(loaded.labels.positions[[gi, ri]])
    assert loaded.marks["orientations"] == pytest.approx(loaded.labels.orientations[[gi, ri]])
    assert loaded.metadata["learned_execution"] is None          # not a learned plan
    # and the path check judges the same path as before (zero clearance = none)
    from tpgpt.hardware.path_check import check_plan, verdict
    assert verdict(check_plan(loaded, _world(), inspire)) is None


def _sticky_run(plan, hand, follow: bool, static_force=3.0, tau=0.3, n_steps=4000):
    """The test run on an arm that does not move until the spring (the labels' stiffness)
    pulls harder than ``static_force`` (N), then closes the rest at ``tau`` (s) -- the
    2026-10-06 17:50 arm, which a 2.4 N pull (6 mm at 400 N/m) did not move."""
    from tpgpt.hardware.ilc import frozen_mask
    from tpgpt.hardware.executor import ArmPathSettings
    from tpgpt.hardware.replay import contact_windows

    m = plan.metrics
    arm = None
    if follow:
        arm = {"positions": plan.arm_path.positions, "margin": plan.arm_margin,
               "contact": frozen_mask(len(plan.labels), contact_windows(int(m["grasp_index"]),
                                                                        int(m["release_index"])))}
    ex = HardwareExecutor(GPPolicy().fit(plan.labels), plan.dt,
                          ExecutorSettings(speed_scale=0.35, lag_tolerance=0.045,
                                           tool_offset=tuple(hand.tool_offset)),
                          arm_path=arm, arm_settings=ArmPathSettings() if follow else None)
    x, R = ex.start_pose
    lowest = np.inf
    for _ in range(n_steps):
        cmd = ex.step(x, R)
        gap = cmd.tcp_position - x
        pull = float(np.linalg.norm(cmd.stiffness @ gap))
        if pull > static_force:
            x = x + gap * (1.0 - static_force / pull) * plan.dt / tau
        lowest = min(lowest, float(hand.grasp_point(x, R)[0][2]))
        if ex.finished:
            break
    return ex, lowest


def test_an_arm_held_by_static_friction_is_pushed_down_to_the_grasp(inspire):
    """The simulation's clamp keeps the command ~6 mm from the arm, which this arm does not
    feel: under it the arm never leaves the pre-grasp. Followed, the held clock hands the gap
    to the correction until the pull breaks the arm free, and it reaches the grasp."""
    plan = grasp_test_plan(TOP_DOWN, ball(), TABLE, inspire, "ball")
    top = float(plan.labels.positions[0][2])
    grasp = float(plan.grasp.tcp[2])
    gated, lowest = _sticky_run(plan, inspire, follow=False, n_steps=1500)
    assert lowest > top - 0.005                                  # never left the pre-grasp
    assert gated.stalled and gated.phase < 0.2                   # the watchdog, early on
    followed, lowest = _sticky_run(plan, inspire, follow=True)
    assert lowest < grasp + 0.005                                # down to the grasp
    assert followed.finished and not followed.blocked
