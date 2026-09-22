"""The whole-path check: the simulation's rules, the real arm, the shelf world."""

import numpy as np
import pytest

from tpgpt.hardware import path_check as pc
from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.fr3 import HOME_Q
from tpgpt.hardware.shelf import ShelfModel, ShelfWorld

#: fr3_hand_tcp pointing straight down, fingers closing along base y.
DOWN = np.diag([1.0, -1.0, -1.0])


def _path(start, end, n=60):
    s = np.linspace(0.0, 1.0, n)[:, None]
    positions = np.asarray(start) + s * (np.asarray(end) - np.asarray(start))
    return positions, np.repeat(DOWN[None], n, axis=0)


def _shelf(board_top: float, x0: float = 0.35, depth: float = 0.3):
    """A single board spanning y in [-0.25, 0.25], x in [x0, x0 + depth]."""
    return ShelfModel.from_config({"shelf": {
        "measured": True, "front_left": [x0, 0.25], "front_right": [x0, -0.25],
        "depth": depth, "bottom": 0.0, "top": board_top, "wall_thickness": 0.0,
        "back_thickness": 0.0,
        "boards": [{"name": "board", "top": board_top, "thickness": 0.02}],
    }})


def _check(positions, rotations, world, grasp=20, release=40, **kw):
    gripper = np.where((np.arange(len(positions)) >= grasp)
                       & (np.arange(len(positions)) <= release), 1.0, -1.0)
    kw.setdefault("object_points", np.zeros((0, 3)))
    return pc.path_feasibility(positions, rotations, gripper, FRANKA_HAND, world,
                               grasp_index=grasp, release_index=release, closed_width=0.04,
                               stride=2, **kw)


def test_constants_are_the_simulations():
    from tpgpt.experiments.pipeline import MIN_REACHABLE_FRACTION
    from tpgpt.grasp import filters

    assert pc.REACH_TOLERANCE == filters.REACH_TOLERANCE
    assert pc.CONTACT_DEPTH_TOLERANCE == filters.CONTACT_DEPTH_TOLERANCE
    assert pc.CLOUD_CONTACT_RADIUS == filters.CLOUD_CONTACT_RADIUS
    assert pc.GRASP_CONTACT_WINDOW == filters.GRASP_CONTACT_WINDOW
    assert pc.MIN_REACHABLE_FRACTION == MIN_REACHABLE_FRACTION


def test_a_clear_path_is_admissible():
    positions, rotations = _path([0.45, 0.0, 0.45], [0.45, 0.0, 0.30])
    r = _check(positions, rotations, ShelfWorld.build(_shelf(0.10)))
    assert r["reachable_fraction"] == 1.0 and r["violations"] == 0
    assert pc.verdict(r) is None


def test_a_path_through_a_board_is_a_collision_named_by_panel():
    positions, rotations = _path([0.45, 0.0, 0.45], [0.45, 0.0, 0.15])
    r = _check(positions, rotations, ShelfWorld.build(_shelf(0.25)))
    assert r["violations"] > 0
    assert any("in shelf board" in k for k in r["faults"])
    assert pc.verdict(r).startswith("collision")


def test_a_path_out_of_reach_is_a_kinematic_failure():
    positions, rotations = _path([1.2, 0.0, 0.3], [1.3, 0.0, 0.3])
    r = _check(positions, rotations, ShelfWorld.build(None))
    assert r["reachable_fraction"] < pc.MIN_REACHABLE_FRACTION
    assert pc.verdict(r).startswith("kinematics")


def test_stop_early_decides_the_same_way():
    positions, rotations = _path([1.2, 0.0, 0.3], [1.3, 0.0, 0.3])
    world = ShelfWorld.build(None)
    full, early = _check(positions, rotations, world), _check(positions, rotations, world,
                                                              stop_early=True)
    assert pc.verdict(full).split(":")[0] == pc.verdict(early).split(":")[0] == "kinematics"
    assert early["decided_early"]


def test_the_carried_object_may_rest_on_a_board_but_not_sink_into_it():
    board = _shelf(0.20)
    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.30])
    # A 4 cm cube held 5 cm below the TCP: its bottom ends 1 mm above the board.
    g = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01, -0.02:0.021:0.01].reshape(3, -1).T
    grasp = 5
    below = positions[grasp] + np.array([0.0, 0.0, -0.079]) + g
    resting = _check(positions, rotations, ShelfWorld.build(board), grasp=grasp, release=59,
                     object_points=below)
    assert not any("carried object" in k for k in resting["faults"])
    sunk = _check(positions, rotations, ShelfWorld.build(board), grasp=grasp, release=59,
                  object_points=below + [0.0, 0.0, -0.02])
    assert any("carried object in shelf board" in k for k in sunk["faults"])


def test_the_observed_cloud_on_the_shelf_is_an_obstacle():
    shelf = _shelf(0.10)
    positions, rotations = _path([0.45, 0.0, 0.45], [0.45, 0.0, 0.20])
    # Something standing on the board, 12 cm tall and 12 cm across the closing
    # axis, right under the path -- too wide to pass between the open fingers.
    thing = np.array([[0.45 + dx, dy, z] for dx in (-0.02, 0.0, 0.02)
                      for dy in np.arange(-0.06, 0.061, 0.01)
                      for z in np.arange(0.11, 0.23, 0.01)])
    world = ShelfWorld.build(shelf, scene_points=thing)
    # Its bottom layer, within 15 mm of the board it stands on, is taken as the
    # board (the model covers it); the rest of it stays an obstacle.
    bottom = thing[:, 2] <= 0.10 + 0.015
    assert world.cloud_report["removed_as_shelf"] == int(bottom.sum())
    assert len(world.cloud) == int((~bottom).sum())
    r = _check(positions, rotations, world)
    assert any("in the seen points" in k for k in r["faults"])


def test_the_move_onto_the_path_is_checked_from_the_measured_joints():
    # The path itself is clear, above a board at 0.25; getting there from a TCP
    # below the board means passing through it.
    shelf = _shelf(0.25)
    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.32])
    from tpgpt.hardware.fr3 import default_model

    start = default_model().solve_ik(np.array([0.45, 0.0, 0.18]), DOWN, seed=HOME_Q,
                                     max_iterations=400)
    assert start.reachable
    r = _check(positions, rotations, ShelfWorld.build(shelf), start_q=start.qpos)
    assert r["first_violation"] == -1
    assert r["approach_in"]["faults"]
    assert "onto the path" in pc.verdict(r)


def test_verdict_ranks_collision_before_kinematics():
    r = {"violations": 1, "first_violation": 7, "faults": {"carry: x": 1},
         "reachable_fraction": 0.1}
    assert pc.verdict(r).startswith("collision")


@pytest.mark.parametrize("fraction", [0.59, 0.61])
def test_the_reachability_threshold(fraction):
    r = {"violations": 0, "first_violation": None, "faults": {}, "reachable_fraction": fraction}
    assert (pc.verdict(r) is None) == (fraction >= 0.60)


# ---------------------------------------------------------------- the arm itself
#: A within-limits pose that folds the forearm into the base (link 5 104 mm deep
#: in link 0), found by random search over the joint limits.
FOLDED = np.array([1.958, 1.518, -0.328, -2.969, 0.372, 4.022, 2.689])


def test_penetration_is_exact_for_boxes():
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    assert pc.penetration(cube, cube + [0.9, 0, 0]) == pytest.approx(0.1)
    assert pc.penetration(cube, cube + [1.2, 0, 0]) < 0


def test_the_real_arm_at_rest_never_collides_with_itself():
    """40 at-rest poses of the real arm (robot-only data) and home: no self fault.

    The pairs that overlap by construction (joint neighbours, the hand's own
    pieces) are excluded; this is what shows the exclusion is enough.
    """
    from pathlib import Path

    from tpgpt.hardware.fr3 import default_model

    fr3 = default_model()
    data = np.load(Path(__file__).resolve().parents[1] / "data" / "fr3_still_samples.npz")
    for q in [*data["joint_positions"], HOME_Q]:
        pieces, _ = pc.robot_pieces(fr3, q, FRANKA_HAND, 0.08)
        assert pc._self_faults(pieces, "test", pc.CONTACT_DEPTH_TOLERANCE) == {}


def test_a_folded_arm_is_a_self_collision():
    from tpgpt.hardware.fr3 import default_model

    pieces, _ = pc.robot_pieces(default_model(), FOLDED, FRANKA_HAND, 0.08)
    faults = pc._self_faults(pieces, "carry", pc.CONTACT_DEPTH_TOLERANCE)
    assert "carry: self fr3_link0 <-> fr3_link5" in faults


def test_joint_neighbours_are_not_faults():
    assert pc._joined("fr3_link3", "fr3_link4") and pc._joined("fr3_link7", "hand_body")
    assert pc._joined("hand_body", "hand_left_pad")
    assert not pc._joined("fr3_link0", "fr3_link5") and not pc._joined("fr3_link5", "hand_body")


# --------------------------------------------------------- the table, exactly
def test_the_arm_going_below_the_table_is_a_collision():
    positions, rotations = _path([0.45, 0.0, 0.30], [0.45, 0.0, 0.05])
    world = ShelfWorld.build(None, support_height=0.12)             # a table at 12 cm
    r = _check(positions, rotations, world, grasp=59, release=59)
    assert any("in the table" in k for k in r["faults"])


def test_the_carried_object_may_rest_on_the_table_but_not_be_pushed_into_it():
    positions, rotations = _path([0.45, 0.0, 0.20], [0.45, 0.0, 0.30])
    g = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01, 0.0:0.041:0.01].reshape(3, -1).T
    cube = positions[5] + np.array([0.0, 0.0, -0.10]) + g            # bottom at z = 0.10
    resting = _check(positions, rotations, ShelfWorld.build(None, support_height=0.103),
                     grasp=5, release=59, object_points=cube)
    assert not any("carried object" in k for k in resting["faults"])
    pushed = _check(positions, rotations, ShelfWorld.build(None, support_height=0.12),
                    grasp=5, release=59, object_points=cube)
    assert any("carried object in the table" in k for k in pushed["faults"])


def test_the_move_onto_the_path_must_not_go_through_the_object_to_be_picked():
    from tpgpt.hardware.fr3 import default_model

    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.32])
    start = default_model().solve_ik(np.array([0.45, 0.0, 0.10]), DOWN, seed=HOME_Q,
                                     max_iterations=400)
    assert start.reachable
    # The object stands between the arm's start (TCP at 10 cm) and the path (above 32 cm).
    obj = np.array([[0.45 + dx, dy, z] for dx in (-0.04, 0.0, 0.04) for dy in (-0.06, 0.0, 0.06)
                    for z in np.arange(0.20, 0.26, 0.01)])
    r = _check(positions, rotations, ShelfWorld.build(None), grasp=40, release=50,
               object_points=obj, start_q=start.qpos)
    assert r["first_violation"] == -1
    assert any("approach_in" in k and "target's cloud" in k for k in r["faults"])


# ------------------------------------ where the arm cannot hold the planned pose
#: fr3_hand_tcp pointing back at the robot base: the position is reachable at
#: (0.45, 0, 0.35-0.45), this orientation is not (IK stops 24.5 mm short).
BACK = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_an_unreachable_orientation_is_checked_where_the_executor_relaxes_it():
    positions, _ = _path([0.45, 0.0, 0.45], [0.45, 0.0, 0.35])
    rotations = np.repeat(BACK[None], len(positions), axis=0)
    r = _check(positions, rotations, ShelfWorld.build(None))
    assert r["orientation_relaxed"] > 0
    # The relaxed postures are checked, not skipped: here IK folds the hand back
    # onto link 5 to point it at the base, which the self check reports.
    assert any("(relaxed)" in k for k in r["faults"])


def test_a_stretch_out_of_reach_is_checked_along_the_line_the_executor_steers():
    a = np.linspace([0.45, -0.10, 0.30], [0.45, -0.10, 0.30], 10)
    out = np.linspace([1.40, 0.0, 0.30], [1.40, 0.0, 0.30], 20)      # far out of reach
    b = np.linspace([0.45, 0.10, 0.30], [0.45, 0.10, 0.30], 30)
    positions = np.vstack([a, out, b])
    rotations = np.repeat(DOWN[None], len(positions), axis=0)
    clear = _check(positions, rotations, ShelfWorld.build(None), grasp=55, release=58)
    assert clear["skipped"] > 0 and clear["violations"] == 0
    # A thin wall standing between y = -0.10 and y = +0.10, where the arm is
    # steered straight across once the out-of-reach stretch is skipped.
    wall = ShelfModel.from_config({"shelf": {
        "measured": True, "front_left": [0.30, 0.01], "front_right": [0.60, 0.01],
        "depth": 0.01, "bottom": 0.0, "top": 0.40, "wall_thickness": 0.0,
        "boards": [{"name": "wall", "top": 0.40, "thickness": 0.40, "support": False}]}})
    r = _check(positions, rotations, ShelfWorld.build(wall), grasp=55, release=58)
    assert any("(steered)" in k for k in r["faults"])
