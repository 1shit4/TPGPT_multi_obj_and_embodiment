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
    assert len(world.cloud) == len(thing)
    r = _check(positions, rotations, world)
    assert any("in the shelf cloud" in k for k in r["faults"])


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
