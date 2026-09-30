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
    sim = _check(positions, rotations, ShelfWorld.build(None), criterion="sim")
    assert sim["reachable_fraction"] < pc.MIN_REACHABLE_FRACTION
    assert pc.verdict(sim).startswith("kinematics")
    strict = _check(positions, rotations, ShelfWorld.build(None), stop_early=True)
    assert strict["first_kinematic"] == 0 and pc.verdict(strict).startswith("kinematics")


def test_stop_early_decides_the_same_way():
    positions, rotations = _path([1.2, 0.0, 0.3], [1.3, 0.0, 0.3])
    world = ShelfWorld.build(None)
    full = _check(positions, rotations, world, criterion="sim")
    early = _check(positions, rotations, world, stop_early=True, criterion="sim")
    assert pc.verdict(full).split(":")[0] == pc.verdict(early).split(":")[0] == "kinematics"
    assert early["decided_early"]


def _cube_set_down(depth_into_board: float):
    """A 4 cm cube held below the TCP, lowered onto a board at 0.20 so that its
    base ends ``depth_into_board`` inside the board's top face."""
    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.30])
    g = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01, -0.02:0.021:0.01].reshape(3, -1).T
    grasp = 5
    # Rigid with the hand and the hand does not turn, so the cube keeps its
    # offset from the TCP: choose it so the base ends at 0.20 - depth.
    dz = (0.20 - depth_into_board + 0.02) - positions[-1][2]
    cube = positions[grasp] + g + [0.0, 0.0, dz]
    return positions, rotations, grasp, cube


@pytest.mark.parametrize("depth, destination, expect", [
    (0.003, 0.20, None),                          # its base, 3 mm into the destination board
    (0.003, 0.35, "touches board"),               # the same contact, but not the destination
    (0.019, 0.20, "pressed into board"),          # 19 mm in: pushed through it
])
def test_the_object_is_set_down_only_on_its_base_on_the_destination(depth, destination, expect):
    positions, rotations, grasp, cube = _cube_set_down(depth)
    r = _check(positions, rotations, ShelfWorld.build(_shelf(0.20)), grasp=grasp, release=59,
               object_points=cube, destination_height=destination)
    carried = [k for k in r["faults"] if "carried object" in k]
    if expect is None:
        assert carried == []
    else:
        assert any(expect in k for k in carried), carried


def test_an_object_turned_onto_its_side_is_not_set_down_on_its_base():
    """Which points may touch is decided by the object's own base face (the points
    it stood on at the pick), wherever the carry has turned it -- not by a cone
    around the vertical."""
    board = _shelf(0.20)
    g = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01, -0.02:0.021:0.01].reshape(3, -1).T
    base = g[:, 2] <= g[:, 2].min() + pc.BASE_BAND
    world = ShelfWorld.build(board)
    upright = g + [0.45, 0.0, 0.20 + 0.02 - 0.003]
    kinds = {k for k, _, _ in pc._object_contacts(upright, base, world, board, None, 0.001, 0.005)}
    assert kinds == {"base_rest"}
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # onto its side
    on_side = g @ Rx.T + [0.45, 0.0, 0.20 + 0.02 - 0.003]
    kinds = {k for k, _, _ in pc._object_contacts(on_side, base, world, board, None, 0.001, 0.005)}
    # The base now stands vertical; its bottom edge still touches, but so does a
    # face that is not the base -- which is the fault.
    assert "rest" in kinds


def test_the_observed_cloud_on_the_shelf_is_an_obstacle():
    shelf = _shelf(0.10)
    positions, rotations = _path([0.45, 0.0, 0.45], [0.45, 0.0, 0.20])
    # Something standing on the board, 12 cm tall and 12 cm across the closing
    # axis, right under the path -- too wide to pass between the open fingers.
    # Sampled every 5 mm, as the planner's own cloud is (3 mm voxels): a contact
    # needs MIN_CLOUD_CONTACT_POINTS points CLOUD_CONTACT_SLACK inside a piece.
    thing = np.array([[0.45 + dx, dy, z] for dx in np.arange(-0.02, 0.021, 0.005)
                      for dy in np.arange(-0.06, 0.061, 0.005)
                      for z in np.arange(0.11, 0.23, 0.005)])
    world = ShelfWorld.build(shelf, scene_points=thing)
    # Its bottom layer, within 15 mm of the board it stands on, is taken as the
    # board (the model covers it); the rest of it stays an obstacle.
    bottom = shelf.explained(thing)          # the model's own 15 mm rule, not a hand copy of it
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


def test_a_blocked_move_onto_the_path_is_reported_but_does_not_reject_the_path():
    """approach_rejects=False (user, 2026-09-29): the parked pose is not the grasp's fault."""
    shelf = _shelf(0.25)
    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.32])
    from tpgpt.hardware.fr3 import default_model

    start = default_model().solve_ik(np.array([0.45, 0.0, 0.18]), DOWN, seed=HOME_Q,
                                     max_iterations=400)
    r = _check(positions, rotations, ShelfWorld.build(shelf), start_q=start.qpos,
               approach_rejects=False)
    assert r["approach_in"]["faults"]                  # still checked, still reported
    assert r["approach_in"]["rejects"] is False
    assert r["violations"] == 0 and pc.verdict(r) is None
    assert not any(k.startswith("approach_in") for k in r["faults"])


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
    assert any("carried object pressed into table" in k for k in pushed["faults"])


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
    assert any("approach_in" in k and "object to be picked" in k for k in r["faults"])


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
    clear = _check(positions, rotations, ShelfWorld.build(None), grasp=55, release=58,
                   criterion="sim")
    assert clear["skipped"] > 0 and clear["violations"] == 0
    # A thin wall standing between y = -0.10 and y = +0.10, where the arm is
    # steered straight across once the out-of-reach stretch is skipped.
    wall = ShelfModel.from_config({"shelf": {
        "measured": True, "front_left": [0.30, 0.01], "front_right": [0.60, 0.01],
        "depth": 0.01, "bottom": 0.0, "top": 0.40, "wall_thickness": 0.0,
        "boards": [{"name": "wall", "top": 0.40, "thickness": 0.40, "support": False}]}})
    r = _check(positions, rotations, ShelfWorld.build(wall), grasp=55, release=58,
               criterion="sim")
    assert any("(steered)" in k for k in r["faults"])
    # The strict criterion does not skip: a position out of reach rejects the path.
    strict = _check(positions, rotations, ShelfWorld.build(None), grasp=55, release=58,
                    stop_early=True)
    assert pc.verdict(strict).startswith("kinematics") and "position" in pc.verdict(strict)


# ------------------------------------------------------------ the stage rules
def _cube(centre, half=0.02, step=0.01):
    g = np.mgrid[-half:half + 1e-9:step, -half:half + 1e-9:step, -half:half + 1e-9:step]
    return np.asarray(centre) + g.reshape(3, -1).T


def test_the_object_may_touch_the_table_only_until_it_lifts_off_or_as_its_destination():
    up = np.linspace([0.45, 0.0, 0.20], [0.45, 0.0, 0.30], 30)
    down = np.linspace([0.45, 0.0, 0.30], [0.45, 0.0, 0.20], 30)
    positions = np.vstack([up, down])
    rotations = np.repeat(DOWN[None], len(positions), axis=0)
    cube = _cube([0.45, 0.0, 0.12])                   # base at 0.10, 3 mm into a table at 0.103
    world = ShelfWorld.build(None, support_height=0.103)
    # Lifted, carried, and brought back down onto the table it is NOT meant for.
    back = _check(positions, rotations, world, grasp=0, release=59, object_points=cube,
                  destination_height=0.30)
    assert any("carried object touches table" in k for k in back["faults"])
    # The same motion when the table IS the destination: set down on its base.
    placed = _check(positions, rotations, world, grasp=0, release=59, object_points=cube,
                    destination_height=0.103)
    assert not [k for k in placed["faults"] if "carried object" in k]


def test_the_held_shift_moves_the_object_when_the_jaws_close():
    up = np.linspace([0.45, 0.0, 0.20], [0.45, 0.0, 0.30], 30)
    down = np.linspace([0.45, 0.0, 0.30], [0.45, 0.0, 0.20], 30)
    positions = np.vstack([up, down])
    rotations = np.repeat(DOWN[None], len(positions), axis=0)
    cube = _cube([0.45, 0.0, 0.12])                   # base 3 mm into a table at 0.103
    world = ShelfWorld.build(None, support_height=0.103)
    kw = dict(grasp=0, release=59, object_points=cube, destination_height=0.30)
    assert any("carried object touches table" in k for k in _check(positions, rotations, world,
                                                                    **kw)["faults"])
    # held 10 mm higher in the hand, it comes back down 10 mm above the table
    lifted = _check(positions, rotations, world, held_shift=[0.0, 0.0, 0.010], **kw)
    assert not [k for k in lifted["faults"] if "carried object" in k]


def test_at_the_pick_the_fingers_may_touch_the_object_and_the_palm_may_not():
    positions, rotations = _path([0.45, 0.0, 0.30], [0.45, 0.0, 0.30], n=30)   # hand holds still
    fr3 = __import__("tpgpt.hardware.fr3", fromlist=["default_model"]).default_model()
    # A slab across the closing axis, level with the pads: the closed fingers press on it.
    slab = np.array([[0.45 + dx, dy, 0.30 - 0.008] for dx in (-0.005, 0.0, 0.005)
                     for dy in np.arange(-0.05, 0.051, 0.004)])
    r = _check(positions, rotations, ShelfWorld.build(None), grasp=20, release=29,
               object_points=slab)
    assert not any(k.startswith("pick:") for k in r["faults"])     # fingers at the pick: fine
    # Something reaching up into the palm at the pick is a fault.
    palm = np.array([[0.45, 0.0, 0.30 + z] for z in np.arange(0.07, 0.12, 0.005)])
    r = _check(positions, rotations, ShelfWorld.build(None), grasp=20, release=29,
               object_points=np.vstack([slab, palm]))
    assert any(k.startswith("pick:") and "hand_body" in k for k in r["faults"])


def test_in_the_retreat_the_fingers_may_not_go_back_into_the_placed_object():
    hold = np.linspace([0.45, 0.0, 0.30], [0.45, 0.0, 0.30], 20)
    rise = np.linspace([0.45, 0.0, 0.30], [0.45, 0.0, 0.40], 12)
    dive = np.linspace([0.45, 0.0, 0.40], [0.45, 0.0, 0.25], 16)    # back down, into it
    positions = np.vstack([hold, rise, dive])
    rotations = np.repeat(DOWN[None], len(positions), axis=0)
    # A block 100 mm across the closing axis, just below the fingertips at the
    # release: wider than the open fingers (80 mm), so going back down hits it.
    cube = np.array([[0.45 + dx, dy, z] for dx in np.arange(-0.02, 0.021, 0.005)
                     for dy in np.arange(-0.05, 0.051, 0.005)
                     for z in np.arange(0.24, 0.281, 0.005)])
    clean = _check(positions[:32], rotations[:32], ShelfWorld.build(None), grasp=2, release=19,
                   object_points=cube)
    assert not any("placed object" in k for k in clean["faults"])
    r = _check(positions, rotations, ShelfWorld.build(None), grasp=2, release=19,
               object_points=cube)
    assert any("retreat" in k and "placed object" in k for k in r["faults"])


def test_the_face_set_down_is_the_one_down_at_the_release_not_at_the_pick():
    """A 20 x 40 x 60 mm block picked standing on its 20 x 40 end, turned by the
    hand during the carry, and set down on a long side: the face that touches is
    the one lowest in the planned release pose, so this is a correct placement."""
    from scipy.spatial.transform import Rotation, Slerp

    g, rise, over, down = 5, 20, 20, 15
    n = g + 1 + rise + over + down
    r = n - 1
    side = Rotation.from_euler("x", -90, degrees=True).as_matrix() @ DOWN
    turn = Slerp([0, 1], Rotation.from_matrix([DOWN, side]))(np.linspace(0, 1, rise)).as_matrix()
    rotations = np.vstack([np.repeat(DOWN[None], g + 1, axis=0), turn,
                           np.repeat(side[None], over + down, axis=0)])
    # Picked in front of the shelf (its front edge is at x = 0.35), lifted and
    # turned, carried over the board, lowered onto it.
    positions = np.vstack([np.linspace([0.28, 0.10, 0.12], [0.28, 0.10, 0.12], g + 1),
                           np.linspace([0.28, 0.10, 0.12], [0.28, 0.0, 0.40], rise),
                           np.linspace([0.28, 0.0, 0.40], [0.45, 0.0, 0.40], over),
                           np.linspace([0.45, 0.0, 0.40], [0.45, 0.0, 0.30], down)])
    block = np.mgrid[-0.01:0.0101:0.005, -0.02:0.0201:0.005, -0.03:0.0301:0.005].reshape(3, -1).T
    obj = positions[g] + block + [0.0, 0.0, -0.035]              # standing, below the fingertips
    # Where it ends up, and how far to shift it so its lowest face is 3 mm into the board.
    local = (obj - positions[g]) @ rotations[g]
    final = positions[r] + local @ rotations[r].T
    obj = obj + [0.0, 0.0, (0.20 - 0.003) - final[:, 2].min()]
    world = ShelfWorld.build(_shelf(0.20, x0=0.35))
    res = _check(positions, rotations, world, grasp=g, release=r, object_points=obj,
                 destination_height=0.20)
    assert res["skipped"] == 0 and res["relaxed_in_pick_or_place"] == 0
    assert not [k for k in res["faults"] if "carried object" in k], res["faults"]


# --- clearance: a gap to keep, for a path the arm follows only approximately ------------
TABLE = -0.0132                    # the touched table, below the robot's base (2026-09-26)


def _lowest_hand_point(position, rotation=DOWN):
    """The lowest point of the open hand with its grasp point at ``position``."""
    from tpgpt.hardware.fr3 import default_model
    from tpgpt.hardware.hand import MAX_OPENING

    fr3 = default_model()
    result = fr3.solve_ik(np.asarray(position), rotation, seed=HOME_Q)
    assert result.reachable
    pieces, _ = pc.robot_pieces(fr3, result.qpos, FRANKA_HAND, MAX_OPENING)
    return min(float(piece.vertices[:, 2].min()) for piece in pieces
               if piece.name.startswith("hand_"))


def _down_to(height_above, surface):
    """A descent that ends with the hand's lowest point ``height_above`` a surface.

    61 waypoints, so the check's stride of 2 tests the last one.
    """
    below = 0.10 - _lowest_hand_point([0.45, 0.0, 0.10])        # grasp point above the hand's bottom
    return _path([0.45, 0.0, 0.30], [0.45, 0.0, surface + height_above + below], n=61)


def test_a_clearance_asks_for_a_gap_to_the_table():
    positions, rotations = _down_to(0.010, TABLE)
    world = ShelfWorld.build(None, support_height=TABLE)
    kw = dict(grasp=60, release=60)
    assert not _check(positions, rotations, world, **kw)["faults"]
    assert not _check(positions, rotations, world, clearance=0.008, **kw)["faults"]
    r = _check(positions, rotations, world, clearance=0.012, **kw)
    assert set(r["faults"]) == {"pick: hand_left_pad in the table",
                                "pick: hand_right_pad in the table"}, r["faults"]


def test_a_clearance_asks_for_a_gap_to_the_shelf():
    positions, rotations = _down_to(0.010, 0.15)
    world = ShelfWorld.build(_shelf(0.15))                       # a board at 15 cm under the hand
    kw = dict(grasp=60, release=60)
    assert not _check(positions, rotations, world, clearance=0.008, **kw)["faults"]
    r = _check(positions, rotations, world, clearance=0.012, **kw)
    assert any("in shelf board" in k for k in r["faults"]), r["faults"]


def test_a_clearance_is_per_waypoint():
    positions, rotations = _down_to(0.010, TABLE)
    world = ShelfWorld.build(None, support_height=TABLE)
    gap = np.full(len(positions), 0.012)
    gap[-8:] = 0.005                                             # small where the hand is low
    assert not _check(positions, rotations, world, grasp=60, release=60, clearance=gap)["faults"]
    with pytest.raises(ValueError):
        _check(positions, rotations, world, grasp=60, release=60, clearance=-0.001)


def test_the_base_needs_no_clearance():
    """It is bolted 13 mm above the table and cannot stray towards it."""
    positions, rotations = _path([0.45, 0.0, 0.40], [0.45, 0.0, 0.30])
    world = ShelfWorld.build(None, support_height=TABLE)
    r = _check(positions, rotations, world, grasp=59, release=59, clearance=0.03)
    assert not r["faults"], r["faults"]


def test_a_clearance_does_not_call_the_set_down_a_collision():
    """The object coming down onto its destination's top face is the task."""
    positions, rotations, grasp, cube = _cube_set_down(0.003)
    r = _check(positions, rotations, ShelfWorld.build(_shelf(0.20)), grasp=grasp, release=59,
               object_points=cube, destination_height=0.20, clearance=0.015)
    assert [k for k in r["faults"] if "carried object" in k] == []


def test_a_clearance_keeps_the_carried_object_off_a_board_edge():
    board = _shelf(0.20)                                         # front edge at x = 0.35
    g = np.mgrid[-0.02:0.021:0.01, -0.02:0.021:0.01, -0.02:0.021:0.01].reshape(3, -1).T
    base = np.zeros(len(g), dtype=bool)
    world = ShelfWorld.build(board)
    beside = g + [0.35 - 0.02 - 0.006, 0.0, 0.19]                # 6 mm in front of the edge
    args = (beside, base, world, board, None, 0.001, 0.005)
    assert pc._object_contacts(*args) == []
    assert pc._object_contacts(*args, clearance=0.004) == []
    assert {k for k, _, _ in pc._object_contacts(*args, clearance=0.010)} == {"deep"}


def test_a_tilt_swings_each_piece_by_its_own_distance_from_the_grasp_point():
    from tpgpt.hardware.fr3 import default_model
    from tpgpt.hardware.hand import MAX_OPENING

    fr3 = default_model()
    pieces, T = pc.robot_pieces(fr3, HOME_Q, FRANKA_HAND, MAX_OPENING)
    pivot = T[:3, 3]
    gaps = pc.piece_gaps(pieces, 0.003, np.radians(2.0), pivot)
    by_name = {p.name: p for p in pieces}
    pad = by_name["hand_left_pad"]
    reach = np.linalg.norm(pad.vertices - pivot, axis=1).max()
    assert reach == pytest.approx(0.057, abs=0.005)
    assert gaps["hand_left_pad"] == pytest.approx(0.003 + np.radians(2.0) * reach)
    wrist = np.linalg.norm(by_name["fr3_link7"].centre - pivot)
    assert gaps["fr3_link3"] == pytest.approx(0.003 + np.radians(2.0) * wrist)   # follows the wrist
    assert gaps["fr3_link0"] == 0.0
    assert pc.piece_gaps(pieces, 0.003)["hand_body"] == 0.003                 # no tilt, no swing


def test_a_tilt_can_refuse_what_the_clearance_alone_admits():
    positions, rotations = _down_to(0.010, TABLE)
    world = ShelfWorld.build(None, support_height=TABLE)
    kw = dict(grasp=60, release=60, clearance=0.005)
    assert not _check(positions, rotations, world, **kw)["faults"]
    # the pads are ~57 mm from the grasp point: 10 deg swings them 10 mm more, 15 in all
    r = _check(positions, rotations, world, tilt=np.radians(10.0), **kw)
    assert any("pad in the table" in k for k in r["faults"]), r["faults"]
