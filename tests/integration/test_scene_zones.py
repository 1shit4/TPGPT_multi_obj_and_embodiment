"""The scene-derived grasp constraints, against the real tabletop shelf.

These need a live simulator because the whole point of the filters they test is
that nothing about the blocked directions is written down: the shelf's panels
are read out of the model at the moment of asking, so the same code gives
different answers on the ``open``, ``cubby`` and ``enclosed`` variants without
knowing that those variants exist.

No grasp planner is involved and no physics is stepped, so these are seconds,
not minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.grasp.filters import free_corridor, hand_envelope
from tpgpt.grasp.grippers import gripper_config_path, resolve_pair
from tpgpt.perception.obstacles import (
    deepest_penetration,
    first_obstruction,
    scene_obstacles,
)
from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

pytestmark = pytest.mark.skipif(
    not gripper_config_path("franka_panda").is_file(),
    reason="GraspGen-X gripper descriptions not installed",
)

OBJECTS = ("cereal", "milk", "can", "bread")


def make_scene(shelf_variant="cubby"):
    env = TabletopShelf(
        robots="Panda", objects=OBJECTS, shelf_variant=shelf_variant,
        has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, seed=0,
    )
    env.reset()
    return env


def test_the_obstacle_set_is_the_solid_geometry_and_only_that():
    """Drawn is not solid. This scene puts a translucent marker at every shelf
    slot, and a geometric query that counts those reports obstacles a hand
    passes straight through."""
    env = make_scene()
    try:
        obstacles = scene_obstacles(env, exclude=("cereal",))
        names = {o.name for o in obstacles}
        assert "shelf_top_back" in names and "table_collision" in names
        assert not any(n.startswith("slot_") for n in names)      # visual only
        assert not any(n.startswith(("robot0", "gripper0")) for n in names)
        assert "cereal_g0" not in names                           # excluded
        assert "milk_g0" in names                                 # a neighbour
    finally:
        env.close()


def test_a_point_inside_the_back_panel_is_reported_inside_it_by_name():
    env = make_scene()
    try:
        obstacles = scene_obstacles(env)
        # ``shelf_top_back`` spans x 0.258 to 0.270; its mid-plane is 0.264 and
        # the panel is 12 mm thick, so the deepest any point can be is 6 mm.
        depth, name = deepest_penetration(np.array([[0.264, 0.0, 1.15]]), obstacles)
        assert name == "shelf_top_back"
        assert np.isclose(depth, 0.006, atol=1e-6)
        assert deepest_penetration(np.array([[0.10, 0.0, 1.15]]), obstacles) == (0.0, None)
    finally:
        env.close()


def test_the_blocked_directions_at_a_slot_are_read_from_the_shelf_not_listed():
    """The cubby's top slot: solid behind, walled on both sides, board below,
    open above and towards the robot. Nothing here is hard-coded -- change the
    shelf and the answer changes with it, which the next test shows."""
    env = make_scene("cubby")
    try:
        origin = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.06]
        answers = {
            label: first_obstruction(env, origin, vec, max_distance=0.5)
            for label, vec in (
                ("+x", (1, 0, 0)), ("-x", (-1, 0, 0)),
                ("+y", (0, 1, 0)), ("-y", (0, -1, 0)),
                ("+z", (0, 0, 1)), ("-z", (0, 0, -1)),
            )
        }
        assert answers["+x"][1] == "shelf_top_back"
        assert answers["+x"][0] < 0.05
        assert answers["+y"][1] == "shelf_top_wall_l"
        assert answers["-y"][1] == "shelf_top_wall_r"
        assert answers["-z"][1] == "shelf_top_board"       # not the slot marker
        assert answers["+z"][1] is None                    # no roof
        assert answers["-x"][1] is None                    # the way the arm comes
    finally:
        env.close()


def test_the_enclosed_shelf_closes_the_way_in_that_the_cubby_leaves_open():
    """A top-down placement is possible in a cubby and impossible under a roof.
    The filter is not told which scene it is in; it casts the same ray."""
    env = make_scene("enclosed")
    try:
        origin = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.06]
        distance, name = first_obstruction(env, origin, (0, 0, 1), max_distance=0.5)
        assert name == "shelf_top_roof"
        assert distance < 0.3
    finally:
        env.close()


def test_the_open_shelf_blocks_nothing_but_its_own_board():
    env = make_scene("open")
    try:
        origin = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.06]
        for vec in ((1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)):
            assert first_obstruction(env, origin, vec, max_distance=0.5)[1] is None
        assert first_obstruction(env, origin, (0, 0, -1), max_distance=0.5)[1] \
            == "shelf_top_board"
    finally:
        env.close()


def test_a_hand_arriving_from_above_is_clear_and_one_arriving_from_behind_is_not():
    """``free_corridor`` is the per-candidate form of the probe above: it sweeps
    a bundle of rays over the hand's own cross-section rather than one down its
    centre line, because a single ray threads the gap between two fingers."""
    env = make_scene("cubby")
    try:
        pair = resolve_pair("panda")
        length, radius = hand_envelope(pair)
        release = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.09]
        # Descending: the hand advances along -z, so it came from above, and
        # above the top slot is open.
        assert free_corridor(env, release, np.array([0.0, 0.0, -1.0]),
                             length, radius) is None
        # Advancing along -x means the hand came from +x -- from behind the
        # shelf, through its back panel.
        assert free_corridor(env, release, np.array([-1.0, 0.0, 0.0]),
                             length, radius) == "shelf_top_back"
        # Advancing along +x means it came from the robot's side, which is the
        # open face of the cubby -- and it is *still* blocked, because the
        # lower cubby's back panel rises 20 mm above the top board (to z=1.101)
        # and the trailing part of the hand meets it. That is the answer this
        # filter exists to give: "clear from the front" is a claim about a
        # point, and a hand is a body.
        assert free_corridor(env, release, np.array([1.0, 0.0, 0.0]),
                             length, radius) == "shelf_bottom_back"
    finally:
        env.close()
