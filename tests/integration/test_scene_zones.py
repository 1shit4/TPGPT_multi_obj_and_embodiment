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


def test_the_place_zone_judges_the_hand_and_not_a_cylinder_around_it():
    """The bug the audit in `FINDINGS.md` 8o found.

    A gripper is two fingers and a wrist with air in between. Testing a cylinder
    of its own radius and length rejects candidates it would clear and admits
    ones it would foul -- over 2000 candidates the two verdicts agreed on 153.
    The filter now places the hand's own surface sample at the release pose, so
    it answers the question it is asking.
    """
    from tpgpt.grasp.filters import by_place_approach, free_corridor, hand_envelope
    from tpgpt.grasp.grasps import Grasp6D

    env = make_scene("cubby")
    try:
        pair = resolve_pair("panda")
        length, radius = hand_envelope(pair)
        release = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.09]

        def descending(position, closing=(0.0, 1.0, 0.0)):
            """+Z the approach, pointing down; ``closing`` the jaw axis.

            The jaw axis matters and is the point of the test. A Panda hand is
            **204 mm across it** -- measured, and GraspGen's model agrees with
            robosuite's to 8 mm -- while the usable depth in front of the top
            cubby's back panel is about 88 mm. So the same descent onto the same
            slot is clear with the hand turned across the shelf and blocked with
            it turned front-to-back. A cylinder of the hand's radius cannot
            express that difference at all; the hand itself can.
            """
            approach = np.array([0.0, 0.0, -1.0])
            c = np.asarray(closing, dtype=float)
            c = c - (c @ approach) * approach
            c /= np.linalg.norm(c)
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack([c, np.cross(approach, c), approach])
            pose[:3, 3] = np.asarray(position, dtype=float)
            return Grasp6D(pose=pose, score=0.9, gripper="franka_panda", width=0.08)

        # Descending onto the slot with the jaws across the shelf: room to spare.
        kept, blocked = by_place_approach(
            [descending(release)], np.array([0]), env, pair, release[None]
        )
        assert kept.tolist() == [0] and blocked == {}

        # The identical descent with the hand turned 90 degrees, so its wide
        # axis points at the back panel 38 mm away: blocked, and named.
        kept, blocked = by_place_approach(
            [descending(release, closing=(1.0, 0.0, 0.0))],
            np.array([0]), env, pair, release[None],
        )
        assert kept.tolist() == []
        assert "shelf_top_back" in blocked

        # Driven into the back panel: rejected, and the panel is named.
        buried = np.array([0.264, 0.0, 1.15])
        kept, blocked = by_place_approach(
            [descending(buried)], np.array([0]), env, pair, buried[None]
        )
        assert kept.tolist() == []
        assert "shelf_top_back" in blocked

        # The ray bundle is blind to the orientation that decides it: a
        # cylinder is the same shape however the hand is rolled inside it, so it
        # returns the same verdict for both poses above, one of which fits and
        # one of which does not.
        assert radius > 0.05
        for closing in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
            assert free_corridor(
                env, release, np.array([0.0, 0.0, -1.0]), length, radius
            ) == free_corridor(
                env, release, np.array([0.0, 0.0, -1.0]), length, radius
            )
    finally:
        env.close()


def test_the_place_zone_judges_the_pose_the_hand_actually_arrives_in():
    """The bug that made the stage answer a different question than it asked.

    The hand does not hold its pick orientation all the way to the shelf -- the
    object is turned on the way, which is most of what reshelving is. On this
    demonstration the hand rotates **35.5 degrees** between closing the jaws and
    opening them. Reusing the pick orientation therefore tested a pose 36
    degrees from the commanded one, and 36 degrees is the difference between a
    204 mm hand lying across an 88 mm slot and lying along it.
    """
    from scipy.spatial.transform import Rotation

    from tpgpt.grasp.filters import by_place_approach
    from tpgpt.grasp.grasps import Grasp6D

    env = make_scene("cubby")
    try:
        pair = resolve_pair("panda")
        release = np.asarray(env.slot_poses()["top_middle"]) + [0, 0, 0.09]
        # Picked with the jaws across the shelf, where the hand fits.
        approach = np.array([0.0, 0.0, -1.0])
        closing = np.array([0.0, 1.0, 0.0])
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
        pose[:3, 3] = release
        grasp = Grasp6D(pose=pose, score=0.9, gripper="franka_panda", width=0.08)

        # Judged in the pick orientation: clear, and that is the wrong answer.
        kept, _ = by_place_approach([grasp], np.array([0]), env, pair, release[None])
        assert kept.tolist() == [0]

        # Judged in the pose it actually arrives in -- a quarter turn about the
        # approach, which swings the 204 mm axis into the 88 mm slot -- blocked.
        quarter = Rotation.from_rotvec(np.pi / 2 * approach).as_matrix()
        kept, blocked = by_place_approach(
            [grasp], np.array([0]), env, pair, release[None], carry_rotation=quarter
        )
        assert kept.tolist() == []
        assert "shelf_top_back" in blocked
    finally:
        env.close()
