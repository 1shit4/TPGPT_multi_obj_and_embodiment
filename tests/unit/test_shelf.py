"""The shelf world: the model built from touched dimensions, and the cloud cropped to it."""

import numpy as np
import pytest

from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.fr3 import HOME_Q, TCP_LINK, default_model
from tpgpt.hardware.shelf import ShelfModel, ShelfWorld, box_surface, robot_mask

# A square shelf 0.5 m wide and 0.3 m deep, open front facing the robot at +x = 0.4.
CONFIG = {
    "shelf": {
        "measured": True,
        "front_left": [0.4, 0.25],
        "front_right": [0.4, -0.25],
        "depth": 0.3,
        "bottom": 0.0,
        "top": 0.6,
        "wall_thickness": 0.02,
        "back_thickness": 0.01,
        "boards": [
            {"name": "low", "top": 0.10, "thickness": 0.02},
            {"name": "high", "top": 0.35, "thickness": 0.02},
            {"name": "roof", "top": 0.60, "thickness": 0.02, "support": False},
        ],
    }
}


@pytest.fixture(scope="module")
def shelf() -> ShelfModel:
    return ShelfModel.from_config(CONFIG)


def test_the_frame_points_into_the_shelf_and_the_panels_land_where_measured(shelf):
    assert np.allclose(shelf.R_base_shelf[:, 1], [1.0, 0.0, 0.0])   # +y into the shelf = +x base
    names = [b.name for b in shelf.boxes]
    assert names == ["left_wall", "right_wall", "back_panel", "low", "high", "roof"]
    assert shelf.supports == ("low", "high")
    high = next(b for b in shelf.boxes if b.name == "high")
    assert np.allclose(high.position, [0.55, 0.0, 0.34])
    assert np.allclose(high.size, [0.25, 0.15, 0.01])


def test_swapped_front_corners_are_refused():
    bad = {"shelf": dict(CONFIG["shelf"], front_left=[0.4, -0.25], front_right=[0.4, 0.25])}
    with pytest.raises(ValueError, match="swapped"):
        ShelfModel.from_config(bad)


def test_penetration_names_the_panel(shelf):
    depth, names = shelf.penetration(np.array([[0.55, 0.0, 0.345], [0.55, 0.0, 0.40],
                                               [0.55, 0.26, 0.3]]))
    assert depth[0] > 0 and names[0] == "high"
    assert depth[1] < 0
    assert depth[2] > 0 and names[2] == "left_wall"


def test_box_surface_samples_lie_on_the_faces():
    half = np.array([0.1, 0.05, 0.01])
    s = box_surface(half, 0.01)
    assert np.allclose(np.abs(s).max(axis=0), half)
    on_a_face = np.any(np.isclose(np.abs(s), half), axis=1)
    assert on_a_face.all()


def test_the_cloud_keeps_only_what_is_on_the_shelf(shelf):
    rng = np.random.default_rng(0)
    on_shelf = np.column_stack([rng.uniform(0.45, 0.65, 200), rng.uniform(-0.2, 0.2, 200),
                                np.full(200, 0.36)])                # something on the high board
    away = np.column_stack([rng.uniform(0.0, 0.3, 100), rng.uniform(-0.5, 0.5, 100),
                            rng.uniform(0.0, 0.5, 100)])           # free space in front
    table = np.column_stack([rng.uniform(0.3, 0.9, 100), rng.uniform(-0.4, 0.4, 100),
                             np.zeros(100)])                        # the table around the foot
    target = on_shelf[:20] + [0.0, 0.0, 0.002]
    world = ShelfWorld.build(shelf, np.vstack([on_shelf, away, table]), target_points=target)
    r = world.cloud_report
    assert r["in_shelf_region"] == 200
    assert r["removed_as_target"] >= 20
    assert len(world.cloud) == r["obstacle_points"] <= 180
    assert r["robot_not_removed"]


def test_the_robot_is_removed_from_its_own_cloud():
    fr3 = default_model()
    T = fr3.tcp(HOME_Q)
    on_hand = T[:3, 3] - T[:3, :3][:, 2] * 0.05       # 5 cm up the hand from the TCP
    elsewhere = np.array([0.6, 0.4, 0.1])
    mask = robot_mask(np.vstack([on_hand, elsewhere]), HOME_Q, fr3, FRANKA_HAND)
    assert mask.tolist() == [True, False]


def test_observed_panels_are_explained_by_the_model(shelf):
    high = next(b for b in shelf.boxes if b.name == "high")
    face = high.position + np.array([[x, y, 0.01] for x in (-0.1, 0.0, 0.1) for y in (-0.2, 0.2)])
    assert shelf.explained(face).all()
    assert not shelf.explained(np.array([[0.55, 0.0, 0.45]]))[0]


def test_tcp_link_is_known():
    assert TCP_LINK in default_model().link_poses(HOME_Q)


# ------------------------------------------------ everything the camera sees
def _visible_scene(shelf):
    rng = np.random.default_rng(1)
    neighbour = np.column_stack([rng.uniform(0.25, 0.30, 60), rng.uniform(0.30, 0.35, 60),
                                 rng.uniform(0.02, 0.10, 60)])      # a box on the table, off the shelf
    table = np.column_stack([rng.uniform(0.2, 0.9, 300), rng.uniform(-0.5, 0.5, 300),
                             rng.normal(0.0, 0.003, 300)])          # table at z = 0, noisy
    far = np.array([[2.0, 0.0, 0.5], [0.0, -1.8, 1.0]])             # the lab, out of reach
    fr3 = default_model()
    T = fr3.tcp(HOME_Q)
    arm = T[:3, 3] - T[:3, :3][:, 2] * np.linspace(0.03, 0.06, 5)[:, None]   # on the hand
    return neighbour, table, far, arm, fr3


def test_everything_seen_within_reach_is_an_obstacle(shelf):
    neighbour, table, far, arm, fr3 = _visible_scene(shelf)
    world = ShelfWorld.build(shelf, np.vstack([neighbour, table, far, arm]), robot_q=HOME_Q,
                             fr3=fr3, end_effector=FRANKA_HAND, support_height=0.0)
    r = world.cloud_report
    assert r["scope"] == "visible"
    assert r["outside_reach"] == 2
    assert r["removed_as_table"] >= 290
    assert r["removed_as_robot"] == 5
    assert len(world.cloud) >= 55                # the neighbouring box stays
    assert np.all(world.cloud[:, 2] > 0.015)


def test_without_joint_angles_only_the_shelf_region_is_kept(shelf):
    neighbour, table, far, arm, _ = _visible_scene(shelf)
    world = ShelfWorld.build(shelf, np.vstack([neighbour, table, far, arm]), support_height=0.0)
    r = world.cloud_report
    assert r["scope"] == "shelf" and "robot cannot be removed" in r["scope_fallback"]
    assert len(world.cloud) == 0                 # nothing here is on the shelf


def test_a_wrong_scope_is_refused(shelf):
    with pytest.raises(ValueError):
        ShelfWorld.build(shelf, np.zeros((1, 3)), scope="everything")


# ------------------------------------------------ tape measurements, anchored
def _tape_case():
    """A shelf at a known pose (19 deg off, like the real one), and the tape
    readings a person would take on it, derived from its geometry."""
    yaw = np.radians(-161.0)
    x = np.array([np.cos(yaw), np.sin(yaw)])
    y = np.array([-x[1], x[0]])
    corner_l = np.array([0.80, -0.10])
    W, D, table = 0.52, 0.30, -0.0136
    slot_l = corner_l + x * 0.140 + y * 0.150           # the touched slot marks
    slot_r = slot_l + x * 0.2332
    truth = {"shelf": {
        "measured": True, "front_left": corner_l.tolist(),
        "front_right": (corner_l + x * W).tolist(), "depth": D, "bottom": table,
        "top": table + 0.62, "wall_thickness_left": 0.018, "wall_thickness_right": 0.019,
        "back_thickness": 0.0, "side_front": 0.012, "side_depth": 0.33,
        "boards": [{"name": "bottom_board", "top": 0.0768, "thickness": 0.017},
                   {"name": "top_board", "top": 0.3213, "thickness": 0.017},
                   {"name": "topmost_board", "top": 0.3213 + 0.250, "thickness": 0.017}]}}
    tape = {"slot_left_to_left_panel": 140, "slot_right_to_right_panel": W * 1000 - 140 - 233.2,
            "slot_left_to_front_edge": 150, "slot_right_to_front_edge": 150,
            "inner_width": W * 1000, "left_panel_thickness": 18, "right_panel_thickness": 19,
            "panel_height": 620, "panel_depth": 330, "panel_front_ahead_of_boards": 12,
            "board_depth": 300, "topmost_above_top_board": 250, "board_thickness": 17,
            "apron_height": 0}
    anchors = {"left_slot": slot_l.tolist(), "right_slot": slot_r.tolist(),
               "bottom_board_top": 0.0768, "top_board_top": 0.3213, "table": table}
    return truth, {"shelf": {"measured": True, "anchors": anchors, "tape_mm": tape}}


def test_tape_measurements_rebuild_the_shelf_exactly():
    truth, taped = _tape_case()
    a, b = ShelfModel.from_config(truth), ShelfModel.from_config(taped)
    assert [x.name for x in a.boxes] == [x.name for x in b.boxes]
    for p, q in zip(a.boxes, b.boxes):
        assert np.allclose(p.position, q.position, atol=1e-9), p.name
        assert np.allclose(p.size, q.size, atol=1e-9), p.name
    assert b.supports == ("bottom_board", "top_board", "topmost_board")
    assert abs(b.config["derived"]["width_mismatch_mm"]) < 1e-6


def test_a_tape_that_disagrees_with_the_touched_slots_is_refused():
    _, taped = _tape_case()
    taped["shelf"]["tape_mm"]["inner_width"] += 12            # misread by 12 mm
    with pytest.raises(ValueError, match="disagrees"):
        ShelfModel.from_config(taped)


def test_missing_tape_readings_are_named():
    _, taped = _tape_case()
    del taped["shelf"]["tape_mm"]["panel_height"]
    with pytest.raises(ValueError, match="M8"):
        ShelfModel.from_config(taped)
