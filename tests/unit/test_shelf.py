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
