"""The shelf world: the model built from touched dimensions, and the cloud cropped to it."""

import numpy as np
import pytest

from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.fr3 import HOME_Q, TCP_LINK, default_model
from tpgpt.hardware.shelf import (
    ShelfModel, ShelfWorld, _faces, apply_cloud_fit, box_surface, describe_fit, fit_to_cloud,
    robot_mask,
)

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


def test_m5_is_the_width_and_the_slot_readings_disagreement_is_split():
    _, taped = _tape_case()
    taped["shelf"]["tape_mm"]["slot_left_to_left_panel"] += 4    # M1 read 4 mm long
    model = ShelfModel.from_config(taped)
    assert model.width == pytest.approx(0.52, abs=1e-9)          # M5, not M1 + 233 + M2
    assert model.config["derived"]["width_mismatch_mm"] == pytest.approx(4.0, abs=1e-6)


def test_named_slots_resolve_to_the_touched_marks_on_each_board():
    cfg = {"shelf": {**REAL, "slots": {
        "top_left": {"xy": "left_slot", "board": "top_board"},
        "bottom_right": {"xy": "right_slot", "board": "bottom_board"}}}}
    from tpgpt.hardware.shelf import slots
    got = slots(cfg)
    assert got["top_left"] == (0.6842, -0.2774, 0.3213)          # the touched mark and board
    assert got["bottom_right"] == (0.4636, -0.3529, 0.0768)
    with pytest.raises(KeyError):
        slots({"shelf": {**REAL, "slots": {"x": {"xy": "nope", "board": "top_board"}}}})


# ------------------------------------------------ the placement, from the cloud
def test_a_cloud_fit_moves_the_front_edge_rigidly_and_never_its_height():
    left, right = np.array([0.8, -0.1]), np.array([0.3, -0.3])
    fit = {"shift_along_front_mm": 5.0, "shift_into_shelf_mm": -3.0, "yaw_deg": 1.0}
    l2, r2 = apply_cloud_fit(left, right, fit)
    assert np.linalg.norm(r2 - l2) == pytest.approx(np.linalg.norm(right - left))
    x = (right - left) / np.linalg.norm(right - left)
    y = np.array([-x[1], x[0]])
    mid, mid2 = (left + right) / 2, (l2 + r2) / 2
    assert (mid2 - mid) @ x == pytest.approx(0.005) and (mid2 - mid) @ y == pytest.approx(-0.003)
    x2 = (r2 - l2) / np.linalg.norm(r2 - l2)
    assert np.degrees(np.arctan2(np.cross(x, x2), x @ x2)) == pytest.approx(1.0)


# The real shelf's tape and touched points (config/shelf.yaml, 2026-09-22) and the
# camera's position (camera_calibration.yaml), so the faces the fit gets are the
# ones this camera sees.
REAL = {"measured": True,
        "anchors": {"left_slot": [0.6842, -0.2774], "right_slot": [0.4636, -0.3529],
                    "bottom_board_top": 0.0768, "top_board_top": 0.3213, "table": -0.0136},
        "tape_mm": {"slot_left_to_left_panel": 201, "slot_right_to_right_panel": 142,
                    "slot_left_to_front_edge": 90, "slot_right_to_front_edge": 92,
                    "inner_width": 580, "left_panel_thickness": 18, "right_panel_thickness": 18,
                    "panel_height": 902, "panel_depth": 242, "panel_front_ahead_of_boards": 0,
                    "board_depth": 242, "topmost_above_top_board": 489, "board_thickness": 18,
                    "apron_height": 72}}
CAMERA = np.array([1.196, 0.505, 0.860])


def _seen(model, faces=None, noise=0.003, seed=0):
    """The model's faces that look towards the camera, with depth noise."""
    pts, _ = model.surface_points(0.006)
    _, _, n = _faces(model, pts)
    toward = ((CAMERA - pts) * (n @ model.R_base_shelf.T)).sum(axis=1) > 0
    if faces is not None:
        toward &= np.abs(n[:, faces]).max(axis=1) > 0.5 if np.ndim(faces) else \
            np.abs(n[:, faces]) > 0.5
    rng = np.random.default_rng(seed)
    return pts[toward] + rng.normal(0.0, noise, (int(toward.sum()), 3))


def test_the_fit_finds_where_the_shelf_really_is():
    """The real shelf, but standing 7 mm right, 9 mm nearer and turned 0.8 deg
    from where the tape puts it; with a clutter of points (an object) on the
    top board. The fit recovers the placement to depth-noise accuracy."""
    true = {"shift_along_front_mm": 7.0, "shift_into_shelf_mm": -9.0, "yaw_deg": 0.8}
    cloud = _seen(ShelfModel.from_config({"shelf": {**REAL, "cloud_fit": true}}))
    rng = np.random.default_rng(1)
    cloud = np.vstack([cloud, np.array([0.6, -0.35, 0.36]) + rng.uniform(-0.03, 0.03, (400, 3))])
    report = fit_to_cloud({"shelf": REAL}, cloud, camera=CAMERA)
    fit = report["cloud_fit"]
    assert report["refused"] is None and all(report["fitted"].values())
    assert fit["shift_along_front_mm"] == pytest.approx(7.0, abs=2.0)
    assert fit["shift_into_shelf_mm"] == pytest.approx(-9.0, abs=2.0)
    assert fit["yaw_deg"] == pytest.approx(0.8, abs=0.2)
    assert report["median_abs_after_mm"] < report["median_abs_before_mm"]
    assert "M3" in describe_fit(report)


def test_a_fit_already_in_the_config_does_not_change_the_answer():
    true = {"shift_along_front_mm": 4.0, "shift_into_shelf_mm": 5.0, "yaw_deg": -0.5}
    cloud = _seen(ShelfModel.from_config({"shelf": {**REAL, "cloud_fit": true}}))
    stale = {"shift_along_front_mm": -20.0, "shift_into_shelf_mm": 0.0, "yaw_deg": 1.0}
    a = fit_to_cloud({"shelf": REAL}, cloud, camera=CAMERA)["fit"]
    b = fit_to_cloud({"shelf": {**REAL, "cloud_fit": stale}}, cloud, camera=CAMERA)["fit"]
    assert a == b


def test_a_direction_the_cloud_does_not_see_keeps_the_tapes_value():
    """Only board tops seen (say, from above): they say nothing about x/y."""
    true = {"shift_along_front_mm": 6.0, "shift_into_shelf_mm": 6.0, "yaw_deg": 0.0}
    cloud = _seen(ShelfModel.from_config({"shelf": {**REAL, "cloud_fit": true}}), faces=2)
    report = fit_to_cloud({"shelf": REAL}, cloud, camera=CAMERA)
    assert not any(report["fitted"].values()) and report["refused"]
    assert report["cloud_fit"] is None


def test_a_correction_beyond_any_tape_error_is_refused():
    true = {"shift_along_front_mm": 0.0, "shift_into_shelf_mm": 34.0, "yaw_deg": 0.0}
    cloud = _seen(ShelfModel.from_config({"shelf": {**REAL, "cloud_fit": true}}))
    report = fit_to_cloud({"shelf": REAL}, cloud, camera=CAMERA)
    assert report["refused"] and "beyond" in report["refused"] and report["cloud_fit"] is None


# ------------------------------------------------ where a carried object may rest
def _kinds(shelf, points, base=None):
    from tpgpt.hardware.path_check import _object_contacts

    base = np.ones(len(points), dtype=bool) if base is None else base
    return {(k, w) for k, w, _ in _object_contacts(np.atleast_2d(points), base,
                                                    ShelfWorld.build(shelf), shelf, None,
                                                    0.001, 0.005)}


def test_contacts_are_classified_by_the_face_they_go_through(shelf):
    """The path check's rule (tpgpt.hardware.path_check._object_contacts): the
    top face of a support board is where an object may rest; anything else is
    'deep' (through a board) or 'into' (a panel that is not a support)."""
    assert _kinds(shelf, [[0.55, 0.0, 0.347]]) == {("base_rest", "high")}      # 3 mm into the top
    assert _kinds(shelf, [[0.55, 0.0, 0.343]]) == {("deep", "high")}           # 7 mm into it
    assert _kinds(shelf, [[0.55, 0.0, 0.3315]]) == {("deep", "high")}          # up into its underside
    assert _kinds(shelf, [[0.55, 0.252, 0.30]]) == {("into", "left_wall")}     # 2 mm into a side


def test_a_wide_flat_object_set_down_rests_on_its_base(shelf):
    """The simulation's 60-degree cone reads this box's bottom corners as side
    contacts (corner 150 mm out, 25 mm down from the centre: 80 deg off vertical).
    By faces, all of it rests on the board's top."""
    g = np.mgrid[0.40:0.701:0.01, -0.10:0.101:0.01].reshape(2, -1).T
    bottom = np.column_stack([g, np.full(len(g), 0.348)])        # 2 mm into the high board
    assert _kinds(shelf, bottom) == {("base_rest", "high")}


def test_the_camera_points_on_the_table_and_shelf_are_removed(shelf):
    high = next(b for b in shelf.boxes if b.name == "high")
    on_board = high.position + np.array([[x, y, 0.012] for x in (-0.1, 0.1) for y in (-0.1, 0.1)])
    box_on_board = np.array([[0.55, 0.0, z] for z in (0.40, 0.45, 0.50)])
    table = np.array([[0.3, 0.3, 0.004], [0.2, -0.3, -0.006]])
    world = ShelfWorld.build(shelf, np.vstack([on_board, box_on_board, table]),
                             robot_q=HOME_Q, fr3=default_model(), support_height=0.0)
    r = world.cloud_report
    assert r["removed_as_shelf"] == 4 and r["removed_as_table"] == 2
    assert len(world.cloud) == 3 and world.table_height == 0.0
