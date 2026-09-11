"""The scene's solid geometry as primitives, on shapes with arithmetic answers.

Every case here is one where the right answer can be worked out by hand, which
is the point: this module replaces a point-cloud proximity test with an exact
signed distance, so the thing worth testing is that the distance really is the
one the geometry implies -- sign, magnitude and which obstacle it belongs to.
"""

from __future__ import annotations

import numpy as np

from tpgpt.perception.obstacles import Obstacle, deepest_penetration


def box(centre=(0.0, 0.0, 0.0), half=(0.1, 0.2, 0.05), rotation=None, name="b"):
    return Obstacle(
        name=name,
        kind="box",
        position=np.asarray(centre, dtype=float),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
        size=np.asarray(half, dtype=float),
    )


def test_box_depth_is_distance_to_the_nearest_face():
    # A 200 x 400 x 100 mm box at the origin. A point 10 mm above the centre is
    # 40 mm below the top face, which is the nearest of the six.
    assert box().penetration(np.array([[0.0, 0.0, 0.01]]))[0] == 0.04
    # Dead centre: half the smallest dimension, 50 mm.
    assert box().penetration(np.array([[0.0, 0.0, 0.0]]))[0] == 0.05


def test_a_point_outside_a_box_reports_its_clearance_as_a_negative_depth():
    depth = box().penetration(np.array([[0.0, 0.0, 0.08]]))[0]
    assert np.isclose(depth, -0.03)


def test_a_rotated_box_is_tested_in_its_own_frame():
    # Turned a quarter turn about z, so its 200 mm x-extent now lies along y.
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = box(rotation=turn)
    # 150 mm out along x was outside the unturned box and is inside this one.
    assert box().penetration(np.array([[0.15, 0.0, 0.0]]))[0] < 0
    assert turned.penetration(np.array([[0.15, 0.0, 0.0]]))[0] > 0


def test_cylinder_depth_takes_the_nearer_of_the_wall_and_the_end_cap():
    # Radius 100 mm, half height 300 mm: on the axis the wall is the near one.
    cylinder = Obstacle("c", "cylinder", np.zeros(3), np.eye(3), np.array([0.1, 0.3]))
    assert np.isclose(cylinder.penetration(np.array([[0.0, 0.0, 0.0]]))[0], 0.1)
    # 50 mm from the flat end, the cap is nearer than the wall.
    assert np.isclose(cylinder.penetration(np.array([[0.0, 0.0, 0.25]]))[0], 0.05)
    assert cylinder.penetration(np.array([[0.11, 0.0, 0.0]]))[0] < 0


def test_a_plane_is_a_half_space_below_its_own_normal():
    plane = Obstacle("floor", "plane", np.zeros(3), np.eye(3), np.zeros(3))
    assert np.isclose(plane.penetration(np.array([[0.0, 0.0, -0.2]]))[0], 0.2)
    assert np.isclose(plane.penetration(np.array([[0.0, 0.0, 0.2]]))[0], -0.2)


def test_deepest_penetration_names_the_obstacle_it_found():
    obstacles = [box(name="near", centre=(0.0, 0.0, 0.0)),
                 box(name="far", centre=(1.0, 0.0, 0.0))]
    depth, name = deepest_penetration(np.array([[0.0, 0.0, 0.0]]), obstacles)
    assert name == "near" and depth == 0.05


def test_nothing_inside_anything_reports_clear():
    assert deepest_penetration(np.array([[5.0, 5.0, 5.0]]), [box()]) == (0.0, None)


def test_clearance_turns_a_near_miss_into_a_reported_touch():
    """The margin exists so a grazing pass is not read as free space."""
    point = np.array([[0.0, 0.0, 0.056]])          # 6 mm clear of the top face
    assert deepest_penetration(point, [box()]) == (0.0, None)
    depth, name = deepest_penetration(point, [box()], clearance=0.010)
    assert name == "b" and np.isclose(depth, 0.004)
