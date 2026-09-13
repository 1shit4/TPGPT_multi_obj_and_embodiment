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


def test_a_convex_body_handles_the_three_shapes_robot_hands_are_made_of():
    """Mesh, box, cylinder and sphere, because robosuite's hands use all of them.

    The Rethink gripper's base is a cylinder and the Ability and Schunk hands
    carry spheres, and a hand whose base cannot be represented is a limb the
    collision check cannot see. Dropping it silently would be worse than a slow
    check, so an unrepresentable geom raises instead -- this pins the three that
    do not have to.
    """
    from tpgpt.perception.obstacles import ConvexBody

    eye = np.eye(3)
    centre = np.zeros(3)

    # A 100 mm cube as half spaces.
    half = 0.05
    planes = np.array([[1, 0, 0, -half], [-1, 0, 0, -half],
                       [0, 1, 0, -half], [0, -1, 0, -half],
                       [0, 0, 1, -half], [0, 0, -1, -half]], dtype=float)
    box = ConvexBody("box", 0, planes, half * np.sqrt(3))
    assert box.contains(np.array([[0.0, 0, 0]]), centre, eye)[0]
    assert not box.contains(np.array([[0.06, 0, 0]]), centre, eye)[0]

    # Radius 40 mm, half height 100 mm.
    cyl = ConvexBody("cyl", 0, np.empty((0, 4)), 0.11, kind="cylinder",
                     size=(0.04, 0.1))
    assert cyl.contains(np.array([[0.03, 0, 0.05]]), centre, eye)[0]
    assert not cyl.contains(np.array([[0.05, 0, 0.0]]), centre, eye)[0]   # too wide
    assert not cyl.contains(np.array([[0.0, 0, 0.11]]), centre, eye)[0]   # too tall

    sph = ConvexBody("sph", 0, np.empty((0, 4)), 0.03, kind="sphere", size=(0.03,))
    assert sph.contains(np.array([[0.02, 0, 0]]), centre, eye)[0]
    assert not sph.contains(np.array([[0.031, 0, 0]]), centre, eye)[0]

    # The tolerance shrinks every shape, not only the polytope.
    assert not sph.contains(np.array([[0.025, 0, 0]]), centre, eye, tolerance=0.01)[0]
    assert not cyl.contains(np.array([[0.035, 0, 0]]), centre, eye, tolerance=0.01)[0]
