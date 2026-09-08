"""Grasp filtering, on geometry with known answers.

Each filter is tested against a case constructed so the right answer is
obvious -- a hand buried in a wall, an object wider than the jaws, a grasp
approaching from a direction nothing ever saw. The funnel's *fallback*
behaviour gets as much attention as the filters themselves: a stage that
empties the candidate set and returns nothing turns "the hand would have
collided" into "there is no grasp", which is a worse answer, so it must pass
its input through and say that it did.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.grasp.filters import (
    FilterFunnel,
    by_collision,
    by_jaw_width,
    by_target,
    by_visibility,
    filter_grasps,
    suppress_duplicates,
)
from tpgpt.grasp.grasps import Grasp6D
from tpgpt.grasp.grippers import gripper_config_path, gripper_geometry, resolve_pair

pytestmark = pytest.mark.skipif(
    not gripper_config_path("franka_panda").is_file(),
    reason="GraspGen-X gripper descriptions not installed",
)

PAIR = "panda"


def grasp_at(position, approach=(0, 0, -1), yaw=0.0, score=0.8, gripper="franka_panda"):
    """A grasp at a position, approaching along ``approach``."""
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(approach @ reference) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    closing = np.cross(reference, approach)
    closing /= np.linalg.norm(closing)
    c, s = np.cos(yaw), np.sin(yaw)
    closing = c * closing + s * np.cross(approach, closing)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
    pose[:3, 3] = np.asarray(position, dtype=float)
    return Grasp6D(
        pose=pose, score=score, gripper=gripper,
        width=gripper_geometry(gripper).aperture,
    )


def box_cloud(centre, half, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    points[np.arange(n), axis] = np.sign(points[np.arange(n), axis])
    return np.asarray(centre) + points * np.asarray(half)


class TestVisibility:
    def test_a_grasp_reaching_around_the_far_side_is_dropped(self):
        """The generator invents the surface the camera never saw and proposes
        grasps on it."""
        camera = {"agentview": np.array([1.0, 0.0, 1.0])}
        towards = grasp_at([0, 0, 0.9], approach=[-1, 0, 0])   # camera -> object
        away = grasp_at([0, 0, 0.9], approach=[1, 0, 0])       # object -> camera
        kept = by_visibility([towards, away], np.array([0, 1]), camera)
        assert kept.tolist() == [0]

    def test_side_approaches_survive(self):
        """Around 90 degrees is legitimate and is much of the point of 6-DoF."""
        camera = {"agentview": np.array([1.0, 0.0, 1.0])}
        side = grasp_at([0, 0, 0.9], approach=[0, 1, 0])
        assert by_visibility([side], np.array([0]), camera).tolist() == [0]

    def test_seen_by_any_camera_is_enough(self):
        cameras = {"a": np.array([1.0, 0, 1]), "b": np.array([-1.0, 0, 1])}
        grasp = grasp_at([0, 0, 0.9], approach=[1, 0, 0])
        assert len(by_visibility([grasp], np.array([0]), cameras)) == 1


class TestTarget:
    def test_a_grasp_that_would_hold_nothing_is_dropped(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        pair = resolve_pair(PAIR)
        on = grasp_at([0, 0, 0.85 + gripper_geometry("franka_panda").tcp_depth])
        off = grasp_at([0.5, 0, 0.85 + gripper_geometry("franka_panda").tcp_depth])
        kept = by_target([on, off], np.array([0, 1]), cloud, pair)
        assert kept.tolist() == [0]


class TestJawWidth:
    def test_an_object_wider_than_the_jaws_is_dropped(self):
        pair = resolve_pair(PAIR)
        aperture = gripper_geometry("franka_panda").aperture
        depth = gripper_geometry("franka_panda").tcp_depth
        grasp = grasp_at([0, 0, 0.85 + depth])
        # The width that matters is along this grasp's own closing axis, so the
        # box has to be widened along that axis and not just made big.
        closing = np.argmax(np.abs(grasp.closing))
        narrow_half = np.array([0.03, 0.03, 0.05])
        wide_half = narrow_half.copy()
        narrow_half[closing] = aperture * 0.25
        wide_half[closing] = aperture * 1.5
        narrow = box_cloud([0, 0, 0.85], narrow_half)
        wide = box_cloud([0, 0, 0.85], wide_half)
        assert len(by_jaw_width([grasp], np.array([0]), narrow, pair)) == 1
        assert len(by_jaw_width([grasp], np.array([0]), wide, pair)) == 0

    def test_width_is_measured_along_the_grasps_own_closing_axis(self):
        """A bottle is narrow at the neck and wide at the base; which matters
        depends on where the grasp sits, so a global bounding box will not do.
        """
        pair = resolve_pair(PAIR)
        aperture = gripper_geometry("franka_panda").aperture
        depth = gripper_geometry("franka_panda").tcp_depth
        # A slab: wide in x, narrow in y. A grasp closing across y fits, one
        # closing across x does not, on the very same cloud.
        slab = box_cloud([0, 0, 0.85], [aperture * 1.5, 0.02, 0.05])
        fits, does_not = None, None
        for yaw in np.linspace(0, np.pi, 9):
            grasp = grasp_at([0, 0, 0.85 + depth], yaw=yaw)
            kept = len(by_jaw_width([grasp], np.array([0]), slab, pair)) == 1
            if abs(grasp.closing[1]) > 0.95:
                fits = kept
            if abs(grasp.closing[0]) > 0.95:
                does_not = kept
        assert fits is True and does_not is False

    def test_too_little_cloud_is_not_treated_as_too_wide(self):
        """Absence of evidence is not evidence of a wide object; a sparse cloud
        is a perception problem and belongs in a different report line."""
        pair = resolve_pair(PAIR)
        grasp = grasp_at([0, 0, 0.9])
        assert len(by_jaw_width([grasp], np.array([0]), np.zeros((2, 3)), pair)) == 1


class TestCollision:
    def test_a_hand_inside_a_wall_is_dropped_and_one_beside_it_is_not(self):
        pair = resolve_pair(PAIR)
        wall = box_cloud([0.3, 0.0, 0.9], [0.005, 0.3, 0.3], n=4000)
        inside = grasp_at([0.3, 0.0, 0.9])
        clear = grasp_at([-0.3, 0.0, 0.9])
        kept = by_collision([inside, clear], np.array([0, 1]), wall, pair, corridor=0.0,
                            samples=1)
        assert kept.tolist() == [1]

    def test_the_approach_corridor_is_checked_not_just_the_final_pose(self):
        """A grasp can be clear where it ends and pass through a neighbour on
        the way in."""
        pair = resolve_pair(PAIR)
        # An obstacle above the grasp, in the path of a downward approach.
        obstacle = box_cloud([0.0, 0.0, 1.15], [0.10, 0.10, 0.01], n=4000)
        grasp = grasp_at([0.0, 0.0, 0.90], approach=[0, 0, -1])
        final_only = by_collision([grasp], np.array([0]), obstacle, pair,
                                  corridor=0.0, samples=1)
        with_corridor = by_collision([grasp], np.array([0]), obstacle, pair,
                                     corridor=0.30, samples=6)
        assert len(final_only) == 1
        assert len(with_corridor) == 0

    def test_a_single_stray_point_does_not_veto_a_grasp(self):
        """Depth edges produce flyer points exactly where two objects meet,
        which is where grasps live."""
        pair = resolve_pair(PAIR)
        stray = np.array([[0.0, 0.0, 0.95]])
        grasp = grasp_at([0.0, 0.0, 0.90])
        assert len(by_collision([grasp], np.array([0]), stray, pair,
                                corridor=0.0, samples=1)) == 1


class TestDuplicates:
    def test_near_identical_grasps_are_thinned_to_the_best(self):
        grasps = [
            grasp_at([0, 0, 0.9], score=0.7),
            grasp_at([0.002, 0, 0.9], score=0.9),   # duplicate of the first
            grasp_at([0.5, 0, 0.9], score=0.6),     # far away, distinct
        ]
        kept = suppress_duplicates(grasps, np.array([0, 1, 2]))
        assert len(kept) == 2
        assert 2 in kept
        assert 1 in kept and 0 not in kept          # the better of the pair

    def test_same_position_different_approach_is_not_a_duplicate(self):
        grasps = [
            grasp_at([0, 0, 0.9], approach=[0, 0, -1]),
            grasp_at([0, 0, 0.9], approach=[1, 0, 0]),
        ]
        assert len(suppress_duplicates(grasps, np.array([0, 1]))) == 2


class TestFunnel:
    def test_a_stage_that_would_empty_the_set_falls_back_and_says_so(self):
        """The load-bearing behaviour. Returning nothing turns 'the hand would
        have collided' into 'there is no grasp', which is a worse answer."""
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        # Every grasp is nowhere near the object, so 'on target' empties.
        grasps = [grasp_at([2.0, 0, 0.9]), grasp_at([2.1, 0, 0.9])]
        funnel = filter_grasps(grasps, PAIR, cloud)
        assert len(funnel.survivors) > 0
        assert funnel.flags.get("on target_fell_back")
        assert "FELL BACK" in funnel.describe()

    def test_the_funnel_records_every_stage(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        depth = gripper_geometry("franka_panda").tcp_depth
        grasps = [grasp_at([0, 0, 0.85 + depth], yaw=y) for y in (0.0, 0.5, 1.0)]
        funnel = filter_grasps(grasps, PAIR, cloud)
        names = [s.name for s in funnel.stages]
        assert names[0] == "generated"
        assert "on target" in names and "jaw width" in names and "distinct" in names
        assert all(s.entered >= s.survived for s in funnel.stages)

    def test_survivors_come_back_best_first(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        depth = gripper_geometry("franka_panda").tcp_depth
        grasps = [
            grasp_at([0, 0, 0.85 + depth], yaw=0.0, score=0.5),
            grasp_at([0.10, 0, 0.85 + depth], yaw=0.0, score=0.9),
        ]
        funnel = filter_grasps(grasps, PAIR, cloud)
        scores = [grasps[i].score for i in funnel.survivors]
        assert scores == sorted(scores, reverse=True)

    def test_the_worst_stage_is_reported_for_explaining_a_failure(self):
        funnel = FilterFunnel()
        funnel.add("visibility", "x", np.arange(10), np.arange(9))
        funnel.add("collision", "y", np.arange(9), np.arange(2))
        assert funnel.rejected_by == "collision"

    def test_nothing_rejected_means_no_blame(self):
        funnel = FilterFunnel()
        funnel.add("visibility", "x", np.arange(5), np.arange(5))
        assert funnel.rejected_by is None
