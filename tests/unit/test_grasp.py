"""Gripper registry, grasp records, and the GraspGen-X frame contract.

Offline: no simulator and no grasp server. The gripper configs are read from
the GraspGen-X checkout, so these skip if it is absent.
"""

import numpy as np
import pytest

from tpgpt.grasp.client import PLANNERS, GraspGenClient
from tpgpt.grasp.grasps import (
    Grasp6D,
    alignment_rotation,
    approach_waypoint,
    build_grasp_set,
    grasp_to_eef_pose,
)
from tpgpt.grasp.grippers import (
    GRIPPER_PAIRS,
    VERIFIED_PAIRS,
    gripper_config_path,
    gripper_geometry,
    resolve_pair,
    summary_table,
)

pytestmark = pytest.mark.skipif(
    not gripper_config_path("franka_panda").is_file(),
    reason="GraspGen-X gripper descriptions not installed",
)


def _identity_grasp(position=(0.0, 0.0, 1.0), gripper="franka_panda"):
    pose = np.eye(4)
    pose[:3, 3] = position
    return Grasp6D(pose=pose, score=0.9, gripper=gripper,
                   width=gripper_geometry(gripper).aperture)


class TestGripperRegistry:
    def test_every_pair_has_a_readable_config(self):
        for short, pair in GRIPPER_PAIRS.items():
            geom = gripper_geometry(pair.graspgen)
            assert geom.aperture > 0, short
            assert geom.tcp_depth > 0, short
            assert geom.family in ("parallel_2f", "revolute_2f", "revolute_3f"), short

    def test_the_pairs_span_all_three_kinematic_families(self):
        """The point of the multi-embodiment claim: not just parallel jaws."""
        families = {gripper_geometry(p.graspgen).family for p in GRIPPER_PAIRS.values()}
        assert families == {"parallel_2f", "revolute_2f", "revolute_3f"}

    def test_the_pairs_differ_substantially_in_geometry(self):
        """If every hand were the same size, transport would be trivial."""
        apertures = [gripper_geometry(p.graspgen).aperture for p in GRIPPER_PAIRS.values()]
        depths = [gripper_geometry(p.graspgen).tcp_depth for p in GRIPPER_PAIRS.values()]
        assert max(apertures) > 2 * min(apertures)
        assert max(depths) > 1.8 * min(depths)

    @pytest.mark.parametrize("name", ["panda", "PandaGripper", "franka_panda"])
    def test_pairs_resolve_by_any_of_their_names(self, name):
        assert resolve_pair(name).graspgen == "franka_panda"

    def test_unknown_gripper_lists_what_is_known(self):
        with pytest.raises(KeyError, match="known short names"):
            resolve_pair("nonexistent_hand")

    def test_the_tcp_is_not_the_lowest_point_of_the_hand(self):
        """A documented trap: the Panda's pads reach past its fingertip frame."""
        geom = gripper_geometry("franka_panda")
        assert geom.lowest_point > geom.tcp_depth

    def test_summary_table_covers_every_pair(self):
        table = summary_table()
        for short in GRIPPER_PAIRS:
            assert short in table


class TestFrameContract:
    def test_verified_grippers_are_the_ones_with_a_closing_axis(self):
        assert set(VERIFIED_PAIRS) == {
            k for k, v in GRIPPER_PAIRS.items() if v.closing_axis is not None
        }
        assert "panda" in VERIFIED_PAIRS

    def test_x_closing_grippers_need_no_rotation(self):
        """Measured: grip_site +Z is the approach axis for every gripper, and
        the Panda's jaws close along grip_site +X -- the same basis GraspGen-X
        emits."""
        assert np.allclose(alignment_rotation(GRIPPER_PAIRS["panda"]), np.eye(3))

    def test_y_closing_grippers_rotate_about_the_approach_axis(self):
        rotation = alignment_rotation(GRIPPER_PAIRS["xarm"])
        assert not np.allclose(rotation, np.eye(3))
        # A rotation about Z: the approach axis must be untouched.
        assert np.allclose(rotation @ np.array([0, 0, 1.0]), [0, 0, 1.0])
        assert np.allclose(np.linalg.det(rotation), 1.0)

    def test_unverified_grippers_refuse_rather_than_guess(self):
        """A wrong rotation about the approach axis fails silently -- the pose
        still looks plausible while closing across the wrong dimension."""
        with pytest.raises(ValueError, match="has not been measured"):
            alignment_rotation(GRIPPER_PAIRS["inspire"])

    def test_conversion_translates_by_the_gripper_tcp_depth(self):
        grasp = _identity_grasp()
        position, _ = grasp_to_eef_pose(grasp)
        depth = gripper_geometry("franka_panda").tcp_depth
        assert np.allclose(position, grasp.position + grasp.approach * depth)

    def test_the_same_grasp_needs_a_different_eef_pose_per_gripper(self):
        """The crux of cross-embodiment transport: TCP depth varies nearly 2x,
        so an end-effector pose is not portable between hands."""
        grasp = _identity_grasp()
        panda, _ = grasp_to_eef_pose(grasp, "panda")
        robotiq140, _ = grasp_to_eef_pose(grasp, "robotiq140")
        assert np.linalg.norm(panda - robotiq140) > 0.05

    def test_approach_waypoint_backs_off_along_the_approach_axis(self):
        grasp = _identity_grasp()
        position, _ = grasp_to_eef_pose(grasp)
        waypoint = approach_waypoint(grasp, standoff=0.12)
        assert np.allclose(np.linalg.norm(position - waypoint), 0.12)
        assert np.allclose(waypoint, position - grasp.approach * 0.12)


class TestGraspSet:
    def test_candidates_are_sorted_best_first(self):
        """The server concatenates per-iteration and per-branch results without
        a global sort, so sorting is our job."""
        poses = np.tile(np.eye(4), (4, 1, 1))
        scores = np.array([0.2, 0.9, 0.5, 0.7])
        grasps = build_grasp_set(poses, scores, "franka_panda", "milk")
        assert list(grasps.scores) == [0.9, 0.7, 0.5, 0.2]
        assert grasps.best.score == 0.9

    def test_width_comes_from_the_gripper_config(self):
        """GraspGen-X returns no per-grasp width; it is the hand's aperture."""
        grasps = build_grasp_set(np.eye(4)[None], np.array([0.5]), "robotiq_2f_85", "can")
        assert grasps[0].width == gripper_geometry("robotiq_2f_85").aperture

    def test_an_empty_result_is_representable(self):
        grasps = build_grasp_set(np.zeros((0, 4, 4)), np.zeros(0), "franka_panda", "milk")
        assert len(grasps) == 0 and grasps.best is None
        assert "no grasps" in grasps.describe()

    def test_axes_follow_the_documented_convention(self):
        """+Z approach, +X closing, anchored at the gripper base."""
        pose = np.eye(4)
        grasp = Grasp6D(pose=pose, score=1.0, gripper="franka_panda", width=0.08)
        assert np.allclose(grasp.approach, [0, 0, 1])
        assert np.allclose(grasp.closing, [1, 0, 0])
        assert np.allclose(grasp.tcp_position(),
                           [0, 0, gripper_geometry("franka_panda").tcp_depth])

    def test_serialisation_round_trips_the_essentials(self):
        grasp = _identity_grasp()
        payload = grasp.as_dict()
        assert np.allclose(payload["pose"], grasp.pose)
        assert payload["gripper"] == "franka_panda"


class TestClientValidation:
    """Input validation, which needs no server."""

    def test_rejects_a_badly_shaped_cloud(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            GraspGenClient().infer(np.zeros((10, 4)))

    def test_rejects_an_empty_cloud(self):
        with pytest.raises(ValueError, match="empty"):
            GraspGenClient().infer(np.zeros((0, 3)))

    def test_rejects_non_finite_points(self):
        cloud = np.zeros((5, 3))
        cloud[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GraspGenClient().infer(cloud)

    def test_rejects_an_unknown_planner(self):
        with pytest.raises(ValueError, match="planner must be"):
            GraspGenClient().infer(np.zeros((5, 3)), planner="magic")
        assert "diffusion" in PLANNERS
