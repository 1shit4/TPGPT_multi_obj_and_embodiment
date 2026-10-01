"""A commanded fingertip never goes below the support plane."""

import numpy as np
import pytest

from tpgpt.grasp.grasps import FLOOR_MARGIN, Grasp6D, clamp_to_support, finger_reach
from tpgpt.grasp.grippers import gripper_geometry, resolve_pair
from tpgpt.grasp.grasps import gripper_frame

TABLE = 0.80


def _top_down(contact_height, gripper="franka_panda"):
    """A grasp approaching straight down whose contact sits ``contact_height`` above the table."""
    depth = gripper_geometry(gripper).tcp_depth
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])       # +Z of the grasp points down
    pose[2, 3] = TABLE + contact_height + depth     # base is `depth` back along the approach
    return Grasp6D(pose=pose, score=1.0, gripper=gripper, width=0.08)


class TestClampToSupport:
    def test_the_panda_has_a_measured_reach(self):
        assert finger_reach("panda") == pytest.approx(0.008)

    def test_a_grasp_that_clears_is_returned_unchanged(self):
        g = _top_down(0.10)
        c, shift = clamp_to_support(g, TABLE, "panda")
        assert shift == 0.0 and c is g

    def test_a_grasp_into_the_table_is_lifted_by_exactly_the_shortfall(self):
        """Contact 10 mm up; executed tip = 10 - depth 7.5 - reach 8 = -5.5 mm;
        margin 5 -> short by 10.5 mm -> lifted 10.5 mm along the approach."""
        frame = gripper_frame("panda")
        depth, reach = frame["calibrated_depth"], frame["finger_reach"]
        g = _top_down(0.010)
        c, shift = clamp_to_support(g, TABLE, "panda")
        expected = FLOOR_MARGIN - (0.010 - depth - reach)
        assert shift == pytest.approx(expected, abs=1e-9)
        assert c.position[2] - g.position[2] == pytest.approx(expected, abs=1e-9)
        assert np.allclose(c.rotation, g.rotation)
        # and the clamped grasp now clears by the margin exactly
        _, again = clamp_to_support(c, TABLE, "panda")
        assert again == pytest.approx(0.0, abs=1e-9)

    def test_a_sideways_approach_is_lifted_along_the_normal(self):
        depth = gripper_geometry("franka_panda").tcp_depth
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float).T  # approach along +x
        pose[2, 3] = TABLE + 0.002                  # 2 mm up: short of the margin by 3
        pose[0, 3] = -depth
        g = Grasp6D(pose=pose, score=1.0, gripper="franka_panda", width=0.08)
        c, shift = clamp_to_support(g, TABLE, "panda")
        assert shift == pytest.approx(FLOOR_MARGIN - 0.002, abs=1e-9)
        assert c.position[2] - g.position[2] == pytest.approx(shift)
        assert c.position[0] == pytest.approx(g.position[0])

    def test_an_unmeasured_reach_means_no_reach_not_a_guess(self, monkeypatch):
        import tpgpt.grasp.grasps as gr

        monkeypatch.setattr(gr, "gripper_frame", lambda short: None)
        assert finger_reach("panda") == 0.0
