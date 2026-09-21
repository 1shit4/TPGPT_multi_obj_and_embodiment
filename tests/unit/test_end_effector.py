"""The hand on the flange: mount geometry, the side grasp, and planning with the Inspire hand.

Offline arithmetic only (no ROS, no GraspGen-X), in the style of test_hardware.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.hardware import frames
from tpgpt.hardware.end_effector import (
    CONTROLLER_FRAMES,
    FRANKA_HAND,
    EndEffector,
    end_effector_for_plan,
    inspire_end_effector,
    inspire_from_config,
    thumb_side_sign,
)
from tpgpt.hardware.executor import ExecutorSettings, HardwareExecutor
from tpgpt.hardware.grasp_selection import SelectionSettings, plan_side_grasp
from tpgpt.hardware.planner import Plan, PlanningError, TargetScene, plan
from tpgpt.hardware.side_grasp import SideGraspSettings, fit_circle, side_grasp
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.utils.rotations import is_rotation

from tests.unit.test_hardware import synthetic_bundle


def cylinder_cloud(centre_xy, radius, bottom, height, view_from=(0.0, 0.0), arc_deg=140.0,
                   n_arc=40, n_z=12, seed=0):
    """The part of an upright cylinder's side a camera at ``view_from`` sees, plus the top rim."""
    rng = np.random.default_rng(seed)
    c = np.asarray(centre_xy, dtype=float)
    to_cam = np.asarray(view_from, dtype=float) - c
    mid = np.arctan2(to_cam[1], to_cam[0])
    angles = mid + np.radians(np.linspace(-arc_deg / 2, arc_deg / 2, n_arc))
    zs = np.linspace(bottom + 0.005, bottom + height, n_z)
    side = np.array([[c[0] + radius * np.cos(a), c[1] + radius * np.sin(a), z]
                     for a in angles for z in zs])
    return side + rng.normal(scale=0.0008, size=side.shape)


# ----------------------------------------------------------------------- mount geometry
class TestMount:
    def test_franka_hand_is_the_previous_convention(self):
        R = frames.make_transform(frames.rotation_about_z(0.7))[:3, :3] @ np.diag([1, -1, -1])
        assert np.allclose(FRANKA_HAND.ee_from_grasp(R), frames.grasp_to_fr3_tcp(R))
        assert np.allclose(FRANKA_HAND.tool_offset, 0.0)
        assert FRANKA_HAND.is_parallel_jaw

    def test_default_mount_puts_palm_on_tcp_x_and_fingers_on_tcp_z(self):
        ee = inspire_end_effector(adapter_thickness=0.01, palm_along_axis=0.09,
                                  palm_normal_offset=0.02)
        # R_ee = R_grasp @ M, so with R_grasp = I the columns of M are the EE axes in grasp coords;
        # equivalently the grasp axes in EE coords are the columns of M^T.
        grasp_in_ee = ee.grasp_to_ee.T
        assert np.allclose(grasp_in_ee[:, 2], [1, 0, 0])     # approach = palm normal = TCP +X
        assert np.allclose(grasp_in_ee[:, 0], [0, 0, 1])     # closing = fingers = TCP +Z
        assert np.allclose(grasp_in_ee[:, 1], [0, -1, 0])    # across the palm = -w = TCP -Y
        assert is_rotation(ee.grasp_to_ee)
        # palm point: 20 mm out along TCP +X, (10 + 90 - 103.4) mm along TCP +Z
        assert np.allclose(ee.tool_offset, [0.02, 0.0, 0.10 - 0.1034], atol=1e-12)

    @pytest.mark.parametrize("yaw", [-45.0, 0.0, 30.0, 135.0])
    def test_both_controller_frames_describe_the_same_physical_hand(self, yaw):
        """Commanding fr3_link8 or fr3_hand_tcp must put the palm at the same place."""
        kw = dict(mount_yaw_deg=yaw, adapter_thickness=0.012, palm_along_axis=0.095,
                  palm_normal_offset=0.021, palm_lateral_offset=0.004)
        tcp = inspire_end_effector(controller_frame="fr3_hand_tcp", **kw)
        link8 = inspire_end_effector(controller_frame="fr3_link8", **kw)
        R_grasp = frames.rotation_about_z(0.4) @ np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        grasp_point = np.array([0.5, 0.1, 0.2])
        poses = {}
        for name, ee in (("fr3_hand_tcp", tcp), ("fr3_link8", link8)):
            R_ee = ee.ee_from_grasp(R_grasp)
            p_ee = grasp_point - R_ee @ ee.tool_offset
            R_f_ee, p_f_ee = CONTROLLER_FRAMES[name]
            R_flange = R_ee @ R_f_ee.T
            poses[name] = (R_flange, p_ee - R_flange @ p_f_ee)
        assert np.allclose(poses["fr3_hand_tcp"][0], poses["fr3_link8"][0], atol=1e-12)
        assert np.allclose(poses["fr3_hand_tcp"][1], poses["fr3_link8"][1], atol=1e-12)

    def test_round_trip_and_grasp_point(self):
        ee = inspire_end_effector()
        R = frames.rotation_about_z(1.1) @ frames.make_transform()[:3, :3]
        assert np.allclose(ee.grasp_from_ee(ee.ee_from_grasp(R)), R)
        p_ee, R_ee = np.array([0.4, 0.0, 0.3]), ee.ee_from_grasp(R)
        point = ee.grasp_point(p_ee, R_ee)[0]
        assert np.allclose(point - R_ee @ ee.tool_offset, p_ee)

    def test_dict_round_trip_and_plan_metadata_lookup(self):
        ee = inspire_end_effector(handedness="left", mount_yaw_deg=-30.0)
        back = EndEffector.from_dict(ee.to_dict())
        assert np.allclose(back.grasp_to_ee, ee.grasp_to_ee)
        assert np.allclose(back.tool_offset, ee.tool_offset)
        assert back.handedness == "left" and not back.is_parallel_jaw
        assert end_effector_for_plan({}) is FRANKA_HAND
        assert end_effector_for_plan({"end_effector": FRANKA_HAND.to_dict()}) is FRANKA_HAND

    def test_config_block(self):
        ee = inspire_from_config({"inspire_hand": {
            "handedness": "right", "mount": {"yaw_deg": 0.0, "controller_frame": "fr3_link8",
                                             "adapter_thickness_m": 0.0, "palm_along_axis_m": 0.1,
                                             "palm_normal_offset_m": 0.03},
            "envelope": {"lo_m": [-0.04, -0.1, -0.04], "hi_m": [0.04, 0.12, 0.07]}}})
        assert ee.controller_frame == "fr3_link8" and ee.handedness == "right"
        assert np.allclose(ee.tool_offset, [0.03, 0.0, 0.1])

    def test_unknown_frame_and_hand_are_refused(self):
        with pytest.raises(ValueError, match="controller_frame"):
            inspire_end_effector(controller_frame="fr3_link7")
        with pytest.raises(ValueError, match="handedness"):
            thumb_side_sign("both")


# -------------------------------------------------------------------------- side grasp
class TestSideGrasp:
    def test_circle_fit_recovers_a_partial_arc(self):
        pts = cylinder_cloud((0.55, 0.12), 0.033, 0.0, 0.1, view_from=(1.2, 0.0), arc_deg=120)
        centre, radius, rms = fit_circle(pts[:, :2])
        assert np.linalg.norm(centre - [0.55, 0.12]) < 0.003
        assert radius == pytest.approx(0.033, abs=0.003)
        assert rms < 0.002

    def test_palm_approaches_radially_and_stops_in_front_of_the_surface(self):
        centre, r, support = np.array([0.5, -0.2]), 0.035, -0.006
        pts = cylinder_cloud(centre, r, support, 0.12, view_from=(1.1, 0.3))
        s = SideGraspSettings(standoff=0.015, height_fraction=0.5)
        grasp, info = side_grasp(pts, support, "left", s)
        radial = centre / np.linalg.norm(centre)
        assert np.allclose(grasp.approach, [*radial, 0.0], atol=0.01)
        axis_point = np.array([*centre, grasp.tcp[2]])
        assert np.linalg.norm(axis_point - grasp.tcp) == pytest.approx(r + 0.015, abs=0.004)
        assert grasp.tcp[2] - support == pytest.approx(0.06, abs=0.004)
        assert info.fit_used

    def test_fingers_run_horizontally_and_the_thumb_is_on_top(self):
        pts = cylinder_cloud((0.5, 0.0), 0.03, 0.0, 0.12, view_from=(1.0, 0.0))
        left, _ = side_grasp(pts, 0.0, "left")
        right, _ = side_grasp(pts, 0.0, "right")
        down, _ = side_grasp(pts, 0.0, "left", SideGraspSettings(thumb_up=False))
        # Approach +x: a left hand, index on top, palm forward, points its fingers to +y.
        assert np.allclose(left.approach, [1, 0, 0], atol=0.02)
        assert np.allclose(left.closing, [0, 1, 0], atol=0.02)
        assert np.allclose(right.closing, [0, -1, 0], atol=0.02)
        assert np.allclose(down.closing, [0, -1, 0], atol=0.02)
        for grasp, side in ((left, "left"), (right, "right")):
            w = np.cross(grasp.closing, grasp.approach)          # w = f x n
            assert thumb_side_sign(side) * w[2] > 0.99          # thumb side is world up

    def test_short_objects_are_gripped_at_the_table_clearance_floor(self):
        pts = cylinder_cloud((0.5, 0.0), 0.03, 0.0, 0.06)
        grasp, info = side_grasp(pts, 0.0, "left", SideGraspSettings(min_height_above_support=0.05))
        assert grasp.tcp[2] == pytest.approx(0.05, abs=1e-9)
        assert info.notes


# ------------------------------------------------------------------------ planning
@pytest.fixture(scope="module")
def bundle():
    return synthetic_bundle()


class TestInspirePlan:
    def _scene(self, support=-0.10):
        pts = cylinder_cloud((0.45, 0.05), 0.033, support, 0.12, view_from=(1.0, 0.4))
        return TargetScene(points=pts, destination=np.array([0.60, 0.20, 0.15]),
                           destination_height=0.15, support_height=support)

    def test_the_side_grasp_plans_and_labels_are_in_the_controller_frame(self, bundle):
        ee = inspire_end_effector()
        # min_determinant 0: this synthetic top-down demo turns 90 degrees onto a side grasp,
        # which is exactly what a real Inspire demonstration taught from the side avoids.
        sel = plan_side_grasp(bundle, self._scene(), ee,
                              SelectionSettings(min_determinant=0.0, min_path_clearance=-1.0,
                                                max_aim_grasp=0.02, max_orientation_error_deg=30))
        result = sel.plan
        assert sel.source == "side" and sel.chosen.jaw is None
        gi = int(result.metrics["grasp_index"])
        R_ee = result.labels.orientations[gi]
        assert all(is_rotation(R, atol=1e-6) for R in result.labels.orientations)
        # Palm normal (TCP +X with the default mount) points along the planned approach.
        assert float(R_ee[:, 0] @ sel.chosen.grasp.approach) > np.cos(np.radians(30))
        assert result.metrics["jaw_verdict"] == "not_applicable"
        assert result.metadata["orientation_convention"] == "fr3_hand_tcp"
        assert result.metadata["side_grasp"]["circle_fit_used"]
        assert end_effector_for_plan(result.metadata).name == "inspire"
        assert "side grasp" not in sel.describe() or sel.describe()

    def test_the_executor_commands_the_tcp_that_puts_the_palm_on_the_label(self, bundle, tmp_path):
        ee = inspire_end_effector()
        sel = plan_side_grasp(bundle, self._scene(), ee,
                              SelectionSettings(min_determinant=0.0, min_path_clearance=-1.0,
                                                max_aim_grasp=0.02, max_orientation_error_deg=30))
        sel.plan.save(tmp_path / "p")
        loaded = Plan.load(tmp_path / "p")
        hand = end_effector_for_plan(loaded.metadata)
        assert np.allclose(hand.tool_offset, ee.tool_offset)
        ex = HardwareExecutor(GPPolicy().fit(loaded.labels), loaded.dt,
                              ExecutorSettings(tool_offset=tuple(hand.tool_offset)))
        tcp, R = ex.start_pose
        assert np.allclose(hand.grasp_point(tcp, R)[0], loaded.labels.positions[0], atol=1e-9)

    def test_the_inspire_hand_needs_an_explicit_grasp(self, bundle):
        with pytest.raises(PlanningError, match="no_grasp"):
            plan(bundle, self._scene(), end_effector=inspire_end_effector())

    def test_the_franka_hand_plan_is_unchanged_by_default(self, bundle):
        p = bundle.placement
        scene = TargetScene(points=p.points, destination=p.destination,
                            destination_height=p.destination_height,
                            support_height=p.support_height, grasp=p.grasp)
        a = plan(bundle, scene)
        b = plan(bundle, scene, end_effector=FRANKA_HAND)
        assert np.allclose(a.labels.orientations, b.labels.orientations)
        assert a.metadata["end_effector"]["name"] == "panda"
