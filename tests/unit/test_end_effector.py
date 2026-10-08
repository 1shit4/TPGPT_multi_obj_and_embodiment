"""The hand on the flange: the Franka Hand unchanged, the Inspire RH56DFX-2R in GraspGen-X's frame.

Offline arithmetic only (no ROS, no GraspGen-X server), in the style of test_hardware.py:
the mount composition, the hand's verdict on a grasp (``judge_hand``), and planning with
the Inspire hand -- in particular the roll of an asymmetric hand, which keeps the
generator's grasp and turns the demonstrating jaw instead.
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
)
from tpgpt.hardware.executor import ExecutorSettings, HardwareExecutor
from tpgpt.hardware.fr3 import HOME_Q, default_model
from tpgpt.hardware.grasp_selection import judge_hand
from tpgpt.hardware.planner import JAW_TURN, Plan, PlanningError, TargetScene, plan
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.sim.keypoints import GraspFrame
from tpgpt.utils.rotations import is_rotation

from tests.unit.test_hardware import synthetic_bundle


@pytest.fixture(scope="module")
def inspire():
    try:
        return inspire_end_effector()
    except FileNotFoundError as exc:                       # no GraspGen-X checkout
        pytest.skip(str(exc))


def _grasp_world(ee, T_tcp):
    """The grasp frame and grasp point in the base frame, with fr3_hand_tcp at ``T_tcp``."""
    return ee.grasp_from_ee(T_tcp[:3, :3]), T_tcp[:3, 3] + T_tcp[:3, :3] @ ee.tool_offset


# ----------------------------------------------------------------------- mount geometry
class TestMount:
    def test_franka_hand_is_the_previous_convention(self):
        R = frames.make_transform(frames.rotation_about_z(0.7))[:3, :3] @ np.diag([1, -1, -1])
        assert np.allclose(FRANKA_HAND.ee_from_grasp(R), frames.grasp_to_fr3_tcp(R))
        assert np.allclose(FRANKA_HAND.tool_offset, 0.0)
        assert FRANKA_HAND.is_parallel_jaw and FRANKA_HAND.symmetric
        assert FRANKA_HAND.graspgen == "franka_panda"
        assert FRANKA_HAND.open_state == pytest.approx(0.08)

    def test_at_home_the_default_mount_hangs_the_hand_palm_towards_the_base(self, inspire):
        """Yaw 135 deg, as seen on the robot (2026-10-05): fingers down, the palm towards the
        robot's base (-X), the thumb and index finger on the robot's left (+Y) -- a right
        hand -- and the wrist face 16 mm below the flange."""
        poses = default_model().link_poses(HOME_Q)
        Rg, pg = _grasp_world(inspire, poses["fr3_hand_tcp"])
        H = inspire.model.link_poses("open")["hand_base_link"]
        assert Rg @ -H[:3, 1] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)    # fingers
        assert Rg @ -H[:3, 0] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)    # palm normal
        wrist = Rg @ H[:3, 3] + pg
        assert wrist == pytest.approx(poses["fr3_link8"][:3, 3] - [0.0, 0.0, 0.016], abs=1e-9)
        tips = inspire.model.fingertip_points("open")
        assert (Rg @ tips["index"] + pg)[1] > (Rg @ tips["pinky"] + pg)[1]
        assert (Rg @ tips["thumb"] + pg)[1] > (Rg @ tips["pinky"] + pg)[1]
        # GraspGen-X's approach is the bisector of fingers-down and the palm's facing.
        # (to 1e-6: the URDF writes its angles to six decimals)
        assert Rg[:, 2] == pytest.approx([-np.sqrt(0.5), 0.0, -np.sqrt(0.5)], abs=1e-6)

    def test_the_grasp_point_from_fr3_hand_tcp(self, inspire):
        # 60 mm out on the palm side (TCP -X since the yaw is 135) and 65 mm along the hand
        # from the TCP, 10 mm across (16 mm adapter).
        assert inspire.tool_offset * 1000 == pytest.approx([-60.10, -10.0, 64.63], abs=0.01)
        assert is_rotation(inspire.grasp_to_ee)

    def test_a_depth_correction_moves_the_grasp_point_back_along_the_approach(self, inspire):
        deeper = inspire_end_effector(calibrated_depth=0.01)
        approach_in_ee = inspire.grasp_to_ee[2, :]          # R_g = R_ee M^T: column 2 of M^T
        assert deeper.tool_offset - inspire.tool_offset == pytest.approx(-0.01 * approach_in_ee)
        assert np.allclose(deeper.grasp_to_ee, inspire.grasp_to_ee)

    def test_both_controller_frames_put_the_hand_in_the_same_place(self):
        kw = dict(mount_yaw_deg=-30.0, adapter_thickness=0.015, lateral=(0.002, -0.001))
        tcp = inspire_end_effector(controller_frame="fr3_hand_tcp", **kw)
        link8 = inspire_end_effector(controller_frame="fr3_link8", **kw)
        R_F, p_F = CONTROLLER_FRAMES["fr3_hand_tcp"]
        T_bF = frames.make_transform(frames.rotation_about_z(0.4) @ np.diag([1, -1, -1]),
                                     [0.4, 0.1, 0.3])
        T_bE = T_bF.copy()
        T_bE[:3, :3], T_bE[:3, 3] = T_bF[:3, :3] @ R_F, T_bF[:3, :3] @ p_F + T_bF[:3, 3]
        Ra, pa = _grasp_world(tcp, T_bE)
        Rb, pb = _grasp_world(link8, T_bF)
        assert Ra == pytest.approx(Rb, abs=1e-12) and pa == pytest.approx(pb, abs=1e-12)

    def test_the_adapter_spans_the_flange_face_to_the_wrist_face(self, inspire):
        adapter = next(b for b in inspire.boxes() if b.name == "body_adapter")
        R_FG = np.asarray(inspire.details["grasp_axes_in_flange"])
        p_FG = np.asarray(inspire.details["grasp_point_in_flange_m"])
        z = (adapter.corners() @ R_FG.T + p_FG)[:, 2]
        assert z.min() == pytest.approx(0.0, abs=1e-9) and z.max() == pytest.approx(0.016, abs=1e-9)

    def test_the_round_trip_rebuilds_the_hand_and_refuses_a_changed_one(self, inspire, tmp_path):
        back = EndEffector.from_dict(inspire.to_dict())
        assert np.allclose(back.tool_offset, inspire.tool_offset)
        assert np.allclose(back.grasp_to_ee, inspire.grasp_to_ee)
        assert back.kind == "multi_finger" and not back.symmetric
        tampered = inspire.to_dict()
        tampered["tool_offset"] = (np.asarray(tampered["tool_offset"]) + 0.002).tolist()
        with pytest.raises(ValueError, match="re-plan"):
            EndEffector.from_dict(tampered)
        with pytest.raises(ValueError, match="side-grasp"):
            EndEffector.from_dict({**inspire.to_dict(), "kind": "inspire_side"})
        assert end_effector_for_plan({}) is FRANKA_HAND

    def test_from_config_reads_the_mount_and_the_description(self):
        ee = inspire_from_config({"inspire_hand": {
            "handedness": "right",
            "mount": {"yaw_deg": -40.0, "adapter_thickness_m": 0.02, "lateral_m": [0.001, 0.0],
                      "measured": True},
            "graspgenx": {"gripper": "inspire_hand", "calibrated_depth_m": 0.005},
            "joint_map": {"index": {"q_at_0": 1.7, "q_at_1000": 0.03, "verified": True}}}})
        spec = ee.model_spec
        assert spec["mount_yaw_deg"] == -40.0 and spec["adapter_thickness"] == 0.02
        assert spec["mount_measured"] and ee.calibrated_depth == 0.005
        assert spec["joint_map"]["index"] == [1.7, 0.03, True]

    def test_only_the_right_hand_and_known_frames(self):
        with pytest.raises(ValueError, match="right hand"):
            inspire_end_effector(handedness="left")
        with pytest.raises(ValueError, match="controller_frame"):
            inspire_end_effector(controller_frame="fr3_link7")

    def test_the_hand_reaches_further_round_the_object_than_the_jaw(self, inspire):
        assert inspire.footprint_margin > FRANKA_HAND.footprint_margin


# ------------------------------------------------------------------- the hand's verdict
def _ball(centre, radius, n=400, seed=0, solid=False):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if solid:
        v *= rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / 3.0)
    return np.asarray(centre) + radius * v


TOP_DOWN = GraspFrame(tcp=np.array([0.5, 0.0, 0.2]), approach=np.array([0.0, 0.0, -1.0]),
                      closing=np.array([1.0, 0.0, 0.0]))


def _at(grasp, p_local):
    return grasp.tcp + grasp.rotation @ np.asarray(p_local, dtype=float)


class TestHandVerdict:
    def test_a_ball_in_the_pocket_is_held_between_thumb_and_fingers(self, inspire):
        why, rec = judge_hand(_ball(TOP_DOWN.tcp, 0.03), TOP_DOWN, -1.0, inspire)
        assert why is None
        assert rec["contact"]["thumb"] and rec["contact"]["index"] and rec["opposed"]
        assert rec["points_in_pocket"] >= 8

    @pytest.mark.parametrize("where, verdict", [
        ((0.3, 0.3, 0.3), "empty"),                         # nothing near the hand
        ((-0.011, -0.010, -0.074), "collides"),             # inside the palm
        ((-0.066, 0.018, -0.016), "lands_on_object"),       # on the open index finger
        ((-0.035, 0.0, -0.010), "no_opposition"),           # only the fingers reach it
    ])
    def test_each_rejection_is_named(self, inspire, where, verdict):
        why, _ = judge_hand(_ball(_at(TOP_DOWN, where), 0.005, solid=True), TOP_DOWN, -1.0, inspire)
        assert why is not None and why.startswith(verdict), why

    def test_the_table_is_checked_open_and_while_closing(self, inspire):
        ball = _ball(TOP_DOWN.tcp, 0.03)
        # Top-down, the open hand reaches ~15 mm below the grasp point, closed ~26.
        why, _ = judge_hand(ball, TOP_DOWN, TOP_DOWN.tcp[2] - 0.015, inspire)
        assert why.startswith("hits_support: the open hand")
        why, _ = judge_hand(ball, TOP_DOWN, TOP_DOWN.tcp[2] - 0.020, inspire)
        assert why.startswith("hits_support: closing")
        assert judge_hand(ball, TOP_DOWN, TOP_DOWN.tcp[2] - 0.035, inspire)[0] is None


# ------------------------------------------------------------------------ planning
@pytest.fixture(scope="module")
def bundle():
    return synthetic_bundle()


def _scene(bundle, grasp):
    p = bundle.placement
    return TargetScene(points=p.points, destination=p.destination,
                       destination_height=p.destination_height,
                       support_height=p.support_height, grasp=grasp)


def _turned(g):
    return GraspFrame(tcp=g.tcp, approach=g.approach, closing=-np.asarray(g.closing))


class TestInspirePlan:
    def test_the_inspire_hand_needs_an_explicit_grasp(self, bundle, inspire):
        with pytest.raises(PlanningError, match="no_grasp"):
            plan(bundle, _scene(bundle, None), end_effector=inspire)

    def test_the_franka_hand_plan_is_unchanged_by_default(self, bundle):
        scene = _scene(bundle, bundle.placement.grasp)
        a = plan(bundle, scene)
        b = plan(bundle, scene, end_effector=FRANKA_HAND)
        assert np.allclose(a.labels.orientations, b.labels.orientations)
        assert a.metadata["end_effector"]["name"] == "panda"
        assert a.metrics["hand_closed_q"] is None

    def test_an_opposed_grasp_turns_the_jaw_not_the_hand(self, bundle, inspire):
        """A grasp whose closing opposes the demonstration's: the sim (and the Franka Hand)
        executes it half-turned; the Inspire hand executes it as given, with the
        demonstrating jaw turned instead. Same map, same path, same carried object."""
        opposed = _turned(bundle.placement.grasp)
        jaw = plan(bundle, _scene(bundle, opposed), end_effector=FRANKA_HAND)
        hand = plan(bundle, _scene(bundle, opposed), end_effector=inspire)
        assert jaw.metrics["roll"] == "target"
        assert hand.metrics["roll"] == "source_jaw_turned"
        # The Franka Hand runs the turned grasp; the Inspire hand runs the generator's.
        assert jaw.grasp.closing == pytest.approx(-opposed.closing)
        assert hand.grasp.closing == pytest.approx(opposed.closing)
        assert hand.metrics["min_det"] == pytest.approx(jaw.metrics["min_det"])
        assert hand.labels.positions == pytest.approx(jaw.labels.positions)
        R_jaw = FRANKA_HAND.grasp_from_ee(jaw.labels.orientations)
        R_hand = inspire.grasp_from_ee(hand.labels.orientations)
        assert R_hand == pytest.approx(R_jaw @ JAW_TURN, abs=1e-9)
        gi, ri = jaw.metrics["grasp_index"], jaw.metrics["release_index"]
        carry_jaw = R_jaw[ri] @ R_jaw[gi].T
        carry_hand = R_hand[ri] @ R_hand[gi].T
        assert carry_hand == pytest.approx(carry_jaw, abs=1e-9)
        assert hand.metrics["orientation_error_grasp_deg"] == pytest.approx(
            jaw.metrics["orientation_error_grasp_deg"], abs=1e-6)

    def test_an_agreeing_grasp_is_not_turned_at_all(self, bundle, inspire):
        hand = plan(bundle, _scene(bundle, bundle.placement.grasp), end_effector=inspire)
        assert hand.metrics["roll"] == "none"
        assert hand.grasp.closing == pytest.approx(bundle.placement.grasp.closing)

    def test_the_plan_records_where_the_fingers_stop(self, bundle, inspire):
        hand = plan(bundle, _scene(bundle, bundle.placement.grasp), end_effector=inspire)
        q = hand.metrics["hand_closed_q"]
        assert len(q) == 6
        assert set(hand.metrics["hand_contacts_predicted"]) == {"pinky", "ring", "middle",
                                                                "index", "thumb"}
        # where each chain meets the object, for a timed close (inspire_hand.timed_speeds)
        fraction = hand.metrics["hand_contact_fraction"]
        assert set(fraction) == set(hand.metrics["hand_contacts_predicted"])
        assert all(0.0 <= f <= 1.0 for f in fraction.values())
        assert hand.metrics["jaw_verdict"] == "not_applicable"
        assert all(is_rotation(R, atol=1e-6) for R in hand.labels.orientations)

    def test_the_executor_commands_the_tcp_that_puts_the_grasp_point_on_the_label(
            self, bundle, inspire, tmp_path):
        p = plan(bundle, _scene(bundle, bundle.placement.grasp), end_effector=inspire)
        p.save(tmp_path / "p")
        loaded = Plan.load(tmp_path / "p")
        hand = end_effector_for_plan(loaded.metadata)
        assert np.allclose(hand.tool_offset, inspire.tool_offset)
        ex = HardwareExecutor(GPPolicy().fit(loaded.labels), loaded.dt,
                              ExecutorSettings(tool_offset=tuple(hand.tool_offset)))
        tcp, R = ex.start_pose
        assert np.allclose(hand.grasp_point(tcp, R)[0], loaded.labels.positions[0], atol=1e-9)


def test_the_configured_inspire_hand_grasps_20_mm_further_back():
    """config/inspire_hand.yaml: calibrated_depth_m -0.02, the depth sweep's result (2026-10-07):
    over a top-down grasp the open hand stops 20 mm higher than GraspGen-X's own pose puts it."""
    import yaml
    from pathlib import Path

    from tpgpt.hardware.end_effector import inspire_from_config
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline"
                              / "config" / "inspire_hand.yaml"))
    try:
        hand = inspire_from_config(cfg)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert hand.calibrated_depth == pytest.approx(-0.02)
    cfg["inspire_hand"]["graspgenx"]["calibrated_depth_m"] = 0.0
    generator = inspire_from_config(cfg)
    R, t = np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.0, 0.05])    # approach straight down
    assert hand.lowest_point(t, R, "open") - generator.lowest_point(t, R, "open") == \
        pytest.approx(0.02, abs=1e-9)
