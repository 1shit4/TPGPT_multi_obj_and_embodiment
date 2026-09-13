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
    GripperPair,
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
    def test_every_gripper_frame_is_measured(self):
        from tpgpt.grasp.grippers import MEASURED_PAIRS

        assert set(MEASURED_PAIRS) == set(GRIPPER_PAIRS)

    def test_verified_means_it_lifts_not_merely_that_it_converts(self):
        """Two hands convert cleanly and lift nothing.

        Keeping the two lists separate is what stops a hand that cannot execute
        a grasp from silently joining a campaign and looking like a transport
        failure.

        **The Inspire hand** does not actuate: its fingers travel 0.7 to 6.5 mm
        against 29 to 90 mm for every other hand, so a single open/close command
        never closes it on anything.

        **The UMI** was verified until ROBOTICS_NOTES 7.32 and is not any more.
        Re-calibrated on a scene whose objects actually settle, it has no
        working depth band at all: 0 of 13 swept offsets lift the reference can,
        the best lift is 5.4 mm against a 50 mm threshold, and most samples are
        *negative* -- the hand pushes the can down. Its stored 0.030 offset had
        been measured in a scene where objects were dropped 25 mm and ejected by
        an interpenetrating arm. It is also the only hand that cannot be placed
        at the shared home pose, reaching 0 of 48 candidates where the other
        eight reach 48 of 48.
        """
        assert "panda" in VERIFIED_PAIRS
        for excluded in ("inspire", "umi"):
            assert excluded in GRIPPER_PAIRS, f"{excluded} left the registry"
            assert excluded not in VERIFIED_PAIRS, (
                f"{excluded} converts but does not lift; it must not be verified"
            )
        assert len(VERIFIED_PAIRS) == 7

    def test_an_axis_aligned_gripper_keeps_its_approach_and_closing_axes(self):
        """The Panda's jaws close along grip_site X with Z the approach -- the
        same basis GraspGen-X emits.

        Its measured alignment is a half turn about the approach axis rather
        than the identity, because the sign of a jaw axis is arbitrary: a
        parallel gripper closing along +X and along -X is the same grasp, and
        the measurement has no reason to prefer one. So the invariant is that
        the approach axis survives and the closing axis stays in the plane, not
        that the matrix is the identity.
        """
        rotation = alignment_rotation(GRIPPER_PAIRS["panda"])
        assert np.allclose(rotation @ [0, 0, 1.0], [0, 0, 1.0], atol=1e-3)
        assert abs((rotation @ [1.0, 0, 0])[2]) < 1e-3
        assert abs(abs((rotation @ [1.0, 0, 0])[0]) - 1.0) < 1e-3

    def test_off_axis_grippers_rotate_about_the_approach_axis(self):
        rotation = alignment_rotation(GRIPPER_PAIRS["xarm"])
        assert not np.allclose(rotation, np.eye(3))
        # The XArm closes along Y, so its alignment is a quarter turn about the
        # approach axis and leaves that axis alone.
        assert np.allclose(rotation @ np.array([0, 0, 1.0]), [0, 0, 1.0], atol=1e-3)
        assert np.allclose(np.linalg.det(rotation), 1.0)

    def test_a_flipped_gripper_reverses_its_approach_axis(self):
        """The UMI's fingers lie along grip_site -Z, so its alignment must turn
        the frame end for end. A rotation purely about Z could not, which is
        why the contract stopped being an angle and became a full basis."""
        rotation = alignment_rotation(GRIPPER_PAIRS["umi"])
        assert (rotation @ np.array([0, 0, 1.0]))[2] < -0.9
        assert np.allclose(np.linalg.det(rotation), 1.0)

    def test_unverified_grippers_refuse_rather_than_guess(self):
        """A wrong rotation about the approach axis fails silently -- the pose
        still looks plausible while closing across the wrong dimension.

        Every gripper in the registry is now measured, so this uses an
        explicitly unmeasured pair. The guard still has to hold: it is what
        stops a newly added hand from silently defaulting to identity.
        """
        unmeasured = GripperPair("SomeNewHand", "some_new_hand", ("Panda",), None)
        assert not unmeasured.frame_verified
        with pytest.raises(ValueError, match="has not been measured"):
            alignment_rotation(unmeasured)

    def test_every_registered_gripper_can_be_converted(self):
        """Nine hands, three kinematic families, no refusals."""
        for short, pair in GRIPPER_PAIRS.items():
            rotation = alignment_rotation(pair)
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9), short
            assert np.isclose(np.linalg.det(rotation), 1.0), short

    def test_multi_finger_hands_are_flagged_as_such(self):
        """One closing axis is an approximation for a hand, not a description.

        The measurement reports how well it holds: every parallel jaw scores
        553 or above, every multi-finger hand under 7.
        """
        assert not GRIPPER_PAIRS["robotiq3f"].single_axis
        assert not GRIPPER_PAIRS["inspire"].single_axis
        assert GRIPPER_PAIRS["panda"].single_axis
        assert GRIPPER_PAIRS["yumi"].single_axis

    def test_conversion_puts_the_object_at_the_grasp_contact_point(self):
        """The invariant the whole contract exists to hold.

        Wherever ``grip_site`` ends up, the point where *this hand* holds an
        object has to land on the grasp's own contact point. That is one
        equation, and it is the same one for a Panda, a three-finger Robotiq
        and a hand whose site sits at its wrist.
        """
        from tpgpt.grasp.grasps import contact_offset

        for short, pair in GRIPPER_PAIRS.items():
            grasp = _identity_grasp()
            grasp.gripper = pair.graspgen
            position, rotation = grasp_to_eef_pose(grasp, short)
            held = position + rotation @ contact_offset(pair)
            wanted = grasp.position + grasp.approach * gripper_geometry(
                pair.graspgen
            ).tcp_depth
            assert np.allclose(held, wanted, atol=1e-9), short

    def test_a_hand_can_be_named_by_its_registry_key(self):
        """Passing the string ``"panda"`` must mean the same as passing the pair.

        The lookup behind these is by object *identity*, so a string used to
        fall through to "never measured" and return a zero offset -- a
        perfectly plausible number, which made a 41 mm frame error read as the
        arm missing its target.
        """
        from tpgpt.grasp.grasps import alignment_rotation, contact_offset

        for short, pair in GRIPPER_PAIRS.items():
            assert np.allclose(contact_offset(short), contact_offset(pair)), short
            if pair.frame_verified:
                assert np.allclose(
                    alignment_rotation(short), alignment_rotation(pair)
                ), short

    def test_a_measured_hand_never_reports_a_zero_offset(self):
        """Zero means "not measured". Every verified hand has a real number."""
        from tpgpt.grasp.grasps import contact_offset

        for short, pair in GRIPPER_PAIRS.items():
            if pair.frame_verified:
                assert np.linalg.norm(contact_offset(short)) > 1e-4, short

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
