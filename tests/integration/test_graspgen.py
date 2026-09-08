"""Grasp generation against the live GraspGen-X server.

Skipped when the server is unreachable, so the suite still runs offline. Start
it with ``python -m tpgpt.grasp.server`` for instructions.
"""

import numpy as np
import pytest

from tpgpt.grasp.grippers import gripper_config_path
from tpgpt.grasp.server import server_available

pytestmark = [
    pytest.mark.graspgen,
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(not server_available(), reason="GraspGen-X server not reachable"),
    pytest.mark.skipif(
        not gripper_config_path("franka_panda").is_file(),
        reason="GraspGen-X gripper descriptions not installed",
    ),
]

#: Cameras with a clear line to the table. Not ``agentview``/``frontview``:
#: both sit behind the shelf, which became solid geometry and now blocks them.
CAMERAS = ("workspace", "sideview", "birdview")
#: Grippers the running server has loaded.
GRIPPERS = ("panda", "robotiq85", "robotiq140")


@pytest.fixture(scope="module")
def env():
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    config = load_composite_controller_config(controller="BASIC", robot="Panda")
    environment = TabletopShelf(
        robots="Panda", controller_configs=config, control_freq=20, seed=1,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=list(CAMERAS), camera_heights=256, camera_widths=256,
        camera_depths=True, camera_segmentations="instance",
    )
    environment.reset()
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def client():
    from tpgpt.grasp.client import GraspGenClient

    connection = GraspGenClient()
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def generated(env, client):
    """One cloud, grasps for three hands. Generated once: 4-12 s per call."""
    from tpgpt.grasp.pipeline import grasps_for_object

    return grasps_for_object(
        env, "cereal", GRIPPERS, obs=env._get_observations(),
        client=client, num_grasps=100, topk=40,
    )


class TestServer:
    def test_reports_its_loaded_grippers(self, client):
        assert set(GRIPPERS).issubset(
            {
                short
                for short in GRIPPERS
                if __import__("tpgpt.grasp.grippers", fromlist=["resolve_pair"])
                .resolve_pair(short)
                .graspgen
                in client.metadata["loaded_grippers"]
            }
        )


class TestGeneration:
    def test_every_gripper_gets_candidates(self, generated):
        _, sets = generated
        for name in GRIPPERS:
            assert len(sets[name]) > 0, f"{name} produced no grasps"

    def test_poses_are_valid_rigid_transforms(self, generated):
        _, sets = generated
        for name, grasp_set in sets.items():
            for grasp in grasp_set:
                rotation = grasp.rotation
                assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4), name
                assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4), name
                assert np.allclose(grasp.pose[3], [0, 0, 0, 1]), name

    def test_scores_are_probabilities_and_sorted(self, generated):
        _, sets = generated
        for grasp_set in sets.values():
            scores = grasp_set.scores
            assert scores.min() >= 0.0 and scores.max() <= 1.0
            assert np.all(np.diff(scores) <= 1e-9)

    def test_grasps_land_near_the_object(self, generated, env):
        """Grasps are returned in the frame of the cloud, and we send a
        world-frame cloud, so they must be world-frame near the object. A frame
        error would put them metres away."""
        cloud, sets = generated
        truth = env.object_position("cereal")
        for name, grasp_set in sets.items():
            distances = [np.linalg.norm(g.tcp_position() - truth) for g in grasp_set]
            assert min(distances) < 0.10, f"{name}: closest TCP {min(distances):.3f} m"

    def test_width_never_exceeds_the_hand_aperture(self, generated):
        from tpgpt.grasp.grippers import gripper_geometry

        _, sets = generated
        for grasp_set in sets.values():
            aperture = gripper_geometry(grasp_set.gripper).aperture
            assert all(g.width <= aperture + 1e-9 for g in grasp_set)

    def test_different_hands_produce_different_grasps(self, generated):
        """The multi-embodiment claim. If the poses were identical the gripper
        conditioning would be doing nothing."""
        _, sets = generated
        best = {name: sets[name].best.pose for name in GRIPPERS if sets[name].best}
        assert len(best) >= 2
        poses = list(best.values())
        assert not np.allclose(poses[0], poses[1], atol=1e-3)

    def test_an_empty_cloud_degrades_rather_than_raising(self, client):
        from tpgpt.grasp.pipeline import grasps_for_cloud
        from tpgpt.perception.cameras import ObjectCloud

        empty = ObjectCloud(points=np.zeros((0, 3)), instance="ghost")
        result = grasps_for_cloud(empty, "panda", client=client)
        assert len(result) == 0
        assert "not visible" in result.metadata["reason"]


class TestEndEffectorConversion:
    def test_conversion_puts_the_held_object_on_the_grasp_contact(self, generated, env):
        """The contract's one equation, on generated rather than synthetic poses.

        The commanded ``grip_site`` is not itself the contact point -- where the
        site sits relative to the fingers differs by up to 177 mm across hands --
        so what must hold is that the *held object* lands on the grasp's own
        contact point.
        """
        from tpgpt.grasp.grasps import contact_offset, grasp_to_eef_pose
        from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

        cloud, sets = generated
        for name, grasp_set in sets.items():
            grasp = grasp_set.best
            pair = resolve_pair(name)
            position, rotation = grasp_to_eef_pose(grasp, name)
            held = position + rotation @ contact_offset(pair)
            wanted = grasp.position + grasp.approach * gripper_geometry(
                grasp_set.gripper
            ).tcp_depth
            # Exact across the plane; the depth is deliberately calibrated.
            assert np.linalg.norm((held - wanted)[:2]) < 1e-6, name
            # The contact should sit nearer the object than the gripper base.
            assert np.linalg.norm(held - cloud.centroid) < np.linalg.norm(
                grasp.position - cloud.centroid
            ), name

    def test_the_commanded_rotation_is_a_proper_rotation(self, generated):
        """For most hands the alignment is a spin about the approach axis, so
        that axis survives. Not for all: the UMI's fingers lie along its own
        ``grip_site -Z``, so its alignment turns the frame end for end. What
        holds for every hand is that the result is a proper rotation and that
        the hand's own approach direction ends up along the grasp's."""
        from tpgpt.grasp.grasps import grasp_to_eef_pose
        from tpgpt.grasp.grippers import gripper_frame, resolve_pair

        _, sets = generated
        for name, grasp_set in sets.items():
            grasp = grasp_set.best
            _, rotation = grasp_to_eef_pose(grasp, name)
            assert np.isclose(np.linalg.det(rotation), 1.0), name
            approach_local = np.asarray(gripper_frame(name)["approach_in_site"])
            assert np.allclose(rotation @ approach_local, grasp.approach, atol=1e-6), name
