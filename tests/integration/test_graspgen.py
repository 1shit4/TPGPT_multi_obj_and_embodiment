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

CAMERAS = ("agentview", "frontview", "birdview")
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
    def test_conversion_puts_the_tcp_where_the_fingers_close(self, generated, env):
        """Risk-2 acceptance: the converted end-effector position must be the
        fingertip frame, i.e. on the object, not the gripper base which stands
        off by 103-195 mm depending on the hand."""
        from tpgpt.grasp.grasps import grasp_to_eef_pose
        from tpgpt.grasp.grippers import gripper_geometry

        cloud, sets = generated
        for name, grasp_set in sets.items():
            grasp = grasp_set.best
            position, _ = grasp_to_eef_pose(grasp, name)
            depth = gripper_geometry(grasp_set.gripper).tcp_depth
            assert np.isclose(np.linalg.norm(position - grasp.position), depth, atol=1e-9)
            # The TCP should sit near the observed surface, the base should not.
            assert np.linalg.norm(position - cloud.centroid) < np.linalg.norm(
                grasp.position - cloud.centroid
            ), name

    def test_the_approach_axis_survives_the_rotation(self, generated):
        """The alignment rotation is about the approach axis, so it must leave
        that axis untouched -- only the closing direction changes."""
        from tpgpt.grasp.grasps import grasp_to_eef_pose

        _, sets = generated
        for name, grasp_set in sets.items():
            grasp = grasp_set.best
            _, rotation = grasp_to_eef_pose(grasp, name)
            assert np.allclose(rotation[:, 2], grasp.approach, atol=1e-9), name
