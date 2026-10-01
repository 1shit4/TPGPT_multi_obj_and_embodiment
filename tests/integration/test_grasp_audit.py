"""The observed checker must leave a live simulator's state unchanged."""
import numpy as np
import pytest

pytest.importorskip("robosuite")
from tpgpt.grasp.grippers import gripper_config_path

pytestmark = [pytest.mark.sim, pytest.mark.skipif(
    not gripper_config_path("franka_panda").is_file(), reason="gripper descriptions unavailable")]


def test_observed_path_restores_live_simulator():
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf
    from tpgpt.grasp.filters import path_feasibility_observed
    from tpgpt.grasp.grasps import contact_offset, to_grasp_convention
    from tpgpt.grasp.grippers import resolve_pair
    env = TabletopShelf(robots="Panda", objects=("can",),
                        has_offscreen_renderer=False, use_camera_obs=False, seed=0)
    try:
        env.reset()
        data = env.sim.data
        grip = env.robots[0].gripper["right"]
        site = env.sim.model.site_name2id(grip.naming_prefix + "grip_site")
        R = np.asarray(data.site_xmat[site]).reshape(3, 3).copy()
        p = np.asarray(data.site_xpos[site]).copy() + R @ contact_offset("panda")
        before = {key: np.asarray(getattr(data, key)).copy() for key in ("qpos", "qvel", "ctrl")}
        time = float(data.time)
        result = path_feasibility_observed(env, np.tile(p, (3, 1)),
            np.tile(to_grasp_convention(R, "panda"), (3, 1, 1)), resolve_pair("panda"),
            np.empty((0, 3)), np.empty((0, 3)), 1, 2, critical_waypoints=True, validate_entry=True)
        assert result["checked_waypoints"] == [0, 1, 2]
        assert result["entry_connection"]["checked"]
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(data, key), value)
        assert float(data.time) == time
    finally:
        env.close()


def test_post_settling_rejects_an_object_left_on_the_table():
    from robosuite.controllers import load_composite_controller_config
    from tpgpt.experiments.pipeline import build_scene
    from tpgpt.experiments.settling import observe_settling
    from tpgpt.sim.replay import make_position_controller_config
    config = make_position_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda"))
    env = build_scene(("can",), controller_config=config, camera_size=32)
    try:
        before = env.object_position("can").copy()
        result = observe_settling(env, "can", "top_middle", seconds=1.)
        assert not result["passed"] and not result["supported"]
        assert result["released"]
        np.testing.assert_allclose(env.object_position("can"), before, atol=.003)
    finally:
        env.close()


def test_disabling_post_selection_camera_observations_preserves_dynamics():
    from tpgpt.experiments.pipeline import build_scene
    from tpgpt.experiments.forensics import disable_execution_cameras
    trajectories = []
    for disabled in (False, True):
        np.random.seed(123)
        env = build_scene(("can",), camera_size=32, seed=0)
        try:
            assert any(o.is_enabled() and o.modality == "image" for o in env._observables.values())
            if disabled:
                disable_execution_cameras(env)
                assert not any(o.is_enabled() and o.modality == "image" for o in env._observables.values())
            states = [env.sim.data.qpos.copy()]
            for _ in range(12):
                env.step(np.zeros(env.action_dim))
                states.append(env.sim.data.qpos.copy())
            trajectories.append(np.asarray(states))
        finally:
            env.close()
    np.testing.assert_array_equal(*trajectories)
def test_explicit_development_placement_moves_shelf_and_object():
    from tpgpt.experiments.pipeline import build_scene
    import numpy as np

    env = build_scene(("meat",), world="ycb", pick_pose=(-.22, -.10, 60.),
                      shelf_translation=(-.04, .02, 0.))
    try:
        np.testing.assert_allclose(env.slot_poses()["top_middle"][:2], [.14, .02])
        np.testing.assert_allclose(env.object_position("meat")[:2], [-.22, -.10], atol=.015)
        assert env._recorded_settled_state() is None
    finally:
        env.close()


def test_explicit_rest_orientation_is_seated_and_physically_free():
    from pathlib import Path
    if not Path("assets/ycb_mjcf/brick.xml").exists():
        pytest.skip("optional foam-brick asset is not downloaded")
    from tpgpt.experiments.pipeline import build_scene
    quat = np.array([.6852912981, .7156159151, .0883992621, .1022510112])
    env = build_scene(("brick",), world="ycb", pick_pose=(-.23, -.14, 50.),
                      object_rest_quat=2 * quat, camera_size=32)
    try:
        assert env._recorded_settled_state() is None
        np.testing.assert_allclose(np.linalg.norm(env.object_rest_quat), 1.)
        vertices = env._geom_vertices(env.object_body_ids["brick"])
        assert .07 < np.ptp(vertices[:, 2]) < .09
        # The historical table_top property is the shelf's height reference,
        # 25 mm above the table collision surface; inspect physical geometry.
        table = env.sim.model.geom_name2id("table_collision")
        surface = env.sim.data.geom_xpos[table, 2] + env.sim.model.geom_size[table, 2]
        assert abs(vertices[:, 2].min() - surface) < .003
        joint = env.sim.model.joint_name2id(env.objects[0].joints[0])
        assert env.sim.model.jnt_type[joint] == 0  # MuJoCo free joint
    finally:
        env.close()
