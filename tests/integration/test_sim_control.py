"""Simulation backend, scene and Cartesian impedance control (paper Sec. V).

Marked ``sim`` because they build a MuJoCo environment; the unit suite stays
free of the simulator.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    from tpgpt.sim.backend import make_reshelving_env

    environment = make_reshelving_env(seed=0)
    environment.reset()
    yield environment
    environment.close()


@pytest.fixture
def controller(env):
    from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController

    env.reset()
    ctl = CartesianImpedanceController(env)
    ctl.reset()
    return ctl


class TestScene:
    def test_observation_exposes_object_and_goal(self, env):
        obs = env.reset()
        for key in ("product_pos", "product_quat", "goal_pos", "robot0_eef_pos"):
            assert key in obs

    def test_goal_slot_is_reachable_and_above_the_table(self, env):
        obs = env.reset()
        assert obs["goal_pos"][2] > env.table_top
        # The Panda's reachable envelope saturates near x = 0.24 in this arena.
        assert obs["goal_pos"][0] < 0.24

    def test_seeding_is_reproducible(self):
        from tpgpt.sim.backend import managed_env

        def sample(seed):
            with managed_env(seed=seed) as e:
                obs = e.reset()
                return np.concatenate([obs["product_pos"], obs["goal_pos"]])

        assert np.allclose(sample(5), sample(5))
        assert not np.allclose(sample(5), sample(6))

    def test_randomisation_spans_the_table_ii_ranges(self):
        from tpgpt.sim.backend import managed_env

        positions = []
        for seed in range(12):
            with managed_env(seed=seed) as e:
                positions.append(e.reset()["product_pos"])
        positions = np.array(positions)
        spread = positions.max(axis=0) - positions.min(axis=0)
        assert spread[0] > 0.1 and spread[1] > 0.2

    def test_scene_settles(self, env):
        env.reset()
        before = env.product_position.copy()
        for _ in range(40):
            env.step(np.zeros(env.action_dim))
        assert np.linalg.norm(env.product_position - before) < 0.01


class TestCartesianImpedance:
    def test_deflection_under_load_follows_the_commanded_stiffness(self, env, controller):
        """The defining property: ``dx = K^-1 F`` under a known external load."""
        body = env.sim.model.body_name2id("gripper0_right_eef")
        target, rotation, _, _ = controller.eef_state()

        def settle(K, D, force, n_free=150, n_load=250):
            env.sim.data.xfrc_applied[body, :3] = 0
            for _ in range(n_free):
                env.step(controller.action(target, K, D, rotation_desired=rotation))
            free = controller.eef_state()[0]
            for _ in range(n_load):
                env.sim.data.xfrc_applied[body, :3] = force
                env.step(controller.action(target, K, D, rotation_desired=rotation))
            loaded = controller.eef_state()[0]
            env.sim.data.xfrc_applied[body, :3] = 0
            return loaded - free

        force = np.array([10.0, 0.0, 0.0])
        for k in (200.0, 400.0, 800.0):
            K = np.eye(3) * k
            D = np.eye(3) * (2 * np.sqrt(k) * 0.9)
            measured = settle(K, D, force)
            predicted = np.linalg.solve(K, force)
            assert measured[0] == pytest.approx(predicted[0], rel=0.15)

    def test_rotated_stiffness_deflects_off_axis(self, env, controller):
        """The reason this controller exists.

        Transport rotates the stiffness ellipsoid (``K_hat = J_perp K J_perp^T``),
        so a force along one axis produces deflection along another. robosuite's
        variable-impedance OSC takes a diagonal gain vector and cannot represent
        this at all.
        """
        body = env.sim.model.body_name2id("gripper0_right_eef")
        target, rotation, _, _ = controller.eef_state()
        theta = np.pi / 4
        R = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        K = R @ np.diag([200.0, 800.0, 800.0]) @ R.T
        D = R @ np.diag([2 * np.sqrt(200) * 0.9, 2 * np.sqrt(800) * 0.9,
                         2 * np.sqrt(800) * 0.9]) @ R.T
        assert np.abs(K - np.diag(np.diag(K))).max() > 100  # genuinely off-diagonal

        for _ in range(150):
            env.step(controller.action(target, K, D, rotation_desired=rotation))
        free = controller.eef_state()[0]
        force = np.array([10.0, 0.0, 0.0])
        for _ in range(250):
            env.sim.data.xfrc_applied[body, :3] = force
            env.step(controller.action(target, K, D, rotation_desired=rotation))
        measured = controller.eef_state()[0] - free
        env.sim.data.xfrc_applied[body, :3] = 0

        predicted = np.linalg.solve(K, force)
        assert np.allclose(measured, predicted, atol=0.005)
        # The deflection is not parallel to the force.
        assert abs(measured[1]) > 0.3 * abs(measured[0])

    def test_orientation_error_is_zero_for_identical_rotations(self):
        from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController
        from scipy.spatial.transform import Rotation

        R = Rotation.random(random_state=0).as_matrix()
        assert np.allclose(CartesianImpedanceController.orientation_error(R, R), 0.0)

    def test_action_is_normalised_by_the_controller_output_range(self, controller):
        """robosuite rescales actions onto ``output_max``; the two must agree."""
        target, rotation, _, _ = controller.eef_state()
        action = controller.action(target + 1.0, np.eye(3) * 1e6, np.eye(3), rotation_desired=rotation)
        assert np.abs(action[:-1]).max() <= 1.0
