"""The GraspGen-X to robosuite frame contract, verified in physics.

Deliberately uses a **synthesised** grasp rather than a generated one, so it
tests the conversion and nothing else. A generated candidate that fails to lift
could be failing for any number of reasons -- unreachable approach, collision
with the table, a pose derived from a one-sided point cloud -- and none of those
would tell you whether ``grasp_to_eef_pose`` is correct.

Needs no GraspGen-X server: the grasp is constructed, not requested.
"""

import numpy as np
import pytest

from tpgpt.grasp.grasps import Grasp6D, approach_waypoint, grasp_to_eef_pose
from tpgpt.grasp.grippers import gripper_config_path, gripper_geometry, resolve_pair

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        not gripper_config_path("franka_panda").is_file(),
        reason="GraspGen-X gripper descriptions not installed",
    ),
]


def synthetic_top_down(contact, yaw, graspgen_name):
    """A grasp approaching straight down, closing along ``yaw``.

    Built in GraspGen-X's own convention: ``+Z`` approach, ``+X`` closing,
    anchored at the gripper base, which sits ``tcp_depth`` back along the
    approach axis from the contact point.
    """
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
    pose[:3, 3] = contact - approach * gripper_geometry(graspgen_name).tcp_depth
    return Grasp6D(
        pose=pose, score=1.0, gripper=graspgen_name,
        width=gripper_geometry(graspgen_name).aperture,
    )


def execute_grasp(env, controller, grasp, gripper, stiffness=600.0):
    """Approach, close, lift. Returns the height the object gained."""
    position, rotation = grasp_to_eef_pose(grasp, gripper)
    pre_grasp = approach_waypoint(grasp, standoff=0.12, gripper=gripper)
    K = np.eye(3) * stiffness
    D = np.eye(3) * (2 * np.sqrt(stiffness) * 0.9)

    def drive(target, steps, gripper_command, settle=0):
        start = controller.eef_state()[0]
        for step in range(steps):
            alpha = (step + 1) / steps
            env.step(controller.action(
                start + alpha * (target - start), K, D, gripper=gripper_command,
                rotation_desired=rotation, rotational_stiffness=60.0,
                rotational_damping=12.0,
            ))
        for _ in range(settle):
            env.step(controller.action(
                target, K, D, gripper=gripper_command, rotation_desired=rotation,
                rotational_stiffness=60.0, rotational_damping=12.0,
            ))

    drive(pre_grasp, 45, -1.0, settle=10)
    drive(position, 40, -1.0, settle=25)
    for _ in range(25):
        env.step(controller.action(
            position, K, D, gripper=1.0, rotation_desired=rotation,
            rotational_stiffness=60.0, rotational_damping=12.0,
        ))
    drive(position + np.array([0.0, 0.0, 0.15]), 45, 1.0, settle=10)


def make_env(gripper_short):
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.controllers.cartesian_impedance import make_torque_controller_config
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    pair = resolve_pair(gripper_short)
    config = make_torque_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    return TabletopShelf(
        robots="Panda", gripper_types=pair.robosuite, controller_configs=config,
        objects=("can", "milk"), control_freq=20, seed=2,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )


@pytest.mark.parametrize("gripper,obj", [("panda", "can"), ("robotiq85", "can")])
def test_a_converted_grasp_actually_picks_the_object_up(gripper, obj):
    """The acceptance test for the frame contract.

    Two hands with different TCP depths (103 vs 136 mm) and different jaw
    apertures. If the conversion dropped the depth term or rotated about the
    wrong axis, the fingers would close beside or above the object.
    """
    from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController

    env = make_env(gripper)
    try:
        env.reset()
        controller = CartesianImpedanceController(env)
        controller.reset()
        start_height = env.object_position(obj)[2]
        contact = env.object_position(obj) + np.array([0.0, 0.0, 0.02])
        grasp = synthetic_top_down(contact, 0.0, resolve_pair(gripper).graspgen)

        execute_grasp(env, controller, grasp, gripper)
        lifted = env.object_position(obj)[2] - start_height
        assert lifted > 0.05, f"{gripper} on {obj} lifted only {lifted * 100:.1f} cm"
    finally:
        env.close()


def test_the_conversion_reaches_the_commanded_pose():
    """Separates a conversion error from a tracking error: whatever the
    contract produces, the arm must be able to get there."""
    from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController

    env = make_env("panda")
    try:
        env.reset()
        controller = CartesianImpedanceController(env)
        controller.reset()
        contact = env.object_position("can") + np.array([0.0, 0.0, 0.02])
        grasp = synthetic_top_down(contact, 0.0, "franka_panda")
        position, rotation = grasp_to_eef_pose(grasp, "panda")

        K, D = np.eye(3) * 600, np.eye(3) * (2 * np.sqrt(600) * 0.9)
        for _ in range(140):
            env.step(controller.action(
                position, K, D, gripper=-1.0, rotation_desired=rotation,
                rotational_stiffness=60.0, rotational_damping=12.0,
            ))
        reached, achieved, _, _ = controller.eef_state()
        assert np.linalg.norm(reached - position) < 0.03

        from tpgpt.utils.rotations import rotation_geodesic

        angle = float(np.degrees(rotation_geodesic(achieved[None], rotation[None])[0]))
        assert angle < 5.0, f"orientation off by {angle:.1f} degrees"
    finally:
        env.close()
