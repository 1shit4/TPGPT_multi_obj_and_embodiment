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

from tpgpt.grasp.grasps import (
    approach_waypoint,
    contact_offset,
    grasp_to_eef_pose,
)
from tpgpt.grasp.grippers import (
    MEASURED_PAIRS,
    VERIFIED_PAIRS,
    gripper_config_path,
    gripper_frame,
    resolve_pair,
)
from tpgpt.grasp.verify import (
    LIFT_THRESHOLD,
    lift_height,
    make_env,
    synthetic_top_down,
)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(
        not gripper_config_path("franka_panda").is_file(),
        reason="GraspGen-X gripper descriptions not installed",
    ),
]


@pytest.mark.parametrize("gripper", VERIFIED_PAIRS)
def test_a_converted_grasp_actually_picks_the_object_up(gripper):
    """The acceptance test for the frame contract, across every hand.

    Eight hands spanning three kinematic families, jaw openings from 50 to
    125 mm and fingertip depths from 103 to 195 mm -- including a three-finger
    Robotiq, a Yumi whose site sits 28 mm from its config depth, and a UMI whose
    fingers point backwards from its own ``grip_site``. If the conversion
    dropped the contact offset or rotated about the wrong axis, the fingers
    would close beside or above the object.
    """
    lifted = lift_height(gripper)
    assert lifted > LIFT_THRESHOLD, f"{gripper} lifted only {lifted * 100:.1f} cm"


def test_the_hand_that_converts_but_cannot_grasp_is_excluded():
    """The Inspire hand is measured, convertible, and lifts nothing.

    Five fingers driven by a single open/close command do not pinch a can from
    above. It lifted 0.0 cm across 24 combinations of object, grasp yaw and
    contact offset, and its depth calibration found no working band at all.
    Keeping it out of ``VERIFIED_PAIRS`` is what stops that from surfacing later
    as a transport failure.
    """
    assert "inspire" in MEASURED_PAIRS
    assert "inspire" not in VERIFIED_PAIRS
    assert gripper_frame("inspire")["calibrated_depth"] is None
    # It still converts: the frame is known, the hand simply cannot execute.
    position, rotation = grasp_to_eef_pose(
        synthetic_top_down(np.array([0.0, 0.0, 0.9]), 0.0, "inspire_hand"), "inspire"
    )
    assert np.all(np.isfinite(position)) and np.isclose(np.linalg.det(rotation), 1.0)


@pytest.mark.parametrize("gripper", VERIFIED_PAIRS)
def test_the_contact_point_lands_where_the_grasp_asked(gripper):
    """Geometry, no physics: the one equation the contract has to satisfy.

    Wherever ``grip_site`` ends up, the point where this hand holds an object
    must coincide with the grasp's own contact point.
    """
    pair = resolve_pair(gripper)
    contact = np.array([0.1, -0.2, 0.9])
    grasp = synthetic_top_down(contact, 0.3, pair.graspgen)
    position, rotation = grasp_to_eef_pose(grasp, gripper)
    held = position + rotation @ contact_offset(pair)
    # The calibrated depth deliberately shifts the hand along the approach, so
    # the offset is exact in the plane and calibrated along the axis.
    lateral = np.linalg.norm((held - contact)[:2])
    assert lateral < 1e-6, f"{gripper} holds the object {lateral * 1000:.1f} mm off axis"


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
            env.step(
                controller.action(
                    position, K, D, gripper=-1.0, rotation_desired=rotation,
                    rotational_stiffness=60.0, rotational_damping=12.0,
                )
            )
        reached = controller.eef_state()[0]
        assert np.linalg.norm(reached - position) < 0.02
    finally:
        env.close()


def test_a_standoff_backs_off_along_the_approach_axis():
    grasp = synthetic_top_down(np.array([0.0, 0.0, 0.9]), 0.0, "franka_panda")
    position, _ = grasp_to_eef_pose(grasp, "panda")
    pre_grasp = approach_waypoint(grasp, standoff=0.1, gripper="panda")
    assert np.allclose(pre_grasp, position - grasp.approach * 0.1)


def test_multi_dof_hands_receive_a_command_per_actuator():
    """The UMI takes two gripper values and the Inspire hand six.

    Emitting one silently truncates the action -- robosuite pads the rest with
    zeros, so the fingers half close and the object slides out.
    """
    from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController

    for gripper, expected in (("panda", 1), ("umi", 2), ("inspire", 6)):
        env = make_env(gripper)
        try:
            env.reset()
            controller = CartesianImpedanceController(env)
            controller.reset()
            assert controller.gripper_dof == expected, gripper
            action = controller.action(
                controller.eef_state()[0], np.eye(3) * 500, np.eye(3) * 40, gripper=1.0
            )
            assert len(action) == env.action_dim, gripper
        finally:
            env.close()
