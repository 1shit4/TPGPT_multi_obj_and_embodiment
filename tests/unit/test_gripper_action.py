"""The gripper command must fill the whole gripper block, not its last entry.

``tpgpt.grasp.grippers.gripper_action`` exists because two measurement sites
wrote ``action[-1] = 1.0`` and so drove **one** actuator. robosuite lays an
action vector out as the arm's degrees of freedom followed by the gripper's, so
for a hand with ``dof > 1`` that commands one finger and leaves the rest at
zero. The Inspire hand was consequently recorded as travelling 0.7 mm and
written up as a hand that does not actuate; driven properly it travels 58.5 mm.

These are unit tests against a double, so they pin the **shape** of the command.
The double cannot test the claim that this changes physics -- a double built
from the same assumption as the code cannot test that assumption, which is the
lesson of ROBOTICS_NOTES.md section 7.28. That half is
``tests/integration/test_gripper_actuation.py``, which mounts the real hands.
"""

import numpy as np
import pytest

from tpgpt.grasp.grippers import GRIPPER_PAIRS, gripper_action


class _Env:
    def __init__(self, action_dim):
        self.action_dim = action_dim


class _Gripper:
    def __init__(self, dof):
        self.dof = dof


@pytest.mark.parametrize("arm_dof, dof", [(7, 1), (7, 2), (7, 6), (7, 10), (7, 20)])
def test_the_whole_gripper_block_carries_the_command(arm_dof, dof):
    env, grip = _Env(arm_dof + dof), _Gripper(dof)
    action = gripper_action(env, grip, 1.0)

    assert action.shape == (arm_dof + dof,)
    assert np.all(action[-dof:] == 1.0), "the gripper block must be fully driven"
    assert np.all(action[:arm_dof] == 0.0), "the arm must not be commanded"
    assert int(np.count_nonzero(action)) == dof


def test_a_single_dof_hand_is_unaffected():
    """The seven ``dof == 1`` hands must be bitwise unchanged by the fix.

    For them the last entry *is* the whole block, which is what makes the
    repair checkable against the registry as it already stands.
    """
    env, grip = _Env(8), _Gripper(1)
    old = np.zeros(8)
    old[-1] = 1.0
    assert np.array_equal(gripper_action(env, grip, 1.0), old)


def test_the_scalar_form_under_drives_a_multi_dof_hand():
    """Pins *why* this helper exists rather than only what it does."""
    env, grip = _Env(13), _Gripper(6)
    scalar = np.zeros(13)
    scalar[-1] = 1.0

    assert int(np.count_nonzero(scalar)) == 1
    assert int(np.count_nonzero(gripper_action(env, grip, 1.0))) == 6
    assert not np.array_equal(gripper_action(env, grip, 1.0), scalar)


def test_opening_fills_the_block_too():
    env, grip = _Env(13), _Gripper(6)
    assert np.all(gripper_action(env, grip, -1.0)[-6:] == -1.0)


def test_a_gripper_without_a_dof_attribute_is_treated_as_one():
    """A ``None`` or absent ``dof`` must not silently command the whole vector."""

    class _Bare:
        pass

    assert int(np.count_nonzero(gripper_action(_Env(8), _Bare(), 1.0))) == 1

    class _NoneDof:
        dof = None

    assert int(np.count_nonzero(gripper_action(_Env(8), _NoneDof(), 1.0))) == 1


def test_every_registered_hand_has_a_dof_the_helper_can_size():
    """A registry entry whose class cannot report ``dof`` would be driven wrong."""
    from robosuite.models.grippers import GRIPPER_MAPPING

    for short, pair in GRIPPER_PAIRS.items():
        cls = GRIPPER_MAPPING.get(pair.robosuite)
        assert cls is not None, f"{short}: {pair.robosuite} is not a robosuite gripper"
        dof = int(cls().dof)
        assert dof >= 1, f"{short}: dof {dof}"
