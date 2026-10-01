from types import SimpleNamespace as NS

import numpy as np
import pytest

from tpgpt.sim.joint_policy import rollout_joint_policy


@pytest.fixture
def joint_case(monkeypatch):
    import tpgpt.sim.joint_policy as m
    controller = NS(qpos_index=[0])
    env = NS(robots=[NS(composite_controller=NS(part_controllers={"right": controller}))],
             sim=NS(data=NS(qpos=np.zeros(1))), control_freq=20,
             step=lambda action: None)
    monkeypatch.setattr(m, "_site_position", lambda *a: np.zeros(3))
    monkeypatch.setattr(m, "_site_rotation", lambda *a: np.eye(3))
    monkeypatch.setattr(m, "_joint_action", lambda *a: np.zeros(1))
    monkeypatch.setattr(m, "solve_ik", lambda *a, **kw: NS(qpos=np.zeros(1), reachable=False))
    calls = []
    def predict(x, phase):
        calls.append((x.copy(), phase.copy()))
        return NS(reference=np.array([[.008, 0., 0.]]), orientation=np.eye(3)[None],
                  gripper=np.array([-1.]), time_rate=np.ones(1),
                  velocity=np.zeros((1, 3)), velocity_std=np.zeros((1, 3)))
    policy = NS(labels=NS(positions=np.zeros((2, 3)), time_belief=np.array([0., .025]),
                         orientations=np.tile(np.eye(3), (2, 1, 1))), predict=predict)
    return env, policy, calls


def test_measured_tracking_advances_despite_tighter_ik_flag(joint_case):
    env, policy, calls = joint_case
    result = rollout_joint_policy(env, policy, approach_steps=0, settle_steps=0,
                                  score=lambda e: (True, np.zeros(3)), max_steps=4)
    assert result.success
    assert result.metadata["termination"] == "phase_complete"
    assert result.metadata["unreachable_steps"] == 2
    assert calls[1][1][0] > calls[0][1][0]
    np.testing.assert_array_equal(calls[0][0], [[0., 0., 0.]])


def test_ik_stall_cannot_be_scored_as_completed_placement(joint_case):
    env, policy, _ = joint_case
    result = rollout_joint_policy(env, policy, approach_steps=0, settle_steps=0,
                                  score=lambda e: (True, np.zeros(3)), max_steps=4,
                                  require_ik_reachable=True, stall_steps=2)
    assert not result.success
    assert result.metadata["termination"] == "tracking_stalled"
