"""The planner stops and asks for a re-home when the chosen grasp's move onto the path is blocked."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _blocked(approach_in):
    pytest.importorskip('rclpy')
    from ycb_pointcloud_pipeline.tpgpt_planner_node import TpgptPlannerNode
    chosen = SimpleNamespace(path_check={'approach_in': approach_in})
    return TpgptPlannerNode._approach_blocked(SimpleNamespace(chosen=chosen))


def test_a_clear_move_onto_the_path_is_not_blocked():
    assert _blocked({'steps': 12, 'unreachable': 0, 'faults': {}, 'rejects': False}) is None


def test_a_move_through_seen_points_asks_for_a_re_home():
    why = _blocked({'steps': 12, 'unreachable': 0, 'rejects': False,
                    'faults': {'approach_in: fr3_link4 in the seen points': 3}})
    assert 'fr3_link4 in the seen points' in why


def test_unreachable_steps_are_blocked_too():
    why = _blocked({'steps': 12, 'unreachable': 2, 'faults': {}, 'rejects': False})
    assert '2 of its 12 steps out of reach' in why


def test_when_the_move_rejected_grasps_itself_there_is_nothing_left_to_ask():
    # approach_rejects_grasp:=true -- the old rule: a chosen grasp's move is clear by construction
    assert _blocked({'steps': 12, 'unreachable': 0, 'rejects': True,
                     'faults': {'approach_in: x': 3}}) is None


def _at_home(q, tolerance=0.1):
    pytest.importorskip('rclpy')
    from ycb_pointcloud_pipeline.tpgpt_planner_node import TpgptPlannerNode
    logged = []
    node = SimpleNamespace(
        joints=None if q is None else (0.0, q),
        get_parameter=lambda name: SimpleNamespace(value=tolerance),
        get_logger=lambda: SimpleNamespace(info=logged.append))
    return TpgptPlannerNode._arm_at_home(node), logged


def test_a_homed_arm_rejects_a_grasp_whose_move_is_blocked():
    """The user, 2026-10-07: re-homing a homed arm achieves nothing, so the grasp is passed
    over. The joints after joint_homing on 10-07 (0.07 rad short on joint 7) count as home."""
    from ycb_pointcloud_pipeline.joint_homing import FRANKA_HOME
    homed = [0.019, -0.775, 0.017, -2.315, -0.048, 1.616, 0.856]
    assert max(abs(a - b) for a, b in zip(homed, FRANKA_HOME)) < 0.1
    at_home, logged = _at_home(homed)
    assert at_home and logged
    parked = list(FRANKA_HOME)
    parked[0] += 0.4                                   # turned away: re-homing can help
    assert not _at_home(parked)[0]
    assert not _at_home(None)[0]                        # no joints: the old rule
