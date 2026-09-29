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
