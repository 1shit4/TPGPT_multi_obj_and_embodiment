"""The start-distance guard applies only when the arm has moved since planning (2026-10-03)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _refusal():
    from ycb_pointcloud_pipeline.tpgpt_executor_node import start_distance_refusal
    return start_distance_refusal


HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def test_a_far_start_runs_when_the_arm_is_where_it_was_planned_from():
    """The 10-03 refusal: homed, planned from home, 0.602 m from the start."""
    q = np.array(HOME) + 0.01
    assert _refusal()(0.602, HOME, q, 0.6, 0.05) is None


def test_a_far_start_is_refused_when_the_arm_has_moved_since():
    q = np.array(HOME)
    q[3] += 0.2
    why = _refusal()(0.7, HOME, q, 0.6, 0.05)
    assert why and 'moved since the plan was made' in why and 're-plan' in why


def test_without_joints_the_distance_still_guards():
    assert _refusal()(0.7, None, HOME, 0.6, 0.05)
    assert _refusal()(0.7, HOME, None, 0.6, 0.05)
    assert _refusal()(0.5, None, None, 0.6, 0.05) is None
