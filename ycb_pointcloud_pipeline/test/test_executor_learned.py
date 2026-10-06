"""A learned plan runs at the settings its labels were learned under (audit B1, step 6)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LEARNED = {'speed_scale': 0.35, 'rotational_stiffness': 30.0, 'transit_rotational_scale': 1.0,
           'use_transported_stiffness': True, 'offset_compensation': 'none',
           'arrival_hold': False}


def _plan(tmp_path, learned=LEARNED):
    (tmp_path / 'plan.json').write_text(json.dumps({'metadata': {'learned_execution': learned}}))
    return str(tmp_path)


def _module():
    pytest.importorskip('rclpy')
    from ycb_pointcloud_pipeline import tpgpt_executor_node
    return tpgpt_executor_node


def _defaults():
    return {'speed_scale': 0.35, 'rotational_stiffness': 30.0, 'transit_rotational_scale': -1.0,
            'use_transported_stiffness': False, 'offset_compensation': 'z',
            'arrival_hold': False}


def test_defaults_take_the_learned_values(tmp_path):
    ex = _module()
    got = ex.learned_settings(_plan(tmp_path), _defaults(), _defaults())
    # arrival_hold is not one: a learned plan always holds, at its marks (follow_arm_path)
    assert got == {k: v for k, v in LEARNED.items() if k != 'arrival_hold'}
    assert type(got['use_transported_stiffness']) is bool


def test_a_value_equal_to_the_learned_one_is_accepted(tmp_path):
    ex = _module()
    given = {**_defaults(), 'transit_rotational_scale': 1.0, 'offset_compensation': 'none'}
    assert ex.learned_settings(_plan(tmp_path), given, _defaults()) == {
        k: v for k, v in LEARNED.items() if k != 'arrival_hold'}


def test_any_other_value_is_refused(tmp_path):
    ex = _module()
    given = {**_defaults(), 'speed_scale': 0.5, 'transit_rotational_scale': 0.25}
    with pytest.raises(SystemExit, match='speed_scale.*transit_rotational_scale'):
        ex.learned_settings(_plan(tmp_path), given, _defaults())


def test_a_plan_that_is_not_learned_changes_nothing(tmp_path):
    ex = _module()
    assert ex.learned_settings(_plan(tmp_path, None), _defaults(), _defaults()) == {}
    assert ex.learned_settings(str(tmp_path / 'missing'), _defaults(), _defaults()) == {}


def test_the_node_runs_a_learned_plan_with_its_stiffness_labels(tmp_path):
    ex = _module()
    import rclpy
    from rclpy.parameter import Parameter
    rclpy.init()
    try:
        node = ex.TpgptExecutorNode(parameter_overrides=[
            Parameter('plan_dir', value=_plan(tmp_path))])
        try:
            assert node.p('offset_compensation') == 'none'
            assert node.p('transit_rotational_scale') == 1.0
            # decided in __init__, before the controller is chosen
            assert node.use_stiffness and node.variable_stiffness
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_a_learned_plan_keeps_the_tracking_guard_but_not_the_not_following_one():
    """2026-10-03: at the start of the retreat the learned command moved 30 mm and the hand
    9.4 mm, as in the replay; the "not following" guard (30 %) stopped a run on its path."""
    import numpy as np
    ex = _module()
    p = {'max_tracking_error': 0.08, 'progress_window': 40, 'min_progress_ratio': 0.3}.get
    arm, command = np.zeros(3), np.zeros(3)
    learned, taught = ex.safety_monitor(p, True), ex.safety_monitor(p, False)
    reasons = {'learned': None, 'taught': None}
    for k in range(60):                       # the command moves 1.5 mm a step, the arm 0.45
        command = command + [0.0015, 0, 0]
        arm = arm + [0.00045, 0, 0]
        for name, mon in (('learned', learned), ('taught', taught)):
            reasons[name] = reasons[name] or mon.check(arm, command)
    assert reasons['learned'] is None and 'not following' in reasons['taught']
    assert 'exceeds' in ex.safety_monitor(p, True).check(np.zeros(3), np.array([0.09, 0, 0]))


def test_a_grasp_test_runs_at_the_transport_settings_and_says_so(tmp_path):
    """A grasp test carries its source bundle's settings (as the Franka Hand's transport runs
    were executed); the refusal names the grasp test, not a learned attractor."""
    ex = _module()
    (tmp_path / 'plan.json').write_text(json.dumps(
        {'metadata': {'learned_execution': LEARNED, 'task': 'grasp_test'}}))
    assert ex.learned_settings(str(tmp_path), _defaults(), _defaults()) == {
        k: v for k, v in LEARNED.items() if k != 'arrival_hold'}
    with pytest.raises(SystemExit, match="grasp test runs at the transport's"):
        ex.learned_settings(str(tmp_path), {**_defaults(), 'speed_scale': 0.5}, _defaults())
