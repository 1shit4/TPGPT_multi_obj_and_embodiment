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
    assert got == LEARNED
    assert type(got['use_transported_stiffness']) is bool


def test_a_value_equal_to_the_learned_one_is_accepted(tmp_path):
    ex = _module()
    given = {**_defaults(), 'transit_rotational_scale': 1.0, 'offset_compensation': 'none'}
    assert ex.learned_settings(_plan(tmp_path), given, _defaults()) == LEARNED


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
