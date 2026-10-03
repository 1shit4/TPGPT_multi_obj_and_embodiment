"""The grip force is per object (the user, 2026-10-03), from config/grasp_forces.yaml."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
TABLE = os.path.join(os.path.dirname(__file__), '..', 'config', 'grasp_forces.yaml')


def _grip_force():
    from ycb_pointcloud_pipeline.tpgpt_executor_node import grip_force
    return grip_force


@pytest.mark.parametrize('name,force', [('pudding_box', 7.5), ('apple', 80.0), ('banana', 40.0),
                                        ('foam_brick', 20.0), ('mug', 50.0)])
def test_each_object_gets_its_own_force(name, force):
    assert _grip_force()(0.0, name, TABLE)[0] == force


def test_an_unlisted_object_gets_the_default_and_a_given_force_wins():
    assert _grip_force()(0.0, 'tomato_soup_can', TABLE)[0] == 30.0
    assert _grip_force()(0.0, None, TABLE)[0] == 30.0
    assert _grip_force()(15.0, 'foam_brick', TABLE) == (15.0, 'grasp_force parameter')
