"""The executor sends CRISP the posture the path check verified (2026-10-02)."""

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _node(tmp_path, dry):
    pytest.importorskip('rclpy')
    from rclpy.parameter import Parameter

    from ycb_pointcloud_pipeline.tpgpt_executor_node import TpgptExecutorNode
    (tmp_path / 'plan.json').write_text(json.dumps({'metadata': {}}))
    return TpgptExecutorNode(parameter_overrides=[
        Parameter('plan_dir', value=str(tmp_path)), Parameter('dry_run', value=dry),
        Parameter('use_transported_stiffness', value=True)])


def _posture():
    from tpgpt.hardware.executor import PostureTarget
    from tpgpt.hardware.fr3 import HOME_Q, default_model
    fr3 = default_model()
    q = np.tile(HOME_Q, (3, 1))
    q[:, 0] = [0.0, 0.2, 0.4]
    return PostureTarget([0, 4, 8], q, fr3.lower, fr3.upper), HOME_Q


def _joint_state(q):
    from sensor_msgs.msg import JointState
    names = [f'fr3_joint{k}' for k in range(1, 8)]
    return JointState(name=names[::-1], position=[float(v) for v in q[::-1]])


def test_every_pose_sent_carries_the_checked_posture_and_a_hold_releases_it(tmp_path):
    import rclpy
    rclpy.init()
    try:
        node = _node(tmp_path, dry=False)
        try:
            # the variable-stiffness controller's own topic, as its target_pose
            assert node.joint_topic == '/tpgpt_cartesian_impedance_controller/target_joint'
            posture, home = _posture()
            node.posture, node.posture_label = posture, 6.0
            measured = home.copy()
            measured[0] = 0.1
            node._joints = (time.monotonic(), _joint_state(measured))
            node.publish(np.array([0.4, 0.0, 0.3]), np.eye(3))
            assert node.posture_sent[0] == pytest.approx(0.3)      # label 6: between 0.2 and 0.4
            assert np.allclose(node.posture_sent[1:], home[1:])
            # an abort's hold: the target goes to where the arm is, and nothing pulls after
            from geometry_msgs.msg import PoseStamped
            pose = PoseStamped()
            pose.header.frame_id = 'base'
            pose.pose.orientation.w = 1.0
            node._pose = (time.monotonic(), pose)
            node.hold_current_pose()
            assert node.posture is None and np.allclose(node.posture_sent, measured)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_a_dry_run_publishes_it_where_no_controller_listens(tmp_path):
    import rclpy
    rclpy.init()
    try:
        node = _node(tmp_path, dry=True)
        try:
            assert node.posture_pub.topic_name == '/tpgpt/executor/commanded_posture'
            posture, home = _posture()
            node.posture, node.posture_label = posture, None
            node.posture_start_q, node.posture_fraction = home + 0.2, 0.5   # mid-approach
            node.publish(np.array([0.4, 0.0, 0.3]), np.eye(3))
            assert np.allclose(node.posture_sent, 0.5 * (home + 0.2 + posture.planned(0)))
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
