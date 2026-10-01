"""A policy run on the robot keeps every pose and joint state it received (2026-10-02)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_the_full_rate_record_is_saved_in_run_time(tmp_path):
    pytest.importorskip('rclpy')
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState

    from ycb_pointcloud_pipeline.tpgpt_executor_node import TpgptExecutorNode

    (tmp_path / 'plan.json').write_text(json.dumps({'metadata': {}}))
    rclpy.init()
    try:
        node = TpgptExecutorNode(parameter_overrides=[Parameter('plan_dir', value=str(tmp_path))])
        try:
            origin = 1000.0
            path = str(tmp_path / 'robot_x_highrate.npz')
            node._run_record = (path, origin)
            node._highrate = {'poses': [], 'joints': []}
            for k in range(5):
                pose = PoseStamped()
                pose.header.stamp.sec, pose.header.stamp.nanosec = 1000, k * 4_000_000
                pose.pose.position.x, pose.pose.orientation.w = 0.1 * k, 1.0
                node._pose_cb(pose)
                joints = JointState()
                joints.header.stamp.sec, joints.header.stamp.nanosec = 1000, k * 10_000_000
                # out of order, as a JointState may come: the record is in fr3_joint1..7 order
                joints.name = [f'fr3_joint{i}' for i in (7, 1, 2, 3, 4, 5, 6)]
                joints.position = [7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                joints.velocity = [0.0] * 7
                joints.effort = [0.5] * 7
                node._joint_cb(joints)
            node.save_run_record()
            assert node._run_record is None and node._highrate is None
            rec = np.load(path)
            assert np.allclose(rec['pose_t'], [0.0, 0.004, 0.008, 0.012, 0.016])   # since the start
            assert np.allclose(rec['pose_position'][:, 0], [0.0, 0.1, 0.2, 0.3, 0.4])
            assert np.allclose(rec['joint_position'][0], [1, 2, 3, 4, 5, 6, 7])
            assert rec['joint_effort'].shape == (5, 7) and 'pose_stamp' not in rec
            node.save_run_record()                     # a second call (abort after end) is a no-op
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
