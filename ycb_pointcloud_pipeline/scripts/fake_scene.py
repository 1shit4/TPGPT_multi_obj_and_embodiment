#!/usr/bin/env python3
"""Replay a recorded capture as the camera and the arm, for offline ROS dry runs.

The planner node needs an object cloud (/ycb_pointclouds/<object>), the scene
(/ycb_scene_cloud) -- both in the camera's optical frame, as segmentation_node
publishes them -- and /joint_states. A plan directory the planner wrote keeps its
capture (clouds.npz: the camera's object cloud and the obstacles, base frame) and the
joints it planned from (plan.json joint_states_at_plan). This publishes them again,
converted back into the optical frame through the INSTALLED calibration's inverse, so
the planner's own conversion puts them where they were.

**Never on the robot's domain**: it refuses to run with ROS_DOMAIN_ID 61 (the robot),
since it publishes /joint_states. Use another domain, without host_ros_env.sh:

    source /opt/ros/jazzy/setup.bash && source ~/thor_ws/install/setup.bash
    export ROS_DOMAIN_ID=88
    ~/venvs/tpgpt/bin/python ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/scripts/fake_scene.py \\
        ~/tpgpt_plans/apple_20261003_221827 --object apple

then, in another terminal on the same domain, the planner (-p allow_missing_joint_states
not needed) and the executor in dry run.
"""

import argparse
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", help="a plan directory holding clouds.npz (and plan.json)")
    ap.add_argument("--object", default=None, help="object name (default: from the directory)")
    ap.add_argument("--joints", nargs=7, type=float, default=None,
                    help="joint angles to publish (default: the capture's, else home)")
    ap.add_argument("--rate", type=float, default=5.0, help="cloud rate, Hz")
    a = ap.parse_args(argv)
    if os.environ.get("ROS_DOMAIN_ID", "0") == "61":
        sys.exit("ROS_DOMAIN_ID is 61, the robot's: fake_scene publishes /joint_states. "
                 "Use another domain (export ROS_DOMAIN_ID=88).")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import PointCloud2  # noqa: F401  (message type of the publishers)

    from ycb_pointcloud_pipeline import camera_calibration as cc
    from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
    from ycb_pointcloud_pipeline.ros_utils import xyz_to_pointcloud2

    capture = os.path.expanduser(a.capture.rstrip("/"))
    name = a.object or os.path.basename(capture).rsplit("_2026", 1)[0]
    clouds = np.load(os.path.join(capture, "clouds.npz"))
    q = a.joints
    if q is None and os.path.isfile(os.path.join(capture, "plan.json")):
        q = json.load(open(os.path.join(capture, "plan.json")))["metadata"].get(
            "joint_states_at_plan")
    if q is None:
        from tpgpt.hardware.fr3 import HOME_Q
        q = HOME_Q.tolist()
    cal = cc.CameraCalibration.load(default_calibration_yaml())
    T_opt_base = cc.invert_transform(cal.T_base_camera)
    frame = cal.camera_child_frame

    def optical(points):
        return cc.transform_points(T_opt_base, np.asarray(points, dtype=float))

    obj, scene = optical(clouds["observed"]), optical(clouds["obstacles"])

    rclpy.init()
    node = Node("fake_scene")
    obj_pub = node.create_publisher(PointCloud2, f"/ycb_pointclouds/{name}", 5)
    scene_pub = node.create_publisher(PointCloud2, "/ycb_scene_cloud", 2)
    joint_pub = node.create_publisher(JointState, "/joint_states", 10)

    def clouds_cb():
        stamp = node.get_clock().now().to_msg()
        obj_pub.publish(xyz_to_pointcloud2(obj, frame, stamp))
        scene_pub.publish(xyz_to_pointcloud2(scene, frame, stamp))

    def joints_cb():
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = [f"fr3_joint{k}" for k in range(1, 8)]
        msg.position = [float(v) for v in q]
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        joint_pub.publish(msg)

    node.create_timer(1.0 / a.rate, clouds_cb)
    node.create_timer(0.02, joints_cb)
    node.get_logger().info(f"{name}: {len(obj)} object points, {len(scene)} scene points in "
                           f"{frame}; joints {np.round(q, 3).tolist()} (domain "
                           f"{os.environ.get('ROS_DOMAIN_ID', '0')})")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
