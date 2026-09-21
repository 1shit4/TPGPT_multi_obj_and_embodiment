#!/usr/bin/env python3
"""
Publish the robot base -> camera transform from config/camera_calibration.yaml
on /tf_static, so RViz shows the camera's clouds next to the robot model and any
TF-based tool agrees with the calibration the planner uses.

  base -> camera_link                    (always, from base_to_camera)
  camera_link -> camera_color_optical_frame
                                         (only when camera_link_to_optical.source
                                          is 'yaml'; otherwise realsense2_camera
                                          publishes it with factory values)

Run:
    ros2 run ycb_pointcloud_pipeline camera_tf_publisher --ros-args \
        -p calibration_yaml:=/path/to/camera_calibration.yaml
"""

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.ros_utils import matrix_to_transform_stamped


def default_calibration_yaml():
    try:
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'),
                            'config', 'camera_calibration.yaml')
    except Exception:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_calibration.yaml')


class CameraTfPublisher(Node):
    def __init__(self):
        super().__init__('camera_tf_publisher')
        self.declare_parameter('calibration_yaml', default_calibration_yaml())
        path = self.get_parameter('calibration_yaml').value
        cal = cc.CameraCalibration.load(path)
        stamp = self.get_clock().now().to_msg()
        transforms = [matrix_to_transform_stamped(cal.T_base_camera, cal.base_frame,
                                                  cal.camera_child_frame, stamp)]
        if not cal.child_is_optical and cal.link_to_optical_source == 'yaml':
            transforms.append(matrix_to_transform_stamped(
                cal.T_link_optical_yaml, cal.camera_child_frame, cal.optical_frame, stamp))
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transforms)
        t = cal.T_base_camera[:3, 3]
        q = cc.matrix_to_quat_xyzw(cal.T_base_camera[:3, :3])
        self.get_logger().info(
            f'Publishing {cal.base_frame} -> {cal.camera_child_frame}: '
            f't={t.round(4).tolist()} q_xyzw={q.round(5).tolist()} from {path}'
            + (' [PLACEHOLDER]' if cal.camera_is_placeholder else ''))
        if len(transforms) > 1:
            self.get_logger().info(f'Also publishing {cal.camera_child_frame} -> '
                                   f'{cal.optical_frame} from the YAML (source: yaml)')


def main(args=None):
    rclpy.init(args=args)
    node = CameraTfPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
