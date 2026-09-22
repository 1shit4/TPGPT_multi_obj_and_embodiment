"""Ground the shelf model's placement in the camera's cloud (tpgpt.hardware.shelf.fit_to_cloud).

The shelf's dimensions come from a tape and its heights from touching; its x/y placement and yaw
rest on the four tape readings from the slot marks (M1-M4), the hardest to hold square. This node
collects a few /ycb_scene_cloud frames, removes the robot (at the /joint_states of each frame),
fits the correction that puts the model's faces on the observed points, prints it with the
per-face offsets, and writes it to ~/tpgpt_calibration/shelf_fit_<time>.yaml. It never edits
shelf.yaml: copy the printed `cloud_fit:` block under `shelf:` there, rebuild, restart the planner.

Before running: the shelf EMPTY, the arm out of the camera's view (it is removed from the cloud
too, but a hand resting against a panel would be read as the panel), segmentation_node running.

    source ~/venvs/tpgpt/bin/activate      # it imports tpgpt
    ros2 run --prefix python3 ycb_pointcloud_pipeline shelf_fit
    # with the robot off (the robot then cannot be removed -- keep it out of view):
    ros2 run --prefix python3 ycb_pointcloud_pipeline shelf_fit --ros-args -p allow_missing_joint_states:=true

In RViz (fixed frame = the calibration's base frame): /shelf_fit/model_tape (orange, as the tape
places it), /shelf_fit/model_fitted (green), /shelf_fit/cloud (the points used).
"""

import os
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, PointCloud2
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
from ycb_pointcloud_pipeline.ros_utils import (
    CalibrationProvider, box_marker, pointcloud2_to_xyzrgb, xyz_to_pointcloud2,
)
from ycb_pointcloud_pipeline.tpgpt_planner_node import default_config_path


class ShelfFitNode(Node):
    def __init__(self):
        super().__init__('shelf_fit')
        d = self.declare_parameter
        d('calibration_yaml', default_calibration_yaml())
        d('shelf_config', default_config_path('shelf.yaml'))
        d('scene_cloud_topic', '/ycb_scene_cloud')
        d('n_frames', 8)                       # clouds merged (depth noise averages down)
        d('voxel_size', 0.004)                 # m
        d('joint_states_topic', '/joint_states')
        d('joint_states_max_age', 1.0)         # s
        d('allow_missing_joint_states', False)
        d('out_dir', os.path.expanduser('~/tpgpt_calibration'))
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        with open(p('shelf_config')) as f:
            self.shelf_raw = yaml.safe_load(f) or {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.calib = CalibrationProvider(self, p('calibration_yaml'), self.tf_buffer)
        self.base = self.calib.cal.base_frame
        self.joints = None
        self.frames = []
        self.done = False

        from tpgpt.hardware.end_effector import FRANKA_HAND
        from tpgpt.hardware.fr3 import default_model
        self.fr3, self.hand = default_model(), FRANKA_HAND

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.tape_pub = self.create_publisher(MarkerArray, '/shelf_fit/model_tape', latched)
        self.fit_pub = self.create_publisher(MarkerArray, '/shelf_fit/model_fitted', latched)
        self.cloud_pub = self.create_publisher(PointCloud2, '/shelf_fit/cloud', latched)
        self.create_subscription(JointState, p('joint_states_topic'), self._joint_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, p('scene_cloud_topic'), self._cloud_cb, 2)
        self.get_logger().info(f"Shelf {p('shelf_config')}; collecting {p('n_frames')} frames of "
                               f"{p('scene_cloud_topic')}. Shelf empty, arm out of view.")

    def _joint_cb(self, msg):
        index = {n: i for i, n in enumerate(msg.name)}
        names = [f'fr3_joint{k}' for k in range(1, 8)]
        if all(n in index for n in names):
            self.joints = (time.monotonic(),
                           np.array([msg.position[index[n]] for n in names], dtype=float))

    def _cloud_cb(self, msg):
        if self.done:
            return
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        T = self.calib.T_base_optical(msg.header.frame_id or self.calib.cal.optical_frame)
        if T is None:
            return
        fresh = (self.joints is not None
                 and time.monotonic() - self.joints[0] <= float(p('joint_states_max_age')))
        if not fresh and not p('allow_missing_joint_states'):
            self.get_logger().error(
                f"no fresh {p('joint_states_topic')}: the robot cannot be removed from the cloud. "
                'Robot container up and host_ros_env.sh sourced? With the robot off and out of '
                'view: -p allow_missing_joint_states:=true', throttle_duration_sec=5.0)
            return
        xyz, _ = pointcloud2_to_xyzrgb(msg)
        pts = cc.transform_points(T, xyz)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        removed = 0
        if fresh:
            from tpgpt.hardware.shelf import robot_mask
            mine = robot_mask(pts, self.joints[1], self.fr3, self.hand)
            removed = int(mine.sum())
            pts = pts[~mine]
        self.frames.append(pts)
        self.get_logger().info(f'frame {len(self.frames)}/{p("n_frames")}: {len(pts)} points'
                               + (f', {removed} removed as the robot' if fresh else
                                  ', robot NOT removed'))
        if len(self.frames) >= int(p('n_frames')):
            self.done = True
            self._fit()

    def _fit(self):
        from tpgpt.hardware.shelf import ShelfModel, describe_fit, fit_to_cloud
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        pts = cc.voxel_downsample(np.vstack(self.frames), float(p('voxel_size')))
        shelf = dict(self.shelf_raw.get('shelf', self.shelf_raw))
        report = fit_to_cloud({'shelf': shelf}, pts)
        log = self.get_logger()
        for line in describe_fit(report).splitlines():
            log.info(line)
        if report['refused']:
            log.error('The fit was refused; shelf.yaml should not be changed from this run.')

        stamp = self.get_clock().now().to_msg()
        tape = {k: v for k, v in shelf.items() if k != 'cloud_fit'}
        for pub, cfg, color in (
                (self.tape_pub, tape, (1.0, 0.55, 0.1, 0.35)),
                (self.fit_pub, {**tape, 'cloud_fit': report['fit']}, (0.1, 0.9, 0.3, 0.35))):
            model = ShelfModel.from_config({'shelf': cfg})
            markers = MarkerArray()
            for k, b in enumerate(model.boxes):
                markers.markers.append(box_marker('shelf', k, b.position, b.rotation,
                                                  2.0 * np.asarray(b.size), self.base, stamp,
                                                  color))
            pub.publish(markers)
        self.cloud_pub.publish(xyz_to_pointcloud2(pts, self.base, stamp, (0, 200, 255)))

        os.makedirs(p('out_dir'), exist_ok=True)
        path = os.path.join(p('out_dir'), time.strftime('shelf_fit_%Y%m%d_%H%M%S.yaml'))
        record = {'shelf_config': p('shelf_config'), 'frames': len(self.frames),
                  'points': int(len(pts)), 'report': _plain(report)}
        with open(path, 'w') as f:
            yaml.safe_dump(record, f, sort_keys=False)
        log.info(f'Written {path}. RViz: /shelf_fit/model_tape (orange), /shelf_fit/model_fitted '
                 '(green), /shelf_fit/cloud. Ctrl-C to exit.')
        if not report['refused']:
            f = report['cloud_fit']
            log.info('To use it, put this under `shelf:` in config/shelf.yaml, then rebuild:\n'
                     '  cloud_fit:                         # shelf_fit, '
                     + time.strftime('%Y-%m-%d') + '\n'
                     f"    shift_along_front_mm: {f['shift_along_front_mm']}\n"
                     f"    shift_into_shelf_mm: {f['shift_into_shelf_mm']}\n"
                     f"    yaw_deg: {f['yaw_deg']}")


def _plain(v):
    """numpy scalars and tuples -> YAML-safe Python."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def main(args=None):
    rclpy.init(args=args)
    node = ShelfFitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
