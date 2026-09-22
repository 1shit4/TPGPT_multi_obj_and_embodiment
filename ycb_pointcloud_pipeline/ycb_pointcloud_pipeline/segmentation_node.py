#!/usr/bin/env python3
"""
ROS2 node: subscribes to the RealSense camera node's color + aligned depth +
camera_info topics, and every `process_period_sec` (default 3 s) runs
Grounded-SAM on the latest synchronized frame, restricted to the objects in
config/target_objects.yaml. Each detected object's masked depth pixels are
deprojected and published as a PointCloud2 on /ycb_pointclouds/<name>.

An annotated image with the detections is published on
/ycb_detections/image (view it with rqt_image_view).

The whole depth frame of the same cycle, subsampled every `scene_cloud_stride`
pixels, is published on /ycb_scene_cloud (same frame and stamp as the object
clouds). The planner crops it to the shelf and uses it as the observed obstacle
cloud (tpgpt.hardware.shelf.ShelfWorld).

Run (after building the workspace and sourcing it):
    ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
"""

import json
import os
import threading

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
import message_filters

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.pointcloud_utils import (
    CameraIntrinsics, mask_to_pointcloud, points_to_ros2_pointcloud2, save_pcd_ascii,
)


def _default_target_objects_path():
    try:
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'),
                            'config', 'target_objects.yaml')
    except Exception:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'target_objects.yaml')


def _default_calibration_path():
    try:
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'),
                            'config', 'camera_calibration.yaml')
    except Exception:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_calibration.yaml')


def load_target_objects(yaml_path):
    """Entries may be a plain string ("banana") or {name: ..., prompt: ...}.
    `name` is used for the topic, `prompt` is the text given to Grounding DINO."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    targets = []
    for entry in data.get('target_objects', []):
        if isinstance(entry, dict):
            name = str(entry['name']).strip()
            prompt = str(entry.get('prompt', name.replace('_', ' '))).strip()
        else:
            name = str(entry).strip()
            prompt = name.replace('_', ' ')
        targets.append({'name': name.lower().replace(' ', '_'), 'prompt': prompt})
    if not targets:
        raise ValueError(f"No target_objects found in {yaml_path}")
    return targets


class YCBSegmentationNode(Node):
    def __init__(self):
        super().__init__('ycb_segmentation_node')

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('target_objects_yaml', _default_target_objects_path())
        self.declare_parameter('process_period_sec', 3.0)
        self.declare_parameter('detector_mode', 'grounded_sam')  # 'grounded_sam' | 'dummy'
        self.declare_parameter('grounding_model', 'IDEA-Research/grounding-dino-base')
        self.declare_parameter('sam_model', 'facebook/sam-vit-large')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('depth_scale', 1000.0)  # RealSense: mm -> m
        self.declare_parameter('depth_trunc', 2.0)     # ignore points farther than this (m)
        self.declare_parameter('save_pcd', False)
        self.declare_parameter('output_dir', os.path.expanduser('~/ycb_pointclouds'))
        self.declare_parameter('show_debug_window', False)
        # The whole frame as the planner's observed obstacle cloud (/ycb_scene_cloud).
        # Pixels eroded off each object mask before it is deprojected. The simulation
        # does this (tpgpt.perception.cameras.DEFAULT_MASK_EROSION = 1) and measured why:
        # a mask's boundary pixels take their depth partly from whatever is behind the
        # object, so they fly off towards the camera. On a 5 x 5 x 9 cm box the fused
        # extents were [17.3, 8.8, 9.7] cm unerooded and [5.8, 6.2, 8.9] cm with one
        # pixel. Two pixels can empty the mask of a small or distant object, so the
        # erosion is skipped for any mask it would leave under `mask_min_pixels`.
        self.declare_parameter('mask_erosion', 1)
        self.declare_parameter('mask_min_pixels', 60)
        # Flying pixels: at a silhouette a pixel's depth interpolates between the object
        # and the background behind it, so it lands on the ray between them. Measured
        # 2026-09-23: an upright YCB can (68 mm across) fused to a 116 mm blob with a
        # 60 mm comet tail streaming away from the camera; a flat banana, whose silhouette
        # has a small depth step, showed none. Such pixels sit on a steep depth ramp, so
        # pixels whose neighbouring depths change faster than this are dropped. On a good
        # frame the 99th percentile of the gradient over an object is about 8 mm/px.
        self.declare_parameter('depth_gradient_max_mm_px', 8.0)   # 0 disables
        self.declare_parameter('publish_scene_cloud', True)
        self.declare_parameter('scene_cloud_stride', 4)   # every 4th pixel: ~25k points at 848x480
        # Intrinsics used for deprojection: config/camera_calibration.yaml
        # `intrinsics.source` = camera_info (default) or yaml (the placeholder
        # values in that file). An empty path disables the YAML entirely.
        self.declare_parameter('camera_calibration_yaml', _default_calibration_path())

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.depth_scale = p('depth_scale')
        self.depth_trunc = p('depth_trunc')
        self.save_pcd = p('save_pcd')
        self.output_dir = p('output_dir')
        self.show_debug_window = p('show_debug_window')
        if self.save_pcd:
            os.makedirs(self.output_dir, exist_ok=True)

        yaml_path = p('target_objects_yaml')
        self.targets = load_target_objects(yaml_path)
        self.get_logger().info(
            f"Loaded {len(self.targets)} target(s) from {yaml_path}: "
            f"{[t['name'] for t in self.targets]}")

        mode = p('detector_mode')
        if mode == 'dummy':
            from ycb_pointcloud_pipeline.ycb_detector import DummyDetector
            self.detector = DummyDetector(self.targets)
            self.get_logger().warn("detector_mode='dummy': publishing a fixed center-box cloud")
        elif mode == 'grounded_sam':
            from ycb_pointcloud_pipeline.ycb_detector import GroundedSAMDetector
            self.detector = GroundedSAMDetector(
                self.targets, grounding_model=p('grounding_model'), sam_model=p('sam_model'),
                box_threshold=p('box_threshold'), text_threshold=p('text_threshold'),
                logger=self.get_logger())
        else:
            raise ValueError(f"Unknown detector_mode: {mode!r}")

        # One publisher per target, created up front so topics are listable
        self.cloud_pubs = {
            t['name']: self.create_publisher(PointCloud2, f"/ycb_pointclouds/{t['name']}", 1)
            for t in self.targets
        }
        self.image_pub = self.create_publisher(Image, '/ycb_detections/image', 1)
        # X6 record: the mask each cloud was deprojected from (same stamp as the
        # cloud) and a JSON line per cycle with scores and models. Extra topics only.
        self.mask_pubs = {
            t['name']: self.create_publisher(Image, f"/ycb_masks/{t['name']}", 1)
            for t in self.targets
        }
        self.info_pub = self.create_publisher(String, '/ycb_detections/info', 5)
        self.scene_pub = self.create_publisher(PointCloud2, '/ycb_scene_cloud', 1)

        self.bridge = CvBridge()
        self.intrinsics = None
        self.calibration = None
        cal_path = p('camera_calibration_yaml')
        if cal_path:
            self.calibration = cc.CameraCalibration.load(cal_path)
            if self.calibration.intrinsics_source == 'yaml':
                y = self.calibration.intrinsics_yaml
                self.intrinsics = CameraIntrinsics(fx=y.fx, fy=y.fy, cx=y.cx, cy=y.cy,
                                                   width=y.width, height=y.height)
                self.get_logger().warn(
                    f"Deprojecting with YAML intrinsics from {cal_path}: fx={y.fx:.2f} "
                    f"fy={y.fy:.2f} cx={y.cx:.2f} cy={y.cy:.2f} ({y.width}x{y.height})"
                    + (' [PLACEHOLDER]' if self.calibration.intrinsics_is_placeholder else ''))
        self._info_checked = False
        self._lock = threading.Lock()
        self._latest = None  # (color_msg, depth_msg)

        # Sensor callbacks only stash the newest frame; they run in their own
        # callback group so a slow inference never backs them up.
        sensor_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(CameraInfo, p('camera_info_topic'), self.camera_info_cb,
                                 qos_profile_sensor_data, callback_group=sensor_group)
        color_sub = message_filters.Subscriber(self, Image, p('color_topic'),
                                               qos_profile=qos_profile_sensor_data,
                                               callback_group=sensor_group)
        depth_sub = message_filters.Subscriber(self, Image, p('depth_topic'),
                                               qos_profile=qos_profile_sensor_data,
                                               callback_group=sensor_group)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.rgbd_cb)

        self.create_timer(p('process_period_sec'), self.process,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.cycle = 0
        self.get_logger().info(
            f"Ready. Processing latest frame every {p('process_period_sec')} s from "
            f"{p('color_topic')} + {p('depth_topic')}")

    def camera_info_cb(self, msg):
        if not self._info_checked:
            self._info_checked = True
            info = cc.Intrinsics.from_camera_info_fields(msg.k, msg.width, msg.height,
                                                         msg.distortion_model, msg.d)
            if self.calibration is not None:
                worst, text = self.calibration.intrinsics_yaml.compare(info)
                tol = float((self.calibration.raw.get('verification') or {})
                            .get('max_intrinsics_mismatch_px', 2.0))
                (self.get_logger().info if worst <= tol else self.get_logger().warn)(
                    f"Intrinsics YAML - camera_info: {text}"
                    + ('' if worst <= tol else f' -> MISMATCH above {tol} px'))
            if info.has_distortion():
                self.get_logger().warn(f"camera_info reports distortion {info.distortion}; "
                                       "deprojection assumes rectified images")
        if self.intrinsics is None:
            self.intrinsics = CameraIntrinsics.from_camera_info(msg)
            i = self.intrinsics
            self.get_logger().info(f"Camera intrinsics (camera_info): fx={i.fx:.1f} "
                                   f"fy={i.fy:.1f} cx={i.cx:.1f} cy={i.cy:.1f} "
                                   f"({i.width}x{i.height})")

    def rgbd_cb(self, color_msg, depth_msg):
        with self._lock:
            self._latest = (color_msg, depth_msg)

    def _depth_gradient(self, depth):
        """Per-pixel depth change, mm per pixel, ignoring invalid depth."""
        d = np.where(depth > 0, depth.astype(np.float32) / float(self.depth_scale), np.nan)
        gx = np.abs(np.gradient(d, axis=1))
        gy = np.abs(np.gradient(d, axis=0))
        return np.fmax(np.nan_to_num(gx, nan=0.0), np.nan_to_num(gy, nan=0.0)) * 1000.0

    def _eroded(self, mask, label):
        """The mask without its boundary pixels, whose depth is part background."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        n = int(p('mask_erosion'))
        if n <= 0:
            return mask
        import cv2

        small = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=n)
        kept = int(small.sum())
        if kept < int(p('mask_min_pixels')):
            self.get_logger().warn(
                f'{label}: {n} px erosion would leave {kept} pixels (min '
                f"{p('mask_min_pixels')}); using the mask as detected, so its edge pixels "
                'will fly towards the camera', throttle_duration_sec=10.0)
            return mask
        return small.astype(bool)

    def process(self):
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is None:
            self.get_logger().warn("No synchronized color+depth frame received yet",
                                   throttle_duration_sec=10.0)
            return
        if self.intrinsics is None:
            self.get_logger().warn("Waiting for camera_info", throttle_duration_sec=10.0)
            return

        color_msg, depth_msg = frame
        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.shape[:2] != rgb.shape[:2]:
            self.get_logger().error(
                f"Depth {depth.shape[:2]} and color {rgb.shape[:2]} differ -- make sure the "
                "depth topic is aligned_depth_to_color", throttle_duration_sec=10.0)
            return
        if (self.intrinsics.width, self.intrinsics.height) != (rgb.shape[1], rgb.shape[0]):
            self.get_logger().error(
                f"Intrinsics are for {self.intrinsics.width}x{self.intrinsics.height} but the "
                f"image is {rgb.shape[1]}x{rgb.shape[0]}; intrinsics are resolution specific, "
                "refusing to deproject", throttle_duration_sec=10.0)
            return

        t0 = self.get_clock().now()
        detections = self.detector.detect_and_segment(rgb)
        dt = (self.get_clock().now() - t0).nanoseconds / 1e9
        self.cycle += 1

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if p('publish_scene_cloud'):
            stride = max(int(p('scene_cloud_stride')), 1)
            every = np.zeros(depth.shape[:2], dtype=bool)
            every[::stride, ::stride] = True
            # Unfiltered: the 5 mm density filter would strip a cloud this sparse.
            # The planner filters its shelf crop at a matching radius.
            scene, scene_colors = mask_to_pointcloud(rgb, depth, every, self.intrinsics,
                                                     depth_scale=self.depth_scale,
                                                     depth_trunc=self.depth_trunc,
                                                     remove_outliers=False)
            if scene.shape[0]:
                self.scene_pub.publish(points_to_ros2_pointcloud2(
                    scene, scene_colors, color_msg.header.frame_id, color_msg.header.stamp))

        limit = float(p('depth_gradient_max_mm_px'))
        steep = self._depth_gradient(depth) > limit if limit > 0 else None
        published = []
        for det in detections:
            det['mask'] = self._eroded(det['mask'], det['label'])
            if steep is not None:
                kept = det['mask'] & ~steep
                if int(kept.sum()) >= int(p('mask_min_pixels')):
                    det['flying_pixels'] = int(det['mask'].sum() - kept.sum())
                    det['mask'] = kept
                else:
                    self.get_logger().warn(
                        f"{det['label']}: the depth-gradient filter would leave "
                        f'{int(kept.sum())} pixels; keeping the eroded mask',
                        throttle_duration_sec=10.0)
            points, colors = mask_to_pointcloud(rgb, depth, det['mask'], self.intrinsics,
                                                depth_scale=self.depth_scale,
                                                depth_trunc=self.depth_trunc)
            det['n_points'] = points.shape[0]
            if points.shape[0] == 0:
                continue
            msg = points_to_ros2_pointcloud2(points, colors, color_msg.header.frame_id,
                                             color_msg.header.stamp)
            self.cloud_pubs[det['label']].publish(msg)
            mask_msg = self.bridge.cv2_to_imgmsg(det['mask'].astype(np.uint8) * 255,
                                                 encoding='mono8')
            mask_msg.header = color_msg.header
            self.mask_pubs[det['label']].publish(mask_msg)
            published.append(f"{det['label']}({points.shape[0]} pts, {det['score']:.2f})")
            if self.save_pcd:
                save_pcd_ascii(os.path.join(self.output_dir,
                                            f"{det['label']}_{self.cycle:05d}.pcd"),
                               points, colors)

        stamp = color_msg.header.stamp
        self.info_pub.publish(String(data=json.dumps({
            'cycle': self.cycle,
            'stamp_ns': int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
            'detector_mode': p('detector_mode'), 'grounding_model': p('grounding_model'),
            'sam_model': p('sam_model'), 'box_threshold': p('box_threshold'),
            'text_threshold': p('text_threshold'), 'target_objects_yaml': p('target_objects_yaml'),
            'prompts': {t['name']: t['prompt'] for t in self.targets},
            'detections': [{'label': d['label'], 'score': float(d['score']),
                            'n_points': int(d.get('n_points', 0)),
                            'box': [float(v) for v in d['box']]} for d in detections],
        })))
        self.get_logger().info(
            f"[{self.cycle}] inference {dt:.2f} s, published: {published or 'nothing'}")
        self._publish_debug_image(rgb, detections, color_msg.header)

    def _publish_debug_image(self, rgb, detections, header):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for det in detections:
            overlay = bgr.copy()
            overlay[det['mask']] = (0, 255, 0)
            bgr = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)
            x0, y0, x1, y1 = (int(v) for v in det['box'])
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(bgr, f"{det['label']} {det['score']:.2f}", (x0, max(y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        msg.header = header
        self.image_pub.publish(msg)
        if self.show_debug_window:
            cv2.imshow("YCB Detections", bgr)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YCBSegmentationNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # A Ctrl-C can arrive again while tearing down; don't dump a traceback for it
        try:
            if node.show_debug_window:
                cv2.destroyAllWindows()
            executor.shutdown()
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
