"""X6: a ROS node that captures perception frames and robot state for the record.

* **Perception frames.** segmentation_node publishes each object cloud with the
  colour image's stamp, and (since X6) the mask on /ycb_masks/<object> with the
  same stamp. While a capture is running this node buffers the colour and aligned
  depth images, so every cloud used can be saved together with the exact RGB,
  depth and mask it was deprojected from. Only clouds stamped *after* the capture
  started are accepted, so each one is a fresh segmentation of a fresh frame.
* **Robot state.** /joint_states, the Franka robot state and the gripper's joint
  states arrive at up to 1 kHz. They are subscribed *raw* (bytes, no
  deserialisation) and decoded only when sampled at the control rate, because a
  Python callback that converts every message starves the 20 Hz control loop of
  the GIL (measured in tpgpt_executor_node: 1.8 s stalls).
"""

import collections
import json
import threading
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from std_msgs.msg import String

ARM_JOINTS = tuple(f'fr3_joint{i}' for i in range(1, 8))


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class X6CaptureNode(Node):
    def __init__(self, object_name, topics, robot_state_topic=None, gripper_state_topic=None,
                 **kwargs):
        super().__init__('x6_capture_node', **kwargs)
        self.object_name = object_name
        self.topics = topics
        self._lock = threading.Lock()
        self._capturing = False
        self._colors = collections.deque(maxlen=240)
        self._depths = collections.deque(maxlen=240)
        self._masks = collections.deque(maxlen=30)
        self._infos = collections.deque(maxlen=30)
        self._clouds = []
        self.camera_info = None
        self._raw = {}

        self.create_subscription(Image, topics['color'], self._color_cb, qos_profile_sensor_data)
        self.create_subscription(Image, topics['depth'], self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, topics['camera_info'],
                                 lambda m: setattr(self, 'camera_info', m), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, f'/ycb_pointclouds/{object_name}', self._cloud_cb, 5)
        self.create_subscription(Image, f'/ycb_masks/{object_name}', self._mask_cb, 5)
        self.create_subscription(String, '/ycb_detections/info', self._info_cb, 5)

        from geometry_msgs.msg import PoseStamped
        self._pose_seen = None
        self.create_subscription(PoseStamped, topics.get('current_pose', '/current_pose'),
                                 lambda b: setattr(self, '_pose_seen', time.monotonic()), 1,
                                 raw=True)
        self._raw_types = {'joints': JointState}
        self.create_subscription(JointState, topics.get('joint_states', '/joint_states'),
                                 lambda b: self._raw.__setitem__('joints', b), 5, raw=True)
        if gripper_state_topic:
            self._raw_types['gripper'] = JointState
            self.create_subscription(JointState, gripper_state_topic,
                                     lambda b: self._raw.__setitem__('gripper', b), 5, raw=True)
        self.robot_state_type = None
        if robot_state_topic:
            try:
                from franka_msgs.msg import FrankaRobotState
                self.robot_state_type = FrankaRobotState
                self._raw_types['robot_state'] = FrankaRobotState
                self.create_subscription(FrankaRobotState, robot_state_topic,
                                         lambda b: self._raw.__setitem__('robot_state', b), 5,
                                         raw=True)
            except ImportError:
                self.get_logger().warn('franka_msgs not importable: no wrench / end-effector mass')

    # ------------------------------------------------------------------ callbacks
    def _color_cb(self, msg):
        if self._capturing:
            self._colors.append(msg)

    def _depth_cb(self, msg):
        if self._capturing:
            self._depths.append(msg)

    def _mask_cb(self, msg):
        self._masks.append(msg)

    def _info_cb(self, msg):
        self._infos.append(msg)

    def _cloud_cb(self, msg):
        with self._lock:
            if self._capturing:
                self._clouds.append(msg)

    # ------------------------------------------------------------------ robot state
    def _decode(self, key):
        data = self._raw.get(key)
        if data is None:
            return None
        try:
            return deserialize_message(data, self._raw_types[key])
        except Exception:  # noqa: BLE001 - a malformed sample is recorded as missing
            return None

    def joint_positions(self):
        msg = self._decode('joints')
        if msg is None:
            return None
        pos = dict(zip(msg.name, msg.position))
        if not all(j in pos for j in ARM_JOINTS):
            return None
        return [float(pos[j]) for j in ARM_JOINTS]

    def gripper_width(self):
        """Sum of the finger joint positions (m), from the gripper's own joint states
        or from /joint_states if the fingers are published there."""
        for key in ('gripper', 'joints'):
            msg = self._decode(key)
            if msg is None:
                continue
            fingers = [p for n, p in zip(msg.name, msg.position) if 'finger' in n]
            if fingers:
                return float(sum(fingers))
        return None

    def robot_state(self):
        """(wrench [fx fy fz tx ty tz] in base, end-effector mass kg, F_T_EE xyz) or Nones."""
        msg = self._decode('robot_state')
        if msg is None:
            return None, None, None
        w = msg.o_f_ext_hat_k.wrench
        wrench = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        mass = float(msg.inertia_ee.inertia.m)
        p = msg.f_t_ee.pose.position
        return wrench, mass, [p.x, p.y, p.z]

    def wait_for_pose(self, timeout, max_age=0.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            seen = self._pose_seen
            if seen is not None and time.monotonic() - seen < max_age:
                return True
            time.sleep(0.05)
        return False

    def wait_for_robot_state(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._raw.get('robot_state') is not None:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ perception
    def capture(self, n_frames, timeout):
        """Wait for `n_frames` clouds of the object segmented from frames taken after
        this call, and return them with their RGB, depth and mask.

        Returns (frames, freshness) where each frame is a dict with keys
        cloud, color, depth, mask (any may be None), info (json dict or None).
        """
        start = self.get_clock().now()
        start_ns = start.nanoseconds
        with self._lock:
            self._clouds = []
            self._capturing = True
        self._colors.clear()
        self._depths.clear()
        accepted, freshness = [], 'stamp_after_request'
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline and len(accepted) < n_frames:
                time.sleep(0.05)
                with self._lock:
                    pending, self._clouds = self._clouds, []
                for msg in pending:
                    if stamp_ns(msg.header) >= start_ns:
                        # Match RGB / depth now: a capture spans ~n_frames x the
                        # segmentation period (~9 s), longer than the 240-frame
                        # buffer holds at 30 fps, so the first frame's image would
                        # be evicted by the end.
                        accepted.append((msg, self._match_images(msg)))
            if len(accepted) < n_frames and not accepted:
                freshness = 'none_received'
            # give the colour / depth / mask buffers a moment to hold the matching frames
            time.sleep(0.1)
            frames = [self._assemble(msg, images) for msg, images in accepted[:n_frames]]
        finally:
            with self._lock:
                self._capturing = False
        return frames, freshness

    @staticmethod
    def _nearest(buffer, ns, tol_ns):
        best, gap = None, None
        for m in list(buffer):
            d = abs(stamp_ns(m.header) - ns)
            if gap is None or d < gap:
                best, gap = m, d
        return best if gap is not None and gap <= tol_ns else None

    def _match_images(self, cloud):
        ns = stamp_ns(cloud.header)
        return (self._nearest(self._colors, ns, 0), self._nearest(self._depths, ns, 50_000_000))

    def _assemble(self, cloud, images):
        ns = stamp_ns(cloud.header)
        color, depth = images
        # an image that arrived after its cloud is still in the buffer
        if color is None or depth is None:
            late_color, late_depth = self._match_images(cloud)
            color = late_color if color is None else color
            depth = late_depth if depth is None else depth

        info = None
        for m in reversed(list(self._infos)):
            try:
                data = json.loads(m.data)
            except ValueError:
                continue
            if int(data.get('stamp_ns', -1)) == ns:
                info = data
                break
        return {'cloud': cloud, 'color': color, 'depth': depth,
                'mask': self._nearest(self._masks, ns, 0),
                'info': info, 'camera_info': self.camera_info}


def cloud_to_base(calib, cloud_msg):
    """(points_base, rgb) using the same CalibrationProvider the planner uses."""
    from ycb_pointcloud_pipeline import camera_calibration as cc
    from ycb_pointcloud_pipeline.ros_utils import pointcloud2_to_xyzrgb
    T = calib.T_base_optical(cloud_msg.header.frame_id or calib.cal.optical_frame)
    if T is None:
        return None, None, None
    xyz, rgb = pointcloud2_to_xyzrgb(cloud_msg)
    return cc.transform_points(T, xyz), rgb, T


def save_frames(directory, frames, calib, prefix=''):
    """Write rgb/depth/mask/cloud of each captured frame; the last one also under the
    brief's plain names (rgb.png, depth.npy, mask.png). Returns per-frame metadata."""
    import os

    import cv2
    from cv_bridge import CvBridge

    from ycb_pointcloud_pipeline.x6_common import write_json

    bridge = CvBridge()
    os.makedirs(directory, exist_ok=True)
    meta = []
    for k, f in enumerate(frames):
        tag = f'{prefix}frame{k}'
        entry = {'index': k, 'cloud_frame_id': f['cloud'].header.frame_id,
                 'has_rgb': f['color'] is not None, 'has_depth': f['depth'] is not None,
                 'has_mask': f['mask'] is not None, 'detection': f['info']}
        pts, rgb, T = cloud_to_base(calib, f['cloud'])
        if pts is not None:
            np.save(os.path.join(directory, f'{tag}_cloud_base.npy'), pts.astype(np.float32))
            entry['n_points'] = int(len(pts))
            entry['T_base_optical'] = T.tolist()
        if f['color'] is not None:
            bgr = bridge.imgmsg_to_cv2(f['color'], desired_encoding='bgr8')
            cv2.imwrite(os.path.join(directory, f'{tag}_rgb.png'), bgr)
        if f['depth'] is not None:
            depth = bridge.imgmsg_to_cv2(f['depth'], desired_encoding='passthrough')
            np.save(os.path.join(directory, f'{tag}_depth.npy'), np.asarray(depth))
            entry['depth_encoding'] = f['depth'].encoding
        if f['mask'] is not None:
            mask = bridge.imgmsg_to_cv2(f['mask'], desired_encoding='mono8')
            cv2.imwrite(os.path.join(directory, f'{tag}_mask.png'), mask)
        meta.append(entry)
    if frames:
        last = len(frames) - 1
        for src, dst in ((f'{prefix}frame{last}_rgb.png', f'{prefix}rgb.png'),
                         (f'{prefix}frame{last}_depth.npy', f'{prefix}depth.npy'),
                         (f'{prefix}frame{last}_mask.png', f'{prefix}mask.png')):
            s = os.path.join(directory, src)
            if os.path.isfile(s):
                import shutil
                shutil.copyfile(s, os.path.join(directory, dst))
        info = frames[last]['camera_info']
        if info is not None:
            write_json(os.path.join(directory, f'{prefix}camera_info.json'), {
                'K': list(info.k), 'D': list(info.d), 'width': info.width, 'height': info.height,
                'distortion_model': info.distortion_model, 'frame_id': info.header.frame_id})
    return meta


def fuse_frames(frames, calib, method_perception, workspace_min, workspace_max):
    """Fuse captured clouds exactly as the planner does (ros_utils.fuse_clouds)."""
    from ycb_pointcloud_pipeline.ros_utils import fuse_clouds
    clouds = []
    for f in frames:
        pts, _, _ = cloud_to_base(calib, f['cloud'])
        if pts is not None:
            clouds.append(pts)
    if not clouds:
        return np.zeros((0, 3)), {'n_raw': 0}
    p = method_perception
    return fuse_clouds(clouds, workspace_min, workspace_max, float(p['voxel_size']),
                       float(p['outlier_radius']), int(p['outlier_min_neighbors']),
                       int(p['max_points']))


__all__ = ['X6CaptureNode', 'cloud_to_base', 'fuse_frames', 'save_frames']
