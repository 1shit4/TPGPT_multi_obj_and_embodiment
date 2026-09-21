#!/usr/bin/env python3
"""
Verify (and optionally re-estimate) the camera intrinsics and the robot base ->
camera extrinsics before any TPGPT plan is executed. Nothing here moves the
robot; it only reads /current_pose, the camera and RViz clicks.

Checks, from cheapest to most conclusive
----------------------------------------
1. INTRINSICS: camera_info vs. the YAML values (fx, fy, cx, cy, resolution) and a
   warning if camera_info carries non-zero distortion (deprojection ignores it).

2. TCP OVERLAY IMAGE  /tpgpt/verify/overlay  (view with rqt_image_view)
   fr3_hand_tcp from /current_pose is projected into the colour image with the
   calibration: a cross at the TCP plus its x (red), y (green), z (blue) axes,
   and the base origin axes when visible. If the extrinsics and intrinsics are
   right, the cross sits between the fingertips wherever the arm is. The text
   line reports the depth measured at that pixel against the depth the
   calibration predicts.

3. SCENE CLOUD IN BASE  /tpgpt/verify/scene_cloud_base  +  TABLE PLANE
   The whole aligned depth image, subsampled, transformed into base. In RViz
   (fixed frame `base`, RobotModel displayed) the table, the robot's own links
   and objects must coincide with reality. A RANSAC plane fitted inside
   `table_roi_*` reports the table tilt against base +z and its height; with
   verification.expected_table_height set, both are checked against tolerances.

4. OBJECT CLOUDS IN BASE  /tpgpt/base_clouds/<name>
   Every /ycb_pointclouds/<name> republished in base, with its centroid, lowest
   point (should equal the table height) and extent logged.

5. TOUCHED-POINT CORRESPONDENCES (quantitative, and a calibration by itself)
   The latest RViz click is paired with the TCP pose at the moment of the
   service call, so the order is CLICK FIRST, THEN TOUCH:
   a. With the arm out of view, click a distinct feature (a box corner, the top
      of a small ball) on /tpgpt/verify/scene_cloud_base in RViz with the
      "Publish Point" tool (/clicked_point).
   b. Hand-guide the fingertip centre (or `probe_offset` point) onto that
      same feature and hold it there.
   c. ros2 service call /calibration_check_node/record_correspondence std_srvs/srv/Trigger
   Repeat for >= 4 well-spread points (not collinear, different heights).
      ros2 service call /calibration_check_node/solve_extrinsics std_srvs/srv/Trigger
   prints the residual of the CURRENT calibration at every point (the
   verification) and a least-squares base -> camera fit with its residuals,
   as a ready-to-paste YAML block (also written to `output_dir`).

Run (robot container up in `robot` or `fake` mode, cameras running):
    ros2 run ycb_pointcloud_pipeline calibration_check_node
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
from ycb_pointcloud_pipeline.ros_utils import (
    CalibrationProvider, cylinder_marker, pointcloud2_to_xyz, pose_to_matrix,
    sphere_list_marker, text_marker, transform_msg_to_matrix, xyz_to_pointcloud2,
)
from ycb_pointcloud_pipeline.segmentation_node import (
    _default_target_objects_path, load_target_objects,
)


class CalibrationCheckNode(Node):
    def __init__(self):
        super().__init__('calibration_check_node')
        self.declare_parameter('calibration_yaml', default_calibration_yaml())
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('current_pose_topic', '/current_pose')
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        self.declare_parameter('target_objects_yaml', _default_target_objects_path())
        self.declare_parameter('period', 1.0)
        self.declare_parameter('depth_stride', 4)
        self.declare_parameter('depth_trunc', 2.0)
        self.declare_parameter('table_roi_min', [0.2, -0.6, -0.3])   # base frame, m
        self.declare_parameter('table_roi_max', [1.0, 0.6, 0.15])
        # Where the touching point is in fr3_hand_tcp coordinates. [0,0,0] =
        # fingertip centre. For touching with the closed fingertips' ends use
        # roughly [0, 0, 0.01].
        self.declare_parameter('probe_offset', [0.0, 0.0, 0.0])
        self.declare_parameter('max_click_age', 120.0)
        self.declare_parameter('output_dir', os.path.expanduser('~/tpgpt_calibration'))

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.calib = CalibrationProvider(self, p('calibration_yaml'), self.tf_buffer)
        self.cal = self.calib.cal
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color = self.depth = self.info_intr = None
        self.optical_frame = self.cal.optical_frame
        self.tcp = None           # (monotonic, T_base_tcp)
        self.click = None         # (monotonic, PointStamped)
        self.pairs = []           # list of dicts
        self._intrinsics_reported = None

        self.create_subscription(Image, p('color_topic'), self._color_cb, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic'), self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('camera_info_topic'), self._info_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 1)
        self.create_subscription(PointStamped, p('clicked_point_topic'), self._click_cb, 10)

        self.overlay_pub = self.create_publisher(Image, '/tpgpt/verify/overlay', 1)
        self.scene_pub = self.create_publisher(PointCloud2, '/tpgpt/verify/scene_cloud_base', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/tpgpt/verify/markers', 1)
        self.object_pubs = {}
        for t in load_target_objects(p('target_objects_yaml')):
            name = t['name']
            self.object_pubs[name] = self.create_publisher(
                PointCloud2, f'/tpgpt/base_clouds/{name}', 1)
            self.create_subscription(PointCloud2, f'/ycb_pointclouds/{name}',
                                     lambda msg, n=name: self._object_cb(n, msg), 1)

        self.create_service(Trigger, '~/record_correspondence', self._record_srv)
        self.create_service(Trigger, '~/solve_extrinsics', self._solve_srv)
        self.create_service(Trigger, '~/clear_correspondences', self._clear_srv)
        self.create_timer(p('period'), self._tick)
        self.get_logger().info('Calibration check running. Overlay: /tpgpt/verify/overlay; '
                               'RViz: /tpgpt/verify/scene_cloud_base, /tpgpt/base_clouds/*')

    # ------------------------------------------------------------------ inputs
    def _color_cb(self, msg):
        with self.lock:
            self.color = msg
            if msg.header.frame_id:
                self.optical_frame = msg.header.frame_id

    def _depth_cb(self, msg):
        with self.lock:
            self.depth = msg

    def _info_cb(self, msg):
        intr = cc.Intrinsics.from_camera_info_fields(msg.k, msg.width, msg.height,
                                                     msg.distortion_model, msg.d)
        with self.lock:
            self.info_intr = intr
        key = (intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height)
        if key != self._intrinsics_reported:
            self._intrinsics_reported = key
            self._report_intrinsics(intr)

    def _pose_cb(self, msg):
        # Up to 1 kHz from pose_broadcaster: store only, convert when used.
        self.tcp = (time.monotonic(), msg)

    def _tcp_matrix(self):
        tcp = self.tcp
        if tcp is None or tcp[1].header.frame_id not in ('', self.cal.base_frame):
            return None
        return tcp[0], pose_to_matrix(tcp[1].pose)

    def _click_cb(self, msg):
        with self.lock:
            self.click = (time.monotonic(), msg)
        self.get_logger().info(f'Clicked point in {msg.header.frame_id}: '
                               f'[{msg.point.x:.4f}, {msg.point.y:.4f}, {msg.point.z:.4f}]')

    # ------------------------------------------------------------------ helpers
    def _report_intrinsics(self, intr):
        log = self.get_logger()
        worst, text = self.cal.intrinsics_yaml.compare(intr)
        tol = float(self.cal.raw.get('verification', {}).get('max_intrinsics_mismatch_px', 2.0))
        msg = (f'camera_info intrinsics fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.cx:.2f} '
               f'cy={intr.cy:.2f} {intr.width}x{intr.height}; YAML - camera_info: {text}')
        # rclpy caches one logging context per CALL SITE and raises if the severity
        # changes there, so info and warn must sit on separate lines.
        if worst <= tol:
            log.info(msg)
        else:
            log.warn(msg + f'  -> MISMATCH above {tol}px. Using '
                     f'{self.cal.intrinsics_source} intrinsics.')
        if intr.has_distortion():
            log.warn(f'camera_info has non-zero distortion {intr.distortion}; deprojection '
                     'assumes a rectified image, expect errors towards the image edges.')

    def _intrinsics(self):
        with self.lock:
            info = self.info_intr
        try:
            return self.cal.intrinsics(info)
        except ValueError:
            return None

    def _T_base_optical(self):
        return self.calib.T_base_optical(self.optical_frame)

    def _to_optical(self, click_msg, T_base_opt):
        frame = click_msg.header.frame_id
        p = np.array([click_msg.point.x, click_msg.point.y, click_msg.point.z])
        if frame == self.optical_frame:
            return p
        if frame == self.cal.base_frame:
            # A click on a cloud displayed in base with the CURRENT calibration:
            # inverting that same calibration recovers the camera measurement exactly.
            return cc.transform_points(cc.invert_transform(T_base_opt), p)[0]
        tf = self.tf_buffer.lookup_transform(self.optical_frame, frame, Time())
        return cc.transform_points(transform_msg_to_matrix(tf), p)[0]

    # ------------------------------------------------------------------ periodic
    def _tick(self):
        T = self._T_base_optical()
        intr = self._intrinsics()
        with self.lock:
            color, depth = self.color, self.depth
        tcp = self._tcp_matrix()
        if T is None or intr is None or color is None:
            self.get_logger().info('Waiting for camera images, camera_info and calibration TF',
                                   throttle_duration_sec=10.0)
            return
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        depth_img = None
        if depth is not None:
            depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            self._scene_check(depth_img, intr, T, color, stamp, markers)
        self._overlay(color, depth_img, intr, T, tcp)
        if markers.markers:
            self.marker_pub.publish(markers)

    def _scene_check(self, depth_img, intr, T, color_msg, stamp, markers):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if depth_img.shape[:2] != (intr.height, intr.width):
            self.get_logger().warn(f'depth image {depth_img.shape[:2]} does not match intrinsics '
                                   f'{intr.height}x{intr.width}; is it aligned to colour?',
                                   throttle_duration_sec=10.0)
            return
        pts_opt, pix = cc.depth_image_to_points(depth_img, intr, self.cal.depth_scale,
                                                p('depth_trunc'), stride=int(p('depth_stride')))
        if len(pts_opt) < 100:
            return
        pts = cc.transform_points(T, pts_opt)
        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        colors = rgb[pix[:, 0], pix[:, 1]]
        msg = xyz_to_pointcloud2(pts, self.cal.base_frame, stamp, colors)
        self.scene_pub.publish(msg)

        roi, _ = cc.crop_box(pts, p('table_roi_min'), p('table_roi_max'))
        if len(roi) < 200:
            self.get_logger().warn('Too few depth points inside table_roi_* for the plane '
                                   'check (is the placeholder calibration putting the table '
                                   'elsewhere?)', throttle_duration_sec=15.0)
            return
        try:
            n, d, inliers = cc.fit_plane_ransac(roi, threshold=0.006, iterations=200)
        except ValueError:
            return
        tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
        # Height of the plane under the ROI centre.
        centre = roi[inliers].mean(axis=0)
        height = float((d - n[0] * centre[0] - n[1] * centre[1]) / max(n[2], 1e-6))
        ver = self.cal.raw.get('verification', {}) or {}
        expected = self.cal.expected_table_height
        text = (f'table plane: tilt {tilt:.2f} deg from base +z, height {height:+.4f} m '
                f'({int(inliers.sum())}/{len(roi)} inliers)')
        ok = tilt <= float(ver.get('max_table_tilt_deg', 1.5))
        if expected is not None:
            err = height - expected
            text += f', expected {expected:+.4f} m -> error {err * 1000:+.1f} mm'
            ok = ok and abs(err) <= float(ver.get('max_table_height_error_m', 0.005))
        # Separate call sites: `ok` flips whenever the arm moves through the table
        # ROI, and switching severity at one site raises ValueError (see above).
        if ok:
            self.get_logger().info(text, throttle_duration_sec=5.0)
        else:
            self.get_logger().warn(text + '  -> CHECK EXTRINSICS',
                                   throttle_duration_sec=5.0)
        markers.markers.append(cylinder_marker('table_plane', 0, [centre[0], centre[1], height],
                                               0.3, 0.002, self.cal.base_frame, stamp,
                                               (0.2, 0.8, 0.2, 0.3) if ok else (0.9, 0.2, 0.2, 0.3)))
        markers.markers.append(text_marker('table_plane', 1, [centre[0], centre[1], height + 0.05],
                                           f'tilt {tilt:.1f} deg  z {height:+.3f}',
                                           self.cal.base_frame, stamp))
        if self.pairs:
            markers.markers.append(sphere_list_marker(
                'touched_points', 0, [q['p_base_tcp'] for q in self.pairs], self.cal.base_frame,
                stamp, (1.0, 0.6, 0.0), 0.015))

    def _overlay(self, color_msg, depth_img, intr, T_base_opt, tcp):
        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8').copy()
        T_opt_base = cc.invert_transform(T_base_opt)

        def px(p_base):
            p_opt = cc.transform_points(T_opt_base, p_base)
            uv, inside = cc.project(p_opt, intr)
            return uv, inside, p_opt[:, 2]

        def draw_axes(origin, R, length, label):
            pts = np.vstack([origin, origin + length * R[:, 0], origin + length * R[:, 1],
                             origin + length * R[:, 2]])
            uv, inside, _ = px(pts)
            if not inside[0]:
                return False
            o = tuple(int(v) for v in uv[0])
            for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                if inside[i + 1]:
                    cv2.arrowedLine(bgr, o, tuple(int(v) for v in uv[i + 1]), c, 2, tipLength=0.2)
            cv2.putText(bgr, label, (o[0] + 6, o[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return True

        draw_axes(np.zeros(3), np.eye(3), 0.15, 'base')
        line = 'no /current_pose'
        if tcp is not None and time.monotonic() - tcp[0] < 1.0:
            T_tcp = tcp[1]
            probe = T_tcp[:3, 3] + T_tcp[:3, :3] @ np.asarray(self.get_parameter('probe_offset').value)
            uv, inside, z = px(probe)
            if inside[0]:
                u, v = int(uv[0, 0]), int(uv[0, 1])
                cv2.drawMarker(bgr, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                draw_axes(T_tcp[:3, 3], T_tcp[:3, :3], 0.06, 'tcp')
                line = f'TCP pixel ({u},{v}) predicted depth {z[0]:.3f} m'
                if depth_img is not None:
                    win = depth_img[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].astype(float)
                    win = win[win > 0] / self.cal.depth_scale
                    if win.size:
                        line += (f', measured {np.median(win):.3f} m (between the fingers this '
                                 'can be the background)')
            else:
                line = 'TCP is outside the camera image (or behind it) with this calibration'
        for i, q in enumerate(self.pairs):
            uv, inside, _ = px(q['p_base_tcp'])
            if inside[0]:
                cv2.circle(bgr, (int(uv[0, 0]), int(uv[0, 1])), 6, (0, 140, 255), 2)
            uv2, inside2 = cc.project(q['p_optical'], intr)
            if inside2[0]:
                cv2.circle(bgr, (int(uv2[0, 0]), int(uv2[0, 1])), 3, (255, 0, 255), -1)
        placeholder = ' [PLACEHOLDER EXTRINSICS]' if self.cal.camera_is_placeholder else ''
        cv2.putText(bgr, line + placeholder, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        out.header = color_msg.header
        self.overlay_pub.publish(out)

    def _object_cb(self, name, msg):
        T = self.calib.T_base_optical(msg.header.frame_id or self.optical_frame)
        if T is None:
            return
        pts = cc.transform_points(T, pointcloud2_to_xyz(msg))
        if len(pts) == 0:
            return
        self.object_pubs[name].publish(xyz_to_pointcloud2(pts, self.cal.base_frame,
                                                          msg.header.stamp, (255, 200, 0)))
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        self.get_logger().info(
            f'{name}: {len(pts)} pts, centroid {pts.mean(axis=0).round(3).tolist()} m, '
            f'lowest (2%) z {lo[2]:+.3f} m (should be the table height), extent '
            f'{((hi - lo) * 1000).round(0).tolist()} mm', throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ services
    def _record_srv(self, request, response):
        T = self._T_base_optical()
        tcp = self._tcp_matrix()
        with self.lock:
            click = self.click
        now = time.monotonic()
        if T is None or tcp is None or now - tcp[0] > 1.0:
            response.success, response.message = False, 'no fresh /current_pose or calibration TF'
            return response
        if click is None or now - click[0] > self.get_parameter('max_click_age').value:
            response.success, response.message = False, ('no recent /clicked_point; click the '
                                                         'feature in RViz with Publish Point')
            return response
        try:
            p_opt = self._to_optical(click[1], T)
        except Exception as exc:
            response.success, response.message = False, f'cannot express click in optical: {exc}'
            return response
        T_tcp = tcp[1]
        p_tcp = T_tcp[:3, 3] + T_tcp[:3, :3] @ np.asarray(self.get_parameter('probe_offset').value)
        err = float(np.linalg.norm(cc.transform_points(T, p_opt)[0] - p_tcp))
        # The TCP *orientation* is kept too: the probe point is a fixed offset in
        # the hand frame, so without the rotation a recorded pair cannot be
        # re-derived for a different probe_offset afterwards.
        self.pairs.append({'p_optical': p_opt, 'p_base_tcp': p_tcp,
                           'p_base_raw_tcp': T_tcp[:3, 3].copy(),
                           'R_base_tcp': T_tcp[:3, :3].copy()})
        with self.lock:
            self.click = None   # one click per correspondence
        response.success = True
        response.message = (f'pair {len(self.pairs)}: touched {p_tcp.round(4).tolist()} (base), '
                            f'camera {p_opt.round(4).tolist()} (optical); current calibration '
                            f'error {err * 1000:.1f} mm')
        self.get_logger().info(response.message)
        return response

    def _clear_srv(self, request, response):
        self.pairs.clear()
        response.success, response.message = True, 'cleared'
        return response

    def _solve_srv(self, request, response):
        T_current = self._T_base_optical()
        if len(self.pairs) < 3 or T_current is None:
            response.success = False
            response.message = f'need >= 3 pairs and a calibration TF (have {len(self.pairs)})'
            return response
        P_opt = np.array([q['p_optical'] for q in self.pairs])
        P_base = np.array([q['p_base_tcp'] for q in self.pairs])
        current = np.linalg.norm(cc.transform_points(T_current, P_opt) - P_base, axis=1)
        try:
            T_fit, residual = cc.fit_rigid_transform(P_opt, P_base)
        except ValueError as exc:
            response.success, response.message = False, str(exc)
            return response
        if self.cal.child_is_optical:
            T_link_opt = np.eye(4)
            child = self.cal.camera_child_frame
        else:
            T_link_opt = cc.invert_transform(self.cal.T_base_camera) @ T_current
            child = self.cal.camera_child_frame
        snippet = cc.camera_link_yaml_snippet(T_fit, T_link_opt, child)
        delta_t = np.linalg.norm(T_fit[:3, 3] - T_current[:3, 3]) * 1000
        delta_r = cc.rotation_angle_deg(T_fit[:3, :3], T_current[:3, :3])
        lines = [
            f'{len(self.pairs)} correspondences',
            'CURRENT calibration error per point (mm): '
            + ', '.join(f'{v * 1000:.1f}' for v in current)
            + f'  | rms {np.sqrt(np.mean(current ** 2)) * 1000:.1f} mm',
            'FITTED calibration residual per point (mm): '
            + ', '.join(f'{v * 1000:.1f}' for v in residual)
            + f'  | rms {np.sqrt(np.mean(residual ** 2)) * 1000:.1f} mm',
            f'fitted differs from current by {delta_t:.1f} mm and {delta_r:.2f} deg',
            'If the fitted residuals are small (a few mm) and spread points were used, paste '
            'into camera_calibration.yaml:', snippet,
        ]
        text = '\n'.join(lines)
        out_dir = self.get_parameter('output_dir').value
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, time.strftime('extrinsics_fit_%Y%m%d_%H%M%S.yaml'))
        with open(path, 'w') as f:
            f.write('# ' + '\n# '.join(lines[:-1]) + '\n' + snippet)
            f.write('# correspondences: p_optical -> p_base_tcp | raw_tcp | R_base_tcp\n')
            for q in self.pairs:
                raw = q.get('p_base_raw_tcp')
                rot = q.get('R_base_tcp')
                f.write(f'#   {q["p_optical"].round(5).tolist()} -> '
                        f'{q["p_base_tcp"].round(5).tolist()}'
                        + ('' if raw is None else f' | {raw.round(5).tolist()}')
                        + ('' if rot is None else f' | {rot.round(6).flatten().tolist()}')
                        + '\n')
        self.get_logger().info(text + f'\n(written to {path})')
        response.success, response.message = True, text
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationCheckNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
