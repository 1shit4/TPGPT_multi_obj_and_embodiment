#!/usr/bin/env python3
"""
Teach the TPGPT source demonstration by hand on the FR3, and save it in the
format tpgpt_planner_node loads (tpgpt.hardware.source_bundle.SourceBundle).

  /ycb_pointclouds/<object> --capture_scene--> object cloud in base (arm out of view)
  /current_pose (1 kHz)     --start/stop_recording, mark_grasp/mark_release-->
      raw.npz --tpgpt.hardware.teach--> source_bundle/ + teach.json + RViz markers

Teach modes (parameter `teach_mode`)
  compliant (default): start_teaching loads config/tpgpt_teaching_controller.yaml
      (a CRISP CartesianController with zero stiffness and light damping) and
      strictly switches it in, so the arm floats and can be moved by hand.
      stop_recording and hold switch back to `hold_controller`
      (cartesian_impedance_controller), which CRISP seeds with the measured pose,
      so the arm stays where it was let go. FCI must be active: without it the
      robot container cannot connect and no state is published at all.
  passive: nothing is switched. For Franka's own guiding mode, if /current_pose
      keeps streaming while guiding with FCI active (check with
      `ros2 topic hz /current_pose`).

Gripper (parameter `gripper_mode`): with franka_grasp (default) mark_grasp also
CLOSES the Franka Hand (franka_gripper grasp action, any final width accepted) and
mark_release OPENS it (move action). Outside a recording the keys only drive the
gripper, so the object can be picked up or dropped without recording. The mark
is stamped with the pose when the key arrives; hold the arm still while the
fingers close or open, as the executor pauses there too. gripper_mode none: marks only.
gripper_mode inspire: the same keys call /inspire_hand/close and /inspire_hand/open
(inspire_hand_node) -- the fixed wrap-around grasp with the thumb opposed.

Hand (parameter `end_effector`): panda (default) records fr3_hand_tcp as the fingertip
centre, as before. inspire converts every recorded /current_pose to the Inspire hand's
PALM point and palm frame using config/inspire_hand.yaml (end_effector_config), and
stores that description in raw.npz, so an offline re-conversion uses the same mount.
Teach an Inspire demonstration the way it will be executed: palm against the side of
the upright object, then mark_grasp.

Services (std_srvs/Trigger), normally called by tpgpt_teach_keyboard:
  ~/capture_scene    fuse n_frames NEW object clouds (arm out of view)
  ~/start_teaching   compliant: load + activate the teaching controller
  ~/start_recording  begin recording /current_pose
  ~/mark_grasp       the object is held from now on; closes the gripper
  ~/mark_release     the object is let go from now on; opens the gripper
  ~/stop_recording   end recording, switch to the hold controller, convert, save, plot
  ~/capture_placed   optional: fuse clouds of the placed object, cross-check the placement
  ~/hold             switch to the hold controller now (aborts a recording)

RViz (fixed frame base, latched):
  /tpgpt/teach/markers                  demo label points (blue free, red holding),
                                        live path while recording (magenta), TCP axes at
                                        start/grasp/release, destination disc, summary text
  /tpgpt/teach/object_cloud             captured object cloud (orange)
  /tpgpt/teach/placed_cloud_predicted   object moved by the taught carry (cyan)
  /tpgpt/teach/placed_cloud_measured    capture_placed cloud (green)

Run with the TPGPT environment (it imports tpgpt), robot container up:
    ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=mustard_bottle
    ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard        # second terminal
"""

import json
import os
import signal
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
from ycb_pointcloud_pipeline.ros_utils import (
    CalibrationProvider, ControllerManagerClient, axes_markers, container_config_path,
    cylinder_marker, fuse_clouds, line_strip_marker, pointcloud2_to_xyz, pose_to_matrix,
    sphere_list_marker, text_marker, xyz_to_pointcloud2,
)

NAN = float('nan')
BLUE, RED, MAGENTA = (0.2, 0.4, 1.0), (1.0, 0.1, 0.1), (1.0, 0.2, 1.0)


class TeachAbort(RuntimeError):
    pass


def _stamp_seconds(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


class TpgptTeachNode(Node):
    def __init__(self):
        super().__init__('tpgpt_teach_node')
        d = self.declare_parameter
        d('calibration_yaml', default_calibration_yaml())
        d('object_name', 'mustard_bottle')
        d('cloud_topic', '')                       # '' -> /ycb_pointclouds/<object_name>
        d('n_frames', 3)                           # NEW clouds fused per capture
        d('capture_timeout', 20.0)                 # s to wait for them
        d('voxel_size', 0.003)
        d('outlier_radius', 0.012)
        d('outlier_min_neighbors', 4)
        d('max_points', 8192)
        d('min_tcp_to_object', 0.15)               # m; warn if the arm may be in view at capture
        d('current_pose_topic', '/current_pose')
        d('base_frame', 'base')
        d('teach_mode', 'compliant')               # compliant | passive
        d('controller_manager', '/controller_manager')
        d('teaching_controller', 'tpgpt_teaching_controller')
        d('teaching_controller_type', 'crisp_controllers/CartesianController')
        d('teaching_controller_params_file', '')   # path INSIDE the robot container
        d('hold_controller', 'cartesian_impedance_controller')
        d('support_height', NAN)                   # NaN -> verification.expected_table_height
        d('workspace_min', [0.15, -0.6, -0.25])    # base frame box; TCP outside aborts recording
        d('workspace_max', [0.85, 0.6, 0.7])
        d('max_pose_age', 0.2)                     # s
        d('max_record_seconds', 120.0)
        d('require_scene', True)                   # capture_scene before start_recording
        # Conversion (tpgpt.hardware.teach)
        d('control_freq', 20)
        # Stiffness labels (tpgpt.hardware.teach): 'directional' = stiffness_along N/m along the
        # direction of motion, stiffness_across across it and in every direction while still;
        # 'static' = stiffness / damping throughout (every demonstration before 2026-09-29).
        d('stiffness_profile', 'directional')      # directional | static
        d('stiffness_along', 350.0)                # N/m, directional
        d('stiffness_across', 600.0)               # N/m, directional
        d('stiffness', 400.0)                      # N/m, static, = stock controller
        d('damping', 40.0)                         # N s/m, static
        d('still_speed', 0.01)                     # m/s
        d('max_dwell', 1.0)                        # s
        d('smoothing_window', 7)                   # samples at control_freq
        d('output_dir', os.path.expanduser('~/tpgpt_demos'))
        d('republish_period', 2.0)
        # Gripper, driven by mark_grasp / mark_release
        d('gripper_mode', 'franka_grasp')           # franka_grasp | inspire | none
        d('gripper_namespace', '/franka_gripper')   # /inspire_hand for gripper_mode inspire
        d('end_effector', 'panda')                  # panda | inspire
        d('end_effector_config', '')                # '' -> installed config/inspire_hand.yaml
        d('open_width', 0.08)                       # m
        d('grasp_speed', 0.05)                      # m/s
        d('grasp_force', 20.0)                      # N

        p = self.p
        from tpgpt.hardware import teach  # noqa: F401  fail at start, not after a demo
        self.mode = p('teach_mode')
        if self.mode not in ('compliant', 'passive'):
            raise SystemExit(f"teach_mode must be 'compliant' or 'passive', got {self.mode!r}")
        self.base = p('base_frame')

        # Three nodes, each spun by its own SingleThreadedExecutor thread (see main):
        #   this node   services, watchdog and RViz timers -- one callback at a time
        #   io_node     controller_manager responses, object clouds, TF, so a service
        #               can block on them without deadlocking its own executor
        #   pose_node   the 1 kHz /current_pose stream, alone
        # rclpy's MultiThreadedExecutor was measured on this machine delivering 15 Hz
        # of a 1 kHz pose stream, and, with the pose on its own thread, still leaving
        # 14 gaps > 50 ms (2 > 200 ms) in 18 s of recording whenever services were
        # called. See the docs for the numbers with this layout.
        self.io_node = rclpy.create_node('tpgpt_teach_io')
        self.pose_node = rclpy.create_node('tpgpt_teach_pose_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.io_node)
        self.calib = CalibrationProvider(self, p('calibration_yaml'), self.tf_buffer)
        self.cmc = ControllerManagerClient(self.io_node, p('controller_manager'), error=TeachAbort)
        self.gripper = None
        if p('gripper_mode') == 'franka_grasp':
            try:
                from franka_msgs.action import Grasp, Move
            except ImportError:
                raise SystemExit('gripper_mode franka_grasp needs franka_msgs on the host, or '
                                 'set gripper_mode:=none')
            ns = p('gripper_namespace').rstrip('/')
            # On io_node: goal/result responses arrive on its thread, not this node's.
            self.gripper = {'grasp': ActionClient(self.io_node, Grasp, f'{ns}/grasp'),
                            'move': ActionClient(self.io_node, Move, f'{ns}/move'),
                            'Grasp': Grasp, 'Move': Move}
        elif p('gripper_mode') == 'inspire':
            from ycb_pointcloud_pipeline.ros_utils import InspireHandClient
            ns = p('gripper_namespace').rstrip('/')
            self.gripper = {'inspire': InspireHandClient(
                self.io_node, ns if ns != '/franka_gripper' else '/inspire_hand')}
        elif p('gripper_mode') != 'none':
            raise SystemExit(f"gripper_mode must be 'franka_grasp', 'inspire' or 'none', got "
                             f"{p('gripper_mode')!r}")
        from ycb_pointcloud_pipeline.tpgpt_planner_node import load_end_effector
        self.end_effector = load_end_effector(p('end_effector'), p('end_effector_config'))[0]
        if p('gripper_mode') != 'none' and \
                (p('gripper_mode') == 'inspire') == self.end_effector.is_parallel_jaw:
            raise SystemExit(f"end_effector {self.end_effector.name} does not match gripper_mode "
                             f"{p('gripper_mode')!r}")

        # --- state (pose callback only stores; everything else reads) ---------
        self.lock = threading.Lock()
        self._pose = None            # (monotonic, PoseStamped)
        self._pose_count = 0
        self.recording = False
        self.record_started = None
        self.samples = []            # [PoseStamped] while recording
        self.grasp_stamp = None
        self.release_stamp = None
        self.clouds = []             # (monotonic, points_base)
        self.object_points = None
        self.teaching_active = False
        self.demo = None             # dict of the last saved demonstration

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, '/tpgpt/teach/markers', latched)
        self.object_pub = self.create_publisher(PointCloud2, '/tpgpt/teach/object_cloud', latched)
        self.predicted_pub = self.create_publisher(
            PointCloud2, '/tpgpt/teach/placed_cloud_predicted', latched)
        self.measured_pub = self.create_publisher(
            PointCloud2, '/tpgpt/teach/placed_cloud_measured', latched)

        self.pose_node.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 10)
        cloud_topic = p('cloud_topic') or f"/ycb_pointclouds/{p('object_name')}"
        self.io_node.create_subscription(PointCloud2, cloud_topic, self._cloud_cb, 5)
        for name, cb in (('capture_scene', self._srv_capture_scene),
                         ('start_teaching', self._srv_start_teaching),
                         ('start_recording', self._srv_start_recording),
                         ('mark_grasp', self._srv_mark_grasp),
                         ('mark_release', self._srv_mark_release),
                         ('stop_recording', self._srv_stop_recording),
                         ('capture_placed', self._srv_capture_placed),
                         ('hold', self._srv_hold)):
            self.create_service(Trigger, f'~/{name}', self._wrap(cb))
        self.create_timer(0.1, self._watchdog)
        self.create_timer(0.5, self._preview)
        self.create_timer(p('republish_period'), self._republish)
        self.get_logger().info(
            f"teach_mode {self.mode}; object clouds on {cloud_topic}; poses on "
            f"{p('current_pose_topic')}. Drive it with: ros2 run ycb_pointcloud_pipeline "
            'tpgpt_teach_keyboard')

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ inputs
    def _pose_cb(self, msg):
        # /current_pose arrives at the controller rate (1 kHz). Only store here: a
        # heavier callback starves the other threads of the GIL (measured in the
        # executor: a matrix conversion per message stalled a 20 Hz loop for 1.8 s).
        self._pose = (time.monotonic(), msg)
        self._pose_count += 1
        if self.recording:
            self.samples.append(msg)

    def _cloud_cb(self, msg):
        T = self.calib.T_base_optical(msg.header.frame_id or self.calib.cal.optical_frame)
        if T is None:
            return
        pts = cc.transform_points(T, pointcloud2_to_xyz(msg))
        with self.lock:
            self.clouds.append((time.monotonic(), pts))
            self.clouds = self.clouds[-50:]

    def latest_pose(self, max_age=None):
        max_age = self.p('max_pose_age') if max_age is None else max_age
        pose = self._pose
        if pose is None or time.monotonic() - pose[0] > max_age:
            raise TeachAbort(f"no fresh pose on {self.p('current_pose_topic')} (older than "
                             f'{max_age} s): is the robot container up with FCI active and '
                             'pose_broadcaster running?')
        if pose[1].header.frame_id not in ('', self.base):
            raise TeachAbort(f"{self.p('current_pose_topic')} is in frame "
                             f'{pose[1].header.frame_id!r}, expected {self.base!r}')
        return pose[1]

    def pose_rate(self, window=0.5):
        start = self._pose_count
        time.sleep(window)
        return (self._pose_count - start) / window

    # ------------------------------------------------------------------ services
    def _wrap(self, fn):
        def callback(request, response):
            try:
                response.success, response.message = True, fn()
                self.get_logger().info(response.message)
            except TeachAbort as exc:
                response.success, response.message = False, str(exc)
                self.get_logger().error(response.message)
            except Exception as exc:  # keep the node alive; report the bug
                response.success = False
                response.message = f'{type(exc).__name__}: {exc}'
                self.get_logger().error(response.message)
            return response
        return callback

    def _fuse_new_clouds(self):
        p = self.p
        requested = time.monotonic()
        n = int(p('n_frames'))
        deadline = requested + float(p('capture_timeout'))
        self.get_logger().info(f'waiting for {n} new object clouds (keep the arm out of view)')
        while True:
            with self.lock:
                clouds = [c for t, c in self.clouds if t >= requested]
            if len(clouds) >= n:
                break
            if time.monotonic() > deadline:
                raise TeachAbort(f'capture: {len(clouds)} of {n} new clouds in '
                                 f"{p('capture_timeout')} s; is segmentation_node detecting "
                                 f"{p('object_name')}?")
            time.sleep(0.1)
        pts, stats = fuse_clouds(clouds[-n:], p('workspace_min'), p('workspace_max'),
                                 p('voxel_size'), p('outlier_radius'),
                                 p('outlier_min_neighbors'), p('max_points'))
        if len(pts) == 0:
            raise TeachAbort('capture: the fused cloud is empty after cropping to the workspace')
        note = ''
        try:
            tcp = pose_to_matrix(self.latest_pose(1.0).pose)[:3, 3]
            gap = float(np.linalg.norm(pts - tcp, axis=1).min())
            if gap < p('min_tcp_to_object'):
                note = (f' WARNING: the TCP is {gap * 1000:.0f} mm from the cloud; the arm may '
                        'be in view or bleeding into the mask')
        except TeachAbort:
            pass
        return pts, stats, note

    def _refuse_capture_while_floating(self, name):
        # A capture blocks this node's executor for several seconds, and with it the
        # watchdog; the arm must not be floating meanwhile.
        if self.recording:
            raise TeachAbort(f'{name}: stop the recording first')
        if self.teaching_active:
            raise TeachAbort(f'{name}: the arm is floating; hold it first (h) and move it out '
                             'of view')

    def _srv_capture_scene(self):
        self._refuse_capture_while_floating('capture_scene')
        pts, stats, note = self._fuse_new_clouds()
        self.object_points = pts
        self.object_pub.publish(xyz_to_pointcloud2(pts, self.base, self.get_clock().now().to_msg(),
                                                   (255, 140, 0)))
        low = float(np.percentile(pts[:, 2], 1.0))
        return (f"scene captured: {stats['n_fused']} points from {stats['n_clouds']} clouds, "
                f'centroid {np.round(pts.mean(axis=0), 3).tolist()} m, lowest z {low:+.3f} m, '
                f'extent {np.round(np.ptp(pts, axis=0) * 1000).astype(int).tolist()} mm' + note)

    def _srv_start_teaching(self):
        pose = self.latest_pose()
        if self.mode == 'passive':
            rate = self.pose_rate()
            return (f'passive mode: nothing switched; /current_pose at {rate:.0f} Hz. Guide the '
                    'arm with Franka guiding mode and check the pose follows.')
        try:
            params_file = (self.p('teaching_controller_params_file')
                           or container_config_path('tpgpt_teaching_controller.yaml'))
        except ValueError as exc:
            raise TeachAbort(f'{exc} (teaching_controller_params_file)')
        name = self.p('teaching_controller')
        controllers = self.cmc.ensure_loaded(name, self.p('teaching_controller_type'), params_file)
        if self.p('hold_controller') not in controllers:
            raise TeachAbort(f"{self.p('hold_controller')} is not loaded, so the arm could not be "
                             'held after teaching; refusing to float it')
        self.cmc.activate(name, controllers)
        self.teaching_active = True
        tcp = pose_to_matrix(pose.pose)[:3, 3]
        return (f'{name} ACTIVE: the arm floats (zero stiffness). TCP at '
                f'{np.round(tcp, 3).tolist()}. Support the arm; e-stop in reach.')

    def _srv_start_recording(self):
        if self.recording:
            raise TeachAbort('already recording')
        if self.p('require_scene') and self.object_points is None:
            raise TeachAbort('call capture_scene first (object at its pick pose, arm out of view)')
        self.latest_pose()
        with self.lock:
            self.samples = []
            self.grasp_stamp = self.release_stamp = None
            self.record_started = time.monotonic()
            self.recording = True
        return 'RECORDING: move to the object, mark_grasp, carry, set it down, mark_release, stop'

    def _command_gripper(self, close):
        """Send a close (grasp) or open (move) goal without blocking; the result is logged."""
        label = 'CLOSE' if close else 'OPEN'
        if self.gripper is None:
            return f'gripper {label} not sent (gripper_mode none)'
        if 'inspire' in self.gripper:
            client = self.gripper['inspire']
            missing = client.available(timeout=0.5)
            if missing:
                raise TeachAbort(f'Inspire hand {label}: services {missing} not available '
                                 '(is inspire_hand_node running?)')
            log = self.get_logger()

            def on_answer(future):
                result = future.result()
                if result is not None and result.success:
                    log.info(f'Inspire hand {label}: {result.message}')
                else:
                    log.error(f'Inspire hand {label} failed: '
                              f"{result.message if result is not None else 'no answer'}")
            client.send(close).add_done_callback(on_answer)
            return f'Inspire hand {label} sent'
        name = 'grasp' if close else 'move'
        client = self.gripper[name]
        if not client.server_is_ready():
            raise TeachAbort(f"gripper {label}: action {self.p('gripper_namespace').rstrip('/')}/{name} not available")
        open_w = float(self.p('open_width'))
        if close:
            goal = self.gripper['Grasp'].Goal()
            # Close until the fingers stop on the object; any final width counts as success.
            goal.width, goal.epsilon.inner, goal.epsilon.outer = 0.0, 0.0, open_w
            goal.speed, goal.force = float(self.p('grasp_speed')), float(self.p('grasp_force'))
        else:
            goal = self.gripper['Move'].Goal()
            goal.width, goal.speed = open_w, float(self.p('grasp_speed'))
        log = self.get_logger()

        def on_goal(future):
            handle = future.result()
            if not handle.accepted:
                log.error(f'gripper {label} goal rejected')
                return

            def on_result(result_future):
                result = result_future.result().result
                if result.success:
                    log.info(f'gripper {label} done')
                else:
                    log.error(f'gripper {label} failed: {result.error}')
            handle.get_result_async().add_done_callback(on_result)
        client.send_goal_async(goal).add_done_callback(on_goal)
        return f'gripper {label} sent'

    def _mark(self, which):
        if not self.recording:
            return f'{self._command_gripper(which == "grasp")} (not recording, nothing marked)'
        pose = self.latest_pose()
        stamp = _stamp_seconds(pose)
        if which == 'grasp':
            if self.grasp_stamp is not None:
                raise TeachAbort('mark_grasp: already marked; one grasp per demonstration '
                                 '(stop and record again to redo it)')
            self.grasp_stamp = stamp
        else:
            if self.grasp_stamp is None:
                raise TeachAbort('mark_release: mark the grasp first')
            if self.release_stamp is not None:
                raise TeachAbort('mark_release: already marked')
            self.release_stamp = stamp
        tcp = pose_to_matrix(pose.pose)[:3, 3]
        text = (f'{which.upper()} marked at t={time.monotonic() - self.record_started:.2f} s, '
                f'TCP {np.round(tcp, 3).tolist()}')
        try:
            return f'{text}; {self._command_gripper(which == "grasp")}'
        except TeachAbort as exc:
            return f'{text}; {exc}'

    def _srv_mark_grasp(self):
        return self._mark('grasp')

    def _srv_mark_release(self):
        return self._mark('release')

    def _srv_hold(self):
        aborted = self._abort_recording('hold requested') if self.recording else ''
        return self._hold() + aborted

    def _hold(self):
        if self.mode == 'passive':
            return 'passive mode: nothing to switch'
        name = self.p('hold_controller')
        self.cmc.activate(name)
        self.teaching_active = False
        return f'{name} ACTIVE: holding the pose the arm was in at the switch'

    def _abort_recording(self, reason):
        with self.lock:
            self.recording = False
            n = len(self.samples)
            self.samples = []
        return f' (recording of {n} samples DISCARDED: {reason})'

    def _srv_stop_recording(self):
        if not self.recording:
            raise TeachAbort('stop_recording: not recording')
        with self.lock:
            self.recording = False
            samples, self.samples = self.samples, []
            grasp, release = self.grasp_stamp, self.release_stamp
        held = self._hold() if self.teaching_active else 'arm not switched'
        try:
            return self._save_demo(samples, grasp, release) + f' | {held}'
        except TeachAbort as exc:
            raise TeachAbort(f'{exc} | {held}')

    # ------------------------------------------------------------------ conversion
    def _support_height(self):
        value = self.p('support_height')
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return float(value), 'parameter'
        if self.calib.cal.expected_table_height is not None:
            return float(self.calib.cal.expected_table_height), 'calibration expected_table_height'
        from tpgpt.hardware.planner import estimate_support_height
        return estimate_support_height(self.object_points), 'estimated from the object cloud'

    def _save_demo(self, samples, grasp, release):
        from tpgpt.hardware.teach import Recording, TeachError, bundle_from_recording
        p = self.p
        if len(samples) < 2:
            raise TeachAbort(f'stop_recording: only {len(samples)} pose samples were recorded')
        stamps = np.array([_stamp_seconds(m) for m in samples])
        if not np.any(stamps > 0):  # a publisher without header stamps: fall back to arrival order
            stamps = np.arange(len(samples)) * 1e-3
        T = np.stack([pose_to_matrix(m.pose) for m in samples])
        support, support_source = self._support_height()
        cal = self.calib.cal
        recording = Recording(
            stamps=stamps, positions=T[:, :3, 3], orientations=T[:, :3, :3],
            grasp_stamp=NAN if grasp is None else grasp,
            release_stamp=NAN if release is None else release,
            object_points=(np.zeros((0, 3)) if self.object_points is None else self.object_points),
            support_height=support,
            metadata={
                'object_name': p('object_name'), 'teach_mode': self.mode,
                'teaching_controller': p('teaching_controller') if self.mode == 'compliant' else None,
                'support_height_source': support_source,
                'calibration_yaml': p('calibration_yaml'),
                'calibration_is_placeholder': cal.camera_is_placeholder,
                'T_base_camera': cal.T_base_camera.tolist(),
                'raw_rate_hz': float((len(stamps) - 1) / max(np.ptp(stamps), 1e-9)),
                'end_effector': self.end_effector.to_dict(),
            },
        )
        out = os.path.join(p('output_dir'), f"{p('object_name')}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out, exist_ok=True)
        recording.save(os.path.join(out, 'raw.npz'))
        self.demo = {'dir': out, 'recording': recording, 'bundle': None, 'info': None,
                     'error': None}
        if grasp is None or release is None:
            self.demo['error'] = 'marks_missing: mark_grasp and mark_release during the recording'
        else:
            try:
                bundle, info = bundle_from_recording(recording, **self._conversion_options())
                self._write_bundle(bundle, info)
            except (TeachError, ValueError) as exc:
                self.demo['error'] = str(exc)
        self._publish_demo()
        if self.demo['error']:
            self._write_report({'error': self.demo['error']})
            raise TeachAbort(f"demonstration NOT converted: {self.demo['error']}. Raw recording "
                             f'kept at {out}/raw.npz (re-convert offline with python -m '
                             'tpgpt.hardware.teach --raw ... --out ...)')
        r = self.demo['bundle'].metadata['teach_report']
        return (f"saved {out}/source_bundle: {r['n_labels']} labels ({r['duration_s']:.1f} s), "
                f"max speed {r['max_speed_mps'] * 100:.0f} cm/s (execution lag "
                f"~{r['expected_execution_lag_at_max_speed_m'] * 1000:.0f} mm at that speed), "
                f"grasp {r['grasp_to_cloud_m'] * 1000:.0f} mm from the cloud, carry "
                f"{r['carry_rotation_deg']:.0f} deg (tilt {r['carry_tilt_deg']:.1f}), destination "
                f"{np.round(r['destination_m'][:2], 3).tolist()} at z {r['destination_height_m']:+.3f}")

    def _conversion_options(self):
        p = self.p
        return {'control_freq': int(p('control_freq')), 'stiffness': float(p('stiffness')),
                'damping': float(p('damping')), 'still_speed': float(p('still_speed')),
                'max_dwell': float(p('max_dwell')), 'smoothing_window': int(p('smoothing_window')),
                'stiffness_profile': str(p('stiffness_profile')),
                'along': float(p('stiffness_along')), 'across': float(p('stiffness_across'))}

    def _write_bundle(self, bundle, info):
        bundle.save(os.path.join(self.demo['dir'], 'source_bundle'))
        self.demo['bundle'], self.demo['info'] = bundle, info
        self._write_report(bundle.metadata['teach_report'])

    def _write_report(self, report):
        with open(os.path.join(self.demo['dir'], 'teach.json'), 'w') as f:
            json.dump({'report': report, 'conversion': self._conversion_options()}, f, indent=2,
                      default=str)

    def _srv_capture_placed(self):
        self._refuse_capture_while_floating('capture_placed')
        if self.demo is None or self.demo['bundle'] is None:
            raise TeachAbort('capture_placed: no converted demonstration yet')
        from tpgpt.hardware.teach import bundle_from_recording
        pts, stats, note = self._fuse_new_clouds()
        recording = self.demo['recording']
        recording.placed_points = pts
        recording.save(os.path.join(self.demo['dir'], 'raw.npz'))
        bundle, info = bundle_from_recording(recording, **self._conversion_options())
        self._write_bundle(bundle, info)
        self._publish_demo()
        predicted = info['place_points']
        return (f"placed object captured ({stats['n_fused']} points): centroid differs from the "
                f"taught carry by {info['placed_cloud_centroid_xy_error_m'] * 1000:.0f} mm in xy; "
                f"lowest z measured {info['placed_cloud_lowest_z_m']:+.3f} vs predicted "
                f'{np.percentile(predicted[:, 2], 1.0):+.3f} m' + note)

    # ------------------------------------------------------------------ safety
    def _watchdog(self):
        if not (self.teaching_active or self.recording):
            return
        try:
            pose = self.latest_pose()
        except TeachAbort as exc:
            self._emergency(f'stale pose: {exc}')
            return
        if not self.recording:
            return
        tcp = pose_to_matrix(pose.pose)[:3, 3]
        lo, hi = np.asarray(self.p('workspace_min')), np.asarray(self.p('workspace_max'))
        if np.any(tcp < lo) or np.any(tcp > hi):
            self._emergency(f'TCP {np.round(tcp, 3).tolist()} left the workspace box '
                            f'{lo.tolist()} .. {hi.tolist()}')
        elif time.monotonic() - self.record_started > self.p('max_record_seconds'):
            self._emergency(f"recording exceeded max_record_seconds ({self.p('max_record_seconds')})")

    def _emergency(self, reason):
        log = self.get_logger()
        text = self._abort_recording(reason) if self.recording else f' ({reason})'
        if self.teaching_active:
            try:
                text = self._hold() + text
            except TeachAbort as exc:
                self.teaching_active = False
                text = f'COULD NOT SWITCH TO THE HOLD CONTROLLER ({exc}); use the e-stop' + text
        log.error('ABORT' + text)

    # ------------------------------------------------------------------ rviz
    def _preview(self):
        if not self.recording:
            return
        samples = self.samples[::50]
        if len(samples) < 2:
            return
        stamp = self.get_clock().now().to_msg()
        pts = [[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in samples]
        markers = MarkerArray()
        markers.markers.append(sphere_list_marker('live_path', 0, pts, self.base, stamp, MAGENTA,
                                                  0.005))
        self.marker_pub.publish(markers)

    def _republish(self):
        if self.demo is not None and not self.recording:
            self._publish_demo()

    def _publish_demo(self):
        stamp = self.get_clock().now().to_msg()
        base = self.base
        demo = self.demo
        clear = Marker()
        clear.header.frame_id, clear.action = base, Marker.DELETEALL
        markers = MarkerArray()
        markers.markers.append(clear)
        rec = demo['recording']
        raw = rec.positions[::20]
        if len(raw) >= 2:
            markers.markers.append(line_strip_marker('raw_path', 0, raw, base, stamp,
                                                     color=(0.7, 0.7, 0.7), width=0.0015))
        if len(rec.object_points):
            self.object_pub.publish(xyz_to_pointcloud2(rec.object_points, base, stamp,
                                                       (255, 140, 0)))
        if rec.placed_points is not None:
            self.measured_pub.publish(xyz_to_pointcloud2(rec.placed_points, base, stamp,
                                                         (60, 220, 60)))
        bundle = demo['bundle']
        if bundle is None:
            markers.markers.append(text_marker(
                'summary', 0, rec.positions[len(rec.positions) // 2] + [0, 0, 0.12],
                f"NOT CONVERTED: {demo['error']}", base, stamp, color=(1.0, 0.3, 0.3)))
            self.marker_pub.publish(markers)
            return

        labels = bundle.tool_labels
        info = demo['info']
        r = bundle.metadata['teach_report']
        gi, ri = info['grasp_index'], info['release_index']
        colors = [RED if g > 0 else BLUE for g in labels.gripper]
        markers.markers.append(sphere_list_marker('demo_points', 0, labels.positions, base, stamp,
                                                  BLUE, 0.006, colors=colors))
        tcp_R = self.end_effector.ee_from_grasp(labels.orientations)
        for ns, i, length in (('tcp_at_start', 0, 0.05), ('tcp_at_grasp', gi, 0.08),
                              ('tcp_at_release', ri, 0.08)):
            markers.markers += axes_markers(ns, 0, labels.positions[i], tcp_R[i], base, stamp,
                                            length=length)
        markers.markers.append(sphere_list_marker('grasp_release', 0,
                                                  labels.positions[[gi, ri]], base, stamp,
                                                  (1.0, 0.9, 0.0), 0.016))
        place = bundle.placement
        markers.markers.append(cylinder_marker(
            'destination', 0, [place.destination[0], place.destination[1],
                               place.destination_height + 0.001],
            0.06, 0.002, base, stamp, (0.0, 0.9, 0.9, 0.6)))
        markers.markers.append(text_marker(
            'summary', 0, labels.positions[gi] + [0, 0, 0.12],
            f"{r['n_labels']} labels, {r['duration_s']:.1f} s | max {r['max_speed_mps'] * 100:.0f} "
            f"cm/s | grasp-cloud {r['grasp_to_cloud_m'] * 1000:.0f} mm", base, stamp))
        self.marker_pub.publish(markers)
        self.predicted_pub.publish(xyz_to_pointcloud2(info['place_points'], base, stamp,
                                                      (0, 220, 255)))


def main(args=None):
    # No rclpy signal handler: on Ctrl-C the context must stay usable so the arm
    # can still be switched to the hold controller before the node exits. SIGINT
    # is set explicitly (a background job inherits it ignored) and SIGTERM, which
    # ros2 launch escalates to, takes the same path instead of killing the node
    # with the arm floating.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TpgptTeachNode()
    background = []
    for child in (node.pose_node, node.io_node):
        executor = SingleThreadedExecutor()
        executor.add_node(child)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        background.append((executor, thread))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.teaching_active and rclpy.ok():
            node.get_logger().warn('shutting down with the arm floating: switching to hold')
            try:
                node.get_logger().info(node._hold())   # io_node is still spinning
            except Exception as exc:  # noqa: BLE001
                node.get_logger().error(f'could not switch to the hold controller: {exc}; '
                                        'use the e-stop')
        executor.shutdown()
        for ex, thread in background:
            ex.shutdown()
            thread.join(timeout=2.0)
        for n in (node.pose_node, node.io_node, node):
            n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
