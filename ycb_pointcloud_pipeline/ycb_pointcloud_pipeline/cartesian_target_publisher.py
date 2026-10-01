#!/usr/bin/env python3
"""
Activate CRISP's cartesian_impedance_controller on the FR3 and stream Cartesian
target poses that move the end effector (fr3_hand_tcp) to a fixed goal.

Sequence:
  1. Read the current end-effector pose from /current_pose (CRISP pose_broadcaster).
  2. Check the goal: inside a workspace box and within `max_distance` of the
     current position. An empty `goal_orientation` keeps the current orientation.
  3. Ask for confirmation, then strictly switch: activate
     cartesian_impedance_controller and deactivate whatever else owns the FR3
     effort interfaces (e.g. joint_impedance_controller). CRISP seeds its target
     with the measured pose on activation, so activation itself does not move the arm.
  4. Stream targets on /target_pose at `publish_rate`, interpolating position
     (smooth profile) and orientation (slerp) from start to goal; the duration
     follows from `max_linear_speed` / `max_angular_speed`.
  5. Keep streaming the goal until Ctrl-C (the controller then keeps holding it).

Aborts (and commands the arm to hold where it is) on a stale pose or a tracking
error above `max_tracking_error`.

Run with the robot container up (./scripts/run.sh robot) and ROS_DOMAIN_ID matching:
    ros2 run ycb_pointcloud_pipeline cartesian_target_publisher --ros-args \
        -p goal_position:="[0.45, 0.0, 0.40]"

Stiffness check (stiffness_mode isotropic | directional; tpgpt.hardware.stiffness_check):
activates tpgpt_cartesian_impedance_controller (variable stiffness) instead, sends a
stiffness every cycle -- isotropic, or 350 N/m along the motion / 600 across and while
still (the stiffness labels' rule, tpgpt.hardware.teach.directional_matrices, 12-value
message) -- and runs hold -> move to the goal -> hold -> move back -> hold, then turns the
hand `turn_deg` about the vertical and back, holding after each turn. It records
/current_pose to ~/tpgpt_stiffness_checks/<time>_<mode>_<k>_rot<k_rot>/ and reports each hold's jitter
(size, frequency) and resting tilt, each move's lag along and across the motion, and each
turn's lag and rotational time constant. Above `max_jitter` of
oscillation it drops to 400 / 30 and holds. Offset check (``two_sided:=true``, 2026-10-01): after the move out and back, the same move
the other way and back, so the start pose is held once arrived from each side; every hold
also records the robot's own estimate of the external force on the hand
(``o_f_ext_hat_k`` on ``robot_state_topic``, from its joint torque sensors, which sit after the
gearboxes). The report gives, per hold, the arm's offset from its command, the force the
spring pulls with (stiffness x offset) and the robot's force estimate. A force estimate as large
as the spring's says the robot's model of what it carries is off (or something real pulls);
one near zero says the force is lost in the joints (friction); an offset that changes with the
side the arm came from says friction too. It imports tpgpt, so run it from the tpgpt venv:
    ros2 run --prefix python3 ycb_pointcloud_pipeline cartesian_target_publisher --ros-args \
        -p stiffness_mode:=isotropic -p translational_stiffness:=600.0 \
        -p goal_offset:="[0.0, 0.15, 0.0]"
"""

import json
import math
import os
import signal
import threading
import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64MultiArray

JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
HARD_MAX_DISTANCE = 0.4  # m; goals farther than this are refused regardless of parameters


class AbortMotion(RuntimeError):
    pass


# ---------------------------------------------------------------- quaternion math
def _normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-9:
        raise AbortMotion('goal_orientation must be a non-zero quaternion [x, y, z, w]')
    return [v / n for v in q]


def _angle_between(q1, q2):
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def _quat_matrix(q):
    """Rotation matrix of a quaternion [x, y, z, w]."""
    import numpy as np

    x, y, z, w = _normalize(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q1, q2, s):
    d = sum(a * b for a, b in zip(q1, q2))
    if d < 0.0:  # take the short way round
        q2, d = [-v for v in q2], -d
    if d > 0.9995:
        return _normalize([a + s * (b - a) for a, b in zip(q1, q2)])
    theta = math.acos(d)
    w1 = math.sin((1.0 - s) * theta) / math.sin(theta)
    w2 = math.sin(s * theta) / math.sin(theta)
    return [w1 * a + w2 * b for a, b in zip(q1, q2)]


class CartesianTargetPublisher(Node):
    def __init__(self):
        super().__init__('cartesian_target_publisher')
        self.declare_parameter('controller_manager', '/controller_manager')
        self.declare_parameter('controller_name', 'cartesian_impedance_controller')
        self.declare_parameter('target_topic', '/target_pose')
        self.declare_parameter('current_pose_topic', '/current_pose')
        self.declare_parameter('base_frame', 'base')
        # Fixed goal for fr3_hand_tcp in base_frame
        self.declare_parameter('goal_position', [0.45, 0.0, 0.40])       # m
        self.declare_parameter('goal_orientation', [0.0])  # [x, y, z, w]; [0.0] = keep current
        self.declare_parameter('workspace_min', [0.20, -0.50, 0.10])     # m
        self.declare_parameter('workspace_max', [0.75, 0.50, 0.80])      # m
        self.declare_parameter('max_distance', 0.30)       # m from the current position
        self.declare_parameter('max_linear_speed', 0.05)   # m/s
        self.declare_parameter('max_angular_speed', 0.25)  # rad/s
        self.declare_parameter('min_duration', 3.0)        # s
        self.declare_parameter('publish_rate', 50.0)       # Hz
        self.declare_parameter('max_tracking_error', 0.12)  # m, abort above this
        self.declare_parameter('require_confirmation', True)
        # --- stiffness check (step 3 of the execution-gap plan; tpgpt.hardware.stiffness_check)
        # 'stock': the behaviour above, unchanged. 'isotropic' / 'directional': load and
        # activate the variable-stiffness controller, send a stiffness every cycle, and run
        # hold -> move to the goal -> hold -> move back -> hold, recording /current_pose and
        # reporting each hold's jitter and each move's lag along and across the motion.
        self.declare_parameter('stiffness_mode', 'stock')   # stock | isotropic | directional
        self.declare_parameter('translational_stiffness', 600.0)  # N/m; across when directional
        self.declare_parameter('stiffness_along', 350.0)    # N/m along the motion (directional)
        self.declare_parameter('rotational_stiffness', 30.0)  # N m/rad
        self.declare_parameter('stiffness_controller_name', 'tpgpt_cartesian_impedance_controller')
        self.declare_parameter('stiffness_controller_type', 'crisp_controllers/CartesianController')
        self.declare_parameter('stiffness_controller_params_file', '')  # path INSIDE the container
        self.declare_parameter('goal_offset', [0.0])        # [dx, dy, dz] m from the start; [0.0] = goal_position
        self.declare_parameter('hold_seconds', 5.0)         # each hold of the sequence
        self.declare_parameter('return_to_start', True)
        # The turn: the hand turns this far about the vertical and back, at up to turn_rate
        # (deg/s; the 18:39 demonstration turns at most 5.6 deg/s at speed_scale 0.35), to
        # measure the rotational spring's resting tilt and turning lag. 0 = no turn.
        self.declare_parameter('turn_deg', 15.0)
        self.declare_parameter('turn_rate', 6.0)
        # Safety cut: when the strongest oscillation at 3 Hz or faster (peak to peak over the
        # last second, from its spectrum) stays above this (m) for 1.5 s, the stiffness drops
        # to 400 / 30 and the arm holds where it is (tpgpt.hardware.stiffness_check).
        self.declare_parameter('max_jitter', 0.004)
        self.declare_parameter('record_dir', '')            # '' -> ~/tpgpt_stiffness_checks/<time>_<mode>
        # the offset check: the start reached from both sides, and the robot's force estimate
        self.declare_parameter('two_sided', False)
        self.declare_parameter('robot_state_topic', '/franka_robot_state_broadcaster/robot_state')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cm = p('controller_manager').rstrip('/')
        self.mode = str(p('stiffness_mode'))
        if self.mode not in ('stock', 'isotropic', 'directional'):
            raise SystemExit(f"stiffness_mode must be stock, isotropic or directional, not {self.mode!r}")
        self.controller = p('controller_name')
        self.target_topic = p('target_topic')
        self.stiffness_pub = None
        if self.mode != 'stock':
            # the variable-stiffness controller listens on private topics
            self.controller = p('stiffness_controller_name')
            self.target_topic = f'/{self.controller}/target_pose'
            self.stiffness_topic = f'/{self.controller}/target_stiffness'
            self.stiffness_pub = self.create_publisher(Float64MultiArray, self.stiffness_topic, 1)
        self.frame = p('base_frame')

        self._lock = threading.Lock()
        self._pose = None  # (monotonic time, position, orientation [x, y, z, w])
        self._samples = None  # while recording: [(stamp, pos, quat, cmd_pos, cmd_speed, segment)]
        # last commanded position, its speed, the segment, the commanded orientation, its rate
        self._cmd = ([0.0] * 3, 0.0, 'none', [0.0, 0.0, 0.0, 1.0], 0.0)
        # The 250 Hz pose stream on its own node, spun alone on its own thread (see main): on a
        # shared executor this node received 96 of ~240 poses a second, with gaps up to 0.5 s,
        # while the check ran -- the teach node's measurement and fix (tpgpt_teach_node).
        self.pose_node = rclpy.create_node(f'{self.get_name()}_pose')
        self.pose_node.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 100)
        self._wrench = None   # while recording: [(monotonic time, force xyz, torque xyz)]
        if p('stiffness_mode') != 'stock':
            try:
                from franka_msgs.msg import FrankaRobotState
                self.pose_node.create_subscription(FrankaRobotState, p('robot_state_topic'),
                                                   self._state_cb, 10)
            except ImportError:            # franka_msgs not installed: no force estimate
                pass
        self.target_pub = self.create_publisher(PoseStamped, self.target_topic, 1)
        self.cli_list = self.create_client(ListControllers, f'{self.cm}/list_controllers')
        self.cli_switch = self.create_client(SwitchController, f'{self.cm}/switch_controller')
        self.moving = False

    # ------------------------------------------------------------------ helpers
    def _pose_cb(self, msg):
        if msg.header.frame_id not in ('', self.frame):
            return
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        o = msg.pose.orientation
        with self._lock:
            self._pose = (time.monotonic(), pos, [o.x, o.y, o.z, o.w])
            if self._samples is not None:
                stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
                cmd_pos, cmd_speed, segment, cmd_quat, cmd_rate = self._cmd
                self._samples.append((stamp, pos, [o.x, o.y, o.z, o.w], list(cmd_pos),
                                      cmd_speed, segment, list(cmd_quat), cmd_rate))

    def _state_cb(self, msg):
        w = msg.o_f_ext_hat_k.wrench
        with self._lock:
            if self._wrench is not None:
                self._wrench.append((time.monotonic(), [w.force.x, w.force.y, w.force.z],
                                     [w.torque.x, w.torque.y, w.torque.z]))

    def measured(self, max_age=0.5):
        with self._lock:
            pose = self._pose
        if pose is None or time.monotonic() - pose[0] > max_age:
            raise AbortMotion(f'No fresh end-effector pose on '
                              f'{self.get_parameter("current_pose_topic").value} '
                              '(is pose_broadcaster active?)')
        return pose[1], pose[2]

    def call(self, client, request, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise AbortMotion(f'Service {client.srv_name} not available -- is the robot '
                              'container running with the same ROS_DOMAIN_ID?')
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise AbortMotion(f'Service {client.srv_name} timed out')
            time.sleep(0.01)
        return future.result()

    def list_controllers(self):
        return {c.name: c for c in self.call(self.cli_list, ListControllers.Request()).controller}

    def publish_target(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in pos)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = (float(v) for v in quat)
        self.target_pub.publish(msg)

    def hold_current_pose(self):
        try:
            pos, quat = self.measured(max_age=1.0)
            if self.stiffness_pub is not None:
                # back to the stock gains: softer, and known not to vibrate (400 / 30)
                self.stiffness_pub.publish(Float64MultiArray(data=[400.0] * 3 + [30.0] * 3))
            self.publish_target(pos, quat)
            self.get_logger().warn('Commanded the controller to hold the current pose.')
        except AbortMotion:
            self.get_logger().error('Could not read the current pose to hold it; the '
                                    'controller keeps its last target.')

    # ------------------------------------------------------------------ steps
    def build_goal(self, start_pos, start_quat):
        goal_pos = [float(v) for v in self.get_parameter('goal_position').value]
        offset = [float(v) for v in self.get_parameter('goal_offset').value]
        if len(offset) == 3:
            goal_pos = [a + b for a, b in zip(start_pos, offset)]
        elif offset != [0.0]:
            raise AbortMotion('goal_offset must be [dx, dy, dz] or [0.0] (use goal_position)')
        if len(goal_pos) != 3:
            raise AbortMotion('goal_position must be [x, y, z]')
        lo = self.get_parameter('workspace_min').value
        hi = self.get_parameter('workspace_max').value
        points = [('goal', goal_pos)]
        if self.get_parameter('two_sided').value:     # the other side of the start, mirrored
            points.append(('other side', [2.0 * a - b for a, b in zip(start_pos, goal_pos)]))
        for what, point in points:
            for axis, v, a, b in zip('xyz', point, lo, hi):
                if not a <= v <= b:
                    raise AbortMotion(f'{what} {axis}={v:.3f} m is outside the workspace box '
                                      f'[{a}, {b}]')
        dist = math.dist(goal_pos, start_pos)
        max_dist = min(self.get_parameter('max_distance').value, HARD_MAX_DISTANCE)
        if dist > max_dist:
            raise AbortMotion(f'goal is {dist:.3f} m from the current position; max_distance '
                              f'is {max_dist} m')

        goal_quat = list(self.get_parameter('goal_orientation').value)
        if len(goal_quat) == 4:
            goal_quat = _normalize(goal_quat)
        elif goal_quat == [0.0]:
            goal_quat = list(start_quat)
        else:
            raise AbortMotion('goal_orientation must be [x, y, z, w] or [0.0] (keep current)')
        return goal_pos, goal_quat

    def activate(self):
        controllers = self.list_controllers()
        if self.controller not in controllers:
            raise AbortMotion(f'{self.controller} is not loaded in {self.cm} (it is spawned '
                              'inactive by fr3_crisp.launch.py)')
        if controllers[self.controller].state == 'active':
            self.get_logger().info(f'{self.controller} is already active')
            return
        if controllers[self.controller].state != 'inactive':
            raise AbortMotion(f'{self.controller} is {controllers[self.controller].state}, '
                              'expected inactive')
        effort_interfaces = {f'{j}/effort' for j in JOINTS}
        to_deactivate = [c.name for c in controllers.values()
                         if c.name != self.controller and c.state == 'active'
                         and effort_interfaces & set(c.claimed_interfaces)]
        self.get_logger().info(f'Switching: activate [{self.controller}], '
                               f'deactivate {to_deactivate} (strict)')
        req = SwitchController.Request()
        req.activate_controllers = [self.controller]
        req.deactivate_controllers = to_deactivate
        req.strictness = SwitchController.Request.STRICT
        req.timeout.sec = 5
        if not self.call(self.cli_switch, req).ok:
            raise AbortMotion('switch_controller failed')
        if self.list_controllers()[self.controller].state != 'active':
            raise AbortMotion(f'{self.controller} did not become active')

    def stream_to_goal(self, start_pos, start_quat, goal_pos, goal_quat):
        dist = math.dist(goal_pos, start_pos)
        angle = _angle_between(start_quat, goal_quat)
        duration = max(self.get_parameter('min_duration').value,
                       1.5 * dist / self.get_parameter('max_linear_speed').value,
                       1.5 * angle / self.get_parameter('max_angular_speed').value)
        period = 1.0 / self.get_parameter('publish_rate').value
        max_err = self.get_parameter('max_tracking_error').value
        self.get_logger().info(
            f'Moving {dist:.3f} m / {math.degrees(angle):.1f} deg over {duration:.1f} s: '
            f'[{", ".join(f"{v:.3f}" for v in start_pos)}] -> '
            f'[{", ".join(f"{v:.3f}" for v in goal_pos)}]')

        start = time.monotonic()
        reached_logged = False
        while rclpy.ok():
            t = time.monotonic() - start
            s = min(t / duration, 1.0)
            s = 0.5 - 0.5 * math.cos(math.pi * s)  # smooth start/stop (peak speed 1.57x avg)
            pos_cmd = [a + s * (b - a) for a, b in zip(start_pos, goal_pos)]
            quat_cmd = _slerp(start_quat, goal_quat, s)
            self.publish_target(pos_cmd, quat_cmd)

            pos_meas, _ = self.measured()
            err = math.dist(pos_cmd, pos_meas)
            if err > max_err:
                raise AbortMotion(f'Tracking error {err:.3f} m exceeds {max_err} m')
            if t > duration + 2.0 and not reached_logged:
                self.moving = False
                _, quat_meas = self.measured()
                self.get_logger().info(
                    f'Goal reached: position error {math.dist(goal_pos, pos_meas) * 1000:.1f} mm, '
                    f'orientation error {math.degrees(_angle_between(goal_quat, quat_meas)):.1f} '
                    'deg. Streaming the goal; Ctrl-C to stop (the controller keeps holding it).')
                reached_logged = True
            time.sleep(period)

    # ------------------------------------------------------------ stiffness check
    def stiffness_for(self, direction, speed):
        """Translational K (base frame) for this cycle: isotropic, or the directional rule."""
        import numpy as np
        from tpgpt.hardware.teach import directional_matrices

        k = float(self.get_parameter('translational_stiffness').value)
        if self.mode == 'isotropic':
            return k * np.eye(3)
        return directional_matrices([direction], [speed],
                                    float(self.get_parameter('stiffness_along').value), k)[0]

    def stiffness_limits(self):
        """``(translational, rotational)``: the variable-stiffness controller's own caps
        (``variable_max_stiffness`` in the installed tpgpt_variable_stiffness_controller.yaml).
        CRISP clamps anything above them, and so does ``crisp_stiffness``."""
        import yaml
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'), 'config',
                            'tpgpt_variable_stiffness_controller.yaml')
        with open(path) as f:
            cfg = yaml.safe_load(f)
        caps = cfg[self.controller]['ros__parameters']['variable_max_stiffness']
        return float(caps['translational']), float(caps['rotational'])

    def publish_stiffness(self, K, quat):
        """K rotated into the measured hand frame; 12 values when directional, else 6."""
        from tpgpt.hardware.executor import crisp_stiffness

        max_t, max_r = self._limits
        values = crisp_stiffness(K, _quat_matrix(quat),
                                 float(self.get_parameter('rotational_stiffness').value),
                                 max_translational=max_t, max_rotational=max_r,
                                 full=self.mode == 'directional')
        self.stiffness_pub.publish(Float64MultiArray(data=[float(v) for v in values]))

    def _segment(self, name, start_pos, end_pos, quat, duration, end_quat=None):
        """Stream one segment: a hold (start == end), a smooth move, or a smooth turn from
        ``quat`` to ``end_quat``; stiffness every cycle."""
        from tpgpt.hardware.stiffness_check import OscillationWatch, recent_oscillation

        period = 1.0 / self.get_parameter('publish_rate').value
        max_err = self.get_parameter('max_tracking_error').value
        max_jitter = float(self.get_parameter('max_jitter').value)
        if getattr(self, '_watch', None) is None:
            self._watch = OscillationWatch(max_jitter, persist=1.5)   # spans segments
        dist = math.dist(start_pos, end_pos)
        direction = ([(b - a) / dist for a, b in zip(start_pos, end_pos)] if dist > 1e-9
                     else [0.0, 0.0, 0.0])
        turn = 0.0 if end_quat is None else _angle_between(quat, end_quat)
        t0 = time.monotonic()
        n = 0
        while rclpy.ok():
            t = time.monotonic() - t0
            if t > duration:
                return
            s = 0.5 - 0.5 * math.cos(math.pi * t / duration)
            profile = math.pi / (2 * duration) * math.sin(math.pi * t / duration)
            speed = dist * profile if dist > 1e-9 else 0.0
            rate = turn * profile
            pos_cmd = [a + s * (b - a) for a, b in zip(start_pos, end_pos)] if dist > 1e-9 \
                else list(start_pos)
            quat_cmd = _slerp(quat, end_quat, s) if turn > 1e-9 else list(quat)
            _, quat_meas = self.measured()
            with self._lock:
                self._cmd = (pos_cmd, speed, name, quat_cmd, rate)
            self.publish_stiffness(self.stiffness_for(direction, speed), quat_meas)
            self.publish_target(pos_cmd, quat_cmd)
            pos_meas, _ = self.measured()
            if math.dist(pos_cmd, pos_meas) > max_err:
                raise AbortMotion(f'{name}: tracking error {math.dist(pos_cmd, pos_meas):.3f} m '
                                  f'exceeds {max_err} m')
            n += 1
            # Vibration cut: the strongest oscillation at 3 Hz or faster over the last second
            # (from its spectrum), above max_jitter for 1.5 s on end (OscillationWatch). The
            # path's own range misfires on a lag building up as friction lets go, and a single
            # reading on the arm catching up at the end of a move; both seen on a mock arm.
            if n % 5 == 0 and time.monotonic() - self._check_t0 > 1.0:
                with self._lock:
                    recent = [(s_[0], s_[1]) for s_ in self._samples[-400:]]
                if recent:
                    stamps, positions = zip(*recent)
                    size, hz = recent_oscillation(stamps, positions, 1.0)
                    if self._watch.update(time.monotonic(), size):
                        raise AbortMotion(f'{name}: the arm oscillates {size * 1000:.1f} mm peak to '
                                          f'peak at {hz:.0f} Hz, above max_jitter '
                                          f'{max_jitter * 1000:.1f} mm for 1.5 s; back to 400 / 30 '
                                          'and holding')
            time.sleep(period)

    def turned(self, quat):
        """``quat`` turned ``turn_deg`` about the base's vertical."""
        from scipy.spatial.transform import Rotation

        turn = Rotation.from_euler('z', float(self.get_parameter('turn_deg').value), degrees=True)
        return list((turn * Rotation.from_quat(quat)).as_quat())

    def check_plan(self, start_pos, start_quat, goal_pos):
        """The sequence: ``(name, kind, from, to, seconds, from quat, to quat)``."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        hold = float(g('hold_seconds'))
        dist = math.dist(goal_pos, start_pos)
        move = max(g('min_duration'), 1.5 * dist / g('max_linear_speed'))
        q = start_quat
        plan = [('hold_start', 'hold', start_pos, start_pos, hold, q, q),
                ('move_out', 'move', start_pos, goal_pos, move, q, q),
                ('hold_goal', 'hold', goal_pos, goal_pos, hold, q, q)]
        end = goal_pos
        if g('return_to_start') or g('two_sided'):
            plan += [('move_back', 'move', goal_pos, start_pos, move, q, q),
                     ('hold_end', 'hold', start_pos, start_pos, hold, q, q)]
            end = start_pos
        if g('two_sided'):
            # the same move the other way, and back: the start reached from the other side
            other = [2.0 * a - b for a, b in zip(start_pos, goal_pos)]
            plan += [('move_out_other', 'move', start_pos, other, move, q, q),
                     ('hold_goal_other', 'hold', other, other, hold, q, q),
                     ('move_back_other', 'move', other, start_pos, move, q, q),
                     ('hold_end_other', 'hold', start_pos, start_pos, hold, q, q)]
        turn_deg = float(g('turn_deg'))
        if turn_deg > 0.0:
            # a smooth turn peaks at pi/2 times its mean rate
            turn_time = max(g('min_duration'), math.pi * turn_deg / (2.0 * float(g('turn_rate'))))
            qt = self.turned(q)
            plan += [('turn_out', 'turn', end, end, turn_time, q, qt),
                     ('hold_turned', 'hold', end, end, hold, qt, qt),
                     ('turn_back', 'turn', end, end, turn_time, qt, q),
                     ('hold_final', 'hold', end, end, hold, q, q)]
        return plan

    def run_stiffness_check(self, start_pos, start_quat, goal_pos):
        """The sequence (:meth:`check_plan`), then the report."""
        import numpy as np
        from tpgpt.hardware.stiffness_check import hold_jitter, hold_tilt, move_lag, turn_lag

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        plan = self.check_plan(start_pos, start_quat, goal_pos)
        with self._lock:
            self._samples = []
            self._wrench = []
        self._check_t0 = time.monotonic()
        wall0 = time.monotonic()
        windows = {}                     # hold name -> (monotonic start, end)
        aborted = None
        try:
            for name, kind, a, b, duration, qa, qb in plan:
                self.get_logger().info(f'{name}: {duration:.1f} s')
                t_seg = time.monotonic()
                self._segment(name, a, b, qa, duration, qb if kind == 'turn' else None)
                if kind == 'hold':
                    windows[name] = (t_seg + 0.5 * duration, time.monotonic())   # settled half
        except (AbortMotion, KeyboardInterrupt) as exc:
            aborted = exc              # keep what was recorded: a cut run is the one to look at
        with self._lock:
            samples, self._samples = self._samples, None
            wrench, self._wrench = self._wrench, None
        if not samples:
            if aborted is not None:
                raise aborted
            raise AbortMotion('no pose was recorded during the check')

        stamps = np.array([s[0] for s in samples])
        stamps = stamps - stamps[0]
        meas = np.array([s[1] for s in samples])
        cmd = np.array([s[3] for s in samples])
        speed = np.array([s[4] for s in samples])
        seg = np.array([s[5] for s in samples])
        quat = np.array([s[2] for s in samples])
        cmd_quat = np.array([s[6] for s in samples])
        rate = np.array([s[7] for s in samples])
        report = {'mode': self.mode, 'translational_stiffness': g('translational_stiffness'),
                  'stiffness_along': g('stiffness_along') if self.mode == 'directional' else None,
                  'rotational_stiffness': g('rotational_stiffness'),
                  'rotational_stiffness_applied': min(float(g('rotational_stiffness')),
                                                      self._limits[1]),
                  'aborted': None if aborted is None else (str(aborted) or 'Ctrl-C'),
                  'segments': {}}
        report['turn_deg'] = float(g('turn_deg'))
        for name, kind, a, b, _, _, _ in plan:
            m = seg == name
            if not m.any():
                continue
            if kind == 'hold':
                report['segments'][name] = {**hold_jitter(stamps[m], meas[m]),
                                            **hold_tilt(stamps[m], cmd_quat[m], quat[m])}
                # the settled half of the hold: offset, the spring's pull, the robot's estimate
                half = m & (stamps >= stamps[m][0] + 0.5 * (stamps[m][-1] - stamps[m][0]))
                offset = np.median(cmd[half] - meas[half], axis=0)
                k = float(g('translational_stiffness'))
                entry = {'offset_mm': (offset * 1000).round(2).tolist(),
                         'offset_norm_mm': round(float(np.linalg.norm(offset)) * 1000, 2),
                         'spring_force_N': (k * offset).round(2).tolist()}
                w0, w1 = windows.get(name, (None, None))
                forces = [f for t_, f, _ in (wrench or []) if w0 is not None and w0 <= t_ <= w1]
                if forces:
                    fe = np.median(np.array(forces), axis=0)
                    entry.update({'robot_external_force_N': fe.round(2).tolist(),
                                  'robot_external_force_norm_N': round(float(np.linalg.norm(fe)), 2),
                                  'robot_state_messages': len(forces)})
                report['segments'][name].update(entry)
            elif kind == 'move':
                u = [bb - aa for aa, bb in zip(a, b)]
                report['segments'][name] = move_lag(cmd[m], meas[m], u, speed[m])
            else:
                sign = 1.0 if name == 'turn_out' else -1.0     # the turn back is about -z
                report['segments'][name] = turn_lag(cmd_quat[m], quat[m], [0.0, 0.0, sign], rate[m])
        out = g('record_dir') or os.path.expanduser(
            time.strftime(f'~/tpgpt_stiffness_checks/%Y%m%d_%H%M%S_{self.mode}_'
                          f"{int(g('translational_stiffness'))}_rot{int(g('rotational_stiffness'))}"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(out, 'samples.csv'), 'w') as f:
            f.write('t,segment,meas_x,meas_y,meas_z,qx,qy,qz,qw,cmd_x,cmd_y,cmd_z,cmd_speed,'
                    'cmd_qx,cmd_qy,cmd_qz,cmd_qw,cmd_rate\n')
            for t, s in zip(stamps, samples):
                f.write(f'{t:.4f},{s[5]},' + ','.join(f'{v:.6f}' for v in (*s[1], *s[2], *s[3]))
                        + f',{s[4]:.5f},' + ','.join(f'{v:.6f}' for v in s[6])
                        + f',{s[7]:.5f}\n')
        if wrench:
            with open(os.path.join(out, 'wrench.csv'), 'w') as f:
                f.write('t,fx,fy,fz,tx,ty,tz\n')
                for t_, fo, to in wrench:
                    f.write(f'{t_ - wall0:.4f},' + ','.join(f'{v:.4f}' for v in (*fo, *to)) + '\n')
        lines = [f'Stiffness check ({self.mode}, {g("translational_stiffness"):.0f} N/m'
                 + (f' across / {g("stiffness_along"):.0f} along' if self.mode == 'directional' else '')
                 + f', rotational {g("rotational_stiffness"):.0f}) -> {out}']
        for name, r in report['segments'].items():
            if 'p2p_mm' in r:
                lines.append(f'  {name:11s} jitter p95 {r["p95_mm"]:.2f} mm, peak-to-peak '
                             f'{r["p2p_mm"]:.2f} mm, strongest {r.get("peak_mm", 0):.2f} mm at '
                             f'{r.get("peak_hz", 0):.1f} Hz ({r["rate_hz"]:.0f} Hz poses); '
                             f'hand tilted {r.get("tilt_median_deg", float("nan")):.2f} deg off '
                             'the command')
                if 'offset_mm' in r:
                    est = (f'robot estimates {r["robot_external_force_norm_N"]:.1f} N '
                           f'{r["robot_external_force_N"]}' if 'robot_external_force_N' in r
                           else 'no robot_state received')
                    lines.append(f'  {"":11s} offset {r["offset_norm_mm"]:.1f} mm {r["offset_mm"]}, '
                                 f'spring pulling {np.linalg.norm(r["spring_force_N"]):.1f} N; {est}')
            elif 'tau_s' in r:
                lines.append(f'  {name:11s} turn lag {r["lag_median_deg"]:.2f} deg '
                             f'(max {r["lag_max_deg"]:.2f}) at up to {r["peak_rate_deg_s"]:.1f} '
                             f'deg/s: time constant {r["tau_s"]:.2f} s, resting part '
                             f'{r["offset_deg"]:.2f} deg; off the turn axis up to '
                             f'{r["off_axis_max_deg"]:.2f} deg')
            elif 'along_median_mm' in r:
                lines.append(f'  {name:10s} lag along the motion {r["along_median_mm"]:.1f} mm '
                             f'(max {r["along_max_mm"]:.1f}), across {r["across_median_mm"]:.1f} mm '
                             f'(max {r["across_max_mm"]:.1f})')
        self.get_logger().info('\n'.join(lines))
        if aborted is not None:
            raise aborted              # main() lowers the stiffness to 400 / 30 and holds
        return plan[-1][3]

    def run_check(self):
        """The stiffness check: variable-stiffness controller, the sequence, the report, hold."""
        from ycb_pointcloud_pipeline.ros_utils import ControllerManagerClient, container_config_path

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        # A stiffness above the controller's caps would be clamped to them without a word, and
        # the report would name a stiffness the arm never had (2026-09-30: runs asked for 80
        # and 100 N m/rad and ran at 60). Refuse instead.
        self._limits = self.stiffness_limits()
        asked = {'translational': max(float(g('translational_stiffness')),
                                      float(g('stiffness_along')) if self.mode == 'directional'
                                      else 0.0),
                 'rotational': float(g('rotational_stiffness'))}
        for (name, value), cap in zip(asked.items(), self._limits):
            if value > cap:
                raise AbortMotion(f'{name} stiffness {value:g} is above the controller\'s cap '
                                  f'{cap:g} (variable_max_stiffness.{name} in '
                                  'config/tpgpt_variable_stiffness_controller.yaml): it would run '
                                  f'at {cap:g}. Raising the cap is a safety decision; nothing was '
                                  'activated.')
        self.get_logger().info('Waiting for the end-effector pose ...')
        deadline = time.monotonic() + 10.0
        while self._pose is None:
            if time.monotonic() > deadline:
                raise AbortMotion(f"No pose received within 10 s on {g('current_pose_topic')}")
            time.sleep(0.05)
        start_pos, start_quat = self.measured()
        goal_pos, _ = self.build_goal(start_pos, start_quat)   # the hand keeps its orientation
        for topic in (self.target_topic, self.stiffness_topic):
            others = sum(1 for info in self.get_publishers_info_by_topic(topic)
                         if info.node_name != self.get_name())
            if others:
                raise AbortMotion(f'{others} other publisher(s) on {topic}; CRISP ignores '
                                  'commands when more than one publisher exists')
        cmc = ControllerManagerClient(self, self.cm, error=AbortMotion)
        try:
            params_file = (g('stiffness_controller_params_file')
                           or container_config_path('tpgpt_variable_stiffness_controller.yaml'))
        except ValueError as exc:
            raise AbortMotion(f'{exc} (stiffness_controller_params_file)')
        cmc.ensure_loaded(self.controller, g('stiffness_controller_type'), params_file)
        if self.mode == 'directional':
            cmc.require_full_stiffness(self.controller)

        k = float(g('translational_stiffness'))
        stiffness = (f'{k:.0f} N/m in every direction' if self.mode == 'isotropic' else
                     f"{g('stiffness_along'):.0f} N/m along the motion, {k:.0f} across it and "
                     'while still')
        dist = math.dist(goal_pos, start_pos)
        if g('require_confirmation'):
            print(f'\nAbout to ACTIVATE {self.controller} (effort control) with stiffness '
                  f"{stiffness}, rotational {g('rotational_stiffness'):.0f} N m/rad, and run:\n"
                  f"  hold {g('hold_seconds'):.0f} s, move fr3_hand_tcp {dist:.3f} m to {goal_pos} "
                  f"({self.frame}), hold{', move back, hold' if g('return_to_start') else ''}"
                  + (f", turn the hand {g('turn_deg'):.0f} deg about the vertical and back, holding "
                     'after each' if g('turn_deg') > 0 else '') + '.\n'
                  + f"  Cut: above {g('max_jitter') * 1000:.1f} mm of oscillation the stiffness drops "
                  'to 400 / 30 and the arm holds.\nKeep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return
        self.activate()
        # Re-read after activation: CRISP seeded its target with this measured pose
        start_pos, start_quat = self.measured()
        self.moving = True
        final = self.run_stiffness_check(start_pos, start_quat, goal_pos)
        self.get_logger().info('Holding the last target with the same stiffness; Ctrl-C to stop '
                               '(the controller keeps holding it).')
        period = 1.0 / g('publish_rate')
        while rclpy.ok():
            _, quat_meas = self.measured()
            self.publish_stiffness(self.stiffness_for([0.0, 0.0, 0.0], 0.0), quat_meas)
            self.publish_target(final, start_quat)
            time.sleep(period)

    def run(self):
        if self.mode != 'stock':
            return self.run_check()
        self.get_logger().info('Waiting for the end-effector pose ...')
        deadline = time.monotonic() + 10.0
        while self._pose is None:
            if time.monotonic() > deadline:
                raise AbortMotion('No pose received within 10 s on '
                                  f'{self.get_parameter("current_pose_topic").value}')
            time.sleep(0.05)
        start_pos, start_quat = self.measured()
        goal_pos, goal_quat = self.build_goal(start_pos, start_quat)

        others = sum(1 for info in self.get_publishers_info_by_topic(self.target_topic)
                     if info.node_name != self.get_name())
        if others:
            raise AbortMotion(f'{others} other publisher(s) on {self.target_topic}; CRISP '
                              'ignores targets when more than one publisher exists')

        if self.get_parameter('require_confirmation').value:
            print(f'\nAbout to ACTIVATE {self.controller} (effort control) and move '
                  f'fr3_hand_tcp {math.dist(goal_pos, start_pos):.3f} m and '
                  f'{math.degrees(_angle_between(start_quat, goal_quat)):.1f} deg to\n'
                  f'  position {goal_pos} ({self.frame})\n  orientation {goal_quat} [x,y,z,w]'
                  '\nKeep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return

        self.activate()
        # Re-read after activation: CRISP seeded its target with this measured pose
        start_pos, start_quat = self.measured()
        self.moving = True
        self.stream_to_goal(start_pos, start_quat, goal_pos, goal_quat)


def main(args=None):
    # No rclpy signal handler: with it, Ctrl-C stopped the executor (so the pose went stale)
    # before KeyboardInterrupt reached this thread, and the final hold never went out. As
    # free_drive does, Ctrl-C and SIGTERM raise KeyboardInterrupt here with callbacks alive.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = CartesianTargetPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spinners = [threading.Thread(target=executor.spin, daemon=True)]
    pose_executor = SingleThreadedExecutor()
    pose_executor.add_node(node.pose_node)
    spinners.append(threading.Thread(target=pose_executor.spin, daemon=True))
    for t in spinners:
        t.start()
    exit_code = 0
    try:
        node.run()
    except AbortMotion as e:
        node.get_logger().error(f'Aborted: {e}')
        if node.moving:
            node.hold_current_pose()
        exit_code = 1
    except KeyboardInterrupt:
        if node.moving:
            node.hold_current_pose()
    finally:
        time.sleep(0.2)  # let a final hold command go out
        executor.shutdown()
        pose_executor.shutdown()
        # Destroying a node its executor thread is still inside aborts the process
        # ("terminate called without an active exception", seen on an early refusal).
        for t in spinners:
            t.join(timeout=2.0)
        node.pose_node.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
