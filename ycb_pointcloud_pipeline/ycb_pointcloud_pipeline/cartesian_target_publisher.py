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
"""

import math
import threading
import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

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

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cm = p('controller_manager').rstrip('/')
        self.controller = p('controller_name')
        self.target_topic = p('target_topic')
        self.frame = p('base_frame')

        self._lock = threading.Lock()
        self._pose = None  # (monotonic time, position, orientation [x, y, z, w])
        self.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 10)
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
            self.publish_target(pos, quat)
            self.get_logger().warn('Commanded the controller to hold the current pose.')
        except AbortMotion:
            self.get_logger().error('Could not read the current pose to hold it; the '
                                    'controller keeps its last target.')

    # ------------------------------------------------------------------ steps
    def build_goal(self, start_pos, start_quat):
        goal_pos = [float(v) for v in self.get_parameter('goal_position').value]
        if len(goal_pos) != 3:
            raise AbortMotion('goal_position must be [x, y, z]')
        lo = self.get_parameter('workspace_min').value
        hi = self.get_parameter('workspace_max').value
        for axis, v, a, b in zip('xyz', goal_pos, lo, hi):
            if not a <= v <= b:
                raise AbortMotion(f'goal {axis}={v:.3f} m is outside the workspace box '
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

    def run(self):
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
    rclpy.init(args=args)
    node = CartesianTargetPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
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
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
