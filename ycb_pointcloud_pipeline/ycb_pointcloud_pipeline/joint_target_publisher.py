#!/usr/bin/env python3
"""
Activate a CRISP joint impedance controller on the FR3 and send 3 joint targets
very close to the robot's current configuration.

Sequence:
  1. Read the current joint positions from /joint_states (robot must be still).
  2. If `joint_impedance_controller` is not loaded, load it into the robot
     container's controller_manager from config/joint_impedance_controller.yaml
     and configure it.
  3. Ask for confirmation, then strictly switch: activate the joint controller and
     deactivate whatever currently owns the FR3 effort interfaces (e.g. CRISP's
     cartesian_impedance_controller). CRISP seeds its target with the measured
     joint positions on activation, so activation itself does not move the arm.
  4. For each target q_start + offset: smoothly interpolate there over
     `move_duration` seconds, hold for `dwell_duration`, report tracking error.
  5. Leave the controller active, holding the last target.

Aborts (and commands the arm to hold where it is) if joint states go stale or the
tracking error grows too large.

Run with the robot container up (./scripts/run.sh robot) and ROS_DOMAIN_ID matching:
    ros2 run ycb_pointcloud_pipeline joint_target_publisher
"""

import math
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ycb_pointcloud_pipeline.ros_utils import (
    JOINTS, ControllerManagerClient, container_config_path, other_publishers,
)

# FR3 limits from franka_description/robots/fr3/joint_limits.yaml
LOWER = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
UPPER = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
HARD_MAX_OFFSET = 0.85  # rad; offsets above this are refused regardless of parameters

# Three targets relative to the start configuration (rad, joint1..joint7):
# rotate the base a little, then also bend the elbow a little, then go back.
DEFAULT_OFFSETS = [
    0.2, 0.4, 0.1, 0.2, 0.4, -0.2, 0.0,
    0.2, 0.3, 0.5, -0.3, 0.1, 0.3, 0.4,
    -0.1, -0.3, 0.4, 0.3, 0.4, 0.5, 0.0,
    -0.4, 0.3, -0.6, 0.3, 0.4, 0.5, 0.0,
    -0.4, 0.3, -0.6, -0.3, -0.4, -0.5, 0.2,
    -0.4, 0.3, 0.3, 0.3, -0.4, -0.5, 0.2,
    0.6, -0.3, 0.3, 0.3, -0.4, -0.5, 0.2,
    0.8, -0.4, -0.6, 0.3, -0.4, -0.5, 0.2,
    0.8, -0.4, -0.6, -0.1, -0.4, -0.5, 0.2,
    0., 0., 0., 0., 0., -0., 0.0,
    0.2, 0.4, 0.1, 0.2, 0.4, -0.2, 0.0,
    0.2, 0.3, 0.5, -0.3, 0.1, 0.3, 0.4,
    -0.1, -0.3, 0.4, 0.3, 0.4, 0.5, 0.0,
    -0.4, 0.3, -0.6, 0.3, 0.4, 0.5, 0.0,
    -0.4, 0.3, -0.6, -0.3, -0.4, -0.5, 0.2,
    -0.4, 0.3, 0.3, 0.3, -0.4, -0.5, 0.2,
    0.6, -0.3, 0.3, 0.3, -0.4, -0.5, 0.2,
    0.8, -0.4, -0.6, 0.3, -0.4, -0.5, 0.2,
    0.8, -0.4, -0.6, -0.1, -0.4, -0.5, 0.2,
    


    




]


class AbortMotion(RuntimeError):
    pass


class JointTargetPublisher(Node):
    def __init__(self, node_name='joint_target_publisher'):
        super().__init__(node_name)
        self.declare_parameter('controller_manager', '/controller_manager')
        self.declare_parameter('controller_name', 'joint_impedance_controller')
        self.declare_parameter('controller_type', 'crisp_controllers/CartesianController')
        # Path of config/joint_impedance_controller.yaml as seen INSIDE the robot
        # container (run.sh mounts ~/thor_ws at /workspaces/thor_ws). Empty = derive it.
        self.declare_parameter('controller_params_file', '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('offsets', DEFAULT_OFFSETS)
        self.declare_parameter('max_offset', 0.8)        # rad
        self.declare_parameter('limit_margin', 0.35)     # rad kept away from joint limits
        self.declare_parameter('move_duration', 4.0)     # s per target
        self.declare_parameter('dwell_duration', 2.0)    # s hold at each target
        self.declare_parameter('publish_rate', 50.0)     # Hz
        self.declare_parameter('max_tracking_error', 0.85)  # rad, abort above this
        self.declare_parameter('max_start_velocity', 0.05)  # rad/s, robot must be still
        self.declare_parameter('require_confirmation', True)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cm = p('controller_manager').rstrip('/')
        self.controller = p('controller_name')
        self.target_topic = f'/{self.controller}/target_joint'

        self._lock = threading.Lock()
        self._joint_state = None  # (monotonic time, {name: (pos, vel)})
        self.create_subscription(JointState, p('joint_states_topic'), self._joint_state_cb, 10)
        self.target_pub = self.create_publisher(JointState, self.target_topic, 1)
        self.cmc = ControllerManagerClient(self, self.cm, error=AbortMotion)

    # ------------------------------------------------------------------ helpers
    def _joint_state_cb(self, msg):
        if not msg.position:
            return
        values = {}
        for i, name in enumerate(msg.name):
            if name in JOINTS and i < len(msg.position):
                vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                values[name] = (msg.position[i], vel)
        if len(values) == len(JOINTS):
            with self._lock:
                self._joint_state = (time.monotonic(), values)

    def measured(self, max_age=0.5):
        with self._lock:
            state = self._joint_state
        if state is None or time.monotonic() - state[0] > max_age:
            raise AbortMotion(f'No fresh FR3 joint states on '
                              f'{self.get_parameter("joint_states_topic").value}')
        values = state[1]
        return [values[j][0] for j in JOINTS], [values[j][1] for j in JOINTS]

    def publish_target(self, q):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(v) for v in q]
        self.target_pub.publish(msg)

    def hold_current_position(self):
        try:
            q, _ = self.measured(max_age=1.0)
            self.publish_target(q)
            self.get_logger().warn('Commanded the controller to hold the current position.')
        except AbortMotion:
            self.get_logger().error('Could not read joint states to hold position; the '
                                    'controller keeps its last target.')

    # ------------------------------------------------------------------ steps
    def build_targets(self, q_start):
        offsets = list(self.get_parameter('offsets').value)
        if len(offsets) == 0 or len(offsets) % 7:
            raise AbortMotion('offsets must contain 7 values per target')
        max_offset = min(self.get_parameter('max_offset').value, HARD_MAX_OFFSET)
        margin = self.get_parameter('limit_margin').value
        targets = []
        for k in range(len(offsets) // 7):
            off = offsets[7 * k: 7 * k + 7]
            if max(abs(o) for o in off) > max_offset:
                raise AbortMotion(f'Target {k + 1}: offset {off} exceeds max_offset '
                                  f'{max_offset} rad')
            q = [s + o for s, o in zip(q_start, off)]
            for j, (v, lo, hi) in enumerate(zip(q, LOWER, UPPER)):
                if not lo + margin <= v <= hi - margin:
                    raise AbortMotion(f'Target {k + 1}: {JOINTS[j]}={v:.3f} is within '
                                      f'{margin} rad of its limit [{lo}, {hi}]')
            targets.append(q)
        return targets

    def ensure_controller_loaded(self):
        try:
            params_file = (self.get_parameter('controller_params_file').value
                           or container_config_path('joint_impedance_controller.yaml'))
        except ValueError as exc:
            raise AbortMotion(f'{exc} (controller_params_file)')
        return self.cmc.ensure_loaded(self.controller,
                                      self.get_parameter('controller_type').value, params_file)

    def move_to(self, q_from, q_to, label, move_t=None):
        if move_t is None:
            move_t = self.get_parameter('move_duration').value
        dwell_t = self.get_parameter('dwell_duration').value
        period = 1.0 / self.get_parameter('publish_rate').value
        max_err = self.get_parameter('max_tracking_error').value

        changes = ', '.join(f'{j[-6:]} {b:+.3f}->{e:+.3f}'
                            for j, b, e in zip(JOINTS, q_from, q_to) if abs(e - b) > 1e-6)
        self.get_logger().info(f'{label}: {changes or "hold"}')
        start = time.monotonic()
        while rclpy.ok():
            t = time.monotonic() - start
            if t > move_t + dwell_t:
                break
            s = min(t / move_t, 1.0)
            s = 0.5 - 0.5 * math.cos(math.pi * s)  # smooth start/stop
            q_cmd = [b + s * (e - b) for b, e in zip(q_from, q_to)]
            self.publish_target(q_cmd)
            q_meas, _ = self.measured()
            err = max(abs(c - m) for c, m in zip(q_cmd, q_meas))
            if err > max_err:
                raise AbortMotion(f'Tracking error {err:.3f} rad exceeds {max_err} rad')
            time.sleep(period)
        q_meas, _ = self.measured()
        errs = [abs(e - m) for e, m in zip(q_to, q_meas)]
        self.get_logger().info(f'{label} reached: max joint error {max(errs):.4f} rad '
                               f'({JOINTS[errs.index(max(errs))]})')

    def run(self):
        self.get_logger().info('Waiting for joint states ...')
        deadline = time.monotonic() + 10.0
        while self._joint_state is None:
            if time.monotonic() > deadline:
                raise AbortMotion('No FR3 joint states received within 10 s')
            time.sleep(0.05)

        q_start, dq = self.measured()
        if max(abs(v) for v in dq) > self.get_parameter('max_start_velocity').value:
            raise AbortMotion(f'Robot is moving (|dq| max {max(abs(v) for v in dq):.3f} '
                              'rad/s); wait until it is still')
        targets = self.build_targets(q_start)
        self.get_logger().info('Start configuration: ' + ' '.join(f'{v:+.3f}' for v in q_start))

        controllers = self.ensure_controller_loaded()

        others = other_publishers(self, self.target_topic)
        if others:
            raise AbortMotion(f'{others} other publisher(s) on {self.target_topic}; CRISP '
                              'ignores targets when more than one publisher exists')

        if self.get_parameter('require_confirmation').value:
            print(f'\nAbout to ACTIVATE {self.controller} (effort control) and move through '
                  f'{len(targets)} target(s), max offset '
                  f'{max(max(abs(t - s) for t, s in zip(q, q_start)) for q in targets):.3f} rad.'
                  '\nKeep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return

        self.cmc.activate(self.controller, controllers)
        self.moving = True
        q_prev = q_start
        for k, q_target in enumerate(targets):
            self.move_to(q_prev, q_target, f'Target {k + 1}/{len(targets)}')
            q_prev = q_target
        self.moving = False
        self.get_logger().info(f'Done. {self.controller} stays active, holding the last '
                               'target.')


def main(args=None, node_cls=JointTargetPublisher):
    rclpy.init(args=args)
    node = node_cls()
    node.moving = False
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    exit_code = 0
    try:
        node.run()
    except AbortMotion as e:
        node.get_logger().error(f'Aborted: {e}')
        if node.moving:
            node.hold_current_position()
        exit_code = 1
    except KeyboardInterrupt:
        if node.moving:
            node.hold_current_position()
    finally:
        time.sleep(0.2)  # let a final hold command go out
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
