#!/usr/bin/env python3
"""
Send the FR3 back to the standard Franka home configuration with the CRISP joint
impedance controller:

    q_home = [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]

Same sequence and safety checks as joint_target_publisher (load/configure the
controller from config/joint_impedance_controller.yaml, confirm,
strict switch), but one target: home, and no tracking-error monitoring: the target is
interpolated to home and the controller is left holding it. Then the Franka Hand
is opened (franka_gripper move action; -p open_gripper:=false to skip; with the
Inspire hand mounted, -p gripper_type:=inspire calls /inspire_hand/open instead). The move duration is
scaled so no joint exceeds `max_joint_velocity`.

    ros2 run ycb_pointcloud_pipeline joint_homing
"""

import math
import time

import rclpy
from rclpy.action import ActionClient

from ycb_pointcloud_pipeline.joint_target_publisher import (
    LOWER, UPPER, AbortMotion, JointTargetPublisher, main as _main,
)
from ycb_pointcloud_pipeline.ros_utils import JOINTS, other_publishers

FRANKA_HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


class JointHoming(JointTargetPublisher):
    def __init__(self):
        super().__init__('joint_homing')
        self.declare_parameter('home', FRANKA_HOME)
        self.declare_parameter('max_joint_velocity', 0.25)  # rad/s, peak along the move
        self.declare_parameter('min_move_duration', 3.0)    # s
        self.declare_parameter('home_tolerance', 0.02)      # rad, already home below this
        self.declare_parameter('open_gripper', True)        # open the Franka Hand once home
        self.declare_parameter('gripper_type', 'franka')     # franka | inspire
        self.declare_parameter('gripper_namespace', '/franka_gripper')
        self.declare_parameter('gripper_open_width', 0.08)  # m
        self.declare_parameter('gripper_speed', 0.05)       # m/s
        self.gripper_move = None

    def connect_gripper(self):
        """Check the gripper move action before the arm moves, so a missing gripper
        is found while nothing has been activated yet."""
        if not self.get_parameter('open_gripper').value:
            return
        if self.get_parameter('gripper_type').value == 'inspire':
            from ycb_pointcloud_pipeline.ros_utils import InspireHandClient
            ns = self.get_parameter('gripper_namespace').value
            client = InspireHandClient(self, ns if ns != '/franka_gripper' else '/inspire_hand')
            missing = client.available(timeout=5.0)
            if missing:
                raise AbortMotion(f'Inspire hand services {missing} not available '
                                  '(start inspire_hand_node, or pass -p open_gripper:=false)')
            self.gripper_move = (client, None)
            return
        try:
            from franka_msgs.action import Move
        except ImportError:
            raise AbortMotion('open_gripper needs franka_msgs on the host, or pass '
                              '-p open_gripper:=false')
        name = self.get_parameter('gripper_namespace').value.rstrip('/') + '/move'
        self.gripper_move = (ActionClient(self, Move, name), Move)
        if not self.gripper_move[0].wait_for_server(timeout_sec=5.0):
            raise AbortMotion(f'gripper action {name} not available '
                              '(or pass -p open_gripper:=false)')

    def open_gripper(self):
        if self.gripper_move is None:
            return
        client, Move = self.gripper_move
        if Move is None:                  # Inspire hand
            ok, message = client.wait(client.send(False), 20.0)
            (self.get_logger().info if ok else self.get_logger().error)(f'Inspire hand open: {message}')
            return
        goal = Move.Goal()
        goal.width = float(self.get_parameter('gripper_open_width').value)
        goal.speed = float(self.get_parameter('gripper_speed').value)
        self.get_logger().info(f'Opening gripper to {goal.width:.3f} m')
        deadline = time.monotonic() + 10.0
        future = client.send_goal_async(goal)
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or not future.result().accepted:
            self.get_logger().error('Gripper open goal was not accepted')
            return
        result_future = future.result().get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not result_future.done():
            self.get_logger().error('Gripper open did not finish within 10 s')
        elif not result_future.result().result.success:
            self.get_logger().error('Gripper open failed: '
                                    f'{result_future.result().result.error}')
        else:
            self.get_logger().info('Gripper open')

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

        q_home = [float(v) for v in self.get_parameter('home').value]
        if len(q_home) != 7:
            raise AbortMotion('home must contain 7 joint values')
        margin = self.get_parameter('limit_margin').value
        for j, (v, lo, hi) in enumerate(zip(q_home, LOWER, UPPER)):
            if not lo + margin <= v <= hi - margin:
                raise AbortMotion(f'home {JOINTS[j]}={v:.3f} is within {margin} rad of its '
                                  f'limit [{lo}, {hi}]')

        delta = max(abs(h - s) for h, s in zip(q_home, q_start))
        self.get_logger().info('Start configuration: ' + ' '.join(f'{v:+.3f}' for v in q_start))
        self.get_logger().info('Home configuration:  ' + ' '.join(f'{v:+.3f}' for v in q_home))
        self.connect_gripper()
        if delta < self.get_parameter('home_tolerance').value:
            self.get_logger().info(f'Already home (max joint difference {delta:.4f} rad).')
            self.open_gripper()
            return

        # Cosine profile peaks at pi/2 x the average velocity.
        move_t = max(self.get_parameter('min_move_duration').value,
                     delta * math.pi / 2 / self.get_parameter('max_joint_velocity').value)

        controllers = self.ensure_controller_loaded()
        others = other_publishers(self, self.target_topic)
        if others:
            raise AbortMotion(f'{others} other publisher(s) on {self.target_topic}; CRISP '
                              'ignores targets when more than one publisher exists')

        if self.get_parameter('require_confirmation').value:
            print(f'\nAbout to ACTIVATE {self.controller} (effort control) and move HOME: '
                  f'largest joint change {delta:.3f} rad over {move_t:.1f} s'
                  f"{', then OPEN the gripper' if self.gripper_move else ''}."
                  '\nMake sure the path to home is clear. Keep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return

        self.cmc.activate(self.controller, controllers)
        self.moving = True
        self.get_logger().info(f'Moving home over {move_t:.1f} s')
        period = 1.0 / self.get_parameter('publish_rate').value
        start = time.monotonic()
        while rclpy.ok():
            s = min((time.monotonic() - start) / move_t, 1.0)
            s = 0.5 - 0.5 * math.cos(math.pi * s)  # smooth start/stop
            self.publish_target([b + s * (e - b) for b, e in zip(q_start, q_home)])
            if s >= 1.0:
                break
            time.sleep(period)
        self.moving = False
        self.get_logger().info(f'Home target sent. {self.controller} stays active, holding home.')
        self.open_gripper()


def main(args=None):
    _main(args, node_cls=JointHoming)


if __name__ == '__main__':
    main()
