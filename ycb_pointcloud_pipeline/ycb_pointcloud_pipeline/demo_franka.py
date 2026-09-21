#!/usr/bin/env python3
"""
Minimal pick-and-place demo with the CRISP joint impedance controller and the
Franka Hand. Both poses are joint configurations written in this file:

    1. open the gripper
    2. move to PICK_POSE
    3. close the gripper (franka_gripper grasp action, any final width accepted)
    4. move to HOME_POSE (standard Franka home), then sway the elbow in the
       null space (hand pose stays fixed, `null_space_amplitude` 0 disables)
    5. move to PLACE_POSE
    6. open the gripper (drop the object)

The controller is left active, holding PLACE_POSE.

Fill in PICK_POSE and PLACE_POSE below before running (the script refuses to move
while they are None). To record a pose, put the arm there (e.g. with free_drive)
and run
    ros2 run ycb_pointcloud_pipeline demo_franka --ros-args -p print_current:=true
which prints the current configuration as a Python list and exits.

Controller loading, confirmation, strict switch, tracking-error abort and
hold-on-abort are the same as joint_target_publisher. Move durations are scaled
so no joint exceeds `max_joint_velocity`.

    ros2 run ycb_pointcloud_pipeline demo_franka
"""

import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient

from ycb_pointcloud_pipeline.joint_homing import FRANKA_HOME
from ycb_pointcloud_pipeline.joint_target_publisher import (
    LOWER, UPPER, AbortMotion, JointTargetPublisher, main as _main,
)
from ycb_pointcloud_pipeline.ros_utils import JOINTS, other_publishers

# ---------------------------------------------------------------- user poses
# Joint positions in rad, fr3_joint1 .. fr3_joint7.
# PLACEHOLDERS: replace None with 7 values, e.g.
#   PICK_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
# Pick -> home -> place: each move is a straight line in joint space, so choose
# poses for which those paths do not drag the object over anything.
PICK_POSE = [0.34, 0.49, 0.822, -1.70, 2.06, 2.671, 1.3324955701828003]   # TODO: gripper around the object, fingers open
PLACE_POSE = [-0.58, 0.745, 0.022, -1.826, -0.0, 2.571, 1.5785]  # TODO: where the object is released
HOME_POSE = FRANKA_HOME  # visited between pick and place

# FR3 kinematics (Craig DH: a, d, alpha per joint, then the flange), used only to
# find the null space of the arm Jacobian for the elbow sway.
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2),
      (0.0825, 0.0, math.pi / 2), (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2),
      (0.088, 0.0, math.pi / 2)]
FLANGE_D = 0.107


def _dh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0.0, 0.0, 0.0, 1.0]])


def flange_pose(q):
    T = np.eye(4)
    for (a, d, alpha), theta in zip(DH, q):
        T = T @ _dh(a, d, alpha, theta)
    return T @ _dh(0.0, FLANGE_D, 0.0, 0.0)


def pose_error(T_goal, T):
    # 6-vector (position, rotation) that moves T towards T_goal, in the base frame
    rot = 0.5 * sum(np.cross(T[:3, i], T_goal[:3, i]) for i in range(3))
    return np.concatenate([T_goal[:3, 3] - T[:3, 3], rot])


def jacobian(q, eps=1e-6):
    T = flange_pose(q)
    J = np.zeros((6, 7))
    for j in range(7):
        dq = np.array(q, dtype=float)
        dq[j] += eps
        J[:, j] = pose_error(flange_pose(dq), T) / eps
    return J


def null_space_path(q0, amplitude, step=0.005):
    """Joint configurations along the arm's self-motion through q0, sampled every
    `step` (unit null-space arc length) for s in [-amplitude, amplitude]. Returns
    (s values, configurations); the flange pose of q0 is held by a small correction."""
    T0 = flange_pose(q0)
    n_steps = max(1, int(math.ceil(amplitude / step)))
    branches = []
    for sign in (1.0, -1.0):
        q, prev = np.array(q0, dtype=float), None
        branch = []
        for _ in range(n_steps):
            J = jacobian(q)
            n = np.linalg.svd(J)[2][-1]
            if prev is None:
                n = n if n[0] >= 0 else -n   # +s turns joint 1 positive
                n = sign * n
            elif n @ prev < 0:
                n = -n
            q = q + n * step + np.linalg.pinv(J) @ pose_error(T0, flange_pose(q))
            prev = n
            branch.append(q.copy())
        branches.append(branch)
    qs = branches[1][::-1] + [np.array(q0, dtype=float)] + branches[0]
    ss = [step * k for k in range(-n_steps, n_steps + 1)]
    return np.array(ss), np.array(qs)


class DemoFranka(JointTargetPublisher):
    def __init__(self):
        super().__init__('demo_franka')
        self.declare_parameter('print_current', False)       # print joint config and exit
        self.declare_parameter('max_joint_velocity', 0.25)   # rad/s, peak along a move
        self.declare_parameter('min_move_duration', 3.0)     # s
        self.declare_parameter('gripper_namespace', '/franka_gripper')
        self.declare_parameter('gripper_open_width', 0.08)   # m
        self.declare_parameter('gripper_speed', 0.05)        # m/s
        self.declare_parameter('grasp_force', 20.0)          # N
        self.declare_parameter('gripper_timeout', 15.0)      # s
        self.declare_parameter('null_space_amplitude', 0.8)  # rad along the null space, 0 = skip
        self.declare_parameter('null_space_period', 8.0)     # s per sway
        self.declare_parameter('null_space_cycles', 5)       # number of sways
        self.gripper = None

    # ------------------------------------------------------------------ gripper
    def connect_gripper(self):
        try:
            from franka_msgs.action import Grasp, Move
        except ImportError:
            raise AbortMotion('demo_franka needs franka_msgs on the host')
        ns = self.get_parameter('gripper_namespace').value.rstrip('/')
        self.gripper = {'Move': Move, 'Grasp': Grasp,
                        'move': ActionClient(self, Move, f'{ns}/move'),
                        'grasp': ActionClient(self, Grasp, f'{ns}/grasp')}
        for name in ('move', 'grasp'):
            if not self.gripper[name].wait_for_server(timeout_sec=5.0):
                raise AbortMotion(f'gripper action {ns}/{name} not available')

    def _wait(self, future, deadline, what):
        while not future.done():
            if time.monotonic() > deadline:
                raise AbortMotion(f'{what} did not finish within '
                                  f'{self.get_parameter("gripper_timeout").value} s')
            time.sleep(0.02)
        return future.result()

    def _gripper_goal(self, client, goal, what):
        deadline = time.monotonic() + self.get_parameter('gripper_timeout').value
        handle = self._wait(client.send_goal_async(goal), deadline, what)
        if not handle.accepted:
            raise AbortMotion(f'{what} goal was not accepted')
        result = self._wait(handle.get_result_async(), deadline, what).result
        if not result.success:
            raise AbortMotion(f'{what} failed: {result.error}')

    def open_gripper(self):
        goal = self.gripper['Move'].Goal()
        goal.width = float(self.get_parameter('gripper_open_width').value)
        goal.speed = float(self.get_parameter('gripper_speed').value)
        self.get_logger().info(f'Opening gripper to {goal.width:.3f} m')
        self._gripper_goal(self.gripper['move'], goal, 'Gripper open')

    def close_gripper(self):
        goal = self.gripper['Grasp'].Goal()
        # libfranka reports success only if width - inner < d < width + outer (strict).
        # With inner = 0, closing on nothing ends at d ~ 0 and fails with an empty
        # error, so give the inner side some slack: any final width counts as success.
        goal.width = 0.0
        goal.epsilon.inner = 0.01
        goal.epsilon.outer = float(self.get_parameter('gripper_open_width').value)
        goal.speed = float(self.get_parameter('gripper_speed').value)
        goal.force = float(self.get_parameter('grasp_force').value)
        self.get_logger().info(f'Closing gripper (force {goal.force:.0f} N)')
        self._gripper_goal(self.gripper['grasp'], goal, 'Gripper close')

    # ------------------------------------------------------------------ motion
    def check_pose(self, name, q):
        if q is None:
            raise AbortMotion(f'{name} is still a placeholder: set it at the top of '
                              'demo_franka.py (-p print_current:=true prints the current '
                              'configuration)')
        if len(q) != 7:
            raise AbortMotion(f'{name} must contain 7 joint values')
        margin = self.get_parameter('limit_margin').value
        for j, (v, lo, hi) in enumerate(zip(q, LOWER, UPPER)):
            if not lo + margin <= v <= hi - margin:
                raise AbortMotion(f'{name} {JOINTS[j]}={v:.3f} is within {margin} rad of '
                                  f'its limit [{lo}, {hi}]')
        return [float(v) for v in q]

    def move_duration(self, q_from, q_to):
        # Cosine profile peaks at pi/2 x the average velocity.
        delta = max(abs(e - b) for b, e in zip(q_from, q_to))
        return max(self.get_parameter('min_move_duration').value,
                   delta * math.pi / 2 / self.get_parameter('max_joint_velocity').value)

    def plan_null_space(self, q_home):
        amplitude = float(self.get_parameter('null_space_amplitude').value)
        if amplitude <= 0.0:
            return None
        ss, qs = null_space_path(q_home, amplitude)
        drift = max(np.linalg.norm(flange_pose(q)[:3, 3] - flange_pose(q_home)[:3, 3])
                    for q in qs)
        if drift > 0.005:
            raise AbortMotion(f'Null-space path moves the hand {drift * 1000:.1f} mm; '
                              'lower null_space_amplitude')
        margin = self.get_parameter('limit_margin').value
        for q in qs:
            for j, (v, lo, hi) in enumerate(zip(q, LOWER, UPPER)):
                if not lo + margin <= v <= hi - margin:
                    raise AbortMotion(f'Null-space sway takes {JOINTS[j]} to {v:.3f}, within '
                                      f'{margin} rad of its limit; lower null_space_amplitude')
        span = ' '.join(f'{lo:+.2f}..{hi:+.2f}' for lo, hi in zip(qs.min(0), qs.max(0)))
        self.get_logger().info(f'Null-space sway joint ranges: {span}')
        return ss, qs

    def null_space_sway(self, plan):
        ss, qs = plan
        amplitude = ss[-1]
        period = float(self.get_parameter('null_space_period').value)
        cycles = max(1, int(self.get_parameter('null_space_cycles').value))
        total = period * cycles
        rate_period = 1.0 / self.get_parameter('publish_rate').value
        max_err = self.get_parameter('max_tracking_error').value
        self.get_logger().info(f'Null-space sway: {cycles} x {period:.1f} s, '
                               f'amplitude {amplitude:.2f}')
        start = time.monotonic()
        while rclpy.ok():
            t = time.monotonic() - start
            if t > total:
                break
            # sine sway under a sin^2 envelope: starts and ends at home with zero velocity
            s = amplitude * math.sin(2 * math.pi * t / period) * math.sin(math.pi * t / total) ** 2
            q_cmd = [float(np.interp(s, ss, qs[:, j])) for j in range(7)]
            self.publish_target(q_cmd)
            q_meas, _ = self.measured()
            err = max(abs(c - m) for c, m in zip(q_cmd, q_meas))
            if err > max_err:
                raise AbortMotion(f'Tracking error {err:.3f} rad exceeds {max_err} rad')
            time.sleep(rate_period)
        self.publish_target(qs[len(ss) // 2])

    def run(self):
        self.get_logger().info('Waiting for joint states ...')
        deadline = time.monotonic() + 10.0
        while self._joint_state is None:
            if time.monotonic() > deadline:
                raise AbortMotion('No FR3 joint states received within 10 s')
            time.sleep(0.05)

        q_start, dq = self.measured()
        if self.get_parameter('print_current').value:
            print('[' + ', '.join(f'{v:.4f}' for v in q_start) + ']')
            return
        if max(abs(v) for v in dq) > self.get_parameter('max_start_velocity').value:
            raise AbortMotion(f'Robot is moving (|dq| max {max(abs(v) for v in dq):.3f} '
                              'rad/s); wait until it is still')

        q_pick = self.check_pose('PICK_POSE', PICK_POSE)
        q_place = self.check_pose('PLACE_POSE', PLACE_POSE)
        q_home = self.check_pose('HOME_POSE', HOME_POSE)
        t_pick = self.move_duration(q_start, q_pick)
        t_home = self.move_duration(q_pick, q_home)
        t_place = self.move_duration(q_home, q_place)
        sway = self.plan_null_space(q_home)
        self.get_logger().info('Start: ' + ' '.join(f'{v:+.3f}' for v in q_start))
        self.get_logger().info('Pick:  ' + ' '.join(f'{v:+.3f}' for v in q_pick))
        self.get_logger().info('Home:  ' + ' '.join(f'{v:+.3f}' for v in q_home))
        self.get_logger().info('Place: ' + ' '.join(f'{v:+.3f}' for v in q_place))

        self.connect_gripper()
        controllers = self.ensure_controller_loaded()
        others = other_publishers(self, self.target_topic)
        if others:
            raise AbortMotion(f'{others} other publisher(s) on {self.target_topic}; CRISP '
                              'ignores targets when more than one publisher exists')

        if self.get_parameter('require_confirmation').value:
            print(f'\nAbout to ACTIVATE {self.controller} (effort control), OPEN the gripper, '
                  f'move to PICK ({t_pick:.1f} s), CLOSE, move to HOME ({t_home:.1f} s)'
                  f'{", SWAY in null space" if sway else ""}, '
                  f'move to PLACE ({t_place:.1f} s), OPEN.\nMake sure all paths are clear. '
                  'Keep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return

        self.cmc.activate(self.controller, controllers)
        self.open_gripper()
        self.moving = True
        self.move_to(q_start, q_pick, 'Pick', move_t=t_pick)
        self.close_gripper()
        self.move_to(q_pick, q_home, 'Home', move_t=t_home)
        if sway:
            self.null_space_sway(sway)
        self.move_to(q_home, q_place, 'Place', move_t=t_place)
        self.open_gripper()
        self.moving = False
        self.get_logger().info(f'Done. {self.controller} stays active, holding PLACE_POSE.')


def main(args=None):
    _main(args, node_cls=DemoFranka)


if __name__ == '__main__':
    main()
