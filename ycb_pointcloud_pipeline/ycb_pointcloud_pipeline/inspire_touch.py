#!/usr/bin/env python3
"""
Record index-fingertip touches for measuring the Inspire hand's mount (plan M3), and fit it.

The hand is put in the pointing pose (`ros2 service call /inspire_hand/point
std_srvs/srv/Trigger`: index straight, the others curled, thumb out of the way) and the
arm hand-guided with free_drive until the index fingertip rests on something whose
position is known. At each touch, type a command and press Enter:

    t [label]        the fingertip rests on the TABLE (its touched height, shelf.yaml)
    p <slot>         the fingertip is on a touched SLOT MARK (top_left, top_right, ...)
    x <x> <y> <z>    the fingertip is on a known point, base frame, m
    u                undo the last touch
    f                fit the mount to the touches so far and print it
    s                save
    q                save and quit

Tilt the hand 15-30 deg in a different direction for each table touch: touches made
straight down cannot tell the mount's yaw from its offset (the fit says so). Six
table touches and the two top slot marks are enough; more is better.

Between touches a status line shows where the CURRENT mount estimate (the yaml's) puts
the index tip: its height above the table, and its distance to the nearest slot mark.
A touch the estimate already agrees with to a millimetre or two needs no fit.

Touches are saved to ~/inspire_mount/touches_<time>.json; fit them again any time with
    python -m tpgpt.hardware.inspire_mount fit ~/inspire_mount/touches_<time>.json

Needs tpgpt: run it from the tpgpt venv with --prefix python3.
    source ~/venvs/tpgpt/bin/activate
    ros2 run --prefix python3 ycb_pointcloud_pipeline inspire_touch
"""

import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

from ycb_pointcloud_pipeline.ros_utils import pose_to_matrix

MAX_POSE_AGE = 0.5          # s
MAX_HAND_AGE = 1.0          # s
POSE_TOLERANCE = 40         # register units from POINT_POSE before a warning


def installed_config(name):
    """Installed config/<name> (what the nodes read), else the source tree's."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'), 'config',
                            name)
    except Exception:  # noqa: BLE001 - package not installed
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', name)


class TouchRecorder(Node):
    def __init__(self):
        super().__init__('inspire_touch')
        self.declare_parameter('hand_config', '')       # '' -> installed inspire_hand.yaml
        self.declare_parameter('shelf_config', '')      # '' -> installed shelf.yaml
        self.declare_parameter('output_dir', os.path.expanduser('~/inspire_mount'))
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        from tpgpt.hardware import inspire_mount as im
        from tpgpt.hardware.shelf import slots
        self.im = im
        self.hand_config = p('hand_config') or installed_config('inspire_hand.yaml')
        shelf_cfg = yaml.safe_load(open(p('shelf_config') or installed_config('shelf.yaml')))
        self.table = float(shelf_cfg['shelf']['anchors']['table'])
        self.slots = {k: np.asarray(v, dtype=float) for k, v in slots(shelf_cfg).items()}
        self.base, self.start = im._base_from_config(self.hand_config)
        self.params = [self.start[k] for k in im.PARAMS]
        self.pose = self.registers = None
        self.touches = []
        os.makedirs(p('output_dir'), exist_ok=True)
        self.path = os.path.join(p('output_dir'), f"touches_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.create_subscription(PoseStamped, '/current_pose', self._pose, 10)
        self.create_subscription(Int16MultiArray, '/inspire_hand/registers', self._registers, 10)
        self.get_logger().info(
            f'table {self.table:+.4f} m, slot marks {sorted(self.slots)}; mount estimate '
            f'{self.start} from {self.hand_config}; saving to {self.path}')

    def _pose(self, msg):
        self.pose = (time.monotonic(), pose_to_matrix(msg.pose))

    def _registers(self, msg):
        self.registers = (time.monotonic(), [int(v) for v in msg.data[:6]])

    # ----------------------------------------------------------------- touches
    def current(self):
        """(T_tcp, registers) if both are fresh, else (None, reason)."""
        now = time.monotonic()
        if self.pose is None or now - self.pose[0] > MAX_POSE_AGE:
            return None, 'no fresh /current_pose (robot container up, ROS_DOMAIN_ID 61?)'
        if self.registers is None or now - self.registers[0] > MAX_HAND_AGE:
            return None, 'no fresh /inspire_hand/registers (inspire_hand_node running?)'
        return (self.pose[1], self.registers[1]), None

    def touch(self, kind, target, label):
        state, why = self.current()
        if state is None:
            print(f'  NOT recorded: {why}')
            return
        T, regs = state
        off = max(abs(a - b) for a, b in zip(regs, self.im.POINT_POSE))
        if off > POSE_TOLERANCE:
            print(f'  WARNING: the hand is not in the pointing pose (registers {regs}, '
                  f'{off} off): call /inspire_hand/point first. Recorded anyway.')
        t = self.im.Touch(kind, list(map(float, target)), T[:3, 3].tolist(),
                          T[:3, :3].tolist(), regs, label)
        self.touches.append(t)
        r = self.im.touch_residual(t, self.params, self.base) * 1000.0
        print(f'  touch {len(self.touches)} ({kind} {label}): the current estimate is off by '
              f'{np.round(r, 1).tolist()} mm')

    def status(self):
        state, why = self.current()
        if state is None:
            return why
        T, regs = state
        probe = self.im.Touch('point', [0.0, 0.0, 0.0], T[:3, 3].tolist(), T[:3, :3].tolist(),
                              regs)
        W = self.im.touch_points(probe, self.params, self.base)
        from tpgpt.hardware.end_effector import inspire_end_effector
        model = inspire_end_effector(**{**self.base, **self.im._mount(self.params)}).model
        tip = W[self.im._tip_index(model, self.im.TOUCH_LINK)]
        name, mark = min(self.slots.items(), key=lambda kv: np.linalg.norm(kv[1] - tip))
        return (f'index lowest point {(W[:, 2].min() - self.table) * 1000:+6.1f} mm above the '
                f'table; tip {np.linalg.norm(mark - tip) * 1000:6.1f} mm from {name}')

    def fit(self):
        if not self.touches:
            print('  no touches yet')
            return
        try:
            result = self.im.fit_mount(self.touches, self.start, self.base)
        except ValueError as exc:
            print(f'  {exc}')
            return
        print(result.describe(self.touches))
        print('\n  for config/inspire_hand.yaml (then colcon build):\n' + result.yaml_block())

    def save(self):
        with open(self.path, 'w') as f:
            json.dump({'table': self.table, 'hand_config': self.hand_config,
                       'mount_estimate': self.start,
                       'touches': [t.as_dict() for t in self.touches]}, f, indent=1)
        print(f'  saved {len(self.touches)} touches to {self.path}')


def prompt_loop(node):
    print(__doc__.split('\n\n')[2])
    while rclpy.ok():
        try:
            line = input(f'[{node.status()}]\n> ').strip()
        except EOFError:
            line = 'q'
        if not line:
            continue
        cmd, *args = line.split()
        try:
            if cmd == 't':
                node.touch('table', [node.table], args[0] if args else f't{len(node.touches) + 1}')
            elif cmd == 'p':
                if not args or args[0] not in node.slots:
                    print(f'  p <slot>, one of {sorted(node.slots)}')
                    continue
                node.touch('point', node.slots[args[0]], args[0])
            elif cmd == 'x':
                node.touch('point', [float(v) for v in args[:3]],
                           args[3] if len(args) > 3 else 'x')
            elif cmd == 'u':
                if node.touches:
                    print(f'  removed {node.touches.pop().label}')
            elif cmd == 'f':
                node.fit()
            elif cmd == 's':
                node.save()
            elif cmd == 'q':
                node.save()
                break
            else:
                print('  t [label] | p <slot> | x x y z [label] | u | f | s | q')
        except (ValueError, IndexError) as exc:
            print(f'  {exc}')


def main(args=None):
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=args)
    node = TouchRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        prompt_loop(node)
    except KeyboardInterrupt:
        node.save()
    finally:
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
