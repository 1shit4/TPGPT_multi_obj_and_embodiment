#!/usr/bin/env python3
"""
Watch the robot's estimate of the external force on the hand, live, on one line.

    ros2 run ycb_pointcloud_pipeline wrench_watch

Prints Fx Fy Fz (N, base frame) averaged over 0.1 s, ten times a second, and Fz's extremes
when stopped (Ctrl-C). Read /franka_robot_state_broadcaster/external_wrench_in_base_frame.

What it is for: the sign test of that estimate (push the hand up: does Fz go up or down
while you push?), and seeing a contact. At rest the number carries static friction: under
the soft joint controller a joint stops short of its target and friction holds up to
~0.6 N m, so after a push the reading settles somewhere new rather than going back
(CLAUDE.md). Only the reading during the push says which way the force points.
"""

import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

TOPIC = '/franka_robot_state_broadcaster/external_wrench_in_base_frame'


class WrenchWatch(Node):
    def __init__(self):
        super().__init__('wrench_watch')
        self.lock = threading.Lock()
        self.buf = []
        self.create_subscription(WrenchStamped, TOPIC, self._cb, 10)

    def _cb(self, msg):
        f = msg.wrench.force
        with self.lock:
            self.buf.append((f.x, f.y, f.z))

    def take(self):
        with self.lock:
            out, self.buf = self.buf, []
        return np.array(out) if out else None


def main(args=None):
    rclpy.init(args=args)
    node = WrenchWatch()
    ex = SingleThreadedExecutor()
    ex.add_node(node)

    def spin():
        try:
            ex.spin()
        except ExternalShutdownException:     # Ctrl-C shuts rclpy down under it
            pass

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    print(f'{TOPIC}, averaged over 0.1 s (Ctrl-C stops):')
    t0, lo, hi, first = time.monotonic(), None, None, None
    try:
        while rclpy.ok():
            time.sleep(0.1)
            f = node.take()
            if f is None:
                sys.stdout.write('\r  (waiting for messages)                          ')
                sys.stdout.flush()
                continue
            m = f.mean(axis=0)
            first = m[2] if first is None else first
            t = time.monotonic() - t0
            lo = (t, m[2]) if lo is None or m[2] < lo[1] else lo
            hi = (t, m[2]) if hi is None or m[2] > hi[1] else hi
            sys.stdout.write(f'\r  t {t:5.1f} s   Fx {m[0]:+6.2f}   Fy {m[1]:+6.2f}   Fz {m[2]:+6.2f} N'
                             f'   (Fz from the start {m[2] - first:+6.2f})   ')
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if lo is not None:
            print(f'  Fz at the start {first:+.2f} N; lowest {lo[1]:+.2f} at {lo[0]:.1f} s, '
                  f'highest {hi[1]:+.2f} at {hi[0]:.1f} s')
        ex.shutdown()
        th.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
