#!/usr/bin/env python3
"""
Weigh what hangs on the FR3's flange, for Desk's end-effector settings (plan M5).

The arm starts near home (run joint_homing first; it moves to home itself), then visits the poses of
tpgpt.hardware.payload.POSES with the joint impedance controller -- home, pose, home, pose,
..., each a joint move of at most 0.8 rad that turns or tilts the hand -- and at each one,
once still, averages the robot's external joint torques
(/franka_robot_state_broadcaster/external_joint_torques) for `sample_s`. Those are what the
robot's model does not explain: with Desk still set for another hand, the payload's weight
error. tpgpt.hardware.payload.fit_payload turns the record into the mass and centre of mass
to type into Desk; the inertia comes from the manufacturer's data.

Every pose was checked offline against the table and the shelf model with the hand on
(`python -m tpgpt.hardware.payload poses`: at least 18 cm above the table, 9 cm from the
shelf). Keep the emergency stop within reach and the space around the hand clear.

    source ~/venvs/tpgpt/bin/activate
    ros2 run --prefix python3 ycb_pointcloud_pipeline payload_measure

The record is saved to ~/inspire_mount/payload_<time>.json; fit it again with
    python -m tpgpt.hardware.payload fit ~/inspire_mount/payload_<time>.json
Run it again after typing the values into Desk: the fit should then find the payload Desk
holds, and the joint offsets alone left.
"""

import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from ycb_pointcloud_pipeline.joint_homing import FRANKA_HOME
from ycb_pointcloud_pipeline.joint_target_publisher import AbortMotion, JointTargetPublisher
from ycb_pointcloud_pipeline.ros_utils import JOINTS, other_publishers


class PayloadMeasure(JointTargetPublisher):
    def __init__(self):
        super().__init__('payload_measure')
        self.declare_parameter('external_torques_topic',
                               '/franka_robot_state_broadcaster/external_joint_torques')
        self.declare_parameter('settle_s', 3.0)          # still for this long before sampling
        self.declare_parameter('sample_s', 2.0)          # averaged over this long
        self.declare_parameter('still_velocity', 0.01)   # rad/s, every joint, while sampling
        self.declare_parameter('max_joint_velocity', 0.25)  # rad/s, peak along a move
        self.declare_parameter('min_move_duration', 3.0)    # s
        # rad: start this close to home. The joint controller leaves the arm a few degrees
        # off its target (2026-10-05 after joint_homing: joint 4 0.056, joint 7 0.071 rad);
        # the node moves to home itself first, and the fit uses the measured joints.
        self.declare_parameter('home_tolerance', 0.3)
        self.declare_parameter('output_dir', os.path.expanduser('~/inspire_mount'))
        self._tau_lock = threading.Lock()
        self._tau = []                                   # (monotonic, [7]) while recording
        self._recording = False
        self.create_subscription(JointState, self.get_parameter('external_torques_topic').value,
                                 self._tau_cb, 10)

    def _tau_cb(self, msg):
        if not self._recording or len(msg.effort) < 7:
            return
        by_name = dict(zip(msg.name, msg.effort)) if msg.name else {}
        tau = [by_name[j] for j in JOINTS] if all(j in by_name for j in JOINTS) \
            else list(msg.effort[:7])
        with self._tau_lock:
            self._tau.append((time.monotonic(), [float(v) for v in tau]))

    def _move(self, q_from, q_to, label):
        delta = max(abs(a - b) for a, b in zip(q_from, q_to))
        move_t = max(self.get_parameter('min_move_duration').value,
                     delta * math.pi / 2 / self.get_parameter('max_joint_velocity').value)
        self.move_to(q_from, q_to, label, move_t=move_t)

    def _sample(self, label, target, approach):
        """Wait until still, then average the external torques and the joints. ``approach``:
        each joint's last commanded motion (+1 up, -1 down): friction's sign."""
        settle, span = self.get_parameter('settle_s').value, self.get_parameter('sample_s').value
        still = self.get_parameter('still_velocity').value
        time.sleep(settle)
        deadline = time.monotonic() + 10.0
        while True:
            _, dq = self.measured()
            if max(abs(v) for v in dq) <= still:
                break
            if time.monotonic() > deadline:
                raise AbortMotion(f'{label}: the arm did not come to rest '
                                  f'(|dq| max {max(abs(v) for v in dq):.3f} rad/s)')
            time.sleep(0.1)
        with self._tau_lock:
            self._tau = []
        self._recording = True
        qs = []
        end = time.monotonic() + span
        while time.monotonic() < end:
            q, dq = self.measured()
            if max(abs(v) for v in dq) > 3 * still:
                self._recording = False
                raise AbortMotion(f'{label}: the arm moved while sampling')
            qs.append(q)
            time.sleep(0.02)
        self._recording = False
        with self._tau_lock:
            taus = [t for _, t in self._tau]
        if len(taus) < 10:
            raise AbortMotion(f'{label}: only {len(taus)} external-torque messages in {span} s '
                              f'on {self.get_parameter("external_torques_topic").value}')
        tau = np.mean(taus, axis=0)
        self.get_logger().info(f'{label}: tau_ext ' + ' '.join(f'{v:+.2f}' for v in tau)
                               + f' N m ({len(taus)} msgs, sd {np.std(taus, axis=0).max():.2f})')
        q_mean = np.mean(qs, axis=0)
        return {'q': q_mean.tolist(), 'tau_ext': tau.tolist(), 'target': list(target),
                'approach': [float(a) for a in approach],
                'tau_sd': np.std(taus, axis=0).tolist(), 'n': len(taus), 'label': label}

    def run(self):
        from tpgpt.hardware import payload as pl

        self.get_logger().info('Waiting for joint states ...')
        deadline = time.monotonic() + 10.0
        while self._joint_state is None:
            if time.monotonic() > deadline:
                raise AbortMotion('No FR3 joint states received within 10 s')
            time.sleep(0.05)
        q_start, dq = self.measured()
        if max(abs(v) for v in dq) > self.get_parameter('max_start_velocity').value:
            raise AbortMotion('Robot is moving; wait until it is still')
        home = list(FRANKA_HOME)
        off_home = max(abs(a - b) for a, b in zip(q_start, home))
        if off_home > self.get_parameter('home_tolerance').value:
            raise AbortMotion(f'{off_home:.3f} rad from home: start near home '
                              '(ros2 run ycb_pointcloud_pipeline joint_homing)')
        margin = self.get_parameter('limit_margin').value
        from ycb_pointcloud_pipeline.joint_target_publisher import LOWER, UPPER
        targets = []
        for k, off in enumerate(pl.POSES):
            q = [h + o for h, o in zip(home, off)]
            for j, (v, lo, hi) in enumerate(zip(q, LOWER, UPPER)):
                if not lo + margin <= v <= hi - margin:
                    raise AbortMotion(f'pose {k}: {JOINTS[j]}={v:.3f} within {margin} rad of a limit')
            targets.append(q)
        controllers = self.ensure_controller_loaded()
        others = other_publishers(self, self.target_topic)
        if others:
            raise AbortMotion(f'{others} other publisher(s) on {self.target_topic}')
        if self.get_parameter('require_confirmation').value:
            print(f'\nAbout to ACTIVATE {self.controller} and visit {len(targets)} poses around '
                  'home (home -> pose -> home), at most 0.8 rad per joint, holding still at each '
                  'to weigh the hand: about 4 minutes.\nKeep the space around the hand clear '
                  'and the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was activated.')
                return
        self.cmc.activate(self.controller, controllers)
        self.moving = True
        self._move(q_start, home, 'to home')
        last = np.sign(np.subtract(home, q_start))       # each joint's last motion
        samples = []
        for k, q in enumerate(targets):
            label = f'pose {k + 1}/{len(targets)}'
            if k:
                self._move(home, q, label)
                d = np.subtract(q, home)
                last = np.where(np.abs(d) > 1e-9, np.sign(d), last)
            samples.append(self._sample(label, q if k else home, last))
            if k:
                self._move(q, home, f'{label}: back home')
                last = np.where(np.abs(d) > 1e-9, -np.sign(d), last)
        self.moving = False
        os.makedirs(self.get_parameter('output_dir').value, exist_ok=True)
        path = os.path.join(self.get_parameter('output_dir').value,
                            f"payload_{time.strftime('%Y%m%d_%H%M%S')}.json")
        modelled = pl.FRANKA_HAND_DESK
        with open(path, 'w') as f:
            json.dump({'modelled': {'mass': modelled.mass, 'com': list(modelled.com),
                                    'inertia': [list(r) for r in modelled.inertia],
                                    'note': "Desk's end-effector values while measuring"},
                       'poses': [list(p) for p in pl.POSES], 'samples': samples}, f, indent=1)
        self.get_logger().info(f'saved {len(samples)} poses to {path}; at home, held')
        try:
            ee = pl._hand_from_config(None)
            cad = pl.inspire_cad_payload(ee)
            fit = pl.fit_payload([(s['q'], s['tau_ext'], s['approach']) for s in samples],
                                 modelled, inertia=cad.inertia)
            print('\n' + fit.describe())
            print(f"\nthe manufacturer's data says {cad.mass:.3f} kg at "
                  + ' '.join(f'{v * 1000:+.1f}' for v in cad.com) + ' mm')
            print('\nfor Desk (Settings > End effector; mass, centre of mass and inertia only):\n'
                  + fit.payload.desk_text())
        except Exception as exc:  # noqa: BLE001 - the record is saved; fit it offline
            self.get_logger().error(f'fit failed ({exc}); python -m tpgpt.hardware.payload fit {path}')


def main(args=None):
    # joint_target_publisher.main's structure, single-threaded: rclpy's MultiThreadedExecutor
    # starves high-rate subscriptions on this machine (CLAUDE.md), and this node reads the
    # joint states every cycle and the external torques at the robot's rate.
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=args)
    node = PayloadMeasure()
    node.moving = False
    executor = SingleThreadedExecutor()
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
        spin_thread.join(timeout=2.0)  # no live spinning thread at teardown ("terminate called")
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
