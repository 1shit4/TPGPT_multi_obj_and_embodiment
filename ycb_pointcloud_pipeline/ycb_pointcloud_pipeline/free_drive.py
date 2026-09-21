#!/usr/bin/env python3
"""
Free drive for the FR3: open the gripper, then switch to a zero-stiffness Cartesian
controller so the arm floats and can be moved by hand.

Sequence:
  1. Wait for /current_pose (the robot container must be up with FCI active).
  2. Open the hand (-p open_gripper:=false skips). gripper_type (default auto = the hand
     x6_experiment.yaml's hands.mounted names, else franka):
       franka  -- franka_gripper move action
       inspire -- /inspire_hand/open (inspire_hand_node must be running): fingers straight,
                  thumb opposed
  3. Load and configure `tpgpt_teaching_controller` from
     config/tpgpt_teaching_controller.yaml if needed: every task stiffness k_* = 0
     and nullspace stiffness 0, light damping only. libfranka compensates gravity.
  4. Ask for 'yes', then strictly switch it in (deactivating whatever owns the FR3
     effort interfaces). The arm now floats.
  5. While floating, type a letter and Enter (Inspire hand only):
       c  close the hand (fingers wrap, then the thumb; stops on contact)
       o  open the hand
     The hand moves in the background; the arm keeps floating meanwhile.
  6. Press Enter alone or Ctrl-C: switch to `hold_controller`
     (cartesian_impedance_controller), which CRISP seeds with the measured pose, so the
     arm stays where it was let go. The hand is left as it is (a held object stays held).

If /current_pose goes stale while floating, it switches to the hold controller.
The hand's mass must be set correctly in Desk, or the arm sinks or rises on its own.

    ros2 run ycb_pointcloud_pipeline free_drive
    ros2 run ycb_pointcloud_pipeline free_drive --ros-args -p gripper_type:=inspire
"""

import select
import signal
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from ycb_pointcloud_pipeline.ros_utils import ControllerManagerClient, container_config_path


class FreeDriveAbort(RuntimeError):
    pass


class FreeDrive(Node):
    def __init__(self):
        super().__init__('free_drive')
        d = self.declare_parameter
        d('controller_manager', '/controller_manager')
        d('free_controller', 'tpgpt_teaching_controller')
        d('free_controller_type', 'crisp_controllers/CartesianController')
        d('free_controller_params_file', '')    # path INSIDE the robot container; '' = derive
        d('hold_controller', 'cartesian_impedance_controller')
        d('current_pose_topic', '/current_pose')
        d('max_pose_age', 0.5)                  # s; older while floating -> hold
        d('open_gripper', True)
        d('gripper_type', 'auto')               # auto | franka | inspire
        d('gripper_namespace', '/franka_gripper')
        d('gripper_open_width', 0.08)           # m
        d('gripper_speed', 0.05)                # m/s
        d('require_confirmation', True)

        self.floating = False
        self._pose_time = None
        self.gripper_type = self._resolve_gripper_type()
        self.inspire = None                      # InspireHandClient once connected
        self.create_subscription(PoseStamped, self.p('current_pose_topic'), self._pose_cb, 10)
        self.cmc = ControllerManagerClient(self, self.p('controller_manager'), error=FreeDriveAbort)

    def p(self, name):
        return self.get_parameter(name).value

    def _resolve_gripper_type(self):
        kind = str(self.p('gripper_type')).strip().lower()
        if kind == 'auto':
            try:
                from ycb_pointcloud_pipeline.x6_common import X6Config
                mounted = X6Config().raw.get('hands', {}).get('mounted')
            except Exception:  # noqa: BLE001 - no experiment config: the Franka Hand
                mounted = None
            kind = 'inspire' if mounted == 'inspire' else 'franka'
            self.get_logger().info(f'gripper_type auto -> {kind} (x6_experiment.yaml hands.mounted: '
                                   f'{mounted})')
        if kind not in ('franka', 'inspire'):
            raise SystemExit(f"gripper_type must be auto, franka or inspire, not {kind!r}")
        return kind

    def connect_inspire(self):
        from ycb_pointcloud_pipeline.ros_utils import InspireHandClient
        ns = self.p('gripper_namespace')
        client = InspireHandClient(self, ns if ns != '/franka_gripper' else '/inspire_hand')
        missing = client.available(timeout=5.0)
        if missing:
            raise FreeDriveAbort(f'Inspire hand services {missing} not available '
                                 '(start: ros2 run ycb_pointcloud_pipeline inspire_hand_node, '
                                 'or pass -p open_gripper:=false)')
        self.inspire = client
        return client

    def command_inspire(self, close):
        """Send open/close without blocking; the answer is logged when it arrives."""
        label = 'CLOSE' if close else 'OPEN'
        log = self.get_logger()
        try:
            client = self.inspire or self.connect_inspire()
        except FreeDriveAbort as exc:
            log.error(str(exc))
            return

        def done(future):
            result = future.result()
            if result is not None and result.success:
                log.info(f'Inspire hand {label}: {result.message}')
            else:
                log.error(f"Inspire hand {label} failed: {result.message if result else 'no answer'}")
        client.send(close).add_done_callback(done)
        log.info(f'Inspire hand {label} sent')

    def _pose_cb(self, _msg):
        self._pose_time = time.monotonic()

    def pose_fresh(self):
        return (self._pose_time is not None
                and time.monotonic() - self._pose_time <= self.p('max_pose_age'))

    def wait_for_pose(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while not self.pose_fresh():
            if time.monotonic() > deadline:
                raise FreeDriveAbort(f"no pose on {self.p('current_pose_topic')} within "
                                     f'{timeout:.0f} s: is the robot container up with FCI active?')
            time.sleep(0.05)

    def open_gripper(self):
        if not self.p('open_gripper'):
            return
        if self.gripper_type == 'inspire':
            client = self.connect_inspire()
            ok, message = client.wait(client.send(False), 20.0)
            if not ok:
                raise FreeDriveAbort(f'Inspire hand open failed: {message}')
            self.get_logger().info(f'Inspire hand open: {message}')
            return
        try:
            from franka_msgs.action import Move
        except ImportError:
            raise FreeDriveAbort('open_gripper needs franka_msgs on the host, or pass '
                                 '-p open_gripper:=false')
        name = self.p('gripper_namespace').rstrip('/') + '/move'
        client = ActionClient(self, Move, name)
        if not client.wait_for_server(timeout_sec=5.0):
            raise FreeDriveAbort(f'gripper action {name} not available '
                                 '(or pass -p open_gripper:=false)')
        goal = Move.Goal()
        goal.width = float(self.p('gripper_open_width'))
        goal.speed = float(self.p('gripper_speed'))
        self.get_logger().info(f'Opening gripper to {goal.width:.3f} m')
        deadline = time.monotonic() + 10.0
        future = client.send_goal_async(goal)
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or not future.result().accepted:
            raise FreeDriveAbort('gripper open goal was not accepted')
        result_future = future.result().get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not result_future.done():
            raise FreeDriveAbort('gripper open did not finish within 10 s')
        result = result_future.result().result
        if not result.success:
            raise FreeDriveAbort(f'gripper open failed: {result.error}')
        self.get_logger().info('Gripper open')

    def start_floating(self):
        try:
            params_file = (self.p('free_controller_params_file')
                           or container_config_path('tpgpt_teaching_controller.yaml'))
        except ValueError as exc:
            raise FreeDriveAbort(f'{exc} (free_controller_params_file)')
        name = self.p('free_controller')
        controllers = self.cmc.ensure_loaded(name, self.p('free_controller_type'), params_file)
        if self.p('hold_controller') not in controllers:
            raise FreeDriveAbort(f"{self.p('hold_controller')} is not loaded, so the arm could "
                                 'not be held afterwards; refusing to float it')
        if self.p('require_confirmation'):
            print(f'\nAbout to ACTIVATE {name}: ZERO Cartesian stiffness, the arm FLOATS.'
                  '\nSupport the arm and keep the emergency stop within reach.')
            if input("Type 'yes' to continue: ").strip().lower() != 'yes':
                self.get_logger().info('Cancelled; nothing was switched.')
                return False
        self.cmc.activate(name, controllers)
        self.floating = True
        keys = ("'c'+Enter closes the hand, 'o'+Enter opens it, " if self.gripper_type == 'inspire'
                else '')
        self.get_logger().info(f'{name} ACTIVE: free drive. {keys}Enter alone or Ctrl-C holds.')
        return True

    def hold(self):
        name = self.p('hold_controller')
        self.cmc.activate(name)
        self.floating = False
        self.get_logger().info(f'{name} ACTIVE: holding the pose the arm was in at the switch')

    def run(self):
        self.get_logger().info('Waiting for /current_pose ...')
        self.wait_for_pose()
        self.open_gripper()
        if not self.start_floating():
            return
        while rclpy.ok():
            if not self.pose_fresh():
                self.get_logger().error('pose went stale while floating; switching to hold')
                break
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()
                if key in ('c', 'o'):
                    if self.gripper_type == 'inspire':
                        self.command_inspire(key == 'c')
                    else:
                        self.get_logger().warn('c/o drive the Inspire hand only '
                                               '(gripper_type inspire)')
                    continue
                if key:
                    self.get_logger().warn(f"unknown key {key!r}: c = close, o = open, "
                                           'Enter alone = hold')
                    continue
                break
            if not sys.stdin.isatty():
                time.sleep(0.1)
        self.hold()


def main(args=None):
    # No rclpy signal handler, so Ctrl-C leaves the context usable for the switch to
    # hold. SIGTERM takes the same path instead of killing the node with the arm floating.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = FreeDrive()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    exit_code = 0
    try:
        node.run()
    except FreeDriveAbort as exc:
        node.get_logger().error(f'Aborted: {exc}')
        exit_code = 1
    except KeyboardInterrupt:
        pass
    finally:
        if node.floating:
            try:
                node.hold()
            except Exception as exc:  # noqa: BLE001
                node.get_logger().error(f'could not switch to the hold controller: {exc}; '
                                        'use the e-stop')
                exit_code = 1
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
