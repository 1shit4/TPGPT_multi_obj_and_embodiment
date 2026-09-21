#!/usr/bin/env python3
"""
ROS 2 driver for the Inspire RH56 hand: the fixed open/close grasp as services.

    /inspire_hand/open         std_srvs/Trigger  fingers + thumb bend straight, then the
                                                 thumb swings into opposition
    /inspire_hand/close        std_srvs/Trigger  fingers wrap, then the thumb bends; every
                                                 DOF stops on its force threshold
    /inspire_hand/joint_states sensor_msgs/JointState, position = closure per DOF
                               (0 straight .. 1 fully bent), effort = measured force (g)
    /inspire_hand/grasp_state  std_msgs/String (latched): open | closed | unknown

Both services block until the motion has finished (every DOF stopped, or the per-step
timeout) and answer success=false with the reason on a serial error or an actuator
error flag. The close answer names the DOFs that stopped on contact.

The grasp, speed, force, serial port and hand ID come from config/inspire_hand.yaml
(parameter `config_file`); `port`, `hand_id`, `speed` and `force_g` override it.
`dry_run:=true` opens no serial port and answers every call with success, for testing
the rest of the pipeline without the hand.

    ros2 run ycb_pointcloud_pipeline inspire_hand_node
    ros2 service call /inspire_hand/close std_srvs/srv/Trigger
"""

import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ycb_pointcloud_pipeline import inspire_hand as ih


class InspireHandNode(Node):
    def __init__(self, **kwargs):
        super().__init__('inspire_hand', **kwargs)
        d = self.declare_parameter
        d('config_file', '')          # '' -> installed config/inspire_hand.yaml
        d('port', '')                 # '' -> config serial.port
        d('hand_id', -1)              # -1 -> config serial.hand_id
        d('baud', -1)
        d('speed', -1)
        d('force_g', -1)
        d('dry_run', False)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        cfg = ih.load_config(p('config_file') or None)
        serial, grasp = cfg.get('serial', {}), cfg.get('grasp', {})
        self.port = p('port') or serial.get('port', ih.DEFAULT_PORT)
        self.hand_id = p('hand_id') if p('hand_id') >= 0 else int(serial.get('hand_id', ih.DEFAULT_ID))
        self.baud = p('baud') if p('baud') > 0 else int(serial.get('baud', 115200))
        self.speed = p('speed') if p('speed') >= 0 else int(grasp.get('speed', ih.DEFAULT_SPEED))
        self.force = p('force_g') if p('force_g') >= 0 else int(grasp.get('force_g', ih.DEFAULT_FORCE))
        self.step_timeout = float(grasp.get('step_timeout_s', 3.0))
        self.dry = bool(p('dry_run'))

        log = self.get_logger()
        self.hand = ih.InspireHand(self.port, self.hand_id, self.baud, dry_run=self.dry,
                                   log=lambda text: log.debug(text))
        self.lock = threading.Lock()
        if not self.dry:
            st = self.hand.status()      # fail at startup, not at the first grasp
            log.info(f'Inspire hand on {self.port} (ID {self.hand_id}, {self.baud} baud): '
                     f'angles {st["angle"]}, errors {ih.error_text(st) or "none"}')
        else:
            log.warn('dry_run: no serial port opened; open/close only log')
        log.info(f'grasp speed {self.speed}, force {self.force} g, config {cfg["_path"]}')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.state_pub = self.create_publisher(String, '~/grasp_state', latched)
        self.joint_pub = self.create_publisher(JointState, '~/joint_states', 10)
        self.state_pub.publish(String(data='unknown'))
        services = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, '~/open', self._open, callback_group=services)
        self.create_service(Trigger, '~/close', self._close, callback_group=services)
        rate = float(grasp.get('state_rate_hz', 10.0))
        if rate > 0 and not self.dry:
            self.create_timer(1.0 / rate, self._publish_joints,
                              callback_group=ReentrantCallbackGroup())

    # ------------------------------------------------------------------ services
    def _run(self, name, fn, response):
        log = self.get_logger()
        with self.lock:
            try:
                st = fn(self.speed, self.force, self.step_timeout, verbose=False)
            except IOError as exc:
                response.success, response.message = False, f'{name} failed: {exc}'
                log.error(response.message)
                self.state_pub.publish(String(data='unknown'))
                return response, None
        errors = ih.error_text(st)
        if errors:
            response.success, response.message = False, f'{name}: actuator error ({errors})'
            log.error(response.message)
            return response, st
        response.success = True
        return response, st

    def _open(self, request, response):
        response, st = self._run('open', self.hand.grasp_open, response)
        if response.success:
            response.message = 'open'
            self.state_pub.publish(String(data='open'))
            self.get_logger().info('hand OPEN (thumb opposed)')
        if st:
            self._publish_joints(st)
        return response

    def _close(self, request, response):
        response, st = self._run('close', self.hand.grasp_close, response)
        if response.success:
            held = ih.contact_dofs(st)
            response.message = ('closed; contact on ' + ', '.join(held) if held
                                else 'closed; no contact (every finger closed fully)')
            self.state_pub.publish(String(data='closed'))
            self.get_logger().info(f'hand CLOSED: {response.message}')
        if st:
            self._publish_joints(st)
        return response

    # ------------------------------------------------------------------ state
    def _publish_joints(self, st=None):
        if st is None:
            if not self.lock.acquire(blocking=False):
                return                   # a grasp is using the serial line
            try:
                st = self.hand.status()
            except IOError as exc:
                self.get_logger().warn(f'status read failed: {exc}', throttle_duration_sec=5.0)
                return
            finally:
                self.lock.release()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'inspire_{n}' for n in ih.DOF_NAMES]
        msg.position = ih.closure(st)
        msg.effort = [float(f) for f in st['force_g']]
        self.joint_pub.publish(msg)

    def destroy_node(self):
        self.hand.close_port()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InspireHandNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
