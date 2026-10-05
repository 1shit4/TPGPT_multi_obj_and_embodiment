#!/usr/bin/env python3
"""
ROS 2 driver for the Inspire RH56DFX-2R hand: the open/close grasp as services.

    /inspire_hand/open         std_srvs/Trigger  right after a close that held something:
                                                 everything at once (the release); else the
                                                 thumb lifts, the fingers open, the thumb
                                                 swings into opposition
    /inspire_hand/close        std_srvs/Trigger  close (``close_mode``); every bending DOF
                                                 stops on its force threshold. ``message``
                                                 is the CloseReport as JSON (contact = the
                                                 DOFs stopped short of the close, on air,
                                                 timed out, errors, angles, forces).
    /inspire_hand/joint_states sensor_msgs/JointState: the six driven URDF joints of
                               GraspGen-X's inspire_hand description, position in rad
                               (config joint_map), effort = measured force in GRAMS.
                               Published while idle (state_rate_hz) and at every status
                               read during a motion.
    /inspire_hand/registers    std_msgs/Int16MultiArray: ANGLE_ACT x6, FORCE_ACT x6,
                               STATUS x6, in DOF order (little, ring, middle, index,
                               thumb_bend, thumb_rot) -- the raw values, for the record.
    /inspire_hand/point        std_srvs/Trigger  the pointing pose for measuring the mount
                                                 (index straight, others curled, thumb out)
    /inspire_hand/grasp_state  std_msgs/String (latched): open | closed | pointing | unknown

Both services block until the motion has finished. ``success`` is false on a serial
error, an actuator error flag, or a DOF still moving at the step timeout. Closing on
air is NOT a failure of the service: the report says so (``opposed`` false) and the
caller decides.

Parameters read at every close, settable while the node runs (``ros2 param set``, or
the executor's per-object force through rcl_interfaces/SetParameters):
    force_g          per-DOF stop threshold, grams, 0-1000 (one value for all six)
    force_g_per_dof  six values in DOF order; empty = force_g for all
    close_mode       simultaneous (the default: it holds) | sequential (pushes the object)

The speed, serial port and hand ID come from config/inspire_hand.yaml (parameter
`config_file`); `port`, `hand_id`, `baud`, `speed`, `force_g` and `close_mode`
override it at start. `dry_run:=true` opens no serial port and answers every call
with success, for testing the rest of the pipeline without the hand.

    ros2 run ycb_pointcloud_pipeline inspire_hand_node
    ros2 service call /inspire_hand/close std_srvs/srv/Trigger
"""

import threading

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Int16MultiArray, String
from std_srvs.srv import Trigger

from ycb_pointcloud_pipeline import inspire_hand as ih


def check_force(value, per_dof) -> str | None:
    """Why a force setting is invalid, or None."""
    if not 0 <= int(value) <= 1000:
        return f'force_g {value} is outside 0-1000 g'
    per = list(per_dof or [])
    if per and (len(per) != 6 or not all(0 <= int(v) <= 1000 for v in per)):
        return f'force_g_per_dof must be empty or six values 0-1000 g, not {per}'
    return None


def joint_state_message(st: dict, joint_map: dict, stamp) -> JointState:
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(ih.URDF_JOINTS)
    msg.position = ih.register_to_joint(st['angle'], joint_map)
    msg.effort = [float(f) for f in st['force_g']]
    return msg


class InspireHandNode(Node):
    def __init__(self, **kwargs):
        super().__init__('inspire_hand', **kwargs)
        d = self.declare_parameter
        d('config_file', '')          # '' -> installed config/inspire_hand.yaml
        d('port', '')                 # '' -> config serial.port
        d('hand_id', -1)              # -1 -> config serial.hand_id
        d('baud', -1)
        d('speed', -1)
        d('force_g', -1)              # -1 -> config grasp.force_g
        d('force_g_per_dof', [0, 0, 0, 0, 0, 0])   # all zero -> force_g for every DOF
        d('close_mode', '')           # '' -> config grasp.close_mode
        d('dry_run', False)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        cfg = ih.load_config(p('config_file') or None)
        serial, grasp = cfg.get('serial', {}), cfg.get('grasp', {})
        self.joint_map = ih.joint_map_from_config(cfg)
        self.port = p('port') or serial.get('port', ih.DEFAULT_PORT)
        self.hand_id = p('hand_id') if p('hand_id') >= 0 else int(serial.get('hand_id', ih.DEFAULT_ID))
        self.baud = p('baud') if p('baud') > 0 else int(serial.get('baud', 115200))
        self.speed = p('speed') if p('speed') >= 0 else int(grasp.get('speed', ih.DEFAULT_SPEED))
        self.step_timeout = float(grasp.get('step_timeout_s', 3.0))
        self.dry = bool(p('dry_run'))
        # Start-up values for the settable parameters, then validation on every change.
        force = p('force_g') if p('force_g') >= 0 else int(grasp.get('force_g', ih.DEFAULT_FORCE))
        mode = p('close_mode') or str(grasp.get('close_mode', ih.DEFAULT_CLOSE_MODE))
        bad = check_force(force, self._per_dof(p('force_g_per_dof')))
        if bad or mode not in ih.CLOSE_MODES:
            raise ValueError(bad or f'close_mode must be one of {ih.CLOSE_MODES}, not {mode!r}')
        self.set_parameters([Parameter('force_g', value=int(force)),
                             Parameter('close_mode', value=mode)])
        self.add_on_set_parameters_callback(self._on_parameters)

        log = self.get_logger()
        self.hand = ih.InspireHand(self.port, self.hand_id, self.baud, dry_run=self.dry,
                                   log=lambda text: log.debug(text))
        self.lock = threading.Lock()
        if not self.dry:
            st = self.hand.status()      # fail at startup, not at the first grasp
            log.info(f'Inspire hand on {self.port} (ID {self.hand_id}, {self.baud} baud): '
                     f'angles {st["angle"]}, errors {ih.error_text(st) or "none"}')
        else:
            log.warning('dry_run: no serial port opened; open/close only log')
        unverified = [n for n, (_, _, ok) in self.joint_map.items() if not ok]
        if unverified:
            log.warning(f'joint_map not yet measured for {", ".join(unverified)}: '
                        'joint_states angles are the URDF limits, not the hand (plan M4)')
        log.info(f'grasp speed {self.speed}, force {force} g, close {mode}, '
                 f'config {cfg["_path"]}')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.state_pub = self.create_publisher(String, '~/grasp_state', latched)
        self.joint_pub = self.create_publisher(JointState, '~/joint_states', 10)
        self.register_pub = self.create_publisher(Int16MultiArray, '~/registers', 10)
        self.state_pub.publish(String(data='unknown'))
        services = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, '~/open', self._open, callback_group=services)
        self.create_service(Trigger, '~/close', self._close, callback_group=services)
        self.create_service(Trigger, '~/point', self._point, callback_group=services)
        rate = float(grasp.get('state_rate_hz', 10.0))
        if rate > 0 and not self.dry:
            self.create_timer(1.0 / rate, self._poll, callback_group=ReentrantCallbackGroup())

    # ---------------------------------------------------------------- parameters
    @staticmethod
    def _per_dof(values):
        values = [int(v) for v in (values or [])]
        return [] if not any(values) else values

    def _on_parameters(self, params):
        force = self.get_parameter('force_g').value
        per = self._per_dof(self.get_parameter('force_g_per_dof').value)
        for prm in params:
            if prm.name == 'force_g':
                force = prm.value
            elif prm.name == 'force_g_per_dof':
                per = self._per_dof(prm.value)
            elif prm.name == 'close_mode' and prm.value not in ih.CLOSE_MODES:
                return SetParametersResult(
                    successful=False, reason=f'close_mode must be one of {ih.CLOSE_MODES}')
        bad = check_force(force, per)
        return SetParametersResult(successful=bad is None, reason=bad or '')

    def _force(self):
        per = self._per_dof(self.get_parameter('force_g_per_dof').value)
        return per if per else int(self.get_parameter('force_g').value)

    # ------------------------------------------------------------------ services
    def _open(self, request, response):
        log = self.get_logger()
        with self.lock:
            try:
                st = self.hand.grasp_open(self.speed, self._force(), self.step_timeout,
                                          verbose=False, on_status=self._publish)
            except IOError as exc:
                response.success, response.message = False, f'open failed: {exc}'
                log.error(response.message)
                self.state_pub.publish(String(data='unknown'))
                return response
        problems = []
        if st and st.get('timed_out'):
            problems.append(f'still moving at the timeout: {", ".join(st["timed_out"])}')
        if ih.error_text(st):
            problems.append(f'actuator error ({ih.error_text(st)})')
        if problems:
            response.success, response.message = False, 'open: ' + '; '.join(problems)
            log.error(response.message)
            self.state_pub.publish(String(data='unknown'))
            return response
        response.success, response.message = True, 'open'
        self.state_pub.publish(String(data='open'))
        log.info('hand OPEN (thumb opposed)')
        return response

    def _point(self, request, response):
        with self.lock:
            try:
                st = self.hand.point(self.speed, self._force(), self.step_timeout,
                                     verbose=False, on_status=self._publish)
            except IOError as exc:
                response.success, response.message = False, f'point failed: {exc}'
                self.get_logger().error(response.message)
                self.state_pub.publish(String(data='unknown'))
                return response
        problem = ((st or {}).get('timed_out') or []) or ih.error_text(st)
        response.success = not problem
        response.message = 'pointing' if not problem else f'point: {problem}'
        self.state_pub.publish(String(data='pointing' if not problem else 'unknown'))
        return response

    def _close(self, request, response):
        log = self.get_logger()
        mode = str(self.get_parameter('close_mode').value)
        with self.lock:
            try:
                report = self.hand.close(mode, self.speed, self._force(), self.step_timeout,
                                         verbose=False, on_status=self._publish)
            except IOError as exc:
                response.success, response.message = False, f'close failed: {exc}'
                log.error(response.message)
                self.state_pub.publish(String(data='unknown'))
                return response
        response.success = self.dry or report.ok
        response.message = report.to_json()
        if not response.success:
            log.error(f'hand CLOSE failed: {report.summary()}')
            self.state_pub.publish(String(data='unknown'))
            return response
        self.state_pub.publish(String(data='closed'))
        if report.opposed or self.dry:
            log.info(f'hand CLOSED: {report.summary()}')
        else:
            log.warning(f'hand CLOSED WITHOUT OPPOSITION: {report.summary()}')
        return response

    # ------------------------------------------------------------------ state
    def _poll(self):
        if not self.lock.acquire(blocking=False):
            return                   # a grasp is using the serial line and publishing
        try:
            st = self.hand.status()
        except IOError as exc:
            self.get_logger().warning(f'status read failed: {exc}', throttle_duration_sec=5.0)
            return
        finally:
            self.lock.release()
        self._publish(st)

    def _publish(self, st):
        if not st:
            return
        self.joint_pub.publish(joint_state_message(st, self.joint_map,
                                                   self.get_clock().now().to_msg()))
        msg = Int16MultiArray()
        msg.data = [int(v) for v in list(st['angle']) + list(st['force_g'])
                    + list(st['status'])]
        self.register_pub.publish(msg)

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
