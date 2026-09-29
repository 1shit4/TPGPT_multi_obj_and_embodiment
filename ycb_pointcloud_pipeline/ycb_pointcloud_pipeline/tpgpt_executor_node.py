#!/usr/bin/env python3
"""
Execute a TPGPT plan on the FR3 through CRISP's Cartesian impedance controller.

The control commands come from TPGPT's own execution loop
(tpgpt.hardware.executor.HardwareExecutor = the per-step body of
tpgpt.sim.rollout.rollout_policy, built from gate_step / advance_step /
advance_phase / StallWatchdog). Each step:

    /current_pose (fr3_hand_tcp, base) --> HardwareExecutor.step()
        --> PoseStamped attractor        --> <controller>/target_pose
        --> [optional] 6 stiffnesses     --> <controller>/target_stiffness
        --> gripper open/close edges     --> /franka_gripper/{grasp,move}
                                              or /inspire_hand/{open,close} (gripper_mode inspire)

Stiffness (parameter `use_transported_stiffness`)
  false (default): controller `cartesian_impedance_controller` as spawned by
         fr3_crisp.launch.py, with its configured gains; poses only, on /target_pose.
  true : controller `tpgpt_cartesian_impedance_controller`, loaded at runtime from
         config/tpgpt_variable_stiffness_controller.yaml (variable_stiffness
         enabled, damping 2*sqrt(K)). The transported 3x3 K is rotated into the
         measured TCP frame and its diagonal sent (CRISP's error is computed in the
         end-effector frame); rotational stiffness follows rollout_policy's
         firm/compliant schedule. The dropped off-diagonal share is logged.

Modes
  dry_run:=true (DEFAULT) -- nothing is activated, nothing is sent to the
         controller or the gripper. The same loop runs against a simulated
         first-order impedance arm starting at the real /current_pose (or the plan
         start if none), and publishes /tpgpt/executor/commanded_pose,
         /tpgpt/executor/simulated_pose and /tpgpt/executor/markers for RViz.
  dry_run:=false -- supervised hardware run: preflight checks, a typed 'yes',
         a strict controller switch (CRISP seeds its target with the measured pose,
         so activation does not move the arm), open gripper, approach at bounded
         speed to the plan start, then the policy.

Hand (from the plan's metadata['end_effector'], written by tpgpt_planner_node)
  The plan's positions are the hand's grasp point (the Franka Hand's fingertip centre,
  or the Inspire hand's palm point) and its orientations are the controller frame
  (fr3_hand_tcp). The hand's tool_offset -- grasp point in controller-frame coordinates --
  is taken from the plan, and the `tool_offset` parameter is ADDED to it (a correction,
  default zero). A plan made for one hand refuses to run with the other hand's
  gripper_mode (inspire <-> franka_grasp / gripper_command).

Aborts hold the arm at its measured pose (the gripper is left as it is, so a held
object is not dropped): stale pose, tracking error, arm not following its target
(SafetyMonitor), commanded pose outside the workspace box, stalled watchdog,
Ctrl-C.

Run with the TPGPT environment and the robot container up (./scripts/run.sh robot):
    source ~/venvs/tpgpt/bin/activate
    ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
        -p plan_dir:=$HOME/tpgpt_plans/<plan> -p dry_run:=true
"""

import csv
import dataclasses
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray

from ycb_pointcloud_pipeline.ros_utils import (
    ControllerManagerClient, axes_markers, container_config_path, line_strip_marker,
    matrix_to_pose_stamped, other_publishers, pose_to_matrix,
)


class AbortRun(RuntimeError):
    pass


class TpgptExecutorNode(Node):
    def __init__(self, **kwargs):
        super().__init__('tpgpt_executor_node', **kwargs)
        d = self.declare_parameter
        d('plan_dir', '')
        d('dry_run', True)
        d('dry_run_realtime', True)
        d('dry_run_resting_error', [0.0, 0.0, 0.0])  # m; simulated arm rests this far from its target
        d('use_transported_stiffness', False)
        # Fixed translational stiffness (N/m) sent to the variable-stiffness controller
        # for the whole run (approach, settle, policy, arrival), when the transported
        # stiffness is not used. At the stock 400 N/m the FR3 trailed its target by a
        # steady 20-30 mm along the path at 1-3 cm/s (joint friction ~10 N at the TCP,
        # independent of speed: the slowest run lagged most), which shut the lag gate,
        # crawled the phase and aborted at arrive_grasp. Lag scales with 1/K.
        # <= 0: previous behaviour (stock controller with its configured gains).
        # 800 (with rotational 60) made the arm vibrate on the approach and at the
        # grasp (2026-09-15, banana_20260915_110456: 1-1.5 mm jitter under a constant
        # command, never above 0.54 mm at 400/30); 600 trades some lag back for that.
        d('translational_stiffness', 400.0)
        d('controller_manager', '/controller_manager')
        d('controller_name', 'cartesian_impedance_controller')
        d('stiffness_controller_name', 'tpgpt_cartesian_impedance_controller')
        d('stiffness_controller_type', 'crisp_controllers/CartesianController')
        d('stiffness_controller_params_file', '')   # path INSIDE the robot container
        d('target_pose_topic', '')                  # '' -> derived from the controller
        d('target_stiffness_topic', '')
        d('current_pose_topic', '/current_pose')
        # Measured joints seed the infeasible-pose fallback's IK (executor.C3); without
        # fresh ones the fallback is off for that step and the log says how often.
        d('joint_states_topic', '/joint_states')
        d('base_frame', 'base')
        # Execution (tpgpt.hardware.executor.ExecutorSettings)
        d('speed_scale', 0.35)
        d('lag_tolerance', 0.045)
        d('speed_limit_factor', 3.0)
        d('belief_correction', 0.2)
        d('rotational_stiffness', 30.0)            # stock CRISP k_rot
        d('rotational_damping', 12.0)
        # < 0 = auto: 0.25 for a demonstrated (simulated) source, 1.0 when the source
        # stiffness is static (taught by hand). With one stiffness throughout, the
        # firm/compliant split compares equal numbers and the rotational gain flipped
        # between 60 and 15 on 64 of 427 steps of a dry run.
        d('transit_rotational_scale', -1.0)
        d('tool_offset', [0.0, 0.0, 0.0])          # m, controller frame, ADDED to the plan hand's offset
        # The policy's own length is 810 labels / 20 Hz / speed_scale = 116 s for the taught
        # demonstration at 0.35, so 120 s aborted a dry run that was running normally, and any
        # stall handling pushed it over. It is a runaway guard, not a schedule: keep it well
        # above the plan's expected duration, which the executor prints before it starts.
        d('max_policy_seconds', 300.0)
        # What to do when the execution watchdog declares the arm stuck (the lag gate
        # held the phase for its whole patience). 'continue': warn, switch the lag gate
        # and its clamp off for the rest of the run, so the task clock runs at the
        # plan's speed and the attractor follows the plan even if the arm lags;
        # max_tracking_error and the progress check still abort. 'abort': stop and hold.
        d('on_stall', 'continue')                   # continue | abort
        # Stiffness limits (must not exceed the controller's variable_max_stiffness)
        d('stiffness_min_translational', 50.0)
        d('stiffness_max_translational', 900.0)
        d('stiffness_max_rotational', 60.0)
        # Form of the target_stiffness message. 'full': the transported K rotated into the hand
        # frame, all of it (12 values; needs the patched CRISP, crisp-full-stiffness.patch in
        # ~/THOR_DOCKER_SETUP, image docker-thor:local since 2026-09-29). 'diagonal': its
        # diagonal only (6 values, any CRISP; the off-diagonal part is dropped and logged).
        # 'auto': full with use_transported_stiffness, else diagonal -- a fixed stiffness is
        # isotropic, so the diagonal is exact and the message works on either image. The
        # pre-patch image IGNORES a 12-value message (it keeps its yaml gains), so before
        # activating anything the executor reads the controller's stiffness-topic description
        # and refuses to run 'full' against it: pass 'diagonal' after rolling the image back.
        d('stiffness_message', 'auto')              # auto | full | diagonal
        # Safety
        # A box the commanded TCP may not leave. **Off by default since 2026-09-23** (the user's
        # call): it is a blunt guard from before the path check existed, and it refused poses the
        # arm reaches perfectly well -- the transported start of a demonstration lands near the
        # base, and the retreat ends at the box's ceiling. What it used to cover is now covered
        # by tpgpt.hardware.path_check, which tests every waypoint and the move onto the path for
        # reachability, self-collision, the table, the shelf model and the seen points. Set
        # workspace_check:=true to bring it back; the box itself is unchanged.
        d('workspace_check', False)
        d('workspace_min', [0.15, -0.6, -0.25])     # base frame box for commanded TCP, m
        d('workspace_max', [0.85, 0.6, 0.7])
        d('max_start_distance', 0.6)                # m, current TCP to plan start
        d('approach_max_linear_speed', 0.05)
        d('approach_max_angular_speed', 0.3)
        d('approach_settle_tolerance', 0.2)
        d('approach_settle_timeout', 5.0)
        d('max_tracking_error', 0.08)
        d('progress_window', 40)
        d('min_progress_ratio', 0.3)
        d('max_pose_age', 0.2)
        d('allow_placeholder_calibration', False)
        # Tracking corrections (hardware only; see _settle_and_measure_offset / _arrive).
        # The stock impedance controller rests 8-15 mm above its target in z (measured
        # on every logged run, top-down and GraspGen alike), so the gripper closed that
        # far above the planned grasp.
        d('settle_time', 1.5)                       # s held at the plan start before measuring
        d('offset_compensation', 'z')               # none | z | xyz: subtract the resting error
        d('max_offset_compensation', 0.03)          # m, bound on any compensation
        d('arrival_hold', False)                    # true: servo onto the grasp/release before the gripper acts
        d('arrival_tolerance', 0.004)               # m, measured TCP to the plan's grasp/release
        d('arrival_timeout', 4.0)                   # s
        d('arrival_gain', 1.5)                      # 1/s, integral gain of the hold
        d('arrival_max_gap', 0.04)                  # m, abort if the command strayed this far
        d('require_confirmation', True)
        # Gripper
        d('gripper_mode', 'franka_grasp')           # franka_grasp | gripper_command | inspire | none
        d('gripper_namespace', '/franka_gripper')   # /inspire_hand for gripper_mode inspire
        d('gripper_timeout', 20.0)                  # s, inspire open/close service answer
        d('open_width', 0.08)
        d('grasp_speed', 0.05)
        d('grasp_force', 20.0)
        d('log_dir', '')

        p = self.p
        if not p('plan_dir'):
            raise SystemExit('Set -p plan_dir:=<directory written by tpgpt_planner_node>')
        self.dry = bool(p('dry_run'))
        self.use_stiffness = bool(p('use_transported_stiffness'))
        self.fixed_stiffness = (0.0 if self.use_stiffness
                                else float(p('translational_stiffness')))
        if self.fixed_stiffness > float(p('stiffness_max_translational')):
            raise SystemExit(f'translational_stiffness {self.fixed_stiffness} exceeds '
                             f"stiffness_max_translational {p('stiffness_max_translational')}")
        form = str(p('stiffness_message'))
        if form not in ('auto', 'full', 'diagonal'):
            raise SystemExit(f"stiffness_message must be auto, full or diagonal, not {form!r}")
        self.full_stiffness = form == 'full' or (form == 'auto' and self.use_stiffness)
        # Either stiffness mode needs the variable-stiffness controller.
        self.variable_stiffness = self.use_stiffness or self.fixed_stiffness > 0
        self.cm = p('controller_manager').rstrip('/')
        self.controller = (p('stiffness_controller_name') if self.variable_stiffness
                           else p('controller_name'))
        self.pose_topic = p('target_pose_topic') or (
            f'/{self.controller}/target_pose' if self.variable_stiffness else '/target_pose')
        self.stiffness_topic = p('target_stiffness_topic') or f'/{self.controller}/target_stiffness'
        self.base = p('base_frame')

        self._lock = threading.Lock()
        self._pose = None     # (monotonic, PoseStamped) -- converted on demand
        self.moving = False
        self.gripper_state = None
        self.sim_T = None     # dry-run simulated TCP pose
        self.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 1)
        self._joints = None   # (monotonic, JointState) -- converted on demand
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, p('joint_states_topic'),
                                 lambda m: setattr(self, '_joints', (time.monotonic(), m)), 1)

        if self.dry:
            self.pose_pub = self.create_publisher(PoseStamped, '/tpgpt/executor/commanded_pose', 1)
            # What would go to the controller, on a topic no controller listens to.
            self.stiffness_pub = (
                self.create_publisher(Float64MultiArray, '/tpgpt/executor/commanded_stiffness', 1)
                if self.variable_stiffness else None)
        else:
            self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 1)
            self.stiffness_pub = (self.create_publisher(Float64MultiArray, self.stiffness_topic, 1)
                                  if self.variable_stiffness else None)
        self.sim_pub = self.create_publisher(PoseStamped, '/tpgpt/executor/simulated_pose', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/tpgpt/executor/markers', 1)
        self.trace_pub = self.create_publisher(Path, '/tpgpt/executor/executed_path', 1)
        self.cmc = ControllerManagerClient(self, self.cm, error=AbortRun)
        self.gripper = None
        # X6 trial record hooks. step_hook(position, rotation) is called after every
        # command sent; stage / last_cmd / policy_fit_seconds describe that step.
        # None by default: nothing is recorded and nothing changes.
        self.step_hook = None
        self.stage = 'init'
        self.last_cmd = None
        self.policy_fit_seconds = None
        self.policy_executor = None

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ inputs
    def _pose_cb(self, msg):
        # pose_broadcaster publishes at the controller rate (up to 1 kHz). Anything
        # heavier than storing the message here starves the control thread of the
        # GIL: measured on fake hardware, a matrix conversion per message stalled
        # the 20 Hz loop for 1.8 s. The conversion happens in measured() instead.
        self._pose = (time.monotonic(), msg)

    def measured(self, max_age=None):
        if self.dry:
            if self.sim_T is None:
                raise AbortRun('dry run has no simulated pose yet')
            return self.sim_T[:3, 3].copy(), self.sim_T[:3, :3].copy()
        max_age = self.p('max_pose_age') if max_age is None else max_age
        pose = self._pose
        if pose is None or time.monotonic() - pose[0] > max_age:
            raise AbortRun(f"no fresh pose on {self.p('current_pose_topic')} "
                           f'(older than {max_age} s); is pose_broadcaster active?')
        if pose[1].header.frame_id not in ('', self.base):
            raise AbortRun(f'{self.p("current_pose_topic")} is in frame '
                           f'{pose[1].header.frame_id!r}, expected {self.base!r}')
        T = pose_to_matrix(pose[1].pose)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def measured_joints(self):
        """The arm's seven joint angles, or None (dry run, or nothing fresh)."""
        if self.dry or self._joints is None:
            return None
        t, msg = self._joints
        if time.monotonic() - t > self.p('max_pose_age'):
            return None
        index = {n: i for i, n in enumerate(msg.name)}
        names = [f'fr3_joint{k}' for k in range(1, 8)]
        if not all(n in index for n in names):
            return None
        return np.array([msg.position[index[n]] for n in names], dtype=float)

    def wait_for_pose(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pose = self._pose
            if pose is not None:
                return pose_to_matrix(pose[1].pose)
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------ controller
    def ensure_controller(self):
        params_file = None
        if self.variable_stiffness:
            try:
                params_file = (self.p('stiffness_controller_params_file') or
                               container_config_path('tpgpt_variable_stiffness_controller.yaml'))
            except ValueError as exc:
                raise AbortRun(f'{exc} (stiffness_controller_params_file)')
        controllers = self.cmc.ensure_loaded(self.controller, self.p('stiffness_controller_type'),
                                             params_file,
                                             missing_hint='fr3_crisp.launch.py spawns it inactive')
        if self.full_stiffness:
            # before anything is activated: the pre-patch CRISP would silently ignore 12 values
            self.cmc.require_full_stiffness(self.controller)
        return controllers

    # ------------------------------------------------------------------ gripper
    def setup_gripper(self):
        mode = self.p('gripper_mode')
        ns = self.p('gripper_namespace').rstrip('/')
        if self.dry or mode == 'none':
            return
        if mode == 'franka_grasp':
            try:
                from franka_msgs.action import Grasp, Move
            except ImportError:
                raise AbortRun('gripper_mode franka_grasp needs franka_msgs on the host (see '
                               'THOR_DOCKER_SETUP README), or set gripper_mode:=none')
            self.gripper = {'grasp': ActionClient(self, Grasp, f'{ns}/grasp'),
                            'move': ActionClient(self, Move, f'{ns}/move'),
                            'Grasp': Grasp, 'Move': Move}
            for name in ('grasp', 'move'):
                if not self.gripper[name].wait_for_server(timeout_sec=5.0):
                    raise AbortRun(f'gripper action {ns}/{name} not available')
        elif mode == 'inspire':
            from ycb_pointcloud_pipeline.ros_utils import InspireHandClient
            client = InspireHandClient(self, ns if ns != '/franka_gripper' else '/inspire_hand')
            missing = client.available(timeout=5.0)
            if missing:
                raise AbortRun(f'Inspire hand services {missing} not available; start '
                               'ros2 run ycb_pointcloud_pipeline inspire_hand_node')
            self.gripper = {'inspire': client}
        elif mode == 'gripper_command':
            try:
                from control_msgs.action import GripperCommand
            except ImportError:
                raise AbortRun('gripper_mode gripper_command needs control_msgs on the host')
            self.gripper = {'command': ActionClient(self, GripperCommand, f'{ns}/gripper_action'),
                            'GripperCommand': GripperCommand}
            if not self.gripper['command'].wait_for_server(timeout_sec=5.0):
                raise AbortRun(f'gripper action {ns}/gripper_action not available')
        else:
            raise AbortRun(f'unknown gripper_mode {mode!r}')

    def command_gripper(self, close, object_width, wait=False):
        """Send an open/close goal. Non-blocking unless `wait`."""
        self.gripper_state = close
        label = 'CLOSE' if close else 'OPEN'
        if self.gripper is None:
            self.get_logger().info(f'gripper {label} (not sent: '
                                   f"{'dry run' if self.dry else 'gripper_mode none'})")
            return True
        if 'inspire' in self.gripper:
            client = self.gripper['inspire']
            future = client.send(close)
            self.get_logger().info(f'gripper {label} sent (Inspire hand)')
            if wait:
                ok, message = client.wait(future, float(self.p('gripper_timeout')))
                if not ok:
                    raise AbortRun(f'Inspire hand {label} failed: {message}')
                self.get_logger().info(f'Inspire hand: {message}')
            return True
        open_w = float(self.p('open_width'))
        if 'grasp' in self.gripper:
            if close:
                goal = self.gripper['Grasp'].Goal()
                # Success for any final width in [0, open_width]: the cloud only
                # gives a lower bound on the object's width.
                goal.width = float(np.clip(object_width, 0.0, open_w))
                goal.epsilon.inner = goal.width
                goal.epsilon.outer = open_w - goal.width
                goal.speed = float(self.p('grasp_speed'))
                goal.force = float(self.p('grasp_force'))
                future = self.gripper['grasp'].send_goal_async(goal)
            else:
                goal = self.gripper['Move'].Goal()
                goal.width, goal.speed = open_w, float(self.p('grasp_speed'))
                future = self.gripper['move'].send_goal_async(goal)
        else:
            goal = self.gripper['GripperCommand'].Goal()
            goal.command.position = 0.0 if close else open_w / 2.0
            goal.command.max_effort = float(self.p('grasp_force'))
            future = self.gripper['command'].send_goal_async(goal)
        self.get_logger().info(f'gripper {label} sent')
        if wait:
            deadline = time.monotonic() + 10.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not future.done() or not future.result().accepted:
                raise AbortRun(f'gripper {label} goal was not accepted')
            result_future = future.result().get_result_async()
            while not result_future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not result_future.done():
                raise AbortRun(f'gripper {label} did not finish in 10 s')
        return True

    # ------------------------------------------------------------------ output
    def publish(self, position, rotation, stiffness_values=None):
        stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(matrix_to_pose_stamped(position, rotation, self.base, stamp))
        if self.stiffness_pub is not None and stiffness_values is not None:
            self.stiffness_pub.publish(Float64MultiArray(data=[float(v) for v in stiffness_values]))
        if self.step_hook is not None:
            self.step_hook(position, rotation)

    def hold_current_pose(self):
        if self.dry:
            return
        try:
            pos, rot = self.measured(max_age=1.0)
            self.publish(pos, rot)
            self.get_logger().warn('Commanded the controller to hold the current pose.')
        except AbortRun:
            self.get_logger().error('Could not read the pose to hold it; CRISP keeps its last '
                                    'target.')

    def check_workspace(self, position, what):
        if not self.p('workspace_check'):
            return
        lo, hi = np.asarray(self.p('workspace_min')), np.asarray(self.p('workspace_max'))
        if np.any(position < lo) or np.any(position > hi):
            raise AbortRun(f'{what} {np.round(position, 3).tolist()} is outside the workspace box '
                           f'{lo.tolist()} .. {hi.tolist()}')

    def simulate(self, cmd_position, cmd_rotation, K, D, dt):
        """Dry run plant: the first-order impedance arm of rollout_impedance."""
        A = np.linalg.solve(D, K)
        x = self.sim_T[:3, 3]
        rest = np.asarray(self.p('dry_run_resting_error'), dtype=float)
        self.sim_T[:3, 3] = x + A @ (cmd_position + rest - x) * dt
        self.sim_T[:3, :3] = cmd_rotation
        self.sim_pub.publish(matrix_to_pose_stamped(self.sim_T[:3, 3], self.sim_T[:3, :3],
                                                    self.base, self.get_clock().now().to_msg()))

    # ------------------------------------------------------------------ run
    def run(self):
        """Execute the plan. False if the operator cancelled at the prompt."""
        from tpgpt.hardware.end_effector import end_effector_for_plan
        from tpgpt.hardware.executor import (ExecutorSettings, HardwareExecutor, SafetyMonitor,
                                             approach_trajectory, crisp_applied_gains,
                                             crisp_stiffness, logged_stiffness,
                                             stiffness_projection_loss)
        from tpgpt.hardware.planner import Plan
        from tpgpt.policy.gp_policy import GPPolicy

        log = self.get_logger()
        p = self.p
        plan = Plan.load(p('plan_dir'))
        dt = plan.dt
        m = plan.metrics
        log.info(f"Plan {p('plan_dir')}: {plan.labels.describe()}, dt {dt} s, min det "
                 f"{m['min_det']:.3f}, aim {m['aim_grasp_m'] * 1000:.1f} mm")
        if plan.metadata.get('calibration_is_placeholder', True) and not self.dry \
                and not p('allow_placeholder_calibration'):
            raise AbortRun('this plan was made with PLACEHOLDER camera extrinsics; calibrate '
                           'and re-plan (or set allow_placeholder_calibration:=true knowingly)')

        # --- the hand the plan was made for ----------------------------------------
        hand = end_effector_for_plan(plan.metadata)
        mode = p('gripper_mode')
        if mode != 'none' and (mode == 'inspire') == hand.is_parallel_jaw:
            raise AbortRun(f'this plan is for the {hand.name} hand but gripper_mode is {mode!r}; '
                           f"use gripper_mode:={'inspire' if not hand.is_parallel_jaw else 'franka_grasp'}"
                           ' (or none)')
        tool_offset = np.asarray(hand.tool_offset, dtype=float) + np.asarray(p('tool_offset'),
                                                                             dtype=float)
        log.info(f'Hand {hand.name}: grasp point {np.round(tool_offset * 1000, 1).tolist()} mm '
                 f'from {hand.controller_frame} (plan {np.round(hand.tool_offset * 1000, 1).tolist()}'
                 f" + parameter {np.round(np.asarray(p('tool_offset')) * 1000, 1).tolist()})")

        # --- preflight: the whole plan must be inside the workspace ------------
        for i, pos in enumerate(plan.labels.positions):
            self.check_workspace(pos, f'plan label {i}')

        transit_scale = float(p('transit_rotational_scale'))
        if transit_scale < 0:
            static = plan.metadata.get('source_stiffness') == 'static'
            transit_scale = 1.0 if static else 0.25
            log.info(f'transit_rotational_scale auto -> {transit_scale} (source stiffness '
                     f"{plan.metadata.get('source_stiffness', 'demonstrated')})")

        t0 = time.monotonic()
        policy = GPPolicy().fit(plan.labels)
        self.policy_fit_seconds = time.monotonic() - t0
        log.info(f'Policy fitted in {time.monotonic() - t0:.1f} s')
        settings = ExecutorSettings(
            belief_correction=float(p('belief_correction')),
            speed_limit_factor=float(p('speed_limit_factor')),
            lag_tolerance=float(p('lag_tolerance')),
            rotational_stiffness=float(p('rotational_stiffness')),
            rotational_damping=float(p('rotational_damping')),
            transit_rotational_scale=transit_scale,
            speed_scale=float(p('speed_scale')),
            tool_offset=tuple(float(v) for v in tool_offset),
        )
        executor = HardwareExecutor(policy, dt, settings)
        self.policy_executor = executor
        start_pos, start_rot = executor.start_pose
        object_width = float(m.get('object_width_closing_m', 0.04))

        # --- where the arm is -------------------------------------------------
        if not self.dry:
            deadline = time.monotonic() + 5.0
            while self.measured_joints() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if self.measured_joints() is None:
                raise AbortRun(f"no fresh {p('joint_states_topic')}: the infeasible-pose fallback "
                               'needs the arm\'s joint angles. Is the robot container up?')
        T_now = self.wait_for_pose(3.0 if self.dry else 10.0)
        if T_now is None:
            if not self.dry:
                raise AbortRun(f"no pose on {p('current_pose_topic')} within 10 s")
            log.warn('dry run: no /current_pose, starting the simulated arm 10 cm above the plan '
                     'start')
            T_now = np.eye(4)
            T_now[:3, :3], T_now[:3, 3] = start_rot, start_pos + [0, 0, 0.10]
        if self.dry:
            self.sim_T = T_now.copy()
        cur_pos, cur_rot = T_now[:3, 3], T_now[:3, :3]
        gap = float(np.linalg.norm(cur_pos - start_pos))
        if gap > p('max_start_distance'):
            raise AbortRun(f'the TCP is {gap:.3f} m from the plan start (max_start_distance '
                           f"{p('max_start_distance')}); move the arm closer first")

        # --- controller and gripper -------------------------------------------
        controllers = None
        if not self.dry:
            controllers = self.ensure_controller()
            others = other_publishers(self, self.pose_topic)
            if others:
                raise AbortRun(f'{others} other publisher(s) on {self.pose_topic}; CRISP ignores '
                               'targets when more than one publisher exists')
            self.setup_gripper()

        approach_p, approach_R = approach_trajectory(
            cur_pos, cur_rot, start_pos, start_rot, dt,
            float(p('approach_max_linear_speed')), float(p('approach_max_angular_speed')))
        summary = (
            f"\n{'DRY RUN (nothing moves)' if self.dry else 'HARDWARE RUN'}\n"
            f"  controller      {self.controller} -> {self.pose_topic}"
            f"{' + ' + self.stiffness_topic if self.variable_stiffness else ''}\n"
            f"  stiffness       "
            f"{('TRANSPORTED (full 3x3 in TCP frame, 12-value message)' if self.full_stiffness else 'TRANSPORTED (diagonal in TCP frame)') if self.use_stiffness else f'FIXED {self.fixed_stiffness:.0f} N/m translational' if self.fixed_stiffness > 0 else 'controller defaults'}\n"
            f"  approach        {gap:.3f} m over {len(approach_p) * dt:.1f} s\n"
            f"  policy          {len(plan.labels)} labels at speed_scale {settings.speed_scale} "
            f"(~{len(plan.labels) * dt / settings.speed_scale:.0f} s)\n"
            f"  grasp at        {np.round(plan.labels.positions[int(m['grasp_index'])], 3).tolist()}"
            f"  release at {np.round(plan.labels.positions[int(m['release_index'])], 3).tolist()}\n"
            f"  hand            {hand.name}, gripper {p('gripper_mode') if not self.dry else 'not sent'}\n"
        )
        print(summary)
        if not self.dry and p('require_confirmation'):
            print('Keep the emergency stop within reach.')
            if input("Type 'yes' to ACTIVATE the controller and execute: ").strip().lower() != 'yes':
                log.info('Cancelled; nothing was activated.')
                return False

        if not self.dry:
            self.cmc.activate(self.controller, controllers)
        self.moving = True
        self.offset = np.zeros(3)
        if p('gripper_mode') != 'none':
            self.command_gripper(False, object_width, wait=not self.dry)

        log_path = self._open_log(p('plan_dir'))
        writer = csv.writer(open(log_path, 'w', newline=''))
        writer.writerow(['t', 'stage', 'phase', 'gate', 'behind', 'expected', 'clamped', 'gripper',
                         'meas_x', 'meas_y', 'meas_z', 'cmd_x', 'cmd_y', 'cmd_z',
                         'att_x', 'att_y', 'att_z', 'k_tcp_x', 'k_tcp_y', 'k_tcp_z', 'k_rot',
                         'k_offdiag_share'])
        realtime = not self.dry or p('dry_run_realtime')
        started = time.monotonic()
        trace = []
        timing = {'last': None, 'worst': 0.0}

        def wait_period(next_t):
            now = time.monotonic()
            if timing['last'] is not None:
                timing['worst'] = max(timing['worst'], now - timing['last'])
            timing['last'] = now
            if not realtime:
                return next_t + dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -dt:
                log.warn(f'control loop overran by {-delay * 1000:.0f} ms',
                         throttle_duration_sec=2.0)
            return max(next_t + dt, time.monotonic() - dt)

        # --- approach ----------------------------------------------------------
        log.info(f'Approach: {len(approach_p)} steps')
        self.stage = 'approach'
        next_t = time.monotonic()
        k0 = self.fixed_stiffness if self.fixed_stiffness > 0 else 400.0
        K0 = np.eye(3) * k0
        D0 = np.eye(3) * 2.0 * np.sqrt(k0)       # CRISP's damping for d_pos: -1 (400 -> 40)
        for pos, rot in zip(approach_p, approach_R):
            self.check_workspace(pos, 'approach target')
            meas_p, _ = self.measured()
            err = float(np.linalg.norm(meas_p - pos))
            if err > p('max_tracking_error'):
                raise AbortRun(f'approach tracking error {err * 1000:.0f} mm')
            self.publish(pos, rot, crisp_stiffness(K0, rot, float(p('rotational_stiffness')),
                                                   float(p('stiffness_min_translational')),
                                                   float(p('stiffness_max_translational')),
                                                   float(p('stiffness_max_rotational')),
                                                   full=self.full_stiffness))
            if self.dry:
                self.simulate(pos, rot, K0, D0, dt)
            writer.writerow([f'{time.monotonic() - started:.3f}', 'approach', 0, 1, err, 0, 0, -1,
                             *meas_p, *pos, *pos, k0, k0, k0, 0, 0])
            next_t = wait_period(next_t)
        settle_deadline = time.monotonic() + float(p('approach_settle_timeout'))
        self.stage = 'approach_settle'
        while True:
            meas_p, _ = self.measured()
            err = float(np.linalg.norm(meas_p - start_pos))
            if err <= p('approach_settle_tolerance'):
                break
            if time.monotonic() > settle_deadline:
                raise AbortRun(f'the arm did not settle at the plan start ({err * 1000:.0f} mm)')
            self.publish(start_pos, start_rot)
            if self.dry:
                self.simulate(start_pos, start_rot, K0, D0, dt)
            next_t = wait_period(next_t)
        self.offset = self._settle_and_measure_offset(start_pos, start_rot, K0, D0, dt,
                                                      wait_period, writer, started)
        next_t = time.monotonic()

        # --- policy ------------------------------------------------------------
        log.info('Executing the transported policy')
        self.stage = 'policy'
        safety = SafetyMonitor(max_tracking_error=float(p('max_tracking_error')),
                               progress_window=int(p('progress_window')),
                               min_progress_ratio=float(p('min_progress_ratio')))
        gripper_closed = False
        policy_deadline = (time.monotonic() + float(p('max_policy_seconds'))
                           if realtime else float('inf'))
        losses = []
        while not executor.finished:
            if time.monotonic() > policy_deadline:
                raise AbortRun('max_policy_seconds exceeded')
            meas_p, meas_R = self.measured()
            cmd = executor.step(meas_p, meas_R, self.measured_joints())
            if executor.stalled and p('on_stall') == 'continue':
                log.warn(f'execution watchdog declared the arm stuck at phase {executor.phase:.3f} '
                         f'after {executor.steps} steps; on_stall=continue: lag gate OFF, '
                         'following the plan at its own speed from here')
                executor.law = dataclasses.replace(executor.law, lag_tolerance=None)
                executor.settings = dataclasses.replace(executor.settings, lag_tolerance=None)
                executor.stalled = False
                executor.finished = executor.phase >= executor.settings.stop_time_belief
            reason = safety.check(executor.tool_position(meas_p, meas_R), cmd.attractor)
            if reason:
                raise AbortRun(reason)
            self.check_workspace(cmd.tcp_position, 'commanded TCP')
            if self.fixed_stiffness > 0:
                # the policy's rotational schedule is kept; translation is fixed
                cmd.stiffness, cmd.damping = K0, D0
            values = crisp_stiffness(cmd.stiffness, meas_R, cmd.rotational_stiffness,
                                     float(p('stiffness_min_translational')),
                                     float(p('stiffness_max_translational')),
                                     float(p('stiffness_max_rotational')),
                                     full=self.full_stiffness)
            # Share of K_tcp off its diagonal: dropped by the 6-value form, applied by the 12.
            loss = stiffness_projection_loss(cmd.stiffness, meas_R)
            losses.append(loss)
            self.last_cmd = cmd
            closed = cmd.gripper > 0
            if closed != gripper_closed:
                if p('arrival_hold'):
                    index = int(m['grasp_index'] if closed else m['release_index'])
                    self._arrive(plan, executor, index, cmd, values, dt, wait_period, writer,
                                 started, 'arrive_grasp' if closed else 'arrive_release')
                if p('gripper_mode') != 'none':
                    self.command_gripper(closed, object_width, wait=not self.dry)
                gripper_closed = closed
                next_t = time.monotonic()
                meas_p, meas_R = self.measured()     # the hold moved the arm; log where it is now
            self.publish(cmd.tcp_position - self.offset, cmd.tcp_rotation, values)
            if self.dry:
                # The simulated arm runs the gains CRISP applies for this message, not the
                # policy's own K: the same when nothing is dropped or clamped.
                K_run, D_run = crisp_applied_gains(values, meas_R)
                self.simulate(cmd.tcp_position - self.offset, cmd.tcp_rotation, K_run, D_run, dt)
            trace.append((cmd.tcp_position.copy(), cmd.tcp_rotation.copy()))
            k_diag, k_rot = logged_stiffness(values)
            writer.writerow([f'{time.monotonic() - started:.3f}', 'policy', f'{cmd.phase:.5f}',
                             f'{cmd.gate:.4f}', f'{cmd.behind:.5f}', f'{cmd.expected:.5f}',
                             int(cmd.clamped), int(cmd.gripper), *meas_p, *cmd.tcp_position,
                             *cmd.attractor, *k_diag, k_rot, f'{loss:.4f}'])
            if len(trace) % 10 == 0:
                self._publish_trace(trace)
            next_t = wait_period(next_t)

        self._publish_trace(trace)
        if executor.stalled:
            raise AbortRun(f'the execution watchdog declared the arm stuck at phase '
                           f'{executor.phase:.3f} after {executor.steps} steps')
        self.moving = False
        text = (f'Policy complete: {executor.steps} steps, final phase {executor.phase:.3f}, '
                f'sag re-baselines {executor.watchdog.rebaselines}; infeasible-pose fallback: '
                f'{executor.relaxed_steps} relaxed steps, {executor.skips} skips '
                f'({executor.skipped_steps} steps steering), '
                f'{executor.feasibility_unavailable} gate-shut steps without joint angles')
        if self.use_stiffness and losses:
            text += (f"; stiffness off-diagonal share {'applied (full matrix)' if self.full_stiffness else 'dropped'}: "
                     f'median {np.median(losses):.2f}, max {np.max(losses):.2f}')
        text += f"; worst loop period {timing['worst'] * 1000:.0f} ms (nominal {dt * 1000:.0f})"
        log.info(text + f'. Log: {log_path}. CRISP keeps holding the last target.')
        # Keep the final target for a moment so the last command is received.
        self.stage = 'final_hold'
        for _ in range(int(1.0 / dt)):
            self.publish(cmd.tcp_position - self.offset, cmd.tcp_rotation, values)
            time.sleep(dt)
        return True

    def _settle_and_measure_offset(self, start_pos, start_rot, K, D, dt, wait_period, writer,
                                   started):
        """Hold the plan start for `settle_time`, then return the compensation vector.

        The resting error (measured minus commanded, averaged over the last third
        of the hold, arm stationary) is what the controller leaves standing: gravity
        compensation and payload mismatch show up in z, joint friction in x/y with a
        sign that follows the last motion. Only the components named by
        `offset_compensation` are kept, bounded by `max_offset_compensation`.
        """
        p, log = self.p, self.get_logger()
        self.stage = 'settle'
        mode = str(p('offset_compensation'))
        if mode not in ('none', 'z', 'xyz'):
            raise AbortRun(f'offset_compensation must be none, z or xyz, not {mode!r}')
        samples = []
        next_t = time.monotonic()
        n = max(int(float(p('settle_time')) / dt), 3)
        for k in range(n):
            meas_p, _ = self.measured()
            if k >= 2 * n // 3:
                samples.append(meas_p - start_pos)
            self.publish(start_pos, start_rot)
            if self.dry:
                self.simulate(start_pos, start_rot, K, D, dt)
            writer.writerow([f'{time.monotonic() - started:.3f}', 'settle', 0, 1,
                             float(np.linalg.norm(meas_p - start_pos)), 0, 0, -1, *meas_p,
                             *start_pos, *start_pos, 400, 400, 400, 0, 0])
            next_t = wait_period(next_t)
        resting = np.mean(samples, axis=0)
        offset = np.zeros(3)
        if mode == 'z':
            offset[2] = resting[2]
        elif mode == 'xyz':
            offset = resting.copy()
        limit = float(p('max_offset_compensation'))
        if np.linalg.norm(offset) > limit:
            offset *= limit / np.linalg.norm(offset)
        log.info(f'Resting error at the plan start: {np.round(resting * 1000, 1).tolist()} mm '
                 f'(measured - commanded); compensating {np.round(offset * 1000, 1).tolist()} mm '
                 f'({mode})')
        return offset

    def _arrive(self, plan, executor, index, cmd, values, dt, wait_period, writer, started,
                stage):
        """Servo onto the plan's grasp (or release) pose before the gripper acts.

        The policy clock is paused. The target is the transported label itself, not
        the executor's attractor: the attractor can be dragged off the path when the
        lag clamp engages (a red-cup run closed 82 mm from its grasp that way). A
        bounded integral term removes whatever resting error the static
        compensation leaves at this pose.
        """
        from tpgpt.hardware.executor import crisp_applied_gains, logged_stiffness

        p, log = self.p, self.get_logger()
        self.stage = stage
        labels = plan.labels
        target_R = np.asarray(labels.orientations[index], dtype=float)
        target_p = executor.tcp_from_tool(labels.positions[index], target_R)
        gap = float(np.linalg.norm(cmd.tcp_position - target_p))
        if gap > float(p('arrival_max_gap')):
            raise AbortRun(f'{stage}: the commanded pose is {gap * 1000:.0f} mm from the planned '
                           f'{"grasp" if stage == "arrive_grasp" else "release"} when the gripper '
                           f'should act (arrival_max_gap {float(p("arrival_max_gap")) * 1000:.0f} '
                           'mm); the arm lagged and the executor clamped. Lower speed_scale.')
        tol, gain = float(p('arrival_tolerance')), float(p('arrival_gain'))
        limit = float(p('max_offset_compensation'))
        integral = np.zeros(3)
        inside = 0
        deadline = time.monotonic() + float(p('arrival_timeout'))
        next_t = time.monotonic()
        # Walk the command from where it is onto the target at approach speed.
        step = float(p('approach_max_linear_speed')) * dt
        command = cmd.tcp_position.copy()
        while True:
            meas_p, _ = self.measured()
            error = target_p - meas_p
            dist = float(np.linalg.norm(error))
            inside = inside + 1 if dist <= tol else 0
            if inside >= 3:
                log.info(f'{stage}: within {dist * 1000:.1f} mm of the plan '
                         f'(integral {np.round(integral * 1000, 1).tolist()} mm)')
                break
            if time.monotonic() > deadline:
                log.warn(f'{stage}: still {dist * 1000:.1f} mm from the plan after '
                         f'{float(p("arrival_timeout")):.1f} s ({np.round(error * 1000, 1).tolist()} '
                         'mm); acting anyway')
                break
            delta = target_p - command
            command = command + delta * min(1.0, step / max(float(np.linalg.norm(delta)), 1e-9))
            if float(np.linalg.norm(target_p - command)) < 1e-6:
                integral = integral + gain * dt * error
                if np.linalg.norm(integral) > limit:
                    integral *= limit / np.linalg.norm(integral)
            out = command - self.offset + integral
            self.check_workspace(out, f'{stage} target')
            self.publish(out, target_R, values)
            if self.dry:
                # the dry-run arm holds target_R, so it is also the frame CRISP would use
                self.simulate(out, target_R, *crisp_applied_gains(values, target_R), dt)
            k_diag, k_rot = logged_stiffness(values)
            writer.writerow([f'{time.monotonic() - started:.3f}', stage, f'{cmd.phase:.5f}', 1,
                             dist, 0, 0, int(cmd.gripper), *meas_p, *out, *command,
                             *k_diag, k_rot, 0])
            next_t = wait_period(next_t)
        # Keep what the hold learned for the rest of the run: the object's weight after
        # a grasp and the pose-dependent error both persist.
        self.offset = self.offset - integral
        self.stage = 'policy'

    def _open_log(self, plan_dir):
        directory = self.p('log_dir') or os.path.join(plan_dir, 'executions')
        os.makedirs(directory, exist_ok=True)
        tag = 'dry' if self.dry else 'robot'
        return os.path.join(directory, time.strftime(f'{tag}_%Y%m%d_%H%M%S.csv'))

    def _publish_trace(self, trace):
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.frame_id, path.header.stamp = self.base, stamp
        path.poses = [matrix_to_pose_stamped(pos, rot, self.base, stamp) for pos, rot in trace[::2]]
        self.trace_pub.publish(path)
        markers = MarkerArray()
        markers.markers.append(line_strip_marker('executed', 0, [t[0] for t in trace], self.base,
                                                 stamp, color=(1.0, 1.0, 0.0), width=0.003))
        markers.markers += axes_markers('commanded_tcp', 0, trace[-1][0], trace[-1][1], self.base,
                                        stamp, length=0.05)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TpgptExecutorNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    exit_code = 0
    try:
        node.run()
    except AbortRun as exc:
        node.get_logger().error(f'Aborted: {exc}')
        if node.moving:
            node.hold_current_pose()
        exit_code = 1
    except KeyboardInterrupt:
        if node.moving:
            node.hold_current_pose()
    finally:
        time.sleep(0.2)  # let a final hold command go out
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
