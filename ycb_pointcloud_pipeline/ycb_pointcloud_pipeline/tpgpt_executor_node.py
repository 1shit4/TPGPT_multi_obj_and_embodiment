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
import json
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


def _installed_config(name):
    """Installed config/<name> (as tpgpt_planner_node.default_config_path), else the source's."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'), 'config', name)
    except Exception:  # noqa: BLE001 - package not installed
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', name)


class AbortRun(RuntimeError):
    pass


#: The settings a learned plan's labels are right at (audit B1, step 6): a bundle from
#: ``python -m tpgpt.hardware.replay bundle`` has labels that are an attractor learned by
#: replaying the demonstration, and records them (plan.json ``metadata.learned_execution``).
#: The lag they were learned against depends on each. A parameter left at its default takes
#: the learned value; one set to anything else is refused.
LEARNED_SETTINGS = ('speed_scale', 'rotational_stiffness', 'transit_rotational_scale',
                    'use_transported_stiffness', 'offset_compensation')


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) < 1e-9


def learned_settings(plan_dir: str, given: dict, defaults: dict) -> dict:
    """What a learned plan must run at, from its plan.json, against the parameters ``given``.

    Returns ``{name: value}`` (empty for a plan that is not learned, or no plan.json -- the
    plan's own load reports that). Raises SystemExit naming every parameter set to a value
    other than its default and the learned one.
    """
    try:
        with open(os.path.join(os.path.expanduser(plan_dir), 'plan.json')) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return {}
    learned = (info.get('metadata') or {}).get('learned_execution') or {}
    out, clashes = {}, []
    for name in LEARNED_SETTINGS:
        if name not in learned:
            continue
        want, default = learned[name], defaults[name]
        if not _same(given[name], default) and not _same(given[name], want):
            clashes.append(f'{name}:={given[name]!r} (learned at {want!r})')
        out[name] = type(default)(want)
    if clashes:
        raise SystemExit("this plan's labels are a learned attractor, right only at the settings "
                         f"they were learned under: {', '.join(clashes)}. Leave these at their "
                         'defaults (the learned values are then used).')
    return out


class TpgptExecutorNode(Node):
    def __init__(self, **kwargs):
        super().__init__('tpgpt_executor_node', **kwargs)
        self._defaults = {}
        self._learned = {}

        def d(name, value):
            self._defaults[name] = value
            return self.declare_parameter(name, value)
        d('plan_dir', '')
        d('dry_run', True)
        d('dry_run_realtime', True)
        d('dry_run_resting_error', [0.0, 0.0, 0.0])  # m; simulated arm rests this far from its target
        # The simulated arm's lag, as a multiple of CRISP's nominal time constant. This arm
        # measured about 2 (audit B2 step 3); 1 is the spring model the simulation uses.
        d('dry_run_lag_scale', 1.0)
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
        # A learned plan (one with an arm path) is followed against the path the arm traced,
        # not with the simulation's lag gate and clamp (tpgpt.hardware.executor.ArmPathSettings;
        # the user, 2026-10-01), and always holds at its marks before the gripper acts.
        d('follow_arm_path', True)
        d('path_pace_span', 0.02)                   # m behind the path (beyond its margin) where the clock stops
        d('path_pacing', True)                      # false: the clock never waits for the arm
        d('path_hold', True)                        # false: no arrival hold; the gripper acts on time
        d('path_correction', True)                  # bring an arm beside its path back
        d('path_correction_gain', 1.5)              # 1/s
        d('path_correction_max', 0.03)              # m
        d('path_bound_slack', 0.01)                 # m, on top of the plan's lead + the correction cap
        d('path_blocked_steps', 40)                 # steps stopped or at the cap without closing in
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
        # Replay mode (tpgpt.hardware.replay; audit B1): instead of a plan, stream the pending
        # pass of a replay directory -- a taught demonstration's attractor, in time, at the
        # replay's speed_scale, under its own stiffness labels -- and record where the arm was.
        # No policy, no lag gate: the simulation's teacher records its demonstration the same way.
        d('replay_dir', '')
        # The shelf the replay's taught path (with its margin) is checked against before each
        # pass ('' = no check): the installed config/shelf.yaml.
        d('replay_shelf_config', _installed_config('shelf.yaml'))
        # A dry run of a replay checks and summarises the pass and stops: a simulated arm's
        # recording saved into the replay would count as a run of it. True simulates the pass
        # and records it there anyway -- for tests of the replay itself, never on a real one.
        d('replay_record_dry', False)
        # With the object, a pass first guides it onto its taught place: the camera's view of it
        # is fitted with its scan, compared with the fit of the taught cloud, and the operator
        # is told how far to move and turn it (tpgpt.hardware.replay.placement_offset); the pass
        # waits until it is in place. False skips it (no camera).
        d('replay_place_check', True)
        d('replay_object', '')          # '' = from the demonstration's directory name
        # Stop the pass once this label is reached (-1: the whole pass). A partial pass is a
        # diagnostic -- saved as partial_<n>.npz, never a run the update learns from.
        d('replay_until_label', -1)
        # the camera picture the object is aligned on (placement guide)
        d('replay_color_topic', '/camera/camera/color/image_raw')
        d('replay_camera_info_topic', '/camera/camera/color/camera_info')

        p = self.p
        self.replay = bool(p('replay_dir'))
        if not p('plan_dir') and not self.replay:
            raise SystemExit('Set -p plan_dir:=<directory written by tpgpt_planner_node> '
                             '(or -p replay_dir:=<replay directory> to replay a demonstration)')
        if not self.replay:
            self._learned = learned_settings(
                p('plan_dir'), {n: p(n) for n in LEARNED_SETTINGS}, self._defaults)
            if self._learned:
                self.get_logger().info(
                    'Learned labels: running at the settings they were learned under: ' +
                    ', '.join(f'{n} {v}' for n, v in self._learned.items()))
        self.dry = bool(p('dry_run'))
        # a replay always sends the demonstration's own stiffness labels
        self.use_stiffness = bool(p('use_transported_stiffness')) or self.replay
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
        # While a replay pass runs, every pose and joint-state message is also kept, as
        # received (converted after the pass: see _pose_cb), for the pass's diagnostic record.
        self._highrate = None  # {'poses': [...], 'joints': [...]} while recording
        self._pending_record = None  # (replay dir, pass, 20 Hz record) of a pass cut short
        self.create_subscription(PoseStamped, p('current_pose_topic'), self._pose_cb, 1)
        self._joints = None   # (monotonic, JointState) -- converted on demand
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, p('joint_states_topic'), self._joint_cb, 1)

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
        if name in self._learned:
            return self._learned[name]
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ inputs
    def _pose_cb(self, msg):
        # pose_broadcaster publishes at the controller rate (up to 1 kHz). Anything
        # heavier than storing the message here starves the control thread of the
        # GIL: measured on fake hardware, a matrix conversion per message stalled
        # the 20 Hz loop for 1.8 s. The conversion happens in measured() instead.
        self._pose = (time.monotonic(), msg)
        if self._highrate is not None:
            self._highrate['poses'].append(msg)

    def _joint_cb(self, msg):
        self._joints = (time.monotonic(), msg)
        if self._highrate is not None:
            self._highrate['joints'].append(msg)

    def _take_highrate(self):
        """Stop keeping messages and return those kept, as arrays (header stamps, s)."""
        log, self._highrate = self._highrate, None
        log = log or {'poses': [], 'joints': []}
        stamp = lambda m: m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec  # noqa: E731
        out = {}
        if log['poses']:
            out['pose_stamp'] = np.array([stamp(m) for m in log['poses']])
            out['pose_position'] = np.array([[m.pose.position.x, m.pose.position.y,
                                              m.pose.position.z] for m in log['poses']])
            out['pose_quaternion'] = np.array([[m.pose.orientation.x, m.pose.orientation.y,
                                                m.pose.orientation.z, m.pose.orientation.w]
                                               for m in log['poses']])
        names = [f'fr3_joint{k}' for k in range(1, 8)]
        rows = [m for m in log['joints'] if all(n in m.name for n in names)]
        if rows:
            order = [rows[0].name.index(n) for n in names]
            out['joint_stamp'] = np.array([stamp(m) for m in rows])
            for field in ('position', 'velocity', 'effort'):
                vals = [list(getattr(m, field)) for m in rows]
                if all(len(v) == len(rows[0].name) for v in vals):
                    out[f'joint_{field}'] = np.array(vals)[:, order]
        return out

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
        A = np.linalg.solve(D, K) / float(self.p('dry_run_lag_scale'))
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
        following = plan.arm_path is not None and bool(p('follow_arm_path'))
        arm_path = arm_settings = None
        if following:
            from tpgpt.hardware.executor import ArmPathSettings
            arm_path = {'positions': plan.arm_path.positions, 'margin': plan.arm_margin}
            arm_settings = ArmPathSettings(
                pace_span=float(p('path_pace_span')), pacing=bool(p('path_pacing')),
                correction=bool(p('path_correction')),
                correction_gain=float(p('path_correction_gain')),
                correction_max=float(p('path_correction_max')),
                bound_slack=float(p('path_bound_slack')),
                blocked_steps=int(p('path_blocked_steps')))
            on = [name for name, key in (('pacing', 'path_pacing'), ('correction', 'path_correction'),
                                         ('hold at the marks', 'path_hold')) if p(key)]
            log.info(f"Learned plan: following its arm path ({', '.join(on) or 'nothing added'}) "
                     "instead of the simulation's lag gate and clamp")
        elif plan.arm_path is not None:
            log.warn('Learned plan with follow_arm_path:=false: the lag gate and clamp will slow '
                     'and cut its command (audit B1 step 8)')
        executor = HardwareExecutor(policy, dt, settings, arm_path=arm_path,
                                    arm_settings=arm_settings)
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
        where = plan.arm_path if plan.arm_path is not None else plan.labels
        summary = (
            f"\n{'DRY RUN (nothing moves)' if self.dry else 'HARDWARE RUN'}\n"
            f"  controller      {self.controller} -> {self.pose_topic}"
            f"{' + ' + self.stiffness_topic if self.variable_stiffness else ''}\n"
            f"  stiffness       "
            f"{('TRANSPORTED (full 3x3 in TCP frame, 12-value message)' if self.full_stiffness else 'TRANSPORTED (diagonal in TCP frame)') if self.use_stiffness else f'FIXED {self.fixed_stiffness:.0f} N/m translational' if self.fixed_stiffness > 0 else 'controller defaults'}\n"
            f"  approach        {gap:.3f} m over {len(approach_p) * dt:.1f} s\n"
            f"  policy          {len(plan.labels)} labels at speed_scale {settings.speed_scale} "
            f"(~{len(plan.labels) * dt / settings.speed_scale:.0f} s)\n"
            f"  grasp at        {np.round(where.positions[int(m['grasp_index'])], 3).tolist()}"
            f"  release at {np.round(where.positions[int(m['release_index'])], 3).tolist()}"
            f"{' (the arm; the learned command leads it)' if plan.arm_path is not None else ''}\n"
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
                         'k_offdiag_share', 'path_index', 'path_deviation', 'path_across',
                         'corr_x', 'corr_y', 'corr_z'])
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
        follow_log = []
        policy_deadline = (time.monotonic() + float(p('max_policy_seconds'))
                           if realtime else float('inf'))
        losses = []
        while not executor.finished:
            if time.monotonic() > policy_deadline:
                raise AbortRun('max_policy_seconds exceeded')
            meas_p, meas_R = self.measured()
            cmd = executor.step(meas_p, meas_R, self.measured_joints())
            if following and executor.blocked:
                raise AbortRun(f'blocked: {executor.blocked_reason}; holding here')
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
                if p('arrival_hold') or (following and p('path_hold')):
                    index = int(m['grasp_index'] if closed else m['release_index'])
                    held = self._arrive(plan, executor, index, cmd, values, dt, wait_period,
                                        writer, started,
                                        'arrive_grasp' if closed else 'arrive_release',
                                        learned=following)
                    if following:
                        # carry on from where the hold left the command, not jump back
                        executor.correction = executor.correction + (held + self.offset
                                                                     - cmd.tcp_position)
                        cap = float(p('path_correction_max'))
                        if np.linalg.norm(executor.correction) > cap:
                            executor.correction *= cap / np.linalg.norm(executor.correction)
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
                             *cmd.attractor, *k_diag, k_rot, f'{loss:.4f}',
                             *((f"{cmd.trace['path_index']:.2f}", f"{cmd.trace['deviation']:.5f}",
                                f"{cmd.trace['across']:.5f}",
                                *np.round(cmd.trace['correction'], 5)) if following else ())])
            if following:
                follow_log.append((cmd.trace['across'], cmd.behind, cmd.gate, cmd.clamped,
                                   float(np.linalg.norm(cmd.trace['correction']))))
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
        if follow_log:
            across, behind, pace, bounded, corr = (np.array(v, dtype=float) for v in zip(*follow_log))
            text += (f'; arm path: beside it median {np.median(across) * 1000:.1f} mm, worst '
                     f'{across.max() * 1000:.1f}; along it from {behind.min() * 1000:+.1f} (ahead) to '
                     f'{behind.max() * 1000:+.1f} mm (behind); clock slowed on {int((pace < 1).sum())} steps '
                     f'(slowest {pace.min():.2f}); correction up to {corr.max() * 1000:.1f} mm; '
                     f'command bounded on {int(bounded.sum())} steps')
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

    # ------------------------------------------------------------------ replay
    def run_replay(self):
        """Stream the pending pass of `replay_dir` and record where the arm was (audit B1).

        As the simulation's `record_demonstration` does: the attractor moves label by label
        in time, linearly between labels, with no policy and no lag gate, under the
        demonstration's own stiffness labels (12-value message) and the executor's rotational
        schedule; the gripper acts at the teacher's marks while the clock waits (as the
        executor does, audit C7). The orientation commanded is the pass's attractor
        orientation, learned like the position. What is recorded, per control step: the
        label index being commanded, the measured grasp point and hand orientation (the
        plan's grasp convention, as the labels), the commanded attractor and the rotational
        stiffness. `python -m tpgpt.hardware.replay update <dir>` then computes the next pass.
        """
        from tpgpt.hardware import replay as rp
        from tpgpt.hardware.end_effector import end_effector_for_plan
        from tpgpt.hardware.executor import (SafetyMonitor, approach_trajectory,
                                             crisp_applied_gains, crisp_stiffness)
        from tpgpt.hardware.frames import slerp_rotation
        from tpgpt.policy.rollout import compliance_norm, stiffness_threshold

        log, p = self.get_logger(), self.p
        root = os.path.expanduser(p('replay_dir'))
        info, bundle = rp.load_replay(root)
        k, run = rp.pending_pass(root)
        runs = rp.planned_runs(info, k)
        attractor = rp.load_attractor(root, k)
        attractor_R = rp.load_attractor_rotations(root, k, bundle)
        labels = bundle.tool_labels
        n = len(attractor)
        dt = float(info['dt'])
        label_period = dt / float(info['speed_scale'])
        if abs(float(p('speed_scale')) - float(info['speed_scale'])) > 1e-9:
            log.warn(f"speed_scale {p('speed_scale')} ignored: a replay runs at its own "
                     f"{info['speed_scale']} (replay.json)")
        hand = end_effector_for_plan(bundle.metadata)
        mode = p('gripper_mode')
        if mode != 'none' and (mode == 'inspire') == hand.is_parallel_jaw:
            raise AbortRun(f'this demonstration is for the {hand.name} hand but gripper_mode is '
                           f'{mode!r}')
        offset = np.asarray(hand.tool_offset, dtype=float) + np.asarray(p('tool_offset'), dtype=float)
        tcp_R = np.asarray(hand.ee_from_grasp(attractor_R), dtype=float)   # commanded
        tcp_from_tool = lambda pos, R: np.asarray(pos) - np.asarray(R) @ offset  # noqa: E731
        tool_from_tcp = lambda pos, R: np.asarray(pos) + np.asarray(R) @ offset  # noqa: E731
        # the executor's rotational schedule (rollout_policy): firm, compliant while carrying
        transit_scale = float(p('transit_rotational_scale'))
        if transit_scale < 0:
            transit_scale = 1.0 if info.get('stiffness') == 'static' else 0.25
        threshold = stiffness_threshold(labels)
        compliance = [compliance_norm(K, D) for K, D in zip(labels.stiffness, labels.damping)]
        k_rot = float(p('rotational_stiffness'))

        def rotational(i):
            carrying = labels.gripper[i] > 0 and compliance[i] > 0.0 and compliance[i] >= threshold
            return k_rot * transit_scale if carrying else k_rot

        def message(i, meas_R):
            return crisp_stiffness(labels.stiffness[i], meas_R, rotational(i),
                                   float(p('stiffness_min_translational')),
                                   float(p('stiffness_max_translational')),
                                   float(p('stiffness_max_rotational')), full=self.full_stiffness)

        schedule = np.array([rotational(i) for i in range(n)])
        for i in range(n):
            self.check_workspace(tcp_from_tool(attractor[i], tcp_R[i]), f'replay label {i}')
        start_pos, start_rot = tcp_from_tool(attractor[0], tcp_R[0]), tcp_R[0]

        # --- where the arm is -------------------------------------------------
        T_now = self.wait_for_pose(3.0 if self.dry else 10.0)
        if T_now is None:
            if not self.dry:
                raise AbortRun(f"no pose on {p('current_pose_topic')} within 10 s")
            log.warn('dry run: no /current_pose, starting the simulated arm 10 cm above the start')
            T_now = np.eye(4)
            T_now[:3, :3], T_now[:3, 3] = start_rot, start_pos + [0, 0, 0.10]
        if self.dry:
            self.sim_T = T_now.copy()
        cur_pos, cur_rot = T_now[:3, 3], T_now[:3, :3]
        gap = float(np.linalg.norm(cur_pos - start_pos))
        if gap > p('max_start_distance'):
            raise AbortRun(f'the TCP is {gap:.3f} m from the replay start (max_start_distance '
                           f"{p('max_start_distance')}); move the arm closer first")
        # Before anything moves: the taught path, with this pass's margin for how far the arm
        # strays from it, against the shelf model and the table -- and the move onto it from
        # where the arm is (tpgpt.hardware.replay.check_pass). Not the attractor: the arm
        # does not go where the spring pulls, it lags it.
        shelf_config = p('replay_shelf_config')
        with_object = mode != 'none'
        if shelf_config:
            verdict, _ = rp.check_pass(root, k, shelf_config, rotational=schedule,
                                       with_object=with_object, start_q=self.measured_joints(),
                                       until=int(p('replay_until_label')))
            clearance, tilt, source = rp.pass_margin(root, k)
            around = (f'{clearance.min() * 1000:.1f}-{clearance.max() * 1000:.1f} mm and '
                      f'{np.degrees(tilt.min()):.1f}-{np.degrees(tilt.max()):.1f} deg')
            if source.startswith('the bracket'):
                what = (f'{source} (from where it went to {rp.BRACKET_OVERSHOOT * 100:.0f} % past '
                        f'the taught path), {around} around each path')
            else:
                what = f'the taught path with a margin of {around} ({source})'
            margin_text = what
            if verdict:
                raise AbortRun(f'pass {k}: {what}: {verdict} (shelf {shelf_config}); not replayed')
            log.info(f'pass {k}: clear of the shelf model and the table: {what}'
                     f"{', carrying the object' if with_object else ''}")
        else:
            margin_text = 'NOT CHECKED'
            log.warn('replay_shelf_config empty: the path is NOT checked')
        # Where the commanded path keeps a joint inside CRISP's joint-limit repulsion zone: the
        # repulsion then pushes against the task spring (2026-09-30: joint 7 at its limit from
        # label 0 to 156 of the 18:39 demonstration, where the arm jittered at the pick).
        from tpgpt.hardware.fr3 import HOME_Q
        start_q = self.measured_joints()
        zones = rp.joint_limit_zones([tcp_from_tool(a, R) for a, R in zip(attractor, tcp_R)],
                                     tcp_R, HOME_Q if start_q is None else start_q)
        for z in zones:
            log.warn(f"joint {z['joint']} is within {rp.JOINT_LIMIT_SAFE_RANGE} rad of its limit "
                     f"(closest {z['closest_rad']:.3f}) from label {z['first']} to {z['last']}: "
                     "CRISP's joint-limit repulsion pushes against the wrist spring there")
        until = int(p('replay_until_label'))
        guide = (self._object_guide(info, bundle)
                 if with_object and p('replay_place_check') else None)
        controllers = None
        if not self.dry:
            controllers = self.ensure_controller()
            for topic in (self.pose_topic, self.stiffness_topic):
                others = other_publishers(self, topic)
                if others:
                    raise AbortRun(f'{others} other publisher(s) on {topic}; CRISP ignores '
                                   'commands when more than one publisher exists')
            self.setup_gripper()
        width = float(info.get('object_width_closing_m', 0.04))
        approach_p, approach_R = approach_trajectory(
            cur_pos, cur_rot, start_pos, start_rot, dt,
            float(p('approach_max_linear_speed')), float(p('approach_max_angular_speed')))
        print(f"\n{'DRY RUN (nothing moves)' if self.dry else 'HARDWARE RUN'}: REPLAY pass {k}, "
              f"run {run} of {runs}, of {root}\n"
              f"  demonstration  {info['bundle']}\n"
              f"  labels         {n} at speed_scale {info['speed_scale']} (~{n * label_period:.0f} s), "
              f"grasp {info['grasp_index']}, release {info['release_index']}\n"
              f"  stiffness      the labels' own ({info.get('stiffness')}), "
              f"{'12-value' if self.full_stiffness else '6-value'} message, rotational {k_rot:.0f} "
              f"(x{transit_scale} while carrying)\n"
              f"  path check     the taught path with a margin of {margin_text}\n"
              f"  approach       {gap:.3f} m over {len(approach_p) * dt:.1f} s\n"
              f"  gripper        {mode if not self.dry else 'not sent'} (object {width * 1000:.0f} mm)"
              f"{'' if with_object else ' -- NO OBJECT: take it off the table'}\n")
        placed = None
        if guide:
            placed = self._place_object(guide, interactive=not self.dry)
        if self.dry and not p('replay_record_dry'):
            log.info('Dry run: the pass is checked and summarised only, and nothing is recorded '
                     '(replay_record_dry:=true simulates it into the replay, for tests)')
            return True
        if not self.dry and p('require_confirmation'):
            print('Keep the emergency stop within reach.')
            if input("Type 'yes' to ACTIVATE the controller and replay: ").strip().lower() != 'yes':
                log.info('Cancelled; nothing was activated.')
                return False
        if not self.dry:
            self.cmc.activate(self.controller, controllers)
        self.moving = True
        if mode != 'none':
            self.command_gripper(False, width, wait=not self.dry)
        realtime = not self.dry or p('dry_run_realtime')

        def wait_period(next_t):
            if not realtime:
                return next_t + dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            return max(next_t + dt, time.monotonic() - dt)

        # --- approach and settle at the start ----------------------------------
        next_t = time.monotonic()
        for pos, rot in list(zip(approach_p, approach_R)) + [(start_pos, start_rot)] * int(
                float(p('settle_time')) / dt):
            meas_p, meas_R = self.measured()
            if float(np.linalg.norm(meas_p - pos)) > p('max_tracking_error'):
                raise AbortRun(f'approach tracking error {np.linalg.norm(meas_p - pos) * 1000:.0f} mm')
            values = message(0, meas_R)
            self.publish(pos, rot, values)
            if self.dry:
                self.simulate(pos, rot, *crisp_applied_gains(values, meas_R), dt)
            next_t = wait_period(next_t)

        # --- the pass ---------------------------------------------------------------
        log.info(f'Replaying pass {k}')
        self.stage = 'replay'
        safety = SafetyMonitor(max_tracking_error=float(p('max_tracking_error')),
                               progress_window=int(p('progress_window')),
                               min_progress_ratio=float(p('min_progress_ratio')))
        rec = {name: [] for name in ('step', 'label_index', 'positions', 'rotations', 'commanded',
                                     'commanded_rotations', 'rotational', 'gripper')}
        deadline = time.monotonic() + float(p('max_policy_seconds')) if realtime else float('inf')
        s, step, closed_now = 0.0, 0, False
        next_t = time.monotonic()
        self._highrate = {'poses': [], 'joints': []}   # every message, for the record
        try:
            while True:
                if time.monotonic() > deadline:
                    raise AbortRun('max_policy_seconds exceeded')
                i = min(int(np.floor(s)), n - 1)
                j = min(i + 1, n - 1)
                pos = rp.attractor_at(attractor, s)
                rot = slerp_rotation(tcp_R[i], tcp_R[j], float(s - i)) if j != i else tcp_R[i]
                closed = bool(labels.gripper[i] > 0)
                if closed != closed_now:
                    if mode != 'none':
                        self.command_gripper(closed, width, wait=not self.dry)   # the clock waits
                    closed_now = closed
                    next_t = time.monotonic()
                meas_p, meas_R = self.measured()
                reason = safety.check(tool_from_tcp(meas_p, meas_R), pos)
                if reason:
                    raise AbortRun(reason)
                values = message(i, meas_R)
                target = tcp_from_tool(pos, rot)
                self.check_workspace(target, 'replay target')
                self.publish(target, rot, values)
                if self.dry:
                    self.simulate(target, rot, *crisp_applied_gains(values, meas_R), dt)
                # orientations in the plan's grasp convention, as the labels they are compared with
                for name, value in (('step', step), ('label_index', s),
                                    ('positions', tool_from_tcp(meas_p, meas_R)),
                                    ('rotations', hand.grasp_from_ee(meas_R)),
                                    ('commanded', pos), ('commanded_rotations', hand.grasp_from_ee(rot)),
                                    ('rotational', rotational(i)),
                                    ('gripper', 1.0 if closed_now else -1.0)):
                    rec[name].append(value)
                if s >= n - 1 or (0 <= until <= s):
                    break
                s = min(s + dt / label_period, float(n - 1))
                step += 1
                next_t = wait_period(next_t)
            for _ in range(int(1.0 / dt)):                     # hold the end a moment
                self.publish(target, rot, values)
                if self.dry:
                    self.simulate(target, rot, *crisp_applied_gains(values, meas_R), dt)
                time.sleep(dt if realtime else 0.0)
        except (Exception, KeyboardInterrupt):
            # What the arm did up to the abort -- any abort, a bug's included -- is the record
            # to look at. It is saved by main()
            # after the hold has gone out (converting the messages takes a moment).
            self._pending_record = (root, k, rec)
            raise
        highrate = self._take_highrate()
        if 0 <= until < n - 1:
            self.moving = False
            path = rp.save_diagnostic(root, k, 'partial', **rec, **highrate)
            log.info(f'Partial pass to label {until} recorded ({path}); not a run of pass {k}, '
                     'the update does not use it.')
            return True
        self.moving = False
        if placed is not None:          # how far from its taught place the object stood
            highrate['placement_offset'] = np.array([placed['move_x'], placed['move_y'],
                                                     placed['turn_deg']])
        path = rp.save_recording(root, k, run, **rec, **highrate)
        error, _, _ = rp.pass_error(root, k)
        m = error['at_marks']
        log.info(f"Pass {k} run {run} recorded ({path}). Over its {error['runs']} run(s): off the "
                 f"taught path {error['error_rms_mm']:.1f} mm RMS, {error['error_max_mm']:.1f} worst, "
                 f"tilt {error['tilt_max_deg']:.2f} deg worst; at the grasp "
                 f"{m['grasp']['total_mm']:.1f} mm ({m['grasp']['closing_mm']:.1f} across the jaws), "
                 f"at the release {m['release']['total_mm']:.1f} ({m['release']['closing_mm']:.1f} "
                 f"across the jaws); strayed sideways up to {error['stray_max_mm']:.1f} mm and "
                 f"{error['stray_tilt_max_deg']:.2f} deg")
        exceeded = error.get('margin_exceeded') or {}
        if exceeded.get('labels'):
            log.warn(f"The arm strayed beyond the margin this pass was checked with at "
                     f"{exceeded['labels']} labels (first {exceeded['first']}), by up to "
                     f"{exceeded['worst_mm']:.1f} mm and {exceeded['worst_tilt_deg']:.2f} deg: "
                     f"the margin ({exceeded['checked_with']}) was too small there. The next pass "
                     'is checked with what this one measured.')
        if run < runs:
            log.info(f'Next: run pass {k} again (run {run + 1} of {runs}), the same command.')
        else:
            log.info(f'Next: python -m tpgpt.hardware.replay update {root}')
        return True

    # ------------------------------------------------------------ replay: the object's place
    def _object_guide(self, info, bundle):
        """For a replay with the object: its scan fitted to the taught cloud; the camera's cloud
        of it fitted as it comes (a worker thread, tpgpt.hardware.replay.placement_offset); and
        an image to align it by, /tpgpt/replay/place_overlay -- the camera's picture with the
        taught place drawn green and the latest fit orange, five times a second."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from tpgpt.hardware import replay as rp
        from tpgpt.hardware.object_model import ObjectModel, fit_object
        from ycb_pointcloud_pipeline import camera_calibration as cc
        from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
        from ycb_pointcloud_pipeline.ros_utils import (CalibrationProvider, pointcloud2_to_xyz,
                                                       xyz_to_pointcloud2)

        log, p = self.get_logger(), self.p
        name = p('replay_object') or rp.object_name(info)
        calib = CalibrationProvider(self, default_calibration_yaml())
        T = calib.T_base_optical(calib.cal.optical_frame)
        if T is None:
            raise AbortRun('no camera calibration to see the object with; '
                           '-p replay_place_check:=false to place it by eye')
        model = ObjectModel.load(name)
        support = float(bundle.placement.support_height)
        taught = fit_object(model, np.asarray(bundle.placement.points, dtype=float), support,
                            T[:3, 3])
        if not taught.accepted:
            log.warn(f'the scan fit of the taught {name} was not accepted ({taught.reason}); '
                     'its place is approximate')
        g = {'name': name, 'topic': f'/ycb_pointclouds/{name}', 'model': model, 'taught': taught,
             'support': support, 'camera': T[:3, 3], 'fit': fit_object, 'lock': threading.Lock(),
             'cloud': None, 'image': None, 'info': None, 'now': None, 'offset': None,
             'offset_at': -1.0, 'advice': 'waiting for the camera', 'running': True}

        def on_cloud(msg):
            T_now = calib.T_base_optical(msg.header.frame_id or calib.cal.optical_frame)
            if T_now is not None:
                xyz = pointcloud2_to_xyz(msg)
                with g['lock']:
                    g['cloud'] = (time.monotonic(), xyz @ T_now[:3, :3].T + T_now[:3, 3])

        self.create_subscription(PointCloud2, g['topic'], on_cloud, 1)
        self.create_subscription(Image, p('replay_color_topic'),
                                 lambda m: g.__setitem__('image', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('replay_camera_info_topic'),
                                 lambda m: g.__setitem__('info', m), qos_profile_sensor_data)
        taught_pub = self.create_publisher(PointCloud2, '/tpgpt/replay/taught_object', 1)
        now_pub = self.create_publisher(PointCloud2, '/tpgpt/replay/object_now', 1)
        overlay_pub = self.create_publisher(Image, '/tpgpt/replay/place_overlay', 1)
        taught_msg = xyz_to_pointcloud2(taught.full_cloud, self.base,
                                        self.get_clock().now().to_msg(),
                                        rgb=np.array([0, 255, 0], dtype=np.uint8))
        T_opt_base = cc.invert_transform(T)

        def render():
            import cv2

            taught_pub.publish(taught_msg)
            msg = g['image']
            if msg is None or msg.encoding not in ('rgb8', 'bgr8') or not g['running']:
                return
            rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
            img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1].copy() if msg.encoding == 'rgb8' else img.copy()
            info = g['info']
            intr = (cc.Intrinsics.from_camera_info_fields(info.k, info.width, info.height)
                    if info is not None else calib.cal.intrinsics_yaml)
            now = g['now']
            # outlines, not fills: the real object stays visible inside them
            for pts, colour in ((None if now is None else now.full_cloud, (0, 140, 255)),
                                (taught.full_cloud, (0, 255, 0))):
                if pts is None:
                    continue
                uv, inside = cc.project(cc.transform_points(T_opt_base, pts), intr)
                mask = np.zeros(img.shape[:2], np.uint8)
                for u, v in uv[inside].astype(int):
                    cv2.circle(mask, (int(u), int(v)), 3, 255, -1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, -1, colour, 2)
            advice = g['advice']
            cut = advice.find(': ')
            lines = ([f"green: where the {name} was taught   orange: where the camera sees it"]
                     + ([advice[:cut + 1], '  ' + advice[cut + 2:]] if cut > 0 else [advice]))
            for row, text in enumerate(lines):
                cv2.putText(img, text, (8, 20 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (8, 20 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
            out = Image(header=msg.header, height=img.shape[0], width=img.shape[1],
                        encoding='bgr8', is_bigendian=0, step=img.shape[1] * 3)
            out.data = img.tobytes()
            overlay_pub.publish(out)

        def worker():
            seen, said, said_at = -1.0, None, -1e9
            while g['running'] and rclpy.ok():
                with g['lock']:
                    cloud = g['cloud']
                if cloud is None or cloud[0] <= seen:
                    time.sleep(0.2)
                    continue
                seen = cloud[0]
                fit = fit_object(model, cloud[1], support, g['camera'])
                g['now'] = fit
                now_pub.publish(xyz_to_pointcloud2(fit.full_cloud, self.base,
                                                   self.get_clock().now().to_msg(),
                                                   rgb=np.array([255, 140, 0], dtype=np.uint8)))
                if fit.accepted:
                    off = rp.placement_offset(taught, fit)
                    g['offset'], g['advice'] = off, rp.placement_advice(off)
                else:
                    g['offset'] = None
                    g['advice'] = (f"the camera's cloud does not fit a {name} ({fit.reason}): is "
                                   'it alone and fully in view?')
                g['offset_at'] = time.monotonic()
                if g['running'] and (g['advice'] != said or time.monotonic() - said_at > 5.0):
                    print('  ' + g['advice'], flush=True)
                    said, said_at = g['advice'], time.monotonic()

        g['timer'] = self.create_timer(0.2, render)
        g['thread'] = threading.Thread(target=worker, daemon=True)
        g['thread'].start()
        return g

    def _place_object(self, g, interactive):
        """Guide the object onto its taught place; returns its last offset
        (``replay.placement_offset``), or None if the camera never fitted it.

        Interactive: the operator aligns it by the image (rqt_image_view on
        /tpgpt/replay/place_overlay) and the printed advice, and presses Enter. Within
        ``PLACE_TOLERANCE`` it goes on; outside, the operator may keep aligning or type 'go'
        to start anyway (the camera's own error at the table is a few mm, and the numbers are
        advice: the pass is safe to run -- it is the grasp that suffers). Not interactive (a
        dry run): one look.
        """
        from tpgpt.hardware import replay as rp

        def fresh(after, wait):
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and g['offset_at'] <= after:
                time.sleep(0.2)
            return g['offset_at'] > after

        try:
            if not interactive:
                if not fresh(-1.0, 10.0):
                    print(f"  no cloud on {g['topic']} for 10 s", flush=True)
                return g['offset']
            print(f"\nPut the {g['name']} where it was taught. Watch it on the image:\n"
                  '    ros2 run rqt_image_view rqt_image_view /tpgpt/replay/place_overlay\n'
                  f"the green dots are the taught place, the orange ones where the camera fits the "
                  f"{g['name']} now (every cloud, ~3 s); the line under them is also printed here. "
                  f'Tolerance {rp.PLACE_TOLERANCE_M * 1000:.0f} mm and '
                  f'{rp.PLACE_TOLERANCE_DEG:.0f} deg.', flush=True)
            while True:
                input(f"Press Enter when the {g['name']} matches the green: ")
                if not fresh(time.monotonic(), 8.0):
                    print(f"  no fit since then: are the camera and segmentation running, and the "
                          f"{g['name']} alone in view?", flush=True)
                    continue
                off = g['offset']
                if off is not None and off['in_place']:
                    print(f"  {g['advice']}", flush=True)
                    return off
                answer = input(f"  It reads: {g['advice']}. Enter to keep aligning, 'go' to "
                               'start anyway: ').strip().lower()
                if answer == 'go':
                    self.get_logger().warn(f"starting with the {g['name']} off its taught place: "
                                           f"{g['advice']}")
                    return off
        finally:
            g['running'] = False
            self.destroy_timer(g['timer'])

    def save_pending_record(self):
        """After an aborted replay pass (and its hold): save what it recorded, as aborted_<n>."""
        if self._pending_record is None:
            return
        from tpgpt.hardware import replay as rp

        root, k, rec = self._pending_record
        self._pending_record = None
        path = rp.save_diagnostic(root, k, 'aborted', **rec, **self._take_highrate())
        self.get_logger().error(f'Pass {k} stopped before its end; what it recorded is in {path}')

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
                stage, learned=False):
        """Servo onto the plan's grasp (or release) pose before the gripper acts.

        The policy clock is paused. The target is the transported label itself, not
        the executor's attractor: the attractor can be dragged off the path when the
        lag clamp engages (a red-cup run closed 82 mm from its grasp that way). A
        bounded integral term removes whatever resting error the static
        compensation leaves at this pose.

        A **learned** plan's labels are a command offset from where the hand should be,
        so its target is the plan's mark (``Plan.marks``: the taught path transported),
        the orientation the learned command's own (it is what puts the hand at the planned
        orientation), and the command-to-target check -- which caught the clamp dragging
        the attractor off -- does not apply: its command is meant to be off. Returns the
        last commanded TCP position; for a learned plan the caller hands the residual to
        the executor's correction instead of the static offset.
        """
        from tpgpt.hardware.executor import crisp_applied_gains, logged_stiffness

        p, log = self.p, self.get_logger()
        self.stage = stage
        labels = plan.labels
        if learned:
            row = 0 if stage == 'arrive_grasp' else 1
            target_R = np.asarray(cmd.tcp_rotation, dtype=float)
            target_p = executor.tcp_from_tool(plan.marks['positions'][row],
                                              np.asarray(plan.marks['orientations'][row]))
        else:
            target_R = np.asarray(labels.orientations[index], dtype=float)
            target_p = executor.tcp_from_tool(labels.positions[index], target_R)
        gap = float(np.linalg.norm(cmd.tcp_position - target_p))
        if not learned and gap > float(p('arrival_max_gap')):
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
        # Walk the command from where it is onto the target at approach speed. A learned
        # command sits off the target by the arm's own offset, which it keeps: it is moved by
        # how far the *arm* is from the target, and the integral does the rest.
        step = float(p('approach_max_linear_speed')) * dt
        command = cmd.tcp_position.copy()
        goal = target_p
        if learned:
            goal = command + (target_p - executor.tcp_from_tool(
                executor.tool_position(*self.measured()), target_R))
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
            delta = goal - command
            command = command + delta * min(1.0, step / max(float(np.linalg.norm(delta)), 1e-9))
            if float(np.linalg.norm(goal - command)) < 1e-6:
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
        # a grasp and the pose-dependent error both persist. (A learned plan's executor
        # takes it into its path correction instead: the caller.)
        if not learned:
            self.offset = self.offset - integral
        self.stage = 'policy'
        return out

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
        if node.replay:
            node.run_replay()
        else:
            node.run()
    except AbortRun as exc:
        node.get_logger().error(f'Aborted: {exc}')
        if node.moving:
            node.hold_current_pose()
        node.save_pending_record()
        exit_code = 1
    except KeyboardInterrupt:
        if node.moving:
            node.hold_current_pose()
        node.save_pending_record()
    except Exception:
        # A bug must not leave the arm on a stale target: hold, keep the record, then report.
        # (2026-09-30: a replay pass died on its first step with the arm activated.)
        if node.moving:
            node.hold_current_pose()
        node.save_pending_record()
        raise
    finally:
        time.sleep(0.2)  # let a final hold command go out
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
