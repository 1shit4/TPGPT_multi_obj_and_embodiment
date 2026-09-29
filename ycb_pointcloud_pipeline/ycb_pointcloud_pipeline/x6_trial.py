#!/usr/bin/env python3
"""
One recorded X6 trial: tpgpt_pick_place with experiment:=true.

    ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
        -p experiment:=true -p trial_id:=0001 -p hand:=panda \
        -p object_name:=apple -p destination_name:=slot_a  [-p dry_run:=true] [-p top_down_grasp:=true]

Sequence
  1. load the hand file and capture fresh segmentations of the object (no
     pre-flight checks)
  2. plan ONCE with the same planner code as tpgpt_pick_place (no re-plan)
  3. write perception frames, candidates.json, keypoints.npz, plan.npy,
     plan_overlay.png, map scalars, timings
  4. execute with the same executor code, recording trace.npz every control step
  5. wait for Ctrl-C. Then: capture the settled scene, measure the placement,
     ask the operator the outcome questions, write record.json, exit.

Ctrl-C at ANY point (planning, execution, waiting) stops there, holds the arm if it
was moving, and goes to step 5 with whatever was recorded. Only the parameters in
x6_common.TRIAL_CLI_PARAMETERS may be given; everything else comes from
x6_experiment/config/x6_experiment.yaml.
"""

import csv
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions

from ycb_pointcloud_pipeline import x6_common as xc
from ycb_pointcloud_pipeline import x6_record as xr

TOP_LEVEL_FROZEN = ('experiment', 'demonstration', 'objects', 'destinations', 'method')


# ============================================================================ helpers
def frozen_digest(config):
    return xc.sha256_bytes(json.dumps({k: config.raw.get(k) for k in TOP_LEVEL_FROZEN},
                                      sort_keys=True, default=str).encode())


def pose7(position, rotation):
    if position is None or rotation is None:
        return None
    return list(np.asarray(position, dtype=float)) + list(xr.rotations_to_quat_xyzw(rotation))


def ask(prompt, choices=None, default=None, allow_blank=True):
    """Terminal question. Returns the answer string ('' if blank)."""
    suffix = ''
    if choices:
        suffix = f" [{'/'.join(choices)}]"
    if default is not None:
        suffix += f' (Enter = {default})'
    while True:
        try:
            answer = input(f'{prompt}{suffix}: ').strip()
        except EOFError:
            # stdin closed: a question with choices and no default would loop forever
            if choices and default is None:
                raise KeyboardInterrupt from None
            answer = ''
        if not answer and default is not None:
            return default
        if choices is None:
            if answer or allow_blank:
                return answer
        elif answer.lower() in [c.lower() for c in choices]:
            return answer.lower()
        print(f'  please answer one of {choices}')


def ask_yes_no(prompt, default=None):
    d = None if default is None else ('y' if default else 'n')
    return ask(prompt, ['y', 'n'], d) == 'y'


# ============================================================================ trial
class Trial:
    def __init__(self, config, trial_id, hand, object_name, destination_name, dry_run,
                 top_down_grasp=False):
        self.cfg = config
        self.trial_id = trial_id
        self.hand_name = hand
        self.object_name = object_name
        self.destination_name = destination_name
        self.dry_run = dry_run
        self.top_down_grasp = top_down_grasp
        self.scratch = dry_run or trial_id.lower().startswith('s')
        self.dir = config.trial_dir(trial_id, scratch=self.scratch)
        self.m = config.method
        self.rate = config.control_rate
        self.trace = xr.TraceRecorder(self.rate)
        self.record = {}
        self.status = 'started'
        self.system_abort_reason = None
        self.moved = False
        self.planner = self.executor_node = self.capture = None
        self._threads = []
        self._stop = threading.Event()
        self.planner_spin = threading.Event()
        self.plan = self.selection = None
        self.hand = None
        self.prov = None
        self.frames = []
        self.retry_captures = 0
        self.approach_blocked = False

    # ------------------------------------------------------------------ files
    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def save_record(self):
        self.record['status'] = self.status
        xr.write_record(self.dir, self.record)

    def save_trace(self):
        if len(self.trace):
            self.trace.save(self.path('trace.npz'))

    def save_cartesian_outputs(self):
        """cartesian_positions.csv and trajectory_3d.png in the trial directory. A
        plotting failure is reported and never loses the trial."""
        if not len(self.trace):
            return
        try:
            xr.write_cartesian_positions(self.path('cartesian_positions.csv'), self.trace)
            xr.render_trajectory_3d(self.path('trajectory_3d.png'), self.trace, self.plan,
                                    f'trial {self.trial_id}: {self.object_name} -> '
                                    f'{self.destination_name} ({self.hand_name})')
        except Exception as exc:  # noqa: BLE001
            print(f'WARNING: cartesian_positions.csv / trajectory_3d.png not written: '
                  f'{type(exc).__name__}: {exc}')

    # ------------------------------------------------------------------ run
    def run(self):
        if not self._claim_directory():
            return 2
        self.record = {
            'trial_id': self.trial_id, 'run_id': self.cfg.run_id,
            'step_index_rate_hz': self.rate, 'scratch': self.scratch, 'dry_run': self.dry_run,
            'top_down_grasp': self.top_down_grasp,
            'conditions': {'hand': self.hand_name, 'object': self.object_name,
                           'destination': self.destination_name},
        }
        self.save_record()
        print(f'\nX6 trial {self.trial_id}: hand {self.hand_name}, object {self.object_name}, '
              f'destination {self.destination_name}'
              f"{'  [top-down grasp]' if self.top_down_grasp else ''}"
              f"{'  [SCRATCH - not analysed]' if self.scratch else ''}\n  -> {self.dir}\n")
        try:
            self._load_setup()
            self._start_nodes()
            self._capture_object()
            self._campaign_bookkeeping()
            if not self._plan():
                self.status = 'preflight_failed' if self.approach_blocked else 'planning_failed'
                self._finalize(settle=False)
                return 0
            self._execute()
        except KeyboardInterrupt:
            self._hold_if_moving()
            if self.status in ('started', 'planning', 'planned', 'executing'):
                self.status = 'stopped_by_operator'
                self.system_abort_reason = 'Ctrl-C during ' + (self.executor_node.stage
                                                              if self.executor_node else 'planning')
            self._finalize(settle=self.moved)
            return 0
        except Exception as exc:  # noqa: BLE001 - the directory must survive any failure
            self._hold_if_moving()
            self.status = 'error'
            self.system_abort_reason = f'{type(exc).__name__}: {exc}'
            self.record['error_traceback'] = traceback.format_exc()
            print(traceback.format_exc())
            self._finalize(settle=self.moved)
            return 1
        finally:
            self.save_trace()
            self.save_cartesian_outputs()
            self.save_record()
            self._shutdown_nodes()
        return 0

    def _claim_directory(self):
        if os.path.isdir(self.dir):
            print(f'Trial {self.trial_id} already exists: overwriting {self.dir}')
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)
        return True

    # ------------------------------------------------------------------ setup
    def _load_setup(self):
        """Hand file, open session and git provenance. No checks: nothing here stops
        the trial."""
        self.hand = self.cfg.hand(self.hand_name)
        self.session = xc.current_session(self.cfg)
        self.prov = xc.provenance(self.cfg)

    def _retry_capture(self, seen, wait):
        """The planner's scan-fit retry (TpgptPlannerNode.await_fresh_clouds): capture
        n_frames fresh segmentations, record them with the others, feed them in."""
        per = self.m['perception']
        n = int(per['n_frames'])
        timeout = float(self.m['outcome'].get('capture_timeout_s', 20.0))
        print(f'  scan fit refused; capturing {n} fresh segmentations to try again...')
        frames, freshness = self.capture.capture(n, timeout)
        print(f'  {len(frames)}/{n} fresh clouds ({freshness})')
        self.retry_captures += 1
        self.frames.extend(frames)
        for f in frames:
            self.planner._cloud_cb(f['cloud'])

    def _capture_object(self):
        """Fresh segmentations of the object, used for planning."""
        per = self.m['perception']
        n = int(per['n_frames'])
        timeout = float(self.m['outcome'].get('capture_timeout_s', 20.0))
        print(f'Waiting for {n} fresh segmentations of {self.object_name} (up to {timeout:.0f} s)...')
        frames, freshness = self.capture.capture(n, timeout)
        self.frames = frames
        fused, stats = xr_fuse(frames, self.planner.calib, per, per['workspace_min'],
                               per['workspace_max'])
        print(f'  {len(frames)}/{n} fresh clouds ({freshness}); fused {len(fused)} points; {stats}')

    # ------------------------------------------------------------------ nodes
    def _start_nodes(self):
        from ycb_pointcloud_pipeline.tpgpt_planner_node import TpgptPlannerNode
        from ycb_pointcloud_pipeline.x6_capture import X6CaptureNode
        topics = self.cfg.raw['experiment']['topics']
        self.capture = X6CaptureNode(self.object_name, topics,
                                     robot_state_topic=topics.get('robot_state'),
                                     gripper_state_topic=self.hand['gripper'].get('width_topic'))
        cap_exec = SingleThreadedExecutor()
        cap_exec.add_node(self.capture)
        self._spin(cap_exec, always=True)
        os.makedirs(self.path('_planner_out'), exist_ok=True)
        self.planner = TpgptPlannerNode(parameter_overrides=to_parameters(self._planner_params()))
        plan_exec = MultiThreadedExecutor(num_threads=3)
        plan_exec.add_node(self.planner)
        self.planner_spin.set()
        self._spin(plan_exec, gate=self.planner_spin)
        self._executors = [cap_exec, plan_exec]

    def _spin(self, executor, always=False, gate=None):
        def loop():
            while not self._stop.is_set():
                if always or gate.is_set():
                    try:
                        executor.spin_once(timeout_sec=0.05)
                    except Exception:  # noqa: BLE001 - shutting down
                        if self._stop.is_set():
                            return
                else:
                    time.sleep(0.05)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._threads.append(t)

    def _shutdown_nodes(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for node in (self.executor_node, self.planner, self.capture):
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:  # noqa: BLE001
                    pass

    def _planner_params(self):
        """The frozen method (x6_common.PLANNER_METHOD_PARAMETERS) plus what only a
        trial decides. Every other planner parameter stays at its node default."""
        cfg = self.cfg
        return {
            **xc.planner_method_parameters(self.m),
            'calibration_yaml': xc.expand(cfg.raw['calibration']['camera_extrinsics_file']),
            'source_bundle_dir': xc.expand(cfg.raw['demonstration']['source_bundle_dir']),
            'shelf_config': cfg.shelf_file,
            'object_name': self.object_name,
            # Resolved by the planner from the shelf file's slot marks, exactly as outside
            # an experiment; the support is inferred (the table).
            'destination_name': self.destination_name,
            'top_down_grasp': bool(self.top_down_grasp),
            'cloud_topic': '',
            # The trial captures its own n_frames (outcome.capture_timeout_s) and feeds them
            # in; the planner must not wait for more on a subscription it is not spinning.
            'frames_wait': 0.0,
            'graspgen_gripper': str(self.hand.get('graspgen_gripper') or 'franka_panda'),
            # Hand on the flange. inspire: scripted side grasp, GraspGen-X is never called.
            'end_effector': str(self.hand.get('end_effector', self.hand_name)),
            'end_effector_config': xc.expand(self.hand['end_effector_config'])
            if self.hand.get('end_effector_config') else '',
            'graspgen_autostart': True,
            'output_dir': self.path('_planner_out'),
            'auto_plan': False,
        }

    def _executor_params(self, plan_dir):
        ex = self.m['execution']
        ints = {'progress_window'}
        params = {'plan_dir': plan_dir, 'dry_run': bool(self.dry_run),
                  'log_dir': xc.expand(f'~/.ros/x6_executor_csv/{self.cfg.run_id}_{self.trial_id}')}
        for k, v in ex.items():
            if k == 'allow_replan':
                continue
            if isinstance(v, bool) or isinstance(v, str):
                params[k] = v
            elif isinstance(v, list):
                params[k] = [float(x) for x in v]
            elif k in ints:
                params[k] = int(v)
            else:
                params[k] = float(v)
        g = self.hand['gripper']
        params.update({'gripper_mode': str(g['mode']), 'gripper_namespace': str(g['namespace']),
                       'tool_offset': [float(v) for v in self.hand['tool_offset']]})
        # Franka Hand goal settings; the Inspire hand's speed/force live in its own config.
        for key, name in (('open_width_m', 'open_width'), ('grasp_speed_mps', 'grasp_speed'),
                          ('grasp_force_n', 'grasp_force'), ('timeout_s', 'gripper_timeout')):
            if key in g:
                params[name] = float(g[key])
        return params

    # ------------------------------------------------------------------ campaign files
    def _campaign_bookkeeping(self):
        cfg = self.cfg
        root = cfg.root
        os.makedirs(root, exist_ok=True)
        demo = cfg.raw['demonstration']
        manifest_path = os.path.join(root, 'manifest.json')
        if not self.scratch:
            if not os.path.isfile(manifest_path):
                src = xc.expand(demo['source_bundle_dir'])
                dst = os.path.join(cfg.demo_copy_dir(), 'source_bundle')
                if not os.path.isdir(dst):
                    shutil.copytree(src, dst)
                    parent = os.path.dirname(src)
                    for extra in ('teach.json', 'raw.npz'):
                        if os.path.isfile(os.path.join(parent, extra)):
                            shutil.copyfile(os.path.join(parent, extra),
                                            os.path.join(cfg.demo_copy_dir(), extra))
                with open(os.path.join(cfg.demo_copy_dir(), 'sha256.txt'), 'w') as f:
                    for name, digest in xc.demo_hashes(dst).items():
                        f.write(f'{digest}  source_bundle/{name}\n')
                os.makedirs(os.path.join(root, 'config_snapshot'), exist_ok=True)
                shutil.copyfile(cfg.path, os.path.join(root, 'config_snapshot', 'x6_experiment.yaml'))
                xc.write_json(manifest_path, {
                    'run_id': cfg.run_id, 'config_path': cfg.path, 'config_sha256': cfg.sha256,
                    'frozen_digest': frozen_digest(cfg), 'frozen_blocks': list(TOP_LEVEL_FROZEN),
                    'demonstration_sha256': xc.demo_hashes(dst),
                    'provenance_at_first_trial': self.prov, 'first_trial_id': self.trial_id,
                    'hands_frozen': {},
                })
            man = xc.read_json(manifest_path)
            if self.hand_name not in man.get('hands_frozen', {}):
                man.setdefault('hands_frozen', {})[self.hand_name] = self.hand['_sha256']
                os.makedirs(os.path.join(root, 'hands'), exist_ok=True)
                shutil.copyfile(self.hand['_path'], os.path.join(root, 'hands',
                                                                  f'{self.hand_name}.yaml'))
                xc.write_json(manifest_path, man)
        # session order
        order = None
        session = getattr(self, 'session', None)
        if session is not None:
            log = os.path.join(cfg.sessions_dir(), session['session_id'], 'session_log.jsonl')
            order = 1
            if os.path.isfile(log):
                with open(log) as f:
                    order = 1 + sum(1 for _ in f)
        cal_file = xc.expand(cfg.raw['calibration']['camera_extrinsics_file'])
        self.record['conditions'].update({
            'destination_xyz': [float(cfg.destination(self.destination_name)[k])
                                for k in ('x', 'y', 'surface_height')],
            'order_in_session': order,
            'session_id': session['session_id'] if session else None,
        })
        self.record['provenance'] = {
            **{k: self.prov[k] for k in ('git_sha', 'dirty', 'modified_files',
                                         'untracked_code_files', 'package_versions', 'venvs',
                                         'python', 'platform')},
            'config_sha256': cfg.sha256, 'frozen_config_digest': frozen_digest(cfg),
            'demonstration_sha256': xc.demo_hashes(demo['source_bundle_dir']),
            'calibration': {'camera_extrinsics_file': cal_file,
                            'camera_extrinsics_sha256': xc.sha256_file(cal_file),
                            'shelf_file': cfg.shelf_file,
                            'shelf_file_sha256': xc.sha256_file(cfg.shelf_file),
                            'hand_frame_file': self.hand['_path'],
                            'hand_frame_file_sha256': self.hand['_sha256']},
        }
        self.save_record()

    # ------------------------------------------------------------------ plan
    def _plan(self):
        from ycb_pointcloud_pipeline.x6_capture import save_frames
        planner = self.planner
        # Plan on exactly the captured clouds, through the planner's own callback.
        self.planner_spin.clear()
        time.sleep(0.2)
        with planner.lock:
            planner.clouds = []
        for f in self.frames:
            planner._cloud_cb(f['cloud'])
        # A refused scan fit is retried on fresh frames (object_model_retries); the
        # planner's own wait would see none, its subscription is not spun here.
        planner.await_fresh_clouds = self._retry_capture
        print('Planning (once; no re-plan in an experiment trial)...')
        self.status = 'planning'
        ok, text = planner._plan()
        frame_meta = save_frames(self.dir, self.frames, planner.calib)
        info = planner.last_plan_inputs or {}
        points = info.get('points')
        if points is not None:
            np.save(self.path('cloud.npy'), np.asarray(points, dtype=np.float32))
        det_info = next((f['info'] for f in reversed(self.frames) if f['info']), None)
        det = None
        if det_info:
            det = next((d for d in det_info.get('detections', [])
                        if d['label'] == self.object_name), None)
        cam = self.frames[-1]['camera_info'] if self.frames else None
        T = planner.calib.T_base_optical(self.frames[-1]['cloud'].header.frame_id) if self.frames else None
        marks = [p for p in self.cfg.raw.get('pick_poses', []) if not xc.is_unfilled(p)]
        self.record['perception'] = {
            'cloud_points': None if points is None else int(len(points)),
            'n_frames_fused': info.get('n_clouds', len(self.frames)),
            'n_frames_captured': len(self.frames), 'retry_captures': self.retry_captures,
            'frames': frame_meta,
            'camera_intrinsics': None if cam is None else list(cam.k),
            'camera_to_base': None if T is None else np.asarray(T).tolist(),
            'segmentation_source': {
                'detector': None if det_info is None else {k: det_info.get(k) for k in (
                    'detector_mode', 'grounding_model', 'sam_model', 'box_threshold',
                    'text_threshold', 'target_objects_yaml')},
                'prompt': None if det_info is None else det_info.get('prompts', {}).get(self.object_name),
                'score': None if det is None else det['score'],
            },
        }
        if points is not None:
            self.record['conditions']['pick_pose'] = xr.pick_pose_from_cloud(points, marks)
            self.record['conditions']['pick_pose_index'] = \
                self.record['conditions']['pick_pose']['nearest_mark_index']
        timing = dict(info.get('timing_s', {}))
        poses, scores = info.get('poses'), info.get('scores')
        if not ok:
            cands = info.get('failed_candidates') or []
            sel = SimpleNamespace(candidates=cands, chosen=None, source=None,
                                  notes=[info.get('error', text)],
                                  stages=info.get('failed_stages'), rank_examined=None,
                                  world=info.get('world'), fell_back=False)
            doc, grasp = xr.candidates_document(sel, poses, scores)
            doc['planning_error'] = info.get('error', text)
            if info.get('failed_fallback') is not None:
                doc['fallback'] = info['failed_fallback'].as_dict()
            xc.write_json(self.path('candidates.json'), doc)
            grasp['planning_error'] = info.get('error', text)
            self.record['grasp'] = grasp
            self.record['timing_s'] = timing
            self.system_abort_reason = f'planning refused: {text}'
            if str(info.get('error', text)).startswith('approach_blocked'):
                # The grasp and its path passed; only the move from where the arm was
                # parked failed. That is the operator's setup, not the method.
                self.approach_blocked = True
                self.system_abort_reason = f'{text} -- run this trial id again once homed'
                self.save_record()
                print(f'\nNOT A TRIAL (preflight_failed, not counted): {text}\n'
                      f'Re-home the arm and run trial {self.trial_id} again; it is overwritten.')
                return False
            self.save_record()
            print(f'\nPLANNING FAILED (counts as a trial, stage "planning"): {text}')
            return False

        selection = info['selection']
        plan = planner.latest_plan
        self.plan, self.selection = plan, selection
        plan_dir = self.path('plan_dir')
        os.rename(planner.plan_dir, plan_dir)
        planner.plan_dir = plan_dir
        shutil.rmtree(self.path('_planner_out'), ignore_errors=True)

        doc, grasp = xr.candidates_document(selection, poses, scores)
        xc.write_json(self.path('candidates.json'), doc)
        xr.save_keypoints(self.path('keypoints.npz'), plan)
        xr.save_plan_array(self.path('plan.npy'), plan)
        map_block, retimed = xr.map_block(plan, planner.source, self.m['map']['cube_half_extent_m'],
                                          self.m['map']['construction'])
        support = info['scene'].support_height
        if support is None:
            support = float(np.percentile(np.asarray(points)[:, 2], 1.0))
        t_screen, reproduced = xr.time_screening(points, poses, scores, support, info['settings'],
                                                 selection.candidates,
                                                 context=info.get('screen_context'))
        timing.update({'filtering': t_screen, 'map_fit': retimed['map_fit'],
                       'label_transport': retimed['label_transport'],
                       'total_plan': sum(timing.get(k, 0.0) for k in
                                         ('perception', 'grasp_generation', 'selection_total'))})
        timing['_note'] = ('perception, grasp_generation, selection_total measured in the planning '
                           'call; filtering (screening), map_fit and label_transport re-timed on '
                           'identical inputs; screening verdicts reproduced: ' + str(reproduced))
        self.record.update({'grasp': grasp, 'map': map_block, 'timing_s': timing})
        self._write_overlay(plan, selection)
        self.status = 'planned'
        self.save_record()
        g = grasp
        print(f"\nPLAN: rank {g['rank_executed']} of {g['n_candidates_generated']} "
              f"({'FELL BACK (' + g['fallback_kind'] + ')' if g['fell_back'] else 'generator grasp'}), "
              f"min det J {map_block['min_det_J']:.3f}, aim {map_block['aim_error_mm']:.1f} mm")
        print(f"      overlay: {self.path('plan_overlay.png')}  (also RViz /tpgpt/plan/*)")
        return True

    def _write_overlay(self, plan, selection):
        import cv2
        from cv_bridge import CvBridge
        frame = next((f for f in reversed(self.frames) if f['color'] is not None), None)
        if frame is None or frame['camera_info'] is None:
            color, info = self.planner._color_msg, self.planner._camera_info
        else:
            color, info = frame['color'], frame['camera_info']
        if color is None or info is None:
            print('WARNING: no colour image; plan_overlay.png not written')
            return
        bgr = CvBridge().imgmsg_to_cv2(color, desired_encoding='bgr8')
        T = self.planner.calib.T_base_optical(color.header.frame_id or
                                              self.planner.calib.cal.optical_frame)
        if T is None:
            return
        img = xr.render_plan_overlay(bgr, info.k, T, plan, selection,
                                     self.record['conditions']['destination_xyz'],
                                     title=f'trial {self.trial_id} {self.hand_name} '
                                           f'{self.object_name} -> {self.destination_name}')
        cv2.imwrite(self.path('plan_overlay.png'), img)

    # ------------------------------------------------------------------ execute
    def _execute(self):
        from ycb_pointcloud_pipeline.tpgpt_executor_node import AbortRun, TpgptExecutorNode
        ex = TpgptExecutorNode(parameter_overrides=to_parameters(
            self._executor_params(self.planner.plan_dir)))
        self.executor_node = ex
        ex_exec = SingleThreadedExecutor()
        ex_exec.add_node(ex)
        self._spin(ex_exec, always=True)
        ex.step_hook = self._step_hook
        self.status = 'executing'
        self.save_record()
        try:
            executed = ex.run()
        except AbortRun as exc:
            self.moved = self.moved or ex.moving or len(self.trace) > 0
            self._hold_if_moving()
            self.system_abort_reason = f'executor abort: {exc}'
            self.status = 'aborted_during_motion' if self.moved else 'aborted_before_motion'
            print(f'\nEXECUTION ABORTED: {exc}')
            self._record_execution_signals(completed=False)
            if not self.moved:
                self._finalize(settle=False)
                return
            self._wait_for_stop()
            self._finalize(settle=True)
            return
        self.moved = len(self.trace) > 0
        if not executed:
            self.status = 'refused_at_confirmation'
            self.system_abort_reason = 'operator did not type yes'
            self._finalize(settle=False)
            return
        self.status = 'executed'
        self._record_execution_signals(completed=True)
        self._retreat()
        self._wait_for_stop()
        self._finalize(settle=True)

    def _step_hook(self, position, rotation):
        from ycb_pointcloud_pipeline.tpgpt_executor_node import AbortRun
        ex = self.executor_node
        try:
            meas_p, meas_R = ex.measured(max_age=1.0)
            ee = pose7(meas_p, meas_R)
        except AbortRun:
            ee = None
        cmd = ex.last_cmd if ex.stage == 'policy' else None
        wrench, _, _ = self.capture.robot_state()
        gripper_cmd = None if ex.gripper_state is None else (1.0 if ex.gripper_state else -1.0)
        self.trace.add(ex.stage, ee, pose7(position, rotation),
                       None if cmd is None else cmd.attractor, self.capture.joint_positions(),
                       gripper_cmd, self.capture.gripper_width(), wrench,
                       None if cmd is None else cmd.phase, None if cmd is None else cmd.gate)
        self.moved = True
        if len(self.trace) % 200 == 0:
            self.save_trace()

    def _record_execution_signals(self, completed):
        ex = self.executor_node
        hx = ex.policy_executor if ex is not None else None
        widths = [w for w, g in zip(self.trace.rows['gripper_width'], self.trace.rows['gripper_command'])
                  if w is not None and g is not None and g > 0]
        self.record.setdefault('timing_s', {})
        self.record['timing_s']['policy_refit'] = None if ex is None else ex.policy_fit_seconds
        self.record['timing_s']['execution'] = self.trace.policy_steps() / self.rate
        self.record['timing_s']['execution_all_stages'] = len(self.trace) / self.rate
        self.record['execution'] = {
            'policy_completed': bool(completed),
            'control_steps_total': len(self.trace),
            'policy_steps': self.trace.policy_steps(),
            'final_phase': None if hx is None else float(hx.phase),
            'stalled': None if hx is None else bool(hx.stalled),
            'gripper_width_after_close_m': widths[-1] if widths else None,
            'gripper_width_min_while_closed_m': min(widths) if widths else None,
            'last_stage': None if ex is None else ex.stage,
        }
        self.save_trace()
        self.save_record()

    def _retreat(self):
        pose = self.m['outcome'].get('retreat_tcp_pose')
        if pose is None or xc.is_unfilled(pose):
            return
        from scipy.spatial.transform import Rotation
        from tpgpt.hardware.executor import approach_trajectory
        ex = self.executor_node
        pos, rot = ex.measured(max_age=1.0)
        tgt_p = np.asarray(pose[:3], dtype=float)
        tgt_R = Rotation.from_quat(pose[3:7]).as_matrix()
        dt = 1.0 / self.rate
        ps, Rs = approach_trajectory(pos, rot, tgt_p, tgt_R, dt,
                                     float(ex.p('approach_max_linear_speed')),
                                     float(ex.p('approach_max_angular_speed')))
        print(f'Retreating to the camera-clear pose ({len(ps) * dt:.1f} s)...')
        ex.stage = 'retreat'
        K, D = np.eye(3) * 400.0, np.eye(3) * 40.0
        for p_, R_ in zip(ps, Rs):
            ex.check_workspace(p_, 'retreat target')
            meas_p, _ = ex.measured()
            if float(np.linalg.norm(meas_p - p_)) > float(ex.p('max_tracking_error')):
                ex.hold_current_pose()
                print('Retreat stopped: tracking error; the arm holds where it is.')
                return
            ex.publish(p_, R_)
            if ex.dry:
                ex.simulate(p_, R_, K, D, dt)
            time.sleep(dt)

    def _hold_if_moving(self):
        ex = self.executor_node
        if ex is not None and (ex.moving or self.moved):
            try:
                ex.hold_current_pose()
            except Exception:  # noqa: BLE001
                pass

    def _wait_for_stop(self):
        print('\n' + '=' * 78)
        print(f'Trial {self.trial_id}: {self.status.upper()}.  The controller holds the arm.')
        print('Clear the camera view of the destination and let the object come to rest,')
        print('then press Ctrl-C to finish the trial (settled capture + questions).')
        print('=' * 78)
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print()

    # ------------------------------------------------------------------ finish
    def _finalize(self, settle):
        """Settled capture (if anything moved), operator questions, record."""
        outcome = self.record.setdefault('outcome', {})
        outcome.update({'status': self.status, 'system_abort_reason': self.system_abort_reason})
        self.save_record()
        try:
            if settle:
                self._settled_capture(outcome)
            self._questions(outcome)
        except KeyboardInterrupt:
            outcome['questions_incomplete'] = True
            print('\nStopped during the questions; writing what was answered.')
        outcome['status'] = self.status
        excluded = bool(outcome.get('excluded'))
        outcome['counted_in_denominator'] = (not excluded and not self.scratch
                                             and self.status != 'preflight_failed')
        self.save_trace()
        self.save_record()
        self._append_logs()
        print(f'\nTrial {self.trial_id} written: {self.dir}\n')

    def _settled_capture(self, outcome):
        from ycb_pointcloud_pipeline.x6_capture import save_frames
        answer = ask('\nArm clear of the view and object at rest? Press Enter to capture the '
                     'settled scene, or type s to skip', default='capture')
        if answer.lower().startswith('s'):
            outcome['settled_capture'] = 'skipped by operator'
            return
        per, oc = self.m['perception'], self.m['outcome']
        n = int(per['n_frames'])
        print(f'Capturing {n} fresh segmentations of the settled {self.object_name}...')
        frames, freshness = self.capture.capture(n, float(oc.get('capture_timeout_s', 20.0)))
        meta = save_frames(self.dir, frames, self.planner.calib, prefix='settled_')
        fused, stats = xr_fuse(frames, self.planner.calib, per, oc['settled_workspace_min'],
                               oc['settled_workspace_max'])
        np.save(self.path('settled_cloud.npy'), np.asarray(fused, dtype=np.float32))
        predicted = None
        if self.plan is not None:
            predicted = self.plan.diagnostics.get('target_place_points')
        err = xr.placement_error(fused, predicted, self.record['conditions']['destination_xyz'])
        outcome['settled_capture'] = {'frames': meta, 'freshness': freshness,
                                      'fused_points': int(len(fused)), 'fusion': stats,
                                      'workspace_min': oc['settled_workspace_min'],
                                      'workspace_max': oc['settled_workspace_max']}
        outcome['placement'] = err
        if err and 'placement_error_mm' in err:
            outcome['placement_error_mm'] = err['placement_error_mm']
            outcome['placement_error_components_mm'] = err['placement_error_components_mm']
            c = err['placement_error_components_mm']
            print(f"Settled object: {len(fused)} points. Placement error (lateral) "
                  f"{err['placement_error_mm']:.1f} mm  (x {c['x']:+.1f}, y {c['y']:+.1f}, "
                  f"z {c['z']:+.1f} mm vs the plan's predicted placed cloud)")
        elif len(fused) == 0:
            print(f'The {self.object_name} was NOT seen in the settled workspace '
                  f'({freshness}, {len(frames)} clouds).')
        else:
            print(f'Settled object seen ({len(fused)} points) but no predicted placement to compare.')

    def _questions(self, outcome):
        stages = list(xc.FAILURE_STAGES)
        print('\n--- Trial outcome (operator) ---')
        if self.status == 'planning_failed':
            outcome['placed'] = False
            outcome['first_failure_stage'] = 'planning'
            outcome['operator_aborted'] = False
        elif self.status in ('refused_at_confirmation', 'aborted_before_motion'):
            outcome['placed'] = False
            outcome['first_failure_stage'] = None
            outcome['operator_aborted'] = self.status == 'refused_at_confirmation'
            reason = ask('Why was it not executed? (safety concern, what you saw)', allow_blank=True)
            outcome['abort_reason'] = reason or self.system_abort_reason
        else:
            placed = ask_yes_no('Was the object placed in the destination slot?')
            outcome['placed'] = placed
            if placed:
                outcome['first_failure_stage'] = None
            else:
                print('  First stage that failed:')
                for k, s in enumerate(stages[1:], 1):
                    print(f'    {k} {s}')
                pick = ask('  Stage number', [str(k) for k in range(1, len(stages))])
                outcome['first_failure_stage'] = stages[int(pick)]
            outcome['operator_aborted'] = self.status == 'stopped_by_operator'
            if self.status in ('stopped_by_operator', 'aborted_during_motion', 'error'):
                reason = ask('Abort reason (Enter = keep the system reason)', allow_blank=True)
                outcome['abort_reason'] = reason or self.system_abort_reason
            else:
                outcome['abort_reason'] = None
            disturbed = ask('Object disturbed before the grasp, mm (estimate from video; Enter = '
                            'unknown)', allow_blank=True)
            try:
                outcome['object_disturbed_before_grasp_mm'] = float(disturbed) if disturbed else None
            except ValueError:
                outcome['object_disturbed_before_grasp_mm'] = None
                outcome['object_disturbed_before_grasp_note'] = disturbed
            outcome['object_disturbed_before_grasp_source'] = 'operator estimate from video'
        reasons = list(self.m['outcome'].get('exclusion_reasons', []))
        print('  Exclude from the denominator ONLY for a cause outside the method:')
        for k, r in enumerate(reasons, 1):
            print(f'    {k} {r}')
        excl = ask('  Exclude? 0 = no, or the reason number', [str(k) for k in range(len(reasons) + 1)],
                   default='0')
        outcome['excluded'] = excl != '0'
        outcome['exclusion_reason'] = reasons[int(excl) - 1] if excl != '0' else None
        if outcome['excluded']:
            outcome['exclusion_detail'] = ask('  Describe what happened', allow_blank=True)
        outcome['operator_notes'] = ask('Notes (anything worth knowing)', allow_blank=True)

    def _append_logs(self):
        cfg = self.cfg
        row = {'trial_id': self.trial_id, 'hand': self.hand_name, 'object': self.object_name,
               'destination': self.destination_name, 'status': self.status,
               'scratch': self.scratch,
               'placed': self.record.get('outcome', {}).get('placed'),
               'excluded': self.record.get('outcome', {}).get('excluded')}
        os.makedirs(cfg.root, exist_ok=True)
        index = os.path.join(cfg.root, 'trial_index.csv')
        # a re-run of the same trial id replaces its earlier row(s)
        rows = []
        if os.path.isfile(index):
            with open(index, newline='') as f:
                rows = [r for r in csv.DictReader(f)
                        if not (r.get('trial_id') == str(row['trial_id'])
                                and r.get('scratch') == str(row['scratch']))]
        with open(index, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row), extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
            w.writerow(row)
        session = getattr(self, 'session', None)
        if session is not None and self.status != 'preflight_failed':
            log = os.path.join(cfg.sessions_dir(), session['session_id'], 'session_log.jsonl')
            with open(log, 'a') as f:
                f.write(json.dumps(row) + '\n')


def xr_fuse(frames, calib, perception, wmin, wmax):
    from ycb_pointcloud_pipeline.x6_capture import fuse_frames
    return fuse_frames(frames, calib, perception, [float(v) for v in wmin],
                       [float(v) for v in wmax])


def to_parameters(mapping):
    return [Parameter(k, value=v) for k, v in mapping.items()]


# ============================================================================ main
def main(args=None):
    argv = sys.argv if args is None else args
    names = xc.cli_parameter_names(argv)
    bad = [n for n in names if n not in xc.TRIAL_CLI_PARAMETERS]
    if bad:
        print(f'REFUSED: parameter(s) {bad} cannot be set on an experiment trial. Only '
              f'{list(xc.TRIAL_CLI_PARAMETERS)} are allowed; everything else comes from the '
              'frozen config.')
        raise SystemExit(2)
    value = lambda n: xc.cli_parameter_value(argv, n)  # noqa: E731
    missing = [n for n in ('trial_id', 'object_name', 'destination_name') if not value(n)]
    config = xc.X6Config(value('x6_config'))
    mounted = config.raw['hands'].get('mounted')
    hand = (value('hand') or mounted or '').strip()
    if not hand:
        missing.append('hand')
    elif mounted and hand != mounted:
        print(f'WARNING: -p hand:={hand} but x6_experiment.yaml says hands.mounted: {mounted}. '
              'Running as requested.')
    if missing:
        print(f'REFUSED: experiment:=true needs -p {":=... -p ".join(missing)}:=...')
        raise SystemExit(2)
    trial_id = value('trial_id').strip().strip('"\'')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', trial_id):
        print(f'REFUSED: trial_id {trial_id!r} may only contain letters, digits, _ and -')
        raise SystemExit(2)
    dry_run = (value('dry_run') or 'false').strip().lower() in ('true', '1', 'yes')
    top_down_grasp = (value('top_down_grasp') or 'false').strip().lower() in ('true', '1', 'yes')

    # No rclpy signal handler: after Ctrl-C the ROS context must stay usable, to hold
    # the arm, capture the settled scene and ask the questions.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    code = 1
    try:
        trial = Trial(config, trial_id, hand, value('object_name').strip(),
                      value('destination_name').strip(), dry_run, top_down_grasp)
        code = trial.run()
    finally:
        rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
