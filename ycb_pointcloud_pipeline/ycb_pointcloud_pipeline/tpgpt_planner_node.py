#!/usr/bin/env python3
"""
TPGPT planning from real perception. Does NOT move the robot.

  /ycb_pointclouds/<object>  (camera_color_optical_frame, from segmentation_node)
      -> base frame via config/camera_calibration.yaml (+ realsense TF)
      -> fuse `n_frames` clouds, crop to the workspace, voxel + outlier filter
      -> grasp: GraspGen-X candidates for the Franka Hand (franka_panda) from the
         GraspGen-X ZMQ server, screened (approach from above, visibility, on
         target, what lies between the pads fits, table, shelf collision, place
         zone, centred, duplicates), then planned best score first; the first
         whose WHOLE transported path the arm can fly without touching the shelf
         wins (tpgpt.hardware.grasp_selection, tpgpt.hardware.path_check -- the
         simulation's _choose_grasp rule). The scripted top-down grasp is used
         only when asked for (top_down_grasp / grasp_fallback_top_down).
      -> what the arm can hit: config/shelf.yaml (the measured shelf, seen or not),
         everything /ycb_scene_cloud sees within reach (target, table and the
         robot at the /joint_states of the capture removed), and the arm itself. /joint_states now also seeds the path check with
         the arm's real configuration and checks the move onto the path.
      -> tpgpt.hardware.planner.plan(): keypoints (grasp-pose cube),
         transportation map, transported policy labels (fr3_hand_tcp)
      -> saved to <output_dir>/<object>_<timestamp>/ for tpgpt_executor_node
      -> RViz markers for inspection before anything moves

Published (latched / transient-local, fixed frame `base`):
  /tpgpt/plan/target_cloud     fused object cloud used for planning (orange)
  /tpgpt/plan/placed_cloud     where the plan will put that object (cyan)
  /tpgpt/plan/markers          grasp candidates (chosen: thick green; planned but
                               refused: orange; screened out: faint red),
                               target keypoints (green), grasp and release
                               fr3_hand_tcp axes (x red, y green = finger travel,
                               z blue = approach), transported path (blue open /
                               red closed), destination disc, metrics text
  /tpgpt/plan/path             nav_msgs/Path of fr3_hand_tcp poses
  /tpgpt/plan/shelf            the shelf model's panels (brown; support boards darker)
  /tpgpt/plan/shelf_cloud      observed obstacle points (magenta)
  /tpgpt/plan/overlay          the camera image with the fused cloud (green dots), the
                               chosen grasp's Franka Hand collision boxes (magenta, open)
                               and TCP (red cross) projected through the calibration; also
                               saved as <plan>/overlay.png. The cloud is drawn in its real
                               colours on /tpgpt/plan/target_cloud.

Inspire RH56DFX-2R (end_effector:=inspire): GraspGen-X grasps for its own description
(config graspgenx.gripper, `inspire_hand`; graspgen_gripper follows the hand), judged on
a model of the hand (tpgpt.hardware.inspire_model) mounted as config/inspire_hand.yaml
says (end_effector_config). The plan records the hand in metadata['end_effector'],
which is where the executor reads the grasp point's offset from.

    ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
        -p end_effector:=inspire -p object_name:=apple -p destination_name:=top_left

Uses the GraspGen-X server (grasp_source:=graspgenx, the default). If it is not
running, the node starts it on the GPU (graspgen_autostart:=true, ~5 s). By hand:
    ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh --daemon

Must run with the TPGPT Python environment (it imports `tpgpt`):
    source ~/venvs/tpgpt/bin/activate
    ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
        -p object_name:=mustard_bottle -p destination:="[0.55, 0.25, 0.02]"
    ros2 service call /tpgpt_planner_node/plan std_srvs/srv/Trigger
"""

import json
import os
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from ycb_pointcloud_pipeline import camera_calibration as cc
from ycb_pointcloud_pipeline.camera_tf_publisher import default_calibration_yaml
from ycb_pointcloud_pipeline.ros_utils import (
    CalibrationProvider, axes_markers, box_marker, colors_for, cylinder_marker, fuse_clouds,
    gripper_segments, hand_axis_segments, line_list_marker, line_strip_marker,
    matrix_to_pose_stamped, piece_segments, pointcloud2_to_xyzrgb, sphere_list_marker,
    text_marker, xyz_to_pointcloud2,
)

NAN = float('nan')


def _finite(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


class TpgptPlannerNode(Node):
    def __init__(self, **kwargs):
        super().__init__('tpgpt_planner_node', **kwargs)
        d = self.declare_parameter
        d('calibration_yaml', default_calibration_yaml())
        # The demonstration every robot run since 2026-10-01 used (54 Franka plans, passed by
        # hand) and X6 uses. The default was the simulation's 09-13 export, from before the
        # move; the Inspire grasp tests took it unseen, and from 2026-10-06 a grasp test reads
        # its execution settings from the bundle (2026-10-06 18:57 ran at the old ones).
        d('source_bundle_dir', os.path.expanduser(
            '~/tpgpt_demos/banana_20260925_183904/source_bundle_learned'))
        d('object_name', 'mustard_bottle')
        d('cloud_topic', '')                      # '' -> /ycb_pointclouds/<object_name>
        d('n_frames', 3)                          # clouds fused per plan
        d('max_cloud_age', 10.0)                  # s; older clouds are discarded
        # s to wait, before the first fit, for n_frames clouds younger than max_cloud_age. On
        # 2026-09-26 most plans fused ONE frame (the object published intermittently, or the
        # plan started right after the node) and fitted worse: the apple's retry on three
        # frames read 6.5 mm where its single frame read 8.4 and was refused.
        d('frames_wait', 12.0)
        d('workspace_min', [0.15, -0.6, -0.25])   # base frame crop of the OBJECT CLOUD, m
        d('workspace_max', [0.85, 0.6, 0.5])
        # The box the EXECUTOR refuses to leave (its own workspace_min/max). A path that leaves
        # it is rejected at selection, so the next candidate is tried instead of the run aborting
        # at the first step. Keep these equal to the executor's, which are NOT the crop above:
        # the crop stops at z 0.5 because objects stand on a table, while a carried object rises
        # to 0.55 m in the taught demonstration.
        # Only used when the executor's own box is on (workspace_check, off by default).
        d('executor_workspace_check', False)
        d('executor_workspace_min', [0.15, -0.6, -0.25])
        d('executor_workspace_max', [0.85, 0.6, 0.7])
        d('voxel_size', 0.003)
        d('outlier_radius', 0.012)
        d('outlier_min_neighbors', 4)
        d('max_points', 8192)
        # The object cloud: the object's YCB scan fitted to the camera's cloud, as the
        # simulation's YCB experiment uses the full scan (tpgpt.hardware.object_model).
        # Falls back to the camera cloud when there is no scan or the fit is refused.
        d('object_model', True)
        # A refused fit is often this camera having a bad moment rather than a wrong pose: the
        # points off the object vary between 11% and 25% frame to frame. So a refusal is retried
        # on fresh clouds, and the best attempt is kept.
        d('object_model_retries', 2)
        d('object_model_retry_wait', 4.0)         # s to wait for new clouds between attempts
        # Scene description, base frame. NaN = estimate from the cloud.
        d('support_height', NAN)                  # NaN -> the table, or a shelf board the object stands on
        d('support_match', 0.030)                 # m: how far above a surface the object's lowest
                                                  # points may sit for it to count as standing there
        d('destination', [0.55, 0.25, 0.0])       # where to put the object (x, y used)
        # A named slot from config/shelf.yaml (top_left, top_right, bottom_left, bottom_right),
        # which resolves to the touched slot mark on that board. It overrides `destination`.
        d('destination_name', '')
        d('destination_height', NAN)              # surface height there; NaN -> destination z
        d('grasp_depth_below_top', 0.03)          # top-down fallback grasp only
        # transport: the demonstration carried to the destination. grasp_test: the grasp on its
        # own -- approach, close, lift 10 cm, set down, release, retreat (tpgpt.hardware.grasp_test;
        # the destination is not used). For making GraspGen-X's grasps work on a new hand.
        d('task', 'transport')
        d('grasp_test_depth_offset', 0.0)         # m: the hand this much further along its approach
        d('grasp_test_library_index', -1)         # >= 0: test only this candidate (a depth sweep)
        d('grasp_test_constructed', False)        # a grasp built by hand (straight down), not GraspGen-X's
        # Grasp selection
        # Hand on the flange: panda (Franka Hand) | inspire (Inspire RH56DFX-2R); both GraspGen-X
        d('end_effector', 'panda')
        d('end_effector_config', '')              # '' -> installed config/inspire_hand.yaml
        d('grasp_source', 'graspgenx')            # graspgenx | top_down (panda only)
        d('top_down_grasp', False)                # true: scripted top-down grasp only, GraspGen-X never called
        d('graspgen_host', '127.0.0.1')
        d('graspgen_port', 5556)
        # GraspGen-X description: '' = the hand's own (franka_panda; the Inspire config's
        # graspgenx.gripper). Set it only to the same name: another hand's grasps are refused.
        d('graspgen_gripper', '')
        d('graspgen_planner', 'diffusion')        # diffusion | graspmoe
        # GraspGen-X is a diffusion sampler: every call returns a different set, so the
        # number asked for decides how reliable planning is, not just how good the best
        # grasp is. Measured live on a banana 2026-09-23: with 100 candidates three calls
        # gave 16, 0 and 0 admissible (planning failed twice); with 400 they gave 5, 38
        # and 38, and all three planned, at ranks 16, 1 and 1, in 271, 22 and 15 s.
        # The object's fixed library instead of a fresh draw, when the scan fit was
        # accepted (tpgpt.hardware.grasp_library): the same grasps every run, so a
        # difference between two runs is a difference in the task.
        d('grasp_library', True)
        d('grasp_library_size', 2000)
        d('graspgen_num_grasps', 400)
        d('graspgen_topk', 200)
        d('graspgen_timeout', 30.0)               # s
        d('graspgen_autostart', True)             # start scripts/run_graspgenx_server.sh if unreachable
        # Approach vs straight down; past 90 the hand rises through the table. The sim caps
        # at 85 (filters.SUPPORT_APPROACH_MAX_DEG) because its fingers hang below the TCP and
        # it has no other table check; the hardware checks the hand's lowest point against the
        # table separately (hits_support, 3 mm), so the cap can sit on the geometric bound.
        # Measured on a YCB can, 2026-09-23: 15 of 100 candidates fall in 85-90 deg and they
        # include the best-scoring one (0.871) -- horizontal grasps around an upright can.
        d('grasp_max_tilt_deg', 90.0)
        # Visibility (approaches from the side nothing saw): 'auto' applies it only when the
        # object's scan was NOT fitted, since a fitted scan has no unseen side. 'true'/'false'
        # force it. On this bench the camera faces the robot, so the filter otherwise removes
        # the robot-side approaches.
        d('check_visibility', 'auto')
        # The sim's 'centred' stage drops grasps more than max_centre_offset from the cloud's
        # box centre, which guards against the moment a heavy object exerts on an off-centre
        # grip. Off here (user, 2026-09-23): these objects are light (apple 68 g, banana 66 g,
        # mug 118 g, soup can 349 g), so the moment is not what limits them, and the stage was
        # removing 10-23 of 100 candidates per run.
        d('check_centred', False)
        d('grasp_max_plans', 100)                 # candidates planned, best score first (sim: 100)
        d('grasp_min_path_clearance', -0.002)     # m, hand vs table/destination along the path
        d('grasp_fallback_top_down', False)       # the simulation has no scripted fallback
        # A fixed base-frame shift added to every planned grasp TCP. It was [-35, 0, -25] mm,
        # set before the 2026-09-21 recalibration. Measured 2026-09-23 (audit A5): it moves the
        # hand 43 mm off the object between the jaw check and the planner, so an obviously good
        # top-down grasp of a YCB can, which passes with no offset (min det 0.61, 61 mm
        # clearance), is refused as 'object_too_wide: the cloud measures 4 mm between the pads'.
        # It caused most plan_refused rejections on every object. Zero unless measured again.
        d('grasp_offset', [0.0, 0.0, 0.0])           # m, base frame
        d('grasp_markers_max', 60)                # candidates drawn in RViz
        d('min_determinant', 0.1)
        # The shelf and the whole-path check (tpgpt.hardware.shelf / path_check)
        d('shelf_config', default_config_path('shelf.yaml'))   # '' = no shelf model
        d('use_scene_cloud', True)                # the observed cloud on the shelf
        d('scene_cloud_topic', '/ycb_scene_cloud')
        d('scene_max_age', 10.0)                  # s
        d('scene_outlier_radius', 0.02)           # m; the scene cloud is ~7 mm apart at 1 m
        d('scene_outlier_min_neighbors', 3)
        d('joint_states_topic', '/joint_states')
        # Without fresh joint angles the robot can't be removed from the scene cloud and
        # the move onto the path can't be checked, so planning is refused. Set true only to
        # plan with the robot container off (offline dry runs); the plan records it.
        d('allow_missing_joint_states', False)
        d('joint_states_max_age', 1.0)            # s
        d('check_path', True)
        # strict: every waypoint reachable in position, pick and place with their orientation;
        # sim: min_reachable_fraction of the waypoints reachable as planned (the simulation's).
        d('kinematic_criterion', 'strict')
        d('min_reachable_fraction', 0.60)         # pipeline.MIN_REACHABLE_FRACTION (sim only)
        d('allow_kinematic_fallback', False)      # never a colliding candidate; off under strict
        # A blocked move from the arm's current pose onto the path no longer rejects the grasp
        # (user, 2026-09-29): it is checked and reported, and if the chosen grasp's move is
        # blocked the planner stops with approach_blocked and asks for a re-home. True brings
        # back rejecting the grasp.
        d('approach_rejects_grasp', False)
        # rad; > 0 refuses a path with joints 1-6 that close to a limit (CRISP pushes back
        # there; path_check.JOINT_MARGIN = 0.1). 0 = off.
        d('grasp_joint_margin', 0.0)
        # m; > 0 refuses a path that brings the wrist that close to full stretch, where the arm
        # cannot be pulled (path_check.REACH_MARGIN, 20 mm since 2026-10-03). 0 = off.
        d('grasp_reach_margin', 0.02)
        # m; > 0: once clear of the placed object by this much in the retreat, the hand must
        # stay that far from it (path_check.RETREAT_CLEARANCE, 10 mm since 2026-10-03). 0 = off.
        d('grasp_retreat_clearance', 0.01)
        d('max_centre_offset', 0.025)             # by_centre_offset on a one-sided cloud
        d('output_dir', os.path.expanduser('~/tpgpt_plans'))
        d('auto_plan', False)                     # plan as soon as n_frames arrived
        d('republish_period', 2.0)
        d('color_topic', '/camera/camera/color/image_raw')
        d('camera_info_topic', '/camera/camera/color/camera_info')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.end_effector = load_end_effector(p('end_effector'), p('end_effector_config'))
        if p('graspgen_gripper') and p('graspgen_gripper') != self.end_effector.graspgen:
            raise SystemExit(f"graspgen_gripper {p('graspgen_gripper')!r} is not the "
                             f"{self.end_effector.name} hand's description "
                             f"({self.end_effector.graspgen!r}): its grasps would be another "
                             "hand's")
        self.gripper = self.end_effector.graspgen
        self.get_logger().info(f'Hand on the flange: {self.end_effector.name} '
                               f'(GraspGen-X {self.gripper}, controller frame '
                               f'{self.end_effector.controller_frame}, grasp point '
                               f'{[round(v * 1000, 1) for v in self.end_effector.tool_offset]} mm)')
        if not self.end_effector.is_parallel_jaw and not self.end_effector.model_spec.get(
                'mount_measured'):
            self.get_logger().warning('the Inspire mount in end_effector_config is NOT measured '
                                      '(mount.measured: false): plans put the hand where the '
                                      'estimated mount says')
        from tpgpt.hardware.source_bundle import SourceBundle
        self.source = SourceBundle.load(p('source_bundle_dir'))
        self.get_logger().info(f"Source bundle {p('source_bundle_dir')}: "
                               f'{self.source.tool_labels.describe()}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.calib = CalibrationProvider(self, p('calibration_yaml'), self.tf_buffer)
        self.base = self.calib.cal.base_frame
        self.lock = threading.Lock()
        self.clouds = []     # (monotonic, points_base)
        self.latest_plan = None
        self.latest_selection = None
        self.latest_colors = None
        self._fused_colors = None
        self.latest_overlay = None
        self.last_plan_inputs = None
        self._color_msg = None
        self._camera_info = None
        self.plan_dir = None
        self.scene = None          # (monotonic, points_base, joint angles at capture)
        self.joints = None         # (monotonic, (7,) fr3_joint1..7)
        self.object_models = {}    # name -> tpgpt.hardware.object_model.ObjectModel
        self.object_fit = None
        self.object_cloud_source = None
        self.grasp_origin = None            # 'library (...)' or 'graspgenx (...)', for the record
        self.shelf_model = None
        shelf_path = p('shelf_config')
        if shelf_path:
            import yaml

            from tpgpt.hardware.shelf import ShelfModel
            with open(shelf_path) as f:
                self.shelf_raw = yaml.safe_load(f) or {}
            self.shelf_model = ShelfModel.from_config(self.shelf_raw)
            self.get_logger().info(
                f'Shelf model {shelf_path}: {len(self.shelf_model.boxes)} panels, supports '
                f'{list(self.shelf_model.supports)}'
                + ('' if self.shelf_model.measured else ' -- ESTIMATED dimensions (measured: false)'))
        else:
            self.shelf_raw = {}
            self.get_logger().warn('No shelf model (shelf_config empty): paths are checked '
                                   'against the observed cloud only')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.cloud_pub = self.create_publisher(PointCloud2, '/tpgpt/plan/target_cloud', latched)
        # What the camera saw of the object; target_cloud is what was planned on (the
        # fitted scan when object_model accepts it).
        self.observed_pub = self.create_publisher(PointCloud2, '/tpgpt/plan/observed_cloud',
                                                  latched)
        self.placed_pub = self.create_publisher(PointCloud2, '/tpgpt/plan/placed_cloud', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/tpgpt/plan/markers', latched)
        self.path_pub = self.create_publisher(Path, '/tpgpt/plan/path', latched)
        self.overlay_pub = self.create_publisher(Image, '/tpgpt/plan/overlay', latched)
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Image, p('color_topic'),
                                 lambda m: setattr(self, '_color_msg', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('camera_info_topic'),
                                 lambda m: setattr(self, '_camera_info', m), qos_profile_sensor_data)

        self.shelf_pub = self.create_publisher(MarkerArray, '/tpgpt/plan/shelf', latched)
        self.shelf_cloud_pub = self.create_publisher(PointCloud2, '/tpgpt/plan/shelf_cloud',
                                                     latched)
        if p('use_scene_cloud'):
            self.create_subscription(PointCloud2, p('scene_cloud_topic'), self._scene_cb, 2,
                                     callback_group=ReentrantCallbackGroup())
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, p('joint_states_topic'), self._joint_cb,
                                 qos_profile_sensor_data, callback_group=ReentrantCallbackGroup())

        topic = p('cloud_topic') or f"/ycb_pointclouds/{p('object_name')}"
        self.create_subscription(PointCloud2, topic, self._cloud_cb, 5,
                                 callback_group=ReentrantCallbackGroup())
        self.create_service(Trigger, '~/plan', self._plan_srv,
                            callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(p('republish_period'), self._republish)
        self.get_logger().info(f'Listening on {topic}. Call ~/plan to plan '
                               f"(auto_plan={p('auto_plan')}).")

    # ------------------------------------------------------------------ clouds
    def _cloud_cb(self, msg):
        T = self.calib.T_base_optical(msg.header.frame_id or self.calib.cal.optical_frame)
        if T is None:
            return
        xyz, rgb = pointcloud2_to_xyzrgb(msg)
        pts = cc.transform_points(T, xyz)
        with self.lock:
            self.clouds.append((time.monotonic(), pts, rgb))
            self.clouds = self.clouds[-int(self.get_parameter('n_frames').value):]
            ready = len(self.clouds) >= int(self.get_parameter('n_frames').value)
        self.get_logger().info(f'cloud: {len(pts)} points in {msg.header.frame_id} -> base',
                               throttle_duration_sec=5.0)
        if ready and self.get_parameter('auto_plan').value and self.latest_plan is None:
            threading.Thread(target=self._plan, daemon=True).start()

    def _joint_cb(self, msg):
        index = {n: i for i, n in enumerate(msg.name)}
        names = [f'fr3_joint{k}' for k in range(1, 8)]
        if all(n in index for n in names):
            q = np.array([msg.position[index[n]] for n in names], dtype=float)
            self.joints = (time.monotonic(), q)

    def _joint_problem(self):
        """Why planning must not go ahead for lack of joint angles, or None."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if p('allow_missing_joint_states'):
            return None
        topic = p('joint_states_topic')
        if self.joints is None:
            return (f'no {topic} received: the robot cannot be removed from the scene cloud and the '
                    'move onto the path cannot be checked. Is the robot container up (FCI on), and '
                    'was host_ros_env.sh sourced (ROS_DOMAIN_ID)? To plan with the robot off, set '
                    'allow_missing_joint_states:=true.')
        age = time.monotonic() - self.joints[0]
        if age > float(p('joint_states_max_age')):
            return (f'{topic} is {age:.1f} s old (max {p("joint_states_max_age")} s): the robot '
                    'stopped publishing')
        if p('use_scene_cloud') and self.scene is not None and self.scene[2] is None:
            return ('the latest scene cloud arrived before any joint angles, so the robot cannot be '
                    'removed from it; wait for the next segmentation cycle and plan again')
        return None

    def _scene_cb(self, msg):
        T = self.calib.T_base_optical(msg.header.frame_id or self.calib.cal.optical_frame)
        if T is None:
            return
        xyz, _ = pointcloud2_to_xyzrgb(msg)
        joints = self.joints[1].copy() if self.joints is not None else None
        self.scene = (time.monotonic(), cc.transform_points(T, xyz), joints)

    def shelf_world(self, target_points):
        """The shelf model plus the recent scene cloud on it (tpgpt.hardware.shelf)."""
        from tpgpt.hardware.fr3 import default_model
        from tpgpt.hardware.shelf import ROI_ABOVE_SUPPORT, ROI_MARGIN, ShelfWorld
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        scene_cfg = (self.shelf_raw or {}).get('scene', {}) or {}
        points, q, age = None, None, None
        if p('use_scene_cloud') and scene_cfg.get('use_cloud', True) and self.scene is not None:
            t, pts, q = self.scene
            age = time.monotonic() - t
            if age <= float(p('scene_max_age')):
                points = pts
        world = ShelfWorld.build(
            self.shelf_model, points, target_points=target_points, robot_q=q,
            fr3=default_model(), end_effector=self.end_effector,
            margin=float(scene_cfg.get('roi_margin', ROI_MARGIN)),
            above_support=float(scene_cfg.get('roi_above_support', ROI_ABOVE_SUPPORT)),
            scope=str(scene_cfg.get('scope', 'visible')),
            support_height=self._support_height(target_points))
        if len(world.cloud) > int(p('scene_outlier_min_neighbors')):
            before = len(world.cloud)
            world.cloud = cc.radius_outlier_filter(world.cloud, float(p('scene_outlier_radius')),
                                                   int(p('scene_outlier_min_neighbors')))
            world.cloud_report['removed_as_outliers'] = before - len(world.cloud)
            world.cloud_report['obstacle_points'] = int(len(world.cloud))
        world.cloud_report['scene_age_s'] = age
        if p('use_scene_cloud') and points is None:
            world.cloud_report['scene_missing'] = (f"no {p('scene_cloud_topic')} within "
                                                   f"{p('scene_max_age')} s")
        return world

    @staticmethod
    def _approach_blocked(selection):
        """Why the chosen grasp's move onto the path is blocked, or ``None``."""
        chosen = getattr(selection, 'chosen', None)
        check = (getattr(chosen, 'path_check', None) or {}).get('approach_in') or {}
        if not check or check.get('rejects', True):
            return None               # checked as part of the path, or not at all
        faults, unreached = check.get('faults') or {}, int(check.get('unreachable') or 0)
        if not faults and not unreached:
            return None
        parts = [f'{k} x{v}' for k, v in list(faults.items())[:3]]
        if unreached:
            parts.append(f'{unreached} of its {check.get("steps")} steps out of reach')
        return 'the move from where the arm is now onto the path: ' + ', '.join(parts)

    def _recent_clouds(self):
        max_age = float(self.get_parameter('max_cloud_age').value)
        now = time.monotonic()
        with self.lock:
            return sum(1 for t, _, _ in self.clouds if now - t <= max_age)

    def _wait_for_frames(self):
        """Wait up to ``frames_wait`` s for ``n_frames`` recent clouds; go on with fewer."""
        want = int(self.get_parameter('n_frames').value)
        deadline = time.monotonic() + float(self.get_parameter('frames_wait').value)
        have = self._recent_clouds()
        if have >= want:
            return
        self.get_logger().info(f'{have} of {want} camera frames so far; waiting up to '
                               f"{self.get_parameter('frames_wait').value:.0f} s for the rest")
        while time.monotonic() < deadline:
            time.sleep(0.2)
            have = self._recent_clouds()
            if have >= want:
                return
        self.get_logger().warn(f'planning on {have} of {want} camera frames: the object is '
                               'published intermittently, and the scan fit is weaker on fewer')

    def fused_cloud(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        now = time.monotonic()
        with self.lock:
            recent = [(c, rgb) for t, c, rgb in self.clouds if now - t <= p('max_cloud_age')]
        clouds = [c for c, _ in recent]
        if not clouds:
            raise RuntimeError('no recent object cloud; is segmentation_node detecting '
                               f"{p('object_name')}?")
        pts, stats = fuse_clouds(clouds, p('workspace_min'), p('workspace_max'), p('voxel_size'),
                                 p('outlier_radius'), p('outlier_min_neighbors'), p('max_points'))
        self.get_logger().info(f"fused {len(clouds)} clouds: {stats['n_raw']} -> "
                               f"{stats['n_in_workspace']} in workspace -> {len(pts)} after "
                               'voxel/outlier filtering')
        raw = np.vstack(clouds)
        colored = all(rgb is not None for _, rgb in recent)
        self._fused_colors = colors_for(pts, raw, np.vstack([rgb for _, rgb in recent])
                                        if colored else None)
        return pts, len(clouds)

    # ------------------------------------------------------------------ planning
    def _plan_srv(self, request, response):
        ok, text = self._plan()
        response.success, response.message = ok, text
        return response

    def _plan(self):
        from tpgpt.hardware.grasp_selection import SelectionSettings, select_and_plan
        from tpgpt.hardware.planner import PlanningError, TargetScene
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        log = self.get_logger()
        # Inputs, raw candidates and stage durations of this call, kept for the X6
        # trial record (x6_trial). Read-only bookkeeping; planning does not use it.
        info = self.last_plan_inputs = {'timing_s': {}}
        problem = self._joint_problem()
        if problem:
            log.error(problem)
            info['error'] = problem
            return False, problem
        t_start = time.monotonic()
        self._wait_for_frames()
        try:
            points, n_clouds = self.fused_cloud()
        except RuntimeError as exc:
            log.error(str(exc))
            info['error'] = str(exc)
            return False, str(exc)
        observed, points, n_clouds = self._object_cloud_with_retries(points, n_clouds)
        info['n_clouds'] = n_clouds
        self.observed_pub.publish(xyz_to_pointcloud2(observed, self.base,
                                                     self.get_clock().now().to_msg(),
                                                     (255, 255, 255)))
        info['timing_s']['perception'] = time.monotonic() - t_start
        info['points'] = points
        info['observed_points'] = observed
        info['object_cloud_source'] = self.object_cloud_source
        info['object_model'] = None if self.object_fit is None else self.object_fit.summary()
        destination = np.asarray(p('destination'), dtype=float)
        dest_h = p('destination_height')
        if p('destination_name'):
            from tpgpt.hardware.shelf import slots
            try:
                named = slots(self.shelf_raw)[p('destination_name')]
            except KeyError:
                known = sorted(slots(self.shelf_raw)) if self.shelf_raw else []
                msg = (f"destination_name {p('destination_name')!r} is not a slot in "
                       f"{p('shelf_config')}; known: {known}")
                log.error(msg)
                info['error'] = msg
                return False, msg
            destination = np.asarray(named, dtype=float)
            dest_h = float(named[2])
            log.info(f"destination {p('destination_name')}: "
                     f"{np.round(destination, 4).tolist()} (from the touched slot marks)")
        scene = TargetScene(
            points=points,
            destination=destination,
            destination_height=float(dest_h) if _finite(dest_h) else float(destination[2]),
            support_height=self._support_height(points),
        )
        t0 = time.monotonic()
        poses, scores, unavailable = self._grasp_candidates(points)
        info['timing_s']['grasp_generation'] = time.monotonic() - t0
        info.update({'poses': poses, 'scores': scores, 'unavailable': unavailable,
                     'scene': scene})
        settings = SelectionSettings(
            max_tilt_deg=float(p('grasp_max_tilt_deg')),
            max_plans=int(p('grasp_max_plans')),
            min_determinant=float(p('min_determinant')),
            min_path_clearance=float(p('grasp_min_path_clearance')),
            fallback_top_down=bool(p('grasp_fallback_top_down')) or bool(p('top_down_grasp')),
            grasp_offset=tuple(float(v) for v in p('grasp_offset')),
            check_path=bool(p('check_path')),
            min_reachable_fraction=float(p('min_reachable_fraction')),
            kinematic_criterion=str(p('kinematic_criterion')),
            approach_rejects=bool(p('approach_rejects_grasp')),
            joint_margin=(float(p('grasp_joint_margin')) if float(p('grasp_joint_margin')) > 0
                          else None),
            reach_margin=(float(p('grasp_reach_margin')) if float(p('grasp_reach_margin')) > 0
                          else None),
            retreat_clearance=(float(p('grasp_retreat_clearance'))
                               if float(p('grasp_retreat_clearance')) > 0 else None),
            allow_kinematic_fallback=bool(p('allow_kinematic_fallback')),
            max_centre_offset=float(p('max_centre_offset')),
            check_visibility=self._check_visibility(),
            check_centred=bool(p('check_centred')),
            # The executor refuses to leave this box, so a path that does is no use.
            workspace_min=(tuple(float(v) for v in p('executor_workspace_min'))
                           if p('executor_workspace_check') else None),
            workspace_max=(tuple(float(v) for v in p('executor_workspace_max'))
                           if p('executor_workspace_check') else None),
        )
        info['settings'] = settings
        from tpgpt.hardware.grasp_selection import PathContext, ScreenContext
        from tpgpt.sim.keypoints import carry_transform
        # The target is removed from the scene cloud: what the camera saw of it,
        # and the fitted scan's hidden side.
        world = self.shelf_world(np.vstack([observed, points]))
        log.info(f'Shelf world: {world.cloud_report}')
        if world.cloud_report.get('scene_missing'):
            log.warn(f"{world.cloud_report['scene_missing']}: the shelf model alone")
        joints = self.joints[1] if self.joints is not None else None
        if joints is None:
            # Only reachable with allow_missing_joint_states:=true (see _joint_problem).
            log.warn(f"no {p('joint_states_topic')} (allowed by allow_missing_joint_states): the "
                     'robot is not removed from the scene cloud, the move onto the path is not '
                     'checked, and IK is seeded from home')
        T_cam = self.calib.T_base_optical(self.calib.cal.optical_frame)
        context = ScreenContext(
            camera_position=None if T_cam is None else T_cam[:3, 3],
            world=world, destination=destination[:2],
            destination_height=scene.destination_height,
            carry_rotation=carry_transform(self.source.demonstration_labels)[0])
        path_ctx = PathContext(world=world, end_effector=self.end_effector, start_q=joints)
        info['screen_context'] = context
        info['world'] = world.to_dict()
        t_sel = time.monotonic()
        task = str(p('task'))
        try:
            if task == 'grasp_test':
                from tpgpt.hardware import grasp_test as gt
                support = float(scene.support_height if scene.support_height is not None
                                else np.percentile(points[:, 2], 1.0))
                # executed at the transport's settings: the source bundle's (the user,
                # 2026-10-06: as the Franka Hand's transport runs were)
                execution = (self.source.metadata.get('learned') or {}).get('execution')
                if not execution:
                    raise PlanningError(
                        f"grasp_test: the source bundle {p('source_bundle_dir')} has no execution "
                        "settings (learned.execution), and a grasp test runs at the transport's: "
                        "pass source_bundle_dir:=<the learned bundle the transport runs use>")
                if p('grasp_test_constructed'):
                    selection = gt.constructed_grasp_test(
                        points, support, settings, path_ctx, self.end_effector,
                        p('object_name'), float(p('grasp_test_depth_offset')), execution)
                elif poses is None or not len(poses):
                    raise PlanningError(f'no_executable_grasp: no GraspGen-X candidates '
                                        f'({unavailable})')
                else:
                    index = int(p('grasp_test_library_index'))
                    selection = gt.select_grasp_test(
                        points, poses, scores, support, settings, context, path_ctx,
                        self.end_effector, p('object_name'), float(p('grasp_test_depth_offset')),
                        None if index < 0 else index, execution)
            elif task == 'transport':
                selection = select_and_plan(
                    self.source, scene, poses, scores, settings, unavailable_reason=unavailable,
                    context=context, path=path_ctx,
                    grasp_depth_below_top=float(p('grasp_depth_below_top')))
            else:
                raise PlanningError(f"task must be transport or grasp_test, not {task!r}")
        except PlanningError as exc:
            info['timing_s']['selection_total'] = time.monotonic() - t_sel
            info['error'] = str(exc)
            info['failed_candidates'] = getattr(exc, 'candidates', None)
            info['failed_fallback'] = getattr(exc, 'fallback', None)
            info['failed_stages'] = getattr(exc, 'stages', None)
            log.error(f'Planning refused: {exc}')
            self._save_refused(exc, observed, points, world,
                               destination=(p('destination_name'), scene.destination))
            self._publish_cloud_only(points)
            self._publish_shelf(world)
            return False, str(exc)
        info['timing_s']['selection_total'] = time.monotonic() - t_sel
        blocked = self._approach_blocked(selection)
        if blocked:
            from tpgpt.hardware.planner import PlanningError as _PE
            exc = _PE(f'approach_blocked: {blocked}. The grasp and its path are fine; re-home '
                      'the arm (ros2 run ycb_pointcloud_pipeline joint_homing) and plan again')
            exc.candidates = selection.candidates
            info['error'] = str(exc)
            log.error(f'Planning stopped: {exc}')
            self._save_refused(exc, observed, points, world,
                               destination=(p('destination_name'), scene.destination))
            self._publish_cloud_only(points)
            self._publish_shelf(world)
            return False, str(exc)
        info['selection'] = selection
        result = selection.plan
        log.info(selection.describe())
        for note in selection.notes:
            log.warning(note)
        cal = self.calib.cal
        result.metadata.update({
            'object_name': p('object_name'),
            'n_fused_clouds': n_clouds,
            'object_cloud_source': self.object_cloud_source,
            'object_model': None if self.object_fit is None else self.object_fit.summary(),
            'calibration_yaml': p('calibration_yaml'),
            'calibration_is_placeholder': cal.camera_is_placeholder,
            'T_base_camera': cal.T_base_camera.tolist(),
            'camera_child_frame': cal.camera_child_frame,
            'source_bundle_dir': p('source_bundle_dir'),
            'planning_seconds': time.monotonic() - t0,
            'grasp_source': selection.source,
            'grasp_candidates_from': self.grasp_origin,
            'grasp_score': selection.chosen.score,
            'grasp_candidates': len(selection.candidates),
            'grasp_rejections': selection.rejection_counts(),
            'grasp_notes': selection.notes,
            'grasp_stages': selection.stages,
            'grasp_rank_examined': selection.rank_examined,
            'grasp_fell_back': selection.fell_back,
            'path_check': selection.chosen.path_check,
            'shelf': world.to_dict(),
            'shelf_is_estimate': bool(self.shelf_model is not None
                                      and not self.shelf_model.measured),
            'joint_states_at_plan': None if joints is None else joints.tolist(),
            'allow_missing_joint_states': bool(p('allow_missing_joint_states')),
            'end_effector_config': self.end_effector.details.get('config_file'),
            'graspgen': {n: self.get_parameter(n).value for n in (
                'graspgen_gripper', 'graspgen_planner', 'graspgen_num_grasps', 'graspgen_topk',
                'grasp_max_tilt_deg', 'grasp_max_plans', 'grasp_min_path_clearance',
                'grasp_offset', 'check_visibility', 'check_centred')},
            'parameters': {n: self.get_parameter(n).value for n in (
                'object_model_retries', 'destination_name', 'executor_workspace_check',
                'workspace_min', 'workspace_max', 'executor_workspace_min',
                'executor_workspace_max', 'voxel_size', 'destination',
                'destination_height', 'support_height', 'grasp_depth_below_top',
                'min_determinant', 'n_frames', 'task', 'grasp_test_depth_offset',
                'grasp_test_library_index', 'grasp_test_constructed')},
        })
        out = os.path.join(p('output_dir'),
                           f"{p('object_name')}_{time.strftime('%Y%m%d_%H%M%S')}"
                           + ('_grasptest' if task == 'grasp_test' else ''))
        result.save(out)
        # The camera's own cloud and the obstacles, as a refused plan keeps them: a plan
        # that worked is as much evidence as one that did not, and without these its fit
        # cannot be re-checked nor its scene re-measured (python -m tpgpt.hardware.coverage).
        np.savez_compressed(os.path.join(out, 'clouds.npz'), observed=observed,
                            object_cloud=points, obstacles=world.cloud)
        with open(os.path.join(out, 'grasp_candidates.json'), 'w') as f:
            json.dump({'source': selection.source, 'notes': selection.notes,
                       'chosen_index': selection.chosen.index,
                       'candidates': [c.as_dict() for c in selection.candidates]},
                      f, indent=1, default=float)
        overlay = self._render_overlay(result)
        if overlay is not None:
            import cv2
            cv2.imwrite(os.path.join(out, 'overlay.png'), overlay)
        with self.lock:
            self.latest_plan, self.latest_selection, self.plan_dir = result, selection, out
            self.latest_world = world
            self.latest_colors = self._fused_colors
            self.latest_overlay = overlay
        m = result.metrics
        grasp_text = (f'{selection.describe()}' if not self.end_effector.is_parallel_jaw else
                      f'grasp {selection.source} (score {selection.chosen.score:.3f}, tilt '
                      f'{m["grasp_tilt_deg"]:.0f} deg, {m["jaw_points_between_pads"]} pts between pads)')
        text = (f'Plan saved to {out} [{self.end_effector.name}]: {grasp_text}, '
                f'min det(J) {m["min_det"]:.3f}, keypoint residual '
                f'{m["keypoint_residual_m"] * 1000:.2f} mm, aim at grasp '
                f'{m["aim_grasp_m"] * 1000:.1f} mm / release {m["aim_release_m"] * 1000:.1f} mm, '
                f'orientation error at grasp {m["orientation_error_grasp_deg"]:.1f} deg, '
                f'width between pads {m["object_width_closing_m"] * 1000:.0f} mm (object '
                f'{m["object_extent_closing_m"] * 1000:.0f} mm across), hand clearance '
                f'{m["hand_clearance_pick_m"] * 1000:.0f} mm at pick / '
                f'{m["hand_clearance_place_m"] * 1000:.0f} mm at place, support '
                f'{m["support_height_m"]:+.3f} m'
                f'{" (estimated)" if m["support_height_estimated"] else ""}, lowest path point '
                f'{m["path_min_height_above_support_m"] * 1000:.0f} mm above support')
        if cal.camera_is_placeholder:
            text += ' -- WARNING: extrinsics are a PLACEHOLDER, do not execute this plan'
        log.info(text)
        self._publish_plan(result, selection)
        return True, text

    def _check_visibility(self):
        """Whether to apply the visibility filter for this plan (see the parameter)."""
        value = str(self.get_parameter('check_visibility').value).lower()
        if value in ('true', '1', 'yes'):
            return True
        if value in ('false', '0', 'no'):
            return False
        if value != 'auto':
            raise ValueError(f"check_visibility must be auto, true or false, not {value!r}")
        fitted = self.object_fit is not None and self.object_fit.accepted
        if fitted:
            self.get_logger().info('visibility filter off: the object cloud is the fitted scan, '
                                   'which has no unseen side')
        return not fitted

    def _object_cloud_with_retries(self, observed, n_clouds):
        """Fit the scan, retrying on fresh camera frames while it is refused.

        Returns ``(observed, planning cloud, clouds fused)``, keeping the attempt
        whose fit explained the most of its own cloud.
        """
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        best = None
        for attempt in range(max(int(p('object_model_retries')), 0) + 1):
            points = self._full_object_cloud(observed)
            fit = self.object_fit
            if fit is None or fit.accepted:
                return observed, points, n_clouds
            score = fit.explained
            if best is None or score > best[0]:
                best = (score, observed, points, n_clouds, fit, self.object_cloud_source)
            if attempt == int(p('object_model_retries')):
                break
            self.get_logger().info(
                f'object model refused ({fit.reason}); waiting '
                f"{p('object_model_retry_wait')} s for fresh clouds and trying again "
                f"({attempt + 1} of {p('object_model_retries')})")
            with self.lock:
                seen = len(self.clouds) and self.clouds[-1][0]
            self.await_fresh_clouds(seen, float(p('object_model_retry_wait')))
            try:
                observed, n_clouds = self.fused_cloud()
            except RuntimeError:
                break
        _, observed, points, n_clouds, self.object_fit, self.object_cloud_source = best
        self.get_logger().warn(
            f'object model: no attempt was accepted; the best explained '
            f'{self.object_fit.explained:.0%} of its cloud. Planning on the camera cloud.')
        return observed, points, n_clouds

    def await_fresh_clouds(self, seen, wait):
        """Wait up to ``wait`` s for a cloud that arrived after ``seen`` (its arrival time).

        An X6 trial replaces this with a capture of its own, so the frames a retry
        plans on are among the ones it records (x6_trial.Trial._retry_capture).
        """
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(0.2)
            with self.lock:
                if len(self.clouds) and self.clouds[-1][0] != seen:
                    return

    def _full_object_cloud(self, observed):
        """The fitted YCB scan (tpgpt.hardware.object_model), or the camera cloud.

        Sets ``object_cloud_source`` and ``object_fit`` for the plan record.
        """
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        log = self.get_logger()
        name = p('object_name')
        self.object_fit = None
        if not p('object_model'):
            self.object_cloud_source = 'camera (object_model is false)'
            return observed
        from tpgpt.hardware.object_model import YCB_SCANS, ObjectModel, fit_object
        if name not in YCB_SCANS:
            self.object_cloud_source = f'camera (no YCB scan for {name})'
            log.warn(f'No YCB scan for {name!r} (known: {sorted(YCB_SCANS)}): planning on the '
                     'camera cloud')
            return observed
        support = self._support_height(observed)
        T_cam = self.calib.T_base_optical(self.calib.cal.optical_frame)
        if support is None or T_cam is None:
            self.object_cloud_source = 'camera (no support height or camera pose to fit with)'
            log.warn(f'{self.object_cloud_source}: planning on the camera cloud')
            return observed
        if name not in self.object_models:
            try:
                self.object_models[name] = ObjectModel.load(name)
            except FileNotFoundError as exc:
                self.object_cloud_source = 'camera (scan not downloaded)'
                log.warn(f'{exc}: planning on the camera cloud')
                return observed
        t = time.monotonic()
        fit = fit_object(self.object_models[name], observed, float(support), T_cam[:3, 3])
        self.object_fit = fit
        s = fit.summary()
        detail = (f"rest {s['rest_index']}, heading {s['heading_deg']:.0f} deg, camera points "
                  f"{s['median_mm']:.1f} mm from the scan (median), {s['explained']:.0%} within "
                  f"8 mm, {s['seen']:.0%} of the camera-facing scan seen, fitted height "
                  f"{s['dz_mm']:+.1f} mm off the table"
                  + (f", {s['dropped_occluded']} points behind the scan dropped as depth artefacts"
                     if s.get('dropped_occluded') else '')
                  + f"; {time.monotonic() - t:.1f} s")
        if fit.accepted:
            self.object_cloud_source = 'ycb scan (fitted)'
            log.info(f'Object model: {name} scan fitted ({detail}); planning on its '
                     f'{len(fit.full_cloud)}-point full cloud')
            return fit.full_cloud
        self.object_cloud_source = 'camera (fit refused)'
        log.warn(f'Object model: {name} fit REFUSED, {fit.reason} ({detail}); planning on the '
                 'camera cloud')
        return observed

    def _save_refused(self, exc, observed, points, world=None, destination=None):
        """A refused plan's candidates and clouds, so the refusal can be diagnosed."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        out = os.path.join(p('output_dir'),
                           f"{p('object_name')}_{time.strftime('%Y%m%d_%H%M%S')}_refused")
        try:
            os.makedirs(out, exist_ok=True)
            cands = getattr(exc, 'candidates', None) or []
            with open(os.path.join(out, 'grasp_candidates.json'), 'w') as f:
                json.dump({'error': str(exc), 'stages': getattr(exc, 'stages', None),
                           # Where it was going: without it a run of refusals cannot be
                           # told apart by slot (2026-09-25, read back from the ROS log).
                           'destination_name': None if destination is None else destination[0],
                           'destination': (None if destination is None
                                           else np.asarray(destination[1]).tolist()),
                           'shelf_world': None if world is None else world.cloud_report,
                           'object_cloud_source': self.object_cloud_source,
                           'object_model': (None if self.object_fit is None
                                            else self.object_fit.summary()),
                           'candidates': [c.as_dict() for c in cands]},
                          f, indent=1, default=float)
            np.savez_compressed(os.path.join(out, 'clouds.npz'), observed=observed,
                                object_cloud=points,
                                obstacles=(np.zeros((0, 3)) if world is None else world.cloud))
            self.get_logger().info(f'Refused plan recorded in {out}')
        except Exception as err:  # the record must never mask the refusal itself
            self.get_logger().warn(f'could not record the refused plan: {err}')

    def _grasp_candidates(self, points):
        """GraspGen-X poses and scores in the base frame, or (None, None, reason)."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if p('top_down_grasp'):
            return None, None, 'top_down_grasp is true'
        if p('grasp_source') == 'top_down':
            return None, None, 'grasp_source is top_down'
        if p('grasp_source') != 'graspgenx':
            raise ValueError(f"grasp_source must be graspgenx or top_down, not {p('grasp_source')!r}")
        from tpgpt.grasp.client import GraspGenUnavailable
        from tpgpt.hardware.grasp_selection import request_candidates
        if p('grasp_library') and self.object_fit is not None and self.object_fit.accepted:
            poses, scores = self._library_candidates()
            if poses is not None:
                return poses, scores, None
        problem = self._ensure_graspgen_server()
        if problem:
            self.get_logger().error(f'GraspGen-X unavailable: {problem}')
            return None, None, f'GraspGen-X unavailable: {problem}'
        t = time.monotonic()
        try:
            poses, scores = request_candidates(
                points, host=p('graspgen_host'), port=int(p('graspgen_port')),
                gripper=self.gripper, num_grasps=int(p('graspgen_num_grasps')),
                topk=int(p('graspgen_topk')), planner=p('graspgen_planner'),
                timeout_ms=int(float(p('graspgen_timeout')) * 1000))
        except GraspGenUnavailable as exc:
            self.get_logger().error(f'GraspGen-X unavailable: {exc}')
            return None, None, f'GraspGen-X unavailable: {exc}'
        self.grasp_origin = f'graspgenx (fresh draw of {len(poses)})'
        self.get_logger().info(f'GraspGen-X: {len(poses)} candidates for {len(points)} points in '
                               f'{time.monotonic() - t:.2f} s')
        return poses, scores, None if len(poses) else 'GraspGen-X returned no grasps'

    def _library_candidates(self):
        """The object's fixed library, carried onto the fitted scan, or (None, None).

        The library is asked for once per (object, gripper, size) and kept on
        disk, so the server is only needed the first time an object is planned.
        A failure here is not fatal: the caller falls back to a live draw.
        """
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        log = self.get_logger()
        fit = self.object_fit
        size, gripper = int(p('grasp_library_size')), self.gripper
        try:
            from tpgpt.hardware.grasp_library import cached, library, placed
            on_disk = cached(fit.name, gripper=gripper, size=size, planner=p('graspgen_planner'))
            if not on_disk:
                problem = self._ensure_graspgen_server()
                if problem:
                    log.warn(f'no grasp library for {fit.name} and GraspGen-X is unavailable '
                             f'({problem}): falling back to a live draw')
                    return None, None
                log.info(f'building the {fit.name} grasp library ({size} grasps, {gripper}); '
                         'this happens once')
            t = time.monotonic()
            poses, scores = library(fit.name, gripper=gripper, size=size,
                                    host=p('graspgen_host'), port=int(p('graspgen_port')),
                                    planner=p('graspgen_planner'),
                                    timeout_ms=max(int(float(p('graspgen_timeout')) * 1000),
                                                   300000))
            poses = placed(poses, fit.rotation, fit.position)
        except Exception as exc:                       # a library must never lose a plan
            log.warn(f'grasp library unusable ({exc}): falling back to a live draw')
            return None, None
        self.grasp_origin = f'library ({len(poses)}, {"cached" if on_disk else "built"})'
        log.info(f'Grasp library: {len(poses)} grasps for {fit.name} '
                 f'({"from disk" if on_disk else "built"}) in {time.monotonic() - t:.2f} s')
        return poses, scores

    def _ensure_graspgen_server(self):
        """None if the server answers (starting it first if allowed), else why not."""
        import subprocess

        from tpgpt.grasp.client import GraspGenClient
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        host, port = p('graspgen_host'), int(p('graspgen_port'))

        def alive():
            client = GraspGenClient(host=host, port=port, timeout_ms=2000)
            try:
                return client.is_alive()
            finally:
                client.close()

        if alive():
            return None
        if not p('graspgen_autostart'):
            return f'no server at tcp://{host}:{port} and graspgen_autostart is false'
        if host not in ('127.0.0.1', 'localhost'):
            return f'no server at tcp://{host}:{port}; autostart only works for a local server'
        script = graspgen_server_script()
        if script is None:
            return 'run_graspgenx_server.sh not found in the package share or source tree'
        self.get_logger().info(f'GraspGen-X server not running; starting {script} --daemon')
        env = dict(os.environ, GRASPGEN_SERVER_PORT=str(port))
        try:
            out = subprocess.run(['bash', script, '--daemon'], env=env, capture_output=True,
                                 text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return 'server start timed out after 90 s (see ~/.ros/graspgenx_server.log)'
        if out.returncode != 0:
            return f'server failed to start: {(out.stderr or out.stdout).strip()[-400:]}'
        for _ in range(30):
            if alive():
                self.get_logger().info('GraspGen-X server is up')
                return None
            time.sleep(1.0)
        return 'server started but does not answer (see ~/.ros/graspgenx_server.log)'

    def _support_height(self, points=None):
        """What the object is standing on: the table, or a shelf board it sits on.

        An explicit ``support_height`` wins. Otherwise the candidates are the
        touched table height and the top face of every shelf board, and the one
        chosen is the highest that lies at or just below the object's own lowest
        points (within ``support_match``). A reshelving experiment picks objects
        off boards as well as off the table, and assuming the table puts the
        object's base tens of centimetres below where it is: measured
        2026-09-23, a banana standing on the bottom board fitted 94 mm off.
        """
        value = self.get_parameter('support_height').value
        if _finite(value):
            return float(value)
        table = self.calib.cal.expected_table_height
        if points is None or not len(points) or self.shelf_model is None or table is None:
            return table
        low = float(np.percentile(np.asarray(points)[:, 2], 1.0))
        match = float(self.get_parameter('support_match').value)
        best = table
        for box in self.shelf_model.boxes:
            if box.name not in self.shelf_model.supports:
                continue
            top = float(box.position[2] + box.size[2])
            if top > best and top <= low + match:
                best = top
        if best != table:
            self.get_logger().info(f'the object stands on a shelf board at z {best:+.4f} m, not '
                                   f'the table at {table:+.4f} m (its lowest points are {low:+.4f})')
        return best

    # ------------------------------------------------------------------ rviz
    def _publish_cloud_only(self, points):
        self.cloud_pub.publish(xyz_to_pointcloud2(points, self.base,
                                                  self.get_clock().now().to_msg(), (255, 140, 0)))

    def _republish(self):
        with self.lock:
            result, selection = self.latest_plan, self.latest_selection
        if result is not None:
            self._publish_plan(result, selection)

    def _publish_plan(self, result, selection=None):
        stamp = self.get_clock().now().to_msg()
        base = self.base
        labels = result.labels
        m = result.metrics
        colors = self.latest_colors
        if colors is None or len(colors) != len(result.target_points):
            colors = (255, 140, 0)
        self.cloud_pub.publish(xyz_to_pointcloud2(result.target_points, base, stamp, colors))
        if self.latest_overlay is not None:
            from cv_bridge import CvBridge
            msg = CvBridge().cv2_to_imgmsg(self.latest_overlay, encoding='bgr8')
            msg.header.stamp, msg.header.frame_id = stamp, self.calib.cal.optical_frame
            self.overlay_pub.publish(msg)
        placed = result.diagnostics.get('target_place_points')
        if placed is not None and len(placed):
            self.placed_pub.publish(xyz_to_pointcloud2(placed, base, stamp, (0, 220, 255)))

        markers = MarkerArray()
        markers.markers.append(sphere_list_marker('keypoints_target', 0,
                                                  result.target_keypoints.points, base, stamp,
                                                  (0.1, 0.9, 0.1), 0.01))
        gi, ri = int(m['grasp_index']), int(m['release_index'])
        markers.markers += axes_markers('tcp_at_grasp', 0, labels.positions[gi],
                                        labels.orientations[gi], base, stamp)
        markers.markers += axes_markers('tcp_at_release', 0, labels.positions[ri],
                                        labels.orientations[ri], base, stamp)
        markers.markers += axes_markers('tcp_at_start', 0, labels.positions[0],
                                        labels.orientations[0], base, stamp, length=0.05)
        colors = [(1.0, 0.1, 0.1) if g > 0 else (0.2, 0.4, 1.0) for g in labels.gripper]
        markers.markers.append(line_strip_marker('path', 0, labels.positions, base, stamp,
                                                 colors=colors, width=0.004))
        markers.markers.append(cylinder_marker(
            'destination', 0, [result.destination[0], result.destination[1],
                               result.destination_height + 0.001],
            0.06, 0.002, base, stamp, (0.0, 0.9, 0.9, 0.6)))
        markers.markers.append(text_marker(
            'metrics', 0, labels.positions[gi] + [0, 0, 0.12],
            f"min det {m['min_det']:.2f} | aim {m['aim_grasp_m'] * 1000:.1f} mm | "
            f"width {m['object_width_closing_m'] * 1000:.0f} mm", base, stamp))
        if selection is not None:
            markers.markers += self._candidate_markers(selection, stamp)
        self.marker_pub.publish(markers)
        self._publish_shelf(getattr(self, 'latest_world', None))

        path = Path()
        path.header.frame_id, path.header.stamp = base, stamp
        path.poses = [matrix_to_pose_stamped(pos, rot, base, stamp)
                      for pos, rot in zip(labels.positions[::2], labels.orientations[::2])]
        self.path_pub.publish(path)


    def _render_overlay(self, result):
        """The latest colour image with the plan drawn through the calibration, or None."""
        import cv2
        from cv_bridge import CvBridge
        color, info = self._color_msg, self._camera_info
        if color is None or info is None:
            self.get_logger().warn('no colour image / camera_info yet; overlay.png not written')
            return None
        T = self.calib.T_base_optical(color.header.frame_id or self.calib.cal.optical_frame)
        if T is None:
            return None
        K = np.array(info.k, dtype=float).reshape(3, 3)
        T_opt_base = cc.invert_transform(T)
        img = CvBridge().imgmsg_to_cv2(color, desired_encoding='bgr8').copy()
        h, w = img.shape[:2]

        def project(points):
            q = cc.transform_points(T_opt_base, np.atleast_2d(points))
            z = np.maximum(q[:, 2], 1e-6)
            return np.column_stack([K[0, 0] * q[:, 0] / z + K[0, 2],
                                    K[1, 1] * q[:, 1] / z + K[1, 2]]), q[:, 2] > 0

        uv, front = project(result.target_points)
        for (u, v), f in zip(uv.astype(int), front):
            if f and 0 <= u < w and 0 <= v < h:
                img[v, u] = (0, 255, 0)
        g = result.grasp
        R = g.rotation
        for box in self.end_effector.boxes():
            corners = box.corners() @ R.T + g.tcp
            c2, ok = project(corners)
            if not ok.all():
                continue
            c2 = c2.astype(int)
            body = box.name == 'body' or box.name.startswith('body_')
            for i, j in box.edges():
                cv2.line(img, tuple(c2[i]), tuple(c2[j]),
                         (200, 0, 200) if body else (255, 0, 255), 1, cv2.LINE_AA)
        tcp_uv, ok = project(g.tcp)
        if ok[0]:
            cv2.drawMarker(img, tuple(tcp_uv[0].astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        table = result.support_height
        ring = [g.tcp + [0.03 * np.cos(a), 0.03 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 25)]
        ring = np.array(ring)
        ring[:, 2] = table
        r2, ok = project(ring)
        if ok.all():
            cv2.polylines(img, [r2.astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
        m = result.metrics
        cv2.putText(img, f"grasp TCP z {g.tcp[2]:+.3f} m ({(g.tcp[2] - table) * 1000:.0f} mm above "
                    f"table), tilt {m.get('grasp_tilt_deg', 0):.0f} deg, pad engaged "
                    f"{m.get('jaw_engagement_m', 0) * 1000:.0f} mm", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _publish_shelf(self, world):
        """The shelf model's panels and the observed obstacle points on the shelf."""
        if world is None:
            return
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        if world.model is not None:
            for k, b in enumerate(world.model.boxes):
                support = b.name in world.model.supports
                color = (0.45, 0.28, 0.12, 0.8) if support else (0.75, 0.55, 0.3, 0.45)
                markers.markers.append(box_marker('shelf', k, b.position, b.rotation,
                                                  2.0 * np.asarray(b.size), self.base, stamp,
                                                  color))
        self.shelf_pub.publish(markers)
        self.shelf_cloud_pub.publish(xyz_to_pointcloud2(world.cloud, self.base, stamp,
                                                        (255, 0, 200)))

    def _candidate_markers(self, selection, stamp):
        """Gripper outlines for the candidates: chosen, planned-but-refused, screened out."""
        chosen = selection.chosen
        limit = int(self.get_parameter('grasp_markers_max').value)
        others = [c for c in selection.candidates if c is not chosen][:max(limit, 0)]
        refused, screened = [], []
        ee = self.end_effector
        # The Franka Hand: a jaw wireframe each. A modelled hand: an approach stem and a
        # thumb-side tick for the candidates, the hand's own pieces (open) for the chosen one.
        outline = (gripper_segments if ee.is_parallel_jaw else hand_axis_segments)
        for c in others:
            segs = outline(c.grasp.tcp, c.grasp.approach, c.grasp.closing)
            (refused if c.plan_metrics is not None else screened).append(segs)
        chosen_segs = (outline(chosen.grasp.tcp, chosen.grasp.approach, chosen.grasp.closing)
                       if ee.is_parallel_jaw else
                       piece_segments(ee.boxes('open'), chosen.grasp.rotation, chosen.grasp.tcp))
        markers = []
        for mid, (segs, color, width) in enumerate((
                (screened, (1.0, 0.2, 0.2, 0.25), 0.001),
                (refused, (1.0, 0.6, 0.0, 0.9), 0.002),
                ([chosen_segs], (0.1, 1.0, 0.2, 1.0), 0.002 if not ee.is_parallel_jaw else 0.004))):
            m = line_list_marker('grasp_candidates', mid,
                                 np.concatenate(segs) if segs else np.zeros((0, 2, 3)),
                                 self.base, stamp, color=color, width=width)
            markers.append(m)
        label = (f"{selection.source} score {chosen.score:.2f}" if selection.source == 'graspgenx'
                 else 'top-down fallback')
        markers.append(text_marker('grasp_candidates', 3, chosen.grasp.tcp - chosen.grasp.approach * 0.12,
                                   label, self.base, stamp, color=(0.1, 1.0, 0.2)))
        return markers


def load_end_effector(name, config_file=''):
    """The EndEffector for `end_effector`: the Franka Hand, or the Inspire hand from
    config/inspire_hand.yaml (its mount, GraspGen-X description and joint map)."""
    from tpgpt.hardware.end_effector import FRANKA_HAND, inspire_from_config
    name = str(name).strip().lower()
    if name in ('panda', 'franka', 'franka_hand'):
        return FRANKA_HAND
    if name != 'inspire':
        raise SystemExit(f"end_effector must be 'panda' or 'inspire', not {name!r}")
    import dataclasses

    from ycb_pointcloud_pipeline.inspire_hand import load_config
    cfg = load_config(config_file or None)
    ee = inspire_from_config(cfg)
    return dataclasses.replace(ee, details={**ee.details, 'config_file': cfg['_path']})


def default_config_path(name):
    """Installed config/<name>, else the source tree's."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'), 'config',
                            name)
    except Exception:  # noqa: BLE001 - package not installed
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', name)


def graspgen_server_script():
    """Installed copy of scripts/run_graspgenx_server.sh, else the source tree's."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(get_package_share_directory('ycb_pointcloud_pipeline'), 'scripts',
                             'run_graspgenx_server.sh')
        if os.path.isfile(share):
            return share
    except Exception:  # noqa: BLE001 - package not installed
        pass
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts',
                          'run_graspgenx_server.sh')
    return source if os.path.isfile(source) else None


def main(args=None):
    rclpy.init(args=args)
    node = TpgptPlannerNode()
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
