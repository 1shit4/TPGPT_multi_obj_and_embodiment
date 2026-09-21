# ycb_pointcloud_pipeline (Jetson Thor, Ubuntu 24.04, ROS 2 Jazzy)

```
RealSense D435 ──► realsense2_camera node ──► /camera/camera/color/image_raw
                                              /camera/camera/aligned_depth_to_color/image_raw
                                              /camera/camera/color/camera_info
                                                       │
                                                       ▼
                 ycb_segmentation_node (every 3 s: Grounding DINO + SAM on GPU)
                                                       │
                   /ycb_pointclouds/<name>  (sensor_msgs/PointCloud2, one per YAML object)
                   /ycb_detections/image    (annotated image, for debugging)
```

Only objects listed in `config/target_objects.yaml` are detected. A listed object
that is not visible simply gets no message that cycle.

## 1. One-time setup

### System packages + camera udev rules (needs sudo)
```bash
sudo apt update
sudo apt install -y ros-jazzy-realsense2-camera ros-jazzy-rqt-image-view python3-venv
sudo curl -L -o /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Plug the camera into a **USB 3** port (blue), then check it shows up:
```bash
source /opt/ros/jazzy/setup.bash
rs-enumerate-devices -s        # should list "Intel RealSense D435" with a USB type of 3.x
```

### Python venv with GPU PyTorch
```bash
python3 -m venv ~/venvs/ycb --system-site-packages     # system site pkgs = rclpy, cv_bridge
source ~/venvs/ycb/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # True NVIDIA Thor
```
Model weights (Grounding DINO base ~900 MB, SAM ViT-L ~1.2 GB) download
automatically on the first run into `~/.cache/huggingface`.

### Build
```bash
cd ~/thor_ws
source /opt/ros/jazzy/setup.bash
source ~/venvs/ycb/bin/activate
colcon build --symlink-install --packages-select ycb_pointcloud_pipeline
```
The launch file starts the node with `python3` from PATH, so the venv must be
activated before `ros2 launch` (it provides torch). For `ros2 run`, use
`ros2 run --prefix python3 ycb_pointcloud_pipeline segmentation_node`.

## 2. Run
```bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/ycb/bin/activate
source ~/thor_ws/install/setup.bash
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
```

Launch arguments:

| arg | default | meaning |
|---|---|---|
| `period` | `3.0` | seconds between detections / point cloud publishes |
| `targets` | `config/target_objects.yaml` | YAML file with objects to detect |
| `profile` | `640,480,15` | color + depth stream `W,H,FPS` |
| `serial_no` | first camera | RealSense serial number |
| `launch_camera` | `true` | `false` if you run the camera node yourself |
| `detector_mode` | `grounded_sam` | `dummy` = fixed center box, no models (plumbing test) |

The two nodes can also run in separate terminals:
```bash
# terminal 1 - camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
    rgb_camera.color_profile:=640,480,15 depth_module.depth_profile:=640,480,15
# terminal 2 - segmentation
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false
```

## 3. Check the output
```bash
ros2 topic list | grep ycb
ros2 topic hz /ycb_pointclouds/banana          # ~0.33 Hz while a banana is visible
ros2 run rqt_image_view rqt_image_view /ycb_detections/image
rviz2   # Fixed Frame: camera_color_optical_frame, add PointCloud2 -> /ycb_pointclouds/<name>
```

## 4. Choosing objects
Edit `config/target_objects.yaml` (with `--symlink-install` no rebuild is
needed, just restart). Entries are either a plain string or a name + prompt:
```yaml
target_objects:
  - banana                    # topic /ycb_pointclouds/banana
  - name: master_chef_can     # topic /ycb_pointclouds/master_chef_can
    prompt: blue coffee can   # what Grounding DINO looks for
```
Describe what the object *looks like*; YCB brand names are often not recognised.

## 5. Tuning (ROS parameters of `ycb_segmentation_node`)
- `box_threshold` (0.35): raise if you get false detections, lower if objects are missed.
- `depth_trunc` (2.0 m): points farther than this are dropped.
- `grounding_model` / `sam_model`: e.g. `IDEA-Research/grounding-dino-tiny`,
  `facebook/sam-vit-base` for faster inference, `facebook/sam-vit-huge` for better masks.
- `save_pcd` (false) / `output_dir`: also write `.pcd` files.
- One cloud is published per object name (the highest-scoring instance).

## 6. FR3 joint target test (`joint_target_publisher`)

Activates a CRISP **joint impedance** controller (`config/joint_impedance_controller.yaml`,
a `crisp_controllers/CartesianController` with zero Cartesian gains and no nullspace
projection) and moves through 3 targets close to the current configuration
(default: joint1 +0.05 rad; joint1 +0.05 & joint4 -0.05 rad; back to start).

**This moves the real robot.** Supervised session only, e-stop in reach. The gains
in the YAML are soft starting values, not validated for your robot/tool/payload.

One-time on the host: `sudo apt install ros-jazzy-controller-manager-msgs`

```bash
# terminal 1
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
# terminal 2
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh      # ROS_DOMAIN_ID=61
source ~/thor_ws/install/setup.bash
ros2 run ycb_pointcloud_pipeline joint_target_publisher
```

The node loads/configures `joint_impedance_controller` if needed (the container reads the
YAML via the `/workspaces/thor_ws` mount), asks you to type `yes`, strictly switches it
in place of whatever owns the FR3 effort interfaces, then sends the targets with a smooth
4 s interpolation each. It aborts and holds the current position on stale joint states or
tracking error > 0.15 rad, refuses offsets > 0.1 rad or targets within 0.15 rad of a joint
limit, and leaves the controller active holding the last target.

Parameters: `offsets` (7 values per target, rad), `move_duration`, `dwell_duration`,
`max_offset`, `max_tracking_error`, `limit_margin`, `require_confirmation`.

Switch back to Cartesian control afterwards:
```bash
ros2 control switch_controllers --activate cartesian_impedance_controller \
    --deactivate joint_impedance_controller --strict      # needs ros-jazzy-ros2controlcli
```

## 7. FR3 Cartesian goal (`cartesian_target_publisher`)

Activates CRISP's `cartesian_impedance_controller` (spawned inactive by the robot launch)
and streams `geometry_msgs/PoseStamped` targets on `/target_pose` (frame `base`) that move
`fr3_hand_tcp` smoothly to a fixed goal, then keeps streaming the goal until Ctrl-C.

**This moves the real robot.** Supervised session only, e-stop in reach.

```bash
# terminal 1
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
# terminal 2 -- look at where the hand is now, then pick a goal close to it
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh
source ~/thor_ws/install/setup.bash
ros2 topic echo --once /current_pose
ros2 run ycb_pointcloud_pipeline cartesian_target_publisher --ros-args \
    -p goal_position:="[0.45, 0.0, 0.40]"
    # optional: -p goal_orientation:="[1.0, 0.0, 0.0, 0.0]"   ([x,y,z,w]; default keeps current)
```

Safety: goals outside the box `workspace_min`/`workspace_max` (default x 0.2..0.75, y -0.5..0.5,
z 0.1..0.8 m) or more than `max_distance` (0.3 m, hard cap 0.4 m) from the current position are
refused; speed is limited by `max_linear_speed` (0.05 m/s) and `max_angular_speed`
(0.25 rad/s); tracking error > `max_tracking_error` (0.12 m) or a stale `/current_pose`
aborts and holds the current pose. CRISP also clips the pose error to 0.1 m / 0.5 rad.

## 8. TPGPT: transported policy from real perception to the FR3

> **Inspire RH56 hand instead of the Franka Hand:** mounting relative to joint 7, where the
> gripper is switched, the side grasp for upright round objects (no GraspGen-X) and the
> commands are in [INSPIRE_HAND.md](INSPIRE_HAND.md). In short: `inspire_hand_node`, and
> `-p end_effector:=inspire -p gripper_mode:=inspire` on the planner / executor / teach node.

The full guide -- frames, the calibration file, how to verify the point cloud pose,
how to teach the demonstration, what each check measures, and what was validated -- is
[`TPGPT_multi_obj_and_embodiment/docs/hardware_integration.md`](../TPGPT_multi_obj_and_embodiment/docs/hardware_integration.md).
Short version:

```
FR3 hand-guided (FCI on) -> tpgpt_teach_node -> ~/tpgpt_demos/<demo>/source_bundle   (the source demonstration, once)
camera -> segmentation_node -> /ycb_pointclouds/<obj> (optical frame)
       -> [camera_calibration.yaml: base -> camera_link, + realsense TF] -> base frame
       -> tpgpt_planner_node   (grasp from GraspGen-X, keypoints, transportation map, transported labels)
            ^ GraspGen-X server (6DOF_MultiEmbod_Grasp/GraspGenX, GPU, ZMQ :5556; started by the node if needed)
       -> tpgpt_executor_node  (TPGPT's own execution loop -> CRISP /target_pose [+ stiffness])
```

| file / node | role |
|---|---|
| `config/camera_calibration.yaml` | base -> camera extrinsics, colour intrinsics, table height (lab calibration, verified) |
| `camera_tf_publisher` | publishes base -> camera_link from the YAML on /tf_static |
| `calibration_check_node` (`ros2 run`, view in RViz with `rviz/tpgpt.rviz`) | verifies the cloud pose manually; moves nothing |
| `tpgpt_teach_node` (+ `launch/tpgpt_teach.launch.py`) | records a hand-guided demonstration and saves it as a TPGPT source bundle; plots it in RViz |
| `tpgpt_teach_keyboard` | single-key control of the teach node (run in its own terminal) |
| `config/tpgpt_teaching_controller.yaml` | zero-stiffness CRISP controller that lets the arm be moved by hand |
| `tpgpt_planner_node` | chooses a grasp with GraspGen-X and plans from the object cloud; moves nothing |
| `scripts/run_graspgenx_server.sh` | GraspGen-X grasp server (`--daemon`, `--stop`, `--status`; log `~/.ros/graspgenx_server.log`) |
| `tpgpt_executor_node` | executes a saved plan; `dry_run:=true` by default |
| `config/tpgpt_variable_stiffness_controller.yaml` | CRISP controller used when `use_transported_stiffness:=true` |

```bash
# host terminals: source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh + ~/thor_ws/install/setup.bash
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot   # FCI active
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py            # camera + segmentation (ycb venv)

# once: teach the source demonstration by hand (guide section 5)
source ~/venvs/tpgpt/bin/activate                                     # nodes that import tpgpt
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=mustard_bottle
ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard                 # another terminal
#   c capture_scene (arm out of view) -> t float the arm (type yes) -> s record
#   -> g at the grasp -> carry, set down -> r at the release -> x stop (arm held,
#   bundle saved to ~/tpgpt_demos/<demo>/source_bundle, plotted in RViz)
# (alternative: MuJoCo source, ~8 s:
#   cd ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment
#   MUJOCO_GL=egl python -m tpgpt.hardware.export_source --out outputs/hardware/source_bundle)

ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=mustard_bottle -p destination:="[0.70, 0.15, -0.006]"
ros2 service call /tpgpt_planner_node/plan std_srvs/srv/Trigger

ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
    -p plan_dir:=$HOME/tpgpt_plans/<plan> -p dry_run:=true             # nothing moves
# supervised, robot container up, extrinsics calibrated:
#   -p dry_run:=false [-p speed_scale:=0.5]
#   (use_transported_stiffness:=true only for the simulated source; a taught one is static)

# or both in one go: plan, print the executor summary, type 'yes' to execute
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=banana -p destination:="[0.60, -0.25, 0.006]" \
    -p dry_run:=false -p gripper_mode:=none
```

### Grasp selection (GraspGen-X)

The planner no longer uses one scripted top-down grasp. For each plan it:

1. sends the fused object cloud to **GraspGen-X** for the Franka Hand (`franka_panda`: the
   same hand, same pads, 0.1034 m fingertip depth), which returns ~100 scored 6-DoF grasps
   from any direction (0.4-0.8 s on the Thor GPU);
2. **screens** them on the cloud with the FR3 hand's real collision geometry
   (`tpgpt/hardware/hand.py`, from `franka_description`), naming each rejection:
   `from_below` (approach > `grasp_max_tilt_deg` from vertical, through the table),
   `too_wide` (an open finger would land on the object), `collides` (palm in the object),
   `empty` (pads close on air), `shallow` (< 10 mm of the 18.5 mm pads engaged),
   `hits_support` (fingers under the table), `duplicate`;
3. **plans** the survivors best score first and keeps the first whose plan passes:
   `min det(J)` ≥ `min_determinant`, transported path within 10 mm / 15° of the grasp, hand
   above the table and the destination along the whole path.

Width is measured **between the pads**, not over the whole object: a cup wider than the
80 mm hand is gripped by its wall (one pad inside, one outside). The scripted top-down grasp
refuses that cup as `object_too_wide`; GraspGen-X + screening picks a rim grasp.

If the server is down and cannot be started, or nothing passes, the planner falls back to the
top-down grasp and logs a warning (`grasp_fallback_top_down:=false` to refuse instead).
RViz (`/tpgpt/plan/markers`, ns `grasp_candidates`): chosen grasp thick green, planned but
refused orange, screened out faint red. Every candidate and its verdict is saved to
`<plan>/grasp_candidates.json`.

| parameter | default | |
|---|---|---|
| `grasp_source` | `graspgenx` | `top_down` = old scripted grasp |
| `graspgen_host` / `graspgen_port` | `127.0.0.1` / `5556` | |
| `graspgen_autostart` | `true` | start `run_graspgenx_server.sh --daemon` if unreachable |
| `graspgen_planner` | `diffusion` | or `graspmoe` (diffusion + box-fitted grasps) |
| `graspgen_num_grasps` / `graspgen_topk` | 200 / 100 | |
| `grasp_max_tilt_deg` | 85 | lower (e.g. 30) to prefer top-down grasps |
| `grasp_max_plans` | 12 | candidates planned before giving up |
| `grasp_min_path_clearance` | -0.002 | m; hand may graze a surface by this much |
| `grasp_fallback_top_down` | `true` | |

GraspGen-X is a diffusion model with no seed: the same scene gives a different (equally
screened) grasp on each `~/plan` call. Setup of its venv (`~/venvs/graspgenx`, torch cu130) and
assets (`6DOF_MultiEmbod_Grasp/assets/`, 2.3 GB) is in the header of
`scripts/run_graspgenx_server.sh`.

`--prefix python3` must come **before** the package name, with the TPGPT venv active.
No gripper is used while teaching: `g` and `r` only mark when the object was held.
The demonstration is executed with the stock controller's static 400 N/m / 40 N·s/m,
so the arm runs `40/400 · v` behind the taught path (10 mm at 10 cm/s, half that at
the executor's default `speed_scale` 0.5). Teach slowly.
Tests: `python3 -m pytest test/test_camera_calibration.py`.
