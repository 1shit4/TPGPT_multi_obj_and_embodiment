# Operator checklist: perception and planning

One page for a session at the robot, collecting the protocol decided 2026-09-25 to 09-29. It
covers the camera, segmentation and the planner; execution is being reworked separately. Why each
item is here is in `CLAUDE.md` ("Things that will bite you") and the documents named below.

## 1. Before the session

```bash
# T1  robot container (controllers, /current_pose, /joint_states)
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot

# T2  camera, then its runtime parameters (they reset on every restart)
source /opt/ros/jazzy/setup.bash && source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_infra1:=false enable_infra2:=false \
  align_depth.enable:=true \
  rgb_camera.color_profile:=848,480,30 depth_module.depth_profile:=848,480,30
bash ~/calib_setup.sh && bash ~/calib_preflight.sh      # every line must read OK

# T3  segmentation
source /opt/ros/jazzy/setup.bash && source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh
source ~/thor_ws/install/setup.bash && source ~/venvs/ycb/bin/activate
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false
```

- [ ] T3's first lines read `... red apple. red cup with a handle. silver and brown box. brown cube
      with holes.` and `accept threshold 0.35, except apple 0.48, mug 0.54, pudding_box 0.25,
      foam_brick 0.29`. Anything else means an unrebuilt config (`SEGMENTATION.md`).
- [ ] If the camera, table or shelf moved at all: recalibrate first (`CALIBRATION.md`). Nothing
      measured on the old setup carries over.
- [ ] The table around the robot's base is clear. A curled sheet of paper there read 3-4 cm tall
      and became an obstacle.
- [ ] Nothing is lying on the bottom board or standing in `bottom_right` (both seen 2026-09-26).

## 2. Placing the object (the protocol poses)

Put it in the pick area on the table in front of the robot, one object at a time if possible.

| object | how it is set down | why |
|---|---|---|
| mug | **opening up, handle visible to the camera** | upright and upside down look alike from this camera; a hidden handle cannot be fitted |
| pudding box | **standing 88 mm tall on its 70 x 35 mm face** (as since 09-25) -- not on its side (70 mm), not flat | the planner's model is this box (88 x 70 x 35, not the YCB one). Laid lower the fit is refused ("is it lying down?"), stood taller "standing another way?" |
| foam brick | on any face | -- |
| banana, apple | as they rest | -- |

- [ ] Your hands and arms are out of the camera's view (an arm reaching in once read as a mug).
- [ ] The pudding box next to the brick is detected on only about 6 frames in 10; alone, 8-9.

## 3. Each planning run

```bash
# home the arm first: the path's joint angles start from where the arm is
source /opt/ros/jazzy/setup.bash && source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh
source ~/thor_ws/install/setup.bash
ros2 run ycb_pointcloud_pipeline joint_homing

# T4  planner (tpgpt venv); object_name and destination_name as needed
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/banana_20260925_183904/source_bundle \
    -p object_name:=pudding_box -p destination_name:=top_left
```

- [ ] Destinations: `top_left`, `top_right`, `bottom_left`. `bottom_right` is not used.
- [ ] Restart the planner after any rebuild; a running node keeps its old code.
- [ ] Give T3 a few seconds of publishing the object before planning; the planner waits up to 12 s
      for three frames and fits better on three than on one.

## 4. Reading the planner's log

| it says | it means | do |
|---|---|---|
| `Object model: <obj> scan fitted (...)` | the YCB scan is placed on the object | nothing |
| `fit REFUSED ... trying again (1 of 2)` | one frame did not fit; it retries on fresh ones | nothing, usually |
| `no attempt was accepted ... Planning on the camera cloud` | no scan fit, no grasp library | check the object's pose and that it is fully in view |
| `... standing another way?` | the object is not in its protocol pose | set it down as in section 2 |
| `planning on 1 of 3 camera frames` | the object was published intermittently | watch T3; check the pose and the lighting |
| `approach_blocked ... re-home the arm` | the grasp is fine; getting there from where the arm is is not | `joint_homing`, plan again |
| `Plan saved to ...` after 17-60 s, grasp rank 1-8 | normal (2026-09-26: 15 of 15 object-slot pairs) | -- |
| `no_executable_grasp` | no grasp's whole path passed | send me the `_refused` directory's name; it keeps everything needed to replay it |

## 5. When detection goes wrong

T3 says why an object is missing (below its threshold, never seen, a box too big to be it, the
same box as another object, no depth in the mask). To have it fixed, record the frames it gets
wrong and say what was on the table:

```bash
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false \
    save_frames:=all save_frames_dir:=$HOME/ycb_detection_frames/<scene_name>
```
