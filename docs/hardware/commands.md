```bash
source /opt/ros/jazzy/setup.bash 
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
```

```bash
source ~/venvs/ycb/bin/activate   # segmentation node
source ~/venvs/tpgpt/bin/activate # teaching, planner, executor and pick-place nodes (they import tpgpt)
```

Run the Robot, configures all the controllers present in yaml
```bash
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
```

Launch the segmentation and pcld pipeline
```bash
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py
```

Teaching Node
```bash
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=banana
ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard
```

Plan + Exec in one go (plans, prints the executor summary, type `yes` to execute)
```bash
rviz2 -d ~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/rviz/tpgpt.rviz

ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo_file>/source_bundle \
    -p object_name:=banana -p destination:="[0.60, -0.25, 0.006]" \
    -p dry_run:=false -p gripper_mode:=none
```
- No `plan_dir` and no `ros2 service call`: the plan is made as soon as the object clouds
  arrive, saved to `~/tpgpt_plans/<object>_<time>/`, and handed to the executor.
- `dry_run` defaults to `true` (nothing moves, no `yes` prompt); `dry_run:=false` moves the robot.
- On a failed plan, a `no` at the prompt, or an abort before the arm moved, it asks `Re-plan? [y/N]`.
- Every `-p` reaches both nodes; `workspace_min` / `workspace_max` change both.
- The planner checks every candidate's whole path against the shelf (`config/shelf.yaml`, rebuild
  after editing) and the scene cloud on it, and needs `/joint_states` (robot container up). In RViz:
  "Shelf model" and "Shelf cloud (obstacles)". The log line `Shelf world: {...}` gives
  `explained_by_model` (close to 1 for an empty, correctly modelled shelf).
  See `docs/hardware/SHELF_AND_PATH_CHECK.md`.

Planning node (separately)
```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo_file>/source_bundle \
    -p object_name:=banana -p destination:="[0.60, -0.25, 0.006]"

ros2 service call /tpgpt_planner_node/plan std_srvs/srv/Trigger

rviz2 -d ~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/rviz/tpgpt.rviz
```

Exec Node (separately)
```bash
rviz2 -d ~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/rviz/tpgpt.rviz

ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
    -p plan_dir:=$HOME/tpgpt_plans/<plan file> -p dry_run:=false -p gripper_mode:=none
```

To collect the experimentd data
```bash
 ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p experiment:=true -p trial_id:=0031 -p hand:=inspire \
    -p object_name:=apple -p destination_name:=top_shelf

```

free-drive gravity comp trigger
```bash
ros2 run ycb_pointcloud_pipeline free_drive 
```

joint homing 
```bash
ros2 run ycb_pointcloud_pipeline joint_homing 
```