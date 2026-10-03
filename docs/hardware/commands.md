# Running the robot: commands

Every section below assumes the two source lines in **Setup**. Background for the choices here:
`SHELF_AND_PATH_CHECK.md` (shelf, collisions, grasp selection), `hardware_integration.md` (frames,
teaching, execution), `FAITHFULNESS_AUDIT.md` (where the robot differs from the simulation and why).

## 1. Setup

In **every** terminal:
```bash
source /opt/ros/jazzy/setup.bash
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
```
`host_ros_env.sh` sets `ROS_DOMAIN_ID=61`; without it no robot topics are visible.

Then the venv the node needs — they are not interchangeable:

| venv | for |
|---|---|
| `source ~/venvs/ycb/bin/activate` | `segmentation_node` (torch, Grounding DINO + SAM) |
| `source ~/venvs/tpgpt/bin/activate` | planner, executor, pick-place, teaching, `shelf_fit`, `shelf_points.py`, tests |
| (none) | `free_drive`, `joint_homing`, `calibration_check_node`, RViz |

Nodes that import `tpgpt` are launched with `ros2 run --prefix python3` so they use the active venv.
Without the venv: `ModuleNotFoundError: No module named 'tpgpt'`.

**After editing anything in the package, including config:**
```bash
cd ~/thor_ws && colcon build --packages-select ycb_pointcloud_pipeline
```
Nodes read the **installed** copy, and a running node keeps the code it started with — rebuild, then
restart the node.

## 2. Bring-up

The three things that must be running before anything else, and that stay up for the whole session:
the **robot container**, the **camera**, and **segmentation**. Nothing here is specific to teaching,
planning or executing — those come later and each brings its own viewer.

One terminal each. The camera and segmentation are separate so the camera can warm up (and stay up)
while segmentation restarts.

```bash
# T1  robot container (controllers, /current_pose, /joint_states)
```
```bash
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
```
```bash
# T2  camera
```
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_infra1:=false enable_infra2:=false \
  align_depth.enable:=true \
  rgb_camera.color_profile:=848,480,30 depth_module.depth_profile:=848,480,30
  ```
  ```bash
# T3  segmentation (NOT its own camera)
```
```bash
source ~/venvs/ycb/bin/activate
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false \
    save_frames:=misses save_frames_dir:=~/ycb_detection_frames
```
- **`launch_camera:=false` matters.** The pipeline launch starts its own camera otherwise, and two
  drivers fight over the D435: `xioctl(VIDIOC_S_FMT) failed, errno=16 ... Device or resource busy`.
- The intrinsics are valid at **848x480** only.
- Check the object is seen before anything else (about one cloud every 3 s):
```bash
ros2 topic hz /ycb_pointclouds/banana        # the object you are about to use
ros2 topic hz /ycb_scene_cloud
ros2 run rqt_image_view rqt_image_view /ycb_detections/image     # masks and scores
```
- **Objects that are detected** (`config/target_objects.yaml`): `banana`, `apple`,
  `tomato_soup_can`, `pudding_box`, `foam_brick`. Each has a YCB scan, so the planner fits the scan
  to the camera cloud (§6). The mug is commented out (it was matching a cup behind the robot).
- Adding an object: put its YCB directory in `tpgpt/hardware/object_model.py` (`YCB_SCANS`), its
  prompt in `target_objects.yaml`, download the scan into `assets/ycb/`, rebuild, restart
  segmentation.

## 3. Teaching a demonstration

One demonstration is transported to every later object and destination, so teach the task you
actually want — including the harder destination (`FAITHFULNESS_AUDIT.md`, A12: a demonstration
that lowers onto the top board from above cannot reach under it; one that slides in under the
bottom board reaches both).

Camera and segmentation must be running; the object must be detected.

```bash
# T4  teaching node. It starts its OWN RViz on tpgpt.rviz (rviz:=false to suppress it)
source ~/venvs/tpgpt/bin/activate
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=banana

# T5  the keys (a launched node has no keyboard, so they live here)
ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard
```
In that RViz you watch the captured object cloud, the magenta trail of the recording, and the
placement it derives (`/tpgpt/teach/*`).

| key | when | what it does |
|---|---|---|
| `c` | arm out of the camera's view | captures the object's cloud (waits for 3 fresh clouds) |
| `t` | **e-stop in reach** | type `yes`; the arm goes limp (zero stiffness) |
| `s` | | starts recording |
| `g` | at the grasp pose | marks the grasp — fingertips where the robot should hold it |
| `r` | at the release, object set down | marks the release |
| `x` | | stops, switches back to the holding controller, converts and saves |
| `p` | optional, arm out of view | captures the placed object to check the derived placement |
| `h` | any time | hold: back to the holding controller, discards a recording in progress |

Order: `c`, `t`, `s`, move, `g`, carry, set down, `r`, retract, `x`.

While teaching: **under 10 cm/s**, one grasp and one release, don't move the object between `c` and
the grasp, and start and end with the hand in front of the robot (a transported start near the base
is awkward for the executor). No gripper is used — hold the object against the open fingers.

The recording is discarded, and the arm put back on the holding controller, if a pose arrives more
than 0.2 s late, if the TCP leaves the teach node's own workspace box, or after 120 s. Refusals at
`x` (`marks_missing`, `no_motion`, `carry_too_short`, `grasp_not_on_object`, ...) each name their
cause, and `raw.npz` is always written first, so a refusal loses nothing:
`python -m tpgpt.hardware.teach --raw <demo>/raw.npz --out <demo>/source_bundle` re-runs the
conversion offline.

It saves to `~/tpgpt_demos/<object>_<time>/` (`raw.npz`, `source_bundle/`, `teach.json`). Check the
summary it prints:

| field | good | meaning |
|---|---|---|
| `grasp_to_cloud` | < 10 mm | the grasp mark really sat on the object (refused above 50 mm) |
| `max_speed` / lag | < 10 cm/s, lag < 10 mm | the arm runs behind the taught path by `D/K·v` |
| `carry_tilt_ok` | `true` | above 10 deg only the **derived destination height** is approximate |
| `destination` | the board you used | derived as support + (TCP rise); compare with `shelf.yaml` |
| `n_labels` | a few hundred | 20 Hz after trimming still periods |

### 3b. Replaying a demonstration (learning its attractor)

TPGPT's labels are the attractor (the point the controller's spring pulls the hand toward), not
where the hand went, which is all a demonstration taught by hand records. So the robot replays it,
pass by pass, and learns the attractor that makes the arm retrace it (`tpgpt/hardware/replay.py`,
`ilc.py`; audit B1). Before moving, each pass checks the taught path, widened by how far the arm
strays from it, against the shelf model and the table. Pass 0 is run twice, without the object;
later passes with it. Home the arm before each pass (§8).

```bash
source ~/venvs/tpgpt/bin/activate
D=~/tpgpt_demos/banana_20260925_183904
# once per demonstration: the replay directory (pass 0 = the taught path)
python -m tpgpt.hardware.replay init --bundle $D/source_bundle_directional --out $D/replay
# one pass; asks 'yes'. Without the object:
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
  -p replay_dir:=$D/replay -p dry_run:=false -p gripper_mode:=none \
  -p rotational_stiffness:=30.0 -p transit_rotational_scale:=1.0
# with the object (camera and segmentation up, §2):
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
  -p replay_dir:=$D/replay -p dry_run:=false -p gripper_mode:=franka_grasp \
  -p rotational_stiffness:=30.0 -p transit_rotational_scale:=1.0
# when a pass's runs are done: the next attractor, and how the passes went
python -m tpgpt.hardware.replay update $D/replay
python -m tpgpt.hardware.replay status $D/replay
# once learning is done: the demonstration to plan with (§4), the last pass's command as labels
python -m tpgpt.hardware.replay bundle $D/replay --out $D/source_bundle_learned
```
- The wrist at 30 N·m/rad the whole way (`transit_rotational_scale:=1.0`: not softened while
  carrying). At 60 the wrist vibrated as joint 7 neared its limit, and a 15 -> 60 step at the
  set-down came before a reflex stop; softened to 7.5 while carrying, pass 0 is refused (audit C5).
  Write numbers with the `.0`. A plan from `source_bundle_learned` runs at the settings it was
  learned under by itself (these two, its `speed_scale`, its stiffness labels, no offset
  compensation or arrival hold) and refuses other values for them: leave them out.
- **With the object**, before asking `yes` the node guides it onto its taught place. Watch
  `ros2 run rqt_image_view rqt_image_view /tpgpt/replay/place_overlay`: the camera's picture,
  green outline where the object was taught, orange where the camera fits it now; the same
  advice is printed ("move it N mm toward the robot ..., turn it N deg ..."). Press Enter when it
  matches; beyond 10 mm / 10 deg it asks again (Enter keeps aligning, `go` starts anyway).
- After a pass it prints how far the arm was from the taught path and then `Next:` (run it again,
  or `update`). `Aborted: ... collision` before moving: the path does not fit with that margin;
  `... on the way onto the path`: home the arm and run it again.
- `update` warns when a pass came out worse than the one before: `rm -r $D/replay/pass_00N` (the
  pass it names) and `update --gain 0.5`.
- `-p dry_run:=true` previews a pass: the check and the summary, nothing recorded.

## 4. Planning (nothing moves)

```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=banana -p destination_name:=top_left
```
`dry_run` defaults to **true**: it plans, prints the executor summary and simulates the motion.
`Re-plan? [y/N]` appears after a refusal.

Destinations by name resolve to the slot marks the robot touched (`config/shelf.yaml`, `slots:`):
`top_left`, `top_right`, `bottom_left`, `bottom_right`. `-p destination:="[x, y, z]"` still works.

The planner alone, re-planning on demand (useful while adjusting the scene):
```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=banana -p destination_name:=top_left
ros2 service call /tpgpt_planner_node/plan std_srvs/srv/Trigger
```

**What to read in the log**

| line | healthy |
|---|---|
| `Object model: <obj> scan fitted (...)` | accepted; `REFUSED` means it planned on the camera cloud instead |
| `Shelf world: {...}` | `explained_by_model` close to 1 (0.96 on an empty, correctly modelled shelf), `removed_as_robot` > 0 when the arm is in view |
| `grasp from graspgenx: ...` | the chosen grasp, and the tally of what removed the rest |
| `Plan saved to ~/tpgpt_plans/<object>_<time>` | plus `min det(J)` (> 0.1), aim errors, hand clearance at pick and place |

**What to look at in RViz** (fixed frame `base`), in its own terminal — skip it if the teaching
launch already opened one on the same config:
```bash
rviz2 -d ~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/rviz/tpgpt.rviz
```

| display | shows |
|---|---|
| Plan target cloud (orange) | what was planned on: the fitted scan, or the camera cloud |
| Plan observed cloud (white) | what the camera actually saw |
| Shelf model / Shelf cloud (magenta) | the model, and the points it does not explain |
| Plan markers / Plan path | the hand path (blue open, red closed), grasp and release frames, candidates (green chosen, orange refused, faint red screened) |
| Plan placed cloud | where the object ends up |

The overlay image, also saved as `overlay.png` in the plan directory:
```bash
ros2 run rqt_image_view rqt_image_view /tpgpt/plan/overlay
```

**Parameters worth knowing**

| parameter | default | what it does |
|---|---|---|
| `graspgen_num_grasps` / `graspgen_topk` / `grasp_max_plans` | 400 / 200 / 200 | GraspGen-X is a diffusion sampler: more candidates mean planning succeeds more often |
| `object_model` | `true` | fit the YCB scan; `object_model_retries` (2) retries on fresh frames |
| `check_visibility` | `auto` | on only when the scan fit was refused |
| `check_centred` | `false` | off: these objects are light |
| `grasp_max_tilt_deg` | 90 | approach vs straight down; past 90 the hand comes from under the table |
| `support_height` | NaN | inferred: a shelf board the object stands on, else the table |
| `kinematic_criterion` | `strict` | every waypoint reachable; orientation may be relaxed in transit only |
| `allow_missing_joint_states` | `false` | `true` only for offline dry runs with the robot off |

## 5. When a plan is refused

Every refusal writes `~/tpgpt_plans/<object>_<time>_refused/` with `grasp_candidates.json` (every
candidate, its score, tilt, jaw verdict and rejection) and `clouds.npz` (observed, planned-on and
obstacle clouds). Read the tally first:

| tag | meaning | normal? |
|---|---|---|
| `from_below` | approach past 90 deg: the hand comes from under the table | yes, often half the set |
| `hits_support` | the hand's lowest point is below the table (the body is 204 mm wide across the fingers) | yes for low objects |
| `empty` / `shallow` / `too_wide` | the jaws: nothing between the pads / under 10 mm of pad engaged / material under an open finger | yes |
| `duplicate` | same grasp as one already tried | yes |
| `visibility` | approaches from the side the camera never saw | only when the scan fit was refused |
| `grasp_collision` | the hand or its approach meets the scene at the grasp | suspicious on an open table |
| `place_zone` | the hand arrives at the release through the shelf | tells you which panel |
| `collision` | the **whole transported path** hit something; names the stage and the part | the real check |
| `plan_refused` | the map refused: folded (`min det`), aim, orientation, `object_too_wide` | folding means the demonstration does not fit the task |
| `outside_workspace` | a label leaves the executor's box | only if `workspace_check` is on (off by default) |

Inspect a record:
```bash
python3 - <<'EOF'
import json, collections
j = json.load(open('/home/krishna/tpgpt_plans/<dir>_refused/grasp_candidates.json'))
print(j['object_cloud_source'], j['object_model'])
print(j['stages'])
print(collections.Counter((c.get('rejected') or '').split(':')[0] for c in j['candidates']))
for c in j['candidates'][:10]:
    print(round(c['score'], 3), round(c['tilt_deg']), c.get('rejected'))
EOF
```

## 6. Executing on the robot

A dry run really executes the policy against a model of the arm, in real time, so it takes as long
as the real motion (~116 s at `speed_scale` 0.35) and shows `/tpgpt/executor/simulated_pose`.

```bash
# the arm must not move between planning and execution (else the start may be at most 0.6 m away)
ros2 run ycb_pointcloud_pipeline free_drive          # Ctrl-C when done
# or
ros2 run ycb_pointcloud_pipeline joint_homing        # only when the path to home is clear

# plan and execute, with the first-run guards
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/banana_20260925_183904/source_bundle_learned \
    -p object_name:=foam_brick -p destination_name:=top_left \
    -p dry_run:=false -p gripper_mode:=franka_grasp -p grasp_force:=15.0
```
`gripper_mode`: `franka_grasp` (default) | `gripper_command` | `inspire` | `none`. `none` runs the
motion without touching the gripper, which is the safer first move on a new plan.
The grip force is the object's, from `config/grasp_forces.yaml` (the run logs `grip force N N`);
`-p grasp_force:=N` overrides it for one run (as above: 15 N against the brick's 20).

Type `yes` at the prompt. **Hand on the e-stop.** Inspect the plan in RViz before confirming — the
same view as §4:
```bash
rviz2 -d ~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/rviz/tpgpt.rviz
```
During the run it adds the executor's own displays: the commanded pose, the dry-run simulated pose,
and `/current_pose` (the live arm). In a dry run the first two move and the third does not.

An existing plan, without re-planning:
```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
    -p plan_dir:=$HOME/tpgpt_plans/<plan> -p dry_run:=false -p gripper_mode:=franka_grasp
```

Guards, and what they mean when they fire:

| abort | meaning |
|---|---|
| `the TCP is X m from the plan start (max_start_distance 0.6) and the arm has moved since the plan was made` | re-plan from where the arm is |
| `max_policy_seconds exceeded` | the policy is longer than the guard (300 s); check `speed_scale` |
| `execution watchdog declared the arm stuck` | the lag gate; `on_stall:=abort` stops, `continue` (default) switches the gate off for the rest of the run (audit C1) |
| `... outside the workspace box` | only with `workspace_check:=true`; off by default |

## 7. X6 experiment trial

```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p experiment:=true -p trial_id:=0031 -p hand:=inspire \
    -p object_name:=apple -p destination_name:=top_shelf
```
X6 overrides some selection defaults from its frozen `method:` block; see
`x6_experiment/RUNNING_THE_EXPERIMENT.md`. The config still predates the 2026-09-21 move.

## 8. Utilities

```bash
ros2 run ycb_pointcloud_pipeline free_drive                     # float the arm (gravity comp)
ros2 run ycb_pointcloud_pipeline joint_homing                   # go to the home configuration
ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.0, speed: 0.03}"

ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.08, speed: 0.03}"
ros2 run rqt_image_view rqt_image_view /ycb_detections/image    # what segmentation sees
ros2 node list                                                  # the authority on what is running
ros2 daemon stop && ros2 daemon start                           # if node/topic lists look stale
```

---

## 9. Camera calibration

Full background, tolerances and why each step is what it is: `docs/hardware/CALIBRATION.md`.
Everything below assumes the sourcing in §1.

### Before anything: let the camera warm up

**The D435's depth drifts ~10 mm over the first ~15 minutes from a cold driver start.** That is
larger than the calibration's own accuracy, so a cold camera makes a good calibration look broken
and bakes an error into a new one. Start the camera, then wait.
```bash
ps -eo etime,comm | grep realsense          # want > 15 min
python3 ~/calib_tools/warmup_drift.py       # optional: samples the table every ~3 min
```

### Is the calibration still good? (2 minutes, no robot)

Run this whenever something looks off, or before a session. It compares surfaces the camera sees
against heights measured by touching.
```bash
python3 ~/calib_tools/check_calibration.py
```
It prints each surface against its touched anchor, the spread over 5 frames, and — the useful line —
each board's height **relative to the table**, which is what separates a moved shelf from a moved
camera. It also refuses to be trusted if the camera has been up less than 15 min. Or watch `calibration_check_node`'s `table plane:` line (below). Healthy: tilt < 1.5 deg, height
error within a few mm, logged at INFO.

**Reading the result — this is the part that matters:**
- **All surfaces off by a similar amount** -> the camera moved, or it is cold. Not the scene.
- **Only the shelf off, table fine** -> the shelf moved. Compare the shelf's height *relative to
  the table*; a global drift cancels in that number, a real move does not.
- **Errors that vary with position** (e.g. table -9, shelf +4) -> a rotation error in the
  calibration. Refit; do not just shift it.

The table is the control: it cannot move, so any error there is the camera or the calibration.

### Recalibrate

Needed after the camera is moved or bumped. Takes ~30-40 min. Five terminals.

**1. Camera** (terminal 1), then wait 15 min:
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_infra1:=false enable_infra2:=false \
  align_depth.enable:=true pointcloud.enable:=true \
  rgb_camera.color_profile:=848,480,30 depth_module.depth_profile:=848,480,30
```
The intrinsics in `camera_calibration.yaml` are only valid at **848x480**. On this ARM build the
pointcloud filter is `pointcloud__neon_`, not `pointcloud` — `pointcloud.enable:=true` is accepted
and silently does nothing.

**2. Robot container** (needed for `/current_pose`):
```bash
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot
```

**3. Calibration node** (terminal 2 — read its log through the whole session):
```bash
ros2 run ycb_pointcloud_pipeline calibration_check_node
```

**4. Settings + preflight** (terminal 4). These are RUNTIME params, lost on every restart of the
camera, the node, or the machine:
```bash
bash ~/calib_setup.sh        # temporal/spatial filters, depth_stride 2, probe_offset
bash ~/calib_preflight.sh    # EVERY line must read OK before you touch anything
```
`calib_preflight.sh` exists because the two ways to waste a whole session are invisible while you
work: `probe_offset` sitting at zero, and the node running code older than the last build.

**5. RViz** (terminal 3) — a **different config** from the planning one, laid out for clicking
stickers:
```bash
rviz2 -d ~/calibration.rviz
```
Fixed Frame `base`, scene cloud as RGB8 flat squares, RobotModel off. **RGB8 matters** — it is how
you see the stickers.

**6. Put out stickers.** Small, high-contrast, **in the middle of flat surfaces, at least 5 cm from
any edge**. 8-10 of them, spread in x, y **and z** (table, both shelf boards, a box).
- **Never click corners.** Edges give "flying pixels" whose depth interpolates to the background.
  Corners measured 33.7 mm RMS; stickers on flat faces gave ~8 mm.
- Height spread is not optional. All-coplanar points fit beautifully and are wrong off-plane.
- Spread x too: a narrow x range triples the noise amplification.

**7. For each sticker** — arm parked out of the camera's view, wait ~2 s for the cloud to settle:
- In RViz click **Publish Point** on the sticker centre. Zoom in first; click precision is
  calibration precision. Confirm the `Clicked point in base:` line in terminal 2. Re-click freely —
  only the newest click counts.
- Close the gripper, float the arm, put the **centre of the closed fingertip face** on that same
  sticker, and hold — **press the fingertip FACE flat on the surface** so its centre is the contact point — that is
what `probe_offset` models. On a horizontal face the hand is near vertical (tilt < ~30 deg);
on a **vertical** face (an object's front, e.g. on the bottom shelf where the hand cannot
come down from above) the hand is near horizontal, ~80 deg tilt, and that is **correct**.
The error to avoid is the face meeting the surface at an angle, where an edge contacts.

```bash
ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.0, speed: 0.03}"
```
```bash
ros2 run ycb_pointcloud_pipeline free_drive --ros-args -p open_gripper:=false
#   add -p max_pose_age:=2.0 if it trips on "pose went stale while floating"
```
```bash
ros2 service call /calibration_check_node/record_correspondence std_srvs/srv/Trigger
```
Expect `pair N: touched [...] camera [...] current calibration error NNN mm`. The click is free and
repeatable; the record is not. You have 120 s between clicking and recording.
```bash
ros2 service call /calibration_check_node/clear_correspondences std_srvs/srv/Trigger   # wipes ALL
```

**8. Solve:**
```bash
ros2 service call /calibration_check_node/solve_extrinsics std_srvs/srv/Trigger
```
Writes `~/tpgpt_calibration/extrinsics_fit_<time>.yaml`, including each pair's raw TCP position and
rotation, so pairs can be re-derived offline instead of re-touched.

**9. Apply**: paste the printed `base_to_camera:` block into
`ycb_pointcloud_pipeline/config/camera_calibration.yaml`, then
```bash
cd ~/thor_ws && colcon build --packages-select ycb_pointcloud_pipeline
```
and restart the calibration node. Nodes read the **installed** copy, never the source tree.

**10. Verify — do not skip, and do not judge by fit RMS:**
```bash
python3 ~/calib_tools/check_calibration.py                     # surfaces vs touched heights
ros2 run rqt_image_view rqt_image_view /tpgpt/verify/overlay   # move the arm around
```
- `table plane:` line at INFO, tilt < 1.5 deg, error within a few mm.
- The overlay cross sits on the closed fingertip tip **at every pose across the workspace**, not
  just one. This is the only check of lateral accuracy.
- **Fit RMS is a poor guide.** A fit reading -0.5 mm on the table was 9 mm out elsewhere. Pruning
  outlier pairs lowered RMS while making the surfaces worse. Judge by surfaces at several heights.
- If errors vary with position, refit against the point pairs **and** horizontal-plane constraints
  at the touched heights — that is how the current calibration was made.

### Measuring anything by touching

`/current_pose` reports `fr3_hand_tcp`, which sits **between** the closed jaws and cannot be placed
on anything. The touchable point is the fingertip face, 9.5 mm further along TCP +Z.
```bash
python3 ~/calib_tools/touch_point.py     # prints the corrected probe point + hand tilt, live
```
The tilt readout is a guide, not a limit: **press the fingertip FACE flat on the surface** so its centre is the contact point — that is
what `probe_offset` models. On a horizontal face the hand is near vertical (tilt < ~30 deg);
on a **vertical** face (an object's front, e.g. on the bottom shelf where the hand cannot
come down from above) the hand is near horizontal, ~80 deg tilt, and that is **correct**.
The error to avoid is the face meeting the surface at an angle, where an edge contacts.

Reading a point off the cloud instead (no robot, ~5 mm, fits the local plane around your click):
```bash
python3 ~/calib_tools/click_measure.py   # then Publish Point in RViz
```
Accept only patches with `local tilt` under ~3 deg — more means you caught an edge, not a face.

Comparing what the robot touches with what the camera sees, in the shelf's own axes (this is how
the camera was found ~10 mm off at the shelf):
```bash
source ~/venvs/tpgpt/bin/activate
python3 ~/calib_tools/shelf_points.py    # t <label> records a touch, c <label> keeps an RViz click
```

---

## 10. The shelf

Model and collision geometry: `ycb_pointcloud_pipeline/config/shelf.yaml`, explained in
`docs/hardware/SHELF_AND_PATH_CHECK.md`. Tape dimensions: `docs/hardware/shelf_measurements.html`.

The shelf's **dimensions** come from a tape and are trusted. Its **placement** comes from five
numbers the robot touched (`anchors:` — the two slot marks, the two board top surfaces, the table).
`shelf_fit` then corrects x/y and yaw against the cloud; it **never** changes a height.

### After the shelf is moved

**1. Recalibrate the camera first if the table check is off.** Fitting the shelf against a bad
calibration bakes the camera error into the shelf model permanently.

**2. Re-touch the five anchors** with `touch_point.py` (arm floating, closed gripper):
the two slot marks on the top board, both board top surfaces, and the table. Put them under
`shelf: anchors:` in `config/shelf.yaml` and rebuild.

**3. Ground the placement in the cloud** — shelf **empty**, arm out of the camera's view,
segmentation running, in the tpgpt venv:
```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline shelf_fit
#   robot off (cannot remove the arm from the cloud, so keep it well out of view):
ros2 run --prefix python3 ycb_pointcloud_pipeline shelf_fit --ros-args -p allow_missing_joint_states:=true
```
It reads `/ycb_scene_cloud` (from `segmentation_node`), merges 8 frames, prints a `cloud_fit:` block
for `config/shelf.yaml` and writes `~/tpgpt_calibration/shelf_fit_<time>.yaml`. It never edits the
config. In RViz: `/shelf_fit/model_tape` (orange), `/shelf_fit/model_fitted` (green),
`/shelf_fit/cloud`.

**If it REFUSES**, one rigid move could not put every face on its points. That means the tape, the
touched anchors and the cloud genuinely disagree — usually the camera being off at the shelf, or the
shelf having changed shape (an adjustable board moved). Find out which before storing anything; a
refusal is information, not a failure to work around.

**4. Check it.** The planner logs `Shelf world: {...}` — `explained_by_model` should be close to 1
for an empty, correctly modelled shelf. In RViz: "Shelf model" and "Shelf cloud (obstacles)".

### Destinations

The four slots resolve to the touched slot marks on each reachable board:
```bash
-p destination_name:=top_left     # or top_right, bottom_left, bottom_right
```
