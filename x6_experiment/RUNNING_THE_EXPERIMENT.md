# Running X6, step by step

The brief is `HARDWARE_EXPERIMENT.md`, the config is `config/x6_experiment.yaml`, and
data locations are in `experiments.md`.

**One trial = one run of `tpgpt_pick_place` with `experiment:=true`.** You start
each trial yourself, give it the trial id, hand, object and destination, and the
node exits when the trial is finished. To go to the next trial, run the command
again with the next trial id.

Everything below is built except the **robustness re-runs** (Phase 5, step 2). The
**Inspire hand** is integrated in code (2026-09-15) but has not run a trial yet; its
mount, grasp and caveats are in `ycb_pointcloud_pipeline/INSPIRE_HAND.md`.

## At a glance

| item | count |
|---|---|
| Demonstrations | **1**: banana, Franka Hand, into the shelf, recorded once, never again |
| Hands | **2**: `panda` first (all 30 trials), then `inspire` (side grasp, no GraspGen-X) |
| Test objects | **5**: apple, cup, mug, cereal_box, can |
| Pick poses | **3** taped marks, different yaw each, same marks for every object and both hands. **Not typed in**: recorded from the point cloud (centroid + yaw) |
| Destinations | **2**: `slot_a`, `slot_b` (hard-coded in the config) |
| Trials per hand | 5 × 3 × 2 = **30**. Panda `0001`–`0030`, Inspire `0031`–`0060`. **60 total** |
| Instrument checks | **3 × 10** captures: before trial 1, after re-mounting, at the end |
| Robustness check | **6** (3 per hand), after both grids (not built) |

**Rules:**
- One demonstration only.
- Never edit `method:` in the config.
- Running an existing trial id **overwrites** that trial: its folder is deleted and its `trial_index.csv` row replaced.
- Refusing at `yes`, or stopping the node mid-trial, is an **aborted trial**. Its data stays and it counts in the denominator.
- Finish a hand's 30 trials before re-mounting.

---

## Build (after any code change, and before the first session)

The nodes run from the **installed** copy in `~/thor_ws/install`, not from `src/`.
Until you build, `experiment:=true`, the `x6_*` commands and the segmentation mask
topics do not exist.

```bash
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh
cd ~/thor_ws && colcon build --packages-select ycb_pointcloud_pipeline
source ~/thor_ws/install/setup.bash
```

Then **restart `segmentation_node`** (T2 below) so it publishes `/ycb_masks/<object>`
and `/ycb_detections/info`.

The build also re-installs `config/camera_calibration.yaml`; every trial records
that installed file's sha256 in `record.json`.

## Terminals (every session)

```bash
# T1 robot, FCI active
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot

# T2 camera + segmentation with the X6 prompts
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
source ~/venvs/ycb/bin/activate
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py \
    targets:=$HOME/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/x6_experiment/config/x6_target_objects.yaml

# T3 GraspGen-X server
~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh --daemon

# T4 trials and the x6_* tools (TPGPT venv)
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
source ~/venvs/tpgpt/bin/activate

# T5 tripod video camera (manual)
```

T1, T2 and T3 stay up for the whole session. You only re-run commands in T4.

## Calibration and sessions

Calibration is verified **manually**, outside this stack. Trials do not check it and
need no session: with `experiment:=true` and `x6_experiment.yaml` filled in, a trial
records its data. Sessions are optional: `ros2 run ycb_pointcloud_pipeline x6_session
start` only groups the following trials (`session_id`, `order_in_session` in
`record.json`), and `x6_session end` closes it.

---

## Phase 0: one-time setup

1. Fix the shelf, RealSense and tripod. Take **photo 4**.
2. Verify the calibration manually: TCP overlay, table plane, at least 4 touched points including one on each slot.
3. Touch each slot board centre with the fingertip, read `/current_pose`, and fill `destinations.slot_a` and `slot_b` plus `table_height`.
4. Tape the 3 pick marks. Optionally touch them and fill `pick_poses` (used only to label trials in the analysis).
5. Put the can in each slot and confirm it's segmented (≫ 40 points). If a slot can't be seen, choose another slot.
6. Check that each object's prompt detects the right item. Fix the prompts, then freeze them.
7. Fill in the home pose (arm out of camera view), the caliper values in `hands/panda.yaml`, and take **photo 5**.
8. Optional: fill `method.outcome.retreat_tcp_pose` with a TCP pose that clears the camera's view of the slots. This is a `method:` field, so decide **before trial 1**. Left `null`, you clear the view yourself before pressing Ctrl-C.
9. **Robot-only checks** (not testable off the robot), before the first real trial:
   - `ros2 action list | grep -E 'grasp|move'` shows the gripper action servers named in `hands/panda.yaml` (`gripper.namespace`), and the width topic (`gripper.width_topic`) is publishing.
   - `ros2 topic list | grep robot_state` matches `experiment.topics.robot_state` (`/franka_robot_state_broadcaster/robot_state`). If the name differs, the trace has no wrench and the Desk end-effector mass isn't checked. Fix the topic name before trial 1.
   - `ros2 topic hz /ycb_masks/apple` and `ros2 topic echo /ycb_detections/info --once` show the rebuilt segmentation node publishing.

## Phase 1: record the demonstration (once)

```bash
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=banana     # T4
ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard                            # T6
```
1. Keys in order: `c`, `t`, `s`, grasp, `g`, carry onto the shelf (not slot_a or slot_b), `r`, `x`. Take **photo 1** mid-carry.
2. Run `sha256sum ~/tpgpt_demos/banana_<time>/source_bundle/*` and fill the `demonstration:` block in the config (`recorded_dir`, `source_bundle_dir`, the three hashes, `placed_at_note`).

## Phase 1b: order sheet (once per hand)

```bash
ros2 run ycb_pointcloud_pipeline x6_order_sheet --hand panda
```
It writes `<root>/orders/order_panda.csv` (trial id → object, mark, slot, seed 6001)
and prints the list. It refuses if the sheet already exists: the order is generated
once and never regenerated. Print it and tick off trials as you go. Run
`--hand inspire` (seed 6002) before Phase 4.

## Phase 1c: scratch trials (optional, before trial 1)

To try the full trial sequence without touching the campaign data, give a trial id
starting with `s` (e.g. `s001`). It is written under `<root>/scratch/` and never
analysed. Add `-p dry_run:=true` to also skip the robot (simulated execution; the
pose, hand-mount and extrinsics checks are skipped).

## Phase 2: instrument check (before trial 1)

Put the can in `slot_a` and don't touch it. Arm out of the camera view. Session open.
```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline x6_instrument_check --ros-args -p when:=before_first_trial
```
This takes 10 measurements. Each one is the same procedure as a trial's settled
capture: 3 fresh segmentations, fused, centroid in the base frame. It prints each
centroid, then the SD per axis and lateral, and writes
`<root>/instrument/before_first_trial/instrument.json`. Running it again with the same
`when` writes `before_first_trial_repeat1` rather than overwriting. Remove the can
afterwards.

## Phase 3: one trial

Example: trial 0001, Franka Hand, apple on a mark, going to slot_a.

1. Place the apple on a mark. Arm at home. Video running.
2. In T4:
   ```bash
   ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
     -p experiment:=true -p trial_id:=0001 -p hand:=panda \
     -p object_name:=apple -p destination_name:=slot_a
   ```
   Only `experiment`, `trial_id`, `hand`, `object_name`, `destination_name`, `dry_run`,
   `top_down_grasp` (true = scripted top-down grasp, GraspGen-X not called) and `x6_config`
   are accepted. Any other `-p` is refused, because everything else
   comes from the frozen config.
3. The node runs these in order:
   - **Setup** (no pre-flight checks): loads the hand file, captures fresh segmentations of the apple and prints the fused point count.
   - **Capture and plan**: one plan, no re-plan. It saves the perception frames, `cloud.npy`, `candidates.json`, `keypoints.npz`, `plan.npy`, `plan_overlay.png` and the pick pose (from the cloud).
   - It shows the plan. **You type `yes`.** Anything else = `refused_at_confirmation` (counted).
   - **Execution**, recording `trace.npz` every control step.
   - When the policy finishes, the controller **holds the arm where it is** (or moves to `retreat_tcp_pose` if you set one) and prints a banner: *press Ctrl-C to finish the trial*.
4. Clear the camera's view of the slot if needed, let the object come to rest, then **press Ctrl-C**.
5. **Press Enter** to capture the settled scene (or type `s` to skip). It prints the number of settled points and the placement error (lateral, plus x/y/z in mm, against the plan's predicted placed cloud).
6. Answer the questions:
   - Placed in the destination slot? `y`/`n`
   - If `n`: the first stage that failed, 1–7 (approach, reach, grasp, lift, carry, place, settle)
   - Abort reason (only for a stopped or aborted trial; Enter keeps the system's reason)
   - Object disturbed before the grasp, mm (estimate from the video; Enter = unknown)
   - Exclude? `0` = no, or the reason number. Only for a cause outside the method: network drop, E-stop for an external cause, camera driver crash, robot container crash, operator setup error
   - Notes
7. It writes `record.json`, appends a row to `<root>/trial_index.csv` (and to the session log if a session is open), and exits.
8. Take the apple out of the shelf.

## Next trial: same object, another pose

Move the apple to another mark (or keep the mark and change the slot). Run the same command with the next trial id:
```bash
... -p trial_id:=0002 -p hand:=panda -p object_name:=apple -p destination_name:=slot_b
```
You don't restart T1, T2 or T3.

## Next trial: another object

Put the new object on a mark and change `object_name`:
```bash
... -p trial_id:=0003 -p hand:=panda -p object_name:=cup -p destination_name:=slot_a
```

**To randomise the order:** follow `orders/order_panda.csv` (Phase 1b) rather than going object by object.

**If something goes wrong mid-trial:** press Ctrl-C (or the E-stop). The arm holds,
and the node goes straight to step 5 (settled capture, questions, record) with
whatever was recorded. The trial is `stopped_by_operator`, or
`aborted_during_motion` if the executor aborted it. Go to the next trial id; don't
reuse this one. A Ctrl-C during the questions writes what was answered and marks
`questions_incomplete`.

## Phase 4: re-mount and Inspire trials

1. Mount the Inspire hand as `ycb_pointcloud_pipeline/INSPIRE_HAND.md` describes, measure
   the `MEASURE` lengths into `ycb_pointcloud_pipeline/config/inspire_hand.yaml` and the
   `TO_FILL` fields of `hands/inspire.yaml`, rebuild. Update Desk (end effector: the
   Inspire hand's mass and centre of mass). Take **photo 5**.
2. Start the hand driver in its own terminal (T6):
   `ros2 run ycb_pointcloud_pipeline inspire_hand_node`. The T3 GraspGen-X server is not
   needed for this hand.
3. Re-verify the calibration manually if the camera was touched.
4. Run the instrument check again with `when:=after_remount`.
5. `ros2 run ycb_pointcloud_pipeline x6_order_sheet --hand inspire`.
6. Run trials `0031`–`0060` with `-p hand:=inspire`: same marks, same slots, same command
   otherwise. The hand file switches the planner to the side grasp and the executor to
   `gripper_mode: inspire`. **Only upright round objects** suit that grasp (cup, mug,
   can, bottle); a cereal box or an apple on its side is outside what it was built for,
   which is a finding to record, not a setting to change.
7. The one demonstration is a top-down Franka Hand grasp. Carried onto a side grasp the
   map is poorly conditioned (min det(J) 0.007-0.40 on three synthetic scenes), so expect
   `map_not_a_diffeomorphism` refusals. Record them as planning failures.

## Phase 5: end

1. Instrument check with `when:=campaign_end`.
2. Robustness check, 6 trials **[not built]**: re-runs with the 2nd and 3rd admissible grasp. Optional, and only after both grids.
3. **Photo 3**: the same object and the same grasp with both hands.
4. Analysis (no ROS needed):
   ```bash
   ros2 run ycb_pointcloud_pipeline x6_analyse                 # optional: --sim-csv sim.csv
   ```
   It writes `<root>/summary.csv`, `<root>/analysis/table5.csv`, `table5.tex` and
   `analysis.md` (all §5 statistics, instrument repeatability, "What went wrong").
   It can be re-run at any time. Scratch trials are never included.
5. If a session is open: `ros2 run ycb_pointcloud_pipeline x6_session end`.

## `experiment:=false`

Exactly today's behaviour: nothing extra is recorded, the config is not read, and the re-plan prompt is still there. Use it for testing.
