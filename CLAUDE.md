# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

The real-robot (Franka FR3) deployment of **TPGPT**, a transportation-map method that executes one
demonstration on new objects, hands and destinations. The method was built and validated in MuJoCo
(branch `main`); **this is branch `hardware`**, which carries the method plus everything needed to
run it on the robot.

| path | what it is | owned by |
|---|---|---|
| `tpgpt/` (except `tpgpt/hardware/`), `tests/`, `README.md` | **The method**, as the simulation built it. | `main` (sim) |
| `tpgpt/hardware/` | The robot bridge: planner, grasp selection, executor, teach, end effectors. No ROS. | `hardware` |
| `ycb_pointcloud_pipeline/` | **The ROS 2 package** (ament_python): camera segmentation, calibration, and the nodes that call `tpgpt.hardware`. | `hardware` |
| `x6_experiment/` | Config and run guide for the paper's real-robot experiment ("X6"). | `hardware` |
| `docs/hardware/` | Hardware docs: calibration, commands, the X6 brief, the faithfulness audit. | `hardware` |
| `docs/sim/` | The simulation's documents, moved here on this branch: its `CLAUDE.md` as `SIM_CLAUDE.md` (**kept for comparison only**, see below), `ROBOTICS_NOTES.md`, `dynamics_execution.md`, `gripper_configs.md`, `gripper_diversity.md`, `realworld_failures.md`, `ycb_grid.md`, `figures/`. | `main` |

Beside this repo, in `~/thor_ws/src/ycb_point_cloud/`, sits `6DOF_MultiEmbod_Grasp/`: another
project, **used only to run the GraspGen-X server** (`ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh`)
and for its gripper descriptions. Its own repo; never edit it. The parent folder
`~/thor_ws/src/ycb_point_cloud/` is a retired repo (`Aerokrishna/ycb_point_cloud`) whose only
remaining job is a pointer `CLAUDE.md` importing this file; its branch `pre-move-baseline` holds the
pre-2026-09-21 layout.

The robot itself runs in a separate Docker container (`docker-thor-robot`, from
`~/THOR_DOCKER_SETUP`), exposing `ros2_control`, the CRISP Cartesian impedance controllers and
`/current_pose`. The RealSense D435 runs on the **host**, not in the container.

## The simulation's CLAUDE.md, and pulling sim updates

`docs/sim/SIM_CLAUDE.md` is the simulation's guide (1000+ lines: its rules, its pitfalls, its
measurement history). **It does not govern hardware sessions** -- this file does -- but it is the
reference for what the method *is*, so read the relevant part before changing anything that the
simulation also implements. Comments in sim-owned code that say "CLAUDE.md" mean that file, and
ones naming `ROBOTICS_NOTES.md` or `docs/<name>.md` mean `docs/sim/...` (the sim's own code is left
unedited so merges stay clean; git follows the renames when `main` edits those files).

Rules carried over from it that apply here too: run the tests after every change; cite the paper
section in docstrings of anything implementing theory; **document every deviation from the paper or
the simulation, with the measurement that motivated it** (`docs/hardware/FAITHFULNESS_AUDIT.md`);
no wall-clock time in recorded data (step index / control rate); prefer extending an existing module
over a parallel one; write for a reader who was not there -- define terms, show arithmetic, say what
would make a conclusion false.

To bring in the simulation's latest work: `scripts/merge_sim.sh` (on `hardware`, clean tree). It
merges `origin/main`, keeps this `CLAUDE.md` (`.gitattributes`: `CLAUDE.md merge=ours`) and refreshes
`docs/sim/SIM_CLAUDE.md` from upstream. Any other conflict is yours to resolve. **Keep the portability
patch** in `tpgpt/grasp/grippers.py` and `tpgpt/grasp/server.py`: upstream hardcodes
`/home/ishita/...`; the patch falls back to the sibling `6DOF_MultiEmbod_Grasp`.

**Commit before every change** (the user's rule), so any stage can be reverted. There is no global
git identity on the Thor; commit with
`git -c user.name=Ishita -c user.email=ishitachaudhary.tib@gmail.com commit ...`. Never push unless asked.

## Environment

```bash
source /opt/ros/jazzy/setup.bash
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
```
`host_ros_env.sh` sets `ROS_DOMAIN_ID=61` — without it no robot topics are visible.

Three venvs, not interchangeable: `~/venvs/ycb` (segmentation node: torch, Grounding DINO + SAM),
`~/venvs/tpgpt` (planner, executor, teach, pick-place, and all tests), `~/venvs/graspgenx`
(GraspGen-X server). Nodes that import `tpgpt` are launched with `ros2 run --prefix python3` so they
pick up the active venv instead of colcon's shebang.

**Rebuild after every edit, including config:**
```bash
cd ~/thor_ws && colcon build --packages-select ycb_pointcloud_pipeline
```
The install is a plain copy, **not** `--symlink-install` (mixing the two fails with "File exists").
Nodes read the **installed** `share/.../config/*.yaml`, never the source tree — an unrebuilt config
edit is silently ignored. A running node keeps the code it started with, so restart it after a build.

Robot and run commands are in `docs/hardware/commands.md`.

## Tests

```bash
cd ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment
# ROS package (no robot, no ROS graph; ROS sourced: the executor's tests import rclpy)
(source /opt/ros/jazzy/setup.bash && cd ycb_pointcloud_pipeline && ~/venvs/tpgpt/bin/python -m pytest test/ -q)
# The method. test_gripper_actuation.py errors at COLLECTION (robosuite lacks UMIGripper),
# which aborts the whole run unless ignored. testpaths = tests, so the ROS tests are not collected.
~/venvs/tpgpt/bin/python -m pytest tests/ -q --ignore=tests/integration/test_gripper_actuation.py
# Just the hardware bridge (fast, ~10 s, and what most changes touch)
~/venvs/tpgpt/bin/python -m pytest tests/unit/test_hardware.py tests/unit/test_end_effector.py \
    tests/unit/test_grasp_selection.py tests/integration/test_hardware_grasp_selection.py \
    tests/unit/test_fr3.py tests/unit/test_shelf.py tests/unit/test_path_check.py \
    tests/unit/test_franka_regression.py tests/unit/test_inspire_model.py \
    tests/unit/test_inspire_mount.py tests/unit/test_grasp_test.py -q
```
`test_hardware_grasp_selection.py` needs a running GraspGen-X server and skips without one.
`test_fr3.py` checks FK against 40 at-rest samples of the real arm (`tests/data/`).
`test_franka_regression.py` pins the Franka Hand's grasp screening as it was before the Inspire
hand changed shared code (2026-10-04): if it fails, a Franka-path change was not intended.

**Offline ROS dry runs** go on another domain so nothing reaches the robot (`ROS_DOMAIN_ID=88`,
without `host_ros_env.sh`), with fake `/ycb_pointclouds/<obj>`, `/ycb_scene_cloud` and
`/joint_states` publishers. Base-frame points must be sent in the optical frame through the inverse
of the installed calibration. If you start the GraspGen-X server for a test, stop it afterwards.

Known failures, not regressions: TPGPT × 7, all gripper-registry tests, because the local GraspGen-X checkout lacks
4 descriptions (`jaco_3f`, `schunk_svh`, `ability_hand`, `fourier_hand`) and `robosuite_models` is
not installed, plus `test_graspgen.py::TestServer::test_reports_its_loaded_grippers` whenever the
GraspGen-X server is not running (8 in all, 904 passing, 2026-09-24). The two hands on the robot (`franka_panda`, `inspire_hand`) are unaffected.

## Architecture

```
D435 (host) -> segmentation_node -> /ycb_pointclouds/<object>, /ycb_scene_cloud  (optical frame)
                                          |
     camera_calibration.yaml (base <- optical) applied by the consumer
                                          |
   tpgpt_planner_node -> tpgpt.hardware.{planner, grasp_selection, path_check}  (+ GraspGen-X, ZMQ)
        + config/shelf.yaml -> tpgpt.hardware.shelf (model + scene cloud on the shelf)
        + /joint_states -> tpgpt.hardware.fr3 (IK seed; robot removed from the cloud)
                                          |  plan dir: ~/tpgpt_plans/<object>_<time>/
   tpgpt_executor_node -> tpgpt.hardware.executor (+ /joint_states) -> CRISP in the container
```
`tpgpt_pick_place` runs planner then executor in one process; with `-p experiment:=true` it runs one
X6 trial and writes the full trial record.

Two facts that need several files to see, and that decide whether a change is faithful to the method:

- **The execution gates are shared code, deliberately.** `tpgpt/hardware/executor.py` imports
  `gate_step`, `advance_step`, `StallWatchdog`, `ExecutionLaw` straight from
  `tpgpt/policy/rollout.py` — the exact code the simulation runs. Never reimplement them. The only
  deviations live in `tpgpt_executor_node.py`'s parameter defaults: `lag_tolerance` 0.045 (method
  0.035), `rotational_stiffness` 30 (60), `speed_scale` 0.35, `use_transported_stiffness` False,
  and `on_stall: continue`, which **switches the lag gate off** for the rest of a run after a
  stall (the clamp's speed limit stays: with no gate its radius is `speed_limit_factor` x the
  fastest label speed x compliance). **A learned plan uses neither the gate nor the clamp**
  (2026-10-01, audit B1 step 8; `follow_arm_path`): it is followed against the path the arm
  traced -- the clock slows when the arm is behind it, a slow capped correction brings it back
  when it is beside it (capped at 50 mm since 2026-10-03), the run stops and holds only when
  the arm does not move while pushed as hard as allowed, and the holds at the grasp and the
  release bring the hand onto the plan's marks (the release pushes too since 2026-10-03); in
  contact (the replay's contact windows) the correction is frozen. **The run starts only once the
  hand is within 4 mm of the start of its path** (`start_on_path`, 2026-10-03): the joints are
  moved onto the posture the planner checked there with the joint controller, because a pull on
  the hand cannot bring a nearly straight arm the last way out (the 23:28 mug start was 1.8 mm
  inside full stretch and the arm stopped 54 mm short of its command). **Since 2026-10-06 the
  whole move onto the path is made in joints** (the user; `approach_in_joints`): the joint
  controller drives the arm through the postures the planner checked from its joints to the
  start (`Plan.approach_postures`, checked at and between them), when the arm is still within
  `plan_joint_tolerance` of where the plan was made from; otherwise the hand approach as before.
  Commanding only the hand, the arm could settle in other postures than the planner's at a
  straight shoulder (below). Full account, with the equations: `docs/hardware/HOW_EXECUTION_WORKS.md`.
  What the correction took *along* the path while the clock was held is **given back once the arm
  runs ahead** (`path_unwinding`, 2026-10-02): kept, it held the mug 24-31 mm ahead after a
  stick-slip and set it down ~10 labels early. On the robot (mug, top_left, 2026-10-01) it was needed: without it the hand missed the grasp
  by 15 mm and the release by 16, with it by 4 and 2. Both guards judge normal lag by the spring model (2-6 mm), and
  the learned command must lead this arm by 17-35 mm: they would slow it to 0.67 and cut it on 262
  of 488 labels. The policy query, the speed cap and the clock's update are still the shared code.
  **A grasp test is followed too** (2026-10-06, audit D12): its labels are its own arm path (no
  lead, no margin), its grasp and set-down the marks. Under the clamp the first one never left
  the pre-grasp: the command sat 6 mm below the arm (2.4 N at 400 N/m), and static friction held
  it for 101 s. Grasp-test plans made before then have no arm path: re-plan. **And at the
  transport's settings** (the user, 2026-10-06: as the Franka runs were executed): a taught
  demonstration's stiffness (350 along / 600 across and still) and the source bundle's
  `learned.execution` (0.35, rotational 30 x1.0, transported 3x3, no offset compensation). At
  400 N/m all four 18:07/18:27 holds hit their 30 mm cap (12 N) and ran out 5.4-20.7 mm off;
  the Franka's 56 holds at 600 needed at most 22.6 mm. The settings come from the planner's
  source bundle, whose default is now the learned bundle (it was the sim's 09-13 export, which
  the 18:57 test took, so it still ran at 400); a grasp test refuses a bundle without them.
- **The elbow is held on the posture the path check verified** (2026-10-02, audit C8;
  executor `posture_target`). The check records the posture it posed the arm in at each tested
  label (`checked_postures.npz` in the plan) and the executor sends it as CRISP's `target_joint`
  with every pose (within 0.5 rad of the arm, 0.1 rad inside the limits). Without it CRISP's
  null-space spring (2 N m/rad, max 5) pulls toward the posture at activation, home -- the
  sim's rule too -- and on the 00:47/00:50 mug runs it swung joints 1 and 3 50-65 deg as joint
  2 crossed 0 (the hand still), the arm stuck, and the elbow ended 108-110 deg off the checked
  posture. Plans made before 2026-10-02 have no checked postures: re-plan.
- **The planner keeps the wrist 20 mm inside full stretch along the whole path** (2026-10-03;
  planner `grasp_reach_margin`, `path_check.REACH_MARGIN`). Near full stretch a pull on the hand
  barely turns the joints, so the arm cannot be pulled there (the 23:28 mug start was 1.8 mm
  inside it and never reached). Cost on the fifteen captures: 315 -> 246 grasps, no cell emptied,
  bottom_left unchanged (`GRASP_LIBRARY.md` item 11). **The open hand keeps 10 mm from the placed
  object in the retreat** once it has got that far (`grasp_retreat_clearance`,
  `path_check.RETREAT_CLEARANCE`; on since 2026-10-03, once bottom_left was no longer run -- it
  empties bottom_left cells): the taught retreat swung the fingers back to 0.5-8 mm and dragged a
  set-down mug. With the reach margin, the top slots keep 105-149 of 187 (bounded), none empty.
- **Grasp selection is the simulation's rule, and sees the shelf** (since 2026-09-21,
  `docs/hardware/SHELF_AND_PATH_CHECK.md`). `grasp_selection.select_and_plan` walks survivors in
  score order and takes the first whose **whole transported path** passes
  `path_check.path_feasibility`: `filters.path_feasibility_observed` on the FR3 model
  (`tpgpt/hardware/fr3.py`, the live URDF, a hull per link, `solve_ik` transcribed) and the shelf
  world (`tpgpt/hardware/shelf.py`). The arm can hit the **table** (an exact plane), the **shelf**
  (the measured model, seen or not), **everything else the camera sees** within reach (the robot,
  the target, and the camera's points on the table and shelf removed), and **itself** (exact convex
  tests between pieces not joined at a joint). Contacts are allowed **by stage** (pick: fingers on
  the object; carry: the object on its pick support only until it first clears it; place: only the
  face that is lowest in the planned release pose, on the destination surface; retreat: fingers on the placed object only until they
  first clear it). Reachability is **strict** by default: every waypoint reachable in position, the
  pick and place with their orientation, and IK retried first. Full account: `SHELF_AND_PATH_CHECK.md`,
  "How a path is checked". The funnel calls the sim's own `tpgpt.grasp.filters` stage functions
  (visibility, on_target, grasp_collision, place_zone, centred), with the sim's pass-through rule.
  The hardware's jaw, refine, table and duplicate checks stay as hard stages. Deliberate
  differences, all documented there: a collision is never executed as a fallback, the top-down
  grasp is opt-in, and the A4 gates are kept. Importing `tpgpt.grasp.*` loads robosuite (~0.5 s);
  `fr3`, `shelf`, `path_check` and `executor` do not.
- **The object cloud is the object's YCB scan, fitted to the camera cloud** (`tpgpt/hardware/object_model.py`,
  planner `object_model`, default true; audit A8). This is what the sim's YCB experiment does (full scan,
  observed scene). Scans for the banana, apple, mug and tomato soup can are downloaded into
  `assets/ycb/` (gitignored: re-download `<id>_google_16k.tgz` from the YCB site on a new machine).
  A fit the camera doesn't support is refused and the planner falls back to the camera cloud. The plan
  says which (`object_cloud_source`). A refused plan writes `<object>_<time>_refused/` with every
  candidate's rejection.
- **The executor has the sim's infeasible-pose fallback**, on FR3 IK seeded from the measured
  joints (`HardwareExecutor.step(..., joint_positions)`).

The planner's support height is: its `support_height` parameter, else the calibration's
`expected_table_height`, else estimated from the object cloud (biased a few mm high).

## Things that will bite you

- **`/current_pose` is `fr3_hand_tcp`, which sits between the closed jaws** and cannot touch
  anything. The touchable point of the closed Franka Hand is its fingertip face, **9.5 mm further
  along TCP +Z**. Any measurement by touching must add this. See `docs/hardware/CALIBRATION.md`.
- **Runtime parameters reset on every restart** of the camera driver, the calibration node, or the
  machine. `bash ~/calib_setup.sh` reapplies them; `bash ~/calib_preflight.sh` verifies them.
- **On this ARM build the RealSense pointcloud filter is `pointcloud__neon_`**, not `pointcloud`:
  `pointcloud.enable:=true` is accepted and silently does nothing.
- **rclpy caches a logging context per call site** and raises `ValueError: Logger severity cannot be
  changed between calls` if one line logs at two severities. Never write
  `(log.info if ok else log.warn)(...)`; use two separate calls.
- **Checking whether a node is running:** `pgrep -f <name>` also matches your own shell when the
  command line contains that name. Use `ros2 node list` as the authority.
- **rclpy's `MultiThreadedExecutor` starves high-rate subscriptions on this machine**
  (measured 2026-10-05, passively on the live stream, alternating runs): `/current_pose` (250 Hz)
  arrived at 45-51 Hz with 7-11 silences over 0.5 s (up to 1.3 s) per 30 s under
  `MultiThreadedExecutor(num_threads=2)`, at 234-241 Hz never more than 5 ms apart under
  `SingleThreadedExecutor`. That was **`free_drive`'s "pose went stale while floating"**, so it
  now spins single-threaded (its waits all poll futures from the main thread). Still on the
  multi-threaded executor, with high-rate subscriptions and staleness checks of their own:
  `joint_target_publisher` (JointState; used by `joint_homing`), `calibration_check_node`
  (`/current_pose`, images), the planner inside `tpgpt_pick_place`/`x6_trial` (JointState), and
  `segmentation_node`. The executor node and `inspire_touch` were already single-threaded.
  Prefer `SingleThreadedExecutor` in new nodes.
- **The setup was moved on 2026-09-26** (by someone else): the camera was knocked ~46 mm / 2.9 deg
  and recalibrated (9 face-flat correspondences plus the table and top-board planes; on a fresh
  frame table +3.8 mm, top board +2.1 mm), and the shelf re-touched. The shelf came back within
  4 mm, the table within 1.2 mm (-0.0132), but the **top board is back in its original notch,
  0.3211 (gap 244 mm, not 256)**, and the bottom board rests at 0.0731. Across the work area the
  table now reads within +-10 mm of the touched height (a slight roll: -6..-9 mm on the shelf
  side, +6..+10 mm at the far left), inside the 25 mm table band. Demonstrations are in the
  robot's frame and stay valid; the executor does not use the camera. **Every coverage number
  measured before 2026-09-26 had the top board at 0.3310**: bottom_left now has 10 mm less
  headroom than those numbers assumed.
- **The shelf is measured (2026-09-22) and placed from the tape plus robot touches; no
  `cloud_fit`.** The shelf has no back and three boards (bottom, top, topmost). Heights and two
  slot marks were touched; everything else is from a tape (M1–M14,
  `docs/hardware/shelf_measurements.html`, private artifact
  https://claude.ai/artifact/FgicphG5cmo9hvRnoutXpp). `shelf_fit` (a fit of the model to the cloud)
  disagreed with the tape at the side panels. Touching and clicking the slot marks showed the
  **camera** is about 10 mm off sideways at the shelf, and the tape is right. The robot needs the
  true shelf, so `shelf_fit` is a check only. `SHELF_AND_PATH_CHECK.md`, "First live run".
- **The camera is about 10 mm off sideways at the shelf** (`CALIBRATION.md`, open). This applies
  to object clouds there too: expect grasps near the shelf to be off by that much until the
  calibration is refit with correspondences on the shelf.
- **Two selection parameters differ from the sim on purpose** (audit A10): the approach cap is
  **90 deg**, not 85 (the hardware checks the hand against the table separately, and 85 threw away
  the best-scoring band: horizontal grasps around an upright can), and `check_visibility` is
  **`auto`** -- on only when the object's scan was not fitted, since a fitted scan has no unseen
  side, and this camera faces the robot so the filter otherwise removes the robot-side approaches.
- **The executor's workspace box is OFF by default** (`workspace_check`, user 2026-09-23). It was a
  blunt guard from before the path check existed and it refused poses the arm reaches perfectly
  well (a transported demonstration start near the base, a retreat at the box ceiling). What it
  covered is now covered by `path_check`: every waypoint and the move onto the path, for
  reachability, self-collision, the table, the shelf model and the seen points. `workspace_check:=true`
  brings it back, and the planner's `executor_workspace_check` mirrors it at selection.
- **A blocked move onto the path asks for a re-home; it does not reject the grasp** (user,
  2026-09-29; planner `approach_rejects_grasp`, default false). The move from the arm's current
  pose onto the path is still checked for every candidate and reported, but the grasp is judged on
  its path alone; if the chosen grasp's move is blocked the planner stops with `approach_blocked`
  and a `joint_homing` hint. The path's IK is still seeded from the measured joints, so a poorly
  parked arm can still cost grasps on the path itself (the arm against the right wall, 2026-09-26)
  -- **home the arm before planning.** **Except with the arm already at home** (every joint within
  `home_tolerance`, 0.1 rad, of `joint_homing`'s; the user, 2026-10-07): then a blocked move
  rejects that grasp and selection moves on, since re-homing a homed arm achieves nothing.
- **Selection stays in GraspGen-X's score order** (user, 2026-09-29): near-upright grasps are not
  moved ahead, even though steep ones fold more often and cost planning time.
- **Other executor guards still apply**, and they are not the box: `max_start_distance` (0.6 m
  from the plan's first label, **only when the arm has moved since planning**: the planner checks
  the move onto the path from the joints it planned from, so with every joint within
  `plan_joint_tolerance` (0.05 rad) of them any distance is fine; 2026-10-03, a homed arm was
  refused at 0.602 m), `max_tracking_error`, the stall watchdog and `max_policy_seconds`.
- **The robot's CRISP takes the full 3×3 translational stiffness** (2026-09-29, audit B2 step 1):
  `~/THOR_DOCKER_SETUP`, `patches/crisp-full-stiffness.patch`, in image `docker-thor:local`; the
  previous image is kept as `docker-thor:pre-full-stiffness`
  (`docker tag docker-thor:pre-full-stiffness docker-thor:local` rolls back). The executor sends it
  as a 12-value `target_stiffness` whenever `use_transported_stiffness` is on
  (`stiffness_message: auto`); a fixed stiffness still goes as 6 values, exact because it is
  isotropic. **The pre-patch image silently ignores a 12-value message** and keeps its yaml gains:
  after a rollback pass `stiffness_message:=diagonal`. Why it exists: a stiffness soft along the
  motion and stiff across it has off-diagonal terms in the hand frame, and the diagonal dropped
  about a quarter of it -- almost all of it at the set-down. Dry runs publish what would be sent on
  `/tpgpt/executor/commanded_stiffness`. Not yet run on the robot.
- **Taught demonstrations get a directional stiffness by default** (2026-09-29, audit B2 step 2;
  `teach.directional_stiffness`): 350 N/m along the motion, 600 across it and in every direction
  when still, smoothed over ±5 labels (without it the policy's fit of it overshot to 236-980 N/m).
  **Run such a bundle with `use_transported_stiffness:=true`**, or its stiffness never reaches
  the arm; `--stiffness-profile static` reproduces the old constant 400/40 bundles bit for bit.
  The 18:39 demonstration's directional bundle is `~/tpgpt_demos/banana_20260925_183904/source_bundle_directional`
  (its `source_bundle/` is unchanged). X6 was re-synced with the execution rework on 2026-10-01
  (its demonstration is now `source_bundle_learned`); it has not been run.
- **What stiffness does on this arm, measured 2026-09-29** (audit B2 step 3;
  `cartesian_target_publisher -p stiffness_mode:=isotropic|directional`, `commands.md` §8):
  **no vibration at 600, 700 or 800 N/m with rotational 30** (holds within 0.02 mm); the
  2026-09-15 vibration at 800/60 was not the translational stiffness. **The lag is viscous and
  about twice CRISP's nominal `2/sqrt(K)`** (149-221 ms at 600, 229-309 ms at 350 along, against
  82 and 107): 12-17 mm along the motion at execution speed with the directional stiffness. Friction
  is small (~0.5 N) -- the "~10 N, 20-30 mm steady lag" in `tpgpt_executor_node.py`'s comments is
  pre-move and does not hold. After a move the arm stops 3.5-6 mm short, whatever the stiffness.
  The lag gate, whose expected lag uses the nominal time constant, therefore sees ~8 mm of excess
  at full speed. One line at the home pose was measured, without a payload.
  **Rotational, 2026-09-30** (audit B2 step 3b; the check now turns the hand 15 deg): no vibration
  at 15-60 N·m/rad; resting tilt 2.5 / 1.3 / 1.0 / 0.8 deg and turning time constant 0.40-0.76 /
  0.34-0.60 / 0.30-0.44 / 0.25-0.35 s at 15 / 30 / 45 / 60. **60 is a hard cap twice over**
  (`stiffness_max_rotational` and CRISP's `variable_max_stiffness.rotational`): runs asked for 80
  and 100 ran at 60. The check now refuses above the cap.
- **A taught demonstration's attractor is learned by replaying it** (2026-09-29/30, audit B1 steps
  4-5b; `commands.md` §3b): `python -m tpgpt.hardware.replay init|use-checks|update|repeat|status`
  keeps the passes, `tpgpt_executor_node -p replay_dir:=... -p dry_run:=false` runs one (no
  policy, no gate; the demonstration's own stiffness at `speed_scale` 0.35). Position **and
  orientation** are learned (gain 1.0). Pass 0 is run twice (the spread). Before each pass the
  **taught path, widened by how far the arm strays from it**, is checked against the shelf model
  and the table -- not the attractor: pass 0 with the stiffness checks' measurements
  (`replay.MEASURED_ROTATION`, 6.5 mm sideways -- about half what the 18:39 pass 0 actually
  strayed), later passes over a **bracket**: the paths from where the last pass went to 50 % past
  the taught path, each with the run-to-run spread, per point within its segment (audit B1 5c;
  a ball the size of the last stray refused pass 1 at the pick). Nothing is learned from 0.5 s before the grasp mark. **A replay learns the lag at its
  own `speed_scale`**: run the resulting labels at the same one. **A dry run of a replay only
  checks** (`replay_record_dry:=true` simulates into it, for tests only). **Replay at
  `rotational_stiffness:=30.0 transit_rotational_scale:=1.0`** (user, 2026-09-30; audit C5; a
  plan from the learned bundle runs at them by itself): 30 N·m/rad the whole way. At 60 the wrist vibrated at 26 Hz as joint 7
  entered the joint-limit repulsion zone, and the 15 -> 60 step at the set-down preceded the
  reflex; with the carry softened (x0.25) pass 0 is refused at 30 and 45 (link 6 passes 23.5 mm
  under the top board, the pads are near the table while lifting); at 30 x1.0 it passes. **With the object, the pass first guides
  it onto its taught place** (the scan fitted to the camera and to the taught cloud; "move it
  N mm toward the robot ..., turn it N deg ..."), and waits until it is within 10 mm / 10 deg;
  camera and segmentation must be up.
- **The first robot replay pass (2026-09-30 15:16, pass 0, no object, rotational 60) jittered
  descending to the pick and ended at the set-down in libfranka's `joint_velocity_violation`
  reflex** (robot red; FrankaHardwareInterface and every controller deactivated). From the
  container's log (`~/THOR_DOCKER_SETUP/.cache/ros_logs/ros2_control_node_*.log`; the container
  runs `--rm`, so `docker logs` is gone once it stops): the rotational stiffness stepped 15 -> 60 at
  label ~381 as the carry slowed, 1.05 s later joint 5's command hit its 12 N m limit (0.33 rad/s),
  0.4 s after that the reflex. The executor's "no fresh pose" came between the two, a symptom.
  Offline: on the branch from home, **joint 7 sits at its limit from label 0 to 156** (the whole
  approach and pick), inside CRISP's `joint_limit_repulsion` (0.1 rad, 5 N m) -- a candidate for
  the jitter, unconfirmed (the teach recording has no joints; the hand's half-turn twin frees joint
  7 but puts link 6 into the top board's margin). A replay pass now keeps every pose and joint
  state (efforts too), saves `aborted_<n>.npz` on an abort, and `replay_until_label` runs a
  partial pass (`partial_<n>.npz`, checked only over its labels). Callback load does not explain
  a stale pose: a copy of the node's structure fed 3x the robot's rates never saw one older than
  6 ms.
- **Plan with the learned bundle, and know what its check assumes** (2026-09-30, audit B1 step 6;
  `GRASP_LIBRARY.md` item 10). `~/tpgpt_demos/banana_20260925_183904/source_bundle_learned` (from
  `replay bundle`) has pass 3's command as labels, the taught labels, and the path the arm traced
  with its run-to-run margin. The planner executes the command and **checks the arm's path**
  (carried by the same map, margin as clearance, the object centred across the jaws when they
  close); the keypoints come from the taught labels, since the command leads the hand. The
  executor takes the learned settings from the plan and refuses others. On the 15 live captures:
  372 -> 320 admissible -- the taught check had assumed the arm is where it is commanded; one cell
  goes to 0, apple bottom_left (2 -> 0, its carried apple ~1 mm lower onto the bottom board). Most of
  the loss: the arm comes down to the grasp **4-5 mm off across the jaws** (a pad on the apple or
  the mug), which the replay did not learn (the arm does not follow the command there; the size of
  the stop-short, unconfirmed), and the margin against the table just before the grasp (banana
  top_right 33, 51 without; the table keeps only the margin's vertical part, the user's
  call). The arm sits 10-16 mm off its command even when still (6-9 N unmodelled, not lag);
  the learning removes the part that repeats. The arm path is pass 3's carried to a new scene: that it transfers is
  assumed (step 9). **Executed** without the simulation's gate and clamp, against its arm path
  (audit B1 step 8, above); dry-run only so far. The executor's summary reports how far the arm
  was *beside* its path (what matters) and *along* it (ahead is still on the checked path).
- **A plan that leaves the executor's workspace box is now refused at selection**
  (`SelectionSettings.workspace_min/max`, fed from the planner's own parameters). It used to be
  found only by the executor, which aborted the whole run instead of taking the next grasp
  (seen 2026-09-23: label 0 at x = 0.138 against a box starting at 0.15).
- **`refine_grasp` may also lift the hand**, not only push it deeper, when the generator's pose
  puts the fingertips through the table. On a 37 mm banana that accounted for 44 of 100
  candidates, several failing by a millimetre.
- **The exact geometry absorbs its own camera points at the error the camera actually has**: the
  table band is **25 mm** (its noise spans 7 mm typically, 24 mm at worst on the glossy sheet) and
  the shelf's explained distance is **30 mm** (its points sit 17-32 mm from the model because the
  camera is ~10 mm off there). At the old 15 mm both left phantom obstacles hanging in front of
  the surfaces -- 50 beside a banana, 1400 in front of the boards -- and those were what refused
  otherwise good plans ("the hand meets the shelf", "the carried object touches seen points" by
  one to five points). Cost: an object flatter than 25 mm on the table, or within 30 mm of a shelf
  panel, is not seen as an obstacle. **Re-measured after the 2026-09-26 recalibration: keep both.**
  Table points sit p95 10.8 / p99 20.7 mm from the touched table; shelf points median 6.7, p95 25.6,
  p99 34.5 mm from the model (at 20 mm 1572 would be left as phantoms, at 30 mm 352). Those 352 are
  mostly real things, not error: 141 standing inside bottom_right by the right wall, 198 hovering
  30-50 mm above the bottom board across its width.
- **The support is inferred, not assumed to be the table** (`support_height` NaN): the highest
  shelf board whose top is at or just under the object's lowest points, else the table
  (`support_match`, 30 mm). A banana standing on the bottom board used to fit 94 mm off, because
  the planner put it on the table.
- **The place clearance used to be an infinite plane, and it refused every upper-shelf grasp**
  (fixed 2026-09-24, audit A14). `_verify`'s `path_hits_surface` measured the hand's lowest point
  over the whole retreat against the destination's height; the taught retreat comes back down in
  front of the shelf, 215-314 mm from the slot and 35-53 mm "below" the board, over open floor.
  It now measures only while the hand is over the destination (its footprint plus 50 mm). Upper
  slots went from 0 of 25 admissible to 5-11 on every object and capture tried.
- **The demonstration decides which destinations are reachable, not the filters.** The placed
  orientation is a yaw about the table (audit A12), so the hand's tilt at the release is the tilt
  at the pick. The 01:57 demonstration (lowering onto the top board from above) cannot reach the
  bottom board -- every candidate hit `shelf top_board` while carrying. The 18:20 demonstration
  (sliding in under the bottom board) reaches **both**: bottom 0.969 / min det 0.86, top 0.962 and
  0.936 / min det 0.89 and 0.81, live, 2026-09-23. Prefer the harder demonstration. Measured by
  coverage on 2026-09-25 (`GRASP_LIBRARY.md`, item 6): the 16:35 and 16:52 demonstrations, picked
  at 13 deg instead of 41, admit 166 and 173 grasps across six objects and three slots against
  110 -- the fold all but disappears -- but carry the hand 16-37 mm under the top board inside
  the bottom slot instead of 60, and lose bottom_left (20 -> 9 and 8). **A pick near vertical
  and a hand kept low inside the slot** is what the numbers ask for. One demonstration serves
  every object and slot; choosing one is fine, mixing them per slot is not. **The 18:39
  demonstration is the best measured** (192 admissible; 91/80/21 at top_left/top_right/
  bottom_left, against 45/45/20 for 23:01), with the cleanest carry (the banana turned 1.9 deg in
  the hand).
- **A refused scan fit is retried on fresh camera frames** (`object_model_retries`, 2), because
  the same object can fit on one frame and not the next. The best attempt is kept; if none is
  accepted the planner falls back to the camera cloud, as before.
- **A dry run really executes the policy**, against a first-order impedance model of the arm
  (`tpgpt_executor_node.simulate`), in real time, publishing `/tpgpt/executor/simulated_pose`.
  So a dry run takes as long as the real motion (116 s for the taught demonstration at
  `speed_scale` 0.35) and can hit `max_policy_seconds`, which is why that default is now 300 s.
- **GraspGen-X is a diffusion sampler: each call returns a different set**, so how many you ask
  for decides whether planning succeeds at all, not just how good the best grasp is. Measured live
  on a banana: 100 candidates gave 16, 0 and 0 admissible (two failures); 400 gave 5, 38 and 38,
  all three planning, in 271, 22 and 15 s. **Since 2026-09-24 the draw is gone**: when the scan fit
  is accepted the planner uses the object's **fixed grasp library** -- 2000 grasps asked for once
  in the scan's own frame, kept in `outputs/hardware/grasp_library/`, carried into the scene by the
  fit (`grasp_library:=true`, `tpgpt/hardware/grasp_library.py`, audit A13). A refused fit, or no
  server on a cold cache, falls back to the live draw. `docs/hardware/GRASP_LIBRARY.md`.
- **What a destination admits is now measurable offline** (`python -m tpgpt.hardware.coverage`,
  any capture, any object, any demonstration via `--demo`): of an object's 2000 grasps, how many
  survive each stage and how many have an executable path, on a capture the planner recorded, with
  no camera and no robot. Use it before another evening at the
  robot. Results so far in `GRASP_LIBRARY.md`; raw in `outputs/hardware/coverage_fixed/`.
- **Beyond about 45 deg of grasp tilt the transport map folds.** Measured over 286 planned
  candidates on five objects: 1 fold in 155 below 45 deg, 74 in 131 above it (det(J) changes sign
  -- the sim's own rule, det -0.34 to -0.58). The **tomato soup can plans nowhere** because the
  hand can only grip it around its barrel: 57 of 60 of its grasps are past 60 deg. The likely
  mechanism is A12 (the placed orientation is a yaw). Selection does not know about tilt; it walks
  the generator's score order.
- **The soup can was replaced by the mug** (user, 2026-09-25; `GRASP_LIBRARY.md`). An upright
  vessel too wide to grip around needs a **rim**, and the rim must be in the *scan*, because that
  is what the planner plans against: `005_tomato_soup_can` is modelled sealed (22 sampled points on
  its top face, 3 in the cavity) however open the real can is, so a finger inside its rim is a
  finger inside solid material. `025_mug` is modelled open (0 on the top face, 340 in the cavity, a
  4-5 mm wall), and 1296 of its 2000 grasps are at or under 45 deg against the can's 308. Measured
  on the same capture: mug 3/14/3 (handle at 0 deg) and 7/4/6 (at 90 deg) admissible of 25 across
  top_left/top_right/bottom_left, against 0/0/0 for the upright can. The mug's **heading matters**
  -- the handle decides which side a grasp can come from. A can laid on its side plans too (6/4/0),
  but it rolls off a board.
- **The mug is set down opening-up with its handle visible to the camera** (user, 2026-09-25).
  Both are the fit's needs, measured: (1) from this camera an upright and an upside-down mug differ
  only inside the cup, and the fit chose upside-down on 15 of 20 live frames, so the fit only tries
  the protocol poses (`object_model.PROTOCOL_RESTS`: mug 0; pudding box 1 or 3, on its 88 x 35 face; banana 0, stem
  on the table -- its two faces fit within 0.03 on 42 of 51 captures, at headings up to 41 deg apart;
  apple 0, upright -- nearly round, so the camera cannot tell how it is turned; foam brick 0, 1,
  2 or 5, on a long face, 52 mm tall -- since 10-01 the camera reads part of its top 15-28 mm
  high, and on 10-06 18:07 the fit stood it on its end, 53 mm across where it is 78);
  (2) the D435's infrared bounces inside the glossy cup and puts 15-30 % of the mug's cloud ~80 mm
  too far, behind the mug near the table, and the fit turned the handle onto those points (and
  leaned the mug 13 deg). For the mug (`OPEN_VESSELS`) the fit now has no rest tilt and is run a
  second time without the points >20 mm behind the first fit along their ray. Recorded with the
  handle in four known directions (`~/tpgpt_bags/mug_handles`): within 30 deg on 9/14, 13/14,
  20/22 fits when visible, 2/18 when turned away from the camera -- nothing to fit it to.
  Other objects' fits are unchanged. The planner logs how many points it dropped.
- **The scan fit starts from the camera-facing surface too** (2026-10-02, audit A8;
  `object_model.FACING_CENTRING`). It used to centre every coarse start by the scan's whole
  median on the cloud's -- which lies toward the camera -- and refine only the best 6: the
  pudding box on its long side came out ~50 deg off on all four recent captures, accepted
  (64-66 % explained), and the jaws closed across its corner exactly as planned. Now 81-84 %.
  When a grasp looks wrong at execution, first compare the planned hand with the arm's at the
  close (FK of the logged joints); if they agree, the plan -- usually the fit -- is the suspect.
- **Flat-resting objects are fitted without lean** (`object_model.FLAT_RESTING`: mug, pudding
  box, foam brick; 2026-09-25; **the banana since 2026-10-03**). The rest-tilt search (up to 25 deg)
  is for what can really rest rolled or leaning (apple, can). It rolled the banana 5-21 deg about
  its long axis on most of its 49 captures, lifting its middle 5-15 mm off the table after the band
  the camera reads over its skin; flat it explains as much or more on 48 (median 0.68 -> 0.81). On
  10-03 19:33 the jaws, centred on that lifted middle, closed 10 mm above the real banana -- the arm
  at the plan to 2 mm. On the planner's live fits it also leaned the foam brick 19.5
  and 19.3 deg (on an edge, as it looked) and the upright pudding box 12 deg on both of its plans
  into bottom_left. On the day's seven real pudding-box clouds the new fit accepts all seven, on
  an end and flat; the old one put the box on its long edge on three. Known limit, in a test:
  with one flat face nearly all the camera sees, the box can sit 15-20 mm off along that face.
- **Plan records** (2026-09-25): a planned run now keeps `clouds.npz` too (the camera's object
  cloud and the obstacles), so its fit can be re-checked and its scene re-measured with
  coverage; a refused run records its `destination_name`/`destination`.
- **Our pudding box is not the YCB one: 88 x 70 x 35 mm, against the scan's 112.5 x 89.7 x 38.9**
  (user's tape, 2026-10-03; `object_model.MEASURED_EDGES`). The scan is scaled along its own
  edges when it loads, so the fit, the grasp library (a new one: its cache key is the scaled
  surface) and the detector's `max_size` (0.110) all see the real box. With the full-size scan
  the camera's box was 30-40 mm shorter than the model on every capture, and the fit had to put
  that length at one end or the other: on 10-03 17:32 it put it on the end facing the camera,
  where the camera saw nothing, the plan put the jaws 20 mm inside that end and they **closed
  on air** -- with the arm at the planned grasp to 4 mm. **It lies on its long narrow side, the
  88 x 35 face, 70 mm tall** (`PROTOCOL_RESTS` 1 and 3; 85-97 % explained, ~3.5 mm, the model's
  ends within ~6 mm of the camera's box). **Its silver face reads 14-33 mm high when up** -- a
  band of points over the box, 20-35 % of its cloud -- so the box's cloud looks ~90 mm tall: that
  is why the full-size scan on its side (89.7) seemed to fit, and why the measured box was briefly
  fitted standing (2a196a4, wrong; the user's 18:14 run). Its height allowance is therefore 38 mm
  (`HEIGHT_EXCESS`; lying -3..+33, stood on its end +44..+49), so a box stood on its end is refused
  only when its top reads high too. **Every pudding-box number before 2026-10-03 was for the
  full-size scan** (coverage, bottom_left, "a lying box is not graspable": flat, at 88 x 70, it fits
  the 80 mm opening across 70 -- unmeasured). The brick's prompt can take the box for the brick.
- **Our apple is 70 x 72 mm, not the YCB scan's 75 x 72** (user, 2026-10-03, between two flat
  faces -- "80" by eye was wrong; `object_model.MEASURED_EXTENTS`, scaled in the scan's own axes,
  z its stem axis). The 75 mm model left 2.5 mm each side in the 80 mm open hand, so grasps across
  its middle were refused as too wide and selection fell to the 353rd (score 0.61), which closed
  25 mm off the centre, on the cap, and pushed the apple away (20:24). Measured apple: explained
  median 0.75 on its 15 captures (0.60 before), tonight's first admissible grasp at rank 0-1. The
  generator's top-rated grasps close nearer the centre (top 50: median 12 mm; ranks 200-400: 17).
  The camera sees round objects narrower than they are (60-65 mm for this apple): the rim is lost
  to the edge filters. **And it is fitted upright and flat** (`PROTOCOL_RESTS` 0, `FLAT_RESTING`,
  2026-10-03): the free fit had its stem axis a median 75 deg from vertical over 17 captures (0.75
  explained; upright 0.95), and none of the apple's runs had executed a good grasp. Where the pads
  touch, the surface now slopes 9-12 deg from the closing direction on the 21:54 capture's chosen
  grasps; the 20:24 cap grasp sloped 41-42 deg, past what the rubber holds.
- **Destinations have names**: `destination_name:=top_left|top_right|bottom_left|bottom_right`,
  resolved from the touched slot marks (`config/shelf.yaml`, `slots:`, `shelf.slots()`).
- **`bottom_right` is abandoned** (user, 2026-09-25). Measured: every object scored 0 admissible of
  25 there on every capture, the carried object meeting 73 seen points standing in the slot
  (`GRASP_LIBRARY.md`). The user's reason is the reach -- it is the corner furthest from the base,
  and the arm gets there badly. Whichever it is, the destination set is now **top_left, top_right,
  bottom_left**. The slot stays in `shelf.yaml` and stays plannable; it is simply not one we run.
- **The object clouds vary a lot between frames.** Two captures of the same motionless banana,
  minutes apart, put 11% and 25% of their points in a dense layer ~20 mm above it (specular; no
  outlier filter removes it). The scan fit tolerates it: its thresholds come from seven real
  captures, judged at the set-down pose (`tpgpt/hardware/object_model.py`).
- **Flying pixels at a silhouette.** A pixel on an object's outline takes its depth partly from
  the background, so it lands on the ray between the two: an upright YCB can (68 mm) fused to a
  116 mm blob with a 60 mm tail streaming away from the camera, which broke its scan fit and most
  of its grasps. `segmentation_node` drops pixels whose depth changes faster than
  `depth_gradient_max_mm_px` (default 8; a good frame reads about 8 mm/px at the 99th percentile).
  Flat objects (a banana) show none of this; tall ones do.
- **The arm removes 50 mm of itself from the cloud** (`shelf.SELF_FILTER_MARGIN`; 30 mm until
  2026-09-26, 10 before that). The cable, the status ring and the connectors are not in the link
  meshes, so they survive as obstacles and the arm then "hits the seen points" -- of itself. After
  the camera moved (2026-09-26), 4 points of the elbow sat just outside 30 mm, 40 cm up in the
  middle of the workspace, and refused every grasp to top_left from where the arm was parked
  (0 of 100; 888 contacts, all on those 4 points). Park the arm clear of the objects: anything
  within 50 mm of it at capture time is erased with it.
- **A detection miss now says why** (2026-09-25, `docs/hardware/SEGMENTATION.md`). A target that
  reached no topic used to leave nothing in the log, so "scored 0.34 against a threshold of 0.35",
  "never seen at all", "matched a phrase that fits two prompts" and "detected fine but no depth in
  the mask" all looked the same. `segmentation_node` now detects at a report floor (0.05) and
  accepts at each target's own threshold, logging every target that did not make it, and
  `/ycb_detections/info` carries the per-target report. `save_frames:=misses` keeps the RGB, the
  depth and the report of every frame that went wrong; `prompt_bench` replays them offline to score
  candidate wordings and both caption modes. **The pudding box vanished (2026-09-25) because
  its score sits on the cutoff**: published on 320 of 605 cycles, and at 0.35-0.44 (median 0.39)
  when it was, against the shared 0.35. It has its own `threshold: 0.25` since; the foam brick,
  94 % at a median 0.39, is close to the same line. **The scan fit cannot tell objects apart**:
  fitting all six scans to 22 real captures of five objects, wrong scans were accepted on nearly
  every capture and the best-fitting scan was the wrong object on 13 of 22 (the mug's scan alone
  was accepted on 18 and fit best on 9 of the 17 non-mug captures). So a broad prompt plus "the fit
  picks" does not work here; the detector has to get the object right.
- **A box too big to be the object is refused** (2026-09-25). With the pudding box off the table
  its prompt found the **shelf** -- 72 000-pixel masks against the real box's 2 000, at 0.25-0.40,
  as high as the real box's own 0.33-0.48, so no cutoff can separate them. Every object in
  `target_objects.yaml` has `max_size` (its scan's largest dimension), and a candidate box that
  measures more than `size_margin` (1.5) times that, at the depth inside it, is refused before the
  best box is chosen. Published objects log their measured size. A new object needs a `max_size`
  (`test_every_listed_object_has_a_size`). Size does not separate look-alikes of similar size:
  the apple's prompt landed on the mug at 0.40 against 0.42.
- **The detection prompts are chosen on labelled frames** (2026-09-25, `SEGMENTATION.md`):
  red apple 0.48 / red cup with a handle 0.54 / silver and brown box 0.25 / brown cube with holes
  0.29, for the pudding box on its long side. Over 535 frames in 14 scenes: apple 74/78, mug
  63/70, pudding box on its side 94/112 alone and 51/86 beside the brick, brick 255/266; one false
  detection (a mug on an arm). `detection_replay` records per-frame boxes and `detection_sweep`
  re-scores thresholds offline in seconds. Earlier sets: "white and red box" missed the box end-on
  and let "brown cube" take its brown picture face. The first prompts had the colours wrong (the pudding box is white with a
  red logo, the brick brown); "dark brown block" often produced no box at all (102/180), because
  a box is only named from words that clear Grounding DINO's text_threshold. The pudding box's
  and apple's thresholds sit only just above their worst absent score -- expect the odd false
  detection. Prompts interact through the shared caption -- re-run
  `detection_replay --frames ~/ycb_detection_frames` after changing any of them (scene labels in
  `labels.yaml` there). **Put the mug down opening-up**: it was
  recorded upside down, and the rim grasps the coverage measured exist only upright.
- **A prompt is only as good as its worst frame.** `target_objects.yaml` takes an optional
  per-object `threshold:`, so one object that scores low does not drag the bar down for the others,
  and `caption_mode:=per_object` gives each prompt its own caption (one forward pass each). The
  latter removes the competition between phrases -- which is also what suppresses false positives:
  on a brown rectangle alone, a caption of just "banana." returned a banana at 0.41. Measure with
  `prompt_bench` before changing either.
- **Object clouds are only as good as the mask.** `segmentation_node` erodes each mask by
  `mask_erosion` (default 1 px, the sim's value; audit A9). Without it the boundary pixels take
  their depth partly from the background and fly towards the camera: a 68 mm YCB can fused to
  133 mm wide, which broke the scan fit, the `centred` stage and the jaw check. Restart the node
  after changing it.
- **The scratchpad in `/tmp` does not survive a reboot** (lost 2026-09-25, with the coverage
  runner and the detection replay in it). Anything a doc tells the reader to run belongs in the
  repo; recorded data belongs under `~` (`~/ycb_detection_frames`, `~/tpgpt_plans`).
- **The FR3's meshes are not a host package.** RViz's RobotModel display needs
  `franka_description`, which lives only in the robot container. It was copied out on 2026-09-22
  into `~/thor_ws/install/franka_description` (118 MB, outside git, and colcon leaves it alone), so
  the arm now draws on the host. If the container is rebuilt or the workspace install is wiped:
  `docker cp docker-thor-robot:/opt/docker_thor/install/franka_description ~/thor_ws/install/`,
  then re-source. Without it RViz spams "Could not load resource
  [package://franka_description/meshes/...]" and draws no robot; nothing else is affected.
- **The planner's `workspace_min/max` is a CLOUD CROP, not the executor's limit.** Its z stops at
  0.5 m because objects stand on a table; the executor's box reaches 0.7 m and a carried object
  rises to 0.55 m. The path criterion uses `executor_workspace_min/max` (kept equal to the
  executor's). Feeding it the crop refused 21-23 candidates per run as `outside_workspace`.
- **The demonstration's own start and end sit at the edges of that box**: its transported start
  lands near x = 0.134 (box starts at 0.15) and its retreat ends at z = 0.70 (box ends at 0.70),
  so a few candidates are refused for that alone. Teach with the hand starting and ending in front
  of the robot, or raise the executor's box (a safety parameter, so it is the user's call).
- **`grasp_offset` is zero (A5, decided 2026-09-23).** It used to be [−35, 0, −25] mm, set before
  the recalibration. It is applied *after* the jaw and table checks and *before* the planner, so it
  moved the hand 43 mm off the object and the planner then refused the grasp it had just accepted
  (`object_too_wide: the cloud measures 4 mm between the pads`). It caused most `plan_refused`
  rejections on every object. With it zero, all four objects plan.
- **Contact with *measured* points needs 3 points and 2 mm** (`path_check.CLOUD_CONTACT_SLACK`).
  Every other contact test had a tolerance (1 mm into exact geometry, 5 mm for resting); the
  robot-against-cloud test had none, so a finger pad clipping four points of a can's rim by 1–2 mm,
  114 mm above the board and moving away, refused an otherwise perfect plan.
- **The path check poses the arm at the waypoint, to 1 mm / 0.6 deg** (2026-09-30, audit A3b).
  Reachability is still the sim's 15 mm, but the IK stops as soon as it is inside it, and the
  collision tests used to pose the arm there -- a median 10.8 mm (max 14.8, up to 8.6 deg) from
  the waypoint on the 18:39 path, ten times the 1 mm contact tolerance. On the fifteen live
  captures: 374 -> 372 admissible, the losses real contacts at bottom_left. `path_feasibility`
  also takes `clearance` and per-piece `tilt` (a gap to keep; off in the planner, used by the
  replay): a uniform 5 mm would cost 374 -> 333 and empty bottom_left for the banana and mug.
  `SHELF_AND_PATH_CHECK.md`, "Where the arm is posed, and a margin".
- **Nothing from before the 2026-09-21 camera/shelf move is evidence** (user, 2026-09-22): not the X6
  trials in `outputs/hardware/hw_2026_09_x6/`, their clouds or plans, nor the 2026-09-15 banana
  demo (its object cloud went through the old calibration). Start from fresh captures and a fresh
  demonstration. The only exception is the FK fixture, which is robot-only.
- **X6's frozen `method:` block is the nodes' defaults, value for value** (reworked 2026-09-29).
  The trial passes the planner the keys in `x6_common.PLANNER_METHOD_PARAMETERS` and the executor
  every key of `method.execution`; everything else runs at its default. **Change a planner or
  executor default and `test_x6.py::test_the_frozen_method_is_what_the_nodes_run` fails**:
  re-sync `x6_experiment.yaml` (before trial 1 of a campaign; never during one). Destinations are
  the shelf file's named slots, the prompts are `config/target_objects.yaml` (X6's own copy is
  gone), a refused scan fit is retried on frames the trial captures and saves, and an
  `approach_blocked` plan is `preflight_failed` (not counted): re-home, re-run the same id.
  Grid: apple, mug, pudding box, foam brick x 3 marks x top_left/top_right = 24 per hand
  (`bottom_left` out since 2026-10-03, the user; a third destination, off the shelf, to come).
  **A trial keeps everything in its own directory** (2026-10-07): the executor's logs in
  `plan_dir/executions/` (they went to `~/.ros/x6_executor_csv/` before), the Inspire's fingers
  in `trace.npz` (`hand_*`; its width topic is null) and its close reports in `record.json`.
  **No X6 trial has been run with either hand yet**: the Franka's 10-01..10-03 runs are plain
  `tpgpt_pick_place` plan directories (no settled capture, no operator outcome).
- **`auto_plan:=true` re-plans every second** after a refusal (each cloud arrival starts a plan
  while `latest_plan` is unset). Old behaviour; `tpgpt_pick_place` calls the planner once.
- **The planner and the executor refuse to run without fresh `/joint_states`**: robot container
  up, FCI on, and `host_ros_env.sh` sourced (ROS domain 61). For offline dry runs with the robot off,
  pass `-p allow_missing_joint_states:=true`.
- **`franka_robot_state_broadcaster` logs "Failed to lock the realtime publisher"** several times a
  minute. Drops messages on `/franka_robot_state_broadcaster/robot_state` (wrench, EE mass), which
  the X6 recorder reads. Unresolved.

## Where to read next

- `docs/hardware/CALIBRATION.md` — camera calibration: procedure, tooling, current values, measured
  shelf geometry.
- `docs/hardware/commands.md` — how to launch the robot, pipeline, teaching, planning and execution.
- `docs/hardware/OPERATOR_CHECKLIST.md` — one page for a session: the protocol poses, homing,
  what the planner's log lines mean and what to do about each.
- `docs/hardware/FAITHFULNESS_AUDIT.md` — every place the hardware implementation differs from the
  simulation's, with file:line on both sides, severity, fix order and status.
- `docs/hardware/FRESH_START.md` — what counts as evidence after the move (nothing before it), and
  the staged test order: sensing, a new demo, planning dry runs, table grasps, then the shelf.
- `docs/hardware/GRASP_LIBRARY.md` — the fixed per-object grasp library (why the draw had to go,
  how it is cached and used) and coverage: what each shelf slot admits, measured, and the four
  things that measurement exposed.
- `docs/hardware/SEGMENTATION.md` — the detector: prompts, per-object thresholds, caption modes,
  and how to find out why an object was detected on one frame and not the next.
- `docs/hardware/SHELF_AND_PATH_CHECK.md` — the shelf model, the scene cloud, the whole-path check
  and grasp selection: which side's method was kept and why, how to measure and check the shelf,
  parameters, and what was verified.
- `docs/hardware/HARDWARE_EXPERIMENT.md` + `docs/hardware/HANDOVER_PROMPT.md` — the X6 experiment
  brief and its rules (one demonstration, no per-hand tuning, never force a grasp rank, clean tree).
- `x6_experiment/RUNNING_THE_EXPERIMENT.md` — operating the X6 harness.
- `docs/hardware/INSPIRE_HAND.md` — the Inspire RH56DFX-2R (a right hand): GraspGen-X
  grasps on a model of the hand, the mount, the measurements M0-M9, grasp tests, transport.
- `docs/hardware/hardware_integration.md` — the operating manual of the bridge: frames, verifying the
  cloud, **teaching a demonstration (keys, order, refusals)**, planning, execution.
- `docs/hardware/HOW_EXECUTION_WORKS.md` — execution in plain words: the run step by step, the
  learned bundle, the sim's lag gate vs the follow mode, every path correction and why it exists.
- `docs/sim/dynamics_execution.md` — why the executor's law and gates are what they are.
- `docs/sim/SIM_CLAUDE.md` — the simulation's own guide (reference, not instructions).

## Current state (2026-10-04): the Inspire hand

- **The Franka Hand runs are done; the Inspire RH56DFX-2R is mounted** -- a **right** hand (the
  2026-09-15 notes said left: wrong). Plan: `~/.claude/plans/this-repo-is-executing-calm-trinket.md`;
  run guide `docs/hardware/INSPIRE_HAND.md`; deviations `FAITHFULNESS_AUDIT.md` section D.
- **The scripted side grasp is gone.** The Inspire hand takes GraspGen-X's `inspire_hand` grasps,
  judged on a model of the hand from GraspGen-X's own URDF (`tpgpt/hardware/inspire_model.py`:
  26-DOP per link, the URDF's palm boxes, the adapter) by `grasp_selection.judge_hand`; the plan
  records where its fingers stop (`hand_closed_q`). An asymmetric hand keeps the generator's grasp:
  the demonstrating jaw takes the half turn instead (`plan.metrics["roll"]`, audit D3).
- **Measured on the robot and the bench (2026-10-05; `config/inspire_hand.yaml`):**
  - **The mount, by touch** (M3: index and little fingertips, 15 touches, rms 1.7 mm): yaw
    137.1 +- 1.4 deg (the first estimate, -45, was half a turn off), thickness 12.45 mm (the
    printed part alone: 16 by ruler), the ring concentric with the flange (the user). **One
    fingertip cannot tell the yaw from an offset** (index alone: 116 deg, little alone: 151, each
    held centred): touch with both (`/inspire_hand/point_little`; `inspire_touch` reads the
    finger), and fit with `--finger-bend`: two fingers still cannot tell an offset along the
    palm's normal from **fingers bent towards the palm -- which they are: 5.2 deg at register
    1000** (the free fit had called it a 10.6 mm offset).
  - **The ring is taped over loose screws** and rocks ~5 mm at the fingertips across the palm
    (across the closing direction; not modelled).
  - **The register map** (ruler, plus the touches' finger bend): fingers 0.12-0.13 -> 1.373 rad,
    thumb 0.082 -> 0.650. **The real open hand is 27 mm narrower along the closing direction
    than GraspGen-X's model** (it opens every joint to 0), and the fingers curl less far than its
    `close` (1.47).
  - **The close is simultaneous** (sequential pushes the object away), 400 g per finger, the
    pudding box 150 (`grasp_forces.yaml` `inspire_grasp`). **The hand reports a force stop as
    STATUS 2, not the manual's 3**: contact is judged by where a DOF stopped
    (`inspire_hand.CONTACT_MARGIN`). **An empty close is an "opposed" fist** (the thumb stops the
    middle and index fingers): a close **holds** something only when a finger stops beyond the
    empty close's angles (`grasp.empty_close_angles`); the release after a held close opens in
    one stage, and every open checks it opened.
- **Desk's payload is the weighed hand** (M5, 2026-10-05; `payload_measure`,
  `tpgpt/hardware/payload.py`): **0.708 kg at (-9, 1, 55) mm** in the flange frame, cable included
  (taped along the last link; the manufacturer's data alone: 0.568 kg at 88 mm). Desk's
  flange-to-TCP stays the Franka Hand's (`fr3_hand_tcp`, what every tool commands). A re-weigh with
  it in Desk agreed within the method's +-0.04 kg / +-15 mm (the arm's own model error at joint 2,
  friction). `payload_measure` reads Desk's values from the robot at its start.
- **The Inspire's grasp-test starts sit at the shoulder's straight point** (joint 2 ~0;
  2026-10-06; answered by the approach in joints, cf243d7 + the lead 22a4cf8: on the robot at
  21:18 the route took 7.5 s, joints within 0.03-0.11 rad of it, 0.031 rad from the checked
  start, and the grasp test held). The start
  posture is where the planner's own path from the arm's joints ends --
  the rule the Franka runs used, and it matched where their arm arrived to 0.06-0.51 rad
  (largest joint). The same brick grasp test puts the Franka's shoulder at +0.30, the Inspire's
  at 0 (its wrist at 3.6 rad, not 2.6; the flange 90 mm higher): there joints 1 and 3 turn about
  one line, the hand's pose does not fix the arm's, and the planner's path and the arm ended on
  opposite sides (19:05: planner -0.48, arm +0.03; the executor refused, 0.76 rad): pulled
  through hand-still motion with no limit, the arm could reach the planner's start on all 15
  Franka runs and on 3 of the 5 brick tests (not 18:57, 19:05: two separate sets of postures).
  The arm now takes the planner's postures in joints on the way there. `start_branch` (nearest the
  arm's joints; 3dd8985) was reverted in 0b0ba0b: it would have moved Franka starts up to
  1.44 rad off the arm (apple 10-03 22:18); the reverted planner reproduces all 29 Franka
  plans' starts of 10-02/10-03 exactly. (The earlier "Franka unchanged" check compared
  admissible counts, not start postures.) The constructed grasp test still tries both sides
  of the object, each thumb side, and ranks them by their start's distance from the arm.
- **The depth sweep's phase 2 (2026-10-07 03:01): apple, mug, pudding box and banana held on
  all 19 runs at 10, 20 and 30 mm back; with the brick (0 to 30) every object holds from 10 to 30
  mm back, middle 20.** **`calibrated_depth_m` is -0.02** (the user, 2026-10-07; a grasp test's
  `grasp_test_depth_offset` now adds to it). In grasp tests the
  hand goes 4-13 mm past the grasp mark (no learned lead; the stored push meets the 350 -> 600
  stiffness step), enough for the thumb to meet the table under a banana grasp planned 4-5 mm
  above it -- left as is for now (the user, 2026-10-07): measure it on the first Inspire
  transport runs, whose command carries the learned lead. The apple's misses were
  `caption_mode: joint` ("red apple red cup").
- **The depth sweep's phase 1 (brick, 2026-10-07 00:40; `INSPIRE_HAND.md`): library grasp
  #292 held 3/3 at 0, -20 and -30 mm, 1/1 at -10, 0/1 at -40: the band is 0 to -30.** A pinned
  library index is not a fixed physical grasp on the brick: its four long faces look alike and
  the fit picks rest 0, 2 or 5 by capture, so #292 became a grasp from below or into the table
  on 11 of 24 plans (normal selection is unaffected). The brick's height allowance is now
  26 mm (its top read up to +23 lying); a held close no longer fails on one finger still
  creeping into soft foam.
- **The first run at the transport's settings and in joints: 2026-10-06 21:18** (brick,
  constructed; `~/tpgpt_plans/foam_brick_20261006_211807_grasptest`): held through the lift
  (thumb, index, middle); arrival holds 0.8-0.9 s to 3.7-3.8 mm (at 400 N/m they had run out
  their 4 s); beside the path median 1.4 mm, at most 7.9; stiffness 456-607 N/m.
- **The first held grasp: 2026-10-06 18:27** (brick, constructed, fitted lying, 0.87 explained;
  `~/tpgpt_plans/foam_brick_20261006_182715_grasptest`): thumb, index and middle on the brick
  (ring and little on air; the plan expected the ring too), held through the lift, set down,
  released. The arm stayed within 6.4 mm of its path (correction up to 28 mm). **Both arrival
  holds timed out (4 s) 5.7-5.8 mm from their marks, the same way both times** (4 mm along base
  +x, 4 mm too deep): a bias the depth sweep (M6) must account for. Before it: 17:50 never
  descended (the sim's clamp; grasp tests now follow their path, audit D12), 18:07 fitted the
  brick standing and pushed it.
- **The foam brick is fitted lying on a long face only** (`PROTOCOL_RESTS`, `HEIGHT_EXCESS` 22 mm,
  2026-10-06): the camera reads a patch of its top 55-80 mm up (it is 52), growing since 09-26
  (+2 mm then, +18.5 at 18:07), and at 18:07 the fit stood it on its end; the fingers met it
  before closing. All 13 recorded brick captures fit lying. Why the patch grew is not known
  (the brick's clouds also went from ~680 points on 09-26 to ~1700 from 10-01).
- **The foam brick's fit starts along the camera's top face** (`object_model.TOP_FACE_HEADING`,
  2026-10-06): its smeared cloud let a brick turned 90 deg fit as well, and the user saw it turned.
- **The grasp depth is measured:** `calibrated_depth_m` -0.02 (the depth sweep, 2026-10-07).
- **The robot's external torques and wrench are the force it applies, not the push on it**
  (push test 2026-10-05, `wrench_watch`: pushing the hand up made Fz go down). Anything reading
  `external_wrench_in_base_frame` or `external_joint_torques` (the X6 recorder too) must flip
  them for the force on the hand (`payload.TAU_EXT_SIGN`).
- **Static friction under the soft joint controller** (`joint_impedance_controller.yaml`,
  40/20/15/8 N m/rad): the arm stops 0.05-0.08 rad short of a joint target, on the side it came
  from (joint 7 0.07 rad after `joint_homing`), and friction holds up to ~0.6 N m, which the
  robot reports as external torque. Anything read from `external_joint_torques` or the external
  wrench at rest carries it: model it by the approach direction, as `payload.fit_payload` does.
  **Every Inspire coverage number so far is void** (the mount was the -45 deg estimate).
- **The hand's serial driver**: `sudo modprobe usbserial && sudo insmod ~/ch341/ch341.ko` after every
  boot (`insmod` alone fails with "Unknown symbol": it does not load `usbserial`).
  Step-by-step measuring: `docs/hardware/INSPIRE_MEASURING.md`.
- **Grasp-only tests come first**: `-p task:=grasp_test` (`tpgpt/hardware/grasp_test.py`), with
  `grasp_test_constructed`, `grasp_test_library_index` and `grasp_test_depth_offset` for the frame
  check and the depth sweep. Dry-run tested offline end to end (`scripts/fake_scene.py`).
- **The first Inspire transport held and placed (apple -> top_left, 2026-10-07 19:05;
  `INSPIRE_HAND.md` section 6)**, after the camera was knocked and recalibrated (d18d450: the old
  extrinsics were 34 mm out sideways yet read the table and top board right). Overshoot past the
  grasp mark 6.9 mm (the learned lead keeps it to the Franka's range). **The wrist vibrated during
  the carry**: joints 5 and 7 swinging 0.3-0.5 rad/s RMS near 7 Hz as the path passed the wrist
  singularity (joint 6 at pi, their axes aligned); the Franka passed through it calmly on 3 of its
  19 runs, so it is the Inspire's inertia on a motion the controller holds only by its null-space
  spring. Open.
- **Coverage with the measured hand and the calibrated depth (2026-10-07, `GRASP_LIBRARY.md`):**
  9 of 10 object-slot cells have an executable transport, 23 admissible of 250 planned (25 per
  slot, sweep captures, 18:39 learned bundle); at GraspGen-X's own depth 5 cells and 14. Only the
  pudding box into top_right has none in its first 25. The commonest faults: the wrist within
  20 mm of full stretch (top_left) and the arm in the shelf's right wall (top_right). A whole-path
  check takes 20-45 s per Inspire grasp: a live plan at rank 0-7 takes up to ~5 min.
- **Fixed in the Franka path too**: the executor read the object's name from `plan.metrics`, so the
  per-object grip force (8cfb576) would never have applied (no run used it yet).

## State (2026-09-26)

- **All five objects plan live into all three slots** (2026-09-26, 16:42-17:38; plans in
  `~/tpgpt_plans/*_20260926_16*`/`_17*`): banana, apple, mug, pudding box (on its long side),
  foam brick x top_left, top_right, bottom_left -- 15 of 15, each on the 1st-8th grasp planned,
  17-59 s, min det(J) 0.52-0.93, with the 18:39 demonstration, the arm started from home
  (`joint_homing`) and the robot filter at 50 mm. The pudding box also planned into bottom_right.
  **Dry runs only: nothing executed on the robot yet.** That morning, before the filter fix and
  from a parked pose, the apple took up to 849 s and top_left refused.
- **The foam brick does not plan into bottom_left** (live, 2026-10-03: no grasp at any heading or
  position the user tried). The grasps one would expect -- tilted 20-70 deg, pads flat on the 53 mm
  faces -- are in the library (132 of 2000) and reach selection; they are refused because the
  hand's tilt at the pick is its tilt at the release (audit A12): tilted like that it meets the top
  board or the left wall in the bottom slot (`place_zone`, then the path check).
- **bottom_left plans on a thin margin** (measured 2026-09-29, `GRASP_LIBRARY.md` item 9): of the
  fifteen live plans' captures, the top slots admit 22-63 executable grasps each; bottom_left
  admits **2** for banana, apple, mug and foam brick and 4 for the pudding box. What kills it: the
  arm against the top board (10 mm lower since the move), the carried object against the bottom
  board, and 27 unexplained points 30-50 mm above the bottom board, 6-12 cm behind the slot mark.
  Expect bottom_left refusals in the X6 grid before anything else.

## Earlier state (2026-09-21)

- Camera and shelf were moved; **recalibrated and validated** (see `docs/hardware/CALIBRATION.md`).
- `x6_experiment/config/x6_experiment.yaml` predated the move; reworked 2026-09-29 (run
  `hw_2026_09_x6_r2`, three slots) at the user's request, to be revisited when more grippers work.
- **Done 2026-09-21 (audit A1, A2, A3, A6, C3):** the scene cloud, the shelf model, FR3
  kinematics, the whole-path check, the sim's selection rule and funnel stages, and the executor's
  infeasible-pose fallback. Tested offline and in a ROS dry run; **not yet run on the robot, nor on
  any capture from the new setup.**
- **Fresh start agreed 2026-09-22:** a new demonstration in the new setup; the old runs and demo are
  not used. The staged test order is in `docs/hardware/FRESH_START.md`.
- **Shelf measured by tape 2026-09-22** and confirmed by touching again. The camera was found about 10 mm
  off sideways at the shelf (open).
- Open, in the user's order: decide whether to refit the calibration with shelf correspondences;
  decide `grasp_offset` (A5); then add
  grippers; then the executor-default deviations (C1, C2, C4–C7), A4, A7, and B1/B2 (whether the
  paper claims transported stiffness on hardware).
