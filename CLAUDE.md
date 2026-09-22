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
# ROS package (no robot, no ROS graph needed)
(cd ycb_pointcloud_pipeline && ~/venvs/tpgpt/bin/python -m pytest test/ -q)
# The method. test_gripper_actuation.py errors at COLLECTION (robosuite lacks UMIGripper),
# which aborts the whole run unless ignored. testpaths = tests, so the ROS tests are not collected.
~/venvs/tpgpt/bin/python -m pytest tests/ -q --ignore=tests/integration/test_gripper_actuation.py
# Just the hardware bridge (fast, ~10 s, and what most changes touch)
~/venvs/tpgpt/bin/python -m pytest tests/unit/test_hardware.py tests/unit/test_end_effector.py \
    tests/unit/test_grasp_selection.py tests/integration/test_hardware_grasp_selection.py \
    tests/unit/test_fr3.py tests/unit/test_shelf.py tests/unit/test_path_check.py -q
```
`test_hardware_grasp_selection.py` needs a running GraspGen-X server and skips without one.
`test_fr3.py` checks FK against 40 at-rest samples of the real arm (`tests/data/`).

**Offline ROS dry runs** go on another domain so nothing reaches the robot (`ROS_DOMAIN_ID=88`,
without `host_ros_env.sh`), with fake `/ycb_pointclouds/<obj>`, `/ycb_scene_cloud` and
`/joint_states` publishers. Base-frame points must be sent in the optical frame through the inverse
of the installed calibration. If you start the GraspGen-X server for a test, stop it afterwards.

Known failures, not regressions: `test_x6.py` × 2 (`x6_experiment.yaml` lists 6 objects where the
grid expects 5); TPGPT × 7, all gripper-registry tests, because the local GraspGen-X checkout lacks
4 descriptions (`jaco_3f`, `schunk_svh`, `ability_hand`, `fourier_hand`) and `robosuite_models` is
not installed. The two hands on the robot (`franka_panda`, `inspire_hand`) are unaffected.

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
  and `on_stall: continue`, which **switches the lag gate and its clamp off** for the rest of a run
  after a stall.
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
  object's base on the destination surface; retreat: fingers on the placed object only until they
  first clear it). Reachability is **strict** by default: every waypoint reachable in position, the
  pick and place with their orientation, and IK retried first. Full account: `SHELF_AND_PATH_CHECK.md`,
  "How a path is checked". The funnel calls the sim's own `tpgpt.grasp.filters` stage functions
  (visibility, on_target, grasp_collision, place_zone, centred), with the sim's pass-through rule.
  The hardware's jaw, refine, table and duplicate checks stay as hard stages. Deliberate
  differences, all documented there: a collision is never executed as a fallback, the top-down
  grasp is opt-in, and the A4 gates are kept. Importing `tpgpt.grasp.*` loads robosuite (~0.5 s);
  `fr3`, `shelf`, `path_check` and `executor` do not.
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
- **`free_drive` "pose went stale"** means its own callback received nothing for `max_pose_age`
  (0.5 s) — subscriber starvation, while `/current_pose` itself holds 250 Hz. Intermittent.
  `-p max_pose_age:=2.0` works around it; that is a safety parameter, so it is the user's call.
- **The shelf model is an estimate until the tape measurements are in** (`config/shelf.yaml`,
  `measured: false`). The shelf has no back and three boards (bottom, top, topmost). The robot
  touched only the slot marks and the bottom and top boards' heights. The rest comes from a tape
  (M1–M14, `docs/hardware/shelf_measurements.html`), because the arm can't reach the panel tops or
  the topmost board squarely. `shelf.resolve_tape` anchors the tape to the touched slots and
  refuses more than 6 mm of disagreement. Private artifact of that sheet:
  https://claude.ai/artifact/FgicphG5cmo9hvRnoutXpp
- **`grasp_offset` (planner node default [−35, 0, −25] mm) predates the recalibration**: a fixed
  base-frame shift applied to every grasp, probably absorbing the old calibration's error. Measure it
  on fresh captures (plan with it and with 0) before trusting it. Audit item A5, open.
- **Nothing from before the 2026-09-21 camera/shelf move is evidence** (user, 2026-09-22): not the X6
  trials in `outputs/hardware/hw_2026_09_x6/`, their clouds or plans, nor the 2026-09-15 banana
  demo (its object cloud went through the old calibration). Start from fresh captures and a fresh
  demonstration. The only exception is the FK fixture, which is robot-only.
- **X6 trials override the selection defaults** from the frozen `method:` block (`max_plans: 12`,
  `fallback_top_down: true`) via `x6_trial.py`. Don't be surprised; it waits for the X6 rework.
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
- `docs/hardware/FAITHFULNESS_AUDIT.md` — every place the hardware implementation differs from the
  simulation's, with file:line on both sides, severity, fix order and status.
- `docs/hardware/FRESH_START.md` — what counts as evidence after the move (nothing before it), and
  the staged test order: sensing, a new demo, planning dry runs, table grasps, then the shelf.
- `docs/hardware/SHELF_AND_PATH_CHECK.md` — the shelf model, the scene cloud, the whole-path check
  and grasp selection: which side's method was kept and why, how to measure and check the shelf,
  parameters, and what was verified.
- `docs/hardware/HARDWARE_EXPERIMENT.md` + `docs/hardware/HANDOVER_PROMPT.md` — the X6 experiment
  brief and its rules (one demonstration, no per-hand tuning, never force a grasp rank, clean tree).
- `x6_experiment/RUNNING_THE_EXPERIMENT.md` — operating the X6 harness.
- `ycb_pointcloud_pipeline/INSPIRE_HAND.md` — the Inspire RH56 hand: mounting, side grasp, caveats.
- `docs/hardware/hardware_integration.md` — the operating manual of the bridge: frames, verifying the
  cloud, **teaching a demonstration (keys, order, refusals)**, planning, execution.
- `docs/sim/dynamics_execution.md` — why the executor's law and gates are what they are.
- `docs/sim/SIM_CLAUDE.md` — the simulation's own guide (reference, not instructions).

## Current state (2026-09-21)

- Camera and shelf were moved; **recalibrated and validated** (see `docs/hardware/CALIBRATION.md`).
- **`x6_experiment/config/x6_experiment.yaml` is stale and not yet updated:** its destinations,
  table height and object list predate the move. The new shelf geometry is recorded in
  `docs/hardware/CALIBRATION.md` only. The user will rework X6 (including a new `run_id` and
  possibly four slots) later — do not edit it unprompted.
- **Done 2026-09-21 (audit A1, A2, A3, A6, C3):** the scene cloud, the shelf model, FR3
  kinematics, the whole-path check, the sim's selection rule and funnel stages, and the executor's
  infeasible-pose fallback. Tested offline and in a ROS dry run; **not yet run on the robot, nor on
  any capture from the new setup.**
- **Fresh start agreed 2026-09-22:** a new demonstration in the new setup; the old runs and demo are
  not used. The staged test order is in `docs/hardware/FRESH_START.md`.
- Open, in the user's order: measure the shelf (`shelf.yaml`); decide `grasp_offset` (A5); then add
  grippers; then the executor-default deviations (C1, C2, C4–C7), A4, A7, and B1/B2 (whether the
  paper claims transported stiffness on hardware).
