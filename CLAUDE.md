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
- **`free_drive` "pose went stale"** means its own callback received nothing for `max_pose_age`
  (0.5 s) — subscriber starvation, while `/current_pose` itself holds 250 Hz. Intermittent.
  `-p max_pose_age:=2.0` works around it; that is a safety parameter, so it is the user's call.
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
  -- **home the arm before planning.**
- **Selection stays in GraspGen-X's score order** (user, 2026-09-29): near-upright grasps are not
  moved ahead, even though steep ones fold more often and cost planning time.
- **Other executor guards still apply**, and they are not the box: `max_start_distance` (0.6 m
  from the plan's first label -- park the arm nearer, or raise it for a dry-run preview),
  `max_tracking_error`, the stall watchdog and `max_policy_seconds`.
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
  (its `source_bundle/` is unchanged). X6's frozen block still says `use_transported_stiffness:
  false`: re-sync it when the execution rework lands.
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
  the protocol poses (`object_model.PROTOCOL_RESTS`: mug 0; pudding box 2 or 4, on an end);
  (2) the D435's infrared bounces inside the glossy cup and puts 15-30 % of the mug's cloud ~80 mm
  too far, behind the mug near the table, and the fit turned the handle onto those points (and
  leaned the mug 13 deg). For the mug (`OPEN_VESSELS`) the fit now has no rest tilt and is run a
  second time without the points >20 mm behind the first fit along their ray. Recorded with the
  handle in four known directions (`~/tpgpt_bags/mug_handles`): within 30 deg on 9/14, 13/14,
  20/22 fits when visible, 2/18 when turned away from the camera -- nothing to fit it to.
  Other objects' fits are unchanged. The planner logs how many points it dropped.
- **Flat-resting objects are fitted without lean** (`object_model.FLAT_RESTING`: mug, pudding
  box, foam brick; 2026-09-25). The rest-tilt search (up to 25 deg) is for what can really rest
  rolled or leaning (banana, apple, can); on the planner's live fits it leaned the foam brick 19.5
  and 19.3 deg (on an edge, as it looked) and the upright pudding box 12 deg on both of its plans
  into bottom_left. On the day's seven real pudding-box clouds the new fit accepts all seven, on
  an end and flat; the old one put the box on its long edge on three. Known limit, in a test:
  with one flat face nearly all the camera sees, the box can sit 15-20 mm off along that face.
- **Plan records** (2026-09-25): a planned run now keeps `clouds.npz` too (the camera's object
  cloud and the obstacles), so its fit can be re-checked and its scene re-measured with
  coverage; a refused run records its `destination_name`/`destination`.
- **A lying pudding box is not graspable by this hand** (138 x 128 x 39 mm: both horizontal
  dimensions beat the 80 mm opening, and a pinch across 39 mm has the fingertips at the table).
  Of 2000 grasps none survives the jaw check. **It is set down on one of its long narrow sides,
  89 mm tall** (user, 2026-09-25, after live runs; `object_model.PROTOCOL_RESTS` 1 and 3). On its
  end (112 mm) it planned into bottom_left almost nowhere: grasps from above leave no room under
  the top board, and side grasps put the fingers below the board surface or the arm into it. The
  two live plans into bottom_left that evening had it on its side (85 and 91 mm tall in the
  camera's cloud). A box stood another way is refused by the fit (`MAX_HEIGHT_EXCESS`: the cloud
  reaching >20 mm above the fitted top), not planned as the protocol pose. On its side the brick's
  prompt can take it for the brick -- detection was tuned on frames with the box on its end.
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
  Grid: apple, mug, pudding box, foam brick x 3 marks x top_left/top_right/bottom_left = 36 per
  hand. **`bottom_left` is 81 mm from where the 18:39 demonstration set the banana down.**
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
- `ycb_pointcloud_pipeline/INSPIRE_HAND.md` — the Inspire RH56 hand: mounting, side grasp, caveats.
- `docs/hardware/hardware_integration.md` — the operating manual of the bridge: frames, verifying the
  cloud, **teaching a demonstration (keys, order, refusals)**, planning, execution.
- `docs/sim/dynamics_execution.md` — why the executor's law and gates are what they are.
- `docs/sim/SIM_CLAUDE.md` — the simulation's own guide (reference, not instructions).

## Current state (2026-09-26)

- **All five objects plan live into all three slots** (2026-09-26, 16:42-17:38; plans in
  `~/tpgpt_plans/*_20260926_16*`/`_17*`): banana, apple, mug, pudding box (on its long side),
  foam brick x top_left, top_right, bottom_left -- 15 of 15, each on the 1st-8th grasp planned,
  17-59 s, min det(J) 0.52-0.93, with the 18:39 demonstration, the arm started from home
  (`joint_homing`) and the robot filter at 50 mm. The pudding box also planned into bottom_right.
  **Dry runs only: nothing executed on the robot yet.** That morning, before the filter fix and
  from a parked pose, the apple took up to 849 s and top_left refused.
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
