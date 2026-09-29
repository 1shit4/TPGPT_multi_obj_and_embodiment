# Running the transported policy on the real FR3

**What this is.** The bridge from this repository's simulation pipeline to a real
Franka FR3 driven by the CRISP Cartesian impedance controller (started by
`~/THOR_DOCKER_SETUP/scripts/run.sh robot`). The object is seen by a RealSense
camera and segmented by `ycb_pointcloud_pipeline`. Everything that decides where
the arm goes is the **same TPGPT code** the MuJoCo experiments use. The only new
arithmetic is a change of coordinate frame and a conversion to the hand's own axes.
Nothing under `tpgpt/sim`, `tpgpt/experiments` or the existing tests was modified.

**Status, stated plainly (2026-09-22).** The camera and the shelf were moved and
recalibrated on 2026-09-21. The current values and the measured shelf points are in
`CALIBRATION.md`, and **nothing recorded on the setup before that counts as
evidence** (`FRESH_START.md`). Since 2026-09-21 the planner also sees the shelf and
checks every candidate's whole path on the real arm's kinematics
(`SHELF_AND_PATH_CHECK.md`). This document is the operating manual for the rest: the
frames, verifying the cloud, teaching, planning and execution. The validation it
reports (§5.8, §8) is synthetic or on fake hardware, as each row says. The executor
still refuses plans made with anything marked `is_placeholder: true`.

---

## 1. The pipeline, stage by stage

```
source demonstration (once)
  ├─ taught by hand on the FR3 (§5)     tpgpt_teach_node -> tpgpt.hardware.teach      -> ~/tpgpt_demos/<demo>/source_bundle
  └─ or scripted in MuJoCo              tpgpt.hardware.export_source                   -> outputs/hardware/source_bundle

RealSense colour + aligned depth
  └─ segmentation_node (Grounded-SAM)   ->  /ycb_pointclouds/<object>   [camera_color_optical_frame]
       └─ camera_calibration.yaml (base -> camera_link)  +  realsense TF (camera_link -> optical)
            └─ tpgpt_planner_node  ->  object cloud in `base`
                 ├─ grasp               tpgpt.hardware.grasp_selection.select_and_plan
                 │                        GraspGen-X candidates (ZMQ server) -> screened (hand + the sim's funnel)
                 │                        -> planned best score first; the first whose WHOLE path clears the
                 │                           shelf and is >= 60 % reachable wins (tpgpt.hardware.path_check)
                 │                        shelf world: config/shelf.yaml + /ycb_scene_cloud, /joint_states
                 ├─ keypoints           tpgpt.sim.keypoints.scene_keypoints   (grasp-pose cube, FINDINGS §8p)
                 ├─ transportation map  tpgpt.transport.maps.TransportMap     (Sec. III-D/E, App. A)
                 └─ transported labels  tpgpt.transport.labels.transport_labels  -> saved plan
                      └─ tpgpt_executor_node
                           ├─ policy     tpgpt.policy.gp_policy.GPPolicy (refit, Sec. III-B step 2)
                           ├─ each 50 ms tpgpt.hardware.executor.HardwareExecutor.step
                           │             = gate_step -> advance_step -> advance_phase -> StallWatchdog
                           └─ publish    CRISP <controller>/target_pose  [+ target_stiffness]  + gripper
```

**The source demonstration** (the single teaching example that everything is
transported from) can now be **taught by hand on the real arm**, which is the
recommended route and is described in §5. The original route, still available and
still the default of `tpgpt_planner_node`'s `source_bundle_dir`, records it once in
MuJoCo by
`python -m tpgpt.hardware.export_source`. That calls exactly what `pipeline.run`
calls: `record_source_placement`, then `_to_tool_frame` with the Panda contact
offset, then `to_grasp_convention`. It then moves the result from robosuite's
world frame into the source robot's **base** frame. That keeps the affine part of
the map near identity, so it doesn't have to absorb a 0.9 m offset. It takes about
8 s. Measured on this machine: 200 labels at 20 Hz, all eight label families
present. In the base frame the pick is at (0.450, −0.173, −0.009) m, the table at
z = −0.110 m and the shelf destination at (0.76, −0.13), 0.1465 m up.

**The simulation still reproduces.** After the environment was set up (Python
3.12 venv, MuJoCo 3.7.0, robosuite 1.5.2, numpy 1.26 for `cv_bridge`), the
reshelving gate in `configs/reshelving.yaml` placed **9 of 10** seeds, failing
seed 8. That is one of the three seeds documented as failing (8, 11, 16), so the
environment change did not alter the validated result. The script's final JSON
summary then raised a numpy-version serialisation error. It is not fixed here,
because the task was not to touch the experiment code.

---

## 2. Coordinate frames — the part that silently goes wrong

| frame | meaning | where it comes from |
|---|---|---|
| `base` | robot base; CRISP's `base_frame` | robot |
| `camera_link` | RealSense body: x forward, y left, z up | `base_to_camera` in `camera_calibration.yaml` (here calibrated directly to the optical frame) |
| `camera_color_optical_frame` | image axes: z forward, x right, y down | realsense driver's `/tf_static` (factory), or YAML nominal |
| `fr3_hand_tcp` | fingertip centre: +Z approach, +Y finger travel | `franka_description` (0.1034 m along `fr3_hand` z; fingers prismatic along y) |
| grasp convention | +X closing, +Z approach | GraspGen-X / every keypoint cube |

A cloud point becomes a base point by
`p_base = T_base_camera_link · T_link_optical · p_optical`. The
colour-to-link part is read from the driver's TF when
`camera_link_to_optical.source: tf`, because its ~15 mm translation is per-unit
factory data. If your hand-eye calibration is already to the optical frame, set
`child_frame: camera_color_optical_frame` and the link step is skipped.

Grasp convention to TCP is one fixed rotation,
`R_tcp = R_grasp · GRASP_TO_FR3_TCP`, a −90° turn about the shared approach axis
(`tpgpt/hardware/frames.py`). A sign error here would not drop the object, since
a parallel jaw grips the same way after a half turn. It *would* command the
wrist 180° away from the plan, which the planner goes to some length to avoid
(`scene_keypoints`, §7.33). The RViz TCP axes marker shows it: green (TCP y) must
lie along the object's narrow side, blue (z) down.

**With THOR's `run.sh cameras`** the frames are `ground_cam_link` and
`ground_cam_color_optical_frame` and the topics start with `/ground_cam`. Set
`child_frame` and `camera_link_to_optical.child_frame` in the YAML to match, and
the topic parameters of `segmentation_node` and `calibration_check_node`.

---

## 3. The calibration file

`ycb_pointcloud_pipeline/config/camera_calibration.yaml`. The values below are the lab
calibration, checked on the live camera with §4, and are **not** placeholders any
more; the `is_placeholder` mechanism is kept for anyone starting from a fresh file.

* **`base_to_camera`** — `translation` in metres plus a rotation as a quaternion
  `[x, y, z, w]` or roll/pitch/yaw, to the **optical** frame
  (`child_frame: camera_color_optical_frame`). **Current values and how they were
  obtained: `CALIBRATION.md`.** Anything marked `is_placeholder: true` makes every
  node warn and the executor refuse.
* **`intrinsics`** — focal lengths and principal point in pixels for the colour
  stream (this unit's factory values at 848×480). With `source: camera_info`
  (recommended) the live `camera_info` is used and these are only compared
  against it. Any difference above `max_intrinsics_mismatch_px` (2 px) is
  reported, and so is a resolution change, because intrinsics are only valid at
  the resolution they were calibrated for. With `source: yaml` the segmentation
  node deprojects with them. It refuses to deproject images of a different size.
* **`verification.expected_table_height`** — the table surface's z in `base`,
  measured independently; currently **−0.0136 m** (`CALIBRATION.md`). The planner
  also uses it as the support height when `support_height` is not given.
  Destinations on the table should therefore be given at z = −0.0136. The easy way: touch the table with the closed
  fingertips and read `/current_pose` z. When set, the table-plane check becomes
  pass/fail.

---

## 4. Verifying the point cloud pose

Why it matters more here than anywhere else: the transportation map
**interpolates the keypoints exactly** (property (i), Sec. III-D). The keypoints
are a cube centred on the grasp point computed from this cloud. So a translation
error of *e* millimetres in the extrinsics moves the grasp by *e* millimetres, no
less and no more. A rotation error of θ moves a point by θ times its distance from
the camera: 17 mm per degree at 1 m.
The unit test `test_translating_the_scene_translates_the_plan_exactly` checks
that equivariance: shifting the whole scene by (70, −50, 30) mm shifts every
transported label by that amount to within 2 mm. The budget is therefore the
hand's: 80 mm jaws on a 50 mm object leave 15 mm per side on the closing axis.

`calibration_check_node` (run manually; the `tpgpt_verify.launch.py` wrapper was removed) runs five checks.
None moves the robot. They are listed cheapest first.

| # | check | what you look at | passes when | what it catches |
|---|---|---|---|---|
| 1 | intrinsics | log line `YAML - camera_info` | ≤ 2 px on fx, fy, cx, cy; same resolution | wrong resolution profile, wrong unit |
| 2 | **TCP overlay** `/tpgpt/verify/overlay` | a yellow cross + TCP axes drawn at `/current_pose` projected into the image | cross sits between the fingertips at several arm poses | gross extrinsic error, flipped axes, wrong optical convention |
| 3 | **scene cloud + table plane** `/tpgpt/verify/scene_cloud_base` | RViz, fixed frame `base`: the camera's view of the table and robot overlaid on the robot model; log `table plane: tilt … height …` | tilt ≤ 1.5°, height within 5 mm of `expected_table_height` | roll/pitch and height errors |
| 4 | object clouds `/tpgpt/base_clouds/<obj>` | log: centroid, lowest z, extent | lowest z ≈ table height; extent plausible for the object | per-object segmentation bleed, wrong depth scale |
| 5 | **touched points** (services) | `record_correspondence` ×≥4, then `solve_extrinsics` | current-calibration error a few mm at every point | everything, quantitatively; also produces a calibration |

**Check 5, step by step.** Click first, then touch, because the latest click is
paired with the TCP pose at the moment of the service call:

1. With the arm out of view, use RViz's *Publish Point* tool to click a distinct
   feature (a box corner, the top of a small ball) on the scene cloud.
2. Hand-guide the fingertip centre onto that feature. Use `probe_offset` if
   touching with another point of the hand.
3. `ros2 service call /calibration_check_node/record_correspondence std_srvs/srv/Trigger`
   — the reply states that point's error under the current calibration.
4. Repeat for at least 4 points spread in all three directions (not in a line,
   different heights), then
   `ros2 service call /calibration_check_node/solve_extrinsics std_srvs/srv/Trigger`.

The reply lists the **current** calibration's error at every point (the
verification) and a least-squares (Kabsch) refit with its residuals, as a
ready-to-paste `base_to_camera` block. It is also written to
`~/tpgpt_calibration/`. Read the refit's residuals before trusting it. Small
residuals at well-spread points mean the refit is consistent. Large ones mean the
clicks and touches don't correspond, and no calibration will fix that.

**Measured on a synthetic scene with known ground truth** (a renderer draws a
table and a 40×70×120 mm box through the placeholder calibration, with a
link→optical TF deliberately 0.4 mm off the YAML nominal):

| check | result | ground truth |
|---|---|---|
| intrinsics | 0.00 px difference | identical |
| TF vs YAML nominal | reported as 0.4 mm, 0.00° | 0.4 mm |
| table plane | tilt 0.01°, height +0.1 mm | 0°, 0 mm |
| object cloud centroid (base) | [0.463, 0.048, 0.075] m | [0.4625, 0.0484, 0.0755] m (renderer's own) |
| touched points, calibration perturbed by +20 mm and +2° | current error 19–36 mm, RMS 28.3 mm | perturbation present |
| refit from those 5 points with 1 mm noise | residual RMS 0.9 mm; translation within 1.3 mm of truth | [1.0, 0.0, 0.6] |

What would make these checks misleading: a synthetic scene has no depth noise,
no multipath and no motion blur. The table-plane numbers on a real camera will be
millimetres worse, and the tolerances in the YAML are a starting point.

---

## 5. Teaching the demonstration by hand

**Why teach on the real arm.** The simulated demonstration lifts a box 26 cm and
carries it 31 cm along +x onto a shelf. §6 measures what that costs: a destination
that asks the map to turn that motion into a flat sideways slide folds the map
(`min det` −0.041). A demonstration of the task you actually want transports over a
much smaller change, so the map stays close to identity. It also removes three
sim-to-real gaps at once: the grasp is the one a person chose on the real object,
the object cloud comes from the real camera, and the hand is the real Franka Hand.

### 5.1 How the arm can be moved and still be measured

**The FCI must be active.** The FCI (Franka Control Interface, the network link
libfranka uses) is the only way the robot's joint angles leave the controller.
With it off, `run.sh robot` cannot connect, so there is no `/joint_states` and no
`/current_pose`. Hand-guiding in Desk with FCI off moves the arm, but nothing can
record it. So teaching needs FCI on, and then one of two things lets a person move
the arm:

| mode | how the arm is moved | what is switched | status |
|---|---|---|---|
| **`compliant`** (default) | A CRISP `CartesianController` with **zero stiffness** (`config/tpgpt_teaching_controller.yaml`). libfranka already cancels gravity, so no spring pulls the hand anywhere and it floats. | strict switch to `tpgpt_teaching_controller`, and back to `cartesian_impedance_controller` afterwards | switching verified on fake hardware; **not yet felt on the real arm** |
| `passive` | Franka's own guiding mode (Pilot guide button) | nothing | only usable if `/current_pose` keeps streaming while guiding with FCI on. Check first with `ros2 topic hz /current_pose` while guiding. |

Two other ways were considered and **not** built. Teleoperating the arm with a
SpaceMouse through `/target_pose` would record the commanded target exactly (that
is what TPGPT's labels are, §5.5), but it is awkward for a pick-and-place. Tracking
a marker on the hand with the camera needs no FCI, but it would pass every
millimetre of camera calibration error straight into the demonstration.

**What the zero-stiffness controller commands.** Every task stiffness `k_*` is 0
and the nullspace stiffness is 0, so the torque is only damping plus Coriolis
compensation and joint-limit repulsion:
`tau = J^T(−D_task · v_ee) + N(−D_null · dq)`. CRISP uses an explicit damping when
it is > 0 (`damping_for_stiffness` in `utils/command_validation.hpp`), so the
values in the YAML are the drag you feel: 8 N·s/m, which is **2.4 N** at 0.3 m/s,
0.5 N·m·s/rad rotational and 0.2 N·m·s/rad per joint. Raise them if the arm drifts;
lower them if it feels heavy.

**Before the first session:** the hand's mass and centre of mass must be set
correctly in Desk (end-effector settings). With zero stiffness, any error in the
gravity model shows up directly as the arm sinking or rising on its own.

### 5.2 Running a session

```bash
# T1 robot, FCI active
cd ~/THOR_DOCKER_SETUP && HARDWARE_ACK=I_UNDERSTAND_FR3_EFFORT_CONTROL ./scripts/run.sh robot

# T2 camera + segmentation (ycb venv: it needs torch, the TPGPT venv has none)
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
source ~/venvs/ycb/bin/activate
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py

# T3 teaching node + RViz (TPGPT venv: the node imports tpgpt)
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
source ~/venvs/tpgpt/bin/activate
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=mustard_bottle
#   [teach_mode:=passive] [output_dir:=~/tpgpt_demos] [rviz:=false]

# T4 keyboard (ros2 launch gives a node no keyboard, so the keys live here)
source ~/THOR_DOCKER_SETUP/scripts/host_ros_env.sh && source ~/thor_ws/install/setup.bash
ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard
```

| key | service | what happens |
|---|---|---|
| `c` | `capture_scene` | waits for **3 new** object clouds (published after the key press, so a cloud taken while the arm was in view is not used), fuses them in `base` exactly as the planner does, and shows them orange. Warns if the TCP is within 15 cm of the cloud. |
| `t` | `start_teaching` | asks for a typed `yes`, loads and configures the teaching controller if needed, strict switch. Refuses if `cartesian_impedance_controller` is not loaded, because then the arm could not be held afterwards. |
| `s` | `start_recording` | refuses without a captured scene. Stores every `/current_pose` message from here on. A magenta trail is drawn live. |
| `g` | `mark_grasp` | the object counts as held from this pose on |
| `r` | `mark_release` | the object counts as let go from this pose on |
| `x` | `stop_recording` | switches to `cartesian_impedance_controller`, which CRISP starts at the measured pose, so **the arm stays where it is**. Then converts, saves and plots. |
| `p` | `capture_placed` | optional. With the arm out of view, fuses the placed object's cloud and compares it with the placement the node derived (§5.4). |
| `h` | `hold` | switch to the hold controller now. Discards a recording in progress. |

**The order is `c`, `t`, `s`, move to the object, `g`, carry, set it down, `r`,
retract, `x`.** Captures are refused while the arm floats, because a capture
blocks the node for several seconds and with it the watchdog. So take the pick
cloud before `t`, and the placed cloud after `x`.

**No gripper is used.** The `g` and `r` marks record only *when* the object was
held. TPGPT needs exactly that: the gripper channel of the labels, from which
`carry_indices` and `carry_transform` read the grasp and the release. You can move
the object with your other hand against the open fingers, or not move it at all.
The placement is computed from the hand's motion (§5.4), so `capture_placed` is
only meaningful if the object really was moved.

**How to demonstrate well**
* One grasp and one release. A second `g` is refused.
* Slowly: under 10 cm/s. Execution lag grows with speed (§5.5). The summary prints
  the peak speed.
* Keep the grasp pose you want the robot to use: fingertip centre on the object,
  fingers across its narrow side. The TCP at the `g` mark **is** the source grasp.
* Pausing at the grasp and at the release is fine. Still stretches longer than
  1 s are shortened to 1 s (`max_dwell`), and the still start and end are trimmed
  to 0.25 s.
* Set the object down on a surface. The destination height is derived as
  `support height + (TCP z at release − TCP z at grasp)`, so it is right only if
  the object sat on the table when grasped and on a surface when released.
* Don't move the object between `c` and the grasp.

**Safety, what stops a recording.** A pose older than 0.2 s, the TCP leaving
`workspace_min/max` (default x 0.15..0.85, y −0.6..0.6, z −0.25..0.7 m), or more than
120 s of recording. Each switches to the hold controller and discards the
recording. Ctrl-C (SIGINT) or SIGTERM on the node while the arm floats also
switches to hold before exiting. Measured on fake hardware: the switch completes
in about 30 ms, under both signals and through `ros2 launch`. None of this
replaces the emergency stop.

### 5.3 What is saved

`~/tpgpt_demos/<object>_<time>/`:

* `raw.npz`: every recorded pose, the two marks, the object clouds, the table
  height. Nothing is processed, so the conversion can be re-run offline with other
  settings:
  `python -m tpgpt.hardware.teach --raw <demo>/raw.npz --out <demo>/source_bundle [--max-dwell 2 --still-speed 0.005 ...]`
* `source_bundle/`: **the same three files** `export_source` writes
  (`tool_labels.npz`, `placement.npz`, `manifest.json`), loaded by the same
  `SourceBundle.load`. Nothing downstream can tell them apart except through
  `manifest.json` (`source: taught_on_fr3`, `stiffness: directional`, or `static` for a
  bundle converted before 2026-09-29 or with `--stiffness-profile static`).
* `teach.json`: the conversion report (below) and the settings used.

How each field is made (`tpgpt/hardware/teach.py`):

| bundle field | from the recording |
|---|---|
| `positions` | measured `fr3_hand_tcp` (the fingertip centre, the reference TPGPT's keypoints use), resampled to 20 Hz by linear interpolation between message stamps, then Savitzky-Golay smoothed (7 samples = 0.35 s, order 2) |
| `orientations` | measured `fr3_hand_tcp` rotation, slerp-resampled, converted to the grasp convention with `fr3_tcp_to_grasp` (the inverse of the −90° turn in §2) |
| `velocities` | the Savitzky-Golay derivative at the 50 ms control period |
| `stiffness`, `damping` | directional (default): 350 N/m along the motion, 600 across it and in every direction when still, log-Euclidean smoothed over ±5 labels, `D = 2K^½`; or `static`: constant 400 N/m and 40 N·s/m (§5.5) |
| `gripper` | +1 from the `g` mark to the `r` mark, −1 elsewhere |
| `time_belief`, `time_rate` | `linspace(0, 1, M)` and `1 / (dt·(M−1))`, from the step index; the message stamps are discarded after resampling (`docs/sim/SIM_CLAUDE.md` rule 5) |
| `placement.points` | the `capture_scene` cloud |
| `placement.grasp` | TCP position and grasp-convention axes at the first held label |
| `support_height` | `expected_table_height` from the calibration YAML (−0.0136 m), unless given as a parameter |
| `destination`, `destination_height` | the object moved by the taught carry (§5.4) |

### 5.4 Where the object was put, without seeing it

Once the object is held it moves rigidly with the hand, so its motion between the
grasp and the release is the hand's own: `T_obj = T_tcp(release) · T_tcp(grasp)⁻¹`
(`carry_transform`, the same function the simulated pipeline uses). The node
applies it to the captured cloud to get the placed object (cyan in RViz). The
`destination` is set to the centre of that cloud's box, fitted in the placed grasp's
task frame, which is the centre `scene_keypoints` measures its lateral offset from.
So the offset it inherits is **zero**, and a planner `destination` means "put the
object's centre here". The placed height is
`support_height + (z_release − z_grasp)`, which assumes the carry turned the object
only about the vertical. The tilt of the carry is reported, and above 10° the
height is flagged as approximate (`carry_tilt_ok: false`).

`capture_placed` checks this against the camera. It reports the xy distance
between the measured and the predicted placed centroids, and both lowest points.

### 5.5 Stiffness labels, and the lag they leave

**Directional (the default since 2026-09-29).** A hand-guided demonstration runs at
zero stiffness, so there is no stiffness to record; like the simulation's scripted
teacher (350 N/m in transit, 900 at the grasp and insertion), the labels are the
teacher's specification (`teach.directional_stiffness`):
`K = 600·I − 250·w·u uᵀ` -- 350 N/m along the direction of motion `u` (the chord of
the last 10 mm of path), 600 across it, blended in by speed `w` (0 at ≤ 2 mm/s, 1 at
≥ 20 mm/s), so a still hand (the grasp, the release) is stiff in every direction and a
moving one gives way along its motion only. The profile is smoothed with a
log-Euclidean mean over ±5 labels: unsmoothed, the policy's regression of it overshot
to 236..980 N/m between labels; smoothed it stays at 349..609. `D = 2K^½`, the damping
CRISP applies. Numbers chosen from this arm: 350 keeps the ~10 N friction's lag at
29 mm, inside the lag gate's 45 mm; 600 is the highest value below the 800 that
vibrated (with rotational 60; 600/30 to be checked). Run with
`use_transported_stiffness:=true` so the full matrix reaches the arm (§7); the
executor's `transit_rotational_scale` then resolves to 0.25, the simulation's value.
`--stiffness-profile static` (or the node's `stiffness_profile:=static`) gives the
old constant labels below, bit for bit.

**Static.** TPGPT's labels are the **commanded attractor** (the point the impedance spring
pulls toward), not the measured pose. An impedance arm lags a moving attractor by
`K⁻¹·D·v` (`docs/sim/SIM_CLAUDE.md`, §2.7 of `docs/sim/ROBOTICS_NOTES.md`). A hand-guided demonstration has
no attractor, only the path the hand took. Taught bundles still use that
measured path as the attractor (replacing it is the next stage, audit B1). The static
profile uses one stiffness and damping taken
from the controller that will execute it: `cartesian_impedance_controller` runs
400 N/m and 40 N·s/m (`controllers.real.yaml`), and the executor's default
`use_transported_stiffness:=false` uses exactly that controller. So the arm will
run **behind the taught path** by `D/K · v = 40/400 · v`:

| taught speed | lag at `speed_scale` 1.0 | at 0.5 (X6; the node default is 0.35) |
|---|---|---|
| 5 cm/s | 5 mm | 2.5 mm |
| 10 cm/s | 10 mm | 5 mm |
| 20 cm/s | 20 mm | 10 mm |

`speed_scale` slows every label uniformly (§7), so the lag halves with it. The
summary line prints the lag at the demonstration's peak speed.

**One executor setting follows from "static".** The executor holds the hand's
angle firmly at the two ends of the task and relaxes it to 25% while carrying.
It decides which is which by comparing each label's compliance with the midpoint of
the demonstration's range. With one stiffness throughout, that compares equal
numbers. Measured on a dry run of a plan from a taught bundle: while carrying,
the rotational gain flipped between 60 and 15 N·m/rad on **64** steps. The planner
now records `source_stiffness: static` in `plan.json`, and the executor's
`transit_rotational_scale` defaults to *auto*: **1.0 for a static source** (a
constant 60, 0 flips on the same plan) and 0.25 otherwise (the simulated source
still flips twice, firm → compliant → firm, as designed). An explicit value still
overrides it.

### 5.6 Refusals

The raw recording is always saved first, so a refusal loses nothing. Each refusal
names its stage.

| refusal | cause |
|---|---|
| `marks_missing` | `g` or `r` not pressed while recording |
| `release_before_grasp`, `marks_outside_recording`, `carry_too_short` | marks in the wrong order, outside the recording, or no 50 ms step between them |
| `no_motion` | the TCP never moved faster than 1 cm/s |
| `too_few_labels` | fewer than 40 labels (2 s) after trimming |
| `object_not_seen` | fewer than 40 points in the captured cloud |
| `grasp_not_on_object` | TCP at the grasp mark more than 50 mm from the nearest object point: grasp marked away from the object, or the object moved after `c` |

### 5.7 RViz (fixed frame `base`, displays already in `rviz/tpgpt.rviz`)

* **Teach markers** (`/tpgpt/teach/markers`):
  * label points, **blue** while free and **red** while held;
  * a thin grey line for the raw 1 kHz path;
  * yellow spheres at the grasp and the release;
  * TCP axes at the start, grasp and release (x red, y green = finger travel,
    z blue = approach);
  * a cyan disc at the destination;
  * a summary text;
  * a magenta trail while recording.
* **Teach object cloud** (orange), **placed (predicted)** (cyan), **placed (measured)** (green).

### 5.8 What was measured, and how

Nothing in this section involved a real robot or a real camera. Each row says what
stood in for them.

| what | how | result |
|---|---|---|
| conversion recovers a known demonstration | the simulated source's fingertip path and rotations played back as a 1 kHz `fr3_hand_tcp` recording, 2 s still at both ends, 0.2 mm Gaussian noise, marks 10 ms outside the true grasp/release | 214 labels (the source's 200 plus the still margins kept at each end); grasp TCP **0.4 mm** from the source grasp, closing axis identical (dot product 1.000); destination (0.76005, −0.13004) vs source (0.76, −0.13); conversion 0.23 s |
| identity transport | plan the converted bundle onto its own scene | `min det` **1.000**, aim 0.0 / 0.0 mm, keypoint residual 1e-16 m, path identical to 3e-13 mm |
| a moved object | object shifted (+50, +80) mm, destination −50 mm in y | `min det` 0.563, aim 0.0 / 0.0 mm; policy fit 0.21 s |
| the ROS node end to end | `tpgpt_teach_node` (`passive`) against a separate-process 1 kHz fake `/current_pose` replaying that path, and a fake camera publishing the source cloud through the real calibration, driven through all services | received **1000 Hz**, recorded 998 Hz, worst message gap 22 ms; grasp and release landed on labels **57 and 166**, the source's own 57 and 166; `capture_placed` with the carried cloud: centroids **1 mm** apart |
| executor on a taught plan | dry run of the moved-object plan, `speed_scale` 0.5 | 427 steps (2 × 214); jaws closed **5.1 mm** from the planned grasp and opened **3.8 mm** from the planned release; gate never shut |
| compliant switching | CRISP fake-hardware container, domain 88 | teaching controller loaded from the YAML, configured, strict switch both ways; `stop_recording` and a refused conversion both left `cartesian_impedance_controller` active; SIGINT and SIGTERM while floating switched to hold and exited |

**Two defects found only by these runs, both fixed.**
* rclpy's `MultiThreadedExecutor` delivered **15 Hz** of a 1 kHz pose stream in
  which a `SingleThreadedExecutor` in the same setup received all 1000 Hz.
* With only the pose on its own single-threaded executor, 18 s of recording under
  service calls still had **14 gaps over 50 ms and 2 over 200 ms** (worst 268 ms).
  The 0.2 s watchdog would have stiffened a floating arm mid-demonstration. The
  node now runs three single-threaded executors: services and timers; controller
  manager, clouds and TF; and the pose stream. Under the same load that gave 250 Hz
  from the fake controller with the worst gap **6 ms** and no false aborts.

**What would make these misleading.** Fake hardware does not move, so nothing here
tests how the zero-stiffness controller feels, whether the arm drifts with the real
hand payload, or how smooth a human demonstration is. The synthetic recording came
from a scripted, piecewise-linear path, much smoother than a hand. And the executor
dry run's plant is the first-order impedance model, not the arm. The first real
session should start with `c` / `t` / `h` alone, arm supported, to feel the
controller before recording anything.

---

## 6. Planning

```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node --ros-args \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=mustard_bottle -p destination:="[0.70, 0.15, -0.006]"
ros2 service call /tpgpt_planner_node/plan std_srvs/srv/Trigger
```

Leave out `source_bundle_dir` to plan from the simulated demonstration instead.
With a taught bundle, **make the first plan with the object back where it was
taught and `destination` at the taught one** (printed by `stop_recording`, and in
`teach.json` as `destination_m` / `destination_height_m`). That is identity
transport: expect `min det` close to 1 and a path that retraces the demonstration
in RViz, as measured in §5.8 (1.000 on the synthetic scene; a real re-capture of
the cloud differs by a few millimetres of depth noise, so not exactly 1). Anything
far from that means the bundle, the frames or the cloud are wrong. Find out which
before moving the object anywhere.

The node fuses `n_frames` clouds (3 by default), transforms them to `base`, crops
to `workspace_min/max`, voxel-downsamples at 3 mm, drops isolated points and caps
the result at 8192 points. It then chooses the grasp (§6.1) and calls
`tpgpt.hardware.planner.plan` with it. The plan is saved to
`~/tpgpt_plans/<object>_<time>/` and shown in RViz:

* orange: the fused object cloud; cyan: where the plan will put that object
* green spheres: the target keypoints (the grasp cube at pick and at place)
* three axes at the start, the grasp and the release: `fr3_hand_tcp` x red,
  y green (finger travel), z blue (approach)
* the path, blue where the gripper is open and red where it is closed
* gripper outlines of the grasp candidates: chosen thick green, planned but
  refused orange, screened out faint red (namespace `grasp_candidates`)

**Refusals name their stage.** `object_not_seen` (fewer than 40 points),
`object_too_wide` (cloud under an open finger, or more than 80 mm **between the
pads**; a cloud can prove "too wide" but never "fits", §7.36), `hand_collides`
(the palm inside the object cloud), `no_executable_grasp` (§6.1),
`keypoints_degenerate`, and `map_not_a_diffeomorphism`: the map's smallest
Jacobian determinant is below `min_determinant` (0.1), meaning it squashes or
folds space somewhere along the path.

**Measured: which destinations the simulated demonstration supports.** These
numbers belong to the MuJoCo source. A taught demonstration supports a different set,
centred on what it demonstrated, and it has not been measured. The same sweep,
run on a taught bundle, is the way to find out. The simulated demo lifts
its box 26 cm and carries it 31 cm along +x. A 40×70×120 mm box at (0.45, 0.05)
on the table was planned to five destinations at three yaws, 15 plans, no
physics:

| destination (x, y, surface z) | yaw −0.5 | yaw 0.0 | yaw 0.5 |
|---|---|---|---|
| (0.75, 0.05, 0.00) — along the demo's direction | 0.575 | 0.501 | 0.364 |
| (0.65, 0.05, 0.25) — along it and up, like the demo | 0.635 | 0.599 | 0.516 |
| (0.60, 0.30, 0.00) | 0.355 | 0.444 | 0.514 |
| (0.45, −0.25, 0.25) — sideways, up | 0.518 | 0.405 | 0.320 |
| (0.45, −0.25, 0.00) — sideways, same height | 0.089 *refused (< 0.1)* | **−0.041 fold** | **−0.115 fold** |

Cells are `min det(J)`. The clouds were sampled on the box surface without its bottom face, not rendered. The aim, meaning the transported path's distance from the
grasp point at the grasp and from the place point at the release, was **0.0 mm in
every one of the 15**, as the exact interpolation guarantees. The one failing
row asks the map to turn a lift-and-carry into a flat sideways slide *and* absorb
a large frame rotation at the same time. Carrying the same object sideways but
upward is fine. That is a property of this one demonstration, not a bug; a
demonstration closer to the real task would remove it.

**A known perception limitation, measured.** One camera sees only the object's
near faces, so the cloud is lopsided toward the camera. On the synthetic box the
default grasp point landed **11 mm** from the box's true centre, toward the
camera. That is not an extrinsic error: check 4 would read correctly. It is a
biased grasp. It sits inside the 15 mm closing budget for a 50 mm object and
eats most of it. Remedies, none implemented: a second viewpoint, a shape prior,
or passing an externally chosen grasp (`TargetScene.grasp`). GraspGen-X (§6.1)
is trained to complete partial views, which helps, but the screening in §6.1
still sees only the observed cloud.

### 6.1 Choosing the grasp: GraspGen-X, screened on the FR3 hand, verified by planning

The old grasp was a single scripted pose: straight down, across the object's
narrowest horizontal direction, 30 mm below its top. Objects that need anything
else failed, and the width check made it worse: it measured the whole cloud
along the closing axis, so a cup 90 mm across was refused even though its rim
is a 3 mm wall that the 80 mm hand closes on easily.

`tpgpt.hardware.grasp_selection.select_and_plan` replaces it in three steps.

**1. Generate.** The cloud goes to the GraspGen-X ZMQ server
(`ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh`, checkout at
`ycb_point_cloud/6DOF_MultiEmbod_Grasp/GraspGenX`, venv `~/venvs/graspgenx`, GPU)
for the gripper `franka_panda`. That model *is* the FR3's Franka Hand: its URDF
has the same finger pad boxes, the same 0.0584 m finger mount and 0.1034 m
fingertip as `franka_description`, and its published bbox agrees with the hand
model below to 3.5 mm (`tests/unit/test_grasp_selection.py`). A GraspGen pose
sits at the gripper base with `+Z` approach and `+X` closing, so the fingertip
point is `base + 0.1034 · approach`. Its half turn about the approach relative to
`fr3_hand` is irrelevant for a symmetric two-finger hand, and `scene_keypoints`
picks the roll that agrees with the demonstration anyway. About 100 grasps come
back with discriminator scores (a learned estimate that the grasp holds), in
0.4-0.8 s after a ~2 s warm-up.

**2. Screen on the cloud** with the hand's real collision geometry
(`tpgpt.hardware.hand`, every box from `franka_hand.xacro` and the collision mesh
of the hand body, hand fully open because that is how it arrives):

| verdict | meaning |
|---|---|
| `from_below` | approach more than `grasp_max_tilt_deg` (85°) from straight down: the hand would come up through the table (GraspGen-X does not see the table) |
| `too_wide` | cloud inside any part of an open finger (pad, diagonal link, mount), or more than 80 mm − 2·4 mm between the pads |
| `collides` | cloud inside the palm |
| `empty` | fewer than 8 cloud points between the pads: closing on air |
| `shallow` | the material between the pads spans less than 10 mm of the pads' 18.5 mm height along the approach: a pinch on an edge |
| `hits_support` | a hand box corner less than 3 mm above the support plane |
| `duplicate` | within 1 cm and 15° of a better-scored survivor |

Only the cloud **between the pads** decides the width. That region is the pad
face (17.5 mm across, 18.5 mm along the approach) swept over the 80 mm opening.

**3. Plan and verify.** Since 2026-09-21 this is the simulation's rule, and it is
described in full in `SHELF_AND_PATH_CHECK.md`. In brief: the sim's own funnel stages
(visibility, on target, collision with the shelf, place zone, centred) join the
checks above. Survivors are then planned best score first, up to `grasp_max_plans`
(100). The first whose **whole transported path** the real arm can fly is kept:
no collision with the shelf model or the observed cloud at any waypoint, at least
60 % of it reachable by IK, and the move onto the path is checked from the measured
joints. The earlier gates still apply: `min det(J)` ≥ `min_determinant`, aim
≤ 10 mm, orientation ≤ 15°, and hand clearance above the table and destination
planes (audit item A4). With nothing admissible, only a candidate that failed on
reach alone may run (`allow_kinematic_fallback`), never one that collides. The
scripted top-down grasp is opt-in (`grasp_fallback_top_down`, default false) and
passes the same checks. Every candidate and verdict is saved to
`<plan>/grasp_candidates.json`; `plan.json` records the stage tallies, the path
check and the shelf world.

Measurements of this step made before 2026-09-21 (real clouds on the old camera
setup) have been removed. They are not evidence for the current setup
(`FRESH_START.md`).

---

## 7. Execution

```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_executor_node --ros-args \
    -p plan_dir:=$HOME/tpgpt_plans/<plan> -p dry_run:=true
```

**Dry run (default).** Nothing is activated and nothing reaches the controller or
the gripper. The same loop runs against the first-order impedance arm of
`rollout_impedance`, starting at the real `/current_pose`. It publishes the
commanded and simulated poses and the executed path for RViz, and writes a CSV to
`<plan>/executions/`. Measured on the synthetic plan above at `speed_scale` 0.5:
**402 steps** (twice the 200 labels, as halving the speed should give), the jaws
closed with the simulated arm **2.0 mm** from the planned grasp point and opened
**2.2 mm** from the planned release point, the gate was shut on 0.7% of steps and
the clamp never fired.

**Hardware run** (`dry_run:=false`, supervised, robot container up):

1. load the plan; refuse placeholder extrinsics; check every label is inside
   `workspace_min/max`
2. fit the policy (0.2–0.5 s measured)
3. check the TCP is within `max_start_distance` of the plan start
4. load/configure the controller if needed; refuse if another node publishes on
   its target topic (CRISP ignores all targets then)
5. print a summary and ask for a typed `yes`
6. strict controller switch. CRISP seeds its target with the measured pose, so
   activation does not move the arm
7. open the gripper and wait for it
8. **approach**: cosine-profiled motion to the plan start at ≤ 5 cm/s and
   ≤ 0.3 rad/s, then settle within 15 mm
9. **policy**: every 50 ms `HardwareExecutor.step(measured TCP)` → target pose
   (+ stiffness) → gripper edge → CSV row
10. at the end, keep publishing the last target for 1 s. CRISP holds it
    afterwards.

**Stiffness option** (`use_transported_stiffness`):

* `false` (default): the stock `cartesian_impedance_controller` with its
  configured gains (400 N/m, 30 N·m/rad), poses only on `/target_pose`.
* **With a static taught demonstration keep it `false`**: its labels carry exactly the
  stock controller's 400 N/m and 40 N·s/m (§5.5), so there is nothing to transport
  but the map's distortion of one constant. `transit_rotational_scale` then
  resolves to 1.0 on its own (§5.5). **A directional one needs `true`**, or its
  stiffness never reaches the arm.
* `true`: `tpgpt_cartesian_impedance_controller`, loaded at runtime from
  `config/tpgpt_variable_stiffness_controller.yaml`, with private topics. CRISP
  applies stiffness to the pose error **in the end-effector frame**
  (`use_local_jacobian: true`), so the transported 3×3 base-frame `K` is rotated
  into the measured TCP frame, `R^T K R`, and sent every step in one of two forms
  (`stiffness_message`):
  * `full` (the default with this option, since 2026-09-29): all of `R^T K R`, a
    12-value message (the 3×3 row-major, then the rotational stiffness). Needs the
    patched CRISP (`~/THOR_DOCKER_SETUP`, `crisp-full-stiffness.patch`, in
    `docker-thor:local`); its principal values are clamped to 50–900 N/m, and CRISP
    damps it with `2 K^(1/2)`.
  * `diagonal`: its diagonal only (6 values, any CRISP), clamped to 50–900 N/m,
    damped with `2√k` per axis; the off-diagonal part is dropped. **Use this after
    rolling the image back to `docker-thor:pre-full-stiffness`**, which ignores a
    12-value message and keeps its yaml gains.

  Either way the `k_offdiag_share` column and the summary give the share of `K`
  off the hand-frame diagonal (applied by `full`, dropped by `diagonal`), and a
  dry run publishes what would be sent on `/tpgpt/executor/commanded_stiffness`;
  its simulated arm runs the gains CRISP applies for that message
  (`executor.crisp_applied_gains`). Rotational stiffness follows
  `rollout_policy`'s schedule: 60 at the ends of the task, 60 × 0.25 while
  carrying. Damping is CRISP's `2√K`, because CRISP cannot take the transported
  `D`. For the demonstration's 350 N/m that is 37.4 N·s/m against the
  demonstrated 33.7.

**Aborts** hold the arm at its measured pose. The gripper is left as it is, so a
held object is not dropped. They fire on: a pose older than 0.2 s; a commanded
pose outside the workspace box; approach tracking error over 80 mm; the
`StallWatchdog` declaring the arm stuck; `SafetyMonitor`; Ctrl-C.

### What differs from the simulated rollout, and why

* **The loop is the simulated one.** `HardwareExecutor` calls `gate_step`,
  `advance_step`, `advance_phase` and `StallWatchdog` in `rollout_policy`'s
  order. On the analytic arm it is **bit-for-bit identical** to
  `rollout_impedance`, with and without a load
  (`test_it_is_the_simulated_loop_bit_for_bit`, `test_it_matches_under_load_too`).
* **The infeasible-pose fallback runs on the FR3 model.** Its two tiers (relax
  the orientation, or steer to the next reachable label) ask
  `tpgpt.hardware.fr3.FR3.solve_ik`, seeded from the measured `/joint_states`, in
  place of MuJoCo's `solve_ik`. Without fresh joint angles it stays off for that
  step, and the completion log counts how often.
* **`speed_scale`** slows the whole task uniformly. Velocities, the speed clamp
  and the phase clock are all scaled, so the gate's expected lag
  (`‖K⁻¹D‖·v`) matches the speed actually commanded. At 1.0 it is the
  simulated loop exactly.
* **`SafetyMonitor` is added outside the loop, because the loop cannot see an
  arm that does not move.** Measured: `rollout_impedance` with
  `blocked_steps=(0, 10000)` (an arm that never moves) *completes the task*. The
  watchdog forgives the standing offset as sag once, the gate reopens, the clamp
  keeps the attractor 48 mm ahead of the motionless arm, and the phase (advanced
  from the attractor) reaches 1.0 after 253 steps, gripper schedule included. In
  MuJoCo a pulled arm always moves, so this never mattered. On a robot an
  inactive controller, a protective stop or a collision is exactly that case.
  `SafetyMonitor` aborts when the measured grasp point is more than 80 mm from
  its attractor, or when the arm travels less than 30% of its target's path over
  40 steps. Tested: it stops a motionless arm before phase 0.5, and stays silent
  for an arm that follows.
* **Gripper.** `franka_msgs` Grasp with `epsilon` covering 0–80 mm, because the
  cloud only gives a lower bound on the object's width and a 5 mm tolerance (the
  `GripperCommand` default) would report a successful grasp as failed. Closing is
  triggered by the policy's gripper channel crossing zero, as in
  `rollout_policy`.

---

## 8. What was validated, and how

| layer | what was run | result |
|---|---|---|
| TPGPT unit tests | `pytest tests/unit` | 552 pass. The 3 failures are pre-existing and need GraspGen-X assets at `/home/ishita/...` |
| hardware unit tests | `tests/unit/test_hardware.py` (31) | frames, bundle I/O, planner equivariance and aim, executor ≡ `rollout_impedance`, safety, approach, stiffness projection |
| calibration unit tests | `ycb_pointcloud_pipeline/test/test_camera_calibration.py` (16) | deprojection round trip, optical rotation = realsense rpy, Kabsch, RANSAC plane, YAML chain, point → pixel → point exact |
| ROS, synthetic camera | fake RealSense + fake `/current_pose` on an isolated domain (88) | table/centroid/TF numbers in §4; planner saved a plan (min det 0.406, aim 0.0 mm); dry run as in §7 |
| CRISP, fake hardware | `docker-thor:local` fake bringup, host network, domain 88 | loaded + configured `tpgpt_cartesian_impedance_controller` from the container path; strict switch both ways; CRISP logged `Variable stiffness received`; loop period median 50.0 ms, max 51 ms; since fake hardware does not move, the approach aborted at 81 mm and held, as designed; placeholder plan refused |
| teaching, conversion | synthetic 1 kHz recording of the simulated path (§5.8) | grasp 0.4 mm, destination 0.05 mm from the source; identity plan `min det` 1.000, aim 0.0 mm |
| teaching, ROS node | `tpgpt_teach_node` against a separate-process 1 kHz fake pose and fake camera, domain 89 | 998 Hz recorded; grasp/release labels 57/166 = source; placed cloud 1 mm; 427-step dry run, jaws 5.1 / 3.8 mm from plan |
| teaching, controllers | CRISP fake hardware, domain 88 | `tpgpt_teaching_controller` loaded and strictly switched both ways; SIGINT/SIGTERM while floating hold the arm; 250 Hz under load with worst gap 6 ms. The executor (`use_transported_stiffness:=true`) and `joint_target_publisher` on the shared controller-manager helper: load, configure, switch, then the designed abort and hold (82 mm approach error; 0.152 rad) |
| real robot | — | **not run** |

One bug was found only by the fake-hardware run. `pose_broadcaster` publishes at
the controller rate, and converting every `/current_pose` message to a matrix in
the callback starved the 20 Hz loop of Python's GIL for 1.8 s. The callbacks now
only store the message.

## 9. Open items

* **Teaching on the real arm (§5):** the first session should only float and hold
  (`t`, then `h`) to feel the zero-stiffness controller and check the payload in
  Desk. Also check once whether `/current_pose` streams during Franka guiding mode
  with FCI on (`teach_mode:=passive`).
* Execution lag with static stiffness is `D/K·v` = 10 mm at 10 cm/s (§5.5). If the
  measured grasp error says it matters, the next step is to record the attractor
  instead (teleoperation) or add `K⁻¹·D·v` to the taught path.
* Build `franka_msgs` on the host (THOR README) for the gripper, or run with
  `gripper_mode:=none` first.
* First real runs: `gripper_mode:=none`, `speed_scale` 0.3–0.5, no object; then
  an object; stiffness option last.
* `workspace_min/max` in the executor and planner are generic boxes. Collision with
  the shelf and the observed scene is now checked along the whole path
  (`SHELF_AND_PATH_CHECK.md`); the shelf model still needs measuring.
* The one-sided-cloud grasp bias (§6).
* A demonstration closer to the real task should widen the range of destinations
  the map supports (§6 table). Teaching one is now possible (§5). Whether it does is
  unmeasured: rerun the §6 destination sweep on a taught bundle.
