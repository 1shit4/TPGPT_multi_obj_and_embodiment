# The Inspire RH56DFX-2R on the FR3

The second hand of the hardware experiment: an **Inspire RH56DFX-2R, a right hand** (the box
says 2R; with the palm facing you, fingers up, the thumb is on your right). Rebuilt 2026-10-04
around GraspGen-X's grasps. The 2026-09-15 integration (a scripted side grasp, a left hand, an
estimated one-box envelope) is gone; `git show 0aceaec:ycb_pointcloud_pipeline/INSPIRE_HAND.md`
has its record.

**Every hand fact recorded on 2026-09-15 is unverified** (the user, 2026-10-05: that may
have been a different, left hand). Each is measured again here.

**State (2026-10-04):** code done and tested offline, including a full offline ROS dry run of a
grasp test. **Nothing has run on the robot**, and the mount, the register map, the grasp depth
and the close are not measured yet (section 4). Every deviation from the simulation is in
`docs/hardware/FAITHFULNESS_AUDIT.md`, section D.

## 1. What the pipeline does with this hand

- **Grasps** come from GraspGen-X's `inspire_hand` description, the same right hand, like the
  Franka Hand's come from `franka_panda`: a fixed library per object, carried onto the fitted
  scan (`grasp_library.py`).
- **The grasp frame** is GraspGen-X's: `+Z` the approach, the bisector of the palm normal and
  the finger direction (45 deg from each); `+X` closing, the thumb on `+X` and the four fingers
  on `-X`. The grasp point is `fingertip` (0.15 m) along `+Z` from GraspGen-X's pose origin,
  between thumb and fingers, about 37 mm in front of the palm.
- **The hand's model** (`tpgpt/hardware/inspire_model.py`) is GraspGen-X's URDF: one convex
  piece per link (a 26-DOP, 2.7-4.6 mm larger than the link), the URDF's exact palm boxes, and
  the adapter plate. Pieces are `body` (wrist, palm, knuckles, thumb base, adapter) or `finger`;
  only fingers may touch the object at the pick.
- **Screening** (`grasp_selection.judge_hand`) replaces the jaw check. A grasp is refused when:
  - the open hand, or the hand while closing, comes within 3 mm of the table;
  - a body piece is in the object;
  - an open finger is in the object;
  - fewer than 8 points lie between thumb and fingers;
  - closing each finger until it stops on the object, the thumb and a finger do not both
    reach it.
- **The plan** records where the fingers stop (`hand_closed_q`). That is the carried hand in the
  path check.
- **The roll**: the Inspire hand runs GraspGen-X's grasp as proposed. When its closing opposes
  the demonstration's, the demonstration's jaw frame is turned half a turn instead, which the
  Franka jaw allows. The map and path are the same; only the wrist differs
  (`plan.metrics["roll"]`, audit D3).
- **The executor** drives the plan's hand (`gripper_mode` auto) and closes with the object's
  force (`config/grasp_forces.yaml`, `inspire_grasp`, grams per finger). It keeps the hand's
  close report (what each finger touched) in `<log>_hand.json`, and the hand's joints in the
  full-rate record.

## 2. Frames and the mount

```
fr3_link8 (flange) --[mount: yaw, adapter thickness, lateral]--> hand_base_link (wrist face)
hand_base_link --[GraspGen-X URDF world_joint]--> W (GraspGen-X's pose frame)
W --[fingertip 0.15 - calibrated_depth along +Z]--> the grasp point (the plan's positions)
fr3_link8 --[Rz(-45 deg), 103.4 mm]--> fr3_hand_tcp (what CRISP commands; a virtual point now)
```

The mount, in `config/inspire_hand.yaml` `mount:`:

- **`yaw_deg`**: the palm normal is `Rz(yaw) * fr3_link8 +X`. At -45 the palm faces
  `fr3_hand_tcp` +X. At the home pose the hand then hangs fingers down, palm forward (robot +X),
  index finger on the robot's right (-Y).
- **`adapter_thickness_m`**: the flange face to the wrist face. The RH56 manual's 217.8 mm to the
  middle fingertip is measured from the wrist face, and the URDF gives 217.9.
- **`lateral_m`**: the wrist face's centre off the flange axis.

With the current values (yaw -45 deg and no lateral offset, estimates; the adapter 16 mm by
ruler) the grasp point sits (60.1, 10.0, 64.6) mm from `fr3_hand_tcp`: 168 mm out along the
flange's axis (152 of them the hand's own), against the Franka Hand's 103 mm.

**How it is held (user, 2026-10-05).** A 3D-printed mount is screwed to the flange, and beige
paper tape wraps the flange and the mount, so their seam cannot be seen: a ruler cannot give
`adapter_thickness_m` (the touches of M3 do, or the printed part's file). The hand's own 52 mm
base ring (clear tape) is screwed to the mount, but its holes have no thread, so the screws do not
grip; the tape holds it and it is **still slightly loose**. Play there turns into an error that
changes with the pose (the hand's weight takes it up a different way in each), magnified about
218 / 26 = 8 times from the ring's edge to the fingertips. M2b measures it; M3's touch residual
shows it too.

**Measured (M2b, user, 2026-10-05): the hand rocks about one line only** -- the fingers swing
side to side across the palm, like waving, about 5 mm at the fingertips end to end. That is
5 / 218 = 1.3 deg in all (+-0.66 deg). Most likely the two loose screws act as a hinge pin
and the hand turns about the line through them; it does not rock towards or away from the
palm, nor twist. What it costs:
- at the grasp point (152 mm along the hand from the ring) +-1.7 mm, **across the closing
  direction** (GraspGen-X's `y`, along the object between the pads): it moves a grasp along
  the object, not into it or off it. Small next to the camera's error (~10 mm sideways at the
  shelf) and the arm's (4-5 mm at the grasp);
- the hand's weight takes up the play whenever the palm's across-direction is not level, so
  where the hand sits within it depends on the pose; the model does not include it;
- in the touches (M3) the index tip can sit up to +-2.5 mm off between touches, so the table
  RMS can exceed 2 mm from the play alone: read it against 3 mm, and guide the arm by the
  robot, never by the hand.

The robot bring-up is unchanged (`load_gripper:=true`, `fr3_hand_tcp` kept). The Franka gripper
node fails without its hand, which is expected and harmless.

### What Inspire's own package says (`~/inspire_mount (2)/`, read 2026-10-05)

The vendor's package for the RH56DFQ/DFX/BFX (June 2025) holds three things: the hand's STEP
models (with and without the wrist module), its URDFs and SDK, and manuals. It describes the
hand only; **the adapter between the FR3 flange and the hand is not in it**, so the adapter's
thickness and the yaw still have to be measured (M2, M3).

**Mounting interface.**
- The hand mounts by a **Ø52 / Ø38 mm ring, 8.1 mm thick** (the 8-19 URDF's `base_link`: its
  bottom face is the base frame's origin), held by radial screws in its side.
- The manual's drawing gives 217.8 mm from that face to the middle fingertip, 80.7 mm palm
  width, and finger segments of 32.0 and 49.3 mm. GraspGen-X's URDF, which the model uses, puts
  its origin at the same face and agrees: 217.9 mm, a 32.1 mm proximal segment.

**The vendor's URDFs, against GraspGen-X's.** The hand on the robot has plain pads, not the
ripple ones (user, 2026-10-05), so of the vendor's files `urdf_right_2025_4_18` is its model;
`RH56DFX_R_8_19` is the ripple-pad version.

- **Geometry, compared by the finger meshes themselves** in the hand's base frame (all three share
  the base face and the palm outline, -38.7..42.8 mm across):
  - Across the palm the proximal segments agree to 0.4 mm (index -26.7, middle -6.9, ring 13.0,
    little 32.9 mm in all three).
  - Along the hand GraspGen-X's sit 1-2 mm nearer the wrist; normal to the palm, 3 mm further out.
  - The thumb base agrees to 2 mm.
  - The joint origins look 10-12 mm apart across the palm. That is only where each file puts them
    along the knuckle hinge, which runs across the palm, and means nothing.
  - **GraspGen-X's geometry stays the model**: within 3 mm of the manufacturer's, and the one its
    grasps are generated for.
- **Kinematics: none of the URDFs reproduces the manual's angle ranges.**

  | URDF | index finger at its limits | thumb bend at its limits |
  |---|---|---|
  | `urdf_right_2025_4_18` (0..1.333 rad, distal x1.117) | 163 -> 35 deg | -40 -> +7 deg |
  | `RH56DFX_R_8_19` | placeholder limits [0, 1], no coupling | -- |
  | the wrist-module URDF | its 1.7 rad limit matches the manual-derived 1.70 | -- |
  | the manual | 176.7 -> 19 deg | 53.6 -> -13 deg |

  The register map was first the manual's ranges solved on GraspGen-X's linkage; the ruler
  (M4, 2026-10-05, below) contradicted it, and the map is now the ruler's fit.

**The wrist module.** It is a 2-DOF wrist (pitch -22.7..+22.1 deg, yaw ±25.5 deg) between the arm
and the hand; its first joint is ~78 mm out from the mounting face, and its registers are
1020/1022 on the same serial line. **If the hand on the robot has it, the mount is not rigid**
and the model needs its two joints.

**The SDK** publishes the raw 0-1000 registers; it has no conversion to joint angles. The
register map (`joint_map`) stays ours to measure (M4).

## 3. Bring-up with this hand

Every terminal as `docs/hardware/commands.md` §1 (sourcing, venvs), then:

```bash
# once after every boot: the USB-RS485 adapter's driver (the kernel lacks it). It depends on the
# kernel's usbserial module, which insmod does not load: without it, "Unknown symbol in module".
sudo modprobe usbserial
sudo insmod ~/ch341/ch341.ko
ls /dev/ttyUSB0
```
```bash
# Th  the hand (one terminal for the whole session)
ros2 launch ycb_pointcloud_pipeline inspire_hand.launch.py
# open / close / pointing pose, by hand:
ros2 service call /inspire_hand/open std_srvs/srv/Trigger
ros2 service call /inspire_hand/close std_srvs/srv/Trigger     # answers the close report (JSON)
ros2 service call /inspire_hand/point std_srvs/srv/Trigger
# settable while it runs (read at each close):
ros2 param set /inspire_hand close_mode simultaneous            # or sequential
ros2 param set /inspire_hand force_g 300                        # grams per finger
```
Every pose change (open, point, `inspire_hand set`) moves the thumb and the fingers in turn, as a
fist does (the user, 2026-10-05: moved together, the thumb obstructs the fingers): closing, the
fingers first and the thumb last; opening, the thumb lifts first, then the fingers open. The
thumb's sideways swing (its **rotation**, the 6th value: 0 opposed, facing the fingers, as every
grasp uses; 1000 swung back flat beside the palm) moves only while the thumb is straight and the
fingers are open. Without the node (it holds the serial line):
`ros2 run ycb_pointcloud_pipeline inspire_hand set <little> <ring> <middle> <index> <thumb bend> <thumb rotation>`.
`free_drive` and `joint_homing` open this hand by themselves (`gripper_type` auto reads X6's
`hands.mounted: inspire`), so they need the hand node running; without it, or to keep the hand
as it is (the pointing pose for M3), pass `-p open_gripper:=false`.

The step-by-step measuring guide: `docs/hardware/INSPIRE_MEASURING.md`.

**Desk**: set the end effector to the Inspire hand plus adapter, with its mass and centre of mass
(M5). Without them, gravity compensation is wrong by the difference.

## 4. Measuring the hand (once, then whenever it is re-mounted)

Each measurement has a pass criterion. Results go into `config/inspire_hand.yaml` (then
`colcon build`) and `x6_experiment/config/hands/inspire.yaml` `measured:`.

| # | what | how | pass |
|---|---|---|---|
| M0 | The hand is the URDF | Caliper on the open hand, against `python -m tpgpt.hardware.inspire_model landmarks`: middle fingertip to wrist face (217.9), knuckle spacing index to little finger (57.8) | within 2 mm. **By ruler 2026-10-05:** middle tip 216 mm below the ring's face, palm 81 mm wide (the model 81.4): agrees |
| M1 | Yaw and handedness | At home (`joint_homing`): fingers down, palm forward, index finger on the robot's right | by eye; no 90 or 180 deg error |
| M2 | Adapter thickness | The flange-mount seam is under tape: from the printed part's file if there is one, else from M3's fit (the ruler gives only the beige band, flange part included) | within 1 mm; into `adapter_thickness_m`. **The printed part alone, by ruler 2026-10-05: 16 mm** (the estimate was 12); M3 refines it |
| M2b | Play at the ring | Rock the palm gently against the held flange; how far the middle fingertip moves (`INSPIRE_MEASURING.md` step 4) | under about 1 mm, or M3's residual and the grasp tests are read with it in mind. **2026-10-05: ~5 mm across the palm only** (section 2) |
| M3 | The mount, by touch | Below | table RMS within 2 mm, marks within 3 mm |
| M4 | Register to joint map | Ruler distances at registers 1000/750/500/250/0: the middle fingertip below the ring's face (thumb swung out), the thumb tip to the index tip (fingers open), against `python -m tpgpt.hardware.inspire_model landmarks --registers ...` (`middle_length`, `thumb_to_index`); 1 mm is about 0.5 deg | within 3 mm; else fit q_at_0/q_at_1000 into `joint_map`. **Done by ruler 2026-10-05** (section 7): fitted, still unverified |
| M5 | Payload | Weigh hand, adapter and screws; centre of mass; set in Desk | force estimate at rest under 1.5 N in 3 orientations |
| M6 | Grasp depth | Grasp-test sweep, below | the middle of the plateau into `graspgenx.calibrated_depth_m` |
| M7 | Close and force | Close on each object in both modes; read the reports | thumb plus a finger in contact; grams per object into `grasp_forces.yaml` |
| M8/M9 | Description and close mode | The same grasp tests with the shipped vs a measured description, sequential vs simultaneous | keep what holds more |

**M3, the mount by touch.** The hand in its pointing pose, the arm floating:
```bash
ros2 service call /inspire_hand/point std_srvs/srv/Trigger
ros2 run ycb_pointcloud_pipeline free_drive
```
```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline inspire_touch
```
Rest the index fingertip on the table and type `t`. Do it six times or more, each with the hand
tilted 15-30 deg in a different direction. Then touch each top slot mark: `p top_left`,
`p top_right`. `f` fits; the yaml block it prints goes into `config/inspire_hand.yaml`. Touches
made straight down cannot separate the yaw from the offset, and the fit says so.

## 5. Grasp tests (the grasp on its own, before any transport)

A grasp test plans one grasp and runs it: in along the approach from 10 cm back, close, lift 10 cm
and hold 2 s, set down, release, back out. The whole-path check judges it like any plan
(`tpgpt/hardware/grasp_test.py`). After a robot run the executor asks what happened (held,
dropped, not picked, pushed) and keeps it with the close report in `<log>_outcome.json`.

```bash
source ~/venvs/tpgpt/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p end_effector:=inspire -p task:=grasp_test -p object_name:=apple -p dry_run:=false
```

- `-p grasp_test_constructed:=true`: a grasp built by hand (straight down, closing across the
  object's narrow side), not GraspGen-X's. It checks the frame chain: the hand must arrive as
  drawn.
- `-p grasp_test_library_index:=N -p grasp_test_depth_offset:=0.01`: the same library grasp,
  1 cm deeper. For the depth sweep (M6): 6 depths 10 mm apart, 3 tries each. Never in X6.
- Dry runs (`dry_run` true, the default) simulate the motion. Offline, with no camera or robot,
  replay a recorded capture on another ROS domain:
  ```bash
  export ROS_DOMAIN_ID=88       # never 61
  ~/venvs/tpgpt/bin/python ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/scripts/fake_scene.py \
      ~/tpgpt_plans/apple_20261003_221827 --object apple
  ```

## 6. Transport

Once grasps hold, the same command without `task` (`transport`, the default) and with
`-p destination_name:=top_left` (or `top_right`) and the learned bundle
(`-p source_bundle_dir:=$HOME/tpgpt_demos/banana_20260925_183904/source_bundle_learned`), as for
the Franka Hand (`commands.md` §6).

## 7. What is known (offline 2026-10-04; the ruler 2026-10-05)

Measured on the 10-03 captures, with the estimated mount and GraspGen-X's shipped description
(`python -m tpgpt.hardware.coverage --hand inspire`, `docs/hardware/GRASP_LIBRARY.md`):

- **About half of GraspGen-X's Inspire grasps approach from below** (more than 90 deg from
  straight down) and are refused at once.
- **The table is the hand's main problem.** GraspGen-X sees the object, not the table, and its
  grasp depth puts the open fingers 15 mm and the closed ones 26 mm below the grasp point on a
  top-down grasp. Every banana grasp (37 mm tall) puts the hand into the table: 1008 of 1008,
  median 46 mm under. The foam brick loses 838 of 930 this way. The Franka Hand's selection
  lifts such grasps (`refine_grasp`); whether the Inspire hand should is open, and waits on the
  depth sweep (M6).
- **Reach is the transport's problem.** Of 10 planned per slot: apple 1 / 0, mug 0 / 3, the rest
  0 (top_left / top_right). The hand is 62 mm longer than the jaw, and the carry most often
  brings the wrist within 20 mm of full stretch.
- **The register map, by ruler (M4, 2026-10-05; `INSPIRE_MEASURING.md` step 3).** The readings
  carry about +-5 mm each (parallax, the ruler's tilt, where on the tip it was read; the hand's
  own shortfall, ANGLE_ACT a few units under the command, is ~0.3 deg), so the map stays
  `verified: false`. They still decide, because the manual-derived map missed them by up to 22 mm.

  | | 1000 | 750 | 500 | 250 | 0 |
  |---|---|---|---|---|---|
  | middle tip below the ring's face, read | 216 | 195 | 168 | 125 | 89 |
  | the manual's map | 218 | 196 | 149 | 103 | 84 |
  | the fitted map (fingers 0.032 -> 1.47) | 218 | 201 | 162 | 118 | 90 |
  | thumb to index, tip centres, read | 120 | 92 | 73 | 64 | -- |
  | the manual's map | 137 | 108 | 82 | 66 | -- |
  | the fitted map (thumb 0.12 -> 0.77) | 121 | 94 | 72 | 64 | -- |

  - **Fingers:** straight at 1000, as modelled (216 against 218). A straight-line map fits every
    reading within 7 mm with the full curl at 1.42-1.51 rad, depending on which point of the tip
    the ruler read; the URDF's own limit, **1.47**, is in that range and is used. The manual's
    1.68-1.77 is wrong for this hand: **GraspGen-X's `close` is the fingers' full curl.** The
    near-closed readings carry no angle (the tip's height barely changes there), so the full
    curl comes from the middle readings. Only the middle finger was read; the other three take
    its value (same mechanism, same URDF limit).
  - **Thumb:** **0.06-0.12 rad (3-7 deg) more bent than modelled at every setting**, however the
    reading is taken (at the very ends: 0.15 at 1000; 12-30 mm back from them, as read: 0.11-0.13;
    at the segments' middles: 0.08, a worse fit). Used: 0.12 open, 0.77 closed (the span, 0.65,
    is near the URDF's 0.6). **When open, the thumb's tip is 17 mm nearer the grasp axis than the
    manual's map had it** (41 mm from it instead of 58; GraspGen-X's own open, pitch 0, is
    straighter still): the open hand is narrower than the generator thinks. A 70 mm ball centred
    on the grasp point has 6 mm to spare on the thumb's side (the old model gave 23).
  - **Thumb rotation:** not measured (the URDF's full 1.308 rad span, the manual's 75 deg).
  - What the map changes: the planner uses only each DOF's two ends (the open hand and the
    farthest close); the middle of the map only turns the hand's reported registers into angles
    (`/inspire_hand/joint_states`, the logs). The coverage numbers above predate it.
- **The simulation never got this hand to hold anything**, but its model barely moved
  (1.5 mm of finger travel). That says nothing about the real hand; the grasp tests are the
  first evidence.
