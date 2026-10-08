# The Inspire RH56DFX-2R on the FR3

The second hand of the hardware experiment: an **Inspire RH56DFX-2R, a right hand** (the box
says 2R; with the palm facing you, fingers up, the thumb is on your right). Rebuilt 2026-10-04
around GraspGen-X's grasps. The 2026-09-15 integration (a scripted side grasp, a left hand, an
estimated one-box envelope) is gone; `git show 0aceaec:ycb_pointcloud_pipeline/INSPIRE_HAND.md`
has its record.

**Every hand fact recorded on 2026-09-15 is unverified** (the user, 2026-10-05: that may
have been a different, left hand). Each is measured again here.

**State (2026-10-06):** the mount, the register map, the close and the payload are measured
(section 4). **The first constructed grasp test held the foam brick** (18:27: thumb, index and
middle on it, held through the lift; both arrival holds ended 5.7-5.8 mm from their marks, 4 mm
too deep). The grasp depth (M6) and the description (M8) are not measured yet. Every deviation from the simulation is in
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

- **`yaw_deg`**: the palm normal is `Rz(yaw) * fr3_link8 +X`. **135** (M1, seen at home by the
  user, 2026-10-05: from behind the robot, "the palm faces me, fingers down, thumb on my left"):
  the palm faces `fr3_hand_tcp` -X, so at home the hand hangs fingers down, the palm towards
  the robot's base (-X), the thumb and index finger on the robot's left (+Y). The first
  estimate, -45, was half a turn off: every grasp would have been executed with the hand
  turned 180 deg about its own axis.
- **`adapter_thickness_m`**: the flange face to the wrist face. The RH56 manual's 217.8 mm to the
  middle fingertip is measured from the wrist face, and the URDF gives 217.9.
- **`lateral_m`**: the wrist face's centre off the flange axis.

With the touched values (yaw 136.7 deg, the adapter 11.0 mm, the ring 10.6 mm off-centre; M3)
the grasp point sits (-70.4, -11.3, 59.6) mm from `fr3_hand_tcp`: 168 mm out along the
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
ros2 param set /inspire_hand close_mode simultaneous            # the default; sequential pushes the object
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
| M1 | Yaw and handedness | At home (`joint_homing`): fingers down; which way the palm faces; the thumb's side | by eye; no 90 or 180 deg error. **2026-10-05: palm towards the base, thumb on the robot's left: yaw 135, not the estimated -45** (a right hand, as the box says) |
| M2 | Adapter thickness | The flange-mount seam is under tape: from the printed part's file if there is one, else from M3's fit (the ruler gives only the beige band, flange part included) | within 1 mm; into `adapter_thickness_m`. **The printed part alone, by ruler 2026-10-05: 16 mm** (the estimate was 12); M3 refines it |
| M2b | Play at the ring | Rock the palm gently against the held flange; how far the middle fingertip moves (`INSPIRE_MEASURING.md` step 4) | under about 1 mm, or M3's residual and the grasp tests are read with it in mind. **2026-10-05: ~5 mm across the palm only** (section 2) |
| M3 | The mount, by touch | Below | table RMS within 2 mm, marks within 3 mm (the ring's play adds up to ~2.5 mm). **Done 2026-10-05, both fingers, 15 touches, the ring concentric: yaw 137.1 +- 1.4 deg, thickness 12.45 +- 0.5 mm, the fingers bent 5.2 deg at 1000; rms table 1.1, marks 2.1 mm: passes** |
| M4 | Register to joint map | Ruler distances at registers 1000/750/500/250/0: the middle fingertip below the ring's face (thumb swung out), the thumb tip to the index tip (fingers open), against `python -m tpgpt.hardware.inspire_model landmarks --registers ...` (`middle_length`, `thumb_to_index`); 1 mm is about 0.5 deg | within 3 mm; else fit q_at_0/q_at_1000 into `joint_map`. **Done by ruler 2026-10-05** (section 7): fitted, still unverified |
| M5 | Payload | The robot weighs it: `payload_measure` holds still in 18 poses around home and fits the mass and centre of mass to its external joint torques, with a constant offset per joint (`tpgpt/hardware/payload.py`); the inertia from the manufacturer's URDF. Expected from the manufacturer's data: 0.568 kg at (-7, 0, 88) mm in the flange frame (hand 533 g, the mount ~35 g guessed); Desk holds the Franka Hand's 0.73 kg at 30 mm | the fit's mass within ~0.05 kg of the manufacturer's; after setting Desk, a second run finds what Desk holds. **First run 2026-10-05 (`payload_20261005_223401.json`): static friction dominated** -- under the soft joint controller every joint stops 0.05-0.08 rad short on the side it came from, and up to 0.6 N m is held and read as external; fitted with a friction term signed by each joint's approach the residuals drop from 0.13-0.53 to 0.02-0.07 N m, and the payload is **0.708 kg at (-9, 1, 55) mm**. The sign: pushing the hand up made the robot's Fz go down (`wrench_watch`, -0.43 -> -2.38 N), so its external torques and wrench are the force it applies, not the push on it; the other sign (0.752 kg at 7 mm) was impossible anyway. 0.14 kg more than the manufacturer's data, and nearer the flange: the hand's cable is taped along the last link. Desk: 0.708 kg, (-0.0092, 0.0014, 0.0548) m, the manufacturer's inertia -- **set by the user 2026-10-05**, and read back from the robot (robot_state inertia_ee). **Re-weighed with it in Desk** (`payload_20261005_233719.json`): 0.658 +- 0.038 kg at 73 +- 10 mm; both runs fitted together (an offset per session): 0.692 kg at (-7, -2, 62) mm. The leftovers show the limit: joint 2 follows the arm's own pose (the robot's model of its arm, -0.4..+0.6 N m) and joint 5 jumps once (friction a single +-f misses), so the real precision is about +-0.04 kg and +-15 mm. Desk is within it (16 g, 7 mm: ~0.2 N, 0.1 N m, below the joints' friction): **done, Desk kept** |
| M6 | Grasp depth | Grasp-test sweep, below | the middle of the plateau into `graspgenx.calibrated_depth_m` |
| M7 | Close and force | Close on each object in both modes; read the reports | thumb plus a finger in contact; grams per object into `grasp_forces.yaml`. **Done on the bench 2026-10-05** (section 7): 400 g, the pudding box 150 |
| M8/M9 | Description and close mode | The same grasp tests with the shipped vs a measured description, sequential vs simultaneous | keep what holds more. **Close mode decided on the bench 2026-10-05: simultaneous** (sequential pushes the object); the description is still to compare |

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
`p top_right`. Then the same table touches with the little finger
(`ros2 service call /inspire_hand/point_little std_srvs/srv/Trigger`; the recorder reads which
finger from the hand's registers). `f` fits; the yaml block it prints goes into
`config/inspire_hand.yaml`.

**Result (2026-10-05).** 15 touches: 8 with the index fingertip
(`touches_20261005_212605.json`), 7 with the little one (`touches_20261005_214936.json`).
Fitted with all four numbers free they say yaw 136.7 deg, thickness 11.0 mm and the ring
**10.6 mm off-centre** (rms 1.6 mm). But the ring is concentric with the flange (the user, who
looked: the beige band, the flange and the ring line up), and in the manufacturer's mesh the
ring is centred on the hand's axis to 0.0 mm. So the 10.6 mm is not in the mount -- and it lies
entirely **along the palm's normal** (+10.6 mm towards the palm, -0.8 across it): both
fingertips sit 10.6 mm further towards the palm than the model puts them. Fingers that at
register 1000 are bent a little towards the palm do exactly that. Fitted that way --
`python -m tpgpt.hardware.inspire_mount fit <both files> --finger-bend` (offset held at 0):

| | value | |
|---|---|---|
| yaw | 137.1 +- 1.4 deg | the user saw ~135 at home |
| thickness | 12.45 +- 0.5 mm | the printed part alone: 16 by ruler |
| the fingers' bend at 1000 | 5.2 +- 0.45 deg (0.091 rad) towards the palm | index alone 3.9, little 7.2 (+-1; they trade off against the yaw) |
| rms | 1.7 mm | table 1.1, slot marks 2.1 (the ring's play: +-2.5 at the tips) |

The bend explains two ruler readings too: the middle finger read 216 mm at 1000, the model with
the bend gives 216.4 (straight, 218.0); and part of the thumb's "extra bend" was the index's (the
thumb's open end refits to 0.082 rad, from 0.12). In the config: the mount above, every finger's
`q_at_1000` + 0.091, the ruler maps refitted with it. The grasp point is (-59.7, -12.2, 61.1) mm
from `fr3_hand_tcp`, within 4 mm of the eye's estimate.

**The open hand is 27 mm narrower than GraspGen-X's model**: at register 1000 the thumb's tip is
48 mm from the grasp axis and the middle fingertip 76 mm on the other side (GraspGen-X's open,
every joint at 0: 63 and 89). GraspGen-X samples grasps for the wider hand; `judge_hand`
checks them on the measured one.

**Why both fingers.** Touches with one fingertip only say where that one point is relative to
the flange: three numbers, however the wrist is tilted. The mount has four, and turning the hand
about the flange axis or shifting it sideways both just move that point. The first live fit
(index only, 8 touches) showed it: yaw +-405 deg, offset +-160 mm, thickness +-0.8 mm. Each
finger alone, held centred, gave 116 and 151 deg. Two fingers 68 mm apart across the palm
separate the yaw from anything that moves both tips alike -- an offset, or a bend. They cannot
separate an offset along the palm's normal from a bend towards the palm: only knowing the
ring is concentric did. With one finger's touches the fit holds the offset (at the config's)
and says so (`--free` frees it).

**The thickness, 12.45 mm, against the ruler's 16.** The hand sits ~3.5 mm deeper than the
printed part's length (a rim, or the ring in a recess), or the model's fingers are a little
long. For the fingertips the touched value is right either way; for the palm it is within those
3.5 mm.

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

- `-p grasp_test_constructed:=true`: a grasp built by hand (straight down), not GraspGen-X's.
  It checks the frame chain: the hand must arrive as drawn. It closes across the object's narrow
  side, else its long side, each with the thumb on either side, and takes the way whose start
  posture is nearest the arm. For this hand the two thumb sides are different arm postures: with
  the mount's yaw (137 deg), tilting only the wrist 45 deg from home makes a clean top-down grasp
  with the **thumb towards the robot**; the thumb away from it needs joint 7 past its range, and
  on the live brick of 2026-10-06 the arm then contorted (shoulder +1.7 rad, links 5 and 7 1.9 mm
  into each other) and the plan was refused. The planned postures can look far from home in
  joints 1 and 3 together: the elbow swivels (the arm has a seventh joint), not a contortion.
- **It is executed in follow mode** (`HOW_EXECUTION_WORKS.md`, "Which plan is executed"): the
  test's labels are its own arm path, and it holds at the grasp and the set-down before the
  hand acts. Plans made before 2026-10-06 18:00 have no arm path: the simulation's clamp then
  holds the command within ~6 mm of the arm, and on the first brick test the arm never left the
  pre-grasp (the static friction held it against a 2.4 N pull). Re-plan.
- **At the transport's settings** (2026-10-06): the stiffness of a taught demonstration (350 N/m
  along the motion, 600 across and when still) and the source bundle's execution settings, as
  the Franka Hand's transport runs. The settings come from the planner's source bundle, which
  defaults to the learned one (`source_bundle_dir`); a bundle without them is refused. Plans
  made before 2026-10-06 19:10 ran at 400 N/m with offset compensation: at 18:07 and 18:27 all
  four arrival holds ran out of push (5.4-20.7 mm off the mark); at 18:57 (the planner's old
  default bundle, so still 400 N/m) both reached 3.8 mm in 2.1 s, the push at 24-29 mm of its
  30.
- `-p grasp_test_library_index:=N -p grasp_test_depth_offset:=0.01`: the same library grasp,
  1 cm deeper (negative: further back). For the depth sweep (M6). Never in X6.
- **The depth sweep (M6), and holding at the fingertips** (2026-10-06). The user's view: every
  object should be held between the fingertips -- thumb, index, middle, sometimes ring and
  little -- the closest the hand comes to the two-finger grasps of the demonstration, and the
  banana must be held too. Counted offline on each object's 2000 GraspGen-X `inspire_hand`
  grasps (the object on the table in its protocol pose, the grasp test's screening, then the
  modelled hand closed and the segments touching the object noted):

  | hand moved back along its approach | brick: pass / fingertips only | banana: pass (all fingertips only) |
  |---|---|---|
  | 0 (GraspGen-X's own depth) | 37 / 6 | 0 (992 from below, 1008 into the table) |
  | 10 mm | 82 / 49 | 1 |
  | 20 mm | 101 / 84 | 6 |
  | 30 mm | 85 / 82 | 14 |

  At GraspGen-X's own depth the apple, mug and pudding box pass 107 / 21 / 58, of which 18 /
  14 / 35 hold with fingertips only. GraspGen-X's grasp point sits between thumb and fingers,
  ~15 mm short of the open fingertips (more when closing): holding at the fingertips is the hand
  stopping 20-30 mm further back, which is what the sweep measures -- one depth for the hand
  (`calibrated_depth_m`, the simulation's rule), not a rule per object. The plan: brick, library
  grasp **#292** (score 0.87; it passes at 0-30 mm back whatever the brick's heading, fingertips
  only at 20 and 30; next 670, 928, 1191, 1297, 1556 if the arm cannot reach it), at 0, -10,
  -20, -30 mm (and -40 if -30 holds), 3 tries each, the brick put back on the same mark; then
  apple, mug, pudding box and banana at the chosen depth and +-10 mm, 2 tries each. The table:
  `~/venvs/tpgpt/bin/python -m tpgpt.hardware.grasp_test summary --object foam_brick --since
  <date_time>` (the depth asked for, where the hand really closed along and across the
  approach, held, the digits on the object).
- **Phase 1, the brick (2026-10-06 22:20 - 10-07 00:40), #292** (approach 19 deg off vertical,
  closing along the brick's length): held 3/3 at 0, 3/3 at -20, 3/3 at -30 (+1 held but aborted,
  below), 1/1 at -10 (#670), 0/1 at -40 (only the index reached it). The hand closed 0.5-1.9 mm
  short of the depth asked for and 2.6-3.8 mm across. **The band that holds is 0 to -30 mm.**
  What went wrong, and why:
  - **A pinned library index is not a fixed grasp on the brick.** Its four long faces look
    alike to the camera, and the fit put it on rest 0, 2 or 5 from capture to capture; #292
    belongs to the scan, so on rest 2 or 5 it is a grasp from below (109 deg) or through the
    table, and the plan was refused (11 of 24 plans). The fit cannot tell the faces apart;
    which it picks is close to chance. Normal selection is unaffected (it screens every grasp
    on the fitted pose); only a pinned sweep is.
  - **One fit turned 35 deg from the camera's top face** (00:23, heading 68 against 183-199,
    0.76 explained against 0.80-0.89): the hint filters the starts, not the refined fit. #292
    then landed on the real brick differently ("a completely different grasp"). Open.
  - **One fit refused** (23:54): the top read +23.0 mm against the 22 allowed; now 26 (all 56
    brick captures: median +14). The planner then fell back to a fresh draw on the camera cloud,
    where #292 does not exist ("no candidate 292 in 200"); a pinned grasp now says why.
  - **One held close aborted** (00:06): the ring still creeping into the foam at the timeout
    (435, 51 g, below its force stop); a close now fails on that only when nothing is held.
- **Phase 2, the other objects (2026-10-07 01:28-03:01), normal selection: all 19 runs held.**
  Apple 2/2 at -10, 1/1 at -20, 1/1 at -30; mug 1/1, 3/3, 1/1; pudding box 1/1 each; banana
  1/1, 4/4, 2/2. With the brick's 0 to -30, **every object holds from 10 to 30 mm back**; the
  middle of the band they share is **-20 mm**. The hand closed within 2.7 mm of the depth asked.
  What the runs showed:
  - **The hand goes 4-13 mm past the grasp mark on the way down** (median ~6; pudding box
    10-12), then the hold lifts it back before the close; the Franka's transport runs went at
    most 7 mm past, mostly 1-4 and in the hold. The cause, as predicted: a grasp test's
    command has no learned lead, so the push that moves the arm down is stored in the follow
    correction (350 N/m along the motion) and carries the hand on when it stops and stiffens
    to 600. On the banana the planner left the hand 4-5 mm above the table at the grasp (its
    minimum is 3), so the thumb met the table before the close (the user saw it, more with the
    thumb on the inner curve and a steeper tilt: #284, 12 deg). Open; an execution change, the
    user's call.
  - **The apple was not detected without a mug in view**: the segmentation node ran
    `caption_mode: joint` until the 02:30 restart (the saved frames say so; the mode is read only
    at start), and in one caption "red apple" and "red cup with a handle" both matched the apple
    ("red apple red cup", refused as ambiguous: 167 of 199 frames missed, best 0.59). With the mug
    in view "red cup" went to it and the apple was published on 143 of 177. Launch with
    `caption_mode:=per_object` and check: `ros2 param get /ycb_segmentation_node caption_mode`.
  - **One apple plan was refused with the arm at home** (01:35, `approach_blocked`: link 5 against
    link 1 on the planned move onto grasp #19's path, 3 at the postures, 29 between them); the
    next capture planned the same grasp. Re-homing, which the message advises, cannot help a
    homed arm.
  - **Decided (the user, 2026-10-07): `calibrated_depth_m` -0.02**, the middle of the shared
    band; `grasp_test_depth_offset` now adds to it. The overshoot is left for now, to be measured
    on the first transport runs. With the arm at home, a grasp whose route is blocked is passed
    over instead of stopping the plan.
  - **The pudding box at -10 and -20 looked alike** because they were different grasps (#187,
    #89): normal selection takes the first testable grasp at each depth. They closed 9.4 and
    21.4 mm back along their own approaches.
- Dry runs (`dry_run` true, the default) simulate the motion. Offline, with no camera or robot,
  replay a recorded capture on another ROS domain:
  ```bash
  export ROS_DOMAIN_ID=88       # never 61
  ~/venvs/tpgpt/bin/python ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/ycb_pointcloud_pipeline/scripts/fake_scene.py \
      ~/tpgpt_plans/apple_20261003_221827 --object apple
  ```

## 6. Transport

**The first Inspire transport: apple -> top_left, 2026-10-07 19:05** (after the camera was
knocked and recalibrated, d18d450; `~/tpgpt_plans/apple_20261007_190234`). Planned on the 11th
grasp tried (132 s, min det 0.61); the scene clean under the new calibration (98 % of the
camera's shelf points explained by the shelf model, 86 unexplained, 788 removed as the robot).
The move onto the path in joints took 10 s; the apple was held (middle, index) and set down.
Beside the arm path median 3.5 mm, at most 25.5 (correction up to 36 mm). The hand went 6.9 mm
past the grasp mark before the hold (grasp tests 4-13, the Franka's transports at most 7: the
learned command carries the lead); the grasp hold ran its 4 s, 15.4 -> 5.2 mm from the mark,
the release hold 1.4 s to 3.8 mm.

**The wrist vibrated during the carry** (the user felt it). In the 250 Hz pose log: the hand's
8-60 Hz motion up to 0.47 mm RMS at the TCP (0.015 typical), around labels 300-337 with the
apple in the hand (release at 393), strongest near 7 Hz; the hand turning 0.6-0.8 deg RMS (4-30
Hz) against 0.004 when quiet; **joints 5 and 7 swinging 0.30 and 0.49 rad/s RMS** (4-15 Hz)
against 0.004-0.005. There joint 6 was 3.11 rad: **the axes of joints 5 and 7 1.9 deg apart, the
wrist's singular posture** (joint 6 = pi; the run passed within 0.0 deg). There, joints 5 and 7
turning against each other barely move the hand, so the Cartesian controller holds that motion
only through its weak null-space posture spring (2 N m/rad). **The singularity alone does not
make it shake**: two of the Franka's 19 transports passed through it too (0.0 deg) and a third
within 6, with joints 5+7 wobbling 0.002-0.007 rad/s near it. The difference is the hand's
inertia on that weakly held motion -- 0.71 kg, off the flange axis and 168 mm out, on a taped
mount that rocks -- against the Franka Hand's compact 0.73 kg at 30 mm. Left as is for now (small:
sub-mm at the TCP); if it grows, the planner could keep the axes of joints 5 and 7 apart along
the path, as it keeps the wrist inside full stretch (the user's call).

**The mug dropped in the carry: mug -> top_right, 2026-10-07 21:29**
(`~/tpgpt_plans/mug_20261007_212936`; the user: "while grasping the other fingers pushed and
displaced the mug"). The plan took the 22nd grasp (the 21 before it failed the path check into
top_right), tilted 54 deg: the thumb on the outside wall 72 mm up, the index pushing up from
36 mm on the far side, a squeeze 44 deg off horizontal against a vertical wall. The two mug runs
into top_left that night (19:54, 21:33) took a grasp tilted 12 deg, squeezing 10 deg off
horizontal. The plan expected ring, middle, index and thumb on the mug; the close report had
index and thumb only (the middle closed through where the mug had been, the ring on air). The
arm was pushed 11 mm sideways in the 1.7 s after the close command. The run was killed in the
carry: its log ends mid-line and the full-rate record was never written (stop a run with one
Ctrl-C and wait for "Full-rate record ...").

What set it apart, over the 45 Inspire grasps run by then (`FAITHFULNESS_AUDIT.md` D13, D14):
- **Not the grip angle as the plan can know it.** The angle between the squeeze and the surface
  at the predicted contacts is 40 deg for this mug, 48 for two banana grasps that held, 36 for
  four bricks that held: the predicted contact on a curved object is good to a few mm only. It is
  recorded on every candidate (`grip_angle_deg`), not used.
- **The fingers leading the thumb.** Closing at one speed, a finger covers 1.0-1.2 of its travel
  per second and the thumb 0.8 (the full-rate hand records); this grasp's fingers were predicted
  to meet the mug at 25-30 % of their travel and the thumb at 58 %: the fingers pushed it for
  ~0.45 s before the thumb arrived. No other grasp of 43 had the fingers ahead by more than 0.15.
  **A timed close** (`-p inspire_timed_close:=true` on the executor; off by default) slows each
  finger so that all the plan expects on the object arrive with the thumb. Not yet run.

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
- **The register map (M4 by ruler, M3 by touch; 2026-10-05; `INSPIRE_MEASURING.md` step 3).**
  The ruler readings carry about +-5 mm each (parallax, the ruler's tilt, where on the tip it
  was read; the hand's own shortfall, ANGLE_ACT a few units under the command, is ~0.3 deg), so
  the map stays `verified: false`. They still decide, because the manual-derived map missed them
  by up to 22 mm. The fingers' open end comes from the mount touches (section 4: 5.2 deg towards
  the palm at 1000); the rest is fitted to the ruler with it.

  | | 1000 | 750 | 500 | 250 | 0 |
  |---|---|---|---|---|---|
  | middle tip below the ring's face, read | 216 | 195 | 168 | 125 | 89 |
  | the manual's map | 218 | 196 | 149 | 103 | 84 |
  | the measured map (fingers 0.123 -> 1.373) | 216 | 197 | 162 | 124 | 95 |
  | thumb to index, tip centres, read | 120 | 92 | 73 | 64 | -- |
  | the manual's map | 137 | 108 | 82 | 66 | -- |
  | the measured map (thumb 0.082 -> 0.650) | 119 | 94 | 73 | 61 | -- |

  - **Fingers:** at 1000, 0.12-0.13 rad (5 deg plus the manual's 2) short of straight -- the
    touches, and the ruler's 216 against a straight 218 agrees. Full curl **1.373 rad**, from the
    middle readings (the near-closed ones carry no angle: the tip's height barely changes there);
    within 6 mm of every reading. That is short of the URDF's limit, 1.47: **GraspGen-X's `close`
    curls further than the real fingers.** The manual's 1.68-1.77 is wrong for this hand. Only the
    middle finger was read; the other three take its value (same mechanism).
  - **Thumb:** 0.082 rad when open (the manual's 0.028), 0.650 closed (0.690); within 3.3 mm of
    the readings. Before the fingers' bend was known it fitted 0.12 -> 0.77: part of the thumb's
    apparent bend was the index's.
  - **Thumb rotation:** not measured (the URDF's full 1.308 rad span, the manual's 75 deg).
  - What the map changes: the planner uses only each DOF's two ends (the open hand and the
    farthest close); the middle of the map only turns the hand's reported registers into angles
    (`/inspire_hand/joint_states`, the logs). The coverage numbers above predate it.
- **Closing on the five objects, by hand on the bench (M7/M9, the user, 2026-10-05).** Each
  object held between thumb and fingers, then `close`:

  | object | force | held | where the DOFs stopped (little ring middle index thumb; 0 = full curl) |
  |---|---|---|---|
  | banana, in the palm | 400 g | yes | 220-245, 191-244, 219-244, 369-406, 499-508 (three closes) |
  | banana, by the fingertips (as from a table) | 400 g | yes | 0 (air), 537, 585, 588, 698 |
  | apple | 400 g | yes | 545-586, 671-745, 687-769, 617-705, 795-826 |
  | mug | 400 g | yes | 0, 0 (air), 425, 411, 542 |
  | foam brick | 400 g | yes, but it turns about the contact points at any force | 0 (air), 517-519, 621-667, 659-716, 731-772 |
  | pudding box | 150 g | yes | 0-268, 394-430, 474-478, 410-516, 572-641 |

  - **`simultaneous` is the close** (default since then). `sequential` pushes every object
    away: the fingers close with nothing opposing them until the thumb comes.
  - **The hand reports a force stop as STATUS 2** ("reached the target"), not the manual's 3:
    every DOF of every close above read 2, stopped 190-830 units short of 0. The first close
    reports therefore called every grasp "closed on air". Contact is now judged by where a DOF
    stopped (more than 30 units short of 0, `inspire_hand.CONTACT_MARGIN`); the force reading
    after a stop cannot tell either (it read -77 to 785 g on stopped fingers at a 400 g
    threshold). Without the fix the grasp tests' `abort_on_empty_close` would have aborted
    every grasp.
  - **The release opens in one stage** after a close that held something: the open had gone
    in two (the thumb lifted first), which leaves the fingers pressing the object alone at the
    place (the user). From a fist closed on air it is still the thumb first.
  - The thumb's rotation actuator read 590-934 g while holding the apple: the apple pushes the
    thumb sideways. The little finger closed on air on four of the five objects: the thumb,
    index and middle fingers do the holding.

- **An empty close is an "opposed" fist** (2026-10-05): [0, 0, 180, 361, 493] -- little and ring
  curl fully, the thumb lands on the middle and index fingers and stops them. So "the thumb and
  a finger stopped" does not mean an object. **Held** now means a finger stopped more than 30
  units beyond those angles (`grasp.empty_close_angles`): every hold that day had one 190-590
  beyond (the mug's middle finger 398-425 against 180; the banana's little finger 206-245
  against 0). The thumb is left out: in a palm grasp it closes over the fingers as in the fist.
  The model puts its thumb 5.5 mm into the index finger at the empty fist's angles -- the
  ruler map's accuracy. **The empty fist is not repeatable**: at 150 g it stopped [0, 2, 140,
  386, 526] once and [0, 0, 349, 361, 491] once -- the second reads as held. The reference is
  kept at the 400 g fist so no real grasp reads as empty (the mug's middle finger, 398-425, is
  the closest); `held` therefore errs towards "held". Its uses tolerate that: the release checks
  that every DOF opened and otherwise opens again in order (that 150 g fist's one-stage open
  left the fingers under the thumb), and `abort_on_empty_close` only saves time -- the trial's
  outcome is what the operator records. **`held` is a run-time check of the real close, not a
  grasp-selection filter**: selection judges a grasp before the robot moves, on the hand model
  and the cloud (`judge_hand`).

- **The simulation never got this hand to hold anything**, but its model barely moved
  (1.5 mm of finger travel). That says nothing about the real hand; the grasp tests are the
  first evidence.
