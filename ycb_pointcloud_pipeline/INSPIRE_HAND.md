# Inspire RH56 left hand on the FR3

> **Status 2026-10-04 -- being replaced.** The hand on the robot is an **RH56DFX-2R, a right
> hand** (the box, and the thumb is on your right with the palm facing you, fingers up); this
> document's "left hand" is wrong. The scripted side grasp, the one-box envelope and the
> estimated mount below are being replaced by GraspGen-X grasps on a measured model of the hand
> and a measured mount. Until the rewrite, read this file as the 2026-09-15 record only.

What was built (2026-09-15), how to mount the hand, where the gripper is switched,
and how to run teaching, planning, execution and X6 trials with it. **None of this
has run on the robot yet.** Only the hand's open/close grasp was tested on the real
hand, detached from the arm.

## 1. Mounting the hand on joint 7

The FR3's last link ends in a flange. Its frame, `fr3_link8`, has **+Z pointing out of
the flange face, along the joint-7 axis**. Its axes are joint 7's, moved 0.107 m along
that axis. The Franka Hand was bolted on turned **-45° about +Z**. Its fingertip
point `fr3_hand_tcp` sits **103.4 mm** out along +Z (`franka_description` `fr3.urdf.xacro`:
`rpy_ee="0 0 -pi/4"`, `tcp_xyz="0 0 0.1034"`).

Mount the Inspire hand like this:

| What | Setting |
|---|---|
| Hand axis (wrist → fingertips) | **coaxial with joint 7**, pointing away from the flange |
| Rotation about joint 7 | **palm faces `fr3_hand_tcp` +X**, i.e. `fr3_link8` +X turned -45° about +Z (`mount.yaw_deg: -45`) |
| Physical check, Franka home pose (`ros2 run ycb_pointcloud_pipeline joint_homing -p open_gripper:=false`) | fingers point **down**, palm faces **forward, away from the robot base** (base +X), thumb towards the **robot's left** (base +Y) |
| RViz check | TF display, frame `fr3_hand_tcp`: the palm faces the **red** arrow and the fingers run along the **blue** one |
| Adapter plate | robot side: the FR3 flange pattern. The Franka Hand's flange is ISO 9409-1-31.5-4-M5 (4 × M5 on a Ø31.5 mm circle, one Ø6 mm dowel); **check this against the FR3 datasheet before machining**. Hand side: the RH56 wrist base (Ø52 mm in the manual's drawing; take the hole pattern from Inspire's mounting drawing) |
| Cable | the hand's GX12 cable (24 V + RS485) runs along the arm to the CH340 adapter; leave slack for ±170° of joint 7 |

**Why -45°.** It is the Franka Hand's own orientation. With it, `fr3_hand_tcp`'s axes
line up with the palm: +X = palm normal, +Z = along the fingers. That is the rotation
the code assumes by default. Any other yaw works too, provided `mount.yaw_deg` says
what was actually built.

The home-pose check relies on `fr3_hand_tcp` +X pointing along base +X at the Franka
home pose (the Franka Hand's usual home orientation, rotation `diag(1, -1, -1)`). Confirm
it once in RViz before relying on it.

### Measure these, then write them into `config/inspire_hand.yaml`

```
          joint-7 axis (fr3_link8 +Z) ──────────────────────────►
 flange ┃adapter┃ hand base ─── palm ──────────── fingertips
 face   ┃       ┃                ▲ palm_normal_offset_m (axis -> palm surface)
        ├───────┼────────────────┤
  adapter_thickness_m  palm_along_axis_m (hand mounting face -> palm centre)
```

| Key (`mount:`) | Meaning | Default, an estimate |
|---|---|---|
| `adapter_thickness_m` | flange face → hand mounting face | 0.012 |
| `palm_along_axis_m` | hand mounting face → palm centre, along the axis | 0.095 |
| `palm_normal_offset_m` | hand axis → palm **surface** | 0.020 |
| `palm_lateral_offset_m` | palm centre off the axis, towards the little finger | 0.0 |

The defaults come from the manual's drawing (217.8 mm from mounting face to fingertip,
80.7 mm palm width), not from the hand. **The planner stops the palm exactly where
these numbers put it.** A 10 mm error means 10 mm into the object or 10 mm short of it.
From them the code computes the palm point in `fr3_hand_tcp` coordinates. With the
defaults that is `[20.0, 0.0, 3.6] mm`: 20 mm along +X, and 12 + 95 - 103.4 = 3.6 mm
along +Z.

## 2. Where the gripper is switched

| Place | Franka Hand | Inspire hand |
|---|---|---|
| **Franka Desk** → Settings → End effector | Franka Hand | **Other / none**: the Inspire hand + adapter **mass**, **centre of mass** (roughly 0.08 m along +Z) and flange-to-EE transform 0.1034 m along Z, rotated -45°. The mass matters most while teaching: the arm floats with zero stiffness, so a wrong mass makes it sink or rise |
| **Robot bringup** (`THOR_DOCKER_SETUP`, `./scripts/run.sh robot`) | unchanged | **unchanged.** It still loads the Franka Hand description (`load_gripper` defaults to true), so `fr3_hand_tcp` exists and every CRISP controller keeps `end_effector_frame: fr3_hand_tcp`. The `franka_gripper` node finds no hand, logs an error and exits; nothing else depends on it |
| **Hand driver** | `franka_gripper` (in the robot container) | `ros2 run ycb_pointcloud_pipeline inspire_hand_node` on the host |
| **Planner** `end_effector` | `panda` | `inspire` |
| **Executor** `gripper_mode` / `gripper_namespace` | `franka_grasp` / `/franka_gripper` | `inspire` / `/inspire_hand` |
| **Teach node** `end_effector`, `gripper_mode` | `panda`, `franka_grasp` | `inspire`, `inspire` |
| **joint_homing** | default | `-p gripper_type:=inspire` (or `-p open_gripper:=false`) |
| **free_drive** | automatic: `gripper_type` defaults to `auto`, which reads `hands.mounted` in `x6_experiment.yaml` | automatic (`mounted: inspire`). While the arm floats: `c`+Enter closes the hand, `o`+Enter opens it, Enter alone holds |
| **X6 trials** | `-p hand:=panda` | `-p hand:=inspire` (reads `x6_experiment/config/hands/inspire.yaml`) |

**Cleaner alternative, not the default.** Launch with `load_gripper:=false`, so the URDF
shows a bare flange. Then set `end_effector_frame: fr3_link8` in
`THOR_DOCKER_SETUP/packages/fr3_crisp_bringup/config/controllers.real.yaml`
(`pose_broadcaster` and `cartesian_impedance_controller`), in this package's
`config/tpgpt_variable_stiffness_controller.yaml`, `tpgpt_teaching_controller.yaml` and
`joint_impedance_controller.yaml`, and `mount.controller_frame: fr3_link8` in
`config/inspire_hand.yaml`. The mount geometry gives the same palm pose in either frame
(unit test `test_both_controller_frames_describe_the_same_physical_hand`). Everything
else in the pipeline still assumes `fr3_hand_tcp`, so this path needs more checking.

## 3. The grasp

**Open and close** are the fixed poses tested on the real hand
(`ycb_pointcloud_pipeline/inspire_hand.py`):

| DOF | open | close |
|---|---|---|
| little, ring, middle, index | 1000 (straight) | 0, each stops at `force_g` |
| thumb bend | 1000 | 0, bends **after** the fingers |
| thumb rotation | 0 (opposed, facing the fingers) | 0 |

**Where the arm goes** (`tpgpt.hardware.side_grasp`, no GraspGen-X):

1. A circle fitted to the object cloud's footprint gives the axis. The camera sees
   only the front, so the centroid would sit up to `2r/π` in front of the true axis:
   21 mm for a 33 mm-radius can.
2. The palm approaches **horizontally, straight out from the robot base** towards
   that axis.
3. The palm stops **`standoff_m` (15 mm)** in front of the surface, at
   **`height_fraction` (0.5)** of the object's height. The floor is
   `min_height_above_support_m` (50 mm), because the hand lies on its side and half its
   80 mm palm hangs below the grasp point.
4. Thumb and index on top, little finger nearest the table (`thumb_up: true`). Fingers
   run horizontally round the object.

So the palm arrives beside the object, not above it, and the close wraps the fingers
round it with the thumb opposing.

**Palm frame** (the grasp convention the map transports): +Z = palm normal (approach),
+X = wrist → fingertips (the "closing" axis), +Y = across the palm. The fingers are the
closing axis because in a side grasp they are horizontal. The keypoint task frame is
built from the table normal and the closing axis, and refuses a vertical one; the first
version used the vertical palm width and planning failed on exactly that.

## 4. Running it

Build after any change: `cd ~/thor_ws && colcon build --packages-select ycb_pointcloud_pipeline`.
The CH340 serial adapter needs the out-of-tree `ch341` driver loaded (`/dev/ttyUSB0`).

```bash
# hand alone
ros2 run ycb_pointcloud_pipeline inspire_hand scan            # finds ID 2, 115200
ros2 run ycb_pointcloud_pipeline inspire_hand --speed 300 cycle
ros2 run ycb_pointcloud_pipeline inspire_hand_node            # /inspire_hand/open, /close
ros2 service call /inspire_hand/close std_srvs/srv/Trigger

# teach an Inspire demonstration (palm against the side of the object, then g)
ros2 launch ycb_pointcloud_pipeline tpgpt_teach.launch.py object_name:=tomato_soup_can \
    end_effector:=inspire gripper_mode:=inspire

# plan + execute (dry run first)
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p end_effector:=inspire -p gripper_mode:=inspire -p gripper_namespace:=/inspire_hand \
    -p source_bundle_dir:=$HOME/tpgpt_demos/<demo>/source_bundle \
    -p object_name:=tomato_soup_can -p destination:="[0.60, -0.25, 0.006]" -p dry_run:=true

# X6 trial
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_pick_place --ros-args \
    -p experiment:=true -p trial_id:=0031 -p hand:=inspire \
    -p object_name:=red_cup -p destination_name:=top_shelf -p dry_run:=true
```

Before the first real run, check the plan in RViz (`/tpgpt/plan/markers`): the axes at
the grasp should show red (palm normal) pointing at the object and blue (fingers)
horizontal. The overlay (`overlay.png`) draws the hand's envelope box at the grasp.

## 5. What to expect, and what is not known

* **A top-down demonstration transports badly onto a side grasp.** The map has to turn
  the demonstrated grasp 90°. On three synthetic scenes (the unit-test demo onto
  cylinder clouds), min det(J) was **0.007, 0.40 and 0.20**; the planner refuses
  anything under 0.1. For the Inspire hand, **teach a demonstration with the Inspire
  hand, from the side**. X6's rule of one Franka Hand demonstration does not allow
  that, so X6 Inspire trials will show this as planning failures.
* **Hand clearance.** The envelope is one box round the open hand, including the
  opposed thumb (80 mm out of the palm). On the synthetic scenes the transported path
  took its lowest corner 6-9 mm below the table before the grasp, which the default
  `grasp_min_path_clearance` (-2 mm) refuses. That is the check doing its job. If it
  refuses real plans, raise `min_height_above_support_m`; do not loosen the check.
* **Not checked by any code:** joint limits and reachability of the sideways wrist pose
  (joint 7 turns the palm; the flange has to lie horizontal), collisions of the arm
  with the object, and whether Desk's payload matches the real hand.
* **Measurements outstanding:** the four mount lengths, the envelope, Desk mass/CoM.
* `/inspire_hand/joint_states` reports closure 0-1 per DOF, not a width, so X6's
  `gripper_width` column stays empty for this hand.
