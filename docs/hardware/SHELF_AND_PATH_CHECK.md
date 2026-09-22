# The shelf, the whole-path check and grasp selection on the robot

Built 2026-09-21 (commits `f5f1e77`..`a647573` on branch `hardware`). This closes items A1, A2, A3,
A6 and C3 of `FAITHFULNESS_AUDIT.md`. The code is in `tpgpt/hardware/{fr3,shelf,path_check,
grasp_selection,executor}.py`; the ROS wiring is in `segmentation_node`, `tpgpt_planner_node` and
`tpgpt_executor_node`.

## What the planner now does, in one pass

1. **Screen** GraspGen-X's candidates (`grasp_selection.screen_candidates`). Two kinds of stage:
   - **Hard, hardware-built:** `from_below`, then the jaws after refinement (`too_wide`,
     `collides`, `empty`, `shallow`), then `hits_support`, then `duplicate`. These are kept
     because they were built on real one-camera clouds, where GraspGen-X stops 2–3 mm short of the
     object. The sim's `by_jaw_width` assumes a complete scan.
   - **Soft, the sim's own functions from `tpgpt.grasp.filters`:** `visibility`, `on_target`,
     `grasp_collision` (the sim funnel's `collision`, renamed), `place_zone` and `centred`. They
     follow the sim's rule: a stage that would reject every remaining candidate passes them all
     through and records `fell_back: true`.
2. **Choose** by the sim's rule (`pipeline._choose_grasp`): walk the survivors in score order (up to
   100), plan each, and take the **first whose whole transported path is admissible**: no forbidden
   contact at any stage, and every waypoint reachable in position, with the pick and place reachable
   with their orientation ("How a path is checked", below).
3. **Execute** with the sim's fallback for poses the arm can't hold. While the lag gate is shut,
   the executor relaxes the orientation, or steers to the next reachable label and jumps the phase
   on arrival.

Every rejection is named (`candidate.rejected`). The plan's `plan.json` metadata records the stage
tallies, the rank examined, any fallback, the chosen path check, the shelf world, and whether the
shelf is still an estimate.

## Which side's method was kept, and why

| question | simulation | robot, now | why |
|---|---|---|---|
| What the arm can hit | an **observed** scene cloud, robot subtracted; robot-robot contacts ignored | (1) **everything the camera sees** within reach (target, table and robot removed); (2) where it can't see, **only the shelf**: the measured model, seen or unseen; (3) **its own body**, checked exactly between every pair of pieces not joined at a joint | The user's statement of the bench (2026-09-22). The camera sees part of the shelf and little of the arm's workspace: cloud-only is blind at the unseen shelf, and treating unseen space as occupied would reject almost every path. The shelf is known by touch, independent of the camera. Self-collision is added because a folded arm is as real a collision as the shelf; at 40 real at-rest poses only joint neighbours' hulls touch, so excluding exactly those gives no false faults. Without joint angles the robot can't be removed from the cloud, so it falls back to the shelf region only, and says so. |
| The arm's shape and reach | MuJoCo's geoms and `solve_ik` | the FR3's own URDF (live `/robot_description`), a convex hull per link, and `solve_ik` transcribed | Same semantics as the sim: MuJoCo also collides a mesh by its hull. FK matches `/current_pose` to 0.01 mm median, 0.18 mm worst, at rest. |
| Where the IK path starts | the sim's fixed home | the **measured joints**, and the straight move onto the path is checked too (`approach_in`) | The real arm starts wherever it was left. This also puts the IK on the configuration branch the impedance controller will actually be on. |
| Exact geometry vs cloud tolerances | 1 mm on exact geometry, 5 mm on clouds | the same numbers: 1 mm on the model, 5 mm on the cloud and for the carried object | Constants are restated without importing robosuite, and a test asserts they match. |
| Place-zone release pose | one provisional pose for all candidates (the hand at the object's top above the slot) | each candidate's own release: its grasp point's offset from the object base, turned by the demonstration's carry rotation | The same question, asked exactly. It is cheap. |
| Centre of mass | exact (simulator) | the centre of the cloud's horizontal box, with a 25 mm margin instead of 15 mm | The sim's `by_centre_offset` docstring prescribes exactly this for perception. A one-sided cloud under-reports extent by 11–21 mm (`ROBOTICS_NOTES` 7.36). |
| Nothing admissible | execute the candidate whose first fault comes latest | refuse. The option `allow_kinematic_fallback` (off by default) takes a candidate that failed on kinematics alone; **never** one that collides | On the robot, a fault is a collision; and under the strict criterion a kinematic failure means the grasp or place pose, or some position, can't be reached. |
| Scripted top-down grasp | none | off by default; when on, it passes the same checks | The X6 brief forbids overriding the generator's choice. |
| Jaw / refine / table / duplicates | `by_jaw_width`, 20 mm / 20° duplicates | the hardware checks, unchanged | Validated on real partial clouds. A7 remains open. |
| min det / aim / orientation / plane gates | none (recorded only) | still applied (`_verify`) | Audit A4, not decided yet. |

## The shelf model: `ycb_pointcloud_pipeline/config/shelf.yaml`

Rebuild after editing (`colcon build --packages-select ycb_pointcloud_pipeline`). The planner reads
the installed copy.

**Measured 2026-09-22** (`measured: true`): heights and the two slot marks were touched by the
robot on 2026-09-21 (about 1 mm). Every other dimension is from a tape (M1–M14), taken by the user.
Two cross-checks agree:
- the width by the slots, M1 + 233.2 + M2 = 576 mm, against the inner width M5 = 580 mm: −3.8 mm;
- the apron plus the board, 72 + 18 = 90 mm, against the touched bottom board above the table,
  90.4 mm.

The model has two side panels (18 mm, 902 mm tall, 242 mm deep, flush with the boards), three
18 mm boards 242 mm deep (bottom at 0.0768 m, top at 0.3213 m, topmost 489 mm above the top), a
72 mm apron under the bottom board, and no back or roof.

**M1–M4 were hard to hold square** (the user). They are the only readings that place the shelf in
x/y and set its yaw, so they are corrected from the camera (`shelf_fit`, below) before the shelf
is used.

### Measuring it

The shelf has **no back and three boards** (bottom, top, topmost). The arm can't touch the side
panels' tops or the topmost board squarely without risk, so the shelf's own dimensions come from a
**tape**. The shelf is placed in the robot frame by the points already touched on 2026-09-21: the
two slot marks on the top board, and both board heights (`anchors` in `shelf.yaml`).

`docs/hardware/shelf_measurements.html` (published as a private artifact, openable on a phone at
the shelf) has top, front and side drawings with each measurement M1–M14 marked. It also has a
fill-in table with live checks, and produces the `tape_mm:` block to paste into `shelf.yaml`.
`shelf.resolve_tape` then:

- steps from each touched slot mark square to the front edge (M3, M4) and along it to each side
  panel (M1, M2);
- takes panel thicknesses, height and depth, board depth and thickness, the topmost board's height
  above the top board, and an optional front apron from the tape;
- **refuses** a result whose width by the slots (M1 + 233.2 + M2) disagrees with the tape's inner
  width (M5) by more than 6 mm. That usually means a slot mark was re-found in the wrong place, or a
  reading is wrong.

The width used is **M5**, the one direct reading across the shelf. The −3.8 mm by which M1 + M2
disagree with it is split between the two ends.

### Grounding the placement in the cloud: `shelf_fit`

The camera sees the shelf's faces directly: the right panel's inner face, the left panel's outer
face, the panels' and boards' front edges, the apron and the board tops. `shelf.fit_to_cloud`
finds the rigid correction that puts the model's faces on those points. It is a yaw about the front
edge's midpoint plus a shift along the front edge and into the shelf. Heights are never moved: they
were touched, and the camera's bottom-board reading is 5 mm off (`CALIBRATION.md`).

How it works:
- it takes the cloud points within 40 mm of the tape-placed model, and fits each point's distance
  to the nearest face by robust least squares (Cauchy, 4 mm scale), then again within 20 mm;
- a direction is fitted only if the cloud constrains it: at least 150 points squarely on side
  faces for the shift along the front, as many on front faces for the shift into the shelf, and
  front faces spanning at least 150 mm for the yaw. Board tops say nothing about x/y. A direction
  not fitted keeps the tape's value;
- points are matched only to faces turned towards the camera, so a thin panel can't be slid
  through its own points (the first live run matched the left panel's outer-face points to its
  hidden inner face and returned a 23 mm correction);
- a result is **refused** if any well-seen side or front face stays more than 3 mm off after the
  fit (no single placement fits the tape's dimensions to the cloud), or above 30 mm or 3° (beyond any tape error: suspect the calibration, an
  object or the arm against a panel, or a mis-keyed reading). It is also refused if the fitted
  model still sits more than 6 mm (median) from the shelf's points;
- the fit always starts from the tape, ignoring any `cloud_fit` already in the config, so re-running
  it gives the same absolute answer, not a correction on top of a correction.

Run it with the **shelf empty and the arm out of view**. The robot is also removed from the cloud
at `/joint_states`, but a hand resting on a panel would be read as the panel. Segmentation must be
running (`/ycb_scene_cloud`):

```bash
ros2 run --prefix python3 ycb_pointcloud_pipeline shelf_fit
```

It prints:
- the correction;
- the median distance of the shelf points to the model, before and after;
- **M1–M4 as the cloud places the shelf**, beside the tape's values;
- the median offset of each face seen, before and after.

It writes all of it to `~/tpgpt_calibration/shelf_fit_<time>.yaml`. In RViz, `/shelf_fit/model_tape`
(orange) and `/shelf_fit/model_fitted` (green) are drawn over `/shelf_fit/cloud`. It never edits
`shelf.yaml`: paste the printed `cloud_fit:` block under `shelf:`, rebuild, and restart the planner.

**Judge it by the faces, not the correction.** After the fit, each well-seen face should read
within about ±2 mm, both panels' faces agreeing. What would make the result wrong:
- a calibration error, which moves the cloud and so the fitted shelf with it (the calibration was
  validated to about 5 mm on these boards);
- one panel reading −3 mm and the other +3 mm after the fit, which means the width (M5) is off
  rather than the placement.

Tested on synthetic clouds of the real shelf, seen from the calibrated camera position with 3 mm
depth noise and a clutter object: a 7 / −9 mm, 0.8° misplacement came back within 1.5 mm and 0.1°
(`tests/unit/test_shelf.py`). A ROS dry run on a fake `/ycb_scene_cloud` (domain 88) recovered
5 / 8 mm, −0.6° as 5.0 / 7.3 mm, −0.54°.

**First live run (2026-09-22, shelf empty): refused.** Faces seen, offset from the tape model:
- top board: front face +3.8 mm, top +2.5 mm. So the front edge (M3, M4) and the top board are
  confirmed.
- bottom board: top +11 mm. This board is badly seen (`CALIBRATION.md`), so it is not evidence.
- side panels, along the front edge: the left panel's outer face is 11 mm further left than the
  model, and the right panel's inner face 25 mm further left. The top board's surface also ends
  10–20 mm short of the model's right panel.

So the cloud puts both panels towards the left, and the far one more. That fits either explanation:
- M1/M2 (and M5) are off;
- the camera is off horizontally at the shelf. A yaw error of about 1° grows with range: the
  left panel is about 0.8 m from the camera, the right about 1.4 m.

The calibration validated heights, not horizontal positions this far out.

**Settled the same evening: the camera is the main cause, and the tape stands.** Both slot marks
were touched again and clicked in the cloud (`~/calib_tools/shelf_points.py`,
`~/tpgpt_calibration/shelf_points_20260922_221426.yaml`). Offsets in mm, as (along the front,
into the shelf, height):

| | left mark | right mark |
|---|---|---|
| new touch minus the 2026-09-21 touch | −4.2, −1.0, +0.4 | −2.4, −2.7, −0.9 |
| click minus touch | **−11.3**, −19.5, +7.5 | **−10.2**, −9.8, −1.3 |

What this shows:
- The marks were found again within about 4 mm, so the anchors hold.
- The camera places both marks **about 10 mm to the left along the front edge**. That explains the
  left panel's 11 mm completely.
- For the right panel, 10 mm of its 25 mm is explained. Take 10 mm off the cloud's reading of where
  the top board's surface ends (560–570) and it gives 570–580, against the tape's 580. So the
  width is right. The rest of that panel's 25 mm comes from its inner face, which the camera sees
  poorly (9.6 mm RMS about a fitted plane, and the plane turned 6°).
- The "into" and height parts of the clicks are not used. A click lands on the mark's *colour*,
  which carries any colour-to-depth registration error, and a single point is noisy. The
  geometric front face of the top board, which has no such error, reads +3.8 mm.

**Decision: no `cloud_fit`.** The robot moves in true coordinates, so the shelf must be where the
robot touched it, not where the camera sees it. The shelf is placed from the tape plus the touches.
`shelf_fit` stays as a check. It would be worth re-running only after the calibration improves at
the shelf.

**Consequence for everything else:** the camera is about 10 mm off sideways at the shelf, so object
clouds there are too. Recorded as an open calibration issue (`CALIBRATION.md`), and it bears on A5
(`grasp_offset`). An optional robot touch on the right panel's inner face would settle that
panel's remaining ≤ 10 mm.

### Checking it

Run the planner (or `tpgpt_pick_place` with `dry_run:=true`) and look at RViz, fixed frame `base`:

- `/tpgpt/plan/shelf`: the model's panels, with support boards drawn darker.
- `/tpgpt/plan/shelf_cloud`: the observed obstacle points (magenta). With the shelf empty,
  those on the shelf should lie on the panels, and nothing should remain on the arm or the table.

The planner logs `Shelf world: {...}`. **`explained_by_model`** is the fraction of observed shelf
points within 15 mm of the model's surface. With an empty shelf it should be close to 1. A low
value means the shelf has moved or the model is wrong. `unexplained_points` counts the rest: objects
on the shelf, or model error.

## How a path is checked for collisions (plain account)

The plan is a path of a few hundred hand poses (waypoints). The check walks it, testing every 4th
waypoint (the sim's spacing). At each one:

1. **The arm's posture.** Inverse kinematics finds the seven joint angles for the planned hand pose,
   starting from the previous waypoint's answer. If it fails, it is retried (400 iterations, then 3
   starting postures near the previous answer, never a jump of more than 0.5 rad on any joint, which
   the controller couldn't follow).
   - If only the **orientation** can't be reached, the executor will relax it, so the check tests
     the posture it will actually take, marked `(relaxed)`.
   - If even the **position** can't be reached, the path is rejected (see Reachability).
2. **The robot's pieces** at that posture: the 8 arm links as convex shapes around their real
   collision meshes, and the hand as boxes. On the Franka Hand, the palm (`hand_body`) is separate
   from the finger boxes. Fingers are open before the grasp and closed to the object's width while
   holding.
3. **The world:**
   - **the table:** an exact flat surface at the touched height;
   - **the shelf:** the measured model;
   - **everything else the camera sees** within reach: its points, minus the robot, the target, and
     the camera's points on the table and the shelf;
   - **the robot itself.**

**In every stage** the robot must not touch the table, the shelf, the seen points, or itself (except
links joined at a joint, and the hand's own parts). "Touch" means more than 1 mm into exact geometry,
or any seen point inside a piece.

**What each stage allows.** The stages come from the gripper schedule: *g* is the close, *r* the
release.

| stage | waypoints | the object and the robot | the object and the world |
|---|---|---|---|
| **onto the path** | the move from where the arm is to the plan's start | no part of the robot may touch the object | — (it stands where it is) |
| **approach** | before *g* − 3 | no part of the robot may touch it | — |
| **pick** | *g* − 3 … *g* | the **fingers** may touch it; the palm and links may not | it stands on its support |
| **carry** | after *g*, before *r* − 12 | held by the fingers; **no arm link** may touch it | it may rest on the surface it was picked from **until it first clears it**, then never again. It touches nothing else. |
| **place** | *r* − 12 … *r* | held by the fingers; no arm link may touch it | only its **place face** may touch, and only the **destination surface** (the destination board's top face, or the table if that is the destination), at most 5 mm in. Another face touching (it's being put down on its side), another surface, a side panel, or any seen point is a fault. |
| **retreat** | after *r* | the object stays where it was set down and is an obstacle. The **fingers** may touch it **until they first clear it**, then never again; the palm and links never. | — |

**The place face** is the face the plan sets the object down on: its cloud points within 10 mm of
its lowest point **in the planned release pose**. It is *not* the face the object stood on at the
pick. A plan may turn the object (a box picked lying flat and shelved upright), and the release pose
is the map's transport of the demonstration's release, so the face down in it is the one the method
means to set down. The object's cloud is carried rigidly with the hand, so the check can ask, at
every waypoint of the place, whether a point of *another* face is touching (tilting in, dropping on
an edge). This replaces the sim's rule, which called a contact "support" when it lay within 60° of
straight down from the object's centre, and misreads the bottom corners of a wide flat object as
side hits. "Until it first clears" is measured, not counted: the object clears its
support when its lowest point is more than 5 mm above it, and the fingers clear the placed object
when none of them touches it.

**Limits of this.**
- The Inspire hand is modelled as one envelope, so its fingers can't be told from its palm. The
  envelope counts as fingers.
- The camera sees the object from one side, so its place face is the lowest part of what was seen,
  turned into the release pose.
- Link 5's convex hull is 1.78× its real volume. Postures that fold the hand back toward the forearm
  may be reported as self-collisions that aren't real; splitting link 5 into two parts would settle
  it.

**Reachability** (`kinematic_criterion`, the user's choice 2026-09-22):
- **`strict`** (the default):
  - every tested waypoint, and every step of the move onto the path, must be reachable in
    **position**;
  - the orientation may be relaxed in **transit** only (approach and carry), and is checked at that
    posture;
  - in the **pick and place** it must be reached.
- **`sim`**: at least 60 % of the waypoints reachable as planned (the simulation's rule). The rest
  are checked where the executor will put the arm, including the straight line it steers along past
  positions it can't reach.
- **The least-bad fallback** (`allow_kinematic_fallback`) is **off**: under `strict` it would execute
  a plan the criterion refused.
- **Reported:** `orientation_relaxed`, `relaxed_in_pick_or_place`, `skipped` (position out of reach)
  and `first_kinematic`.

## The scene cloud

`segmentation_node` publishes `/ycb_scene_cloud` each cycle: every 4th pixel of the same depth frame
as the object clouds (~25k points), in the optical frame. The planner keeps a cloud only if it is
under `scene_max_age` (10 s) old. It keeps every point within the arm's reach (1.15 m from the
shoulder), then removes:

- the robot, at the joint angles recorded with that cloud;
- the target, within 15 mm of its cloud;
- the table, everything below its height + 15 mm. The cloud's table lies within about 10 mm of the
  touched plane (4.9 mm offset, 0.61° tilt). The cost: anything flatter than 15 mm lying on the table
  isn't an obstacle.
- the shelf, points within 15 mm of the model's panels. The cost: the bottom 15 mm of an object
  standing on a board. The gain: a panel mis-measured by more than 15 mm keeps its real points as
  obstacles.

It then applies a radius-outlier filter at 20 mm.

**Joint angles are required.** The planner refuses to plan without fresh `/joint_states` (under
1 s old), or when the scene cloud arrived before any joint angles. Without them the robot can't be
removed from its own picture. They are always published while the robot container runs, so their
absence means the container is down, FCI is off, or the terminal is on the wrong ROS domain
(`host_ros_env.sh`). `allow_missing_joint_states:=true` permits planning with the robot off
(offline dry runs). The cloud then keeps only the shelf region, and the plan records it. The
executor refuses a real run without them.

## Parameters (planner node)

`shelf_config` (installed `shelf.yaml`; `''` = none), `use_scene_cloud` (true),
`scene_cloud_topic`, `scene_max_age`, `scene_outlier_radius`, `scene_outlier_min_neighbors`,
`joint_states_topic`, `allow_missing_joint_states` (false), `joint_states_max_age` (1 s),
`check_path` (true), `kinematic_criterion` (`strict`), `min_reachable_fraction` (0.60, `sim`
only), `allow_kinematic_fallback` (false), `max_centre_offset` (0.025), `grasp_max_plans` (100),
`grasp_fallback_top_down` (false).

Executor node: `joint_states_topic`. Without fresh joints the fallback is off for that step, and
the completion log counts how often that happened.

**X6 overrides these.** The frozen `method:` block in `x6_experiment/config/x6_experiment.yaml`
still sets `max_plans: 12` and `fallback_top_down: true`, and `x6_trial.py` applies them over the
node defaults. That file is left for the X6 rework.

## What was verified, and what was not

- **Tests:** 106 hardware tests pass (`tests/unit/test_{fr3,shelf,path_check,grasp_selection,
  hardware}.py`), including FK against 40 at-rest samples of the real arm. The ROS package's tests
  pass, apart from the two known stale-X6-config failures.
- **Offline ROS dry run** (ROS domain 88, synthetic inputs, fake `/joint_states`, the real
  GraspGen-X server): the whole chain runs. That is, the scene cloud is cropped to the shelf, the
  funnel stages and the path check run, and every rejection is named. Checks of the saved X6 plans
  and object clouds were also run, but those were captured with the **previous camera and shelf
  setup** and are **not evidence** for this one (user, 2026-09-22). Anything about grasp success,
  the shelf or `grasp_offset` has to be measured afresh.
- **Not run on the robot, and not yet on a single capture from the new setup.** Not verified yet:
  the whole pipeline on fresh camera data, robot self-removal from a real scene cloud, the measured
  shelf, the executor's fallback on real hardware, and the path-check timing on a real shelf.
- **Still valid from before:** the FK fixture (`tests/data/fr3_still_samples.npz`). It holds joint
  angles against `/current_pose`, which neither the camera nor the shelf enters.
