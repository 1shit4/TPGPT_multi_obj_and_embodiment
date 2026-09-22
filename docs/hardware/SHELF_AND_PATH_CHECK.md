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
   100), plan each, and take the **first whose whole transported path is admissible**. Admissible
   means no collision at any tested waypoint, and at least 60 % of the waypoints within the arm's
   reach.
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
| Nothing admissible | execute the candidate whose first fault comes latest | only a candidate that failed on **kinematics alone** (`allow_kinematic_fallback`); **never** one that collides | On the robot, a fault is a collision with the shelf. |
| Scripted top-down grasp | none | off by default; when on, it passes the same checks | The X6 brief forbids overriding the generator's choice. |
| Jaw / refine / table / duplicates | `by_jaw_width`, 20 mm / 20° duplicates | the hardware checks, unchanged | Validated on real partial clouds. A7 remains open. |
| min det / aim / orientation / plane gates | none (recorded only) | still applied (`_verify`) | Audit A4, not decided yet. |

## The shelf model: `ycb_pointcloud_pipeline/config/shelf.yaml`

Rebuild after editing (`colcon build --packages-select ycb_pointcloud_pipeline`). The planner reads
the installed copy.

**Currently only the board heights, the slot marks and the shelf's yaw are measured** (the touched
points in `CALIBRATION.md`). Width, depth, the side panels and the topmost board are **estimates**
(`measured: false`) until the tape measurements below are in. Every
plan says so in its notes and in `metadata['shelf_is_estimate']`. Until you measure them, the check
may report a collision at an estimated wall that isn't there, or miss one that is. The estimate
also assumes **no roof** above the top board (an open-top two-tier shelf, from `CALIBRATION.md`'s
"recessed under the top shelf"). A roof that isn't there would block every top-board placement,
so it is left out until it is seen or touched.

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

While every `tape_mm` value is `null`, the estimate below it stays in use (now without a back panel,
and without the topmost board, whose height is unknown). Once one value is filled, all required ones
must be. Then set `measured: true` and rebuild.

### Checking it

Run the planner (or `tpgpt_pick_place` with `dry_run:=true`) and look at RViz, fixed frame `base`:

- `/tpgpt/plan/shelf`: the model's panels, with support boards drawn darker.
- `/tpgpt/plan/shelf_cloud`: the observed obstacle points (magenta). With the shelf empty,
  those on the shelf should lie on the panels, and nothing should remain on the arm or the table.

The planner logs `Shelf world: {...}`. **`explained_by_model`** is the fraction of observed shelf
points within 15 mm of the model's surface. With an empty shelf it should be close to 1. A low
value means the shelf has moved or the model is wrong. `unexplained_points` counts the rest: objects
on the shelf, or model error.

## The scene cloud

`segmentation_node` publishes `/ycb_scene_cloud` each cycle: every 4th pixel of the same depth frame
as the object clouds (~25k points), in the optical frame. The planner keeps a cloud only if it is
under `scene_max_age` (10 s) old. With `scene.scope: visible` (the default), it keeps every point
within the arm's reach (1.15 m from the shoulder) and removes:

- the table: everything below the support height + 15 mm. The table is the support plane the
  planner checks separately, and the cloud's table lies within about 10 mm of it (4.9 mm offset,
  0.61° tilt). The cost: anything flatter than 15 mm lying on the table isn't an obstacle.
- the target: points within 15 mm of its cloud;
- the robot: the FR3 hulls and hand boxes at the `/joint_states` recorded with that cloud.

It then applies a radius-outlier filter at 20 mm. Without joint angles the robot can't be removed,
so only the shelf region is kept (`scope: shelf`: the model's box plus 3 cm, and 0.30 m above the
highest support board). The report says so. `explained_by_model` is always measured on the shelf
region.

## Parameters (planner node)

`shelf_config` (installed `shelf.yaml`; `''` = none), `use_scene_cloud` (true),
`scene_cloud_topic`, `scene_max_age`, `scene_outlier_radius`, `scene_outlier_min_neighbors`,
`joint_states_topic`, `check_path` (true), `min_reachable_fraction` (0.60),
`allow_kinematic_fallback` (true), `max_centre_offset` (0.025), `grasp_max_plans` (100),
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
