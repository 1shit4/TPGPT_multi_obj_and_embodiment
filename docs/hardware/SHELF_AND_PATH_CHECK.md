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
| What the arm can hit | an **observed** scene cloud, robot subtracted | the **measured shelf model**, plus the observed cloud **inside the shelf's region only** | The camera sees part of the shelf and little of the arm's workspace. Cloud-only is blind at the unseen shelf. Treating unseen space as occupied would reject almost every path, since most of the arm's motion is out of view. The shelf is known by touch (±1 mm, independent of the camera), and it is the only thing on the bench the arm can hit. The cloud keeps what the model can't know (an object standing in a slot, a panel the model gets wrong) wherever the camera sees it. Outside the shelf region nothing counts as an obstacle. |
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

**Currently only the board heights and the shelf's yaw are measured** (the four slot points in
`CALIBRATION.md`). Width, depth, walls, back and top are **estimates** (`measured: false`). Every
plan says so in its notes and in `metadata['shelf_is_estimate']`. Until you measure them, the check
may report a collision at an estimated wall that isn't there, or miss one that is. The estimate
also assumes **no roof** above the top board (an open-top two-tier shelf, from `CALIBRATION.md`'s
"recessed under the top shelf"). A roof that isn't there would block every top-board placement,
so it is left out until it is seen or touched.

### Measuring it

Close the gripper, run `python3 ~/calib_tools/touch_point.py` (a live fingertip-face readout, with
the 9.5 mm probe offset applied), and free-drive the arm (`ros2 run ycb_pointcloud_pipeline
free_drive`). Keep the hand tilt under 30°.

| field | touch |
|---|---|
| `front_left` [x, y] | The front edge of a board, at the inner face of the wall on the robot's left as it faces the open front (the "left slot" side). Touch the corner where the board's front edge meets the wall. |
| `front_right` [x, y] | The same corner at the other wall. The builder refuses a swapped pair. |
| `depth` | Touch the back panel's inner face at board height. `depth` is its distance from the front edge line. |
| `top` | The top of the side walls. If there is a roof above the top board, add it under `boards` with `support: false`. |
| `wall_thickness`, `back_thickness`, board `thickness` | A ruler at an exposed edge (0 for an open back). |
| `bottom` | Already the touched table height. |

Then set `measured: true`. Anything else solid goes under `extra_boxes` (lo/hi corners, shelf or
base frame).

### Checking it

Run the planner (or `tpgpt_pick_place` with `dry_run:=true`) and look at RViz, fixed frame `base`:

- `/tpgpt/plan/shelf`: the model's panels, with support boards drawn darker.
- `/tpgpt/plan/shelf_cloud`: the observed points kept on the shelf (magenta). With the arm out of
  view and the shelf empty, these should lie on the panels.

The planner logs `Shelf world: {...}`. **`explained_by_model`** is the fraction of observed shelf
points within 15 mm of the model's surface. With an empty shelf it should be close to 1. A low
value means the shelf has moved or the model is wrong. `unexplained_points` counts the rest: objects
on the shelf, or model error.

## The scene cloud

`segmentation_node` publishes `/ycb_scene_cloud` each cycle: every 4th pixel of the same depth frame
as the object clouds (~25k points), in the optical frame. The planner keeps a cloud only if it is
under `scene_max_age` (10 s) old. It crops it to the shelf region (the model's box plus 3 cm,
reaching 0.30 m above the highest support board so objects standing on an open-top shelf stay in).
It then removes points within 15 mm of the target's cloud, removes the robot using the FR3 hulls and
hand boxes at the `/joint_states` recorded with that cloud, and applies a radius-outlier filter at
20 mm. Without joint angles, the robot is not removed, and the report says so.

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
