# Hardware vs MuJoCo: implementation differences that break faithfulness

Audit of 2026-09-21. TPGPT at `d3275d7`. Compared stage by stage:

- **Sim reference:** `tpgpt/experiments/pipeline.py::run` + `_choose_grasp`,
  `tpgpt/grasp/filters.py::filter_grasps`, `tpgpt/sim/demo.py::record_demonstration`,
  `tpgpt/sim/rollout.py::rollout_policy`.
- **Hardware:** `tpgpt/hardware/{grasp_selection,planner,teach,executor}.py`, plus
  `ycb_pointcloud_pipeline/.../tpgpt_{planner,executor}_node.py` and the X6 `method:` block.

**What this list leaves out:** differences the port forces: a one-sided camera cloud instead of the
sim's full YCB scan, the bounded-speed approach from wherever the arm is, `SafetyMonitor`, and
wall-clock loop timing. Every item below could be implemented the way the sim does it. (CRISP taking
only a diagonal stiffness was on this list until 2026-09-29; see B2 in the Status table.)

Severity: **H** changes which grasp is executed or what a trial measures. **M** changes behaviour in
some runs. **L** is a small numerical mismatch.

## Status

| item | status |
|---|---|
| A1, A2, A3, A6, C3 | **Addressed 2026-09-21** (commits `f5f1e77`..`a647573`). Details, and where the robot deliberately keeps a different answer, are in `SHELF_AND_PATH_CHECK.md`. The kept differences: obstacles are everything seen within reach, plus the measured shelf model where unseen, plus self-collision (which the sim ignores), instead of a cloud-only scene; the least-bad fallback only for kinematics-only failures; the top-down grasp opt-in; the hardware jaw, refine, table and duplicate checks kept hard; per-candidate release poses in the place zone; a 25 mm centre margin on the cloud's box centre. |
| A5 | **Decided 2026-09-23: `grasp_offset` is now zero.** Measured on saved captures of all four objects: with [−35, 0, −25] mm an obviously good top-down grasp of a YCB can (jaws `fits`, 66 mm between the pads, 104 points, 62 mm above the table) is refused by the planner as `object_too_wide: the cloud measures 4 mm between the pads`, because the offset moves the hand 43 mm off the object *between* the jaw check and the planner. It caused most `plan_refused` rejections on every object. With it zero, all four objects plan; the banana and the mug take the top-scored candidate (rank 1). `refine_grasp` remains as the only thing that moves a grasp, and it is checked at the pose it produces. |
| A5 (old text) | **Was open.** `grasp_offset` ([−35, 0, −25] mm, planner default) is a fixed base-frame shift, the same for every object, set before the recalibration, probably to absorb the old calibration's error. Whether it helps or hurts must be measured on fresh captures from the new setup: plan the same scene with it and with 0, no motion. |
| A11 | **Added 2026-09-23**: the fit's free-space test, and refine lifting a low grasp. |
| A10 | **Deliberately different 2026-09-23** (this table's A10 row): approach cap 90 deg, visibility `auto`. |
| A9 | **Addressed 2026-09-22** (commit pending). The mask is eroded 1 px before deprojection, as the sim does. Restart `segmentation_node` for it to take effect. |
| A14 | **Fixed 2026-09-24**: the place-side hand clearance was an infinite plane and refused every upper-shelf grasp; it now measures the board under the hand. |
| A13 | **Extended 2026-09-24**: the object's grasps are a fixed library keyed on its scan, not a fresh draw per plan, and coverage against a destination is now measurable offline (`GRASP_LIBRARY.md`). |
| A4, A7, B1, B2, C1, C2, C4–C7 | Open. B1/B2 are being worked on (2026-09-29, plan: attractor labels from a replayed demonstration, directional stiffness); first step below. |
| B2, step 1 | **The full transported stiffness reaches the arm (2026-09-29).** The sim's controller takes full `K` and `D` matrices (`tpgpt/sim/controllers/cartesian_impedance.py`); stock CRISP took only a diagonal in the hand frame, which for a stiffness soft along the motion and stiff across it drops a quarter of the matrix (measured on the 18:39 demonstration's path at 350/600 N/m: 421 N/m along the motion instead of 350 before the grasp, 468/486 -- almost isotropic -- at the set-down). CRISP is now patched to take a symmetric 3x3 (`~/THOR_DOCKER_SETUP`, `patches/crisp-full-stiffness.patch`, principal values clamped to [0, 900], damping `2 K^(1/2)`), and the executor sends `R^T K R` whole when `use_transported_stiffness` is on (`stiffness_message: auto`). Remaining difference from the sim: damping is CRISP's `2 K^(1/2)` (critical for a unit mass), not the transported `D`; the sim's demonstration uses `1.8 sqrt(k)`. Verified: unit tests, CRISP on fake hardware, and a ROS dry run whose simulated arm is identical to the diagonal form's for today's constant stiffness (difference 1e-13 mm). Not yet run on the robot. |
| A8 | **Addressed 2026-09-22** (commits `f6a34e4`, `60df37d`), pending a live object that fits. The object cloud is the object's YCB scan fitted to the camera cloud (`tpgpt/hardware/object_model.py`, planner `object_model`, default true), as in the sim's YCB experiment. A fit the camera doesn't support is refused and the camera cloud is used, recorded as `object_cloud_source`. First live fit (a banana on the white sheet) was refused: median 5.0 mm, 78% of points explained, against about 3 mm and at least 93% for a correct fit at 5 mm synthetic noise. |
| X6 | **Resolved 2026-09-29.** The frozen `method:` block mirrors the planner's and the executor's defaults value for value (`max_plans` 100, `fallback_top_down` false, the grasp library, the scan fit and its retries, the executor's parameters), and `test_x6.py::test_the_frozen_method_is_what_the_nodes_run` fails when a default drifts from it. So an X6 trial runs the method exactly as a run outside an experiment does, executor deviations C1-C7 included, instead of the mixed set frozen on 2026-09-15. |

---

## A. Grasp selection (the largest group)

| # | Sim | Hardware | Sev |
|---|---|---|---|
| **A1. No scene** | `scene_point_cloud(env, exclude=target)`, with the robot's own body subtracted (`self_filtered`). Candidates are checked against it by `by_collision` and by `path_feasibility_observed` at every waypoint. | No scene cloud exists. `TargetScene` holds only the object cloud. Collision is checked only against the object (jaws) and an **infinite flat plane** at the support and destination heights. The shelf's sides, back and underside, and any neighbouring object, are invisible. | **H** |
| **A2. Selection rule** | `_choose_grasp`: walks the funnel survivors (up to **100**) in score order. It takes the **first whose whole transported path is admissible** (no collision at any waypoint, reachable fraction ≥ 0.60). If none is admissible it takes the **least-bad candidate** (latest first fault) and flags it. | `select_and_plan`: tries only the top **12** (`max_plans`). It admits a candidate on different criteria (A4). If none passes, it falls back to **a scripted top-down grasp** (`default_grasp`) that the generator never proposed. | **H** |
| **A3. No kinematic check** | `path_feasibility_observed` solves IK along the path and rejects `reachable_fraction < MIN_REACHABLE_FRACTION = 0.60` (`pipeline.py:131`). | None. An unreachable path shows up only at execution, as a shut gate and then a stall (which then triggers E1). This needs only FR3 kinematics, not a simulator. | **H** |
| **A4. Extra acceptance gates** | The map is refused only if det(J) **changes sign**. Aim and orientation error are *recorded*, never gated. | `_verify` also rejects on `min_det < 0.1`, `aim_grasp > 10 mm` and `orientation_error_grasp > 15°` (`grasp_selection.py:323`, `planner.py:307`). This can reject a grasp the sim would execute. | M |
| **A5. The grasp is moved after generation** | The executed grasp is the generator's pose (the roll is resolved in `scene_keypoints`). | `refine_grasp` slides each grasp up to 25 mm deeper and up to 10 mm sideways. Then `grasp_offset` adds **[−35, 0, −25] mm in the base frame** to every grasp TCP (`tpgpt_planner_node.py:130`) *before* the keypoint cube is built. The offset ignores the grasp's orientation. It is **not in the frozen X6 `method:` block**, and it was probably tuned against the old calibration. | **H** |
| **A6. Missing funnel stages** | visibility (`MAX_BACKWARD_DEG` 100), on-target (4 cm), place-zone corridor (with `carry_rotation`), centred on the CoM (15 mm). | None of these. | M |
| **A12. The placed orientation is a yaw** | `placement_rotation` makes the target's task frame at release equal the source's. `task_frame`'s third column is the **support normal**, so the rotation maps that normal onto itself: it is a yaw about the table. The source's placed block, by contrast, is the object moved by the demonstration's *actual* carry rotation, tipping included. | **Unchanged, and a limitation to know.** A demonstration that tips its object onto a different face (picked from its side, set down on its base) is not reproduced on the target: the target keeps the attitude it was picked in, and the map has to absorb the difference. It does not bite on this bench, where both ends rest on horizontal surfaces (user, 2026-09-23). **It may, however, be what folds the map for tilted grasps**: measured 2026-09-24 over 286 planned candidates on five objects, a grasp tilted 0-45 deg from straight down folds the map (det(J) changes sign, the sim's own rule) in 1 case of 155, and one tilted past 45 deg in 74 of 131. A near-horizontal pick yawed about the table at the release has its keypoint cube rotated nearly into the table plane, which would explain it; unproven. It is why the tomato soup can, which this hand can only grip around its barrel, plans nowhere (`GRASP_LIBRARY.md`). The hand's *tilt* is likewise inherited from the pick, which is why the destination a demonstration can reach is decided by the demonstration, not by the filters: a demonstration that lowers onto the top board from above cannot reach under it (measured: every candidate hit `shelf top_board` during the carry), while one that slides under the bottom board reaches both boards (measured the same day: bottom 0.969 / min det 0.86, top 0.962 and 0.936 / min det 0.89 and 0.81). | M |
| **A10. The approach cap and the visibility filter** | `SUPPORT_APPROACH_MAX_DEG = 85`: the geometric bound is 90 (past it the hand rises through the table), lowered to 85 because the sim's fingers hang below the TCP and it has no separate table check. `by_visibility` always on (`MAX_BACKWARD_DEG` 100). | **Deliberately different, 2026-09-23.** Cap **90** (`grasp_max_tilt_deg`): the hardware checks the hand's lowest point against the table separately (`hits_support`, 3 mm), so the reason for 85 is covered twice. Measured on a YCB can: of 100 candidates, 15 fall in 85–90 deg and they include the best-scoring one (0.871), all horizontal grasps around an upright can. Visibility is `auto`: applied only when the object's **scan was not fitted**, since a fitted scan has no unseen side; on this bench the camera faces the robot, so it otherwise removes exactly the robot-side approaches (15–28 of 100 measured). | M |
| **A11. The object's pose is estimated, and low grasps are lifted** | Not applicable: the sim knows the true pose, and its `by_support_approach` is the only table test. | `object_model` fits the scan (A8) and judges it by three measured numbers: the camera's points on the scan (median, fraction explained) and the fraction of the scan's **visible** surface standing in front of what the camera measured along the same ray (free space). Visibility is a z-buffer, not a normal test, because a mug's far wall faces the camera but is hidden. Measured over all pairs of the four scans: a correct fit reads in-front ≤ 0.07, a wrong one ≥ 0.13. `refine_grasp` may also lift a grasp out until the hand clears the table. | M |
| **A9. Mask erosion** | `DEFAULT_MASK_EROSION = 1` pixel off every instance mask before deprojection (`tpgpt/perception/cameras.py`). Measured: a 5 x 5 x 9 cm box fused to [17.3, 8.8, 9.7] cm unerooded and [5.8, 6.2, 8.9] cm with one pixel, because boundary pixels take their depth partly from the background and fly towards the camera. | **Was none.** `segmentation_node` deprojected the mask as detected. Measured here 2026-09-22: a YCB tomato soup can (68 x 68 x 102 mm) fused to 133 x 99 x 110 mm, its centre pulled ~25 mm off the object, which then failed the scan fit (58% explained), the `centred` stage (23 of 51) and the planner's own jaw check. **Fixed**: `mask_erosion` (default 1, the sim's value), skipped for a mask it would leave under `mask_min_pixels` (60). | **H** |
| **A8. The object cloud** | In the YCB world, **the full scan at the true pose** (`pipeline.run`, "The object's cloud is the full scan; the scene's stays observed"). One-sided camera clouds pushed hands onto rims; the full scan halved never-lifted runs (`docs/sim/ycb_grid.md`). The scene cloud stays observed, and visibility still uses the real camera positions. | Was the camera's one-sided cloud. Now the YCB scan fitted to it: pose estimated (resting face, position, heading), since the robot has no true pose. Bounded by the camera calibration (~10 mm sideways near the shelf). **2026-09-25, the mug:** the fit is restricted to the poses the protocol sets objects down in (`PROTOCOL_RESTS`: mug opening-up, pudding box on an end), because from this camera an upright and an upside-down mug differ only inside the cup and the fit chose upside-down on 15 of 20 live frames. For the mug alone (`OPEN_VESSELS`) the rest tilt is off and the pose is fitted a second time without the camera points lying >20 mm behind the first fit's surface along their ray -- the D435's infrared bouncing inside the glossy cup put 15-30 % of the cloud ~80 mm too far, and the fit turned the handle onto those points. On a recording with the handle in four known directions, the handle is now within 30 deg on 42 of 50 fits where it is visible (3-13 of 50 before); hidden, it stays unknowable, so the protocol keeps it visible. Other objects' fits are bit-identical on every capture. **Also 2026-09-25:** objects that rest on a flat face (`FLAT_RESTING`: mug, pudding box, foam brick) are fitted without the rest tilt, which leaned the brick 19.5 deg and the upright pudding box 12 deg on the planner's live fits; the banana, apple and can keep it. | **H** |
| **A13. Where the candidates come from** | `tpgpt/grasp/cache.py` caches GraspGen-X's answer per (cloud, gripper, size): in simulation the object's cloud is the same every episode, so the cache hits and the draw is effectively fixed. | **Deliberately extended, 2026-09-24.** The camera's cloud is never twice the same, so that cache would never hit and every plan asked the sampler a fresh question -- measured on one banana: 100 candidates gave 16, 0 and 0 admissible, 400 gave 5, 38 and 38. Since the planner works on the fitted **scan** (A8), the grasps are asked for once in the scan's own frame, cached with the sim's own module, and carried into the scene by the fit (`tpgpt/hardware/grasp_library.py`, `grasp_library:=true`, 2000 grasps). The selection rule and every filter are unchanged; a refused fit or an unreachable server falls back to the live draw. `docs/hardware/GRASP_LIBRARY.md`. | M |
| **A14. The place clearance** | No such gate: the sim judges a path by `path_feasibility_observed` against the scene it can see. | **Was wrong, fixed 2026-09-24.** `_verify`'s `path_hits_surface` measured the hand's lowest point over the *whole retreat* against an infinite plane at the destination's height. The taught retreat comes back down in front of the shelf (traced: 8 mm above the board at the release, up to 70 mm, then 35-53 mm "below" it while 215-314 mm away, over open floor), so the gate refused **every** upper-shelf candidate -- 25 of 25 on four banana captures and on the apple and the foam brick -- while `path_check` passed the same plans. Now measured only while the hand is within the object's footprint plus 50 mm of the destination (`planner.PLACE_FOOTPRINT_MARGIN`). After the fix those slots admit 5-11 of 25. `GRASP_LIBRARY.md`. | **H** |
| **A7. Parameters and algorithms that differ** | duplicates 20 mm / 20°. `by_jaw_width` at the grasp height, `JAW_MARGIN` 5 mm. | duplicates 10 mm / 15°. `hand.check_jaws` (a pad-box test), `JAW_MARGIN` **0** (`hand.py:166`). | L |

A1 and A3 together meant the hardware could reject a grasp only on jaw and plane criteria, never on
the shelf. (The X6 trials run before the move executed ranks 27, 84 and 10 of 100. They are
recorded for history only; that setup no longer exists.)

## B. The source demonstration (the taught X6 banana demo)

| # | Sim | Hardware | Sev |
|---|---|---|---|
| **B1. Labels are the measured pose, not the attractor** | `record_demonstration` labels with the **commanded attractor** (`demo.py:223`). Its docstring records that labelling the measured pose was a live bug (the lag was applied twice and the gripper closed on air). | `teach.py` uses the **measured** hand path as the attractor. The teacher had no lag, so the executing arm trails the taught path by `K⁻¹D·v` everywhere except at dwells (the gate forces arrival there). The effect differs from the sim's bug, but it is the same kind of deviation. It could be compensated by labelling `measured + K⁻¹D·v`. | M |
| **B2. Stiffness is not demonstrated** | The demonstration varies stiffness: 350 N/m in free space and 900 N/m at the insertion, with D = 1.8√K (`demo.py:69-70`). The transported stiffness (Sec. III-G) and the firm/compliant rotational schedule both depend on this profile. | Static 400 / 40 throughout (`teach.py:89-92`). The transported-stiffness channel carries no information. The node then forces `transit_rotational_scale` to 1.0 for a static source, whereas the sim's own expression on these labels would apply 0.25 whenever the jaws are closed. **Variable impedance is never exercised on hardware.** | **H** for the paper's claims |
| **B3. Pauses and smoothing** | Scripted dwells are kept whole. Velocities are `np.gradient` of the attractor. | Still stretches are capped at 1 s (`MAX_DWELL`), and the path is Savitzky-Golay smoothed. | L |

## C. Execution

The execution arithmetic is shared code and is faithful (see "Checked and faithful" below). The
deviations are in the node around it.

| # | Sim | Hardware | Sev |
|---|---|---|---|
| **C1. What happens after a stall** | The watchdog ends the run as `blocked`, a failure (`rollout.py:475`). | `on_stall: continue` (`tpgpt_executor_node.py:130, 596`) switches **the lag gate and its clamp off** for the rest of the run, which then follows the plan open-loop. Frozen in the X6 `method:` block, at this default, since 2026-09-29. | **H** |
| **C2. Static offset compensation** | None. Standing sag is handled inside the method, by the watchdog's `static_sag` re-baselining. | The resting error at the plan start is measured and **subtracted from every command** (`offset_compensation: z`, frozen in X6; applied at `:629`). The shared loop never sees it. The gate and watchdog judge a lag that is smaller than the controller's real position error, so part of what the sim handles inside the method is removed before the method sees it. | M |
| **C3. No infeasible-pose fallback** | When the gate shuts, it relaxes the orientation or skips to the next feasible label (`rollout.py:140, 383`). This uses IK. | Off. Same root cause as A3: it needs FR3 IK, not MuJoCo. | M |
| **C4. The gate's expected lag uses the wrong stiffness** | The controller runs the predicted K, D, and the gate's `expected = ‖K⁻¹D‖·v` uses the same K, D. | The gate uses the label K, D (400/40 → τ = 0.100 s). At the node default the robot runs the same 400 N/m with D = 2√K = 40, so the two agree. The gap was X6's fixed 600 N/m (τ = 0.082 s, override at `:610`: the gate tolerated about 22 % more lag than the arm had), which X6 no longer sets (2026-09-29). It returns whenever the fixed stiffness differs from the labels'. | L |
| **C5. Rotational gains** | k_rot 60, d_rot 12. | k_rot 30. `rotational_damping` is ignored because CRISP runs with d = −1, i.e. 2√k. | L |
| **C6. Speed** | Runs at the demonstrated speed. | `speed_scale` 0.35 (node default, and X6 since 2026-09-29; it was 0.5). The phase clock and the velocities are scaled consistently. | L |
| **C7. The gripper pauses the clock** | The jaws close while the loop keeps stepping through the demonstration's dwell labels. | `command_gripper(..., wait=True)` blocks the loop, so the phase clock stops while the gripper acts (`:625`). | L |

---

## Checked and faithful

- The execution loop: `gate_step`, `advance_step`, `advance_phase`, `belief_position`,
  `query_position`, `StallWatchdog`, `ExecutionLaw`, `stiffness_threshold`. These are imported from
  `tpgpt/policy/rollout.py`, in the same order as `rollout_policy`, with the integrate law and the
  query at the attractor. The X6 config froze `lag_tolerance` at the method's 0.035 until 2026-09-29; it now runs the executor's 0.045, like every run outside an experiment (a listed deviation, `CLAUDE.md`).
- Keypoints: `grasp_cube` / `grasp`, half extent 0.02, `include_contacts=False` (equivalent to the
  sim's `_select_parts(("box",))`), and the roll resolved in `scene_keypoints`.
- `TransportMap`, `check_diffeomorphism` on the tool path, and `transport_labels`.
- Source frames: `export_source` repeats `pipeline.run`'s `_to_tool_frame` and
  `to_grasp_convention` exactly. The taught bundle uses the same conventions.
- Grasp generation: diffusion, 200 samples, top-k 100. The TCP depth is 0.1034 m on both sides.
  Candidates are ranked by discriminator score.
- The support-approach tilt limit is 85° on both sides. `MIN_CLOUD_POINTS` is 40 on both sides.
  The policy is fitted with `GPPolicy().fit` on the transported labels.

## Suggested order of fixes

1. ~~**A1 + A3 + C3 + A2.**~~ Done, with A6 (see Status). Publish a scene cloud with the robot subtracted, and add FR3 IK. Then
   replace the selection with `_choose_grasp`'s rule: first admissible at every waypoint, least-bad
   fallback. `path_feasibility_observed` and `by_collision` need only numpy and scipy plus
   kinematics.
2. **A5.** Re-derive or remove `grasp_offset` against the new calibration, and freeze it in
   `method:`.
3. **C1.** Set `on_stall: abort` (the sim's behaviour) and freeze it.
4. **B2 / B1.** Decide whether the paper claims transported stiffness on hardware. If it does, the
   taught demonstration needs a stiffness profile (for example firm around the grasp and release
   marks), and its labels need the lag compensation.
5. **A4, A6, A7, C2, C4–C7.** Align with the sim, or record each as a stated deviation.
