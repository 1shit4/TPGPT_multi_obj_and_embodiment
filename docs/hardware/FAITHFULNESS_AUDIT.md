# Hardware vs MuJoCo: implementation differences that break faithfulness

Audit of 2026-09-21. TPGPT at `d3275d7`. Compared stage by stage:

- **Sim reference:** `tpgpt/experiments/pipeline.py::run` + `_choose_grasp`,
  `tpgpt/grasp/filters.py::filter_grasps`, `tpgpt/sim/demo.py::record_demonstration`,
  `tpgpt/sim/rollout.py::rollout_policy`.
- **Hardware:** `tpgpt/hardware/{grasp_selection,planner,teach,executor}.py`, plus
  `ycb_pointcloud_pipeline/.../tpgpt_{planner,executor}_node.py` and the X6 `method:` block.

**What this list leaves out:** differences the port forces: a one-sided camera cloud instead of the
sim's full YCB scan, CRISP taking only a diagonal stiffness, the bounded-speed approach from wherever
the arm is, `SafetyMonitor`, and wall-clock loop timing. Every item below could be implemented the
way the sim does it.

Severity: **H** changes which grasp is executed or what a trial measures. **M** changes behaviour in
some runs. **L** is a small numerical mismatch.

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
| **A7. Parameters and algorithms that differ** | duplicates 20 mm / 20°. `by_jaw_width` at the grasp height, `JAW_MARGIN` 5 mm. | duplicates 10 mm / 15°. `hand.check_jaws` (a pad-box test), `JAW_MARGIN` **0** (`hand.py:166`). | L |

A1 and A3 together are why the hardware rejects so heavily and in the wrong places. Trials 0001, 0002
and 0006 executed ranks 27, 84 and 10, rejected on jaw and plane criteria, never on the shelf.

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
| **C1. What happens after a stall** | The watchdog ends the run as `blocked`, a failure (`rollout.py:475`). | `on_stall: continue` (`tpgpt_executor_node.py:130, 596`) switches **the lag gate and its clamp off** for the rest of the run, which then follows the plan open-loop. This setting is **not frozen in the X6 `method:` block.** | **H** |
| **C2. Static offset compensation** | None. Standing sag is handled inside the method, by the watchdog's `static_sag` re-baselining. | The resting error at the plan start is measured and **subtracted from every command** (`offset_compensation: z`, frozen in X6; applied at `:629`). The shared loop never sees it. The gate and watchdog judge a lag that is smaller than the controller's real position error, so part of what the sim handles inside the method is removed before the method sees it. | M |
| **C3. No infeasible-pose fallback** | When the gate shuts, it relaxes the orientation or skips to the next feasible label (`rollout.py:140, 383`). This uses IK. | Off. Same root cause as A3: it needs FR3 IK, not MuJoCo. | M |
| **C4. The gate's expected lag uses the wrong stiffness** | The controller runs the predicted K, D, and the gate's `expected = ‖K⁻¹D‖·v` uses the same K, D. | The gate uses the label K, D (400/40 → τ = 0.100 s). The robot runs a fixed 600 N/m with D = 2√K (X6), so τ = 0.082 s (override at `:610`). The gate therefore tolerates about 22 % more lag than the arm actually has. | L |
| **C5. Rotational gains** | k_rot 60, d_rot 12. | k_rot 30. `rotational_damping` is ignored because CRISP runs with d = −1, i.e. 2√k. | L |
| **C6. Speed** | Runs at the demonstrated speed. | `speed_scale` 0.35 (node default) / 0.5 (X6). The phase clock and the velocities are scaled consistently. | L |
| **C7. The gripper pauses the clock** | The jaws close while the loop keeps stepping through the demonstration's dwell labels. | `command_gripper(..., wait=True)` blocks the loop, so the phase clock stops while the gripper acts (`:625`). | L |

---

## Checked and faithful

- The execution loop: `gate_step`, `advance_step`, `advance_phase`, `belief_position`,
  `query_position`, `StallWatchdog`, `ExecutionLaw`, `stiffness_threshold`. These are imported from
  `tpgpt/policy/rollout.py`, in the same order as `rollout_policy`, with the integrate law and the
  query at the attractor. The X6 config freezes `lag_tolerance` at 0.035, the method's value.
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

1. **A1 + A3 + C3 + A2.** Publish a scene cloud with the robot subtracted, and add FR3 IK. Then
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
