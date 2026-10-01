# Grasp-selection audit of this checkout

This is a local investigation, not a reproduction of the unavailable 94/210
experiment. Evidence lives in [outputs/grasp_audit](../outputs/grasp_audit/).
The 74-object expansion remains gated on a trustworthy baseline.

The six reconstructed policy cases succeed **2/6**; position replay using identical
pinned effective grasps and labels succeeds **5/6**. Correcting target geometry and
removing fallback does not improve the measured success rate. The audit confirms
source/target-cloud and raw/effective-pose mismatches, a stride-four sampling miss,
and an overly broad simulator contact allowance. Strict refusals remain frequent,
including cases that physically succeed. The evidence supports targeted execution
and perception work before broadening the object sweep.

## Reproduction contract

The starting commit is `e6d694e7f1b04782195344e4c7e6419eeac8f9be`.
The existing edit to `reshelving_pipeline.py` and the scratch scripts were
preserved. [preexisting.patch](../outputs/grasp_audit/preexisting.patch) records
the pre-existing tracked change. [inventory.json](../outputs/grasp_audit/inventory.json)
records local manifests, caches and videos with hashes; package versions are in
[environment.json](../outputs/grasp_audit/environment.json).

The new successful seed-zero source recording is frozen in
[source](../outputs/grasp_audit/source/source.json), including numeric labels,
placement, grasp frame, support heights and source capture. All new comparisons
reuse it. Labels SHA-256:
`da3cd63e566b157d83cee37699ebe24b4340ca6f31914592f2aa4cab4f598dd5`.
Placement SHA-256:
`e004e828e4e8f8917914c7f26f8f0dfd7062ba0132c7f2334b3bad5d3796dc83`.

The old debug videos have no matching source/cache/effective-pose manifest.
Their conditions are reconstructed from filenames, the local render script and
campaign rows, with uncertainty retained in
[provenance_matches.json](../outputs/grasp_audit/provenance_matches.json).
New runs are reconstructions, even where their outcomes match the videos.
The local YCB manifests include 84, 83, 91, 95, 66 and 90 successes under different
settings, and a position-replay manifest with 93; none establishes the missing
94-success experiment.

## Implemented controls

`python -m tpgpt.experiments.forensics CELLS OUT --source SOURCE --variant VARIANT`
accepts explicit YCB cells, an immutable source artifact and optional rendering.
It uses cached candidates only. A missing asset/cache is a setup error, not a
fresh candidate draw. Each cell commits a JSON checkpoint; restart rejects changed
settings, source hashes or Python code hashes. Code snapshots accompany new runs.

Variants are sequential interventions:

| Variant | Geometry sent to path check | Waypoints | Empty feasible set |
|---|---|---|---|
| `legacy` | source placement (historical bug) | stride four, historical endpoints | fallback |
| `target` | target object | same as legacy | fallback |
| `waypoints` | target object | every waypoint, critical endpoints explicit | fallback |
| `fallback` | target object | every waypoint | refusal |
| `strict` | target object | every waypoint | refusal; also checks entry and target palm contacts |
| `effective` | target object | every waypoint | strict checks using support-clamped poses throughout the funnel |

The `fallback` variant isolates refusal behavior. It is an ablation, with entry
and stricter contact checks explicitly disabled. The `strict` ablation adds
those checks and disables the simulator place-zone stage, so it must not be interpreted as a one-factor estimate of fallback.
The public pipeline corrects target geometry by default; exact historical
reproduction requires all settings in the runner's `legacy` variant.

Forced diagnostics are separately labeled and require both the canonical candidate
hash (poses plus scores) and the post-support-clamp effective pose. A transported
label hash can additionally pin every numeric label family before execution.
Normal strict selection never restores rejected candidates. The legacy place-zone funnel stage uses simulator shelf geometry. It is retained
in `legacy`, `target`, `waypoints` and the isolated `fallback` ablation, and disabled
in corrected strict selection. Strict selection uses the observed transported path
instead; its separate simulator query is diagnostic only.

## Filter audit and limits

- Candidate poses and original indices, stage inputs/rejections/restorations,
  sampled waypoint identities, first collision, skipped prefix, IK failures,
  halo removal count and entry-connection evidence are recorded.
- Target object geometry is a full mesh scan. Scene obstacles are observed clouds,
  with the target excluded. The path query subtracts the robot’s initial geometry;
  the initial funnel uses the raw scene cloud. Calling
  the complete input “observed geometry” would be incorrect.
- Grasp frames are converted to wrist frames before IK, using the gripper-specific
  alignment and contact offset. Carried points use `(points - grasp_position) @ R`
  in the hand frame and `position + local @ R.T` at each carried waypoint.
- The legacy target check omits all near-grasp target contacts. Strict checking
  permits named finger/pad contacts but rejects palm/arm penetration. This remains
  a geometry approximation; fingers use the initial simulator joint configuration,
  not a simulated grasp-dependent closure.
- Cloud halo removal removes points initially within 5 mm of the target from the
  carried-object obstacle query, but retains them for the robot query. This can
  erase real nearby obstacles as well as segmentation artifacts. Counts are
  reported; it is not a proven segmentation correction.
- Cloud support allowance is directional: nearby points below the object are
  allowed; sideways ones are rejected. It is not a full contact mechanics model.
- Strict entry checks sample joint interpolation from the initial arm configuration
  to the accepted join, at at most 0.02 radians per joint per sample. This checks
  a proposed connection, not the actual impedance-controlled trajectory. Actual
  approach contacts must be evaluated from the execution trace.
- The inherited minimum reachable-waypoint fraction is 0.6; unreachable samples
  are recorded but this is not an all-waypoint IK certificate.
- Neither every-waypoint checking nor entry interpolation is continuous collision
  certification. Robot self-collisions, observation holes, configuration-dependent
  finger geometry and dynamics are not certified by the observed-cloud filter.
- The simulator diagnostic retains historical allowances (including broad gripper
  contact and best-effort IK contact checks). A disagreement is evidence to inspect,
  not automatic proof that one checker is correct.

The forensic trace uses simulation timestamps at every **control step**, including
approach and settling, rather than every internal MuJoCo integration substep.
Overview and close side renders share those timestamps. The close camera is changed
only for rendering and restored before the next perception observation. Event
frames include first target contact, closure, lift above 20 mm, release and settling;
an event that never occurs is recorded as absent.

## Visual observations before interventions

Inspected eight-frame sequences from all six requested existing cases, plus event
frames from the new Panda reconstruction. These are observations, not causal
claims:

| Case | Visible behavior | Evidence |
|---|---|---|
| Panda/meat | contacts/tips can; leaves it on table | [old sequence](../outputs/grasp_audit/grip_panda_meat_top_left_sheet.jpg), [new side events](../outputs/grasp_audit/baseline/cell_000/sideview_events.jpg) |
| Rethink/meat | descends near can; retreats without carrying | [sequence](../outputs/grasp_audit/grip_rethink_meat_top_middle_sheet.jpg) |
| Robotiq85/meat | grasps and carries can into shelf region | [sequence](../outputs/grasp_audit/ok_robotiq85_meat_sheet.jpg) |
| Robotiq85/sugar | lifts box and reaches shelf region | [sequence](../outputs/grasp_audit/robotiq85_sugar_top_middle_sheet.jpg) |
| XArm/hammer | lifts hammer; long handle/head rotates beside shelf | [sequence](../outputs/grasp_audit/xarm_hammer_top_middle_sheet.jpg) |
| Panda/banana | lifts banana; object is later obscured by shelf | [sequence](../outputs/grasp_audit/panda_banana_topleft_sheet.jpg) |

The new Panda/meat baseline chose candidate 19 without path fallback. First target
contact occurred at recorded execution frame 80, closure at 187 and release at 430.
Its object moved 17.2 mm before closure and lifted only 6.6 mm. Measured hand error
at closure was 24.0 mm, while commanded error was 1.8 mm. The peak object height
preceded closure, consistent with tipping/pushing rather than successful pickup.
[Trace](../outputs/grasp_audit/baseline/cell_000/trace.npz),
[measurements](../outputs/grasp_audit/baseline/cell_000/measurements.json),
[filter record](../outputs/grasp_audit/baseline/cell_000/result.json),
[overview video](../outputs/grasp_audit/baseline/cell_000/workspace.mp4),
[side video](../outputs/grasp_audit/baseline/cell_000/sideview.mp4).

This supports an execution/contact hypothesis; it does not yet distinguish
controller tracking, a physically blocked approach and a poor effective grasp.
The baseline pilot videos are synchronized to the rollout probe and omit the
initial approach; the completed runner records all control steps for subsequent
runs. The pilot must not be presented as having that later instrumentation.

## Draft corrections

1. Replace “observed geometry” with “full target scan plus observed scene cloud.”
2. Report fallback executions explicitly; a selected candidate is not necessarily
   feasible. Include strict refusals in the scheduled-cell denominator.
3. Replace “whole-path/every-waypoint certification” with the actual discrete
   stride and endpoint policy; even stride one is not continuous certification.
4. Describe the source as the frozen local seed-zero recording, not an unavailable
   paper/experiment demonstration. Do not transfer a historical success rate to it.
5. Distinguish scheduled success, conditional execution success, setup errors and
   refusals. A scene-build error must remain in the denominator.
6. Identify candidates by cache hash and effective support-clamped pose, not index
   alone; filtering the raw pose does not validate the executed pose.

## Broader sweep gate

The five locally downloaded YCB assets are sugar, meat, mug, hammer and banana.
The prepared 35-cell and 210-cell grids are in
[grid_35.json](../outputs/grasp_audit/grid_35.json) and
[grid_210.json](../outputs/grasp_audit/grid_210.json). The 74-object expansion is
not authorized by this baseline alone. Before choosing a fixed ten-object set,
record asset availability, mesh/scale/mass/collision provenance and development
selection criteria; then freeze the set and choose fresh evaluation conditions.
Do not choose the ten objects retrospectively by evaluation success.

## Baseline measurements

All six scheduled reconstructions executed. Two succeeded, four failed final
placement, and none refused or had a setup error: overall and conditional execution
success were both 2/6. These counts are not estimates for the old 210-cell campaign.
The synchronized [trace plot](../outputs/grasp_audit/baseline/traces.png) separates
hand tracking from object lifting. Green and red lines mark closure and release.

| Cell | Candidate | Final outcome | Hand error at closure | Lift | Final XY error |
|---|---:|---|---:|---:|---:|
| Panda/meat, top left | 19 | failure | 24.0 mm | 6.6 mm | 512.1 mm |
| Rethink/meat, top middle | 0 | failure | 28.2 mm | 5.5 mm | 434.2 mm |
| Robotiq85/meat, top middle | 27 | success | 10.6 mm | 409.7 mm | 5.2 mm |
| Robotiq85/sugar, top middle | 54 | success | 7.9 mm | 349.1 mm | 35.3 mm |
| XArm/hammer, top middle | 12 | failure after lift | 11.2 mm | 269.6 mm | 143.7 mm |
| Panda/banana, top left | 48 | failure after lift | 4.4 mm | 407.1 mm | 71.2 mm |

Each row's evidence bundle contains `result.json` (filter and candidate record),
`measurements.json`, `trace.npz`, `frames.json`, `transported_labels.npz`,
`workspace.mp4`, `sideview.mp4` and both event sheets:
[Panda/meat](../outputs/grasp_audit/baseline/cell_000/result.json),
[Rethink/meat](../outputs/grasp_audit/baseline/cell_001/result.json),
[Robotiq85/meat](../outputs/grasp_audit/baseline/cell_002/result.json),
[Robotiq85/sugar](../outputs/grasp_audit/baseline/cell_003/result.json),
[XArm/hammer](../outputs/grasp_audit/baseline/cell_004/result.json),
[Panda/banana](../outputs/grasp_audit/baseline/cell_005/result.json).
All six baseline side-view event sequences have been inspected. The hammer hangs
outside the cubby at release; the banana's release is partly occluded, so its exact
final support contact cannot be inferred from the video alone.

## Runtime instrumentation validation

After selection, the forensic runner disables camera **observables** that are no
longer used by the executor. Candidate generation and all observed-cloud checks
still receive the original camera observations. Explicit diagnostic video renders
remain enabled. A live-simulator regression compared 12 steps with these camera
observables enabled and disabled and found exactly equal joint-position trajectories.
This optimization changes no controller gains, physical parameters or object choices.
The baseline is also repeated under the completed instrumentation before interpreting
its intervention comparisons.

## Paired interventions and pinned execution

The [synchronized evidence viewer](../outputs/grasp_audit/report.html) links videos,
inspected event frames, complete control-step traces and candidate decisions.
[paired_results.json](../outputs/grasp_audit/paired_results.json) records identity
checks and individual transitions. All six policy/replay pairs match source,
candidate-cache hash, effective pose and transported-label hash exactly.

| Six scheduled cases | Success | Refusal | Setup error | Execution failure | Conditional execution success |
|---|---:|---:|---:|---:|---:|
| Historical inputs and legacy fallback | 2/6 | 0 | 0 | 4 | 2/6 |
| Target-cloud correction only | 1/6 | 0 | 0 | 5 | 1/6 |
| Then every waypoint | 1/6 | 0 | 0 | 5 | 1/6 |
| Then remove fallback | 0/6 | 3 | 0 | 3 | 0/3 |
| Entry/contact strict checks | 0/6 | 5 | 0 | 1 | 0/1 |
| Then filter effective support-clamped poses | 0/6 | 5 | 0 | 1 | 0/1 |
| Pinned position replay of original grasps, diagnostic | 5/6 | 0 | 0 | 1 | 5/6 |

Target correction changes sugar from a successful candidate 54 to candidate 75,
which fails execution. Hammer switches from candidate 12 to a rejected fallback
candidate 90 and still fails. Every-waypoint checking detects a Robotiq85/meat
collision at waypoint 65 that stride four misses, but legacy fallback restores a
candidate that physically succeeds. This is why refusal counts must accompany
headline success rates: stricter checking can reject successful executions.

Removing fallback refuses Robotiq85/meat, XArm/hammer and Panda/banana. Hammer's
initial collision stage rejected every surviving candidate; the historical funnel
had restored them. Entry/contact strict checking additionally rejects Panda/meat and
Robotiq85/sugar for target contacts. Rethink/meat still executes and fails.

Pinned position replay changes Panda/meat, Rethink/meat and Panda/banana from failure
to success. The two Robotiq85 controls remain successful; hammer still fails
(final XY error 229 mm, versus policy's 144 mm). This supports executor sensitivity
for the three rescued failures. It does not establish that changing one controller
gain would fix them, and no gains or physical parameters were tuned here.

The six complete baseline control-step traces reproduce the pilot hand, command,
gripper and phase arrays **bit for bit**:
[reproduction checks](../outputs/grasp_audit/baseline_reproduction.json).
Actual initial approach traces contain no penetrating robot/scene contacts at the
sampled control instants, although Panda and Rethink still exhaust their 345-step
approach budget without reaching the desired start. This distinguishes a clear
sampled connection from dynamic arrival:
[baseline entry evidence](../outputs/grasp_audit/legacy_verified_actual_entry.json),
[strict entry evidence](../outputs/grasp_audit/strict_verified_actual_entry.json).

## Inspected filter disagreements

For sugar candidate 54 at waypoint 68, one scene point causes the carried-object
rejection. It is 6.153 mm from the initial target scan, just beyond the 5 mm halo
removal radius, and 178.4 mm above the table. For hammer candidate 12, the single
point is 5.265 mm from the initial scan and 22.9 mm above the table. The cloud plots
and identical-pose close renders have been inspected:

- Sugar: [cloud plot](../outputs/grasp_audit/disagreement_robotiq85_sugar/cloud_disagreement.png),
  [close render](../outputs/grasp_audit/disagreement_robotiq85_sugar/close.png),
  [numeric evidence](../outputs/grasp_audit/disagreement_robotiq85_sugar/evidence.json).
- Hammer: [cloud plot](../outputs/grasp_audit/disagreement_xarm_hammer/cloud_disagreement.png),
  [close render](../outputs/grasp_audit/disagreement_xarm_hammer/close.png),
  [numeric evidence](../outputs/grasp_audit/disagreement_xarm_hammer/evidence.json).

Pixel provenance was recovered by rebuilding the identical initial scene and
matching the original unprojected depth raster exactly. Sugar's point comes from
sideview pixel (137, 152), hammer's from birdview (106, 124), and the newly inspected
meat point from workspace (158, 85), in top-down row/column coordinates. All three
have background segmentation ID 0, target ID 1, and are **one pixel outside the
target mask**. Their RGB patches have been inspected:
[sugar](../outputs/grasp_audit/disagreement_robotiq85_sugar/sideview_pixel_close.png),
[hammer](../outputs/grasp_audit/disagreement_xarm_hammer/birdview_pixel_close.png),
[meat](../outputs/grasp_audit/disagreement_robotiq85_meat/workspace_pixel_close.png).
Each directory also contains `pixel_provenance.json` with the exact zero-distance
world match and mask neighborhood.

Meat candidate 27 is rejected at waypoint 65 by a point 5.008 mm from the initial
scan, only 0.008 mm beyond the halo threshold. Its cloud plot and identical-pose
render were also inspected:
[evidence](../outputs/grasp_audit/disagreement_robotiq85_meat/evidence.json),
[plot](../outputs/grasp_audit/disagreement_robotiq85_meat/cloud_disagreement.png),
[render](../outputs/grasp_audit/disagreement_robotiq85_meat/close.png).
This establishes pixel-edge provenance and strongly supports residual target-edge
observations, but does not isolate whether depth rasterization, segmentation
rasterization or finite surface sampling creates the discrepancy. Enlarging the
halo until the experiment succeeds could hide real obstacles and was not done.

There is a separate confirmed allowance error in the legacy simulator diagnostic:
it treats any `gripper0...` geom as a finger. Sugar's raw identical-pose contacts
include **5.119 mm of palm/object penetration**, which that diagnostic labels
allowed. Hammer has no raw contacts at the inspected pose. Thus “simulator says
clear” is not interchangeable with “no geometric overlap.” Strict observed checking
rejects palm contact (candidate 75 fails at waypoint 67); its regression test
separates palm/arm from finger/pad contacts.

Forced diagnostic policy execution of the rejected sugar candidate 54 still
succeeds with the original cache, effective pose and full labels pinned:
[result](../outputs/grasp_audit/rejected_sugar/cell_000/result.json),
[overview](../outputs/grasp_audit/rejected_sugar/cell_000/workspace.mp4),
[close side view](../outputs/grasp_audit/rejected_sugar/cell_000/sideview.mp4),
[control-step contacts](../outputs/grasp_audit/rejected_sugar/cell_000/simulation_steps.json).
The first-contact, closure, lift, release and settling images were inspected;
release is occluded in the close side view, so it cannot independently certify
set-down. This run is explicitly excluded from ordinary-selection success counts.


## Support-clamped filtering intervention

A second implementation mismatch was measured: the early funnel checked raw cached
grasps, while execution later clamped them above the support. For the hammer,
candidate 12 shifted by (−6.034, +3.556, +21.037) mm. On the saved self-filtered cloud,
raw candidates all failed collision checking, while effective candidates 12 and 90
passed. This exploratory comparison uses the path cloud rather than the funnel's
raw cloud, so the final `effective` run measures the correction through the actual
pipeline. Across the 35-cell subset, seven funnels change intermediate decisions,
but no final outcome changes. [Comparison](../outputs/grasp_audit/clamp_filter_comparison.json).

The corrected public strict mode now filters support-clamped poses without changing
the cached candidates or scores. Historical variants explicitly retain raw-pose
filtering. This is a separate intervention, not a controller or grasp-generation
change.


## Scheduled campaign coverage

The design uses seven hands × five downloaded objects × three pick placements
(`P0`, `P1`, `P2`) × two destinations (`top_middle`, `top_left`). Every reconstructed
cell fixes seed zero, a 600-step policy budget and both closure/release gates. These
are the six-condition design dimensions, not a claim of matching historical
800-step campaign settings. The 35-cell subset fixes `P0` and `top_middle`.

Only 32 unique hand/object/pick cache entries are available, covering 64 of the 210
scheduled conditions. The other 146 are recorded setup errors. The 35-cell subset
has 24 executable cells and 11 missing caches. No server was called and no candidates
were regenerated. [Asset inventory](../outputs/grasp_audit/assets.json) records the
local object and hand assets. In the corrected runs, missing-cache checkpoints
are revalidated against unchanged scene/generation/perception code, the same source,
and continued cache absence; executed results are never reused across interventions.

| Run | Scheduled | Success | Refusal | Setup error | Execution failure | Overall success | Conditional execution success |
|---|---:|---:|---:|---:|---:|---:|---:|
| 35 legacy | 35 | 9 | 0 | 11 | 15 | 25.71% | 37.50% |
| 35 entry/contact strict | 35 | 1 | 19 | 11 | 4 | 2.86% | 20.00% |
| 35 corrected strict (effective poses) | 35 | 1 | 19 | 11 | 4 | 2.86% | 20.00% |
| 210 legacy | 210 | 26 | 0 | 146 | 38 | 12.38% | 40.63% |
| 210 entry/contact strict | 210 | 3 | 56 | 146 | 5 | 1.43% | 37.50% |
| 210 corrected strict (effective poses) | 210 | 3 | 56 | 146 | 5 | 1.43% | 37.50% |

Setup errors stay in the scheduled denominator. Conditional execution success divides
only by successes plus execution failures. These are local cache-coverage-limited
measurements and cannot support a rate claim for a complete 210-cell experiment.


## Verification and rerunning

The final serial full suite passed **800 tests**, with **10 skips** and seven
warnings: [full output](../outputs/grasp_audit/final_tests.log). The ten skips require
an unavailable GraspGen-X server; cached candidate simulation is available. The
warnings include existing approach-arrival timeouts. New regressions cover differing
source/target clouds, collisions between stride-four samples, critical endpoints,
entry interpolation, strict versus legacy fallback, allowed finger/support versus
forbidden palm/obstacle contact, simulator state restoration, candidate and label
pins, source tampering, checkpoint recovery, scene-build errors, effective-pose
filtering and camera-observable dynamics equivalence.

Use `/home/sra/mujoco_env/bin/python` with `PYTHONPATH=.`, `MUJOCO_GL=egl`,
`OPENBLAS_NUM_THREADS=1`, `NUMBA_CACHE_DIR=/tmp/tpgpt-numba`,
`MPLCONFIGDIR=/tmp/tpgpt-mpl`, `TPGPT_GRASP_CACHE=outputs/grasp_cache_s0`, and
`GRASPGENX_GRIPPER_CFG_DIR=/home/sra/gripper_descriptions_hf`. For example:

```sh
/home/sra/mujoco_env/bin/python -m tpgpt.experiments.forensics \
  outputs/grasp_audit/representatives_frozen.json outputs/grasp_audit/new_effective_run \
  --source outputs/grasp_audit/source --variant effective
```

Use a new output directory when code/settings change. Every run includes code hashes
and a source snapshot. The output scripts and input JSONs preserve the exact
comparisons; generated artifacts remain local under the repository's ignored
`outputs/` directory.


All 210 scheduled checkpoints completed in each variant. Correcting the funnel's
pose basis changes no final outcomes relative to the entry/contact strict ablation.
From legacy to corrected strict, 23 of 26 successes become refusals, three remain
successes, 33 of 38 execution failures become refusals, and five remain failures.
All 146 setup errors persist. This measures a substantial loss of acceptance of
physically successful cases, not an established collision-certification improvement.
Per-cell transitions and source/cache identity checks are in
[paired_results.json](../outputs/grasp_audit/paired_results.json).

## Complete video evidence

The final [viewer](../outputs/grasp_audit/report.html) uses full-approach recordings
under `baseline_complete/`, with synchronized overview/close views and every-control-step
contacts. Inspection uses both event frames and eight uniformly spaced frames from
each complete overview video. This is frame-based inspection, not a claim to have
visually examined every integration step. Close views can be blocked by the robot
at release; missing visibility is not treated as a clear contact check.

Panda/meat and Rethink/meat tip the can near closure, fail to lift it 20 mm and leave
it on the table. Robotiq85/meat lifts and carries it into the shelf, with the final
contact obscured. Robotiq85/sugar lifts the box and the final close settling frame
clearly shows it standing upright on the shelf. The former pilot clips remain
available for provenance; their timestamps are not interchangeable with these
full-approach clips.

The final close view resolves the banana's previously occluded outcome: it lies on
the shelf after release but finishes **71 mm from the requested slot**. This is a
placement-accuracy failure, not a failure to lift. Hammer is lifted but its head
hangs outside the shelf at settling. The identical-grasp replay rescue for banana,
and persistent failure for hammer, should be interpreted in that light.

All six full rendered baselines reproduce the earlier hand, command, grip and phase
arrays exactly: [render repeatability](../outputs/grasp_audit/final_render_repeatability.json).
The inspected full-frame evidence is linked in the viewer; numeric placement and
contact measurements resolve events obscured in either camera.


The target-only accepted sugar candidate 75 was also executed as a separately
labeled, pinned diagnostic. Its trace reproduces the target-only run exactly. The
inspected video shows pickup followed by a stalled carry before the shelf; release
never occurs and is explicitly absent from the events. Actual contacts include
palm/object penetration, but the trace alone does not establish that overlap as the
cause of the tracking failure.
[Result](../outputs/grasp_audit/accepted_sugar75/cell_000/result.json),
[events](../outputs/grasp_audit/accepted_sugar75/cell_000/events_sheet.jpg),
[contacts](../outputs/grasp_audit/accepted_sugar75/cell_000/simulation_steps.json),
[actual-contact audit](../outputs/grasp_audit/actual_contacts.json).

The 24 executable cells shared by the 35-cell and 210-cell legacy grids also match
hand/command/grip/phase arrays exactly:
[grid repeatability](../outputs/grasp_audit/grid_repeatability.json). All scheduled
checkpoint totals, paired source/cache identities, report/viewer links and the
preserved pre-existing diff were checked. After clarifying geometry-provenance
metadata and documentation, the 17 audit unit tests passed again.

The remaining development priorities are to resolve depth/segmentation edge
residuals without erasing real obstacles, model finger configuration and allowed
contacts consistently, and diagnose execution tracking with the pinned replay
controls. Until those checks and missing-cache coverage are addressed, neither a
74-object expansion nor a fresh ten-object evaluation is presented as validated.
