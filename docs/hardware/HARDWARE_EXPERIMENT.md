# Hardware experiment (X6) — brief for the operator and for the agent running it

**Hand this file to the agent that has the robot. It is self-contained.**

---

## 0. What this experiment is for

The paper claims that a single demonstration can be executed on an object it
never saw, by a hand with different geometry, at a destination it never visited,
with no further demonstration and no training. Simulation carries the breadth of
that claim. **This experiment carries its credibility**: a cross-embodiment claim
evaluated only in simulation is discounted, because a simulated gripper is
position-commanded with no force feedback and a simulated object has perfect
geometry.

The experiment therefore has one job: **show the same single demonstration
executing on two end-effectors of materially different morphology, across
objects and destinations, and record enough to say what failed when it failed.**

It is *not* a tuning exercise. Nothing about the method may be adjusted per hand,
per object or per trial.

---

## 1. The distinction that governs the design: pick pose is not grasp

These are two different things and conflating them will produce an experiment
that measures nothing.

| | what it is | who chooses it |
|---|---|---|
| **pick pose** | where the *object physically sits* in the workspace at the start of the trial — its position on the table and its yaw | **the operator**, by placing the object |
| **grasp** | which 6-DoF grasp *on that object* the arm takes | **the pipeline**, from the point cloud of the object as placed |

So *varying the pick pose needs no code at all*: put the object somewhere else.
Each placement produces a different point cloud, hence a different candidate set,
hence a different chosen grasp and a different map. That variation is exactly
what the experiment wants, and it arrives for free.

**Do not override the pipeline's grasp choice.** The selection rule — walk the
generator's ranked list and execute the first candidate admissible at every
waypoint of the transported path — *is part of the method*. Forcing rank 2 or 3
would be evaluating a different system. When rank 1 is rejected by the
feasibility check and rank 2 runs, that is the method working as designed, and
the record must say which rank ran and why the earlier ones were rejected.

**One optional robustness check, reported as a sentence and not a table.** On a
small subset (suggest 6 trials), after the normal trial completes, re-run the
same scene executing the 2nd and 3rd *admissible* candidates. This measures how
much the outcome depends on which admissible grasp is taken. Run it only after
the main grid is complete; it is not part of the headline result.

---

## 2. Design

**One demonstration, recorded once.**

- Kinesthetic teaching on the Franka with the **parallel-jaw hand**, gravity
  compensation on, guided by hand through a complete pick-and-place into a shelf.
- Record the commanded attractor at 20 Hz with the full label set: position,
  velocity, end-effector orientation, Cartesian stiffness, damping, gripper
  command.
- **This demonstration is used for every trial in the experiment, including every
  trial with the second end-effector.** The second end-effector never records a
  demonstration. Re-recording per hand destroys the claim the paper makes.
- Record it once, store it, hash it, and check the hash at the start of every
  session.

**Factors.**

| factor | levels | how varied |
|---|---|---|
| end-effector | **2** — the Franka parallel-jaw hand, and a second hand of different morphology | mounted |
| object | **5** (6 if time) — spread over size, aspect ratio, taper and mass; **none is the demonstrated object** | placed |
| pick pose | **3** per (hand, object) — spread across the reachable region, not clustered, with a different yaw each | placed by the operator |
| destination | **2** — two shelf slots differing in height and lateral position | commanded |

**As configured (2026-09-29, `x6_experiment/config/x6_experiment.yaml`).** Two
departures from the table above, both from measurement on the current setup:

- **4 objects**, not 5: apple, mug, pudding box (on a long narrow side), foam brick. They
  are the objects that plan into every slot (2026-09-26) other than the demonstrated
  banana. The fifth candidate, the tomato soup can, plans nowhere upright: the hand can
  only grip it around its barrel, past the ~45 deg of tilt where the map folds
  (`GRASP_LIBRARY.md`).
- **3 destinations**, not 2: `top_left`, `top_right`, `bottom_left`. `bottom_right`
  admitted no grasp on any capture and is abandoned (user, 2026-09-25). **`bottom_left` is
  81 mm from where the demonstration set the banana down** (same board), so it tests a
  new object and hand more than a new destination; the two top slots are 248 mm higher.

So 4 × 3 × 3 = **36 trials per hand**, 72 in all. Expect this to change again when the
second hand works.

**Trial count (as briefed).** 2 × 5 × 3 × 2 = **60 trials**. At roughly 3 minutes per trial
including reset, about 3 hours of running, realistically 2–3 sessions.

**Order.** Randomise the order of (object, pick pose, destination) within each
hand; complete one hand before re-mounting. Record the order actually run.

**Fallback if the second hand cannot be mounted.** Any end-effector that changes
the tool-centre offset and the jaw aperture is sufficient — printed fingertip
extensions of a different length and width will do. That is the axis the map is
supposed to be blind to. A single-hand hardware section validates the pipeline
but not the paper's central claim, so this matters more than adding a sixth
object.

---

## 3. What every trial must record

One directory per trial. Nothing is recomputed later from memory.

```
outputs/hardware/<run_id>/
  manifest.json               # provenance + settings, written once per run
  demonstration/              # the single demonstration, + its sha256
  trials/trial_0001/
    record.json               # all scalars (schema below)
    rgb.png  depth.npy        # the perception frame the grasp was computed from
    mask.png  cloud.npy       # object segmentation and its point cloud (N x 3)
    candidates.json           # the FULL ranked candidate list + per-stage verdicts
    keypoints.npz             # source and target keypoint sets, all four frames
    plan.npy                  # transported waypoints, N x 7 (xyz + quat)
    trace.npz                 # execution trace, per control step
    settled_rgb.png  settled_cloud.npy   # the scene after the object comes to rest
    video.mp4                 # fixed-camera video of the whole trial
  summary.csv                 # one row per trial, flattened
```

### `record.json`

```jsonc
{
  "trial_id": "0001",
  "run_id": "hw_2026_09_xx",
  "step_index_rate_hz": 20,          // timestamps come from step index, never wall clock

  "conditions": {
    "hand": "panda | inspire",
    "object": "…",
    "destination": "top_left | top_right | bottom_left",
    "pick_pose_index": 0,            // which of the 3 operator placements
    "order_in_session": 7
  },

  "provenance": {
    "git_sha": "…", "dirty": false, "untracked_code_files": [],
    "package_versions": {…},
    "demonstration_sha256": "…",
    "calibration": { "camera_extrinsics_sha256": "…", "hand_frame_file_sha256": "…" }
  },

  "perception": {
    "cloud_points": 1234,
    "camera_intrinsics": [...], "camera_to_base": [[...]],
    "segmentation_source": "…"
  },

  "grasp": {
    "n_candidates_generated": 100,
    "rank_executed": 1,              // 1 = the generator's top candidate
    "rejections": [                  // why every earlier rank was rejected
      {"rank": 0, "stage": "path_feasibility", "detail": "waypoint 143 in collision"}
    ],
    "executed_pose_grasp_convention": [[...]],   // 4x4
    "executed_pose_wrist_convention": [[...]],   // 4x4
    "generator_score": 0.87,
    "fell_back": false               // true if nothing was fully admissible
  },

  "map": {
    "construction": "cube_grasp_pose",
    "cube_half_extent_m": 0.02,
    "min_det_J": 0.94,
    "det_J_at_grasp": 0.99, "det_J_at_release": 0.97,
    "keypoint_residual_mm": 0.0004,
    "aim_error_mm": 0.0004,
    "orientation_error_at_grasp_deg": 0.0,
    "source_to_target_rotation_deg": 41.2
  },

  "timing_s": {                      // supports the "no training" claim with a number
    "perception": 0.8, "grasp_generation": 6.1, "filtering": 2.0,
    "map_fit": 0.03, "label_transport": 0.01, "policy_refit": 0.12,
    "total_plan": 9.1, "execution": 22.4
  },

  "outcome": {
    "placed": true,
    "placement_error_mm": 14.2,
    "placement_error_components_mm": {"x": 6.1, "y": -11.0, "z": 3.2},
    "first_failure_stage": null,     // approach|reach|grasp|lift|carry|place|settle
    "object_disturbed_before_grasp_mm": 1.4,
    "operator_notes": "…",
    "operator_aborted": false, "abort_reason": null
  }
}
```

### `trace.npz` — one row per control step

`step_index`, `ee_pose` (measured, 7), `attractor_pose` (commanded, 7),
`joint_positions` (7), `gripper_command`, `gripper_width` (if readable),
`wrench` (6, if an F/T sensor is present). **No wall-clock timestamps** — time is
`step_index / rate`.

### `candidates.json`

The **entire** ranked list with each candidate's pose and score, and for each
filter stage the verdict and reason. This is what lets the analysis say how often
rank 1 was admissible, and it costs nothing to store.

---

## 4. Measuring placement error credibly

This number appears in the paper, so the instrument must be characterised.

1. **Define the destination frame from an AprilTag on the shelf**, not on the
   object. The paper claims no markers on the object and hardware must honour it.
2. **Measure the settled object pose** by segmenting it from the post-settle
   RGB-D frame and taking the centroid of its cloud, expressed in the destination
   frame. Report the lateral error and the components separately.
3. **Characterise the instrument before the first trial.** Place one object at a
   known position and measure it **ten times without moving it**, re-running
   segmentation each time. Report the standard deviation. Put that number in the
   paper. *A 14 mm placement error means nothing until the reader knows the ruler
   is good to 2 mm.*
4. Re-measure the instrument at the end of the campaign and report both. If they
   differ, the calibration drifted and the trials between them are suspect.

---

## 5. Analysis to produce

Deliver a single `analysis.md` plus the tables below, computed from `summary.csv`.

| quantity | method |
|---|---|
| success rate, overall and per hand | proportion with a **Wilson 95% interval** (not the normal approximation — n is small) |
| difference between the two hands | **Fisher's exact test**. Unpaired, because the two hands see different grasps |
| placement error per hand | median with a **bootstrap 95% CI**; **Mann-Whitney U** between hands |
| failure attribution | count by `first_failure_stage`, with exact binomial intervals |
| does map quality predict success? | **point-biserial correlation** between `min_det_J` and `placed`; predicted to be near zero |
| how often was rank 1 executed? | histogram of `rank_executed`, plus the rejection reasons |
| cost of a new hand | median and range of `timing_s.map_fit + label_transport + policy_refit` |
| sim-to-real agreement | **Fisher's exact test** on matching (hand, object) conditions against the simulated grid; state plainly whether simulation over- or under-predicts |

**Table 5 (goes in the paper).**

| hand | object | trials | placed | median error (mm) | dominant failure stage |

with a totals row carrying Wilson intervals, and the instrument repeatability
quoted in the caption.

**Report every trial.** Aborted trials, dropped objects and operator errors are
listed with their reason and **counted in the denominator**, unless the abort was
caused by something outside the method — a network drop, an E-stop — in which
case it is listed separately and excluded, with the reason recorded.

---

## 6. Photographs and video

The figures cannot be reshot after the robot is dismantled, so capture these
during the campaign.

**Fixed rig.** One tripod camera, **the same position and framing for every
trial**, running video throughout. Inconsistent framing is what makes a
hardware figure look amateurish, and it cannot be fixed afterwards.

| # | shot | when | for |
|---|---|---|---|
| 1 | The kinesthetic demonstration being given — operator's hand on the arm, object in the gripper, mid-carry. Wide enough to show arm, object and shelf | once, at recording | **Fig. 1 top**, Fig. 6 |
| 2 | A **4-frame strip** per successful (hand, object): approach; jaws closed on the object; mid-carry; object released on the shelf | one good trial per pair | **Fig. 1 body**, Fig. 6 |
| 3 | **Both end-effectors at the same grasp pose on the same object**, photographed from an identical viewpoint | once, deliberately staged after the grid | **the money shot** — it shows the morphology gap in a way no number does |
| 4 | Workspace overview: arm, table, shelf, both destination slots, camera | once | Fig. 3 / setup |
| 5 | Each end-effector alone, on a plain background, same scale | once each | inset showing aperture and fingertip depth |
| 6 | Every failure, at the moment of failure | from the trial video | failure analysis; may not reach the paper but will be asked about |

**Also record, once:** the two hands' measured jaw aperture and tool-centre
offset, with a caliper. Those numbers go in the paper as the span of morphology
covered, and they must be measured rather than taken from a datasheet.

---

## 7. Rules that must not be broken

1. **One demonstration.** Never re-record per hand, per object or per session.
   Check its hash at the start of every session.
2. **No per-hand, per-object or per-trial tuning** of anything: not the cube
   extent, not a controller gain, not a filter threshold. If something must
   change to make a hand work at all, that is a finding — record it and report
   it; do not absorb it.
3. **Commit before running.** Every trial records the git SHA, the dirty files
   and the untracked code files. A trial whose provenance says the tree was dirty
   is a scratch trial and cannot enter the paper.
4. **Timestamps come from the step index and the control rate.** Never from the
   wall clock.
5. **Record the failures.** A campaign with no recorded failures is a campaign
   that was not recorded properly.
6. **Do not stop early on a bad run.** If a hand is doing badly, finish its grid.
   The failure attribution is a result.
