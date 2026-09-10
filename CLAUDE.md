# CLAUDE.md — working rules and context for the TPGPT project

## What this project is

A research-grade implementation of **Franzese, Prakash, Della Santina, Kober,
"Generalizable Motion Policies through Keypoint Parameterization and
Transportation Maps"** (arXiv:2404.13458v2, `TPGPT.pdf` in this directory).

Long-term goal: transport policies across different **objects** and different
**embodiments** from a **single** source demonstration, with a publishable
implementation and analysis. Everything here is built to serve that.

Robotics-side findings, results and decisions live in **`ROBOTICS_NOTES.md`**.
This file is process, environment and code rules.

## Environment — read this before running anything

| Item | Value |
|---|---|
| Interpreter | `/home/ishita/mujoco_env/bin/python` (Python 3.10) |
| **Not** the default | `python3` resolves to conda 3.13, which has none of the packages |
| Simulator | MuJoCo 3.7.0 + robosuite 1.5.2 (source install at `/home/ishita/robosuite`) |
| Grasp generation | GraspGen-X, **out of process**. See "Two-process setup" below. |
| GL backend | `MUJOCO_GL=egl`. **`osmesa` is broken on this box** (PyOpenGL cannot bind). |
| Compute | **CPU only. No GPU.** Never add a CUDA dependency. |
| Memory | Tight: ~2.5-3.5 GiB free, several GiB of swap already in use. |

```bash
export MUJOCO_GL=egl
/home/ishita/mujoco_env/bin/python -m pytest tests/unit -q          # fast, no sim
/home/ishita/mujoco_env/bin/python -m pytest tests -q               # everything
```

### Two-process setup for grasp generation

GraspGen-X cannot run in `mujoco_env`. It needs Python 3.11 with
`diffusers==0.11.1` and `huggingface-hub==0.25.2`, which conflict with this
project's stack, and its model is 1.6 GB. It runs as a **long-lived ZMQ server**
in its own conda environment; TPGPT talks to it with a torch-free client.

```bash
python -m tpgpt.grasp.server                    # status, and how to start it
/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/scripts/run_server.sh --daemon
```

- Server: `127.0.0.1:5556`, CPU, **2.6 GB resident**, 3.3 s model load,
  4-12 s per inference. Budget that against MuJoCo's ~1 GB.
- Assets come from the sibling checkout via `GRASPGENX_GRIPPER_CFG_DIR` and
  `GRASPGENX_PROJECT_ROOT`. Nothing is duplicated into this repo.
- `franka_panda`, `robotiq_2f_85` and `robotiq_2f_140` are what a running
  server usually has resident, but **any registered gripper works without a
  restart**: `zmq_server.py:110` loads a sampler *lazily* on the first `infer`
  request naming it, then caches it. `python -m tpgpt.grasp.server` reports
  `loaded grippers`, which is what has been asked for so far, not a whitelist.
  Budget the first call for that hand accordingly.
- `tests/integration/test_graspgen.py` **skips** when the server is unreachable,
  so the suite still runs offline. Never start the server implicitly from a test.

### Memory and CPU rules
- Check `free -h` before any long simulation run.
- Render at 256x256 by default; stream frames to disk, never accumulate a long
  episode of frames in RAM.
- Always close environments explicitly (`env.close()`).
- Run pytest **serially**. Do not use `pytest-xdist`/`-n auto`.
- Headless physics costs ~160 control-steps/s and ~800 MB RSS; add ~250 MB for
  offscreen rendering.

## How to report back, and how to write anything down

**Write in layman's language, keeping every technical term, number and file
reference.** One explanation, not two: do **not** append a separate
"plain-language summary" at the end. The main text is the accessible one.

- Define a term the first time it appears — "the attractor (the point the spring
  is anchored to, which the arm chases)" rather than "the attractor".
- Show the arithmetic behind any claim that rests on it, not just the result.
- Explain the mechanism step by step rather than stating a conclusion and moving
  on. A paragraph of mechanism beats a sentence of result.

The same applies to everything written into a file the user reads —
`ROBOTICS_NOTES.md`, this file, generated reports, campaign indexes. For every
table, say what was varied, what was held fixed, and what conclusion it supports.
For every number, say how it was measured and with what instrument. For every
conclusion, say what would make it false. Notes that carry only conclusions are
how three findings in `ROBOTICS_NOTES.md` ended up wrong (§7.26).

## Rules for changing code

1. **Run the tests after every change.** The unit suite takes ~10 s; the
   full suite ~4 min. Run the full suite before declaring anything done.
2. **Cite the paper in docstrings.** Every function implementing theory names
   the section or equation it comes from. This is the point of the project.
3. **Deviations from the paper must be documented** in `ROBOTICS_NOTES.md` with
   the measurement that motivated them, not silently absorbed into code.
4. **Measure before concluding, and correct the record when a measurement
   supersedes an earlier one.** `ROBOTICS_NOTES.md` §2.5 contains a worked
   example: an early diagnosis blamed the time belief for a rollout stall, and
   a later measurement showed the cause was under-regularisation. The note now
   carries both, marked as a correction.
5. **No wall-clock time in recorded data.** Timestamps come from the step index
   and the control frequency.
6. **Seed before `env.reset()`**, never after.
7. Prefer extending an existing module over adding a parallel one; check
   `tpgpt/transport/` and `tpgpt/utils/` first.

## Things that will bite you again

Each of these cost real debugging time. Full detail in `ROBOTICS_NOTES.md`.

- **A policy is not a transportation map.** They share the GP code but need
  opposite priors: the map interpolates sparse keypoints exactly; the policy
  must regularise (`noise_variance` 1e-2) and needs a length scale set in
  standardised units, not from label spacing. Getting this wrong fails
  *silently* — the policy fits its labels perfectly and commands zero velocity
  2 cm away.
- **Labels are the commanded attractor, not the measured pose.** An
  impedance-controlled arm lags its attractor by 20-45 mm. Recording the
  measured pose and executing it as an attractor applies that lag twice.
- **robosuite rescales actions onto `output_max`.** Leaving it at the default of
  1 divides every commanded torque by the ratio; the impedance law then looks
  broken when it is only attenuated.
- **The arm's reachable envelope, not the paper, sets the scene geometry.**
  End-effector x saturates near 0.24 m in this arena.
- **Check `free -h` before long runs**, and never accumulate rendered frames in
  memory — stream them with `FrameWriter`.
- **Segmentation instance ids are offset by one.** robosuite adds 1 so that 0
  means background; using the raw index masks the background and yields a 2.5 m
  "object cloud" that looks like a calibration failure, not an off-by-one.
- **Observation images are vertically flipped** relative to the camera matrix,
  and the pixel vector is `[col*z, row*z, z, 1]`. Either mistake moves the cloud
  about 27 cm, in opposite directions.
- **A grasp pose is at the gripper base, not the fingertips.** Converting it
  needs the hand's own TCP depth (103-195 mm across the registry) plus a
  rotation about the approach axis. Never assume identity: unmeasured grippers
  refuse conversion deliberately.
- **Point clouds sent to GraspGen-X must be capped** (8192 points). Its outlier
  removal is `torch.cdist(X, X)` and a large cloud OOM-kills the server.
- **`env.table_top` is not the plane objects rest on.** robosuite's `TableArena`
  puts the *top* surface at `table_offset` and hangs the thickness below it, so
  the property reads 23 mm high. Both scenes derive their shelf heights from it
  consistently, so it is left alone; use `table_offset[2]` for the table
  surface and `slot_poses()` for a shelf board, which is exact.
- **A keypoint built from a thin band of a point cloud is unstable.** Any
  statistic taken over a slice -- a percentile's neighbour, a band centroid --
  moves several millimetres between two samplings of the same object. `phi`
  interpolates keypoints *exactly*, so that wander is not averaged away: it
  becomes a local deformation, and Eq. 11 turns it into gripper rotation. Two
  such bugs cost 14 degrees of yaw and 2 of 20 episodes. Measure only the one
  coordinate that needs measuring; take the rest from the fitted box.
- **`det(J) > 0` fraction is a weak diagnostic; the *minimum* determinant is
  the useful one.** An under-observed object gives a collapsed box, a perfect
  keypoint residual and 100% positive determinants, with `min det(J)` at 0.008
  against 0.67 for a well-observed one.
- **The grasp's *position* along an object must not enter the keypoints.** It
  contradicts the shape keypoints and folds the map. Only the grasp's
  *orientation* belongs there, via the task frame. Detail in
  `ROBOTICS_NOTES.md` section 6.4.
- **The attractor clamp and the lag gate must agree on "too far".** They were
  computed from different speeds -- the clamp from the demonstration's fastest
  label, the gate from the current one -- so the clamp parked the attractor
  beyond the gate's own reopening threshold and then dragged it along behind the
  arm. Neither could resolve; one run froze for 112 of 361 steps. `§7.14`.
- **Measure progress on the phase, not on the gate being zero.** A gate held at
  0.05 by a constant sag is not zero and advances the clock at a twentieth of
  nominal: three runs delivered the object to within 5-30 mm of its slot and
  were scored failures because the clock never reached the segment that opens
  the fingers. `§7.16`.
- **`contact_offset` and `alignment_rotation` look a hand up by identity.**
  They now accept a registry key too; before, a string fell through to "never
  measured" and returned a *zero* offset -- a plausible number, so a 41 mm frame
  error read as the arm missing its target. Zero means unmeasured, never
  "no offset".
- **Do not flip rows when projecting world points onto a camera image.**
  `perception.cameras` flips the depth and mask buffers *before* unprojecting,
  so the camera matrix's rows are already the display-orientation ones. Flipping
  again in the overlay leaves the columns correct and only the height wrong,
  which reads as a keypoint that has drifted rather than a broken projection.
  `§7.17`.
- **GraspGen-X's planner is an unseeded diffusion model.** The same scene gets a
  different candidate set every call, and it decides the task. Anything
  comparative must go through `tpgpt.grasp.cache`, or it is comparing random
  draws. `§7.15`.
- **A cloud's axis-aligned extent is not the object's size.** A 30 x 100 mm box
  yawed 45 degrees measures 92 x 92. Comparing that to a jaw aperture says a
  perfectly graspable object is impossible. `§7.18`.
- **Small objects are barely seen at 256x256.** Cloud size runs 975 points for
  the cereal down to **17** for the lemon, against a `MIN_CLOUD_POINTS` of 40,
  so the smallest objects are rejected outright. That much is a direct
  observation and is solid. Whether cloud size *predicts task success* is
  **not** established — two opposite conclusions were drawn from the same data
  and both are withdrawn. `§7.18`, `§7.26`.
- **Attribute a failure to a stage, not to the outcome.** `policy_stalled` and
  `placed_in_the_wrong_place` name where a run *stopped*. Use
  `tpgpt.experiments.diagnose`, which watches the object and reports the first
  of approach/reach/grasp/lift/carry/place/settle that failed.
- **Transport the fingertip path, not the wrist path.** The keypoints are
  anchored where the hand holds the object; the demonstration is recorded at
  `grip_site`, 41-117 mm away depending on the hand. A map is exact only at its
  keypoints, so warping the wrist spends that guarantee on the wrong point. The
  argument is geometric and does not need a success rate; the size of the
  improvement was measured on withdrawn campaigns and is not currently known.
  `§7.20`.
- **A null result from a diagnostic deserves as much suspicion as a surprising
  one.** The frame problem above was tested early and recorded as *disproved*,
  because the instrument (`contact_offset`) was returning zeros and the test was
  comparing a quantity with itself. Assert the instrument reads something
  non-zero before believing "no difference".
- **Never change the system and the measurement in the same sitting.** Three
  findings in `ROBOTICS_NOTES.md` had to be retracted for exactly this, and
  254 MB of campaign results had to be deleted because the code that produced
  them was never committed and cannot be identified. Commit before running a
  campaign; record the git SHA and the full settings in the manifest. `§7.26`.
- **Prefer a geometric measurement to an end-to-end one.** Aim, `det(J)`,
  keypoint residual and path curvature are all computable from the fitted map
  with no physics: milliseconds instead of 25 seconds, deterministic instead of
  stochastic, and with no rollout in between to confound the answer. Use
  end-to-end runs to *confirm* a result, never to find one. `§7.26`.
- **The impedance lag is a velocity-tracking property, not a settling error.**
  With a parked setpoint this controller settles to zero error. It lags only
  because the setpoint keeps *moving* and `velocity_desired` is never passed, so
  the damper fights absolute velocity rather than velocity error and the spring
  must stay stretched by `K^-1 D v` (~24 mm at 0.25 m/s) to supply the force.
  Before "fixing" it, read `§2.7`: the lag is transport-invariant here and lands
  the arm where the demonstrating arm actually was, so it may be correct.
- **The tabletop scene handed over objects that were still falling, inside the
  robot. Read `§7.32` before trusting any number measured in it.** Three
  defects, all fixed: `SETTLE_STEPS = 60` was 0.12 s where the cereal needed up
  to 0.96 s, so a 150 mm box toppled 35-78 mm *on its own* after handover; the
  arm's rest pose sat inside the sampling region, so objects were created
  interpenetrating the gripper (4.85-26.77 mm on four of six hands) and ejected
  on the first step; and objects are dropped 25 mm, not the 2 mm `z_offset`
  claims, because each declares a `bottom_offset` that much larger than its true
  half height. The scene consequently **differed by up to 154 mm between
  grippers from the same seed**. It is now identical across hands to 0.00 mm.
- **Settling is detected by displacement, never by velocity.** An object at rest
  on the table carries 8-23 mm/s and 0.18-0.65 rad/s of solver jitter
  indefinitely, oscillating in sign, so no meaningful velocity threshold is ever
  met while the object does not move 0.1 mm in a second. And **one still window
  is not enough**: a box balanced on an edge pauses before it tips, which is why
  the gate needs four consecutive windows.
- **A shared joint configuration does not give a shared start pose.** The same
  `init_qpos` puts a 97 mm-deep Panda hand and a 270 mm-deep Robotiq 2F-140 in
  completely different places. `HOME_TCP` is a **fingertip** target and IK solves
  per hand for the wrist behind it, which is what makes every arm start at the
  same point (3-4 mm) — the same conversion the replay does.
- **Gravity compensation during a settle must touch only the arm's DOFs.**
  `qfrc_applied[:] = qfrc_bias` across the whole model makes the *objects*
  weightless: they hang at their placement heights, and every check reports
  perfection — no interpenetration, no cross-gripper spread, all upright —
  because nothing has moved. **A settle that freezes what it is meant to settle
  looks exactly like a settle that works.** Three integration tests caught it
  when the purpose-built verification did not.
- **Every `calibrated_depth` in the registry was re-measured after `§7.32` and
  all seven moved, by up to 30 mm.** `contact_offset` derives from it, so the
  Panda's wrist-to-fingertip offset was **41.1 mm** throughout the project and is
  **11.1 mm**. Any note quoting 41.1 mm predates this. The value feeds
  `_to_tool_frame`, `grasp_to_eef_pose`, the replay's IK and the home pose, and
  19.7 mm of tool-offset error was separately measured turning a 28 mm placement
  into a 232 mm one — so this is not a rounding detail.
- **`VERIFIED_PAIRS` is 7, not 8: the UMI is out.** On the corrected scene it
  lifts nothing (0 of 13 swept depths), reaches **0 of 48** candidate home poses
  where the other eight reach 48 of 48, and had 8 of 8 Tier 2 paths unreachable.
  Convertible, not executable, like the Inspire hand. The reasons are stored in
  `gripper_frames.json`, not left as a silent `None`.
- **`+1` closes every hand in the registry, and the jaw *reading* is what
  differs.** Robosuite's per-gripper `format_action` sign multipliers look
  inverted between hands (`[-1, +1]` for the Panda, `[+1, -1]` for the Robotiq
  2F-140), but they compensate for opposite joint conventions in the two models;
  measured on all nine, the fingers converge on `+1` and reopen on `-1`,
  reversibly to 0.1 mm. What is **not** cross-hand valid is
  `diagnose._jaw_opening`, `sum |qpos|` over the gripper joints: it *falls* on
  closing for the two prismatic hands and *rises* for the five revolute ones,
  it is metres on some and radians on others (0.001 vs 4.9 for "shut"), and it
  has no signal at all on the Rethink (+0.0012) and Yumi (0.0000) whose fingers
  travel 46 mm and 38 mm. Reading it as a width diagnosed a sign bug that does
  not exist. Use `diagnose.jaw_closure_probe`, which is 0 open and 1 closed on
  every hand. `§7.28`.
- **The same seed does not settle the objects identically across hands.** The
  placement sampler is seeded the same, but the scene's 60 settle steps then run
  with a different gripper attached, and the can comes to rest 14.6 mm away
  between a Panda and a Robotiq 2F-140. Every cross-hand comparison carries that
  ~15 mm scene difference by construction; a reproducibility check must be keyed
  per hand or it will flag it as a failure.
- **Never let a campaign be the first test of its own harness.** Four harness
  bugs in one thread were each found only *after* a 12-25 minute physics run had
  produced plausible numbers -- wrist-frame labels, a reused environment, the
  wrong hand mounted, and the jaw units above -- and all four are answerable in
  milliseconds from a fresh scene. `diagnose.replay_preconditions` checks
  `gripper_mounted`, `scene_unstepped`, `object_placement` and
  `closure_calibrated` on every cell and `diagnose.require` aborts before any
  physics. Extend it rather than adding checks by hand. `§7.28`.
- **`measure_frames.main` used to overwrite `gripper_frames.json` wholesale**,
  wiping `calibrated_depth`, which comes from a much more expensive 13-grasp
  physics sweep and which `_physics_verified` reads to decide which hands
  campaigns may use -- and that function **falls back to all measured pairs when
  the list is empty**, so the Inspire hand, which lifts nothing, would quietly
  re-enter every campaign. It merges now. Any new per-hand measurement must
  merge too.
- **The Inspire hand's fingers travel 0.7 mm.** It does not actuate, which is a
  simpler explanation for its 24 failed grasp attempts than the registry
  docstring's "five fingers cannot pinch a can from above". Against 29-90 mm for
  every other hand.
- **A grasp is a pose, not an axis. Never re-derive its closing direction.**
  `task_frame` used to pick the closing axis's sign with a world-axis test
  (`c[1] >= 0`, tie-broken on `c[0]`), run independently at each of
  `scene_keypoints`' four blocks. The demonstration **places its object square
  with the shelf**, so the placed closing axis lands on world `x` at
  `[1, -1e-17, 0]`, the tie-break fires and answers the opposite way to the
  pick, whose `c[1]` is an unambiguous `-0.581`. The source's own two frames
  came out **180 degrees apart**. Once the jaws shut the object is rigid with
  the hand, so a half turn at one end only **reflects it through the grasp
  point**: dead centre nothing moves, `d` off-centre it lands `2d` away --
  4 to 100 mm over 20 cells. `Grasp6D.closing` has a definite sign; use it.
  The origin was upstream: `reshelving_placement` built the source grasp from
  `rotation[:, 0]`, the **product's body axis**, whose sign is undefined, and
  everything downstream was compensating. `§7.33`.
- **Three instruments read clean through that, and none was broken.** The map
  stayed a valid diffeomorphism (`min det` 0.44-1.00, residual ~1e-7). **The aim
  was exactly 0.00 mm at both ends** -- a half turn about the approach leaves
  the grasp point *fixed*, so aim is blind to it by construction. And
  `orientation_transport_error` minimises over `JAW_SYMMETRY`, which **is** that
  half turn, so it read 0.2-4.5 degrees. For anything that *carries* an object
  use `metrics.transport.carry_orientation_error`: it compares the relative
  rotation between grasp and release, where a roll taken at both ends cancels
  and one taken at a single end does not.
- **Keypoints live in the grasp convention; the robot is commanded in the wrist
  convention; convert explicitly.** GraspGen-X emits `+Z` approach and `+X`
  closing, uniform across all nine hands. Each gripper model's `grip_site` is
  its own frame, differing by the measured `alignment_rotation`: **0.2 degrees**
  for the Robotiqs, **90** for the XArm, **180** for the Panda, Yumi and
  Rethink. Neither `alignment_rotation` nor `grasp_to_eef_pose` appeared in the
  Tier 2 path, so the plan was commanded 0.6-91 degrees wrong. Use
  `to_grasp_convention` / `to_wrist_convention`, once at each end. `§7.34`.
- **`flip_target` was a no-op for the life of the project.** The sign resolution
  ran after it and put the sign back; both branches gave identical keypoints, to
  three decimals of `min det` at all eight yaws of a sweep.
  `pipeline._choose_grasp` tries both rolls and keeps the better -- it was
  scoring one option twice. Any "try both and pick the best" search deserves an
  assertion that the two branches actually differ.
- **Removing the sign resolution loses a job it was doing silently.** `min det`
  tracks the source-to-target frame rotation, and **three of twenty maps fold**
  past about 145 degrees. The target's roll is now chosen *once, on the grasp*,
  by agreement with the demonstration -- not four times from a world axis. Over
  the same cells: median 0.944, min 0.456, no folds, against 0.878 and 0.443.
  On a synthetic yaw sweep the cheap criterion still picks the worse roll at
  2 of 8 yaws; the old code folded at those same two and could not be rescued.
- **GraspGen-X declares `symmetric` per gripper and we ignore it.**
  `parallel_2f` and `revolute_2f` are `True`; **`revolute_3f` is `False`** --
  `robotiq3f` and `inspire`. A half turn about the approach is the same grasp
  for two fingers and a *different* grasp for three.
  `orientation_transport_error` applies it unconditionally, so every orientation
  figure for those two hands, including their rows in `§7.29`'s nine-hand table,
  used a symmetry they do not have. **Not yet fixed.** `§7.34`.
- **A closure-*rate* rule cannot detect contact, and object shape is not why.**
  The jaws are position-commanded (`+1` = "go to fully closed") and `closure`
  reports where the fingers *are*, so it asymptotes toward the commanded value
  whether or not anything is between them: on `robotiq140/can` the rate falls
  from 0.343 to 0.012 per waypoint by waypoint 52 while contact is registered at
  **65**, then jumps back to 0.148 *at* contact. The drop marks the jaws
  arriving, not touching. And on four of five hands the close finishes inside a
  single waypoint, so there is no rate to read at all -- `closure` is sampled
  once per waypoint, eight control steps. Gate on the `held` contact channel, or
  sample closure per control step first.
- **`stage_outcome`'s `grasped` flag misses a slow-closing hand.** It looks for
  contact within twelve waypoints either side of the commanded close. A Robotiq
  2F-140 has 125 mm jaws against a 66 mm can, so each finger travels ~30 mm
  before touching anything and first contact came at waypoint **65** against a
  close at 50. The cell is scored "never grasped" while the can is caught and
  carried to 1.266 m. The gripper schedule is inherited from a Panda
  demonstration whose narrower jaws close much sooner -- so this is the
  cross-hand closing schedule (open item 2) surfacing as a *measurement* fault.
- **The warp does not bend the approach, so do not reach for it to explain a
  disturbed object.** A transported path need not descend along the grasp's own
  approach axis, which makes it the obvious suspect when the hand disturbs the
  object before closing. Measured: the tilt is **0.2 to 1.7 degrees** across all
  20 cells, and a clean straight descent to the same pose contacts the object to
  within 0.2-2.5 mm of the warped one. `xarm/cereal` shifts its box 22.1 mm
  before the jaws close and topples it, and the cause is **not established** --
  it is not the warp, and it is not the fingers being forced apart (the negative
  jaw reading is a calibration offset present on all four xarm cells, three of
  which succeed). It reproduces with no map in the loop, so it is a property of
  that grasp, object and hand.
- **A Robotiq flicks the object sideways as it opens.** Its finger pads swing
  inward while parting, so an object between them is nudged rather than let go:
  72.8 mm and 81.9 mm of post-release travel on two cells that fouled nothing
  and were released only 21-40 mm above the board. The sibling project measured
  the same on the same family of hand and saw a mug carried back up 12.9 cm by
  fingers that had correctly opened. A Panda does not do this.
- **The plan commands the hand through the shelf, and `solve_ik` cannot see it.**
  The default scene is the **cubby** variant, so the top slot is a **78 mm gap
  in x** between the bottom cubby's back panel (which rises 20 mm *above* the
  top board) and the top cubby's 180 mm back wall. **15 of 20 cells command the
  hand inside `shelf_top_back`, by 4.8 to 79.9 mm.** The arm jams: on
  `robotiq140/bread` a finger is 8.24 mm into the wall from waypoint 132, which
  is exactly where the tracking error starts to climb, and the object's motion
  collapses to 0.99 mm per waypoint against 5.3 commanded. So every placement is
  a **drop** from a median 40 mm rather than a set-down -- on cells that work the
  contact is the *object* on the *board* at 0.05-0.17 mm.
  **`solve_ik` is joint angles and a Jacobian with no collision model**, which is
  why the release pose "solves to 3.4-4.9 mm" while being physically
  unreachable, and it is why ruling out the workspace envelope (r = -0.028) and
  the warm-started IK chain (2 of 20 cells) was correct and could not find this.
  `by_collision` only ever checks the hand at the **grasp**. The foul does *not*
  predict which cells fail -- `panda/bread` fouls deepest at -79.9 mm and places
  -- so the mechanism is established and its decisiveness is not. `§7.35`.
- **`reachable_fraction` is a property of the whole path, not of a pose.**
  `replay_labels` seeds each IK solve from the previous one, so it answers "can
  the arm move *between* these poses". The direct control places objects at
  3.8 mm while reporting 53% reachable. Its 5 mm tolerance also flickers --
  successful solves land at 2.4-5.0 mm. And `worst_segment` says "approach"
  whenever *nothing* was unreachable, which reads as an accusation.
- **Hold the grasp selection fixed when comparing constructions.**
  `target_placement`'s `filters="approach"` keeps the 45-degree test and ranks
  by agreement with the demonstration; `"full"` runs the whole funnel and ranks
  by GraspGen-X's own score. The chosen candidate's approach mismatch goes from
  a median of 3.2 to 13.8 degrees and its TCP moves a median 15.7 mm, so
  **39 of 40 cells execute a different grasp**. A run that varies this and
  anything else measures neither. `§8k` in `outputs/keypoints_sweep/FINDINGS.md`.
- **Two capabilities exist in the policy and are never used at runtime.**
  `prediction.reference` — the regressed attractor position, the paper's own
  Sec. V formulation and the policy's only restoring term — is fitted and never
  read by `rollout_policy`. `GPPolicy.attractor()` is called only from a unit
  test. Check whether a proposed fix is already implemented before writing it.

## Scope decisions (agreed 2026-09-05)

**In scope now:** the full theory of Sec. III + Appendix A, uncertainty
propagation, the policy refit of Sec. III-B step 2, the **reshelving** scene of
Sec. V-A, the theory figures 2/4/5, a package restructure with a pytest suite,
and git.

**Explicitly deferred** (the layout leaves room for them, no rework needed):
- Sec. IV baselines: Reshaped-KMP, Laplacian Editing, LWT, Ensemble-NN,
  Ensemble Neural Flows, TP-GMM, TP-HMM; Table I; Figs. 6-10 and the
  Mann-Whitney ranking. `tpgpt/metrics/` already provides the metrics they need.
- Sec. V-B dressing (MuJoCo `flexcomp` cloth is viable on CPU — verified at
  8.3x realtime for a 9x9 sheet — but not built).
- Sec. V-C surface cleaning and Sec. V-D DINO keypoints.

**Dependencies:** minimal, CPU-only. `similaritymeasures`, `pyyaml`,
`pytest-cov`, `imageio` were added. Torch was explicitly *not* added.

## Package map (paper section -> module)

| Paper | Module |
|---|---|
| Sec. III-E-a, Eqs. 4-7 (`gamma`) | `tpgpt/transport/affine.py` |
| App. A, Eqs. 14-16 (GP, derivative posterior) | `tpgpt/transport/gp.py` |
| App. A (SV-GPT, inducing points) | `tpgpt/transport/svgp.py` |
| Sec. III-D/F/G, Eqs. 2, 9, 10, 11 (`phi`, labels) | `tpgpt/transport/maps.py` |
| Sec. III-A (label set) | `tpgpt/transport/labels.py` |
| Sec. III-A (keypoint extraction) | `tpgpt/sim/keypoints.py` |
| Sec. III-H, Eqs. 12, 13 | `tpgpt/transport/uncertainty.py` |
| Sec. III-B step 2 (refit `g`) | `tpgpt/policy/` |
| Sec. V (simulation, keypoints, impedance control) | `tpgpt/sim/` |
| Sec. IV metrics | `tpgpt/metrics/` |
| Grasp generation, gripper registry, frame contract | `tpgpt/grasp/` |
| Object clouds and the scene graph | `tpgpt/perception/` |
| Deterministic prompt parsing | `tpgpt/language/` |
| Figs. 2, 4, 5 | `tpgpt/viz/`, `tpgpt/experiments/` |
| Stage attribution and the reach/drift instruments | `tpgpt/experiments/diagnose.py` |
| What code produced a result | `tpgpt/reporting/provenance.py` |

## Current state

> **A large correction landed on 2026-09-07.** Every end-to-end campaign was
> deleted and six sections of `ROBOTICS_NOTES.md` were withdrawn, because the
> code that produced them was never committed and cannot be identified. Read
> `ROBOTICS_NOTES.md` §7.26 before trusting any number about end-to-end
> behaviour, and before running a new campaign. What follows distinguishes
> carefully between what is measured, what is geometry, and what is open.

**Transportation (done, and validated).** All of Sec. III, Appendix A,
uncertainty propagation, the policy refit and the Sec. V-A reshelving
validation. One demonstration transported into 20 randomised scenes:
**17/20 at 9.1 mm**, failing on seeds 8, 11, 16; 4/20 without the nonlinear
stage (p = 3.3e-5).

This is the regression gate — run it after any change to the map, the policy or
the rollout. It survived the deletion above because it is genuinely repeatable:
fixed seeds, no diffusion model in the loop, and it has been re-measured across
several rounds of code change at 17/20 with the same three failing seeds every
time.

**Keypoint extraction (built; the default is defensible, the trade is not
resolved).** A box fitted to the object's own point cloud in a frame built from
the grasp and the support surface. The **box alone** is the default, on two
independent *geometric* measurements that agree (§6.4 and §7.22): adding the jaw
contacts collapses `min det(J)` from ~0.68 to 0.0074 and folds five maps in six,
because the contacts sit centimetres inside the box's own convex hull and the
map cannot satisfy both constraints without bending sharply enough to turn
inside out. **Those two figures predate the scene and frame corrections**; the
mechanism is geometric and holds, the numbers have not been re-measured.

What §7.22 also shows, and what was missed at the time: **the box misses the
grasp point by about 53 mm, and the contact keypoints hit it to about 5 mm.**
The default trades ten-fold aim accuracy for a well-conditioned map. Same
caveat: mechanism yes, numbers unverified on corrected code.

**That trade is resolved, and the default should change — but read the numbers
from the right place.** A fixed 20 mm cube centred on the grasp, oriented by the
full grasp pose, gets *both*: the aim is exact by construction (the cube's centre
**is** the grasp point and `phi` interpolates keypoints exactly) and the map
stays well conditioned. **The per-hand figures that used to sit here came from
§7.29, which is withdrawn** — it predates both the scene correction and the
frame corrections, and its orientation column was read through a metric that
minimises over the very half turn the source frames carried. The corrected
geometry is `min det` median **0.944** against the cloud box's 0.527 over
20 cells, with no folds in either; the corrected execution is **15/20 against
10/20** (`FINDINGS.md` §8j). Composition remains dead on the corrected code:
median keypoint residual ~10 mm, violating property (i) everywhere.

**Two of the nine hands are still measured wrongly** and it has not been fixed:
`robotiq3f` and `inspire` are `revolute_3f` with `symmetric: false` in
GraspGen-X's own config, so a half turn about the approach is a *different*
grasp for them, and `orientation_transport_error` applies it anyway. Any
cross-hand orientation claim covering those two is unsupported.

**Execution has now been measured, twice, and the second time it was worth
having.** Tier 2 across five hands: the grasp-pose cube places **15 of 20**
against the cloud box's 10, grasps 18 against 14, and traverses 18 against 13.
That is on the corrected code; the first attempt gave 12 and 10 and was largely
measuring three coordinate-frame defects (`§7.33`, `§7.34`), whose repair
recovered five cells and whose two quantitatively predicted cases both came out
as predicted. The default in `object_keypoints` still stays `box="cloud"` until
flipping it is its own commit, but the evidence now clearly favours the cube on
every stage rather than trading aim against conditioning.

**Grasping, filtering and end-to-end physics (built end to end; reliability
unmeasured).** Text prompt -> object and shelf -> cloud -> ranked 6-DoF grasps
-> seven filters -> keypoints -> transport -> policy -> execute -> scored, with
per-run HTML reports. Nine gripper pairs registered, eight verified in physics.

**The end-to-end numbers that exist are the Tier 2 replays**, which run under a
stiff position controller with no policy and no attractor integration, so they
are an **upper bound** on what the keypoints can support rather than a rate for
the full pipeline. Read them as 15/20 and 10/20 with that caveat attached;
`outputs/keypoint_replay_v4/`, commit `6fb83d4`, and the per-cell attribution is
in `FINDINGS.md` §8j. Nothing from the campaigns deleted under §7.26 has been
reinstated, and the policy-driven rate remains unmeasured.

**The largest known execution weakness is that every placement is a drop.** The
arm stops 13 to 130 mm short of the commanded release pose, so the jaws open a
median 40 mm above the shelf board. It usually lands anyway, which is why this
went unnoticed, and it is what makes marginal cells flip between runs. `§7.35`.

**Two threads are open, and they are independent of each other.** The interface
between them is the transported label set: keypoints and the map *produce* it,
the policy and the rollout *consume* it. Neither needs the other to be correct,
so they can be built and tested separately. Full detail in `ROBOTICS_NOTES.md`
§8; the short version:

- **Thread A — keypoints and the map.** Answerable as **pure geometry**, no
  simulator: fit the map, transport the labels, and measure how close the plan
  passes to the grasp, the minimum `det(J)`, and the keypoint residual. Runs in
  milliseconds per variant from a cached cloud and a cached grasp, so hundreds
  of designs can be swept offline. This is where the largest known error lives.
- **Thread B — policy execution.** Needs the simulator but **not** the
  keypoints: test it against a fixed reference path, such as the untransported
  demonstration replayed in the source scene, where the ground truth is known.
  Two capabilities already exist unused (`prediction.reference`,
  `GPPolicy.attractor()`) and one control input is never passed
  (`velocity_desired`); see the bite-list above and §2.7-2.8.

**The instrument and the provenance recording are done** (§7.26-7.27). Both
threads can now be measured without repeating the mistake that cost the last
set of campaigns:

- `diagnose.reach_axes` splits the reach error into the grasp's own **closing /
  approach / jaw** axes instead of one straight-line distance. The old scalar
  mixed a millimetre-scale budget with a 130 mm one, and it rated the *better*
  keypoint configuration worse. `diagnose.closing_budget` derives the tolerance
  as `(aperture − object width) / 2` from the target keypoints rather than
  assuming one, and returns `None` — never a plausible default — when it cannot.
- `diagnose.attractor_drift` measures drift **pointwise** against the
  transported path via `metrics.curves.path_deviation`, so it cannot come out
  negative. The measure it replaces was a difference of two separately
  minimised distances and routinely did.
- `reporting.provenance` records the commit, the modified files, the
  **untracked** code files and the package versions into every manifest
  automatically. Untracked files are what actually made the old campaigns
  unidentifiable, and a plain dirty-check does not see them. The index shows a
  red "Not reproducible" banner naming the reason; `run_experiments
  --require-clean` refuses to start.
- Campaign manifests now record `settings.varied` and `settings.fixed`
  separately from `results`, plus every stage threshold, so a table says what it
  measured without anyone reading the source.

**Also open:** a front-approach demonstration for a shelf with a roof, which a
top-down teach cannot solve by construction; and the `umi` hand, whose contact
offset is the only one in the registry with a large lateral component
(`[0.0, -0.035, -0.112]`) and which §7.2 measured as tolerating only 15 mm of
depth error against 120-135 mm for the parallel jaws.

**590 tests pass** in ~5 min; the unit suite alone is 492 in ~17 s. The suite
was unaffected by the campaign deletion, because it tests mechanisms rather than
campaign outcomes — but note that it also failed to catch the three frame
defects of `§7.33`-`§7.34` for the life of the project, and one of its tests
actively *enforced* the wrong behaviour. When a test breaks under a change,
treat it as a question about which of the two is right, not as a verdict.

## History

`git log` starts from the original three-script prototype
(`affine_transform.py`, `warping_transform.py`, `sim_policy_transport.py`),
committed verbatim before the rewrite and removed afterwards. Its known defects
are catalogued in `ROBOTICS_NOTES.md` §1 so they are not reintroduced.
