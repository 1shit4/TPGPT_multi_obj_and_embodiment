# Can this system hold an object with something other than a two-finger jaw?

**Written for a reader with no prior context.** Every experiment below states
what it asks, the conditions it was measured under, what each column means, the
result, and *why* the result came out that way. Section 1 is background; if you
know the setup, start at section 3.

**Provenance.** `manifest.json` in each output directory records the commit that
produced it. Where it reports `reproducible: false` the tree had uncommitted
changes and the numbers are a scratch experiment, not evidence — that rule is
`ROBOTICS_NOTES.md` §7.26, written after 254 MB of results had to be deleted for
exactly this reason.

**Companion documents.** `outputs/keypoints_sweep/FINDINGS.md` asks which
keypoints should pin the transportation map; `docs/dynamics_execution.md` asks
how a transported policy should be executed. This one asks a third question that
is independent of both: **which hands and which objects can this system grasp at
all?** Nothing here involves a transportation map, a policy, or a shelf.

---

> ### ⚠ Status of every experiment
>
> | # | experiment | status |
> |---|---|---|
> | A | §3 the ruler — what a one-actuator command mis-measures | **stands.** 11 of 13 multi-actuator hands were under-driven; 0 of 13 single-actuator hands affected |
> | B | §4 the Inspire and UMI hands, re-measured and re-calibrated | **stands.** Frames repaired and two false claims removed; both hands still lift 0 of 13, and the reason is measured — their fingertips part by 28 mm against a 65 mm can. One 132 mm lift is **withdrawn** as irreproducible |
> | C | §5 two hands paired from halves already on disk | **stands.** `g1three` and `bd` measured and registered; both lift 0 of 13. Every hand with `grip_site` at its base fails and every hand with it at the fingertips passes, 11 of 11 |
> | ⚠ | **the 26–28 mm "fingertip gap" of §4 is WITHDRAWN** | It measured the free space *between* the fingers. GraspGen-X's swept volume is **the region the fingers traverse while closing** (arXiv:2606.00998), which for a curling hand is a completely different and much larger quantity. Every conclusion in §4 that rests on that number is suspended pending re-measurement |
> | — | §6 how reproducible the frame measurement is | **stands.** One hand's contact offset spans 11.7 mm over three identical runs |
> | E | §7 object census — what the cameras see, per object | **stands.** 70 of 70 cells plan; every new object resolves better than the can, and the single-object scene nearly doubles the bread |
> | F | §8 the bench — grasp, close, lift, carry | **stands.** 74 of 294 grasps and 40 of 110 pairs held. The three-finger `robotiq3f` is joint-second of eleven hands and covers 7 of 10 objects; `wrench` and `nut_square` are refused by every hand |
> | D | §9 descriptions, and four defects the multi-finger hands were failing on | **the §4 withdrawal, explained.** The swept volume is the region the fingers *traverse*, not the gap between them: the Inspire hand sweeps 156 mm, not 28. Four defects fixed — a drifting arm, an unsigned closing axis, a bench measuring its own controller, and a closure command that did nothing. The generator measures the right quantity now with a known bias |
> | G | §10 the closure sweep, and how wide each pair's band is | **stands.** 17 of 25 pairs hold somewhere; `+1` is inside the band for 15 of those 17. Per-pair closure is worth **two cells in twenty-five**, both on the `robotiq140` |
> | H | §11 five more hands, and the split that explains them | **stands.** A second three-finger configuration works (7/15). Every hand on a GraspGen-X-written description grips; every hand on a description written here does not — 6 hands, 0/41 |

---

## 1. Why this document exists

The project's cross-embodiment claim — one demonstration, many hands — rests on
a fleet that is almost entirely **two-fingered**. Nine hands are registered in
`tpgpt/grasp/grippers.py`; eight of them are two-finger jaws. The ninth,
`robotiq3f`, is the whole of the finger-count diversity in the project, and the
only other multi-finger pairing, `inspire`, carried a registry note reading
*"it does not actuate"* and was excluded from every campaign.

That is open item **1h** in `outputs/keypoints_sweep/FINDINGS.md` §8z, whose
stated reason is:

> Only one of the nine registered hands is not two-fingered, and no five-finger
> hand can be run at all. [...] robosuite offers Ability, Fourier, SchunkSvh and
> Jaco hands and **GraspGen-X has a description for none of them**, so there are
> no grasp candidates to filter.

A gripper needs two halves to be usable here. It needs a **robosuite model**, so
MuJoCo can simulate it, and it needs a **GraspGen-X description**, so the grasp
planner can propose poses for it. The assessment above is that the two sets
barely overlap. This document tests that assessment and then acts on it.

The second half of the problem is the objects. Four separate rules for closing
the jaws have now been tried and all four fail somewhere, because the window
between "gripping" and "crushing" is different for every combination of hand and
object and no constant sits inside all of them. That history is summarised in
§9, and it is why the closing rule here is **measured per pair** rather than
chosen.

## 2. What it takes to make a hand usable here

A gripper needs **two halves**, and they come from different places.

**The robosuite half** is a MuJoCo model — geometry, joints and actuators — so
the simulator can mount the hand on an arm and close it on something. robosuite
1.5.2 plus the `robosuite_models` package register **26 hands** between them.

**The GraspGen-X half** is a description the grasp planner conditions on, a file
called `config.json` carrying nine keys. GraspGen-X ships **26** of those too,
in `assets/gripper_descriptions/.../x_grippers/`, but they are not the same 26.

On top of those two, this project measures a third thing for each hand and keeps
it in `tpgpt/grasp/gripper_frames.json`: **where the hand's own frame sits**
relative to the pose the planner emits. GraspGen-X always emits a pose with `+Z`
along the approach direction and `+X` along the closing direction; robosuite
commands a site called `grip_site` whose orientation is whatever the person who
wrote that gripper model chose. The rotation between them is measured per hand by
`python -m tpgpt.grasp.measure_frames`, and a hand that has not been measured
**refuses** to convert a grasp rather than assuming the two frames agree — a
wrong rotation about the approach axis produces a pose that still looks entirely
plausible while the jaws close along the object's longest dimension.

Finally, a hand is only allowed into a campaign once it has a **calibrated
depth**: a sweep of thirteen grasps at different pushes along the approach axis,
keeping the middle of the widest band that actually lifts a reference can
(`tpgpt.grasp.verify.calibrate_depth`). That list is `VERIFIED_PAIRS`, and a hand
that lifts nothing stays out of it with its reason written down.

So onboarding a hand is four steps: have both halves, add a row to
`GRIPPER_PAIRS`, measure the frame, calibrate the depth.

---

## 3. Experiment A — the ruler, and what it had been mis-measuring

`outputs/gripper_audit`, commit `062f52e`, clean tree, 26 hands, about 4 minutes.

### Why

Before adding hands to the registry it is worth checking that the instrument
used to admit them works. It did not.

robosuite builds one action vector per step: the arm's joint commands first, the
gripper's last. A gripper with `dof` actuators therefore owns the **last `dof`
entries**. Two places in this package wrote `action[-1] = 1.0` — the last
*single* entry — and so commanded exactly one actuator, leaving the rest at
zero. Those two places are `grippers.measure_closing_angle` and
`measure_frames.measure_frame`, which between them produce every frame number in
the registry.

For a hand with one actuator, the last entry **is** the whole gripper and
nothing was ever wrong. For a hand with more, it means driving one finger and
recording the others as stationary.

This is the same shape of failure as `ROBOTICS_NOTES.md` §7.28, where the jaw
*reading* was not comparable across hands. Here the jaw *command* is not
comparable across hands, and both produce numbers that look like properties of a
hand.

### Conditions, held fixed

Every gripper robosuite can build, mounted on a Panda in the standard `Lift`
task, no cameras, control frequency 20 Hz. Each hand driven fully open for 40
control steps, then fully closed for 40, and the displacement of every gripper
geom taken in the `grip_site` frame. Seeded with `np.random.seed(0)` before the
environment is built and before `reset`, because robosuite samples the object's
placement at reset and an unseeded sample moves the measured travel.

### What was varied

**One thing: the command.** Each hand is measured twice on two freshly built,
identically seeded environments. Once with `action[-1] = ±1` (the defect) and
once with the whole gripper block set to `±1` (the repair, now
`grippers.gripper_action`). Nothing else differs, so the gap between the two
columns is the defect and nothing else.

### What each column means

| column | what it is |
|---|---|
| `dof` | how many actuators the gripper has, from robosuite's own model |
| travel, whole block | the largest distance any gripper geom moved between fully open and fully closed, driving every actuator. **Not a jaw aperture** — the moving set includes knuckles and outer links — but monotone and per hand |
| travel, last entry only | the same measurement with one actuator driven |
| shortfall | how much travel the scalar command failed to produce. Positive means it under-drove the hand |
| fingers lost | how many geoms travelled more than 1 mm under the repaired command but not under the scalar one. This is the column that catches a hand whose *furthest* finger moves either way while another never moves at all |
| registry | the short name in `GRIPPER_PAIRS`, or — if the hand is not registered |

### Result: every multi-actuator hand was under-driven, and no single-actuator hand was touched

**11 of the 13 hands with more than one actuator** were measurably under-driven.
**0 of the 13 hands with one actuator** were affected beyond noise.

#### Hands with more than one actuator

| robosuite hand | dof | travel, whole block | travel, last entry only | shortfall | fingers lost | registry |
|---|---|---|---|---|---|---|
| `SchunkSvhLeftHand` | 20 | **165.13 mm** | 82.08 mm | **+83.06 mm** | 0 | — |
| `SchunkSvhRightHand` | 20 | **150.37 mm** | 55.09 mm | **+95.28 mm** | 0 | — |
| `AbilityLeftHand` | 10 | **230.90 mm** | 82.56 mm | **+148.34 mm** | 0 | — |
| `AbilityRightHand` | 10 | **102.69 mm** | 70.32 mm | **+32.37 mm** | 0 | — |
| `G1ThreeFingerLeftGripper` | 7 | 20.30 mm | 21.05 mm | -0.75 mm | 0 | — |
| `G1ThreeFingerRightGripper` | 7 | **19.56 mm** | 8.34 mm | **+11.22 mm** | 0 | — |
| `FourierLeftHand` | 6 | 87.39 mm | 88.94 mm | -1.55 mm | 0 | — |
| `FourierRightHand` | 6 | **29.66 mm** | 2.64 mm | **+27.03 mm** | 2 | — |
| `InspireLeftHand` | 6 | **27.93 mm** | 11.42 mm | **+16.50 mm** | 0 | — |
| `InspireRightHand` | 6 | **53.36 mm** | 10.04 mm | **+43.32 mm** | 0 | `inspire` |
| `RobotiqThreeFingerDexterousGripper` | 4 | **91.32 mm** | 27.68 mm | **+63.64 mm** | 10 | — |
| `JacoThreeFingerDexterousGripper` | 3 | **14.15 mm** | 13.94 mm | **+0.21 mm** | 12 | — |
| `UMIGripper` | 2 | **44.60 mm** | 44.59 mm | **+0.01 mm** | 2 | `umi` |

#### Hands with one actuator — the control

| robosuite hand | dof | travel, whole block | travel, last entry only | shortfall | fingers lost | registry |
|---|---|---|---|---|---|---|
| `AlohaGripper` | 1 | 28.91 mm | 28.91 mm | -0.00 mm | 0 | — |
| `BDGripper` | 1 | 166.00 mm | 166.00 mm | -0.00 mm | 0 | — |
| `JacoThreeFingerGripper` | 1 | 14.15 mm | 14.15 mm | -0.00 mm | 0 | — |
| `PR2Gripper` | 1 | 49.53 mm | 49.54 mm | -0.01 mm | 0 | — |
| `PandaGripper` | 1 | 39.18 mm | 39.18 mm | +0.00 mm | 0 | `panda` |
| `RethinkGripper` | 1 | 23.06 mm | 23.06 mm | +0.00 mm | 0 | `rethink` |
| `Robotiq140Gripper` | 1 | 58.37 mm | 58.37 mm | -0.00 mm | 0 | `robotiq140` |
| `Robotiq85Gripper` | 1 | 48.75 mm | 48.75 mm | +0.00 mm | 0 | `robotiq85` |
| `RobotiqThreeFingerGripper` | 1 | 93.49 mm | 92.24 mm | +1.25 mm | 0 | `robotiq3f` |
| `XArm7Gripper` | 1 | 42.23 mm | 42.23 mm | +0.00 mm | 0 | `xarm` |
| `YumiLeftGripper` | 1 | 25.00 mm | 25.00 mm | -0.00 mm | 0 | — |
| `YumiRightGripper` | 1 | 25.00 mm | 25.00 mm | +0.00 mm | 0 | `yumi` |
| `Z1Gripper` | 1 | 53.61 mm | 53.61 mm | -0.00 mm | 0 | — |

### Reading it

**The control is what makes this trustworthy.** For a one-actuator hand the two
commands are literally the same vector, so any difference is measurement noise
and nothing else. Twelve of the thirteen agree to within 0.01 mm. The
thirteenth, `RobotiqThreeFingerGripper`, differs by 1.25 mm — that hand's
fingers settle chaotically against each other, and repeating either measurement
reproduces about that much scatter. So the noise floor is about 0.01 mm for a
well-behaved hand and about 1.3 mm for a restless one, and every shortfall in
the multi-actuator table above 2 mm is real.

**The two failures look completely different, and both are silent.**

`AbilityLeftHand` loses **148 mm** of travel, `SchunkSvhRightHand` **95 mm**,
`InspireRightHand` **43 mm** — the driven finger moves and the rest do not, so
the hand simply closes less far.

`UMIGripper` and `JacoThreeFingerDexterousGripper` lose **no travel at all**
(0.01 mm and 0.21 mm) and lose **2 and 12 fingers**. The finger the scalar
command happens to drive travels exactly as far as it always did; what vanishes
is the *other* finger, which never moves. A gripper closing one jaw still
produces a plausible closing axis, a plausible spread and a plausible travel
figure. That is why a two-column table was needed rather than one: a single
travel number cannot tell "closed less" from "closed one-sidedly".

### What it says

**`inspire`'s exclusion from the project was an artefact of this.** Its registry
entry recorded `finger_travel_mm` of 0.7 and a note reading *"it does not
actuate"*, and that note is the stated reason a five-finger hand has never
appeared in any campaign — `CLAUDE.md` carries it as a known fact, and
`FINDINGS.md` §8z item 1h builds on it. Driven on all six of its actuators the
hand travels **53 mm**. It actuates. The ruler was reading its thumb.

**`umi` is the other one, and its symptom was different.** It is recorded as
convertible but not executable, lifting nothing across thirteen swept depths.
Under the scalar command only its left finger moved.

### What it does not settle

That these hands *move* does not mean they *grip*. Travel is a free-air
measurement with nothing between the fingers; whether a repaired frame lets
either hand pick something up is Experiment B, and whether any of the unpaired
hands can be used at all depends on the GraspGen-X half, which is Experiments C
and D.

Nor does the audit say anything about the **quality** of the small-travel hands.
`JacoThreeFingerGripper` moves 14 mm and `G1ThreeFingerRightGripper` 20 mm,
against 166 mm for the `BDGripper`. A hand that barely opens has little room to
accept an object, and that is a reason to expect trouble later rather than a
measurement of it.

### Consequences

* `tpgpt.grasp.grippers.gripper_action` is now the single way a measurement
  commands a hand, so the next hand with more than one actuator cannot
  reintroduce this;
* the executor was **never** affected — `replay._joint_action` already filled
  every gripper dimension and `CartesianImpedanceController.action` broadcasts
  across `gripper_dof` — so no recorded campaign result is invalidated by this;
* the audit is kept as a driver rather than a script, because onboarding any new
  hand should start with these five seconds of measurement rather than with a
  campaign.

---

## 4. Experiment B — the two repaired hands: what changed, and what did not

Commit `2ca4f74` for the frames; the depth sweeps and the pocket measurements
were run from `062f52e`/`2ca4f74` with the scratch scripts named below.

### Why

Experiment A shows the Inspire and UMI hands were measured with one actuator
driven. Their `gripper_frames.json` entries — the rotation between the planner's
frame and robosuite's, and where a grasped object ends up in the hand — were
computed from that measurement. So both were re-measured, and then asked the
only question that matters: **does either hand now pick anything up?**

### Conditions, held fixed

`python -m tpgpt.grasp.measure_frames`, every hand, seeded. Then
`tpgpt.grasp.verify.calibrate_depth`, which is the project's existing admission
test: thirteen synthesised top-down grasps on a reference **can**, at pushes
along the approach axis from −90 mm to +90 mm, keeping the middle of the widest
contiguous band that lifts the can more than 50 mm. A hand with no such band
gets `calibrated_depth: None` and stays out of `VERIFIED_PAIRS`.

### What changed in the frames

The seven single-actuator hands are the control and are **unchanged**: finger
travel moved by at most 0.0045 mm and contact offsets by at most 0.007 mm. Their
`anisotropy` figures move by large ratios and this means nothing — they are
ratios of singular values between 10⁴ and 10¹¹, a perfectly single-axis jaw
divided by numerical dust.

| quantity | `umi` before | after | `inspire` before | after |
|---|---|---|---|---|
| finger travel | 44.78 mm | **89.19 mm** | 6.49 mm | **−1.05 mm** |
| anisotropy | 5.6 × 10⁵ | **2.4 × 10⁸** | 2.9 | 7.3 |
| contact offset | [0.1, **−37.0**, −141.7] mm | [0.0, **2.0**, −141.7] mm | [−0.8, −134.0, 10.2] mm | [4.9, −129.0, 1.4] mm |
| approach in `grip_site` | [0, −0.066, −0.998] | [0, **0.015**, −1.000] | [−0.062, −0.969, 0.238] | [−0.066, −0.980, 0.188] |
| closing axis moved | — | (same axis) | — | **76 degrees** |
| `plus_one_closes` | True | True | True | **False** |

**The UMI was being measured with one jaw, and it shows in three places at
once.** Its travel exactly doubles, because the spread between its fingers had
been changing by one finger's motion instead of two. Its anisotropy rises by
three orders of magnitude — with both fingers moving, a single axis describes it
almost perfectly, which is what a parallel jaw should look like. And its contact
offset loses its lateral component: **−37.0 mm becomes +2.0 mm**. That −37 mm is
the "only contact offset in the registry with a large lateral component" that
`CLAUDE.md` and `ROBOTICS_NOTES.md` §8 both record as a property of the hand
worth investigating. It was never a property of the hand. It was the frame of a
gripper closing on one side.

**The Inspire hand's closure calibration is now honestly broken rather than
dishonestly fine.** `finger_travel_mm` reads −1.05 and `plus_one_closes` reads
False. Neither says the hand opens when told to close. They say that `spread` —
the extent of the moving geoms along one closing axis — does not describe a hand
whose five fingers curl through an arc: they sweep *outward* in extent while
closing *inward* on the thumb. The old +6.5 mm was that same quantity measured
on a hand where only the thumb moved, which is why it looked well behaved.

The consequence is correct and deliberate. `diagnose.jaw_closure_probe` returns
`nan` for an uncalibrated hand rather than a plausible zero, and
`diagnose.replay_preconditions` checks `closure_calibrated`, so the Inspire hand
is now **refused** rather than measured with a ruler that does not fit it.

### Result: both hands still lift nothing — 0 of 13 each

| hand | contact offset used | lifts ≥ 50 mm | best lift |
|---|---|---|---|
| `umi` | closed-fingertip (repaired frame) | 0 / 13 | +0.50 mm |
| `inspire` | closed-fingertip (repaired frame) | 0 / 13 | +5.83 mm |

So the repair does **not**, on its own, rescue either hand. The value of
Experiment A is that a false explanation has been removed, not that a hand has
been gained.

### Following it up: the contact point is measured in the wrong pose

`contact_offset` is the centroid of the distal third of the fingers **once
closed**. For a jaw whose fingers travel 39 mm that is within a few millimetres
of where an object sits. For a hand whose fingers travel 166 mm, or curl into a
palm, the closed fingertip is nowhere near it.

The pocket between the *open* fingers is where an object sits by definition, and
`tpgpt.grasp.describe` measures it by casting rays across the hand. Substituting
its centre for the closed-fingertip centroid:

| hand | closed-fingertip | pocket centre | they differ by | result |
|---|---|---|---|---|
| `umi` | [0.0, 2.0, −141.7] mm | [−0.0, 2.0, −146.2] mm | 4.4 mm | see below |
| `inspire` | [4.9, −129.0, 1.4] mm | [−26.9, −125.1, 40.8] mm | 50.8 mm | 0 / 13 |
| `g1three` | [−0.5, −19.0, −130.2] mm | [22.3, −26.9, −129.3] mm | 24.2 mm | 0 / 13 |
| `bd` | [−24.8, 0.0, −208.2] mm | [30.0, 0.0, −161.1] mm | 72.3 mm | 0 / 13 |

**A 132 mm lift was measured for the UMI here and is withdrawn.** One run of the
three-point probe reported the UMI lifting the can 132.38 mm with the pocket
offset at zero extra depth. It does not reproduce. The same hand, the same
nominal offset and the same depth gave **−0.00 mm** three times in one process
with the offset rounded to four decimals, and **−16.42 mm** with the offset at
full precision — a 0.05 mm change in the commanded offset flipping the outcome
between "misses the can entirely" and "knocks it over". The global random state
was ruled out separately: five different seedings gave −0.00 mm every time.

So the UMI's grasp of a can under this test is **chaotic**, not marginal, and no
number from it should be quoted. This is `ROBOTICS_NOTES.md` §7.37 arriving
before the write-up rather than after it.

### ~~The real obstacle, measured: these hands cannot accept a can on any axis~~ — WITHDRAWN

> **This subsection is wrong, and the error is in the measurement, not the
> arithmetic.** It reports the free space *between* the fingers, and concludes
> from a 26–28 mm reading that neither hand can accept a 65 mm can.
>
> GraspGen-X's swept volume is not that quantity. The paper (arXiv:2606.00998)
> defines it as **"the region traversed by the robot fingers during its grasping
> motion"** — the volume the fingers *sweep through* as they close, not the gap
> they leave. The two nearly coincide for a parallel jaw, which is why the
> figures below reproduce the shipped apertures for the Panda, Yumi and Robotiq
> 2F-140 and why that agreement looked like validation. For a hand whose fingers
> curl inward they are entirely different: the Panda's config declares
> `extents[0] = 0.08` against a finger travel of 0.04 per side, i.e. the **total
> travel of the two fingers**, and `extents2[0] = 0.04`, the travel from
> half-closed.
>
> So the 10.4 mm and 26.1 mm readings describe a gap that is not what the
> representation encodes, and no conclusion about whether these hands can hold
> anything follows from them. The paper also reports GraspGen-X scoring **0.363
> on the Inspire Hand** and 0.404 on the Surge Hand, so five-finger hands are
> within what the released model is expected to handle.
>
> What survives independently of this: both hands lift 0 of 13 on the depth
> sweep, on a can, a lemon and bread. That is a physics result and it stands.
> But its *explanation* is now open, and one candidate is that the sweep uses a
> synthesised top-down grasp converted through an `alignment` these two hands
> measure unreliably (3.2 mm and 5 degrees between runs), rather than a grasp
> GraspGen-X actually proposed for them. Neither hand has yet been run through
> the bench, which is the test that uses real planner grasps.
>
> The subsection is kept rather than deleted because the axis search in it is
> still a correct measurement of the thing it measured, and because
> `ROBOTICS_NOTES.md` §7.26 is about what happens when superseded numbers are
> quietly removed instead of marked.

### The measurement as taken, now known to be of the wrong quantity

The pocket is measured along the axis the fingers *travel* along, which for a
parallel jaw is the axis they grip across, by construction. For a hand whose
fingers curl it need not be. So the measurement was repeated along **twelve
axes** perpendicular to the approach, every 15 degrees, and the widest gap kept:

| hand | gap on the travel axis | widest gap, any axis | at |
|---|---|---|---|
| `panda` | 78.2 mm | 78.2 mm | 0° — and **no other axis has a pocket at all** |
| `yumi` | 50.0 mm | 51.8 mm | −15° (the same axis) |
| `robotiq3f` | 144.2 mm | 146.1 mm | −15° (the same axis) |
| `inspire` | 10.4 mm | **28.0 mm** | **75°** |
| `g1three` | **no pocket** | **26.1 mm** | **90°** |

Two things follow, and they are independent.

**For a two-finger jaw the travel axis is the grasping axis**, and the search
confirms it rather than assuming it — on the Panda every other axis finds no
pocket whatever, because there is nothing on either side of a ray fired across
the fingers.

**For the two curling hands it is not**, by 75 and 90 degrees. But even at their
best axis they open **28.0 mm and 26.1 mm**, against a reference can 65 mm
across. Neither hand can take that can between its fingertips on any axis, at
any depth, with any contact offset. That is why thirteen sweeps found nothing,
and it is not a frame problem.

### What it says

**The registry's original explanation was right and the replacement was wrong.**
The `inspire` entry once read that "five fingers cannot pinch a can from above",
and `CLAUDE.md` replaced that with "it does not actuate, which is a simpler
explanation than its five fingers being unable to pinch", on the strength of the
0.7 mm travel figure. The 0.7 mm was an artefact. The hand travels 53 mm, and
its fingertips still part by only 28 mm — so the original account stands and the
simpler one is withdrawn.

**`calibrate_depth` decides which hands a campaign may use, and it asks one
object.** `REFERENCE_OBJECT = "can"` is a good reference for a jaw opening to 80
or 125 mm and a meaningless one for a hand opening to 28. For those hands the
sweep has been measuring *a can does not fit*, which is a fact about the can.
`lift_height` now places the object it is asked to measure, so a hand can be
calibrated against something it could plausibly hold; the default scene is
untouched, because every stored `calibrated_depth` was measured in it.

### What it does not settle

Whether either hand grips a **smaller** object, which is the obvious next
question and does not follow from anything above. And whether the fingertip
pocket is even the right region for an anthropomorphic hand: a five-finger hand
holds things in its palm, not between its fingertips, so a measurement taken at
the fingertips may be describing the wrong part of the hand. GraspGen-X's own
description of this gripper declares an 80 mm pocket, against the 28 mm measured
here at the fingertips, and that gap is unexplained.

---

## 5. Experiment C — two pairings whose halves were already on disk

Commit `2ca4f74`; `gripper_frames.json` entries for `g1three` and `bd`.

### Why

`FINDINGS.md` §8z item 1h lists two pairings as *unverified rather than ruled
out*: `G1ThreeFingerRightGripper` ↔ `unitree_g1`, a second three-finger hand,
and `BDGripper` ↔ `bd_spot`. Both halves already exist, so neither needs a line
of new geometry — only a registry row, a frame measurement and a depth sweep.
They are the cheapest possible addition to the fleet and were done first for
that reason.

`bd` is worth having despite being two-fingered: it is the **only**
`revolute_2f` hand GraspGen-X declares `symmetric: false`, so it is the only
registered exercise of the asymmetric path for a two-finger jaw — a path §8l
records as having been applied wrongly for the life of the project.

### What each column means

| column | what it is |
|---|---|
| closing angle | rotation about the approach axis from the planner's `+X` to the direction this hand's fingers travel, in `grip_site` coordinates |
| anisotropy | ratio of the first to the second singular value of the finger displacements — how well **one** axis describes the hand. A parallel jaw scores in the hundreds or higher; the registry's threshold for "single axis" is 50 |
| travel | spread between the moving finger geoms, open minus closed |
| `eef_depth` | root body to `grip_site` along the approach axis. **Zero means `grip_site` sits at the gripper's base rather than at its fingertips** |
| lifts | of thirteen swept depths, how many raised the reference can by more than 50 mm |

### Result

| hand | fingers | closing angle | anisotropy | travel | `eef_depth` | contact offset | lifts |
|---|---|---|---|---|---|---|---|
| `g1three` | 3 | 47.0° | **1.2** | 8.1 mm | 0.000 m | [−0.5, −19.0, −130.2] mm | **0 / 13** |
| `bd` | 2 | 0.0° | 85 179 | 52.0 mm | 0.000 m | [−24.8, 0.0, −208.2] mm | **0 / 13** |

Both mount, both actuate, both measure cleanly, and **neither lifts the can**.
`bd` gets closest: four of its thirteen depths raise the can 3.5 to 4.7 mm, so
it is gripping something and losing it, rather than missing entirely.

### Reading it

**`g1three`'s anisotropy of 1.2 is the lowest in the registry by a wide
margin** — the next lowest is `robotiq3f` at 4.3 and every parallel jaw is above
550. An anisotropy of 1.2 means the finger displacements have no dominant
direction at all: the three fingers converge radially, and `closing_angle` is
close to meaningless for that hand. `GripperPair.single_axis` already reports
False for it, which is the registry saying so rather than this document.

It is also the **least reproducible** hand measured here. Two fresh processes
gave 28.4° and 47.0° with 3.4 mm and 8.1 mm of travel.

### A property that separates every hand that works from every hand that does not

| `eef_depth` | hands | calibrated |
|---|---|---|
| > 0 (`grip_site` at the fingertips) | panda, robotiq85, robotiq140, rethink, xarm, robotiq3f, yumi | **7 of 7** |
| = 0 (`grip_site` at the base) | umi, inspire, bd, g1three | **0 of 4** |

Perfect separation, across eleven hands. It is a **correlation with a mechanism
attached**, not a coincidence: when `grip_site` sits at the base, the whole
distance from the wrist to the object — 130 to 208 mm on these four — has to be
carried by `contact_offset`, a quantity measured as the centroid of the fingers
in their *closed* pose. Any error in it is applied at the end of a 200 mm lever.
On `bd` that error is the 24.8 mm lateral component in the table above, which on
a 65 mm can is the difference between the middle and the rim.

It is **not** proof that a base-mounted `grip_site` is the cause. Four hands is
four hands, two of them are also multi-finger hands with a 28 mm pocket, and the
two explanations are not separated by anything measured here.

### Consequences

* both rows stay in `GRIPPER_PAIRS` with `calibrated_depth: None`, so they are
  measured and convertible but excluded from campaigns — the treatment the
  registry already gives a hand that converts and does not lift, with the reason
  recorded rather than left as a silent `None`;
* `measure_frames` grew a `--only` flag, for the reason in section 6 below.

---

## 6. How reproducible is the frame measurement? Not very, and for one hand not at all

Measured while adding the hands of Experiment C, because adding a hand re-runs
the measurement for every hand and it was worth checking what that costs.

### Why it matters

`gripper_frames.json` is a cache. The point of a cache is that reading it twice
gives the same answer, and the point of *re-measuring* is to pick up a change.
If re-measuring an unchanged hand returns a different number, then a diff of
that file cannot distinguish "the hand changed" from "the measurement was taken
again", and substituting one value for another silently changes every campaign
that hand appears in. That is `ROBOTICS_NOTES.md` §7.41 — a pipeline fix between
two campaigns reselecting the grasp in half the cells — in a different place.

### Conditions

`measure_frame` seeds `np.random` before building the environment and before
`reset`, so the object placement is fixed. Three fresh **processes**, identical
code, identical seed, same hand.

### Result: the number of geoms counted as fingers is not stable

| run | geoms classed as moving | contact offset | closing angle | travel |
|---|---|---|---|---|
| 1 | **28** | [−3.29, 6.17, **−9.03**] mm | 1.157° | 83.76 mm |
| 2 | **32** | [−7.55, 4.45, **−20.71**] mm | 0.415° | 84.74 mm |
| 3 | **30** | [−5.96, 4.83, **−9.35**] mm | 0.414° | 84.74 mm |

`RobotiqThreeFingerGripper`. The contact offset's approach component spans
**11.7 mm** across three runs of the same measurement.

The mechanism is visible in the first column. A geom counts as a finger if it
moved more than `MIN_TRAVEL` = 0.2 mm; this hand's three fingers settle against
each other, and several of its geoms sit right on that threshold, so which ones
count changes between runs. `contact_offset` is the centroid of the distal third
of whichever set was counted, so a different set is a different centroid.

That is the same failure as the keypoint instability in `CLAUDE.md` — *any
statistic taken over a thin band moves several millimetres between two samplings
of the same object*. It has now appeared in point clouds, in keypoints, and here
in a gripper's own geometry.

For scale, the same three-run comparison on the other hands:

| hand | spread over repeated measurement |
|---|---|
| parallel jaws (panda, yumi, robotiq85, robotiq140, xarm, rethink) | ~0.02 mm |
| `inspire` | 3.2 mm and 5 degrees |
| `g1three` | 4.7 mm of travel and 18 degrees |
| `robotiq3f` | **11.7 mm** |

### Consequences

`measure_frames` grew a `--only` flag and its docstring now says to prefer it:

```bash
python -m tpgpt.grasp.measure_frames --only g1three,bd
```

Re-measuring a settled hand does not refresh its value, it **draws another
sample**. The two hands of Experiment C were added with `--only`, and the diff
of `gripper_frames.json` for that commit is 111 lines of insertion and **not one
changed line** in any of the nine hands already there.

### What this does not settle

Why the measurement is process-dependent at all. Within a single process,
repeated calls agree to 0.005 mm on this hand; across processes they do not.
The seed is set, so something outside `np.random` differs between processes.
That is unexplained, and it is the reason the advice above is "measure only the
hand you are adding" rather than "run it twice and average".

---

## 7. Experiment E — the object census: what the cameras see, one object at a time

`outputs/bench_plan`, commit `217a584`, clean tree.
`run_grasp_bench --plan-only --ranks 5`, 70 cells, about 20 minutes.

### Why

Six objects were added to `OBJECT_CLASSES` — a hammer, a wrench, a pot, a mug
and two nuts — on the argument that a box, a carton, a can and a loaf are all
things a **parallel jaw** is good at, so a fleet measured only on them cannot
show that more fingers buy anything. A handle, a rim, a hole and an offset
centre of mass can.

Before any of that can be measured, the objects have to be **seen**. This
project already has one object it cannot use for exactly this reason: the lemon
fuses to **17** points in the five-object scene, against a `MIN_CLOUD_POINTS`
floor of 40, and GraspGen-X refuses a cloud that thin outright (§7.18). So the
first question about a new object is not whether it can be grasped but whether
the cameras resolve it at all.

`--plan-only` answers that in seconds a cell rather than minutes: it runs
perception and grasp generation and stops before any physics. It also answers
the second question — whether the planner returns candidates at all — which is
worth knowing before an hour of physics is committed.

### Conditions, held fixed

Seven hands (every pair in `VERIFIED_PAIRS`) × ten objects. **One object on the
table at a time**, cameras at **256 × 256**, three views (`workspace`,
`sideview`, `birdview`) fused with one pixel of mask erosion, seed 0. Grasps
from GraspGen-X through `tpgpt.grasp.cache`, 200 samples drawn, top 100 kept.

The resolution is deliberately unchanged. Raising it to 512 would roughly
quadruple every cloud and lift the lemon over the floor, and it would also change
what GraspGen-X proposes for every object, so it belongs in its own experiment.

### What each column means

| column | what it is |
|---|---|
| cloud points | points in the fused, world-frame object cloud after mask erosion and downsampling |
| range | smallest and largest across the seven hands. The hand is at its home pose, so it occludes a little, and the object does not settle identically under different hands |
| five-object scene | the same object's cloud in the four- or five-object campaign scene, from `ROBOTICS_NOTES.md` §7.18 |

### Result: every object is seen, and the two the campaigns struggle with are the two thinnest

| object | cloud points, median over the seven hands | range | in the five-object scene (§7.18) |
|---|---|---|---|
| `pot` | 2892 | 2838 – 2908 | — (new) |
| `wrench` | 1388 | 1383 – 1388 | — (new) |
| `cereal` | 1031 | 1023 – 1031 | 975 |
| `mug` | 650 | 650 – 650 | — (new) |
| `hammer` | 619 | 507 – 1278 | — (new) |
| `nut_round` | 372 | 367 – 373 | — (new) |
| `milk` | 367 | 366 – 367 | 345 |
| `nut_square` | 340 | 335 – 341 | — (new) |
| `can` | 238 | 237 – 238 | 255 |
| `bread` | 156 | 156 – 156 | 81 |

**70 of 70 cells planned**, every one returning the full 100 candidates. No cell
was rejected for a thin cloud and none was refused by the planner.

### Reading it

**Every new object is seen more thoroughly than the grocery meshes.** The thinnest
of the six, `nut_square` at 340 points, is still better resolved than the `can` at
238 and more than twice the `bread` at 156. So nothing here repeats the lemon's
problem, and the census can be set aside: if one of these objects fails later, it
will not be because the cameras could not find it.

**The single-object scene helps, and the size of the help says what it is.** The
bread goes from 81 points to **156**, very nearly doubling; the cereal, milk and
can barely move (975 → 1031, 345 → 367, 255 → 238). The objects that gain are the
small ones, which is what you would expect if the gain is **occlusion** — a small
object is easily hidden behind a larger neighbour and a large one is not. The can
actually loses 17 points, which is within the variation across hands and is not
a change worth explaining.

**The hammer is the only object whose cloud depends on which hand is mounted**,
and it varies two and a half fold — 507 points with the `rethink` against 1278
with the `robotiq140`. Every other object varies by less than 2%. The hammer is
the one object here with a long thin handle and a head, so how it comes to rest
matters more: the same seed does not settle an object identically under different
grippers (`CLAUDE.md`), and a hammer that settles on its side presents a very
different silhouette from one that settles on its head. That is a property of the
object, not a fault, but it does mean **hammer cells are not cross-hand
comparable in the way the other nine are**.

### What it does not settle

Nothing about grasping. A well-resolved cloud is a precondition, not a result,
and the pot at 2892 points is also the largest and heaviest object here. Whether
any hand can hold these objects is Experiment F.

Nor does it settle the lemon, which is excluded from this grid because it is
excluded from the pipeline. A single-object scene would very likely lift it over
the floor — the bread nearly doubled — but that was not tested and should not be
assumed.

---

## 8. Experiment F — the bench: eleven hands, ten objects, can it hold it?

`outputs/grasp_bench_v2`, commit `4e0c750`, clean tree. 294 grasps, about three
hours. Three planner grasps per hand-object pair.

### Why

Every other driver in this package measures a *transported* plan, which puts
the map, the keypoints, grasp selection, the controller and the shelf in series:
when a cell fails, six things could be responsible. This one removes all of
them. It asks the planner for grasps on the object in front of it, executes
them directly, and reports whether the object was still in the hand at the end.

### Conditions, held fixed

Eleven hands — every pair in the registry — times ten objects. **One object on
the table at a time**, so cells are independent; the cost is that they are
**not** comparable with the four-object campaigns. Cameras at 256 x 256, three
views, seed 0. Grasps from GraspGen-X through `tpgpt.grasp.cache`, put through
the funnel's pick-side stages, screened for reachability, then ranked by the
planner's own score. Arm under **joint-position control through inverse
kinematics**, jaws commanded shut with plain `+1`.

### What each column means

| column | what it is |
|---|---|
| held | the object was still in the hand, above the table, after being lifted 150 mm and carried 200 mm sideways and back |
| lifted then lost | it left the table by more than 20 mm and was dropped |
| never lifted | it never left the table |
| reach error | distance from the gripper site to the commanded grasp pose, at the moment before the jaws move |
| closure at lift | 0 is fully open, 1 is shut on air. Above 1 means pressed past the free-air closed spread |
| slip | how far the object drifted **in the hand's own frame**, where a held object is motionless by definition |

### Result: 74 of 294 grasps held, 40 of 110 pairs

### Held, per cell (of the grasps tried)

| hand | `bread` | `can` | `cereal` | `hammer` | `milk` | `mug` | `nut_round` | `nut_square` | `pot` | `wrench` | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bd` | 0/1 | 0/3 | 0/3 | 0/1 | 0/3 | **1**/3 | 0/3 | 0/3 | 0/2 | 0/1 | **1/23** |
| `g1three` | 0/3 | 0/3 | 0/3 | 0/2 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **0/29** |
| `inspire` | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/2 | 0/3 | 0/3 | 0/1 | **0/27** |
| `panda` | **1**/3 | **1**/3 | **1**/3 | **1**/1 | **2**/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **6/28** |
| `rethink` | 0/3 | **1**/3 | **1**/2 | 0/3 | **2**/3 | **1**/1 | 0/2 | 0/3 | 0/3 | 0/2 | **5/25** |
| `robotiq140` | **2**/3 | **1**/3 | **3**/3 | **1**/1 | **2**/3 | **3**/3 | 0/3 | 0/3 | **1**/3 | 0/3 | **13/28** |
| `robotiq3f` | **1**/3 | **2**/3 | **3**/3 | **2**/3 | **3**/3 | **2**/3 | 0/3 | 0/3 | **2**/3 | 0/3 | **15/30** |
| `robotiq85` | 0/3 | **2**/3 | **2**/3 | **2**/3 | **2**/3 | **3**/3 | 0/1 | 0/3 | 0/3 | 0/3 | **11/28** |
| `umi` | 0/3 | **1**/3 | 0/3 | **1**/1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | 0/1 | **2/24** |
| `xarm` | **2**/3 | **3**/3 | **2**/3 | **3**/3 | **2**/3 | **3**/3 | 0/3 | 0/3 | **3**/3 | 0/3 | **18/30** |
| `yumi` | 0/3 | **2**/3 | 0/1 | 0/3 | 0/3 | 0/1 | **1**/3 | 0/1 | 0/3 | 0/1 | **3/22** |
| **total** | **6**/31 | **13**/33 | **12**/30 | **10**/24 | **13**/33 | **13**/29 | **1**/29 | **0**/31 | **6**/30 | **0**/24 | **74/294** |


### Pairs where at least one grasp held

40 of 110 hand-object pairs held the object with at least one of the grasps tried.

| object | hands that held it | hands that did not |
|---|---|---|
| `bread` | 4/11 — `panda`, `robotiq140`, `robotiq3f`, `xarm` | `bd`, `g1three`, `inspire`, `rethink`, `robotiq85`, `umi`, `yumi` |
| `can` | 8/11 — `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `umi`, `xarm`, `yumi` | `bd`, `g1three`, `inspire` |
| `cereal` | 6/11 — `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `xarm` | `bd`, `g1three`, `inspire`, `umi`, `yumi` |
| `hammer` | 6/11 — `panda`, `robotiq140`, `robotiq3f`, `robotiq85`, `umi`, `xarm` | `bd`, `g1three`, `inspire`, `rethink`, `yumi` |
| `milk` | 6/11 — `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `xarm` | `bd`, `g1three`, `inspire`, `umi`, `yumi` |
| `mug` | 6/11 — `bd`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `xarm` | `g1three`, `inspire`, `panda`, `umi`, `yumi` |
| `nut_round` | 1/11 — `yumi` | `bd`, `g1three`, `inspire`, `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `umi`, `xarm` |
| `nut_square` | 0/11 — — | `bd`, `g1three`, `inspire`, `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `umi`, `xarm`, `yumi` |
| `pot` | 3/11 — `robotiq140`, `robotiq3f`, `xarm` | `bd`, `g1three`, `inspire`, `panda`, `rethink`, `robotiq85`, `umi`, `yumi` |
| `wrench` | 0/11 — — | `bd`, `g1three`, `inspire`, `panda`, `rethink`, `robotiq140`, `robotiq3f`, `robotiq85`, `umi`, `xarm`, `yumi` |


### Reading it

**A three-finger hand is among the best in the fleet.** `robotiq3f` holds 15 of
30 and covers **seven of the ten object classes** — level with `robotiq140` and
one behind `xarm`, and ahead of five of the eight two-finger jaws including the
Panda the demonstration was recorded with. It is the only hand besides `xarm`
and `robotiq140` to hold the `pot`, which has handles rather than flat sides.
That is the finger-count claim this document set out to test, and for three
fingers it holds.

**Two objects are refused by every hand.** `wrench` 0 of 11 and `nut_square` 0
of 11, with `nut_round` managing 1. All three are flat metal parts lying on a
table, and their failure signature is identical and unlike any other: reach
error is small — the arm gets where it was told — and **closure reaches 1.00**,
which means the jaws shut on air. The grasp point is not on graspable material.
A flat part offers a top-down hand only a thin flange, and the fingers would
have to descend past it into the table to enclose anything.

**The objects that work are the ones with a graspable section**: `can` 8 of 11,
then `cereal`, `hammer`, `milk` and `mug` at 6 of 11. Two of those are new here
and neither is a box — the `hammer` has a handle and an offset centre of mass,
the `mug` a thin wall and a rim — so the working set is not just "things a
parallel jaw likes".

**Four hands hold almost nothing**: `inspire` 0 of 27, `g1three` 0 of 29, `bd`
1 of 23, `umi` 2 of 24. These are exactly the four whose `grip_site` sits at the
gripper's base rather than at its fingertips, which is discussed in section 9.

### What separates a grasp that holds from one that does not

### What separates a grasp that holds from one that does not

| quantity | held | lifted then lost | never lifted |
|---|---|---|---|
| reach error, total mm | 3.0 (2.3–71.3) | 8.3 (2.4–180.7) | 51.2 (2.3–505.0) |
| reach error, closing axis mm | 0.3 (0.0–30.7) | 2.8 (0.0–114.4) | 8.0 (0.0–284.4) |
| closure at lift | 0.80 (0.22–1.24) | 0.86 (0.48–1.00) | 1.00 (0.97–25.72) |
| lift height mm | 143 (39–182) | 124 (37–150) | 0 (-841–19) |
| slip max mm | 17.3 (1.9–201.2) | 247.6 (22.0–1869.0) | 248.2 (48.7–1429.6) |
| grip force at close, N | 71.3 (0.0–730.8) | 15.1 (0.0–404.9) | 0.0 (0.0–332.2) |
| survivors in the funnel | 17 (1–29) | 16 (2–28) | 14 (1–31) |

(median, with the full range in brackets)


**Reach error is the strongest single discriminator, and the closing-axis
component is the sharpest form of it**: a held grasp misses by a median of
**0.3 mm** along the axis that decides whether the object ends up between the
jaws, against 8.0 mm for one that never lifts. The scalar distance separates
too (3.0 mm against 51.2 mm) but less cleanly, which is the point section 7.27
of `ROBOTICS_NOTES.md` makes about mixing a millimetre-scale budget with a
130 mm one.

**Closure at the lift separates almost perfectly, and it is not a threshold.**
Grasps that hold reach a median 0.80; grasps that never lift reach **1.00**,
with a minimum of 0.97. A reading of 1.00 means the jaws travelled their whole
range, which they can only do with nothing between them. But the *value* that
means "gripping" is hand-dependent: `robotiq3f` holds objects at 0.92 to 0.99,
where a Panda at 0.99 has closed on air. That is the evidence for a per-pair
closing table rather than one more universal threshold, and it is section 10.

**Grip force is not a discriminator on its own.** The median at the close is
71 N for a held grasp and 0 N for one that never lifts — but the range for
"never lifted" runs to **332 N**. A hand can hit an object very hard and not
hold it, which is what three of the four failing hands do.

### What it does not settle

Whether a *filtered* failure is a hand failure. The bench selects with the
funnel's pick-side stages and screens for reachability, so a pair scored 0 may
still have a grasp neither stage would have kept. `inspire` is the clearest
case: 46 of its 100 candidates on a can are reachable and the funnel kept 8 of
them, none reachable, before the screen was added.

And three grasps per pair is few. Experiment P measured the spread *within* a
pair to exceed the spread *between* pairs, so a 0 of 3 is weaker evidence than
it looks.

### Every grasp

| hand | object | rank | surv | reach mm | closing mm | lift mm | closure | slip mm | force N | held % | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bd` | `bread` | 0 | 9 | 10.0 | -2.5 | -0 | 1.00 | 268.3 | 0.0 | 0 | never_lifted |
| `bd` | `can` | 0 | 31 | 228.4 | +169.7 | -16 | 1.00 | 510.1 | 0.0 | 0 | never_lifted |
| `bd` | `can` | 1 | 31 | 13.4 | -7.7 | -16 | 1.00 | 346.8 | 0.0 | 0 | never_lifted |
| `bd` | `can` | 2 | 31 | 10.5 | -3.6 | -0 | 1.00 | 244.8 | 113.3 | 0 | never_lifted |
| `bd` | `cereal` | 0 | 6 | 3.1 | -0.2 | -64 | 1.00 | 243.6 | 0.0 | 0 | never_lifted |
| `bd` | `cereal` | 1 | 6 | 4.9 | -0.4 | -64 | 1.00 | 243.2 | 0.0 | 0 | never_lifted |
| `bd` | `cereal` | 2 | 6 | 2.6 | +0.1 | -64 | 1.00 | 242.5 | 0.0 | 0 | never_lifted |
| `bd` | `hammer` | 0 | 4 | 18.5 | +2.7 | -0 | 1.00 | 231.9 | 162.9 | 3 | never_lifted |
| `bd` | `milk` | 0 | 24 | 505.0 | +49.6 | -43 | 1.47 | 144.4 | 0.0 | 0 | never_lifted |
| `bd` | `milk` | 1 | 24 | 2.5 | +0.0 | -43 | 1.00 | 243.5 | 0.0 | 0 | never_lifted |
| `bd` | `milk` | 2 | 24 | 2.5 | -0.1 | -43 | 1.00 | 245.2 | 0.0 | 0 | never_lifted |
| `bd` | `mug` | 0 | 28 | 2.7 | -0.0 | 150 | 1.24 | 22.4 | 730.8 | 100 | **held** |
| `bd` | `mug` | 1 | 28 | 66.0 | -14.0 | -11 | 1.00 | 168.8 | 62.6 | 0 | never_lifted |
| `bd` | `mug` | 2 | 28 | 255.3 | +196.7 | -11 | 1.00 | 723.2 | 0.0 | 0 | never_lifted |
| `bd` | `nut_round` | 0 | 31 | 229.1 | +120.8 | 0 | 1.00 | 571.2 | 0.0 | 0 | never_lifted |
| `bd` | `nut_round` | 1 | 31 | 70.2 | +52.7 | 0 | 1.00 | 545.4 | 0.0 | 0 | never_lifted |
| `bd` | `nut_round` | 2 | 31 | 302.6 | +171.0 | 0 | 1.00 | 97.1 | 0.0 | 0 | never_lifted |
| `bd` | `nut_square` | 0 | 20 | 102.8 | +77.6 | 0 | 1.00 | 600.5 | 0.0 | 0 | never_lifted |
| `bd` | `nut_square` | 1 | 20 | 66.3 | +23.0 | 0 | 1.00 | 390.1 | 0.0 | 0 | never_lifted |
| `bd` | `nut_square` | 2 | 20 | 315.5 | +205.9 | 0 | 1.00 | 711.6 | 0.0 | 0 | never_lifted |
| `bd` | `pot` | 0 | 4 | 439.6 | +284.4 | 0 | 1.00 | 468.3 | 0.0 | 0 | never_lifted |
| `bd` | `pot` | 1 | 4 | 11.3 | -4.5 | -800 | 1.00 | 219.3 | 0.0 | 0 | never_lifted |
| `bd` | `wrench` | 0 | 1 | 78.1 | +15.6 | 0 | 1.00 | 403.2 | 0.0 | 0 | never_lifted |
| `g1three` | `bread` | 0 | 7 | 27.0 | +4.2 | -0 | 1.03 | 229.8 | 0.0 | 0 | never_lifted |
| `g1three` | `bread` | 1 | 7 | 40.4 | +14.7 | -0 | 1.02 | 224.9 | 0.0 | 0 | never_lifted |
| `g1three` | `bread` | 2 | 7 | 24.0 | -14.5 | -4 | 1.05 | 175.8 | 0.0 | 0 | never_lifted |
| `g1three` | `can` | 0 | 25 | 8.8 | -5.4 | -800 | 1.05 | 795.6 | 0.0 | 0 | never_lifted |
| `g1three` | `can` | 1 | 25 | 25.9 | +0.5 | -0 | 1.02 | 235.8 | 133.9 | 0 | never_lifted |
| `g1three` | `can` | 2 | 25 | 12.5 | -0.3 | -16 | 1.02 | 252.9 | 0.0 | 0 | never_lifted |
| `g1three` | `cereal` | 0 | 19 | 19.8 | +15.6 | -64 | 0.97 | 250.4 | 0.0 | 0 | never_lifted |
| `g1three` | `cereal` | 1 | 19 | 48.7 | +34.4 | -64 | 1.02 | 295.1 | 0.0 | 0 | never_lifted |
| `g1three` | `cereal` | 2 | 19 | 10.2 | +0.3 | -64 | 1.02 | 283.5 | 0.0 | 0 | never_lifted |
| `g1three` | `hammer` | 0 | 2 | 51.1 | +23.9 | 0 | 1.14 | 238.2 | 0.0 | 0 | never_lifted |
| `g1three` | `hammer` | 1 | 2 | 27.2 | +15.9 | -0 | 1.02 | 249.5 | 96.0 | 0 | never_lifted |
| `g1three` | `milk` | 0 | 18 | 8.5 | +0.7 | -841 | 0.97 | 980.6 | 0.0 | 0 | never_lifted |
| `g1three` | `milk` | 1 | 18 | 10.4 | +1.1 | -841 | 1.03 | 964.5 | 0.0 | 0 | never_lifted |
| `g1three` | `milk` | 2 | 18 | 3.4 | +1.9 | -33 | 1.03 | 257.6 | 0.0 | 0 | never_lifted |
| `g1three` | `mug` | 0 | 22 | 64.8 | +18.6 | 0 | 1.02 | 218.2 | 191.2 | 0 | never_lifted |
| `g1three` | `mug` | 1 | 22 | 6.8 | +2.5 | -11 | 1.03 | 285.7 | 0.0 | 0 | never_lifted |
| `g1three` | `mug` | 2 | 22 | 88.5 | +69.6 | 0 | 1.02 | 261.7 | 299.5 | 0 | never_lifted |
| `g1three` | `nut_round` | 0 | 21 | 96.1 | +42.4 | 0 | 1.13 | 306.3 | 27.9 | 0 | never_lifted |
| `g1three` | `nut_round` | 1 | 21 | 65.9 | +1.5 | 0 | 1.06 | 260.8 | 23.9 | 0 | never_lifted |
| `g1three` | `nut_round` | 2 | 21 | 51.6 | +2.3 | 0 | 1.02 | 216.3 | 0.0 | 0 | never_lifted |
| `g1three` | `nut_square` | 0 | 17 | 66.3 | -12.2 | 0 | 1.06 | 233.0 | 13.8 | 0 | never_lifted |
| `g1three` | `nut_square` | 1 | 17 | 87.9 | -10.0 | 0 | 1.06 | 248.6 | 10.5 | 0 | never_lifted |
| `g1three` | `nut_square` | 2 | 17 | 89.5 | +32.9 | 0 | 1.07 | 268.1 | 12.1 | 0 | never_lifted |
| `g1three` | `pot` | 0 | 8 | 8.7 | -4.6 | 18 | 1.02 | 246.4 | 0.0 | 0 | never_lifted |
| `g1three` | `pot` | 1 | 8 | 10.9 | -7.9 | -0 | 1.02 | 237.6 | 10.2 | 0 | never_lifted |
| `g1three` | `pot` | 2 | 8 | 4.1 | -3.6 | 19 | 1.15 | 240.6 | 0.0 | 0 | never_lifted |
| `g1three` | `wrench` | 0 | 13 | 6.6 | -2.0 | -11 | 1.02 | 242.9 | 0.0 | 0 | never_lifted |
| `g1three` | `wrench` | 1 | 13 | 16.7 | +1.6 | -11 | 1.02 | 234.6 | 0.0 | 0 | never_lifted |
| `g1three` | `wrench` | 2 | 13 | 66.7 | -15.8 | 0 | 1.02 | 202.0 | 0.0 | 0 | never_lifted |
| `inspire` | `bread` | 0 | 18 | 63.7 | -21.8 | -0 | 25.61 | 221.2 | 0.0 | 0 | never_lifted |
| `inspire` | `bread` | 1 | 18 | 24.4 | -2.4 | -0 | 25.61 | 275.2 | 0.0 | 0 | never_lifted |
| `inspire` | `bread` | 2 | 18 | 24.6 | +6.6 | -0 | 25.49 | 243.8 | 0.0 | 0 | never_lifted |
| `inspire` | `can` | 0 | 18 | 38.2 | -3.7 | -16 | 25.72 | 272.8 | 49.6 | 0 | never_lifted |
| `inspire` | `can` | 1 | 18 | 34.5 | +14.0 | -16 | 25.58 | 220.4 | 97.6 | 0 | never_lifted |
| `inspire` | `can` | 2 | 18 | 7.4 | +0.1 | -16 | 25.51 | 243.1 | 14.8 | 0 | never_lifted |
| `inspire` | `cereal` | 0 | 14 | 70.1 | -28.3 | -64 | 25.49 | 345.3 | 0.0 | 0 | never_lifted |
| `inspire` | `cereal` | 1 | 14 | 77.4 | +38.9 | 0 | 25.59 | 231.2 | 219.6 | 0 | never_lifted |
| `inspire` | `cereal` | 2 | 14 | 54.3 | +17.4 | -0 | 25.59 | 218.6 | 193.3 | 0 | never_lifted |
| `inspire` | `hammer` | 0 | 3 | 219.4 | +59.2 | 0 | 25.49 | 322.1 | 0.0 | 0 | never_lifted |
| `inspire` | `hammer` | 1 | 3 | 82.3 | -19.7 | 0 | 25.60 | 170.9 | 0.0 | 0 | never_lifted |
| `inspire` | `hammer` | 2 | 3 | 80.4 | -26.0 | 0 | 25.57 | 151.2 | 0.0 | 0 | never_lifted |
| `inspire` | `milk` | 0 | 13 | 13.0 | +4.9 | -43 | 25.45 | 274.5 | 57.5 | 0 | never_lifted |
| `inspire` | `milk` | 1 | 13 | 8.9 | +3.5 | -43 | 25.53 | 255.1 | 0.0 | 0 | never_lifted |
| `inspire` | `milk` | 2 | 13 | 2.3 | +0.2 | -43 | 25.55 | 258.3 | 0.0 | 0 | never_lifted |
| `inspire` | `mug` | 0 | 3 | 70.3 | -2.2 | 0 | 25.58 | 284.5 | 189.9 | 0 | never_lifted |
| `inspire` | `mug` | 1 | 3 | 97.3 | +86.1 | 0 | 25.46 | 293.7 | 332.2 | 0 | never_lifted |
| `inspire` | `mug` | 2 | 3 | 87.4 | -55.0 | 0 | 25.56 | 229.6 | 219.0 | 0 | never_lifted |
| `inspire` | `nut_round` | 0 | 2 | 185.8 | +125.8 | 0 | 25.62 | 48.7 | 0.0 | 0 | never_lifted |
| `inspire` | `nut_round` | 1 | 2 | 47.3 | +3.1 | 0 | 25.46 | 188.1 | 77.2 | 0 | never_lifted |
| `inspire` | `nut_square` | 0 | 18 | 42.0 | -9.9 | 0 | 25.55 | 235.3 | 0.6 | 0 | never_lifted |
| `inspire` | `nut_square` | 1 | 18 | 51.3 | -8.3 | 0 | 25.37 | 262.9 | 55.7 | 0 | never_lifted |
| `inspire` | `nut_square` | 2 | 18 | 79.0 | -8.0 | 0 | 25.52 | 293.2 | 17.9 | 0 | never_lifted |
| `inspire` | `pot` | 0 | 8 | 73.7 | +15.2 | 0 | 25.67 | 195.1 | 226.4 | 0 | never_lifted |
| `inspire` | `pot` | 1 | 8 | 66.6 | +7.2 | 0 | 25.68 | 195.1 | 259.7 | 0 | never_lifted |
| `inspire` | `pot` | 2 | 8 | 39.8 | +0.6 | 0 | 25.66 | 230.2 | 135.1 | 0 | never_lifted |
| `inspire` | `wrench` | 0 | 1 | 172.8 | -16.5 | 0 | 25.53 | 120.0 | 0.0 | 0 | never_lifted |
| `panda` | `bread` | 0 | 24 | 2.5 | +0.2 | -0 | 1.00 | 252.8 | 1.2 | 0 | never_lifted |
| `panda` | `bread` | 1 | 24 | 2.7 | +0.1 | -0 | 0.99 | 221.7 | 22.7 | 0 | never_lifted |
| `panda` | `bread` | 2 | 24 | 2.7 | +0.0 | 145 | 0.53 | 18.2 | 39.0 | 100 | **held** |
| `panda` | `can` | 0 | 25 | 272.6 | +12.7 | -0 | 1.00 | 226.3 | 0.0 | 0 | never_lifted |
| `panda` | `can` | 1 | 25 | 206.9 | -1.2 | -0 | 1.00 | 215.6 | 0.0 | 0 | never_lifted |
| `panda` | `can` | 2 | 25 | 3.3 | +0.4 | 142 | 0.38 | 2.9 | 40.0 | 100 | **held** |
| `panda` | `cereal` | 0 | 15 | 4.9 | -4.6 | 146 | 0.62 | 14.5 | 32.8 | 100 | **held** |
| `panda` | `cereal` | 1 | 15 | 80.2 | -19.8 | 0 | 1.00 | 240.3 | 0.0 | 0 | never_lifted |
| `panda` | `cereal` | 2 | 15 | 3.5 | -0.2 | 0 | 1.00 | 242.5 | 31.2 | 0 | never_lifted |
| `panda` | `hammer` | 0 | 1 | 2.8 | -0.0 | 136 | 0.51 | 23.3 | 34.9 | 100 | **held** |
| `panda` | `milk` | 0 | 17 | 2.4 | +0.1 | 147 | 0.51 | 1.9 | 39.4 | 100 | **held** |
| `panda` | `milk` | 1 | 17 | 2.3 | -0.1 | -43 | 1.00 | 378.3 | 37.4 | 0 | never_lifted |
| `panda` | `milk` | 2 | 17 | 2.4 | -0.2 | 149 | 0.33 | 5.0 | 39.8 | 100 | **held** |
| `panda` | `mug` | 0 | 11 | 7.2 | -1.2 | 0 | 1.00 | 245.2 | 31.6 | 0 | never_lifted |
| `panda` | `mug` | 1 | 11 | 5.4 | -2.3 | 0 | 1.00 | 238.4 | 27.8 | 0 | never_lifted |
| `panda` | `mug` | 2 | 11 | 15.5 | -3.7 | 0 | 1.00 | 239.7 | 42.8 | 0 | never_lifted |
| `panda` | `nut_round` | 0 | 15 | 75.7 | +3.2 | 0 | 1.00 | 282.9 | 0.0 | 0 | never_lifted |
| `panda` | `nut_round` | 1 | 15 | 95.9 | +51.0 | 0 | 1.00 | 296.0 | 0.0 | 0 | never_lifted |
| `panda` | `nut_round` | 2 | 15 | 96.8 | -2.0 | 0 | 1.00 | 312.5 | 0.0 | 0 | never_lifted |
| `panda` | `nut_square` | 0 | 15 | 167.3 | -103.8 | 0 | 1.00 | 169.0 | 0.0 | 0 | never_lifted |
| `panda` | `nut_square` | 1 | 15 | 106.4 | +59.3 | 0 | 1.00 | 290.1 | 0.0 | 0 | never_lifted |
| `panda` | `nut_square` | 2 | 15 | 94.4 | -42.6 | 0 | 1.00 | 300.7 | 0.0 | 0 | never_lifted |
| `panda` | `pot` | 0 | 6 | 9.4 | +9.3 | 0 | 1.00 | 240.0 | 0.0 | 0 | never_lifted |
| `panda` | `pot` | 1 | 6 | 9.9 | +0.9 | 0 | 1.00 | 254.5 | 2.1 | 0 | never_lifted |
| `panda` | `pot` | 2 | 6 | 2.8 | -0.1 | 0 | 1.00 | 235.8 | 0.0 | 0 | never_lifted |
| `panda` | `wrench` | 0 | 7 | 7.3 | +2.2 | 0 | 1.00 | 250.2 | 0.0 | 0 | never_lifted |
| `panda` | `wrench` | 1 | 7 | 7.8 | +2.8 | 0 | 1.00 | 234.5 | 0.0 | 0 | never_lifted |
| `panda` | `wrench` | 2 | 7 | 19.2 | -6.4 | 0 | 1.00 | 213.0 | 0.0 | 0 | never_lifted |
| `rethink` | `bread` | 0 | 22 | 5.7 | -2.1 | -0 | 1.00 | 353.3 | 22.4 | 0 | never_lifted |
| `rethink` | `bread` | 1 | 22 | 2.7 | -0.0 | -0 | 1.00 | 246.6 | 7.4 | 0 | never_lifted |
| `rethink` | `bread` | 2 | 22 | 6.6 | +2.2 | -0 | 1.00 | 280.1 | 36.0 | 0 | never_lifted |
| `rethink` | `can` | 0 | 24 | 2.3 | +0.1 | 147 | 0.32 | 12.5 | 39.4 | 100 | **held** |
| `rethink` | `can` | 1 | 24 | 2.6 | +0.0 | 145 | 0.48 | 250.1 | 37.2 | 33 | lifted_then_lost |
| `rethink` | `can` | 2 | 24 | 2.4 | -0.0 | 5 | 1.00 | 309.9 | 36.8 | 17 | never_lifted |
| `rethink` | `cereal` | 0 | 13 | 26.1 | -2.2 | 0 | 1.00 | 241.5 | 82.7 | 0 | never_lifted |
| `rethink` | `cereal` | 1 | 13 | 2.8 | +0.1 | 145 | 0.67 | 9.7 | 34.0 | 100 | **held** |
| `rethink` | `hammer` | 0 | 5 | 259.8 | -253.1 | 0 | 1.00 | 271.3 | 0.0 | 0 | never_lifted |
| `rethink` | `hammer` | 1 | 5 | 29.0 | +24.8 | 0 | 1.00 | 394.4 | 61.0 | 0 | never_lifted |
| `rethink` | `hammer` | 2 | 5 | 79.9 | +43.2 | 0 | 1.00 | 221.3 | 0.0 | 0 | never_lifted |
| `rethink` | `milk` | 0 | 17 | 2.4 | -0.1 | 142 | 0.61 | 15.8 | 35.4 | 100 | **held** |
| `rethink` | `milk` | 1 | 17 | 2.6 | +0.1 | -43 | 1.00 | 245.4 | 0.0 | 0 | never_lifted |
| `rethink` | `milk` | 2 | 17 | 4.7 | +0.6 | 145 | 0.56 | 12.8 | 38.2 | 100 | **held** |
| `rethink` | `mug` | 0 | 4 | 3.0 | +0.0 | 149 | 0.86 | 18.1 | 21.9 | 100 | **held** |
| `rethink` | `nut_round` | 0 | 2 | 111.0 | -52.9 | 0 | 1.00 | 335.2 | 0.0 | 0 | never_lifted |
| `rethink` | `nut_round` | 1 | 2 | 10.3 | -3.3 | 0 | 1.00 | 365.5 | 0.3 | 0 | never_lifted |
| `rethink` | `nut_square` | 0 | 13 | 79.8 | -29.1 | 0 | 1.00 | 299.0 | 0.0 | 0 | never_lifted |
| `rethink` | `nut_square` | 1 | 13 | 82.3 | -32.2 | 0 | 1.00 | 295.3 | 0.0 | 0 | never_lifted |
| `rethink` | `nut_square` | 2 | 13 | 23.7 | +22.6 | 0 | 1.00 | 236.7 | 0.0 | 0 | never_lifted |
| `rethink` | `pot` | 0 | 6 | 28.1 | -9.9 | 0 | 1.00 | 258.2 | 39.6 | 0 | never_lifted |
| `rethink` | `pot` | 1 | 6 | 11.1 | +0.9 | 0 | 1.00 | 205.2 | 8.8 | 0 | never_lifted |
| `rethink` | `pot` | 2 | 6 | 9.2 | +0.7 | 0 | 1.00 | 255.6 | 4.2 | 0 | never_lifted |
| `rethink` | `wrench` | 0 | 4 | 85.4 | +79.0 | 0 | 1.00 | 368.0 | 0.0 | 0 | never_lifted |
| `rethink` | `wrench` | 1 | 4 | 106.7 | -67.3 | 0 | 1.00 | 186.0 | 0.0 | 0 | never_lifted |
| `robotiq140` | `bread` | 0 | 9 | 17.8 | +13.3 | -0 | 1.00 | 336.9 | 0.0 | 0 | never_lifted |
| `robotiq140` | `bread` | 1 | 9 | 36.9 | -5.7 | 138 | 0.90 | 17.1 | 12.9 | 100 | **held** |
| `robotiq140` | `bread` | 2 | 9 | 7.3 | -1.0 | 134 | 0.90 | 19.7 | 0.0 | 100 | **held** |
| `robotiq140` | `can` | 0 | 17 | 4.3 | -0.4 | 126 | 0.85 | 416.7 | 0.0 | 33 | lifted_then_lost |
| `robotiq140` | `can` | 1 | 17 | 11.2 | +7.7 | 120 | 0.89 | 144.3 | 7.2 | 67 | lifted_then_lost |
| `robotiq140` | `can` | 2 | 17 | 12.9 | +6.6 | 120 | 0.87 | 29.7 | 6.3 | 100 | **held** |
| `robotiq140` | `cereal` | 0 | 23 | 3.3 | +0.2 | 146 | 0.93 | 29.0 | 31.6 | 100 | **held** |
| `robotiq140` | `cereal` | 1 | 23 | 2.4 | +0.0 | 136 | 0.93 | 31.4 | 32.2 | 100 | **held** |
| `robotiq140` | `cereal` | 2 | 23 | 2.4 | +0.0 | 135 | 0.93 | 30.5 | 30.2 | 100 | **held** |
| `robotiq140` | `hammer` | 0 | 3 | 8.1 | +2.9 | 103 | 0.93 | 45.1 | 0.0 | 100 | **held** |
| `robotiq140` | `milk` | 0 | 25 | 2.3 | -0.1 | 0 | 1.00 | 249.4 | 4.6 | 0 | never_lifted |
| `robotiq140` | `milk` | 1 | 25 | 2.6 | -0.2 | 142 | 0.91 | 24.5 | 27.4 | 100 | **held** |
| `robotiq140` | `milk` | 2 | 25 | 2.4 | -0.0 | 144 | 0.91 | 20.1 | 30.1 | 100 | **held** |
| `robotiq140` | `mug` | 0 | 20 | 2.8 | -0.4 | 130 | 0.80 | 14.9 | 74.6 | 100 | **held** |
| `robotiq140` | `mug` | 1 | 20 | 3.1 | -0.3 | 137 | 0.79 | 23.2 | 67.9 | 100 | **held** |
| `robotiq140` | `mug` | 2 | 20 | 2.6 | -0.3 | 136 | 0.84 | 33.9 | 42.9 | 100 | **held** |
| `robotiq140` | `nut_round` | 0 | 19 | 83.0 | +23.3 | 0 | 1.00 | 241.1 | 2.3 | 0 | never_lifted |
| `robotiq140` | `nut_round` | 1 | 19 | 88.1 | -29.3 | 0 | 0.99 | 265.0 | 0.0 | 0 | never_lifted |
| `robotiq140` | `nut_round` | 2 | 19 | 60.7 | +51.4 | 0 | 1.00 | 164.8 | 71.6 | 0 | never_lifted |
| `robotiq140` | `nut_square` | 0 | 11 | 106.0 | -16.0 | 0 | 1.00 | 302.3 | 0.0 | 0 | never_lifted |
| `robotiq140` | `nut_square` | 1 | 11 | 109.6 | -9.1 | 0 | 1.00 | 320.3 | 0.0 | 0 | never_lifted |
| `robotiq140` | `nut_square` | 2 | 11 | 101.5 | +20.2 | 0 | 1.00 | 303.9 | 0.1 | 0 | never_lifted |
| `robotiq140` | `pot` | 0 | 8 | 20.2 | +2.9 | 39 | 0.94 | 194.5 | 22.3 | 100 | **held** |
| `robotiq140` | `pot` | 1 | 8 | 133.3 | +101.6 | 0 | 1.00 | 156.8 | 0.0 | 0 | never_lifted |
| `robotiq140` | `pot` | 2 | 8 | 18.7 | +3.2 | 0 | 1.00 | 235.0 | 14.5 | 0 | never_lifted |
| `robotiq140` | `wrench` | 0 | 18 | 114.4 | +92.1 | -811 | 1.00 | 1429.6 | 0.0 | 0 | never_lifted |
| `robotiq140` | `wrench` | 1 | 18 | 25.0 | -3.4 | 95 | 0.75 | 79.0 | 0.0 | 0 | lifted_then_lost |
| `robotiq140` | `wrench` | 2 | 18 | 111.0 | -74.6 | -811 | 1.00 | 1334.6 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `bread` | 0 | 28 | 118.0 | -89.0 | -0 | 0.99 | 253.8 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `bread` | 1 | 28 | 40.5 | +4.8 | 135 | 0.98 | 265.0 | 37.3 | 67 | lifted_then_lost |
| `robotiq3f` | `bread` | 2 | 28 | 22.0 | -15.8 | 121 | 0.98 | 31.6 | 64.0 | 100 | **held** |
| `robotiq3f` | `can` | 0 | 14 | 3.1 | +0.1 | 4 | 0.99 | 412.5 | 129.2 | 0 | never_lifted |
| `robotiq3f` | `can` | 1 | 14 | 4.4 | +1.9 | 145 | 0.94 | 19.9 | 196.4 | 100 | **held** |
| `robotiq3f` | `can` | 2 | 14 | 3.0 | -1.6 | 141 | 0.95 | 9.9 | 129.4 | 100 | **held** |
| `robotiq3f` | `cereal` | 0 | 17 | 3.8 | -1.2 | 143 | 0.93 | 19.6 | 150.7 | 100 | **held** |
| `robotiq3f` | `cereal` | 1 | 17 | 2.7 | -0.3 | 144 | 0.94 | 13.2 | 122.7 | 100 | **held** |
| `robotiq3f` | `cereal` | 2 | 17 | 4.1 | -0.9 | 143 | 0.93 | 15.8 | 162.6 | 100 | **held** |
| `robotiq3f` | `hammer` | 0 | 19 | 18.6 | +12.8 | 127 | 0.99 | 23.1 | 22.1 | 100 | **held** |
| `robotiq3f` | `hammer` | 1 | 19 | 24.3 | +10.9 | 129 | 0.95 | 10.9 | 63.4 | 100 | **held** |
| `robotiq3f` | `hammer` | 2 | 19 | 130.5 | +10.4 | 0 | 0.99 | 175.0 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `milk` | 0 | 24 | 2.5 | -0.3 | 148 | 0.92 | 24.8 | 139.0 | 100 | **held** |
| `robotiq3f` | `milk` | 1 | 24 | 2.6 | -0.1 | 151 | 0.94 | 11.3 | 174.3 | 100 | **held** |
| `robotiq3f` | `milk` | 2 | 24 | 2.4 | -0.3 | 146 | 0.93 | 21.3 | 195.8 | 100 | **held** |
| `robotiq3f` | `mug` | 0 | 22 | 2.4 | -0.2 | 139 | 0.74 | 20.5 | 331.8 | 100 | **held** |
| `robotiq3f` | `mug` | 1 | 22 | 13.8 | +8.5 | 122 | 0.79 | 245.0 | 84.5 | 63 | lifted_then_lost |
| `robotiq3f` | `mug` | 2 | 22 | 4.0 | +1.2 | 134 | 0.78 | 32.8 | 256.3 | 100 | **held** |
| `robotiq3f` | `nut_round` | 0 | 25 | 121.4 | -6.1 | 0 | 0.99 | 157.1 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `nut_round` | 1 | 25 | 107.2 | -38.0 | 0 | 1.00 | 333.1 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `nut_round` | 2 | 25 | 98.7 | -18.6 | 0 | 0.99 | 220.7 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `nut_square` | 0 | 7 | 71.7 | +17.1 | 0 | 1.01 | 274.2 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `nut_square` | 1 | 7 | 66.4 | -43.2 | 0 | 1.00 | 254.6 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `nut_square` | 2 | 7 | 104.1 | -29.1 | 0 | 0.99 | 256.4 | 0.1 | 0 | never_lifted |
| `robotiq3f` | `pot` | 0 | 8 | 3.3 | +0.2 | 51 | 0.99 | 201.2 | 16.9 | 100 | **held** |
| `robotiq3f` | `pot` | 1 | 8 | 3.4 | +0.6 | 146 | 0.96 | 11.3 | 115.5 | 100 | **held** |
| `robotiq3f` | `pot` | 2 | 8 | 15.8 | +4.5 | 0 | 1.00 | 181.4 | 9.7 | 0 | never_lifted |
| `robotiq3f` | `wrench` | 0 | 12 | 4.3 | +2.2 | 147 | 0.68 | 22.0 | 0.0 | 0 | lifted_then_lost |
| `robotiq3f` | `wrench` | 1 | 12 | 109.6 | -63.7 | 0 | 0.99 | 512.2 | 0.0 | 0 | never_lifted |
| `robotiq3f` | `wrench` | 2 | 12 | 12.9 | -2.6 | 0 | 0.99 | 241.9 | 0.0 | 0 | never_lifted |
| `robotiq85` | `bread` | 0 | 23 | 2.6 | +0.3 | -0 | 0.99 | 246.6 | 0.0 | 0 | never_lifted |
| `robotiq85` | `bread` | 1 | 23 | 15.0 | -9.8 | -0 | 0.99 | 551.1 | 0.0 | 0 | never_lifted |
| `robotiq85` | `bread` | 2 | 23 | 22.5 | -10.6 | -0 | 1.01 | 429.7 | 15.3 | 0 | never_lifted |
| `robotiq85` | `can` | 0 | 17 | 2.4 | +0.0 | 144 | 0.74 | 7.7 | 83.9 | 100 | **held** |
| `robotiq85` | `can` | 1 | 17 | 259.5 | -0.1 | -0 | 0.99 | 98.0 | 0.0 | 0 | never_lifted |
| `robotiq85` | `can` | 2 | 17 | 2.4 | +0.1 | 146 | 0.73 | 7.3 | 92.8 | 100 | **held** |
| `robotiq85` | `cereal` | 0 | 16 | 2.6 | +0.1 | 141 | 0.88 | 8.4 | 60.5 | 100 | **held** |
| `robotiq85` | `cereal` | 1 | 16 | 94.2 | -15.7 | 0 | 0.99 | 219.1 | 0.0 | 0 | never_lifted |
| `robotiq85` | `cereal` | 2 | 16 | 4.0 | +0.5 | 150 | 0.88 | 23.0 | 39.6 | 100 | **held** |
| `robotiq85` | `hammer` | 0 | 7 | 16.5 | -6.6 | 130 | 0.88 | 13.4 | 15.8 | 100 | **held** |
| `robotiq85` | `hammer` | 1 | 7 | 12.1 | -7.8 | 137 | 0.87 | 18.5 | 0.4 | 100 | **held** |
| `robotiq85` | `hammer` | 2 | 7 | 142.9 | +25.3 | 0 | 1.01 | 229.0 | 0.0 | 0 | never_lifted |
| `robotiq85` | `milk` | 0 | 23 | 2.4 | -0.1 | 149 | 0.79 | 11.1 | 101.8 | 100 | **held** |
| `robotiq85` | `milk` | 1 | 23 | 19.6 | +6.2 | 152 | 0.84 | 9.3 | 48.3 | 100 | **held** |
| `robotiq85` | `milk` | 2 | 23 | 3.1 | -0.1 | -43 | 1.00 | 323.8 | 73.3 | 0 | never_lifted |
| `robotiq85` | `mug` | 0 | 9 | 7.1 | +1.3 | 144 | 0.61 | 15.4 | 121.5 | 100 | **held** |
| `robotiq85` | `mug` | 1 | 9 | 5.5 | -3.0 | 133 | 0.50 | 17.3 | 221.1 | 100 | **held** |
| `robotiq85` | `mug` | 2 | 9 | 35.7 | -26.8 | 178 | 0.59 | 23.4 | 215.1 | 100 | **held** |
| `robotiq85` | `nut_round` | 0 | 1 | 32.5 | +19.9 | 0 | 0.98 | 230.3 | 10.5 | 0 | never_lifted |
| `robotiq85` | `nut_square` | 0 | 15 | 223.3 | +89.2 | 0 | 1.25 | 386.6 | 0.0 | 0 | never_lifted |
| `robotiq85` | `nut_square` | 1 | 15 | 67.9 | -10.0 | 0 | 0.99 | 354.7 | 0.0 | 0 | never_lifted |
| `robotiq85` | `nut_square` | 2 | 15 | 72.6 | -27.8 | 0 | 1.00 | 272.5 | 0.0 | 0 | never_lifted |
| `robotiq85` | `pot` | 0 | 9 | 20.3 | +0.4 | 0 | 1.01 | 237.5 | 32.4 | 0 | never_lifted |
| `robotiq85` | `pot` | 1 | 9 | 2.4 | +0.1 | 48 | 0.92 | 365.8 | 58.2 | 67 | lifted_then_lost |
| `robotiq85` | `pot` | 2 | 9 | 48.7 | +32.9 | 0 | 1.00 | 195.3 | 0.0 | 0 | never_lifted |
| `robotiq85` | `wrench` | 0 | 14 | 16.8 | -5.7 | 0 | 1.00 | 225.5 | 0.0 | 0 | never_lifted |
| `robotiq85` | `wrench` | 1 | 14 | 3.2 | +0.1 | 111 | 0.95 | 253.5 | 0.0 | 0 | lifted_then_lost |
| `robotiq85` | `wrench` | 2 | 14 | 5.4 | -2.0 | 0 | 0.99 | 259.2 | 0.0 | 0 | never_lifted |
| `umi` | `bread` | 0 | 21 | 68.8 | +9.5 | -0 | 1.01 | 190.9 | 0.0 | 0 | never_lifted |
| `umi` | `bread` | 1 | 21 | 53.9 | +6.5 | -4 | 1.01 | 221.5 | 0.0 | 0 | never_lifted |
| `umi` | `bread` | 2 | 21 | 54.4 | +8.2 | -0 | 1.01 | 215.2 | 0.0 | 0 | never_lifted |
| `umi` | `can` | 0 | 29 | 2.3 | +0.1 | -16 | 1.01 | 247.7 | 0.0 | 0 | never_lifted |
| `umi` | `can` | 1 | 29 | 61.0 | +9.0 | 87 | 0.53 | 3.9 | 0.0 | 100 | **held** |
| `umi` | `can` | 2 | 29 | 6.0 | +0.4 | -16 | 1.01 | 241.5 | 0.0 | 0 | never_lifted |
| `umi` | `cereal` | 0 | 26 | 5.0 | -0.8 | -64 | 1.01 | 251.3 | 0.0 | 0 | never_lifted |
| `umi` | `cereal` | 1 | 26 | 5.0 | -0.4 | -64 | 1.01 | 250.7 | 0.0 | 0 | never_lifted |
| `umi` | `cereal` | 2 | 26 | 5.0 | -1.8 | -64 | 1.01 | 250.4 | 0.0 | 0 | never_lifted |
| `umi` | `hammer` | 0 | 1 | 71.3 | -30.7 | 106 | 0.70 | 20.6 | 32.9 | 100 | **held** |
| `umi` | `milk` | 0 | 18 | 7.6 | +5.1 | -43 | 1.01 | 244.1 | 0.0 | 0 | never_lifted |
| `umi` | `milk` | 1 | 18 | 133.1 | +109.7 | 0 | 1.00 | 441.7 | 0.0 | 0 | never_lifted |
| `umi` | `milk` | 2 | 18 | 2.3 | -0.1 | -43 | 1.01 | 250.2 | 23.1 | 0 | never_lifted |
| `umi` | `mug` | 0 | 7 | 172.4 | -17.0 | 8 | 1.01 | 191.3 | 40.8 | 0 | never_lifted |
| `umi` | `mug` | 1 | 7 | 2.4 | +0.1 | -11 | 1.01 | 309.7 | 0.0 | 0 | never_lifted |
| `umi` | `mug` | 2 | 7 | 93.2 | +34.9 | -0 | 1.00 | 319.1 | 0.0 | 0 | never_lifted |
| `umi` | `nut_round` | 0 | 25 | 207.6 | -11.8 | 0 | 1.00 | 448.7 | 0.0 | 0 | never_lifted |
| `umi` | `nut_round` | 1 | 25 | 109.9 | +5.6 | 0 | 1.00 | 294.0 | 0.0 | 0 | never_lifted |
| `umi` | `nut_round` | 2 | 25 | 162.6 | +44.7 | 0 | 1.00 | 113.5 | 0.0 | 0 | never_lifted |
| `umi` | `nut_square` | 0 | 25 | 57.1 | -1.5 | 0 | 1.00 | 160.1 | 0.0 | 0 | never_lifted |
| `umi` | `nut_square` | 1 | 25 | 78.3 | -15.8 | 0 | 1.00 | 582.0 | 0.0 | 0 | never_lifted |
| `umi` | `nut_square` | 2 | 25 | 67.6 | +3.7 | -800 | 1.00 | 240.4 | 0.0 | 0 | never_lifted |
| `umi` | `pot` | 0 | 2 | 180.7 | +114.4 | 37 | 1.00 | 23.8 | 404.9 | 33 | lifted_then_lost |
| `umi` | `wrench` | 0 | 2 | 2.6 | +0.1 | 0 | 1.01 | 262.4 | 0.0 | 0 | never_lifted |
| `xarm` | `bread` | 0 | 16 | 2.6 | -0.6 | 147 | 0.71 | 11.8 | 428.1 | 100 | **held** |
| `xarm` | `bread` | 1 | 16 | 2.6 | +0.3 | -0 | 1.01 | 251.4 | 5.7 | 0 | never_lifted |
| `xarm` | `bread` | 2 | 16 | 2.3 | +0.1 | 146 | 0.71 | 3.2 | 475.3 | 100 | **held** |
| `xarm` | `can` | 0 | 19 | 3.1 | +0.2 | 144 | 0.55 | 2.7 | 723.4 | 100 | **held** |
| `xarm` | `can` | 1 | 19 | 2.6 | -0.3 | 145 | 0.55 | 10.1 | 716.2 | 100 | **held** |
| `xarm` | `can` | 2 | 19 | 2.4 | -0.3 | 146 | 0.56 | 12.0 | 704.9 | 100 | **held** |
| `xarm` | `cereal` | 0 | 14 | 107.8 | -71.4 | 0 | 1.01 | 315.4 | 0.0 | 0 | never_lifted |
| `xarm` | `cereal` | 1 | 14 | 4.1 | -1.0 | 154 | 0.74 | 19.3 | 437.1 | 100 | **held** |
| `xarm` | `cereal` | 2 | 14 | 2.6 | +0.2 | 182 | 0.76 | 50.7 | 387.7 | 100 | **held** |
| `xarm` | `hammer` | 0 | 3 | 2.8 | -0.2 | 138 | 0.67 | 17.3 | 540.6 | 100 | **held** |
| `xarm` | `hammer` | 1 | 3 | 2.9 | +0.1 | 142 | 0.74 | 11.8 | 414.2 | 100 | **held** |
| `xarm` | `hammer` | 2 | 3 | 2.9 | +0.0 | 137 | 0.71 | 15.2 | 457.6 | 100 | **held** |
| `xarm` | `milk` | 0 | 17 | 2.5 | -0.3 | 162 | 0.65 | 30.9 | 587.5 | 100 | **held** |
| `xarm` | `milk` | 1 | 17 | 8.7 | -0.2 | 143 | 0.64 | 40.3 | 728.2 | 100 | **held** |
| `xarm` | `milk` | 2 | 17 | 220.6 | +195.5 | 0 | 1.02 | 54.9 | 0.0 | 0 | never_lifted |
| `xarm` | `mug` | 0 | 14 | 2.4 | +0.0 | 140 | 0.81 | 14.6 | 414.1 | 100 | **held** |
| `xarm` | `mug` | 1 | 14 | 38.3 | +9.5 | 141 | 0.86 | 23.6 | 364.7 | 100 | **held** |
| `xarm` | `mug` | 2 | 14 | 38.7 | +1.2 | 147 | 0.86 | 43.4 | 282.3 | 100 | **held** |
| `xarm` | `nut_round` | 0 | 21 | 108.0 | -15.0 | 0 | 1.02 | 151.5 | 0.0 | 0 | never_lifted |
| `xarm` | `nut_round` | 1 | 21 | 69.6 | -1.7 | 0 | 1.02 | 191.5 | 0.0 | 0 | never_lifted |
| `xarm` | `nut_round` | 2 | 21 | 340.1 | +75.0 | 0 | 1.02 | 119.1 | 0.0 | 0 | never_lifted |
| `xarm` | `nut_square` | 0 | 19 | 301.2 | -79.9 | 0 | 1.02 | 67.8 | 0.0 | 0 | never_lifted |
| `xarm` | `nut_square` | 1 | 19 | 106.2 | -37.5 | 0 | 1.02 | 225.8 | 0.0 | 0 | never_lifted |
| `xarm` | `nut_square` | 2 | 19 | 95.5 | -1.0 | 0 | 1.02 | 271.7 | 0.0 | 0 | never_lifted |
| `xarm` | `pot` | 0 | 6 | 2.3 | +0.0 | 138 | 0.74 | 17.3 | 424.9 | 100 | **held** |
| `xarm` | `pot` | 1 | 6 | 2.5 | +0.2 | 88 | 0.83 | 63.9 | 385.7 | 100 | **held** |
| `xarm` | `pot` | 2 | 6 | 2.7 | -0.4 | 87 | 0.84 | 65.8 | 367.3 | 100 | **held** |
| `xarm` | `wrench` | 0 | 12 | 110.8 | +67.0 | 0 | 1.02 | 301.6 | 0.0 | 0 | never_lifted |
| `xarm` | `wrench` | 1 | 12 | 5.0 | +2.6 | 0 | 1.01 | 269.9 | 0.0 | 0 | never_lifted |
| `xarm` | `wrench` | 2 | 12 | 28.0 | -14.6 | 125 | 0.88 | 80.3 | 0.0 | 0 | lifted_then_lost |
| `yumi` | `bread` | 0 | 7 | 89.8 | -12.1 | -0 | 1.00 | 229.8 | 0.0 | 0 | never_lifted |
| `yumi` | `bread` | 1 | 7 | 4.7 | +2.2 | -0 | 1.00 | 278.9 | 28.3 | 0 | never_lifted |
| `yumi` | `bread` | 2 | 7 | 5.3 | -1.0 | 150 | 0.49 | 1869.0 | 22.9 | 33 | lifted_then_lost |
| `yumi` | `can` | 0 | 14 | 3.3 | -0.2 | 152 | 0.22 | 3.3 | 37.8 | 100 | **held** |
| `yumi` | `can` | 1 | 14 | 5.3 | +1.0 | 146 | 0.22 | 2.2 | 43.9 | 100 | **held** |
| `yumi` | `can` | 2 | 14 | 5.5 | +1.8 | -815 | 1.00 | 1126.5 | 49.9 | 0 | never_lifted |
| `yumi` | `cereal` | 0 | 5 | 264.0 | -6.3 | 0 | 1.00 | 61.3 | 0.0 | 0 | never_lifted |
| `yumi` | `hammer` | 0 | 4 | 6.3 | -0.3 | 0 | 1.00 | 249.1 | 28.4 | 0 | never_lifted |
| `yumi` | `hammer` | 1 | 4 | 4.1 | +0.7 | 0 | 1.00 | 267.3 | 7.1 | 0 | never_lifted |
| `yumi` | `hammer` | 2 | 4 | 142.5 | +25.6 | 0 | 1.00 | 180.2 | 0.0 | 0 | never_lifted |
| `yumi` | `milk` | 0 | 14 | 93.8 | -4.7 | 0 | 1.00 | 214.0 | 4.6 | 0 | never_lifted |
| `yumi` | `milk` | 1 | 14 | 255.0 | +37.3 | 0 | 1.00 | 304.0 | 0.0 | 0 | never_lifted |
| `yumi` | `milk` | 2 | 14 | 2.7 | +0.2 | -43 | 1.00 | 261.0 | 0.0 | 0 | never_lifted |
| `yumi` | `mug` | 0 | 1 | 216.9 | +52.7 | 0 | 1.00 | 79.2 | 0.0 | 0 | never_lifted |
| `yumi` | `nut_round` | 0 | 4 | 5.2 | -0.3 | 0 | 1.00 | 240.9 | 0.0 | 0 | never_lifted |
| `yumi` | `nut_round` | 1 | 4 | 5.5 | -1.4 | 141 | 0.61 | 5.8 | 24.7 | 100 | **held** |
| `yumi` | `nut_round` | 2 | 4 | 3.6 | +0.4 | 0 | 1.00 | 244.8 | 0.0 | 0 | never_lifted |
| `yumi` | `nut_square` | 0 | 1 | 11.0 | -5.8 | 0 | 1.00 | 376.9 | 0.0 | 0 | never_lifted |
| `yumi` | `pot` | 0 | 6 | 16.4 | -7.2 | 0 | 1.00 | 234.8 | 12.5 | 0 | never_lifted |
| `yumi` | `pot` | 1 | 6 | 5.5 | +4.5 | 0 | 1.00 | 241.9 | 0.0 | 0 | never_lifted |
| `yumi` | `pot` | 2 | 6 | 10.5 | +1.8 | 0 | 1.00 | 254.2 | 34.3 | 0 | never_lifted |
| `yumi` | `wrench` | 0 | 2 | 4.5 | -0.8 | 0 | 1.00 | 243.9 | 0.0 | 0 | never_lifted |

---

## 9. Experiment D — describing a hand, and four defects the multi-finger hands were failing on

Commits `4e0c750` and `598a0e5`. Prompted by an objection that a 26 mm
fingertip gap for a five-finger hand was not credible. It was not, and finding
out why turned up four separate faults, **none of them the hand**.

### Why the objection was right

Section 4 measured the space *between* the Inspire hand's fingers as 28 mm and
concluded it could not accept a 65 mm can. That measurement is of the wrong
quantity. GraspGen-X's paper (arXiv:2606.00998) defines its gripper encoding as
**"the region traversed by the robot fingers during its grasping motion"** — the
volume the fingers sweep *through* while closing, not the gap they leave.

The two are nearly the same box for a parallel jaw, and that is precisely how
the error survived: the old measurement reproduced the shipped apertures for the
Panda, the Yumi and the Robotiq 2F-140, and that agreement looked like
validation. The Panda's own config settles the definition beyond argument — its
finger joint travels **0.04 m per side** and `extents[0]` is **0.08**, their
total travel, with `extents2[0]` = 0.04, the travel from half-closed.

Measured properly, the Inspire hand sweeps **156.3 mm** and `g1three`
**127.4 mm**, against declared apertures of 80 and 100. These are not hands that
open by 26 mm. The paper also reports the released model scoring **0.363 on the
Inspire Hand** and 0.404 on the Surge Hand, so five-finger hands are inside what
it is expected to handle.

### The four defects, in the order they were found

**1. The arm does not hold still while a gripper is measured.** A zero arm
action is not a command to stay put. With a heavy hand the wrist wandered
**800 mm** over three forty-step settles, and that motion lands in
`closed - opened` where it reads as finger travel. It is what made the Inspire
hand report `finger_travel_mm` of −1.05 and `plus_one_closes: False`. With the
arm frozen — its own degrees of freedom only, never the gripper's, which would
be §7.32's error — the same hand reads +1.5 mm and True, and **`g1three`'s
anisotropy goes from 1.2 to 116.3**: from "no closing axis at all" to a cleanly
single-axis hand.

**2. The closing axis had no sign.** `finger_axes` takes it from the first right
singular vector of the finger displacements, and **the sign of a singular vector
is arbitrary**. Harmless for a two-finger jaw, where a half turn about the
approach swaps the fingers and is the same grasp — which is why
`closing_angle_deg` is folded into [−90, 90) in the first place. Not harmless
for a hand GraspGen-X declares `symmetric: false`, where it puts the thumb on
the wrong side.

The convention was read from GraspGen-X's own assets rather than guessed.
Loading each multi-finger URDF and driving it to the `open` pose its config
declares puts the **odd finger on +X** every time:

| gripper | bodies on +X | bodies on −X | odd finger |
|---|---|---|---|
| `inspire_hand` | 4 | 8 | +X |
| `unitree_g1` | 3 | 4 | +X |
| `barrett_hand` | 2 | 4 | +X |
| `sharpa_wave` | 5 | 17 | +X |

Resolved at the **open** pose, because a hand that curls brings every fingertip
together and at the closed pose the sides stop meaning anything. `g1three`'s
axis flips 179 degrees; `robotiq3f` moves 1.1; the symmetric hands are
untouched. `ROBOTICS_NOTES.md` §7.33 is this same lesson one level up — *a grasp
is a pose, not an axis*.

Measured as reachability, over five planner grasps on a can:

| hand | as measured | sign flipped |
|---|---|---|
| `inspire` | **0/5** | 2/5 |
| `g1three` | 2/5 | 3/5 |
| `robotiq3f` | 3/5 | 4/5 |
| `panda` (symmetric) | **5/5** | 3/5 |

**3. The bench was measuring the controller, not the gripper.** Two parts. It
never asked whether the arm could reach a candidate — on `inspire/can`, **46 of
100** candidates are reachable and **0 of the 8** the funnel kept were, so the
hand was scored on grasps it never had a chance to attempt. And it drove the arm
by Cartesian impedance, under which the hand finished **43 to 100 mm** from poses
inverse kinematics reports as reachable. It now screens for reachability and
drives by joint position through IK, as the Tier 2 replay does.
`robotiq3f/can` went from 25–32 mm of reach error to **3.0–4.4 mm** and from 1
of 3 grasps held to 2 of 3.

The effect on the whole fleet was large. Against the first bench run:

| hand | before | after |
|---|---|---|
| `xarm` | 7/27 | **18/30** |
| `robotiq85` | 5/22 | **11/28** |
| `robotiq3f` | 13/28 | **15/30** |
| `rethink` | 3/27 | 5/25 |
| `panda` | 6/24 | 6/28 |

**4. `set_closure` is a silent no-op on every anthropomorphic hand.** See
section 10 — it is the mechanism the closing table is commanded through, so it
belongs there.

### Where the generator stands

`describe` now measures the swept volume as the paper defines it. It
**over-reads uniformly** against the shipped figures — panda 104.5 mm against
80.0, robotiq85 131.4 against 85.0, yumi 58.4 against 50.0 — and the cause is
visible rather than mysterious: the box here contains every *moving geom*,
including knuckles and outer links that swing wide, where the shipped value is
the finger travel alone. Tightening it to the distal geoms is the obvious next
step and has not been done.

So the module is **not finished**. What it is now is a measurement of the right
quantity with a known bias, rather than a measurement of the wrong quantity that
happened to agree on five hands.

### What it does not settle

**The Inspire hand still holds nothing, and the cause is not established.**
Six candidates have been tested and eliminated:

| hypothesis | test | verdict |
|---|---|---|
| it does not actuate | 53 mm of finger travel | **no** |
| it cannot open wide enough | 156 mm swept volume against a 65 mm can | **no** |
| it cannot reach its grasps | reaches to 2.3 mm under position control | **no** |
| its actuators are too weak | `kp` 2 → 200, force limit 20 → 50 N: 0/6 both ways | **no** |
| the contact point is wrong | swept centre instead of closed fingertips: 1/6 against 0/6 | marginal |
| the roll is a half turn out | 0/6 against 0/6 | **no** |
| it knocks the object on the way in | approached and descended with the hand held open: **0.0 mm** of object motion | **no** |
| the contact depth is wrong | swept along the approach axis from −60 to +60 mm, two objects, 14 runs: **no band works** | **no** |
| the jaws close too far | closure swept 0.3, 0.5, 0.7, 0.9, `+1`, three objects: no fraction holds | **no** |

One row of the depth sweep is worth keeping rather than averaging away. At
−60 mm on the milk the hand applies **0 N** — it never touches the object at the
close — and the milk still ends **43 mm** down. So something other than the
fingers disturbs it, and the approach test above says it is not the descent
either. That is unexplained and is the thread to pull next.

What is measured is that it reaches the grasp, touches the object on **16 of
27** attempts with 50 to 332 N, and has **0%** contact through the carry. It
grips and the object is expelled. The remaining candidate is where the fingers
meet the object — a contact point for a hand that closes into its palm is not
the centroid of its fingertips, and the swept volume's centre is only a first
guess at a better one.

`g1three` is in the same position at 0 of 29, with the added doubt that its two
models may not be the same hand: GraspGen-X's `unitree_g1` measures
[182, 70, 71] mm across its own base frame against robosuite's
[122, 49, 116] mm. `bd`/`bd_spot` by contrast agree to within 4 mm on every
axis, so the comparison is meaningful and that pairing is sound.

---

## 10. Experiment G — the closing sweep, and what a per-pair table is worth

125 runs: five hands × five objects × five commanded closures (0.3, 0.5, 0.7,
0.9 and plain `+1`), one grasp per cell, otherwise Experiment F's conditions.

### Why

Four closing rules have failed the same way, and `FINDINGS.md` §8z states the
reason: the window between gripping and crushing is per-object **and** per-hand,
and no constant sits inside all of them. The response here is not a fifth
constant but a **measurement** of where each pair's window is.

### One distinction that has to be made first

**Commanded closure and measured closure are different quantities and they
point opposite ways.** Experiment F found that a *measured* closure of 1.00 at
the lift always means failure — the jaws travelled their whole range, which they
can only do with nothing between them. This experiment finds that *commanding*
`+1` is usually fine, because a commanded `+1` that meets an object stops early
and then measures around 0.5.

Both are true. One is an outcome, the other an input, and conflating them would
produce exactly the wrong table.

### And one defect it exposed before it could produce anything

`set_closure` is how a partial closure is commanded, and it is a **silent no-op
on every anthropomorphic hand**. robosuite's grippers close two ways: most
*integrate*, so `format_action` reads `current_action` and adds a step in the
command's direction — for those `set_closure`, which writes `current_action`
directly, is the only way to ask for a partial close. The Inspire, G1, Fourier,
Ability and SchunkSvh hands *pass through*: `format_action` maps the action onto
their actuators and never reads `current_action`.

Measured before the fix, a sweep over all five fractions on `inspire/can` and
`inspire/milk` returned **identical lift, force and carry at every fraction**.
`grippers.commands_position` now reads which kind a hand is, and for a
passthrough hand the fraction is commanded directly as `2f − 1`. The split is
**not** by finger count — `robotiq3f` has three fingers and integrates.

### Result: 17 of 25 pairs hold somewhere, and `+1` is not always inside the band

| hand | object | 0.3 | 0.5 | 0.7 | 0.9 | `+1` | band | chosen |
|---|---|---|---|---|---|---|---|---|
| `panda` | `bread` | n | n | n | n | n | 0/5 | — |
| `panda` | `can` | n | n | n | n | n | 0/5 | — |
| `panda` | `cereal` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `panda` | `milk` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `panda` | `mug` | n | n | n | n | n | 0/5 | — |
| `robotiq140` | `bread` | n | **Y** | **Y** | n | n | 2/5 | 0.7 |
| `robotiq140` | `can` | n | n | **Y** | n | n | 1/5 | 0.7 |
| `robotiq140` | `cereal` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `robotiq140` | `milk` | n | n | n | n | n | 0/5 | — |
| `robotiq140` | `mug` | n | **Y** | **Y** | **Y** | **Y** | 4/5 | 0.9 |
| `robotiq3f` | `bread` | n | n | n | n | n | 0/5 | — |
| `robotiq3f` | `can` | n | n | n | n | n | 0/5 | — |
| `robotiq3f` | `cereal` | n | n | n | n | **Y** | 1/5 | `+1` |
| `robotiq3f` | `milk` | n | n | n | **Y** | **Y** | 2/5 | `+1` |
| `robotiq3f` | `mug` | n | n | n | **Y** | **Y** | 2/5 | `+1` |
| `robotiq85` | `bread` | n | n | n | n | n | 0/5 | — |
| `robotiq85` | `can` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `robotiq85` | `cereal` | n | n | n | **Y** | **Y** | 2/5 | `+1` |
| `robotiq85` | `milk` | n | n | n | **Y** | **Y** | 2/5 | `+1` |
| `robotiq85` | `mug` | n | **Y** | **Y** | **Y** | **Y** | 4/5 | 0.9 |
| `xarm` | `bread` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `xarm` | `can` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `xarm` | `cereal` | n | n | n | n | n | 0/5 | — |
| `xarm` | `milk` | n | n | **Y** | **Y** | **Y** | 3/5 | 0.9 |
| `xarm` | `mug` | n | n | n | **Y** | **Y** | 2/5 | `+1` |

### Reading it

**`+1` is right far more often than not, and the sweep confirms rather than
replaces it.** It is inside the holding band for 15 of the 17 pairs that hold at
all, which is why `FINDINGS.md` §8z's decision to keep it stands.

**The two exceptions are the case a table exists for.** `robotiq140/bread` holds
at 0.5 and 0.7 and **fails at 0.9 and `+1`** — the squeeze-out, the same
phenomenon that makes the bread the worst object in every campaign. The same
hand's `can` holds **only** at 0.7. Both are the widest-jawed hand in the fleet
closing on the two objects it can most easily crush.

So the table's value, scored honestly: per-pair closure holds **17 of 25** pairs
against **15 of 25** for always-`+1`. Two cells, both on one hand.

**What the table is more useful for is margin.** `robotiq140/mug` and
`robotiq85/mug` hold across four of the five fractions; `robotiq140/can` holds at
exactly one. A pair with a band of 1 is a single lucky sample, not a calibrated
pair, and that is worth knowing before a campaign leans on it.

### Consequences

`tpgpt/grasp/closing_table.json` records the measurement — the chosen fraction,
the band width and every fraction that held, per pair, with `+1` as the default.
**It is not wired into any driver.** Doing that is its own commit with its own
before-and-after, and on this evidence it would be worth two cells in
twenty-five, which is not yet a reason to move a default that every existing
result depends on.

### What it does not settle

One grasp per cell. Experiment P measured the spread *within* a pair to exceed
the spread between pairs, so a cell reading 0/5 may be one bad grasp rather than
a bad closure window, and a band of 1 may be noise. Five hands and five objects,
none of them the hands that hold nothing.

---

## 11. Experiment H — five more hands, and the split that explains them

`outputs/grasp_bench_new`, 56 grasps. Five hands × five objects × three grasps,
Experiment F's conditions exactly.

### Why

Three-finger diversity rested on a single hand and five-finger diversity on
none. robosuite has five more multi-finger hands that had never been tried, and
`describe` can author the GraspGen-X description each of them needs. Four were
authored; the fifth needed nothing, because
`RobotiqThreeFingerDexterousGripper` is the **same robosuite XML** as
`robotiq3f` with its fingers driven independently, so it pairs to the same
shipped description.

### Result

| hand | fingers | config | closing | held | reach mm | touched object |
|---|---|---|---|---|---|---|
| `robotiq3f_dex` | 3 | **GraspGen-X's own** (`robotiq_3f`) | integrator | **7/15** | 3.0 | 13/15 |
| `jaco3f` | 3 | authored here | integrator | **0/14** | 3.5 | 1/14 |
| `ability` | 5 | authored here | passthrough | **0/15** | 26.0 | 4/15 |
| `fourier` | 5 | authored here | passthrough | **0/12** | 20.7 | 5/12 |
| `schunk` | 5 | authored here | passthrough | **refused** — uncalibrated closure | — | — |

### Reading it

**A second three-finger configuration works.** `robotiq3f_dex` holds 7 of 15
across four of the five objects it saw — `cereal` 3/3, `milk` 2/3, `mug` 1/3,
`hammer` 1/3, `can` 0/3.

**Every hand running a description GraspGen-X wrote grips; every hand running a
description written here does not.** That is six hands and it is the clearest
signal in this document:

| description | hands | result |
|---|---|---|
| GraspGen-X's own | `robotiq3f`, `robotiq3f_dex` | 15/30 and 7/15 |
| authored by `describe` | `jaco3f`, `ability`, `fourier`, `schunk` | 0/41 |

**`jaco3f` is the proof that this is the description and not the hand.** It
reaches its commanded grasp to a median **3.5 mm** — *better* than the 3.0 mm of
the hand that works — and touches the object **once in fourteen attempts**. The
arm goes exactly where the description says, and the object is not there. A
depth sweep from −80 to +20 mm along its approach axis finds no offset that
rescues it.

The suspect is `fingertip`, which sets where along the approach GraspGen-X puts
the grasp point and which feeds `grasp_to_eef_pose` directly. `describe` takes
it as the far face of the swept box; on the one hand where a shipped value
exists to compare, the Panda, it comes out 93.4 mm against a declared 103.4.

### What it does not settle

Whether the five-finger hands can grasp at all. Three of the four failures are
five-fingered, but so is the fourth failure's cause — `jaco3f` has three fingers
and an integrator close, and fails identically. Until an authored description is
validated against a hand that already works, nothing about the five-finger hands
can be concluded from these zeros.

`schunk` was **refused before any physics**, by the `closure_calibrated`
precondition: its spread travel measures −61.8 mm. That is the Inspire hand's
failure again — `spread` does not describe a hand whose fingers curl — and the
precondition doing its job rather than guessing.

---

## 12. Where this leaves the two questions

### Which hands work

Sixteen hands are registered, up from nine. Eleven have been run through the
bench on ten objects, five more on five objects.

| hand | fingers | held | objects held | note |
|---|---|---|---|---|
| `xarm` | 2 | 18/30 | 7 of 10 | best overall |
| **`robotiq3f`** | **3** | **15/30** | **7 of 10** | |
| `robotiq140` | 2 | 13/28 | 7 of 10 | |
| `robotiq85` | 2 | 11/28 | 5 of 10 | |
| **`robotiq3f_dex`** | **3** | **7/15** | **4 of 5 tried** | same hand as `robotiq3f`, fingers driven independently |
| `panda` | 2 | 6/28 | 5 of 10 | the demonstration's own hand |
| `rethink` | 2 | 5/25 | 5 of 10 | |
| `yumi` | 2 | 3/22 | 2 of 10 | |
| `umi` | 2 | 2/24 | 2 of 10 | |
| `bd` | 2 | 1/23 | 1 of 10 | |
| `inspire` | 5 | 0/27 | 0 | |
| `g1three` | 3 | 0/29 | 0 | |
| `jaco3f` | 3 | 0/14 | 0 | authored description |
| `ability` | 5 | 0/15 | 0 | authored description |
| `fourier` | 5 | 0/12 | 0 | authored description |
| `schunk` | 5 | refused | — | uncalibrated closure |

**Three-finger diversity is established**: two working configurations, covering
7 of 10 objects, competitive with the best parallel jaws. `robotiq3f` is the
only hand besides `xarm` and `robotiq140` to hold the `pot` by its handles.

**Five-finger diversity is not**, and the reason is **not yet known to be the
hands** — see §11. No five-finger hand in this project has ever run on a
description its own authors wrote.

**Three fingers is competitive, not superior.** On the same ten objects `xarm`
holds 18 and `robotiq3f` 15. Nothing here shows that more fingers buy anything;
it shows they cost nothing. Demonstrating an advantage needs objects chosen to
require enclosure rather than pinching, and the three candidates for that here
— `wrench` and the two nuts — defeat every hand.

### Which objects work

| verdict | objects | held by |
|---|---|---|
| **good** | `can` | 8 of 11 hands |
| | `cereal`, `hammer`, `milk`, `mug` | 6 of 11 each |
| marginal | `bread` | 4 of 11 |
| | `pot` | 3 of 11 |
| **refused by every hand** | `wrench`, `nut_square` | 0 of 11 |
| | `nut_round` | 1 of 11 |

**The recommended set is the six good ones**: `can`, `cereal`, `hammer`, `milk`,
`mug`, `bread`. Two of those are new and neither is a box — the `hammer` has a
handle and a centre of mass away from it, the `mug` a thin wall and a rim — so
the working set is not merely "things a parallel jaw likes".

**The three failures are diagnosed, not merely dropped.** One hold in 88
attempts across the fleet, with an identical signature everywhere: reach error
small, closure 1.00, jaws shut on air. All three are flat metal parts lying on a
table. A top-down hand is offered a thin flange and would have to descend past
it into the table to enclose anything. They are the candidates to replace with
sized primitives, which is what a shape-controlled comparison would want anyway.

### The largest open item

`describe` authors a GraspGen-X description that gets a hand into the system —
the planner accepts it, the funnel filters it, the arm flies to its grasps to
3.5 mm — and the hand then touches nothing. Four hands, 0 of 41. Fixing
`fingertip` is the single change that would most extend this fleet, and
`robotiq3f` provides the control to validate it against: author a description
for a hand that already works on a shipped one, and require the authored version
to reproduce its result.
