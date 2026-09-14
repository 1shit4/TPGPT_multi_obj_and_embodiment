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
> | F | §8 the bench — grasp, close, lift, carry | *running* |
> | D | §9 GraspGen-X descriptions authored from the MuJoCo model | *partial* — the generator reproduces a shipped aperture to 3% on clean two-finger jaws and does not describe an anthropomorphic hand |
> | G | §10 the closure sweep, and how wide each pair's band is | *not started* |
> | H | §11 the closure table applied, paired grasp for grasp | *not started* |

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
