# How do you write a GraspGen-X description for a gripper it has never seen?

**Written for a reader with no prior context.** GraspGen-X plans grasps for a
hand by **conditioning on its shape** rather than on weights trained for it, so
one checkpoint serves any gripper it can be told about. What it has to be told
is a `config.json` with nine keys. This document is about producing that file
for a new hand — one robosuite has and GraspGen-X does not, or one that exists
only as CAD because it is about to be 3D printed — and, more importantly, about
**checking that the file is right**, because a wrong one is accepted silently.

**Provenance.** Numbers here come from `tpgpt.grasp.describe` (which writes a
description) and `tpgpt.grasp.validate_config` (which scores one by grasping
with it). Where a run is quoted, its log is named.

---

## 1. Why "the server accepted it" means nothing

Four descriptions were authored from robosuite models and installed. The planner
accepted all four, the filter funnel passed their grasps, and the arm flew to
them to within **3.5 mm**. Across four hands, **0 of 41** grasps held anything.
Meanwhile the two hands running a description GraspGen-X's own authors wrote
held 15 of 30 and 7 of 15.

So a description has to be checked by **grasping with it**. That is what
`validate_config` does, and the rest of this document is what it found.

## 2. What the file contains, and which parts are derivable

| key | what it is | derivable from the model? |
|---|---|---|
| `sweep_volume` | the box **between** the fingers — the gap they close onto — at two closure states. Not the region they traverse; that reading was wrong and section 8 is the correction | **yes** — sections 3, 8, 10 |
| `type` | `parallel_2f`, `revolute_2f`, `revolute_3f` | yes, from the joint type and finger count |
| `symmetric` | whether a half turn about the approach is the same grasp | yes — true for a two-finger jaw |
| `bbox` | the hand's own extent | yes, from MuJoCo's exact per-geom bounds |
| `open` / `close` | joint positions at each extreme | yes |
| `links`, `standoff` | documentation; not consumed at inference | yes |
| **`fingertip`** | the tool-centre depth | **no — section 4** |

Only `sweep_volume`, `type`, `symmetric` and `fingertip` affect anything: the
released checkpoint declares `gripper_backbone: sweep_volume_v2`, so it
conditions on the swept boxes, the depth and the family. The point clouds and
TSDF grids the shipped directories carry are **not** consumed, which is why a
description needs no URDF and no retraining.

## 3. The swept volume, and three ways to get it wrong

GraspGen-X's paper (arXiv:2606.00998) defines it as *"the region traversed by
the robot fingers during its grasping motion"*, as an axis-aligned box, recorded
from fully open and from half closed.

The Panda's shipped description is the check, because its numbers are
unambiguous: its finger joint travels 0.04 m per side and `extents[0]` is
**0.08** — the two fingers' total travel — with `extents[1]` and `extents[2]` at
**0.018**, the pad's own cross-section.

Three errors, each of which produced a plausible-looking file:

**Measuring the gap between the fingers instead of the region they sweep.**
These nearly coincide for a parallel jaw and are unrelated for a hand that
curls. Measuring the gap reproduced the Panda, the Yumi and the Robotiq 2F-140
— which looked like validation — and then reported a five-finger hand opening
by 28 mm.

**Including every moving geom.** Knuckles and outer links move when the hand
closes and sweep a far wider arc than the pads. The Panda's aperture came out at
**104.6 mm** against a declared 80. Restricting to the distal band of the
fingers fixes it.

**Using geom centres rather than solid volumes.** The extent *across* the
closing axis is the pads' own cross-section, and the Panda's two fingers sit at
the same height and depth — so a box fitted to their centres is **0.0 mm** wide
in that direction. A degenerate box is not a small error: it is what the model
conditions on.

The box is therefore a hybrid, and each part is measured where it means
something: **the aperture** is the widest empty interval along the closing axis
within the pad region (topology-free — splitting the pads by the *sign* of their
coordinate assumes they straddle the axis, which a thumb opposing four fingers
does not, and that read the Inspire hand at 6.8 mm against a declared 80); the
**cross-section and depth** are the solid union over the sweep; and `offset` has
its x and y forced to **zero**, because all 26 shipped descriptions are
`[0, 0, z]` — the box is centred on the approach axis by convention.

### Measured against every hand that has both halves

`--validate` authors a description for a hand GraspGen-X already describes and
compares. Aperture, shipped against measured:

| hand | shipped | measured | error |
|---|---|---|---|
| `panda` | 80.0 | 78.9 | **−1.1** |
| `yumi` | 50.0 | 46.3 | −3.7 |
| `robotiq85` | 85.0 | 93.0 | +8.0 |
| `robotiq140` | 125.0 | 116.6 | −8.4 |
| `rethink` | 66.0 | 56.7 | −9.3 |
| `robotiq3f` | 110.0 | 116.7 | +6.7 |
| `xarm` | 85.0 | 60.8 | −24.2 |
| `umi` | 80.0 | 55.8 | −24.2 |
| `g1three` | 100.0 | 17.3 | −82.7 |
| `inspire` | 80.0 | 20.7 | −59.3 |
| `bd` | 60.0 | 20.9 | −39.1 |

**Read the boundary, not the average.** For two-finger jaws the method is good
to about 10 mm on an aperture of 50 to 125 — which is inside the annotation
noise of the shipped files themselves, since those are round numbers a person
dragged in a GUI (80, 18, 18; 66, 10, 40). For anthropomorphic hands it
**fails**, and the failure is structural rather than a tuning problem: a hand
that closes into its palm does not hold things between two opposed pads, so a
box fitted between two opposed pads is not its graspable volume.

## 4. `fingertip` is one scalar, and it has to be calibrated

Every other field is geometry. `fingertip` is the tool-centre depth — how far
along the approach axis the grasp point sits from the gripper's base — and the
hand's own geometry does not fix it.

The Panda shows why. The centre of the region its pads sweep is **93.4 mm**; the
description GraspGen-X ships says **103.4**. Neither is wrong about the hand.
They disagree because *the grasp point* is a convention: the model is trained to
place the base so that `base + depth × approach` lands on the object, and which
point of the object that is — its surface, its centre, the middle of the pads —
is not determined by the gripper.

**What it costs to get wrong.** Executing the *same* grasps at two depths, on a
40 mm cube:

| poses from | tool depth | held |
|---|---|---|
| shipped | 103.4 mm | 4/10 |
| authored | **93.4 mm** | 3/12 |
| authored | **103.4 mm** | **8/12** |

The poses the authored description produces are **good** — better here than the
shipped ones. A 10 mm error in one number took them from 67% to 25%. That is the
whole of the 0-of-41 in section 1.

**So it is swept, not derived.** `describe --calibrate` executes the same grasps
at a spread of depths and keeps the best. The planner never sees `fingertip`, so
the poses are identical at every depth and this is an exact one-variable
experiment rather than a comparison of two draws. On the Panda:

| fingertip | held |
|---|---|
| 83.4 mm | 0/7 |
| 93.4 | 0/7 |
| 103.4 | 0/7 |
| **113.4** | **4/8 (50%)** |
| 123.4 | 4/8 (50%) |
| 133.4 | 4/9 (44%) |

A clean plateau. The chosen 113.4 agrees with an independent arithmetic check:
the shipped 103.4 plus the Panda's own stored `calibrated_depth` of 7.5 is
110.9, and only the **sum** of those two is observable — they trade off exactly.

A maximum at either end of the sweep is the range's edge rather than the hand's
optimum, so that case is flagged; the Yumi hit it first time and the range was
widened.

## 5. What this means for a gripper that exists only as CAD

Inside this project a wrong `fingertip` is absorbed by
`verify.calibrate_depth`, which fits a per-hand correction in physics — so it
costs nothing **once that has been run**, and skipping it is what produced the
0-of-41. On hardware there is no second chance: the description has to be right
on its own.

The recipe is therefore:

1. **Derive the geometry** from the model — `describe` does this, and section 3
   says how far to trust it by gripper family.
2. **Calibrate `fingertip`** with a depth sweep on one reference object. In
   simulation that is `describe --calibrate`; on hardware it is a handful of
   real grasps at a spread of depths, which is the same experiment.
3. **Verify by grasping**, against a hand whose description is known good if one
   exists — `validate_config` scores two descriptions on the same object in the
   same scene, so the difference is the description and nothing else.

### One trap, because it cost a day here

The offset between a gripper's **base frame** and whatever frame your controller
aims is a *constant* of the hand, and it must be expressed in the frame the
planner writes its poses in — not in the model's root-body frame. Those differ
by up to 180 degrees across this registry, and getting it wrong is silent: every
grasp lands somewhere plausible and wrong. Measured that way, **GraspGen-X's own
Panda description held 0 of 31.**

There is a related discrepancy that is nobody's bug: GraspGen-X's Panda URDF and
robosuite's Panda XML put the gripper's base **17.50 mm** apart, constant across
every grasp. Two third-party models of the same hand simply disagree about where
its origin is. A description authored from *your own* model does not have this
problem, because it is written in the same frame you will command.

---

# Part two: reading GraspGen-X's own integration path

Everything above was reverse-engineered from the shipped descriptions. Sections
6 onward come from GraspGen-X's own repository -- its README's *Integrating a
New Gripper*, the wizard at `scripts/gripper_config_wizard.py`, and the loader
at `graspgenx/x_grippers.py`. Several conclusions above are **corrected** here,
and the corrections are marked.

## 6. What the model actually reads

`get_gripper_info` (`x_grippers.py:150-200`) builds the conditioning from
exactly this, and nothing else:

| what reaches the model | which key |
|---|---|
| `sweep_volume` = concat(extents, offset), six numbers | `sweep_volume.extents`, `.offset` |
| `sweep_volume_mid` = concat(extents2, offset2), six numbers | `sweep_volume.extents2`, `.offset2` |
| `gripper_type`, mapped to 0/1/2 | `type` |
| `depth` | `fingertip[-1]` -- **the z component only** |
| `symmetric` | `symmetric` |
| the gripper's control points | `bbox`, as width `bbox[1][0]-bbox[0][0]` and depth `bbox[1][-1]` |

`standoff` shifts a separate `grasp_volume` used downstream, not the
conditioning. `links`, the point clouds, the TSDF grid and the PointNet-VAE
embedding are **not consumed** -- `make_sweep_volume_gripper_info` fills them
with zeros. So **thirteen numbers and two flags** are the whole description,
and those are what have to be right for a hand you are going to print.

There is a tenth key this document previously missed: **`base_rotation`**, a 4x4
that takes the gripper's native frame to the canonical one (+Z approach, +X
closing) and is "applied to every downstream computation". None of the 26
curated descriptions carries it -- their URDFs were exported aligned -- so the
loader treats a missing one as identity. For a hand arriving as CAD it is the
difference between a working description and one rotated 90 or 180 degrees.

## 7. Correction: `fingertip` is derivable after all

Section 4 said `fingertip` cannot be derived and must be calibrated against
physics. That is **wrong**. The wizard derives it in one line (`:1270`):

```python
# Derive fingertip from sweep volume offset
self.fingertip = list(self.sv_offset)
```

It is the centre of the open sweep box. GraspGen-X's *other* code path,
`make_sweep_volume_gripper_info`, uses a different rule -- the box's far face,
`offset[2] + extents[2]/2`. Neither is what the curators did. Checked against
all 26:

| rule | exact matches |
|---|---|
| box centre (the wizard's) | 5 of 26 |
| box far face | 1 of 26 |

What they actually did is visible in the residuals: `fingertip - offset[2]`
comes out **-5.0, -10.0, -15.0, -20.0, -25.0, -30.0** -- round numbers. The box
centre is pushed **forward** along the approach axis by a hand-chosen amount,
median **10 mm**, maximum 37 (the UMI). That reconciles with section 4's own
physics: the Panda's calibration landed on 113.4 mm against a shipped 103.4, a
+10 mm push, and within 1 mm of the far-face rule's 112.4.

**So the honest statement for hardware is that `fingertip` is derivable to about
+-10 mm from the sweep box, and the last 10 mm is a design decision about where
along the fingers you want the object to sit.** That is a tolerance, not an
unknown. The physics calibration of section 4 remains useful as a refinement and
is no longer a prerequisite.

## 8. Correction: the swept volume is the gap, not the region traversed

Section 3 was overturned once already, in the wrong direction. The README's step
3 asks a person to *"drag the blue box to enclose the **inner volume** between
the fingertips at the open state"*, and the estimator's own docstring says
*"closing extent = **gap between innermost finger surfaces**"*. The original
reading was right. The Panda cannot settle the question, because it closes fully
to zero, so its total travel and its open aperture are the same 80 mm -- a
coincidence of that one hand which made a wrong rule look validated.

## 9. The experiment that reframes the problem: their estimator on their own hands

**Why.** Before building an authoring tool, find out what the existing one
achieves. The wizard seeds its box with `estimate_inner_sweep_volume` and then
asks a human to adjust it. If the seed were close, authoring would be solved.

**Conditions held fixed.** GraspGen-X's own estimator, its own 26 URDFs, each at
the `open` joint state its own `config.json` declares, finger links found by its
own `detect_finger_geoms`. Ground truth is the curated `config.json` for the
same hand -- treated as truth because those are the hands the paper reports
working, not because a slider is exact.

**Result.** It does not reproduce them.

| family | n | median aperture error | within 10 mm | depth | offset |
|---|---|---|---|---|---|
| 2-finger (`parallel_2f`, `revolute_2f`) | 20 | 9.5 mm | 10/20 | | 21.5 mm |
| 3-finger and multi (`revolute_3f`) | 6 | **58.8 mm** | **1/6** | | 32.5 mm |
| all | 26 | 19.9 mm | 11/26 | **2.67x too large** | 26.1 mm |

For the Franka Panda it gives `[85.0, 21.0, 53.8]` against a shipped
`[80, 18, 18]`: the aperture good to 5 mm, the depth three times too large. Two
hands degenerate outright -- `bd_spot` finds one moving link and reports a
**0 mm** aperture, and `wuji_hand` picks the wrong closing axis (1, not 0).

**What it says.** The wizard is a six-panel GUI *because* the automatic guess is
not good enough. There is no hidden derivation in the repository: the human
annotation is the method, and it is most load-bearing for exactly the
multi-finger hands. Section 3's boundary -- good on two-finger jaws, failing on
anthropomorphic hands -- is not a defect of this project's code. It is the state
of the art in the reference implementation.

**What it does not settle.** Whether a *better* estimator exists. Section 10.

## 10. A scored authoring rule

`tpgpt/grasp/authoring.py`, scored by `tpgpt/grasp/bench_authoring.py` on the
same 26 hands, against the same ground truth.

| family | n | median aperture error | within 10 mm | median depth err | median offset err |
|---|---|---|---|---|---|
| 2-finger | 18 | **2.5 mm** | **15/18** | 20.1 mm | 7.9 mm |
| 3-finger and multi | 7 | 31.0 mm | 0/7 | 5.4 mm | 4.5 mm |
| all | 25 | **4.9 mm** | **15/25** | 13.5 mm | 7.5 mm |

Three corrections do the work, each measured over all 26:

1. **The box stops at the fingertips.** `offset[2] + extents[2]/2` against the
   finger geometry's own maximum reach has a median difference of **-1.5 mm**,
   with 21 of 26 inside 8.5 mm. The baseline spans the whole finger, which is
   the entire source of its 2.67x depth inflation.
2. **The pocket ends where the gap stops being the aperture**, not only where
   the fingers meet. Stopping only on contact ran the OnRobot RG2's box over 39
   of 40 slices and left its aperture 75 mm short.
3. **The aperture is the pocket's narrowest slice**, because an object must pass
   every slice to reach it. Against median (51.3 mm multi-finger error), mean
   (50.9), 25th percentile (43.2) and 10th (36.2), the minimum gives **31.0**.

`bd_spot` is **refused** with its reason rather than described.

**What it does not settle.** The third choice was made on these 26 hands, so
these are in-sample figures. And a small error here is agreement with people who
knew the hands -- it is not evidence a grasp will hold. Section 12 is that test.

## 11. Two things MuJoCo needs that a URDF does not

Both were found by the Panda control regressing.

**Group finger points by body, not by geom.** GraspGen-X works on URDF link
meshes, one per finger segment; robosuite splits a Panda finger across a
collision hull and a separate pad. Ungrouped, a two-finger hand looks like a
four-finger one and the wrong pair counts as outermost.

**Sample mesh vertices, not bounding boxes.** A box bulges inward wherever the
shape inside it tapers. The Panda's pads face each other at +-39.4 mm, close to
the 80 mm its description declares, while the AABBs of the hulls behind them
register material at +-30 mm -- 10 mm inside the pads, through which no object
could pass. Sampling boxes gave 70.8 mm; the mesh gives **78.9**.

Controls, authored from robosuite models against the curated descriptions:

| hand | authored | curated | error |
|---|---|---|---|
| Panda | 78.9 mm | 80.0 | **-1.1** |
| Robotiq 2F-140 | 125.1 mm | 125.0 | **+0.1** |
| Robotiq 2F-85 | 96.2 mm | 85.0 | +11.2 |

## 12. Do the authored configs actually grasp?

**Why.** Sections 9-11 are geometry. A description is only worth anything if the
grasps it yields can be executed.

**Conditions held fixed.** A plain robosuite `Lift` table, one 40 mm cube, the
object cloud sampled from the cube's known geometry (perception is not under
test). Per cell: 200 proposals, top 100 kept, ranked by the planner's own score,
the top 12 approaching from above executed by `teleport_grasp` -- placed at the
grasp, closed, lifted. Held means the cube rose more than 50 mm. `tcp_depth` is
each config's own `fingertip[-1]`.

**What was varied.** The config, and only the config: the same hand, arm,
object and physics run against a description GraspGen-X's authors wrote and one
`tpgpt.grasp.describe` wrote.

**Result, two identical runs.**

| cell | run 1 | run 2 | pooled | rate |
|---|---|---|---|---|
| panda shipped | 5/12 | 3/9 | 8/21 | 38% |
| panda authored | 1/9 | 2/8 | 3/17 | 18% |
| robotiq140 shipped | 0/8 | 4/9 | 4/17 | 24% |
| robotiq140 authored | 2/7 | 3/8 | 5/15 | 33% |
| robotiq85 shipped | 4/11 | 5/11 | 9/22 | 41% |
| robotiq85 authored | **10/11** | **8/11** | **18/22** | **82%** |
| jaco3f authored (3f) | 0/8 | 0/11 | 0/19 | 0% |
| inspire shipped (5f) | 0/6 | 0/8 | 0/14 | 0% |
| inspire authored (5f) | 0/9 | 0/7 | 0/16 | 0% |

**What it says.**

*Authored descriptions now work.* Section 4 recorded **0 of 41** grasps held
across four authored hands. These hold **26 of 54** on the two-finger hands.
That is the result this document exists to reach.

*Authored versus shipped is not resolved at this n.* Pooled, 26/54 (48%) against
21/60 (35%), Fisher **p = 0.184**. Only `robotiq85` repeats in both runs and
survives a test on its own: 18/22 against 9/22, **p = 0.0122**.

*Do not read a single cell.* `robotiq140 shipped` went **0/8 in one run and 4/9
in the next**, with nothing changed. GraspGen-X's planner is an unseeded
diffusion model, so each run draws a different candidate set -- this is
`ROBOTICS_NOTES.md` §7.15, and it is why these cells were run twice.

*The Inspire cell is the useful control.* The description **GraspGen-X's own
authors wrote** also holds **0**, in both runs. The five-finger failure
therefore survives a description written by the people who built the system: it
is not an authoring failure.

**What it does not settle.** The comparison did not go through
`tpgpt.grasp.cache`, so the two arms of each pair faced different candidate
sets; a paired comparison needs cached draws. The 3-finger `jaco3f` holds
nothing in either run despite touching the cube 8 times across 19 attempts, and
that is undiagnosed.

## 13. Why five-finger hands fail, and it is not the config

`gripper_frames.json` records an **anisotropy** per hand: how one-dimensional
the finger motion is, and therefore whether a single closing axis exists at all.

| hand | anisotropy | closing angle | travel |
|---|---|---|---|
| panda | 9.6e10 | 1.1 deg | 78.4 mm |
| robotiq140 | 1324 | -0.04 deg | 54.8 mm |
| robotiq85 | 605 | 0.00 deg | 28.9 mm |
| jaco3f | 5.46 | -4.4 deg | 19.8 mm |
| robotiq3f | 4.26 | 0.4 deg | 84.7 mm |
| inspire | 5.57 | **-80.0 deg** | 1.5 mm |
| fourier | 4.24 | **-63.2 deg** | 15.6 mm |
| ability | 2.33 | **+31.6 deg** | 57.5 mm |
| schunk | 1.79 | **+51.7 deg** | **-61.8 mm** |

A parallel jaw's fingers move along one line, so the ratio is enormous. Five
fingers curling toward a palm move in genuinely different directions, the ratio
collapses to about 2, and "the closing axis" stops being a well-defined
quantity -- the measured axes for those four hands sit **31 to 80 degrees** from
their own `grip_site` frames, and the SchunkSvh reports *negative* travel and
`plus_one_closes: false`.

A GraspGen-X description assumes **one** closing direction and **two** opposing
groups. A five-finger hand has neither, so there is no correct box to compute
and no amount of estimator work will find one. This is why
`estimate_sweep_box` **refuses** the Ability hand ("no slice has the two finger
groups separated") rather than describing it, and it is consistent with §12's
Inspire control failing on a curated description.

It also explains NVIDIA's own practice: they ship the five-finger `sharpa_wave`
as `revolute_3f`, forcing it into the three-finger family with a box a person
drew for a grasp they had **chosen**. The description encodes an intention, not
a measurement -- which is information the geometry does not contain.

## 14. What to do for a gripper you are going to print

**Two or three fingers.** Author automatically. The rule is at 2.5 mm median
aperture error against hands whose descriptions work, the frame is well defined
(anisotropy at or above 2.7, closing angle within 4.4 degrees), and §12 shows
the resulting grasps hold. Supply `base_rotation` if the CAD frame is not
already +Z approach / +X closing.

**Five fingers.** Run GraspGen-X's wizard by hand. It is not merely easier: a
person choosing where the grasp pocket is supplies information §13 shows the
geometry does not contain. Declare the hand `revolute_3f`, as NVIDIA does for
`sharpa_wave`.

**Either way, the wizard is not sufficient on its own.** It gives GraspGen-X
what it needs to *generate* grasps. It knows nothing about the robot the hand is
bolted to, so it cannot supply the gripper-base to tool-frame transform needed
to *execute* one -- `alignment_rotation` and `contact_offset` here, a mounting
calibration on hardware. That is a separate per-gripper measurement.

### Two traps worth writing down

**The grasp server caches samplers and never invalidates them.**
`zmq_server.py` builds a sampler on the first request naming a gripper and keeps
it. Re-authoring a `config.json` therefore has **no effect** until the server
restarts or the name changes. §12 was run against freshly named copies for this
reason; a validation run that skipped it would silently score the previous
generation.

**Importing GraspGenX triggers a 1.57 GiB checkpoint download** into
`<repo>/ext/graspgenx_checkpoints` on first use, plus the gripper-description
pack. Budget for it before running anything that imports the package fresh.

## 15. The other half: getting a grasp from the planner onto real hardware

Sections 6-14 are about making GraspGen-X *emit* a good grasp. This section is
about *executing* one, which is a separate problem with separate measurements,
and it is the half a `config.json` says nothing about. It is written for the
case that matters here: a gripper you printed, bolted to your own arm.

### What the planner hands you, exactly

GraspGen-X returns a 4x4 pose, and that pose is **the gripper base in the
canonical frame**: `+Z` is the approach direction, `+X` is the closing
direction. That convention is uniform across all of its grippers -- it is what
`base_rotation` exists to guarantee, and if you authored your config with a
`base_rotation` that was wrong, every pose it returns is wrong by that rotation
and nothing downstream can tell.

Your robot controller does not aim the gripper base. It aims a **tool frame** --
a flange, or a TCP you configured. So between what the planner says and what you
command there is a fixed rigid transform with two parts, and both have to be
measured per gripper:

| | what it is | how wrong it can be |
|---|---|---|
| **rotation** | how your tool frame is turned relative to `+Z` approach / `+X` closing | measured across this registry's nine hands: **0.2 degrees** for the Robotiqs, **90** for the XArm, **180** for the Panda, Yumi and Rethink |
| **translation** | how far along the approach axis the grasp point sits from the base | the `fingertip` depth, **103 to 195 mm** across the same nine hands |

In this project those are `alignment_rotation` and `contact_offset` /
`calibrated_depth`, and the conversion is `R_tool = R_grasp @ alignment`.

### Why this is the part that bites

**Every failure mode here is silent.** A wrong rotation does not throw, does not
look like a crash, and does not produce an obviously bad pose. The arm flies
somewhere plausible, closes on nothing, and the natural conclusion is that the
grasp was bad. Three measured examples from this project:

* expressed in the model's root-body frame instead of the grasp frame,
  **GraspGen-X's own Panda description scored 0 of 31** -- a description written
  by its own authors, failing completely on a frame error;
* `contact_offset` looked a hand up by identity and returned a **zero** offset
  for an unmeasured hand. Zero is a plausible number, so a 41 mm frame error
  read as the arm missing its target;
* a **19.7 mm** tool-offset error turned a 28 mm placement into a 232 mm one.

And a discrepancy that is nobody's bug, worth knowing exists: GraspGen-X's Panda
URDF and robosuite's Panda XML place the gripper base **17.50 mm** apart,
constant across every grasp. Two third-party models of the same hand disagree
about where its origin is. **A gripper you authored from your own CAD does not
have this problem**, because the description and the robot are written in the
same frame -- which is one of the genuine advantages of printing your own.

### The procedure for a new printed gripper

1. **Fix the canonical frame in CAD, once.** Decide which axis of your CAD model
   is the approach and which is the closing direction, and record the rotation
   from your export frame to `+Z`/`+X` as `base_rotation` in the config. Do this
   before anything else; every later number is expressed in it.

2. **Get the rotation from CAD, then verify it physically.** The transform from
   the gripper base to your tool frame is fixed by the mounting plate, so it is
   known from the assembly -- you do not need to estimate it. What you do need
   is a check that catches a sign or an axis swap, because those are the errors
   that survive inspection. The cheap one: command the hand to a grasp you
   constructed by hand rather than one the planner produced -- straight down
   onto a cylinder on the table, closing axis across it -- and confirm the hand
   arrives the way you drew it. A 90 or 180 degree error is unmissable there and
   invisible in a planned grasp.

3. **Measure the depth, do not derive it.** Section 7 shows `fingertip` is
   derivable from the sweep box to about **+-10 mm**, and that the curators of
   GraspGen-X's own hands then adjusted it by a hand-chosen 5 to 30 mm. Ten
   millimetres is the difference between gripping an object and brushing it. The
   measurement is a handful of grasps at a spread of depths on one reference
   object, keeping everything else fixed, and taking the depth where it holds --
   `describe --calibrate` in simulation, the identical experiment by hand on
   hardware. Section 4 shows the response is a **plateau** roughly 30 mm wide,
   not a peak, so about six depths at 10 mm spacing locates it.

4. **Choose the reference object to suit the hand.** `calibrate_depth` in this
   project defaults to a 65 mm can, which is a fine reference for a jaw opening
   to 80 or 125 mm and meaningless for a narrower hand, where the sweep measures
   "a can does not fit" rather than a depth. Use something comfortably inside
   the aperture your config declares.

5. **Re-run the calibration whenever the hand changes.** Every
   `calibrated_depth` in this registry moved when the scene was corrected, by up
   to 30 mm, and the Panda's wrist-to-fingertip offset went from 41.1 mm to
   11.1. For a printed gripper, a reprint at a different tolerance or a new pad
   material is that same kind of change.

### What no config can give you

A `config.json` describes the *hand*. The transform above describes the *hand on
your robot*, and nothing in GraspGen-X's schema, wizard or asset pack carries
it -- the wizard never asks what arm you are using. So for every gripper you
print, budget one mounting measurement in addition to the description, and treat
it as the more likely source of a confusing failure, because it fails silently
and the description does not.

## 16. Authored against curated, properly paired

**Why.** Section 12 compared an authored description with a curated one by
calling the planner fresh each time. GraspGen-X's planner is an unseeded
diffusion model, so that compares draws. This routes every candidate set
through `tpgpt.grasp.cache`, so a cell's grasps are fixed on first run and
identical on every rerun, and raises n from 12 to 30.

**Conditions held fixed.** The same `Lift` table, the same 40 mm cube, the same
cloud sampled from its known geometry, the same seed, the same physics. Only
the config changes. The two arms of a pair necessarily see *different*
candidates -- changing the description is what changes what the model proposes,
and that is the thing under test.

**Result, over four draws.**

| arm | held / reachable | rate |
|---|---|---|
| curated (4 draws) | 63/163 | **39%** |
| authored (3 draws) | 36/94 | **38%** |
| authored + 10 mm push (1 draw) | 10/32 | 31% |
| all authored pooled | 46/126 | **37%** |

Authored against curated: Fisher **p = 0.716**. Per hand, nothing separates
either -- panda 34% vs 14%, robotiq140 39% vs 35%, robotiq85 40% vs 55%.

**What it says.** *An automatically authored description performs as well as one
GraspGen-X's own curators drew by hand.* That is the result this document set out
to reach, and it is what licenses pointing the tool at a gripper nobody has
described before.

**The correction it forced.** Section 12 reported `robotiq85` authored beating
curated 18/22 against 9/22 at **p = 0.0122**. That is **withdrawn**. With two
further draws it is 21/38 against 19/48, **p = 0.193**, and its per-draw rates
are **91%, 73%, 19%**. Twenty-two grasps drawn from a single candidate set are
not twenty-two independent samples -- the unit of variance here is the
**candidate set**, and treating grasps as the unit is pseudo-replication.
Caching fixed *reproducibility* without fixing *representativeness*: a cached
draw reruns identically and is still one sample. Three draws is the minimum that
exposes this, and two actively misled.

## 17. `fingertip` has two jobs, and improving one does not improve the whole

**Why.** Section 7 established that the curators push the grasp point forward
from the sweep box's centre by a median 10 mm. Does applying that help?

**Held fixed, then not.** Two experiments, deliberately different:

*With the candidate set pinned* -- same config, same cached grasps, same cube,
same seed, only `tcp_depth` moving:

| push (mm) | 0 | 5 | **10** | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| panda | 1/4 | 1/4 | **3/4** | 2/4 | 3/4 | 1/4 |
| robotiq85 | 1/9 | 4/9 | **5/9** | 1/9 | 2/8 | 2/9 |
| robotiq140 | 7/12 | 7/12 | **7/12** | 6/12 | 5/13 | 5/13 |

10 mm is the optimum on two hands and free on the third, and it lands each near
the curated value -- Panda 105.5 against 103.4, Robotiq 2F-85 133.2 against
136.0. Two independent routes to the same number.

*With the description re-authored* so the push is baked in: **10/32, 31%**,
against 36/94 (38%) without it. Push versus no push, Fisher **p = 0.529**. No
net gain.

**What it says, and it is the most useful thing in this document for anyone
authoring a config.** `fingertip[-1]` is read **twice**, in two unrelated
places:

* it reaches the model as `depth` and sets the gripper's control points, so it
  decides **which poses come back**;
* it is the tool-centre depth, so it decides **where the hand is driven** for a
  given pose.

The pinned sweep measures only the second. Moving the point forward genuinely
places the hand better on a fixed set of grasps. Re-authoring moves the first as
well, and the model then proposes a *different* set -- and the two effects
cancel. **A change that clearly improves one role of this number can be net
neutral on the task**, and an experiment that pins the candidates cannot see it.

**The default keeps the push anyway, and the reason is not the physics.** On
hardware there is no thirty-grasp A/B to run, so what you want is the value most
likely to be right before any test: the one expert curators converged on across
26 hands, which is the box centre plus about 10 mm. The measured net effect is
null (p = 0.529), not negative, and `--calibrate` overrides it per hand.

## 18. What is worth porting, and the answer is nothing yet

The point of authoring descriptions was to add hands. Over four draws:

| hand | fingers | best available | verdict |
|---|---|---|---|
| panda, robotiq85, robotiq140 | 2 | curated and authored tie | already registered; **no reason to switch** |
| `jaco3f` | 3 | authored | **0 of 47** grasps held |
| `inspire` | 5 | curated | 0 of 28; 0 at all six depths |
| `ability`, `schunk`, `fourier` | 5 | -- | cannot be described (section 13) |

**No registry entry should change.** Switching a hand from its curated config to
an authored one would move every existing campaign number for that hand in
exchange for a difference of 39% against 37% at p = 0.716. The honest reading is
that the authoring method is **validated** and that validation does not license
a change.

**`jaco3f` is the closest miss and is still a miss.** Its arm arrives at 3.3 to
4.6 mm and its fingers **sweep the cube aside**: 0.0 mm of lift with 9.5 to
40.9 mm of lateral slide, on every one of six traced grasps. A depth sweep finds
a real optimum at **172.9 mm**, 20 mm past the box centre, where it holds 1 of
7 against 0 of 7 everywhere else -- and that reproduces session one's
independent calibration of 173.6 mm, on a different config, to within 0.7 mm. A
reproducible 14% is a real peak and not a working hand.

**`inspire` is settled.** It holds nothing on the description **GraspGen-X's own
authors wrote**, nothing at any of six depths, and at one of them it knocks the
cube off the table. Section 13 gives the mechanism.

So the state is: the authoring tool is ready for a printed two- or three-finger
gripper, and it unlocks no new hand *in this simulator*, because the hands that
needed it fail for reasons a description cannot address. Adding five-finger
capability means bringing in hands that already carry a curated description --
GraspGen-X ships URDFs for `sharpa_wave` (34 links), `barrett_hand`,
`surge_hand`, `wuji_hand` and `unitree_g1`, and MuJoCo compiles all of them --
which is a robosuite modelling task, not an authoring one. **Section 20 does
that task, and section 21 measures what it bought.**

## 19. Correction: eleven of sixteen registered hands had never been tested

Section 18 concluded that no hand is worth porting. That conclusion was reached
from **five** hands -- the ones with both an authored and a curated description,
because the question being asked was "is my authoring as good as theirs". Eleven
registered hands were never put through this harness at all, four of them
three-finger hands running descriptions GraspGen-X's own authors wrote. Two of
those, `g1three` and `bd`, were paired in a previous session and had **never
been validated by grasping anything**.

Run on the same bench -- same `Lift` table, same 40 mm cube, same cached
candidate sets, same seed:

| hand | fingers | held / reachable | rate | config |
|---|---|---|---|---|
| `xarm` | 2 | 10/15 | **67%** | curated |
| `panda` | 2 | 3/5 | 60% | curated |
| `robotiq85` | 2 | 9/21 | 43% | curated |
| **`robotiq3f_dex`** | **3** | **6/15** | **40%** | curated |
| `robotiq140` | 2 | 6/16 | 38% | curated |
| `yumi` | 2 | 4/12 | 33% | curated |
| **`robotiq3f`** | **3** | **6/20** | **30%** | curated |
| `rethink` | 2 | 1/10 | 10% | curated |
| `umi` | 2 | 2/25 | 8% | curated |
| `bd` | 2 | 1/20 | 5% | curated |
| `g1three` | 3 | 0/8 | 0% | curated |
| `jaco3f` | 3 | 0/7 | 0% | authored |
| `inspire` | 5 | 0/5 | 0% | curated |

**Seven of thirteen hands clear 30%, and two of those are three-finger hands.**
`robotiq3f_dex` at 40% sits above three of the four parallel jaws. The claim
that finger diversity does not work in this system is wrong: it works for three
fingers, on two separate hands, both on curated descriptions.

What remains true is narrower and should be stated as such: **no five-finger
hand grasps**, and **no hand this project authored a description for grasps**
(`jaco3f`, 0/47 across four draws). Those are different failures -- section 13
for the first, section 18 for the second -- and neither licenses the broader
claim.

**The five-finger half of that needs one qualification, added later.** Section
21 benches `sharpa_wave`, a physically five-finger hand ported from GraspGen-X's
own curated description, at **12/81 = 15%** over seven independent candidate
sets. So "no five-finger hand grasps" is too strong as an absolute: one holds
sometimes. It is still true in the sense that matters here, which is that no
five-finger hand grasps *well enough to register* -- 15% sits below every hand
in the table above that was judged worth having, and significantly below
`robotiq3f_dex` at 40% (Fisher p = 0.033).

**The methodological fault is worth more than the numbers.** The
authored-versus-curated question needed hands with both halves, so the hand set
was chosen by what made the *comparison* clean rather than by what would answer
the *project's* question, which is which hands work. A census costs one run and
should have come first: it would have found `robotiq3f_dex` in an hour rather
than at the end, and it would have stopped `g1three` and `bd` sitting in the
registry for a session as untested pairings. **Enumerate the population before
designing the contrast.**

Two entries here also contradict older records and are flagged rather than
silently overwritten: `umi` is recorded elsewhere as lifting nothing (0 of 13)
and reaches 2 of 25 here, and `g1three` and `bd` have no prior grasp measurement
of any kind. All three are single runs on one object at n <= 25, so they rank
hands, they do not settle them.

## 20. Going the other way: porting GraspGen-X's own hands into robosuite

Every route in this document up to here runs **robosuite → GraspGen-X**: take a
hand the simulator already has, and author it a description. Section 19 shows
that route exhausted for the thing it was meant to buy, which is finger
diversity. The robosuite hands still unpaired are anthropomorphic, GraspGen-X
trained only on two- and three-finger families (section 13), and a five-finger
hand has no single closing axis for a description to name.

This section runs the route **backwards**. GraspGen-X ships 26 hands, each with
a URDF *and* a `config.json` its own authors curated. Section 16 measures
curated descriptions at 39% against this project's authored ones at 37% — so a
hand taken this way arrives with the half that works and skips the step that was
failing. What it lacks is the robosuite model, and building one is a *modelling*
job rather than an authoring one.

`tpgpt/grasp/port_gripper.py` does the conversion; `tpgpt/grasp/ported_models.py`
wraps the result as a robosuite `GripperModel`; and
`tpgpt.experiments.run_ported_bench` measures it, in two modes that are
deliberately separate — `--mount-only` (does it load, mount and move) and the
default (does it hold anything). Conflating those two is how "thirteen hands now
work" gets written down when what works is the XML.

### What the porter reads rather than assumes

The config's `open` and `close` give each joint's two end states, which become
the position actuator's `ctrlrange` and the rest pose; `fingertip` gives where
`grip_site` goes. Three joint **roles** fall out of comparing those two poses
against the URDF's joint list, rather than being hardcoded per hand:

| role | how it is recognised | what it gets |
|---|---|---|
| **driven** | named in `close`, with a value **different** from `open` | a position actuator, `ctrlrange` open→close |
| **locked** | named in `close`, with a value **equal** to `open` | an equality constraint pinning it |
| **coupled** | in the URDF, **absent** from the config | an equality constraint tying it to its driver |

The locked case is not a technicality. The Barrett's finger **spread** joint is
one: it is named in the config and does not move between the two poses, so
actuating it would splay the hand open at exactly the moment it should be
closing. The coupled case is the distal links of a real linkage, tied to their
driver at the ratio of their joint-limit spans — which for the Barrett comes out
**0.3443**, its true 1/3 linkage, *derived* rather than typed in.

### The frame needs no measurement, and that is the part worth keeping

Every other hand in this project needed a `measure_frames` pass to find
`alignment_rotation` and `contact_offset`, and §7.34 records **0.6 to 91
degrees** of commanded error from getting it wrong. A ported hand skips that
pass entirely: the porter puts the XML's `eef` body at the description's own
`fingertip` depth along **+Z**, which is the frame GraspGen-X emits poses in, so
converting a planned pose into a `grip_site` target is the identity rotation
plus that depth.

That is *asserted*, not derived, so it is checked rather than believed — §7.13's
lesson being that an unverified frame returns a plausible number rather than an
obviously wrong one. The check is the bench's `reach_mm` column, the distance
between where the arm was told to go and where its end effector actually landed.
On `sharpa_wave` it reads **3.4, 3.5 and 3.6 mm** across three independent
draws, which is the inverse-kinematics tolerance itself. A frame error cannot
hide in that column; it puts the arm somewhere else.

### The trap that costs an afternoon

robosuite's `MujocoXML.resolve_asset_dependency` (`models/base.py:54`) rewrites
every mesh path absolute **against the XML's own folder**, and **ignores the
compiler's `meshdir`**. So a port that loads perfectly in bare MuJoCo fails the
moment robosuite merges it into a robot, with an error about a file that plainly
exists. The porter copies the meshes beside the XML and drops `meshdir`; those
copies are gitignored, being verbatim and regenerable at up to 29 MB a hand.

### What mounted

`run_ported_bench --mount-only`, one Panda each, all gripper degrees of freedom
driven to `-1` then `+1` for 60 control steps each.

**Column glossary.** *travel* is the largest distance any gripper geom moves
between the two poses — the blunt "did anything happen" measure. *diameter* is
the gripper geom set's maximum pairwise distance at each pose; it is **not** a
fingertip gap, because a base geom or a splaying knuckle widens it while the
fingers converge, which is why it is reported and not interpreted. *+1 closes*
compares each driven joint's final position against the two poses the config
itself declares and asks which one it ended nearer; locked joints cast no vote.

| hand | joints | act | travel mm | diameter −1 → +1 mm | +1 closes? |
|---|---|---|---|---|---|
| `barrett_hand` | 8 | 3 | **169.3** | 288.1 → 102.6 | **yes** (3/3) |
| `sharpa_wave` | 22 | 14 | **149.6** | 169.3 → 124.5 | **yes** (14/14) |
| `surge_hand` | 10 | 9 | **95.6** | 144.2 → 86.7 | **yes** (9/9) |
| `ezgripper` | 4 | 1 | **75.3** | 212.5 → 176.0 | **yes** (1/1) |
| `wuji_hand` | 20 | 16 | 94.5 | 201.0 → 151.6 | **no** (16 joints) |
| `schunk_wsg50` | 2 | 1 | 50.5 | 115.8 → 143.0 | **no** |
| `fetch_robot` | 2 | 1 | 49.5 | 95.0 → 110.5 | **no** |
| `arx_x5` | 2 | 1 | 43.2 | 91.2 → 112.2 | **no** |
| `robotiq_hande` | 2 | 1 | 24.7 | 120.1 → 123.6 | **no** |
| `galaxea_g1` | 2 | 1 | 7.1 | 119.8 → 119.8 | no |
| `onrobot_RG6` | 6 | 1 | 1.5 | 175.7 → 175.7 | no |
| `dh_ag95` | 8 | 1 | 0.8 | 129.0 → 129.4 | no |
| `onrobot_RG2` | 6 | 1 | 0.5 | 132.9 → 132.9 | no |
| `piper_hand` | — | — | — | **does not load** | — |

Read it as three groups, not one number:

- **Thirteen of the fourteen hands this project lacked now mount on a Panda**,
  where previously none did. That is the modelling result, and it holds.
- **Nine of those move meaningfully** (over 20 mm of travel). The other four —
  `onrobot_RG2`, `onrobot_RG6`, `dh_ag95`, `galaxea_g1` — travel 0.5 to 7.1 mm.
  All four are **four-bar linkages**: their extra joints are absent from the
  config, so the porter's coupling rule ties them to the driver at a
  joint-limit ratio, and for a closed loop that over-constrains the mechanism
  and locks it. The rule is right for a serial finger and wrong for a loop.
- **Only four reach the pose the config asked for.** This is new, and section
  21 is about what it costs.

`piper_hand` is the single load failure, and it is the porter's fault rather
than the hand's: the URDF refers to its meshes through a nested
`piper_description/meshes/` path which the mesh copier flattens, so `link7.STL`
is not where the rewritten absolute path looks for it.

## 21. Correction: two opposite claims about the same hand, both wrong

Commit `bf489da`'s message says:

> **sharpa_wave grasps: 4 of 12, mean lift 110.8 mm** — comparable to robotiq3f
> at 30%, yumi at 33%, robotiq140 at 38%. The first five-finger hand in this
> project that holds anything.

**That is withdrawn.** So is the correction that first replaced it, which was
reported as "the honest answer is: none" and was not more honest for being less
flattering. Both were single-tail readings of one distribution.

### Every draw

The bench is section 19's: a Panda in a `Lift` env, one 40 mm cube, candidates
from the planner filtered to top-down approaches, each reached by inverse
kinematics, closed on, and lifted. A **draw** is one candidate set. GraspGen-X's
planner is an unseeded diffusion model, so asking it twice asks two different
questions (§7.15) — which makes the candidate set, not the individual grasp, the
unit of variance.

| draw | source | held / reachable | rate | mean lift |
|---|---|---|---|---|
| 1 | scratchpad, quoted in `bf489da` | 4/12 | **33%** | +110.8 mm |
| 2 | scratchpad, `--fresh` | 0/5 | 0% | — |
| 3 | scratchpad, `--fresh` | 1/11 | 9% | — |
| 4 | scratchpad, `--fresh` | 0/10 | 0% | −160.0 mm |
| 5 | `run_ported_bench`, cached | 4/19 | 21% | +172.6 mm |
| 6 | `run_ported_bench`, cached | 0/9 | 0% | −88.9 mm |
| 7 | `run_ported_bench`, cached | 3/15 | 20% | −79.8 mm |
| **pooled** | | **12/81** | **15%** | |

95% interval on the pooled rate: **7.9% to 24.4%**.

- **33%** is the draw quoted in the commit. It sits **above** the interval's
  upper bound — the top of the distribution, not the hand's rate.
- **4%** is draws 2–4 pooled, the figure behind "none". It sits **below** the
  lower bound — the bottom of the same distribution.
- **15%** is all seven draws, and it is the number that should be quoted.

So the fault in the second claim was identical to the fault in the first: a
per-draw rate spanning 0% to 33% was being reported as a property of the hand.
Withdrawing a flattering number on one draw and replacing it with an unflattering
number on three is not the fix. **The fix is that a single draw is not a result,
in either direction**, which is why `run_ported_bench` takes `--draws k` — *k*
independent candidate sets, each keyed into `tpgpt.grasp.cache` by its draw
index so every one of them replays exactly — and why its `--fresh` flag is
documented as making a run non-reproducible rather than as a convenience.

### Where that actually puts the hand

Against section 19's registry census, run on the same bench:

| comparison | their rate | Fisher p |
|---|---|---|
| `sharpa_wave` 15% vs `robotiq3f_dex` | 6/15 = 40% | **0.033** |
| vs `robotiq3f` | 6/20 = 30% | 0.187 |
| vs `yumi` | 4/12 = 33% | 0.211 |
| vs `rethink` | 1/10 = 10% | 1.000 |
| vs `umi` | 2/25 = 8% | 0.511 |
| vs `bd` | 1/20 = 5% | 0.455 |

It is **significantly worse than the best three-finger hand** and
**indistinguishable from the bottom three of the registry**. Not a working hand,
and not a zero: a bottom-quartile hand.

### The lift column says something the held column hides

Mean lift across the three cached draws is **+172.6, −88.9 and −79.8 mm**. A
negative mean lift is the cube ending up *below* where it started, which on a
table means it was swept off it. So the typical outcome of a `sharpa_wave`
attempt is not "fails to hold" but "destroys the scene". The mount audit says
why: its geom diameter fully shut is **124.5 mm** against a 40 mm cube, so the
hand arrives *around* the object rather than onto it, and the fingers meet the
table before they meet each other.

### And most of the ported hands were never asked the right question

Section 20's last column is the finding that outranks all of the above. **Nine
of the thirteen mounted hands do not reach the pose their own config declares
as `close` when commanded `+1`** — including `wuji_hand`, which travels 94.5 mm
and still ends nearer `open` on all sixteen of its driven joints. For the four
two-finger jaws the geom diameter *grows* on the close command (`schunk_wsg50`
115.8 → 143.0 mm, `arx_x5` 91.2 → 112.2), which is what a hand **opening** looks
like.

This matters because it reframes an earlier diagnosis. `barrett_hand`,
`wuji_hand` and `surge_hand` were recorded as mounting and actuating but holding
nothing, and the cause was put down to `grip_site` sitting past the geometry.
That explanation may still be right for the two that *do* close — but for
`wuji_hand` it is answering the wrong question, because the hand was never
commanded shut. **A bench result on a hand that fails this check measures the
sign convention, not the hand.**

It is left as a measurement rather than fixed in the same sitting, deliberately:
`CLAUDE.md`'s rule is never to change the system and the measurement together,
and §7.26 records three findings retracted for exactly that. The concrete next
step is to make the porter's actuator range agree with the config on all
thirteen, re-run `--mount-only` until the last column reads `yes` everywhere it
can, and only then bench.

### The state, in one line each

- **Porting works as modelling.** Thirteen hands that mount and actuate in
  robosuite where previously there were none, frames correct by construction
  and confirmed at 3.4–3.6 mm of reach.
- **Porting has so far produced no hand worth registering.** `sharpa_wave`, the
  only one benched across draws, holds at 15% — below every hand section 19
  found worth having.
- **No registry entry has been changed.** `GRIPPER_PAIRS` and
  `gripper_frames.json` are untouched, so no existing campaign number moves.
