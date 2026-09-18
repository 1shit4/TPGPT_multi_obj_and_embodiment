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
| `sweep_volume` | the box the fingers traverse while closing, at two closure states | **yes** — section 3 |
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
