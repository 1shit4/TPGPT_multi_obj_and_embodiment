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
