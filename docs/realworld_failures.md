# The real-sized scene: what 210 cells did, and why each failure failed

**Run.** Commit `dab4d46`, recorded reproducible with no modified or untracked
code. Seven grippers x five objects x six conditions (three pick poses x two
destinations) = **210 cells**, every one executed end to end by the fitted
policy under Cartesian impedance control. `outputs/campaigns/realworld/real/`.

**Result: 56 placed, 154 failed, 0 refused.**

Every one of the 154 failures has been traced, at every control step, and
attributed to a cause. The traces come from re-running all 210 cells with a
fuller probe; the re-runs chose **the same grasp as the campaign in 210 of 210
cells**, so they describe the same tasks that were scored.

---

## 1. What is new about this scene, and why it was built

Every campaign before this one ran in a scene whose objects and furniture were
smaller than the things they represent, while the gripper was full size.
Measured against the retail articles:

| object | benchmark | real | mass, benchmark -> real |
|---|---|---|---|
| can | 52 x 52 x 80 mm | 66 x 66 x 123 mm | **16 g -> 345 g** |
| cereal | 100 x 30 x 150 | 190 x 80 x 300 | 68 g -> 500 g |
| milk | 40 x 40 x 144 | 70 x 70 x 230 | 21 g -> 1030 g |
| hammer | 127 x 41 x 231 | 117 x 38 x 333 | 66 g -> 700 g |
| mug | 78 dia x 100 | 90 dia x 95 | 300 g -> 350 g |

The mass was further out than the size, and it is not a detail: robosuite's XML
objects declare densities of 50 to 150 kg/m3 against water's 1000, so a
benchmark can weighs **sixteen grams**. Grip is friction against weight, so a
light object is a systematically easier object.

The size decides which grasps exist at all. A real cereal carton is 80 mm across
its narrowest face, which is **exactly** a Panda's full jaw opening; the
miniature is 30 mm and can be taken almost anywhere.

## 2. The scene was verified first, and one check was missing

`tpgpt.experiments.verify_scene` runs the checks of `ROBOTICS_NOTES` 7.32 before
any campaign. On this scene they all pass:

| property | result |
|---|---|
| interpenetration between an object and the robot at placement | **0.00 mm** |
| object rest pose across all seven grippers | **0.024 mm** (bar: 0.1 mm) |
| worst tilt from the commanded pick pose | **0.57 degrees** |
| reach: pick, place and clearance poses, 7 hands x 5 objects x 3 picks | **525 of 525** |

Two defects were found by those checks and fixed before the run. The **hammer**
stood on the end of its handle, because its asset lays the handle along the
body's `z` and a yaw-only placement therefore stands it upright; at benchmark
size it toppled during the settle and was measured lying down, and at 333 mm and
700 g it topples off the table (resting at z = 0.488 against a table surface at
0.800). And the **home pose** left only 50 mm above a 300 mm carton, which is
enough for the fingertips and not for a hand whose unactuated fingers sag while
the objects settle: the Robotiq 3F moved the cereal **4.79 mm** within 250 steps
of reset where six hands moved it 0.061 mm, with **no interpenetration at all**,
so the check that caught the original 7.32 defect could not have caught this one.

**The check that was missing is whether the object fits the destination.** The
top cubby gives 282 mm of clear depth between the board's front edge and the
back panel. A 333 mm hammer's footprint along that axis is 345 mm at 55 degrees
of yaw, 330 at 90 and 355 at 110. Reconstructing the commanded placed pose of
every cell from the recorded grasp and release poses and the recorded rest
pose, **21 of 42 hammer cells are commanded into a pose that does not fit** --
through the back panel by up to 36.7 mm, the side wall by 27.2. That is a
property of the scene, not of the method, and it takes seconds to compute. It is
now part of `verify_scene`.

## 3. The transportation map is not implicated in a single failure

The transported plan passes **0.0 mm from the planned grasp and 0.0 mm from the
planned release, on all 210 cells**. Not approximately: exactly, and by
construction -- the grasp-pose cube is centred on the grasp point and `phi`
interpolates its keypoints exactly, so the aim is a theorem rather than a
measurement.

`min det(J)` is 0.597 to 0.915 by object with no fold anywhere, and it does not
order the outcomes: the mug at 0.616 places 18 of 42 while the cereal at 0.694
places 9 of 42. That is this project's standing result -- `min det` measures
whether the map is a valid deformation, not whether the task succeeds --
reproduced here on a new scene.

**So every millimetre of error is downstream of the map**, and the analysis
below is about the executor, the grip and the scene.

## 4. The causes, and what each one is

| cause | cells | where it sits |
|---|---|---|
| placed | 56 | -- |
| slipped in the grip and was set down in the wrong place | 27 | grip |
| the attractor ran away from the plan | 26 | executor |
| object knocked away before the jaws closed | 19 | approach |
| gripped but the object never rose | 18 | grip |
| fingers touched but took no purchase | 18 | grip |
| jaws closed early; the hand reached the grasp later | 13 | executor |
| the arm lagged too far behind its attractor | 13 | executor |
| lost the object during the carry | 10 | grip |
| shelf refuses the commanded placed pose | 9 | **scene** |
| jaws closed with nothing between them | 1 | approach |

Grouped: **73 of 154 failures are the grip**, 52 are the executor, 20 are the
approach, and 9 are the scene.

### 4.1 The grip is the largest cause, and it is force, not aperture

Of the 98 cells that never lifted the object, **zero** had a slab of material
wider than the jaw at the grasp point. Aperture is not the constraint. What
separates is how hard the fingers end up squeezing:

| | peak grip force after closing |
|---|---|
| cells that placed | **140 N** (median) |
| cells whose grip failed | **51 N** (median) |

And per hand, the split is stark -- a hand that takes hold registers hundreds of
newtons, one that brushes the object registers tens:

| hand | lifted: median peak force | never rose: median peak force |
|---|---|---|
| xarm | 934 N | 32 N |
| robotiq3f | 386 N | 28 N |
| robotiq3f_dex | 196 N | 29 N |
| robotiq85 | 154 N | 42 N |
| robotiq140 | 95 N | 6 N |
| panda | 79 N | 45 N |
| rethink | 29 N | 38 N |

The two narrowest jaws are where it concentrates: **panda 12 of 30 cells and
rethink 12 of 30** fail on the grip, against 0 of 30 for the Robotiq 2F-140. The
Rethink is the only hand whose lifted and failed forces overlap (29 N against
38 N), which is a hand that never grips firmly at all.

**The counter-intuitive number, and it is the useful one.** Cells whose grip
failed were closing on a *thinner* piece of the object than cells that placed --
the object filled a median **33.9%** of the aperture on grip failures against
**49.4%** on successes. A narrow hand cannot span the body of a real-sized
object, so the only candidates that survive the funnel are ones that catch a
rim, an edge or a corner, and a rim is what the fingers slide off. The failure
is not "the object is too big for the hand" but "the only grasps left for this
hand are marginal ones".

*What would make this false:* if peak force were an artefact of contact
stiffness rather than of grip quality, the same force distribution would appear
on cells that lifted successfully. It does not -- the separation is 51 N against
140 N with the two populations barely overlapping.

### 4.2 Misplacement is slip inside the grip, not the arm stopping short

The 27 cells that carried the object and set it down in the wrong place put it a
median **133 mm** from the slot, almost all of it in `x` (components
-132, -8.8, -1.3 mm). The obvious reading -- and the one `ROBOTICS_NOTES` 7.35
would suggest -- is that the arm stopped short of the shelf. **It did not.**

| | median |
|---|---|
| commanded release x | 0.179 m |
| hand's actual x at release | 0.165 m |
| shortfall | **3.0 mm** |
| deepest the hand ever got into the shelf | 0.185 m (155 mm past the front edge) |

The hand arrives. What moves is the object *relative to the hand*:

| | median slip between first firm contact and release |
|---|---|
| cells that placed | **21.6 mm** |
| cells that carried and misplaced | **77.0 mm** (max 583) |

The vertical component of the release error is -1.3 mm and the drop after
release is 3.1 mm, so the object is set down gently -- in the wrong place. This
is a grip failure that survives long enough to look like a placement failure.

*What would make this false:* if the hand's release pose were wrong rather than
the object's position within the grip, the shortfall would show in the hand's
own trajectory. It is 3.0 mm.

### 4.3 The executor: the attractor runs away when the arm is held up

On 26 cells the integrated attractor leaves the plan and does not come back.
Measured as the largest distance from the attractor to the transported path:

| | median max drift |
|---|---|
| cells that placed | **7.4 mm** |
| cells that failed | **39.6 mm** |

On the 66 cells where the hand was far from the grasp when the jaws shut (median
miss 66 mm), that miss splits into **53 mm of attractor drift** and **36 mm of
arm lag**. Since the plan itself is exact, both halves are the executor.

The mechanism is visible in the traces. The lag gate freezes the clock when the
arm falls behind, and on the worst cells it never recovers: the clock is frozen
**0.0% of steps on cells that placed and 3.2% on cells that failed**, nine cells
are frozen for more than a quarter of their steps and **all nine failed**, and
29 cells never reach phase 0.95 -- **all 29 failed**. On `robotiq3f/mug/P1` the
clock is frozen for 149 of 279 steps and the hand never gets past x = -0.134,
never approaching the shelf at all.

### 4.4 The gripper schedule is inherited, and it is almost always fine

The demonstration closes its jaws at phase 0.250. Across the grid the executed
close phase is 0.247 to 0.251 on every cell but six, and those six -- closing
before phase 0.10 -- all failed. On one, `robotiq3f/mug/P1`, the jaws are
commanded shut at **step 5**, 261 mm from the grasp.

This kills a hypothesis worth recording because it was the obvious one. The
policy executor has **no grasp gate**; the replay driver has one. The expectation
was that an open-loop phase schedule would routinely shut the jaws before the
hand arrived. It does not: of the 66 cells that missed at the close, only 17
ever reached the grasp later at all, so in the other 49 the hand was not merely
*late* -- it never got there.

### 4.5 The approach: the Robotiq 3F sweeps the object away

Nineteen cells have the object displaced more than 20 mm **before the jaws are
commanded shut**. Nine of the nineteen are `robotiq3f` and twelve are the milk
carton. The 3F is the widest-bodied hand in the fleet and the milk is the
tallest narrow object, so the hand meets it on the way down.

## 5. Four instruments that misled, three of them built for this analysis

Recorded because each produced a confident wrong answer, and because the pattern
is the same every time: a measurement that returns a plausible number is worse
than one that returns nothing.

- **A bounding box is not a width.** The first width instrument took each
  collision geom's axis-aligned box, so a 66 mm can measured **90.7 mm** along a
  diagonal and half the fleet looked incapable of grasping it. Ray-casting the
  real geometry and grouping the crossings by geom gives 55-62 mm for the can,
  80 for the cereal, 32 for the hammer's handle and 6 for the mug's wall.
- **A default is not a measurement.** The forensic trace showed the jaw closure
  channel reading 0.00 at every step of every cell, including one that lifted
  307 mm at 946 N. The channel was never recorded -- `pipeline.run` builds the
  object probe without the gripper argument -- and the reader's default filled
  it with zeros. Driven directly, every hand closes properly (closure 0 to 1.0,
  joint travel 0.04 to 6.8 rad).
- **`first_failure_stage` blames `reach` on 106 of 154 failures**, because the
  closing budget is `0.25 x (aperture - object width) / 2` and that is about
  **0.6 mm** for an xarm on a real cereal carton. The proxy is over-strict on
  real-sized objects, so the stage attribution is unusable as it stands. This
  whole analysis re-derives causes from traces for that reason. Both
  `m_reach_closing` and `m_closing_budget` are recorded per cell, so it can be
  re-derived without another campaign.
- **Grasp depth below the object's top is not the cause**, although it looked
  like one. Every grasp does sit near the top -- a median 11.3 mm below the top
  surface, about 5% of the object's height -- but it does not predict outcome:
  the correlation with lift is **r = -0.13**, and 30 of the 54 cells grasped
  *above* the rim still lift.

## 6. What to fix, in the order the evidence supports

1. **The grip.** 73 of 154 failures, and the mechanism is measured: narrow hands
   are left with marginal grasps because a central grasp no longer fits, and
   marginal grasps do not generate force. This is a *selection* problem as much
   as a hardware one -- the funnel should prefer grasps whose slab fills more of
   the jaw, which is a ranking term it does not currently have.
2. **The executor's attractor.** 52 failures, with a clean separator (7.4 mm of
   drift on successes against 39.6 mm on failures) and a known mechanism in the
   lag gate freezing the clock. `ROBOTICS_NOTES` 7.40 chose the shipped
   integrator on a surrogate plant where the laws differed by less than the
   seed-to-seed noise; on this scene they would not.
3. **The shelf depth.** 9 cells, all hammer, and the cheapest fix of the three:
   the top cubby needs to be deeper than the longest object, or the object set
   needs a shorter hammer.
4. **The approach clearance for wide-bodied hands.** 19 cells.

## 7. Per-cell table

Columns: **miss at close** is the distance from the hand to the planned grasp
when the jaws were commanded shut; **attr drift** and **arm lag** split that miss
into the attractor leaving the plan and the arm trailing the attractor;
**peak N** is the largest grip force after closing; **hold** is the fraction of
steps from closing to release in contact; **slip** is how far the object moved
relative to the hand between first firm contact and release; **final err** is the
horizontal distance from the object's resting place to the slot centre.

| gripper | object | pick | dest | outcome | cause | miss at close mm | attr drift mm | arm lag mm | peak N | lift mm | hold | slip mm | final err mm | min det |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| panda | can | P0 | left | fail | no purchase | 23 | 2 | 23 | 49 | 6 | 0.29 | 481 | 446 | 0.864 |
| panda | can | P0 | middle | fail | no purchase | 24 | 4 | 20 | 24 | 3 | 0.09 | 422 | 352 | 0.951 |
| panda | can | P1 | left | fail | slipped in the grip | 18 | 2 | 19 | 116 | 245 | 0.72 | 355 | 264 | 0.976 |
| panda | can | P1 | middle | OK | placed | 16 | 1 | 16 | 74 | 342 | 0.98 | 22 | 22 | 0.832 |
| panda | can | P2 | left | fail | no lift | 26 | 12 | 28 | 76 | 4 | 0.15 | 529 | 456 | 0.956 |
| panda | can | P2 | middle | fail | no lift | 22 | 8 | 23 | 74 | 4 | 0.16 | 534 | 484 | 0.970 |
| panda | cereal | P0 | left | fail | attractor ran away | 72 | 68 | 6 | 0 | 0 | 0.00 | - | 446 | 0.609 |
| panda | cereal | P0 | middle | fail | no purchase | 8 | 3 | 8 | 34 | 2 | 0.11 | 495 | 353 | 0.700 |
| panda | cereal | P1 | left | fail | no lift | 24 | 5 | 23 | 58 | 11 | 0.13 | 561 | 414 | 0.295 |
| panda | cereal | P1 | middle | fail | no lift | 23 | 30 | 45 | 55 | 4 | 0.09 | 344 | 390 | 0.859 |
| panda | cereal | P2 | left | fail | no purchase | 27 | 3 | 25 | 40 | 5 | 0.15 | 522 | 460 | 0.754 |
| panda | cereal | P2 | middle | fail | attractor ran away | 72 | 68 | 8 | 45 | 0 | 0.02 | 181 | 487 | 0.321 |
| panda | hammer | P0 | left | fail | dropped in carry | 21 | 2 | 19 | 39 | 106 | 0.37 | 538 | 314 | 0.387 |
| panda | hammer | P0 | middle | fail | slipped in the grip | 10 | 0 | 10 | 82 | 240 | 0.96 | 63 | 136 | 0.981 |
| panda | hammer | P1 | left | fail | knocked away | 86 | 80 | 15 | 45 | 9 | 0.01 | 390 | 436 | 0.319 |
| panda | hammer | P1 | middle | fail | no purchase | 23 | 43 | 36 | 48 | 1 | 0.14 | 367 | 391 | 0.376 |
| panda | hammer | P2 | left | fail | closed early | 50 | 22 | 34 | 168 | 230 | 0.97 | 92 | 207 | 0.937 |
| panda | hammer | P2 | middle | fail | slipped in the grip | 30 | 2 | 31 | 148 | 264 | 0.94 | 98 | 102 | 0.979 |
| panda | milk | P0 | left | fail | attractor ran away | 158 | 118 | 51 | 0 | 0 | 0.00 | - | 445 | 0.614 |
| panda | milk | P0 | middle | fail | attractor ran away | 123 | 101 | 36 | 0 | 0 | 0.00 | - | 353 | 0.617 |
| panda | milk | P1 | left | fail | attractor ran away | 103 | 102 | 9 | 0 | 0 | 0.00 | - | 414 | 0.459 |
| panda | milk | P1 | middle | fail | no lift | 10 | 19 | 12 | 185 | 6 | 0.18 | 510 | 382 | 0.664 |
| panda | milk | P2 | left | fail | no lift | 27 | 14 | 30 | 167 | 4 | 0.27 | 482 | 446 | 0.937 |
| panda | milk | P2 | middle | fail | attractor ran away | 130 | 100 | 36 | 0 | 0 | 0.00 | - | 487 | 0.886 |
| panda | mug | P0 | left | OK | placed | 26 | 3 | 26 | 32 | 336 | 0.96 | 22 | 11 | 0.898 |
| panda | mug | P0 | middle | OK | placed | 14 | 2 | 15 | 54 | 361 | 0.97 | 18 | 8 | 0.804 |
| panda | mug | P1 | left | OK | placed | 6 | 13 | 9 | 75 | 316 | 0.97 | 94 | 16 | 0.509 |
| panda | mug | P1 | middle | OK | placed | 14 | 3 | 12 | 113 | 309 | 1.00 | 47 | 8 | 0.748 |
| panda | mug | P2 | left | fail | attractor ran away | 97 | 61 | 40 | 78 | 7 | 0.03 | 33 | 459 | 0.949 |
| panda | mug | P2 | middle | fail | no purchase | 18 | 16 | 29 | 22 | 6 | 0.36 | 476 | 486 | 0.849 |
| rethink | can | P0 | left | fail | no purchase | 17 | 3 | 18 | 41 | 7 | 0.25 | 501 | 445 | 0.910 |
| rethink | can | P0 | middle | OK | placed | 14 | 1 | 15 | 60 | 339 | 1.00 | 25 | 19 | 0.983 |
| rethink | can | P1 | left | fail | closed early | 42 | 19 | 48 | 65 | 4 | 0.17 | 377 | 415 | 0.612 |
| rethink | can | P1 | middle | fail | no purchase | 16 | 2 | 17 | 13 | 6 | 0.04 | 449 | 391 | 0.916 |
| rethink | can | P2 | left | fail | no purchase | 20 | 9 | 29 | 49 | 5 | 0.33 | 487 | 460 | 0.954 |
| rethink | can | P2 | middle | fail | attractor ran away | 63 | 52 | 24 | 86 | 2 | 0.09 | 437 | 487 | 0.957 |
| rethink | cereal | P0 | left | fail | no purchase | 12 | 6 | 7 | 12 | 2 | 0.06 | 532 | 446 | 0.631 |
| rethink | cereal | P0 | middle | fail | no lift | 11 | 3 | 9 | 53 | 4 | 0.11 | 464 | 353 | 0.742 |
| rethink | cereal | P1 | left | fail | closed early | 52 | 61 | 34 | 29 | 4 | 0.07 | 530 | 418 | 0.709 |
| rethink | cereal | P1 | middle | fail | closed on air | 24 | 55 | 36 | 22 | 7 | 0.04 | 421 | 390 | 0.528 |
| rethink | cereal | P2 | left | fail | no purchase | 17 | 9 | 19 | 38 | 8 | 0.14 | 523 | 459 | 0.696 |
| rethink | cereal | P2 | middle | fail | no lift | 17 | 15 | 22 | 53 | 5 | 0.24 | 521 | 483 | 0.569 |
| rethink | hammer | P0 | left | fail | dropped in carry | 29 | 3 | 28 | 50 | 92 | 0.40 | 493 | 441 | 0.780 |
| rethink | hammer | P0 | middle | fail | attractor ran away | 72 | 85 | 36 | 0 | 0 | 0.00 | - | 353 | 0.875 |
| rethink | hammer | P1 | left | fail | no purchase | 20 | 2 | 19 | 34 | 0 | 0.25 | 486 | 413 | 0.780 |
| rethink | hammer | P1 | middle | fail | no purchase | 22 | 2 | 22 | 38 | 0 | 0.27 | 492 | 390 | 0.630 |
| rethink | hammer | P2 | left | fail | closed early | 46 | 12 | 35 | 204 | 0 | 0.16 | 430 | 460 | 0.928 |
| rethink | hammer | P2 | middle | fail | no lift | 32 | 4 | 35 | 134 | 16 | 0.22 | 516 | 508 | 0.969 |
| rethink | milk | P0 | left | fail | closed early | 66 | 61 | 19 | 0 | 0 | 0.00 | - | 446 | 0.247 |
| rethink | milk | P0 | middle | fail | no lift | 20 | 3 | 20 | 63 | 4 | 0.14 | 391 | 353 | 0.590 |
| rethink | milk | P1 | left | fail | attractor ran away | 103 | 87 | 36 | 0 | 0 | 0.00 | - | 414 | 0.696 |
| rethink | milk | P1 | middle | fail | attractor ran away | 54 | 75 | 35 | 0 | 4 | 0.00 | - | 391 | 0.485 |
| rethink | milk | P2 | left | fail | knocked away | 23 | 25 | 26 | 0 | 7 | 0.00 | - | 522 | 0.993 |
| rethink | milk | P2 | middle | fail | closed early | 54 | 49 | 32 | 43 | 7 | 0.14 | 542 | 485 | 0.979 |
| rethink | mug | P0 | left | fail | slipped in the grip | 18 | 2 | 18 | 29 | 321 | 0.77 | 77 | 83 | 0.548 |
| rethink | mug | P0 | middle | fail | closed early | 42 | 53 | 54 | 53 | 5 | 0.21 | 490 | 354 | 0.472 |
| rethink | mug | P1 | left | fail | slipped in the grip | 15 | 1 | 16 | 29 | 337 | 0.78 | 66 | 73 | 0.738 |
| rethink | mug | P1 | middle | OK | placed | 13 | 2 | 13 | 16 | 342 | 0.96 | 41 | 13 | 0.514 |
| rethink | mug | P2 | left | fail | closed early | 101 | 103 | 35 | 9 | 179 | 0.22 | 444 | 321 | 0.780 |
| rethink | mug | P2 | middle | fail | no purchase | 38 | 26 | 16 | 15 | 6 | 0.10 | 543 | 498 | 0.628 |
| robotiq140 | can | P0 | left | OK | placed | 18 | 1 | 18 | 90 | 330 | 0.97 | 22 | 3 | 0.921 |
| robotiq140 | can | P0 | middle | OK | placed | 24 | 1 | 24 | 82 | 336 | 0.98 | 29 | 2 | 0.940 |
| robotiq140 | can | P1 | left | OK | placed | 14 | 3 | 12 | 78 | 342 | 0.98 | 8 | 16 | 0.988 |
| robotiq140 | can | P1 | middle | OK | placed | 21 | 3 | 19 | 78 | 348 | 0.98 | 13 | 8 | 0.892 |
| robotiq140 | can | P2 | left | OK | placed | 18 | 1 | 17 | 92 | 334 | 0.97 | 10 | 8 | 0.941 |
| robotiq140 | can | P2 | middle | OK | placed | 16 | 1 | 16 | 85 | 336 | 0.98 | 16 | 37 | 0.956 |
| robotiq140 | cereal | P0 | left | OK | placed | 24 | 9 | 17 | 139 | 340 | 0.98 | 32 | 4 | 0.562 |
| robotiq140 | cereal | P0 | middle | fail | arm lagged | 70 | 12 | 58 | 112 | 285 | 0.99 | 170 | 342 | 0.394 |
| robotiq140 | cereal | P1 | left | OK | placed | 32 | 2 | 31 | 142 | 339 | 0.98 | 11 | 18 | 0.967 |
| robotiq140 | cereal | P1 | middle | fail | arm lagged | 75 | 20 | 56 | 64 | 4 | 0.29 | 378 | 390 | 0.974 |
| robotiq140 | cereal | P2 | left | OK | placed | 27 | 9 | 20 | 150 | 325 | 0.99 | 17 | 27 | 0.693 |
| robotiq140 | cereal | P2 | middle | OK | placed | 3 | 23 | 22 | 158 | 321 | 0.99 | 21 | 52 | 0.533 |
| robotiq140 | hammer | P0 | left | fail | shelf refuses | 18 | 7 | 12 | 40 | 136 | 0.72 | 250 | 245 | 0.905 |
| robotiq140 | hammer | P0 | middle | fail | slipped in the grip | 15 | 2 | 17 | 33 | 180 | 0.96 | 117 | 174 | 0.967 |
| robotiq140 | hammer | P1 | left | fail | attractor ran away | 147 | 127 | 22 | 0 | 0 | 0.00 | - | 414 | 0.716 |
| robotiq140 | hammer | P1 | middle | fail | slipped in the grip | 15 | 31 | 20 | 47 | 233 | 0.97 | 10 | 199 | 0.528 |
| robotiq140 | hammer | P2 | left | fail | attractor ran away | 66 | 67 | 4 | 6 | 0 | 0.01 | 551 | 460 | 0.992 |
| robotiq140 | hammer | P2 | middle | fail | slipped in the grip | 37 | 10 | 38 | 94 | 194 | 0.99 | 202 | 183 | 0.945 |
| robotiq140 | milk | P0 | left | fail | attractor ran away | 84 | 78 | 34 | 27 | 6 | 0.21 | 434 | 447 | 0.231 |
| robotiq140 | milk | P0 | middle | OK | placed | 17 | 9 | 12 | 61 | 302 | 0.99 | 26 | 6 | 0.929 |
| robotiq140 | milk | P1 | left | OK | placed | 33 | 14 | 25 | 88 | 303 | 0.98 | 56 | 46 | 0.426 |
| robotiq140 | milk | P1 | middle | OK | placed | 60 | 35 | 28 | 74 | 303 | 0.98 | 35 | 13 | 0.227 |
| robotiq140 | milk | P2 | left | OK | placed | 21 | 2 | 20 | 118 | 321 | 0.98 | 23 | 14 | 0.891 |
| robotiq140 | milk | P2 | middle | fail | knocked away | 28 | 11 | 21 | 0 | 2 | 0.00 | - | 545 | 0.373 |
| robotiq140 | mug | P0 | left | OK | placed | 14 | 3 | 16 | 104 | 344 | 0.99 | 19 | 21 | 0.722 |
| robotiq140 | mug | P0 | middle | OK | placed | 9 | 5 | 11 | 98 | 327 | 0.99 | 12 | 6 | 0.829 |
| robotiq140 | mug | P1 | left | OK | placed | 18 | 1 | 17 | 148 | 348 | 0.99 | 15 | 4 | 0.686 |
| robotiq140 | mug | P1 | middle | OK | placed | 23 | 4 | 20 | 126 | 342 | 0.99 | 12 | 21 | 0.491 |
| robotiq140 | mug | P2 | left | OK | placed | 20 | 5 | 15 | 124 | 340 | 0.99 | 16 | 35 | 0.912 |
| robotiq140 | mug | P2 | middle | fail | slipped in the grip | 28 | 2 | 27 | 136 | 331 | 1.00 | 25 | 67 | 0.818 |
| robotiq3f | can | P0 | left | fail | arm lagged | 71 | 12 | 64 | 101 | 6 | 0.22 | 544 | 474 | 0.866 |
| robotiq3f | can | P0 | middle | fail | slipped in the grip | 16 | 1 | 16 | 396 | 249 | 0.99 | 12 | 109 | 0.911 |
| robotiq3f | can | P1 | left | fail | arm lagged | 63 | 5 | 58 | 58 | 7 | 0.28 | 346 | 340 | 0.680 |
| robotiq3f | can | P1 | middle | OK | placed | 24 | 1 | 22 | 338 | 347 | 0.97 | 27 | 24 | 0.846 |
| robotiq3f | can | P2 | left | OK | placed | 26 | 3 | 23 | 420 | 340 | 0.98 | 26 | 23 | 0.953 |
| robotiq3f | can | P2 | middle | OK | placed | 26 | 2 | 24 | 410 | 333 | 0.98 | 12 | 16 | 0.833 |
| robotiq3f | cereal | P0 | left | fail | knocked away | 88 | 24 | 64 | 54 | 5 | 0.20 | 408 | 445 | 0.683 |
| robotiq3f | cereal | P0 | middle | fail | knocked away | 91 | 44 | 48 | 36 | 6 | 0.22 | 342 | 353 | 0.717 |
| robotiq3f | cereal | P1 | left | fail | dropped in carry | 25 | 3 | 23 | 510 | 169 | 0.53 | 502 | 388 | 0.979 |
| robotiq3f | cereal | P1 | middle | fail | dropped in carry | 24 | 2 | 23 | 545 | 171 | 0.50 | 486 | 372 | 0.977 |
| robotiq3f | cereal | P2 | left | fail | arm lagged | 88 | 24 | 69 | 60 | 3 | 0.26 | 466 | 460 | 0.749 |
| robotiq3f | cereal | P2 | middle | fail | knocked away | 38 | 16 | 23 | 0 | 4 | 0.00 | - | 572 | 0.466 |
| robotiq3f | hammer | P0 | left | fail | shelf refuses | 31 | 4 | 29 | 43 | 177 | 0.71 | 468 | 427 | 0.845 |
| robotiq3f | hammer | P0 | middle | fail | slipped in the grip | 35 | 6 | 29 | 49 | 177 | 0.69 | 269 | 207 | 0.935 |
| robotiq3f | hammer | P1 | left | fail | attractor ran away | 90 | 77 | 14 | 81 | 0 | 0.18 | 443 | 414 | 0.305 |
| robotiq3f | hammer | P1 | middle | fail | knocked away | 212 | 181 | 37 | 0 | 0 | 0.00 | - | 372 | 0.439 |
| robotiq3f | hammer | P2 | left | fail | arm lagged | 56 | 8 | 48 | 56 | 206 | 0.63 | 245 | 187 | 0.896 |
| robotiq3f | hammer | P2 | middle | fail | arm lagged | 55 | 5 | 50 | 130 | 224 | 0.97 | 45 | 121 | 0.914 |
| robotiq3f | milk | P0 | left | fail | knocked away | 62 | 61 | 21 | 0 | 11 | 0.00 | - | 433 | 0.106 |
| robotiq3f | milk | P0 | middle | fail | knocked away | 54 | 36 | 21 | 0 | 8 | 0.00 | - | 321 | 0.210 |
| robotiq3f | milk | P1 | left | fail | knocked away | 26 | 2 | 24 | 0 | 6 | 0.00 | - | 313 | 0.336 |
| robotiq3f | milk | P1 | middle | fail | knocked away | 25 | 1 | 26 | 0 | 6 | 0.00 | - | 338 | 0.582 |
| robotiq3f | milk | P2 | left | fail | slipped in the grip | 19 | 13 | 16 | 647 | 310 | 0.99 | 40 | 83 | 0.603 |
| robotiq3f | milk | P2 | middle | fail | knocked away | 36 | 11 | 35 | 19 | 3 | 0.02 | 523 | 526 | 0.373 |
| robotiq3f | mug | P0 | left | fail | arm lagged | 61 | 13 | 51 | 315 | 4 | 0.25 | 424 | 438 | 0.279 |
| robotiq3f | mug | P0 | middle | fail | closed early | 43 | 7 | 36 | 478 | 300 | 0.99 | 19 | 283 | 0.354 |
| robotiq3f | mug | P1 | left | fail | attractor ran away | 261 | 234 | 36 | 0 | 0 | 0.00 | - | 414 | 0.497 |
| robotiq3f | mug | P1 | middle | fail | arm lagged | 64 | 16 | 48 | 335 | 9 | 0.25 | 392 | 387 | 0.923 |
| robotiq3f | mug | P2 | left | fail | dropped in carry | 34 | 15 | 22 | 377 | 91 | 0.39 | 534 | 509 | 0.577 |
| robotiq3f | mug | P2 | middle | OK | placed | 61 | 1 | 60 | 126 | 256 | 1.00 | 35 | 52 | 0.380 |
| robotiq3f_dex | can | P0 | left | fail | slipped in the grip | 28 | 5 | 28 | 271 | 211 | 0.99 | 19 | 94 | 0.866 |
| robotiq3f_dex | can | P0 | middle | OK | placed | 13 | 1 | 13 | 293 | 350 | 0.97 | 12 | 2 | 0.948 |
| robotiq3f_dex | can | P1 | left | fail | slipped in the grip | 20 | 1 | 19 | 190 | 360 | 0.98 | 20 | 77 | 0.680 |
| robotiq3f_dex | can | P1 | middle | OK | placed | 28 | 2 | 26 | 105 | 346 | 0.98 | 11 | 39 | 0.846 |
| robotiq3f_dex | can | P2 | left | OK | placed | 29 | 2 | 27 | 249 | 346 | 0.97 | 22 | 8 | 0.882 |
| robotiq3f_dex | can | P2 | middle | OK | placed | 25 | 4 | 28 | 284 | 335 | 0.97 | 26 | 7 | 0.762 |
| robotiq3f_dex | cereal | P0 | left | fail | knocked away | 65 | 5 | 60 | 65 | 4 | 0.24 | 437 | 445 | 0.683 |
| robotiq3f_dex | cereal | P0 | middle | fail | arm lagged | 67 | 19 | 48 | 42 | 5 | 0.18 | 354 | 353 | 0.717 |
| robotiq3f_dex | cereal | P1 | left | fail | no lift | 15 | 2 | 16 | 177 | 11 | 0.18 | 476 | 428 | 0.988 |
| robotiq3f_dex | cereal | P1 | middle | fail | no lift | 15 | 1 | 15 | 138 | 12 | 0.18 | 474 | 432 | 0.986 |
| robotiq3f_dex | cereal | P2 | left | fail | dropped in carry | 25 | 5 | 21 | 452 | 52 | 0.22 | 541 | 463 | 0.796 |
| robotiq3f_dex | cereal | P2 | middle | fail | dropped in carry | 10 | 16 | 25 | 508 | 106 | 0.31 | 685 | 596 | 0.644 |
| robotiq3f_dex | hammer | P0 | left | fail | shelf refuses | 27 | 5 | 24 | 44 | 165 | 0.66 | 460 | 409 | 0.842 |
| robotiq3f_dex | hammer | P0 | middle | fail | slipped in the grip | 18 | 2 | 18 | 36 | 334 | 0.98 | 17 | 168 | 0.971 |
| robotiq3f_dex | hammer | P1 | left | fail | no purchase | 28 | 32 | 13 | 16 | 4 | 0.03 | 315 | 400 | 0.463 |
| robotiq3f_dex | hammer | P1 | middle | fail | slipped in the grip | 27 | 3 | 27 | 32 | 225 | 0.76 | 218 | 236 | 0.228 |
| robotiq3f_dex | hammer | P2 | left | fail | shelf refuses | 26 | 1 | 25 | 41 | 271 | 0.82 | 500 | 417 | 0.987 |
| robotiq3f_dex | hammer | P2 | middle | fail | dropped in carry | 33 | 3 | 30 | 41 | 50 | 0.57 | 589 | 532 | 0.982 |
| robotiq3f_dex | milk | P0 | left | fail | dropped in carry | 37 | 46 | 29 | 362 | 94 | 0.54 | 220 | 336 | 0.106 |
| robotiq3f_dex | milk | P0 | middle | fail | knocked away | 53 | 37 | 24 | 0 | 9 | 0.00 | - | 427 | 0.210 |
| robotiq3f_dex | milk | P1 | left | fail | attractor ran away | 215 | 183 | 35 | 0 | 0 | 0.00 | - | 414 | 0.003 |
| robotiq3f_dex | milk | P1 | middle | OK | placed | 6 | 3 | 8 | 159 | 301 | 0.98 | 10 | 18 | 0.903 |
| robotiq3f_dex | milk | P2 | left | fail | attractor ran away | 88 | 87 | 13 | 209 | 6 | 0.14 | 371 | 529 | 0.131 |
| robotiq3f_dex | milk | P2 | middle | fail | knocked away | 26 | 7 | 23 | 0 | 2 | 0.00 | - | 545 | 0.217 |
| robotiq3f_dex | mug | P0 | left | fail | arm lagged | 57 | 28 | 30 | 7 | 4 | 0.10 | 385 | 445 | 0.279 |
| robotiq3f_dex | mug | P0 | middle | fail | dropped in carry | 6 | 8 | 3 | 364 | 136 | 0.43 | 459 | 445 | 0.354 |
| robotiq3f_dex | mug | P1 | left | fail | attractor ran away | 251 | 240 | 36 | 0 | 0 | 0.00 | - | 414 | 0.650 |
| robotiq3f_dex | mug | P1 | middle | fail | no lift | 39 | 4 | 36 | 240 | 6 | 0.32 | 417 | 385 | 0.923 |
| robotiq3f_dex | mug | P2 | left | fail | closed early | 42 | 23 | 20 | 202 | 52 | 0.33 | 494 | 443 | 0.577 |
| robotiq3f_dex | mug | P2 | middle | fail | closed early | 41 | 3 | 43 | 102 | 28 | 0.33 | 474 | 475 | 0.380 |
| robotiq85 | can | P0 | left | OK | placed | 23 | 4 | 22 | 165 | 333 | 0.97 | 22 | 23 | 0.810 |
| robotiq85 | can | P0 | middle | OK | placed | 32 | 26 | 16 | 141 | 305 | 0.97 | 12 | 36 | 0.920 |
| robotiq85 | can | P1 | left | fail | no lift | 17 | 17 | 25 | 65 | 6 | 0.03 | 505 | 481 | 0.845 |
| robotiq85 | can | P1 | middle | OK | placed | 17 | 1 | 16 | 190 | 356 | 0.97 | 5 | 12 | 0.915 |
| robotiq85 | can | P2 | left | OK | placed | 16 | 1 | 15 | 176 | 347 | 0.98 | 10 | 26 | 0.932 |
| robotiq85 | can | P2 | middle | OK | placed | 19 | 2 | 20 | 154 | 356 | 0.98 | 37 | 10 | 0.976 |
| robotiq85 | cereal | P0 | left | OK | placed | 28 | 4 | 24 | 299 | 310 | 0.99 | 55 | 4 | 0.662 |
| robotiq85 | cereal | P0 | middle | fail | arm lagged | 46 | 3 | 43 | 50 | 4 | 0.21 | 378 | 353 | 0.653 |
| robotiq85 | cereal | P1 | left | OK | placed | 24 | 2 | 22 | 156 | 308 | 0.99 | 21 | 15 | 0.924 |
| robotiq85 | cereal | P1 | middle | OK | placed | 22 | 2 | 22 | 138 | 322 | 0.99 | 100 | 26 | 0.863 |
| robotiq85 | cereal | P2 | left | fail | slipped in the grip | 19 | 3 | 17 | 231 | 309 | 0.87 | 583 | 458 | 0.718 |
| robotiq85 | cereal | P2 | middle | OK | placed | 24 | 3 | 22 | 241 | 319 | 0.99 | 33 | 13 | 0.548 |
| robotiq85 | hammer | P0 | left | fail | shelf refuses | 21 | 15 | 7 | 68 | 223 | 0.98 | 69 | 175 | 0.952 |
| robotiq85 | hammer | P0 | middle | fail | slipped in the grip | 12 | 8 | 7 | 62 | 168 | 0.92 | 134 | 194 | 0.958 |
| robotiq85 | hammer | P1 | left | fail | shelf refuses | 32 | 1 | 32 | 83 | 253 | 0.97 | 23 | 389 | 0.855 |
| robotiq85 | hammer | P1 | middle | fail | attractor ran away | 93 | 60 | 36 | 49 | 4 | 0.24 | 174 | 397 | 0.472 |
| robotiq85 | hammer | P2 | left | fail | shelf refuses | 11 | 2 | 11 | 255 | 165 | 0.90 | 359 | 365 | 0.988 |
| robotiq85 | hammer | P2 | middle | fail | slipped in the grip | 11 | 2 | 10 | 57 | 174 | 0.84 | 220 | 215 | 0.870 |
| robotiq85 | milk | P0 | left | fail | attractor ran away | 183 | 168 | 36 | 0 | 0 | 0.00 | - | 445 | 0.783 |
| robotiq85 | milk | P0 | middle | fail | closed early | 60 | 54 | 7 | 42 | 4 | 0.07 | 394 | 353 | 0.856 |
| robotiq85 | milk | P1 | left | fail | knocked away | 177 | 159 | 36 | 0 | 0 | 0.00 | - | 511 | 0.840 |
| robotiq85 | milk | P1 | middle | fail | attractor ran away | 225 | 196 | 36 | 0 | 0 | 0.00 | - | 391 | 0.112 |
| robotiq85 | milk | P2 | left | fail | no lift | 13 | 3 | 15 | 83 | 4 | 0.15 | 475 | 460 | 0.773 |
| robotiq85 | milk | P2 | middle | fail | knocked away | 65 | 42 | 29 | 0 | 7 | 0.00 | - | 389 | 0.618 |
| robotiq85 | mug | P0 | left | fail | slipped in the grip | 27 | 19 | 15 | 47 | 276 | 0.64 | 257 | 67 | 0.214 |
| robotiq85 | mug | P0 | middle | OK | placed | 14 | 2 | 14 | 119 | 338 | 0.99 | 21 | 12 | 0.887 |
| robotiq85 | mug | P1 | left | OK | placed | 21 | 4 | 17 | 126 | 307 | 1.00 | 21 | 57 | 0.866 |
| robotiq85 | mug | P1 | middle | OK | placed | 10 | 2 | 12 | 156 | 310 | 1.00 | 15 | 27 | 0.673 |
| robotiq85 | mug | P2 | left | fail | attractor ran away | 236 | 204 | 36 | 0 | 0 | 0.00 | - | 460 | 0.605 |
| robotiq85 | mug | P2 | middle | fail | knocked away | 49 | 52 | 10 | 51 | 0 | 0.03 | 954 | 769 | 0.577 |
| xarm | can | P0 | left | OK | placed | 17 | 1 | 17 | 1006 | 359 | 0.97 | 5 | 11 | 0.917 |
| xarm | can | P0 | middle | OK | placed | 17 | 2 | 17 | 998 | 362 | 0.97 | 11 | 7 | 0.975 |
| xarm | can | P1 | left | OK | placed | 23 | 1 | 22 | 1027 | 338 | 0.97 | 33 | 35 | 0.728 |
| xarm | can | P1 | middle | OK | placed | 20 | 2 | 21 | 1012 | 346 | 0.98 | 8 | 6 | 0.602 |
| xarm | can | P2 | left | fail | no lift | 25 | 3 | 25 | 242 | 6 | 0.27 | 401 | 460 | 0.877 |
| xarm | can | P2 | middle | fail | slipped in the grip | 21 | 1 | 21 | 1048 | 95 | 0.99 | 8 | 557 | 0.987 |
| xarm | cereal | P0 | left | fail | attractor ran away | 50 | 45 | 21 | 1098 | 160 | 1.00 | 140 | 430 | 0.074 |
| xarm | cereal | P0 | middle | OK | placed | 19 | 3 | 17 | 1097 | 342 | 0.99 | 45 | 5 | 0.730 |
| xarm | cereal | P1 | left | fail | no lift | 22 | 2 | 19 | 247 | 5 | 0.10 | 149 | 415 | 0.787 |
| xarm | cereal | P1 | middle | fail | no lift | 22 | 1 | 21 | 246 | 5 | 0.09 | 176 | 391 | 0.644 |
| xarm | cereal | P2 | left | fail | no purchase | 17 | 3 | 17 | 29 | 17 | 0.03 | 507 | 460 | 0.557 |
| xarm | cereal | P2 | middle | fail | no purchase | 18 | 3 | 16 | 15 | 16 | 0.03 | 607 | 487 | 0.379 |
| xarm | hammer | P0 | left | fail | shelf refuses | 12 | 3 | 12 | 593 | 274 | 0.96 | 15 | 60 | 0.931 |
| xarm | hammer | P0 | middle | fail | slipped in the grip | 9 | 3 | 6 | 574 | 212 | 0.98 | 24 | 179 | 0.991 |
| xarm | hammer | P1 | left | fail | shelf refuses | 8 | 2 | 8 | 664 | 247 | 0.97 | 18 | 404 | 0.533 |
| xarm | hammer | P1 | middle | fail | slipped in the grip | 19 | 0 | 18 | 483 | 178 | 0.97 | 39 | 173 | 0.698 |
| xarm | hammer | P2 | left | fail | attractor ran away | 78 | 82 | 12 | 0 | 0 | 0.00 | - | 460 | 0.866 |
| xarm | hammer | P2 | middle | fail | closed early | 46 | 31 | 24 | 557 | 290 | 0.96 | 59 | 132 | 0.946 |
| xarm | milk | P0 | left | fail | slipped in the grip | 25 | 5 | 21 | 933 | 181 | 0.61 | 448 | 374 | 0.903 |
| xarm | milk | P0 | middle | fail | slipped in the grip | 20 | 2 | 20 | 946 | 308 | 0.85 | 137 | 124 | 0.967 |
| xarm | milk | P1 | left | fail | knocked away | 59 | 46 | 32 | 0 | 6 | 0.00 | - | 354 | 0.027 |
| xarm | milk | P1 | middle | fail | attractor ran away | 114 | 91 | 36 | 0 | 8 | 0.00 | - | 292 | 0.153 |
| xarm | milk | P2 | left | fail | slipped in the grip | 22 | 1 | 22 | 938 | 321 | 0.82 | 63 | 65 | 0.995 |
| xarm | milk | P2 | middle | fail | slipped in the grip | 24 | 2 | 22 | 935 | 253 | 0.70 | 401 | 306 | 0.981 |
| xarm | mug | P0 | left | fail | no purchase | 22 | 24 | 22 | 35 | 10 | 0.06 | 489 | 412 | 0.100 |
| xarm | mug | P0 | middle | OK | placed | 19 | 3 | 19 | 155 | 330 | 0.95 | 43 | 60 | 0.843 |
| xarm | mug | P1 | left | OK | placed | 27 | 3 | 29 | 188 | 329 | 0.98 | 41 | 35 | 0.657 |
| xarm | mug | P1 | middle | fail | arm lagged | 53 | 10 | 51 | 358 | 18 | 0.00 | 124 | 285 | 0.306 |
| xarm | mug | P2 | left | OK | placed | 15 | 2 | 17 | 138 | 363 | 0.95 | 43 | 39 | 0.535 |
| xarm | mug | P2 | middle | OK | placed | 21 | 7 | 18 | 632 | 313 | 0.98 | 79 | 51 | 0.197 |
