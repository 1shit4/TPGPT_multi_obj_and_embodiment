# The YCB grid: 84 of 210, and where the remaining failures sit

**Run.** Commit `a40b3ca`, recorded reproducible with no modified or untracked
code. Seven grippers x five YCB objects x six conditions (three pick poses x two
destinations) = **210 cells**, each executed end to end by the fitted policy
under Cartesian impedance control. `outputs/campaigns/ycb/ycb/`.

**Result: 84 placed, 126 failed, 0 refused.** The same design on the previous
scene -- this project's own sizing of robosuite's benchmark meshes -- placed
**56 of 210**.

The executor was not touched and no filter was added or removed. What changed is
the scene: the objects are the YCB set's own scans at their scanned size and
published mass, their collision geometry is a convex decomposition rather than a
convex hull, the object point cloud is the full scan instead of a one-sided
camera fusion, the top board is deep enough to admit the longest object at every
orientation, and the pick poses clear the robot's own base. Five defects found
while building it are in the commit message for `a40b3ca`.

## 1. Where the gain came from: the objects are now being picked up

The stage each run reached, against the same breakdown on the previous scene:

| phase reached | YCB | previous |
|---|---|---|
| placed | **84** | 56 |
| never lifted (< 20 mm) | **51** | **98** |
| lifted then dropped mid-carry | 17 | 13 |
| carried, then set down in the wrong place | **58** | 43 |

**The number that moved is "never lifted", which halved.** That is the
signature of the change that ought to have caused it: with a one-sided camera
cloud the narrow hands were pushed onto rim and edge grasps, which is what
fingers slide off; with the full scan a central grasp is visible and selectable.
The failures that remain have moved *downstream* -- more runs now get as far as
carrying the object and then put it somewhere wrong, which is a different and
more tractable problem than never having hold of it.

## 2. The gripper spread narrowed, which is the claim the project cares about

| gripper | placed | | gripper | placed |
|---|---|---|---|---|
| `robotiq3f` | 15 / 30 | | `robotiq140` | 13 / 30 |
| `robotiq85` | 14 / 30 | | `xarm` | 12 / 30 |
| `panda` | 13 / 30 | | `robotiq3f_dex` | 11 / 30 |
| `rethink` | 6 / 30 | | | |

The spread is **6 to 15**, against **2 to 19** on the previous scene. The
standing claim is that once a grasp exists, which hand holds the object should
not predict whether the task succeeds; that is considerably more defensible at
this spread than at the old one. The Rethink is still last, and it has the
narrowest jaw in the fleet at 66 mm, so the claim is not yet clean.

**Per object it is not even at all**, and this is where the remaining work is:

| object | placed | note |
|---|---|---|
| `meat` | 32 / 42 | 102 x 83.5 x 60.1 mm, flat-faced, 370 g |
| `banana` | 22 / 42 | curved, light (66 g) |
| `sugar` | 15 / 42 | 176 mm tall, 514 g -- the heaviest |
| `mug` | 12 / 42 | held by a 5 mm wall or a handle, not a silhouette |
| `hammer` | **3 / 42** | 333 mm, 665 g, mass off the grip axis |

**No condition carries the result.** The six conditions score 8, 12, 14, 15, 17
and 18 of 35, so neither a pick pose nor a destination is doing the work.

## 3. The transportation map is exact, again

The transported plan passes **0.000 mm from the planned grasp and 0.000 mm from
the planned release**, median and to a maximum of 0.002 mm, on all 210 cells.
That is the grasp-pose cube construction working as designed: the cube's centre
*is* the grasp point and `phi` interpolates its keypoints exactly, so the aim is
a theorem rather than a measurement.

`min det(J)` is median **0.795**, minimum 0.171, with **no folds anywhere**. It
does separate outcomes here (0.877 on cells that placed against 0.760 on cells
that failed, rank-biserial +0.32), which is weaker than the execution
quantities below and consistent with the project's standing position that it
measures whether the map is a *valid* deformation rather than whether the task
works.

**So no failure in this campaign is a failure of the map**, and the analysis
below is about the executor, the grip and the scene.

## 4. What separates a placed cell from a failed one

Rank-biserial separation, where +1 means the quantity is always larger on cells
that placed:

| quantity | placed | failed | separation |
|---|---|---|---|
| lift height | 337.7 mm | 134.8 mm | **+0.82** |
| largest attractor drift from the plan | 5.6 mm | 42.2 mm | **-0.65** |
| fraction of the carry in contact | 0.977 | 0.560 | **+0.61** |
| reach error at the close | 20.0 mm | 40.1 mm | -0.55 |
| control steps with the clock frozen | 2 | 16.5 | -0.54 |
| orientation error at the release | 0.71 deg | 1.00 deg | -0.43 |
| candidates surviving the funnel | 4 | 4 | -0.06 |

**The executor is again the strongest signal after the lift itself.** The
attractor drifts 5.6 mm from the plan on cells that place and **42.2 mm** on
cells that fail, and the lag gate freezes the clock for 2 steps against 16.5.
Since the plan is exact, every millimetre of that drift is the executor's own.

**How many candidates the funnel left does not separate anything** -- a median
of 4 either way. That is worth stating plainly because funnel strictness has
been a standing suspect, and on this grid it is not distinguishing successes
from failures.

## 5. What is not yet known

**The hammer's 3 of 42 is unexplained.** It is no longer a fit problem: the
board is deep enough that it fits its destination at every yaw, and it settles
at 0.000 mm at all three pick poses. 17 of its 42 cells never lift it. A 33 mm
handle with 665 g of mass off the grip axis is the obvious suspect and it has
**not been measured**, so it is a hypothesis and nothing more.

**The stage attribution is not trustworthy and is not used above.**
`first_failure_stage` blames `reach` on 83 of the 126 failures, because the
closing budget is `0.25 x (aperture - object width) / 2` and that is
sub-millimetre for real-sized objects, so the proxy fires on almost everything
and hides what happened afterwards. The per-cell evidence is recorded
(`m_reach_closing`, `m_closing_budget`), and causes will be re-derived from
per-step traces with `tpgpt.experiments.attribute_failures`, as they were for
the previous grid.

**The funnel's internals are not comparable with the previous grid.** With the
full scan as the object cloud, `cloud_points` is a constant 4096 and the
`visibility` stage -- which previously rejected about 15 of 100 candidates -- is
asking a different question. That is the intended consequence of taking
perception out of grasp selection, not a regression, but it means the stage
tallies cannot be read against the old ones.

**The paired position-control arm has not finished.** Until it does, none of the
executor attribution above is separable from the task by construction; it rests
on within-run correlations only.
