# The YCB grid at commit `606edc7`: the numbers behind the draft's three simulation experiments

**Run.** `outputs/campaigns/ycb_s0_gated/ycb/`, recorded reproducible at
`606edc7` (main), 2026-09-20, on the RTX 4090 box (stock robosuite 1.5.2,
MuJoCo 3.7.0). Seven hands x five YCB objects x three pick poses x two
destinations = **210 cells**, seed 0 (one planner draw; with a fixed pick pose
the physics is deterministic, so a "seed" is a draw), executed end to end by
the fitted policy under Cartesian impedance control with the close and release
gates and the support clamp on (`--close-gate --release-gate`; the clamp is
always on since `606edc7`). Every number below is computed from that
directory's `rows.json` and `cell_*.npz` by the scripts named, so it can be
regenerated. Nothing here is paired against the 84/210 of `docs/ycb_grid.md`
(commit `a40b3ca`): that run drew different candidates on 175 of its 210 cells
and has no rows on this machine.

**Result: 83 placed of 210, Wilson 95 % [0.33, 0.46].** Against 84/210 without
the gates and the clamp, on a different draw: the executor changes of
`ROBOTICS_NOTES.md` 7.43 are a wash on the rate, as they were on the 35-cell
paired comparison, and the failure ledger is what moved.

## 1. Across grippers (draft `tab:matrix`)

`spread open` and `TCP offset` are measured on each robosuite model
(`gripper_frames.json`: `spread_open` is the distance between the finger geom
centres at the open pose, not the inner gap; the offset is the norm of
`contact_offset`, grip_site to the contact centroid). `min det` is the median
over the hand's 30 cells of the minimum determinant of the map Jacobian along
the executed path. `modal failure` is the first failed stage on most of that
hand's failures, by the trace-based attribution of `56dea29`.

| hand | fingers | spread open (mm) | TCP offset (mm) | family | placed | Wilson 95% | error (mm, placed) | min det median | modal failure |
|---|---|---|---|---|---|---|---|---|---|
| panda | 2 | 104 | 3.6 | parallel | 13/30 | [0.27, 0.61] | 26.2 | 0.83 | reach (5) |
| robotiq85 | 2 | 131 | 17.8 | revolute | 13/30 | [0.27, 0.61] | 14.3 | 0.80 | place (7) |
| robotiq140 | 2 | 165 | 30.8 | revolute | 15/30 | [0.33, 0.67] | 25.5 | 0.77 | reach (7) |
| xarm | 2 | 107 | 3.3 | parallel | 13/30 | [0.27, 0.61] | 18.3 | 0.80 | place (8) |
| rethink | 2 | 77 | 5.1 | parallel | 7/30 | [0.12, 0.41] | 28.6 | 0.74 | grasp (11) |
| robotiq3f | 3 | 164 | 22.5 | revolute 3F | 13/30 | [0.27, 0.61] | 17.9 | 0.85 | reach (5) |
| robotiq3f_dex | 3 | 165 | 10.1 | revolute 3F, independent fingers | 9/30 | [0.17, 0.48] | 15.8 | 0.83 | settle (7) |
| **all** | | | | | **83/210** | [0.33, 0.46] | 19.4 | 0.80 | |

**Is the hand a predictor of success?** Logistic regression of success on
gripper, object and destination (`tpgpt.metrics.success_model.factor_effects`,
HC1 covariance -- six conditions is far too few clusters for a cluster-robust
one, see `MIN_CLUSTERS`): gripper Wald chi2 = 8.6 on 6 df, **p = 0.20**; object
chi2 = 33.9 on 4 df, p < 0.001; destination p = 0.62. A plain chi-square on the
7 x 2 gripper table gives 6.5, p = 0.37. The object is a predictor and the hand
is not, at n = 30 per hand. The per-hand spread is 7 to 15 of 30 with the
Rethink -- the narrowest jaw, 77 mm centre to centre -- last, as it was on the
previous grid; that is a real, unexplained weakness of one hand, not a
significant gripper effect.

**Per object:** meat 28/42, banana 26/42, mug 15/42, sugar 10/42, hammer
**4/42**. The hammer is what 7.43 is about: on every hand the commanded
fingertip was at or below the table until the support clamp, and clamped, a
long-fingered hand pinches only the top of a 33 mm handle.

**Where it fails, first stage:** reach 32, grasp 26, approach 20, place 19,
settle 15, carry 8, lift 7. The "reach" count is the trace-based one -- the
hand pushed the object more than 20 mm before closing, or closed more than
40 mm from the grasp -- not the closing-budget proxy that read 83 of 126 on the
previous grid. `min det` correlates with success at point-biserial r = 0.26
(p < 0.001): weakly, as the project's standing position expects of a validity
measure.

## 2. What the keypoints must encode (draft `tab:ladder`)

`python -m tpgpt.experiments.run_ladder --cells outputs/campaigns/ycb_s0_gated/ycb/rows.json --out outputs/ladder_ycb_s0`.
The five constructions fitted to all 210 cells with the grasp **pinned** to
the candidate the campaign executed, so the construction is the only thing
that varies; no physics. Aim is the closest approach of the transported
fingertip path to the target grasp point; orientation is the angular error of
the transported hand at the grasp; residual is the keypoint interpolation
error; `min det` along the executed path; a fold is a map whose Jacobian
changes sign, which the pipeline refuses. Medians with interquartile ranges;
p is Wilcoxon signed-rank against (D), Holm-corrected over the six columns.

| construction | admits | aim (mm) | orient. (deg) | residual (mm) | min det median [IQR] | worst | folded |
|---|---|---|---|---|---|---|---|
| (A) object box | object pose and size | 35.9 [26.4, 48.9] | 16.9 [9.3, 27.2] | 0.00 | 1.05 [0.80, 1.51] | 0.376 | 1 |
| (B) finger contacts | hand geometry | 7.1 [4.5, 12.2] | 11.5 [7.4, 16.1] | 0.00 | 0.93 [0.71, 1.00] | 0.024 | 5 |
| (C) support cube | object position | 48.0 [32.9, 66.7] | 20.7 [14.2, 33.5] | 0.00 | 0.90 [0.76, 0.94] | 0.403 | 0 |
| (E) grasp cube + contacts | grasp position, hand geometry | 0.00 | 10.6 [8.5, 13.2] | 0.00 | 0.30 [0.17, 0.49] | 0.011 | **43** |
| **(D) grasp cube (ours)** | grasp pose | **0.00** | **1.45 [0.91, 1.59]** | 0.00 | 0.80 [0.67, 0.93] | 0.107 | **0** |

Every column except the residual differs from (D) at Holm p < 1e-15 (the
residual is ~1e-7 mm for all: the GP interpolates exactly, and (E)'s
contradiction between cube and contacts shows up as folds, not residual).

**Read it honestly.** (D) is exact in aim *by construction* and wins
orientation by an order of magnitude (1.45 degrees against 10.6-20.7), and it
is the only construction with no fold on 210 cells. It is **not** the
best-conditioned: its `min det` median of 0.80 is below the object box's 1.05
and the contacts' 0.93. The draft's expected-shape text says the grasp cube
"keeps the map well conditioned"; the measurement says it keeps it *valid*
(no folds, worst 0.107) while trading some conditioning for aim and
orientation. (E) is the construction the draft predicted would violate (R1);
what it actually does is fold 43 of 210 maps and halve `min det` on the rest,
which is a stronger statement.

## 3. Is the map blind to the object and the hand? (draft `sec:exp:blind`)

`python -m tpgpt.experiments.map_blindness --campaign outputs/campaigns/ycb_s0_gated/ycb --out outputs/map_blindness_ycb_s0`,
9999 permutations. Map distance is the RMS difference of two cells' `phi(x)-x`
on the common 20^3 grid (or of their transported paths); the predictor is the
distance between the two cells' grasp *pairs* -- root-sum-square over pick and
release of the SE(3) geodesic, 0.1 m per radian -- because the keypoints are
exactly those two poses and the release depends on the object through its
footprint height.

| | fingerprint | path |
|---|---|---|
| Mantel, map ~ grasp pair | r = 0.76, p = 1e-4 | r = 0.88, p = 1e-4 |
| same gripper, unconditioned | r = -0.011 | r = -0.013 |
| same gripper, partial on grasp | **r = 0.004, p = 0.38** | **r = 0.007, p = 0.20** |
| PERMANOVA share, gripper | 1.4 %, p = 0.98 | 1.0 %, p = 1.00 |
| same object, unconditioned | r = 0.078, p = 1e-4 | r = 0.090, p = 1e-4 |
| same object, partial on grasp | r = -0.051, p = 1e-4 | r = -0.079, p = 1e-4 |
| PERMANOVA share, object | 9.7 %, p = 1e-4 | 12.7 %, p = 1e-4 |

**The map is blind to the hand.** Nothing about the gripper survives any
test: partial r within 0.007 of zero, one per cent of variance, p ~ 1. The
object retains a residual association after conditioning that is detectable
at 21,945 pairs but tiny (|r| <= 0.08, under one per cent of variance) and
**negative** -- same-object pairs sit slightly *farther* apart in map space
than their grasp distance predicts, the opposite sign an object-geometry leak
would have. On the 35-cell subset of one condition the same partial r was
-0.02 and 0.00 (p > 0.7). The draft's sentence attributing variance "before and
after conditioning" by PERMANOVA should go: a PERMANOVA on residual distances
is not a valid statistic (a residual can be negative), and the conditioned
claim is the partial Mantel's. Figures in `outputs/map_blindness_ycb_s0/`.

## 4. What would make these wrong

One draw. Two of the three seeds the draft promises are not run; each needs
its own planner cache (`TPGPT_GRASP_CACHE=outputs/grasp_cache_s<k> --seed <k>`).
The ladder and the blindness analysis are geometry on cached grasps and will
reproduce exactly on this cache; the 83/210 will move by a few cells on a new
draw, as 7.41 measured. The per-cell timing for `sec:exp:cost` was not
recorded by this campaign (`timings` is empty in the rows) and needs one line
in `_row`.
