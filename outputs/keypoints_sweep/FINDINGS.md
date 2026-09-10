# Which keypoints should pin the transportation map?

**Written for a reader with no prior context.** Every experiment below states what
it asks, the conditions it was measured under, what each column means, the
result, and *why* the result came out that way. Sections 1 and 2 are background;
if you know the method, start at section 3.

**Provenance.** `manifest.json` in each output directory records the commit.
Where it reports `reproducible: false` the tree had uncommitted changes and the
numbers are a scratch experiment, not evidence — that rule is `ROBOTICS_NOTES`
§7.26, written after 254 MB of results had to be deleted for exactly this.

---

> # ⚠ EVERY NUMBER IN EXPERIMENTS G, H AND I IS WITHDRAWN
>
> **The scene they were measured in was broken.** `ROBOTICS_NOTES.md` §7.32 has
ame.> `TabletopShelf`:
>
> 1. **Objects were still falling at handover.** `SETTLE_STEPS = 60` is 0.12 s;
>    the cereal needed up to 0.96 s. In four of six scenes a 150 mm box toppled
>    **35-78 mm on its own**, with the arm idle, *after* the cloud was captured.
> 2. **Objects were created inside the robot.** The arm's rest pose sat in the
>    sampling region: 4.85 mm (yumi), 12.85 (robotiq85), 20.64 (xarm), 26.77
>    (robotiq140) of interpenetration, ejected on the first physics step.
> 3. **So the scene differed per gripper by up to 154 mm** from the same seed.
>    Every cross-gripper comparison below compared *different worlds*.
>
> And separately, **every `contact_offset` was wrong**. It derives from
> `calibrated_depth`, which was itself calibrated against these moving objects;
> re-measured, all seven hands moved by up to 30 mm. The Panda's
> wrist-to-fingertip offset was 41.1 mm throughout these experiments and is
> **11.1 mm**. That value converts the source demonstration's labels
> (`_to_tool_frame`), so the label path — and the map fitted to it — shifts with
> it. **This is why even Experiment G's pure geometry is withdrawn**, contrary to
> what §8d's own text claims.
>
> **What survives**, and is worth reading:
>
> * the **methods** — what was varied, what was held fixed, how each metric is
>   defined and measured;
> * the **withdrawn-claims record** in §9, which is the point of keeping this
>   document rather than deleting it (§7.26);
> * the observation that `aim` and `orient` are **~0 by construction** for the
>   cube variants and therefore were never evidence;
> * the finding that the geometric metrics correlate with physical contact at
>   `r` between −0.07 and +0.02 — **geometry did not predict physics**, and that
>   conclusion does not depend on the scene being right.
>
> **Do not quote a number from §§8d, 8e or 8f.** The re-runs are on a corrected
> scene, with fresh clouds, fresh grasps and fresh calibrations.

## 1. Background: what a transportation map is, and what keypoints do

The project transports **one** recorded demonstration onto new objects. The
mechanism is a map `phi` that **warps space itself** — not the object, the space,
including the empty air the arm moves through. Push the demonstration's whole
trajectory through `phi` and you get the plan for the new scene.

`phi` is pinned by **keypoints**: paired points, one set on the source scene and
one on the target, and `phi` is required to carry each source keypoint *exactly*
onto its partner (Sec. III-D of the paper) and to interpolate smoothly between
them. Keypoints are therefore the entire interface between "what the scene looks
like" and "where the robot goes".

`phi` is built in two stages, `phi(x) = gamma(x) + psi(gamma(x))`:

- **`gamma`** — a single affine map (rotate, scale, shear, translate) fitted to
  all the keypoints at once. Global: it treats every point in space the same way.
- **`psi`** — a Gaussian-Process residual that adds whatever local bending is
  needed so each keypoint lands exactly.

### 1.1 The quantity everything is judged by: `det(J)`

`J` is the **Jacobian** of `phi` — how much output distance you get per unit of
input distance, in each direction. Its determinant `det(J)` is the **local volume
ratio**: how much a tiny cube of space grows or shrinks under the map.

| `det(J)` | what it means physically |
|---|---|
| `= 1` | space is carried rigidly — no stretch at all |
| `> 1` | space is inflated |
| `0 < det < 1` | space is compressed; `0.1` means tenfold compression |
| `= 0` | a region has been crushed to **zero volume** |
| `< 0` | space has been turned **inside out** — two separate regions now overlap |

A map with `det(J) < 0` anywhere is **folded**. It is not merely inaccurate: it
is not invertible, a trajectory pushed through it can double back on itself, and
— as section 4 shows — the commanded hand orientation can invert. The pipeline
refuses such a map outright.

`J` also carries the **orientation**. Eq. 11 of the paper transports the hand's
rotation as `R_hat = J_perp R`, where `J_perp` is `J`'s rotation part. This has a
consequence that drives most of what follows:

> **`J_perp` is one rotation per point in space.** Whatever rotation it applies
> to the *gripper*, it applies to *everything else at that point* — including the
> world's vertical. A map cannot be made to rotate the hand without rotating the
> space around it.

---

## 2. The source demonstration — measured once, reused everywhere

Every experiment transports this same single demonstration. It was recorded in
the **Reshelving** scene, which is a different scene from the tabletop one the
targets live in; that is the point of the method.

| | |
|---|---|
| scene | `Reshelving`, robot **Panda**, gripper `panda` |
| object | `product_main`, a box **50 x 50 x 90 mm** |
| labels | **200** at 20 Hz — a 10.0 s trajectory |
| grasp TCP (fingertips) | `[-0.110, -0.173, 0.933]` m |
| grasp approach / closing | `[0, 0, -1]` (straight down) / `[0.814, -0.581, 0]` |
| jaws close / open | label **50** / label **159** (109 labels carrying) |
| picked from | table surface, z = **0.802 m** |
| placed at | `[0.200, -0.130, 1.104]`, shelf surface z = **1.059 m** |
| travel | **313 mm** lateral, **257 mm** up |
| peak speed | 0.202 m/s |
| source keypoints | two fixed cubes on `product_main` and `goal_marker` |

Two details matter downstream:

**The demonstration is recorded at the wrist**, at the `grip_site` frame, while a
grasp's TCP is at the **fingertips**. Those are `contact_offset("panda")` =
**41.1 mm** apart. Every experiment below converts the labels to the fingertip
("tool") frame with `_to_tool_frame` before transporting, exactly as
`pipeline.run` does. Section 6.1 shows what happens when that is forgotten.

**The demonstration approaches straight down.** That makes it the reference the
grasp filter measures candidates against (section 2.2).

### 2.1 The target scene

| | |
|---|---|
| scene | `TabletopShelf`, robot Panda, destination slot `top_middle` |
| observation | 256x256 depth + instance segmentation, **one-sided real point clouds** |
| objects | the five robosuite meshes below |

| object | size (mm) | cloud points | notes |
|---|---|---|---|
| cereal | 80 x 80 x 60 | 985 | best observed |
| milk | 50 x 50 x 150 | 399 | tallest |
| can | 50 x 50 x 80 | 217 | closest to the source object |
| bread | 60 x 60 x 60 | 86 | irregular |
| lemon | 75 x 75 x 40 | **17** | below `MIN_CLOUD_POINTS = 40`; the pipeline rejects it |

`fig_keypoints_3d.png` shows the source and target keypoint blocks against their
clouds; `scene_workspace.png`, `scene_sideview.png` and `scene_birdview.png` show
the arena. `fig_transport_3d.png` shows one demonstration transported onto the
cereal, with the source and warped trajectories and both shelf heights.

### 2.2 How the target grasp is chosen

**GraspGen-X**, running as a ZMQ server (`franka_panda`, fp32, CPU), generates
**100 candidates** per object from the object's point cloud. Candidates are then
filtered exactly as `filter_grasps` filters them: anything whose approach
direction is more than `MAX_APPROACH_MISMATCH_DEG = 45` from the demonstration's
own approach is rejected, and the best-aligned survivor is used.

**The filter is not optional and section 5 shows why**: unfiltered, the
planner's top-scoring candidate is routinely 119-174 degrees from the
demonstration's grasp orientation.

The **place** grasp is not chosen separately. It is *derived* from the pick
grasp: once the jaws close the object is rigidly attached, so `place_transform`
puts the object into the slot and `target.grasp.transformed(...)` carries the
grasp along with it. The placed pose is therefore implied by the pick grasp and
the trajectory, and is exact.

---

## 3. The constructions being compared

All four build the same nine "box" keypoints — a centre plus eight corners — per
configuration, and there are four configurations (source-pick, source-place,
target-pick, target-place), so 18 keypoints unless contacts are added.

| # | name | where the box is centred | how its corners are oriented | jaw contacts |
|---|---|---|---|---|
| 0 | `cloud_box` | the **object's centroid**, extent fitted to its point cloud | task frame | none |
| 1 | `cube_task` | the **grasp**, fixed 20 mm half-extent | task frame | none |
| 2 | `cube_grasp_pose` | the **grasp**, fixed 20 mm | **full grasp pose** | none |
| 3 | `cube_grasp_pose_composed` | the **grasp**, fixed 20 mm | full grasp pose | **composed second stage** |

**Variant 0 is today's default and the control.** Its box's *extent is the
object's own*, so a taller object asks `phi` to inflate space in proportion.

**The task frame** (`task_frame`) is built from the support normal and the
closing axis *projected perpendicular to it*. It therefore keeps world-up as one
of its axes and can only ever express a **yaw**; a grasp's out-of-plane approach
tilt is discarded. **The full grasp pose** keeps all of it.

**Composed** means the jaw contacts are pinned by a *second map applied after the
first*, `phi(x) = phi_2(phi_1(x))`, rather than thrown into the same one. Under
composition determinants multiply, `det(J) = det(J_2) det(J_1)`, so each stage
can be checked independently; a convex blend has no such property. The second
stage uses an identity affine so its whole warp lives in the residual, where its
length scale (30 mm here) confines it.

---

## 4. Experiment A — does the object's size belong in the warp?

**Question.** The cloud box's extent *is* the object's. Does transporting onto a
taller object then inflate space, and does that matter?

**Method.** Synthetic box clouds so the object's height can be swept
continuously, source half-height fixed at 45 mm, target swept 20-300 mm. Map
fitted through the full four-block `scene_keypoints` construction; `det(J)`
sampled along the transported trajectory. No physics.

**Result.**

| target/source height ratio | 0.4x | 1.0x | 1.7x | 2.8x | 4.4x | 6.7x |
|---|---|---|---|---|---|---|
| **cloud box** `min det(J)` | 0.598 | 0.844 | 1.030 | 1.551 | 2.693 | **4.160** |
| **grasp cube** `min det(J)` | 0.698 | 0.698 | 0.698 | 0.698 | 0.698 | **0.698** |

**Why.** The cloud box's determinant tracks the size ratio almost linearly,
because its keypoints *are* the object's corners: making the target object 6.7x
taller literally asks `phi` to carry a 90 mm box onto a 300 mm one, and it obeys
by inflating space 4.16-fold. The arm's trajectory has no reason to care how tall
the object is, so this is deformation with no task meaning. A fixed cube is flat
across the whole range because its keypoints do not depend on the object at all.

**What this does not show.** It does *not* fold. An earlier synthetic of mine
claimed the cloud box folds at `det = -0.63`; that was an artifact of a
hand-built scene and is withdrawn (section 6.3). Inflation is not folding, and
the cloud box remains a valid map here.

---

## 5. Experiment B — the four constructions on real objects and real grasps

**Question.** With real point clouds and real planner-chosen grasps, which
construction produces the best-conditioned map, aims at the grasp, and carries
the hand's orientation?

**Conditions.** Source demonstration of section 2, converted to the tool frame.
Target objects cereal, milk, can, bread. Target grasp from GraspGen-X, approach-
filtered as in section 2.2, best-aligned candidate, at the object's mid height.
Slot `top_middle`. Geometry only — no policy, no physics.

**Result.**

| object | construction | `min det(J)` | stage1/stage2 | `aim_map` | orientation | `tilt_mid` |
|---|---|---|---|---|---|---|
| cereal | 0 cloud box | 0.948 | — | 65.1 mm | 4.2° | 2.6° |
| cereal | 1 cube task | 0.976 | — | 0.0 mm | 4.1° | 2.2° |
| cereal | **2 cube grasp pose** | 0.970 | — | 0.0 mm | **0.2°** | 3.3° |
| cereal | 3 composed | 0.970 | 0.97 / 1.00 | 9.4 mm | 2.0° | 2.8° |
| milk | 0 cloud box | 0.584 | — | 70.1 mm | 7.8° | 4.1° |
| milk | 1 cube task | 0.997 | — | 0.0 mm | 7.5° | 2.1° |
| milk | **2 cube grasp pose** | 0.996 | — | 0.0 mm | **0.7°** | 7.0° |
| milk | 3 composed | **0.080** | 1.00 / **0.08** | 2.8 mm | 0.7° | 7.2° |
| can | 0 cloud box | 0.414 | — | 40.2 mm | 3.2° | 3.2° |
| can | 1 cube task | 0.996 | — | 0.0 mm | 1.6° | 1.6° |
| can | **2 cube grasp pose** | 0.996 | — | 0.0 mm | **0.4°** | 2.3° |
| can | 3 composed | **0.169** | 1.00 / **0.17** | 13.0 mm | 0.8° | 2.4° |
| bread | 0 cloud box | 0.528 | — | 26.5 mm | 8.1° | 10.8° |
| bread | 1 cube task | 0.917 | — | 0.0 mm | 4.3° | 7.0° |
| bread | **2 cube grasp pose** | 0.916 | — | 0.0 mm | **0.9°** | 6.4° |
| bread | 3 composed | **0.305** | 0.92 / **0.30** | 9.3 mm | 2.2° | 7.3° |

### What each column means

| column | definition | why it is here |
|---|---|---|
| `min det(J)` | smallest volume ratio anywhere along the transported trajectory | `<= 0` means folded; low means space is being crushed somewhere the arm travels |
| stage1 / stage2 | the same, per composition stage | determinants multiply, so this says **which** stage is responsible |
| `aim_map` | `\|\|phi(source grasp TCP) − target grasp TCP\|\|` | does the map carry the point where the jaws close onto the point where they must close? Frame-independent: both ends are fingertip points |
| orientation | angle between the Eq. 11-transported hand rotation `J_perp R` and the target grasp's own rotation, minimised over the parallel jaw's 180° symmetry | does the hand arrive at the right *angle*. **No keypoint pins a derivative**, so unlike `aim_map` this is never trivially satisfied |
| `tilt_mid` | median angle by which `J_perp` tilts world-up, over the middle half of the path | separates "the hand was rotated where it needed to be" from "the whole trajectory was rotated" |

> **`aim_map` is 0.0 mm for constructions 1 and 2 *by construction*.** The cube's
> centre **is** the target grasp TCP, and `phi` interpolates keypoints exactly.
> That zero is arithmetic, not evidence. The informative half of the comparison
> is the **cloud box's 26-70 mm**, which is a genuine failure to carry the grasp
> point, and the **orientation** column, which nothing pins.

### Results and why

**Construction 2, the grasp-pose cube, wins on every axis.** Determinant 0.916 to
0.996 against the cloud box's 0.414 to 0.948; orientation **0.2 to 0.9 degrees**
against 3.2 to 8.1; grasp point carried exactly.

*Why it aims better than the cloud box.* The cloud box is centred on the object's
**centroid** while the jaws close at the **grasp**, and nothing couples the two.
When the planner picks a grasp that is not at the object's centre — which is the
normal case — nothing in the keypoints carries that point across, and the error
is however far the grasp sits from the centroid: 26 to 70 mm here.

*Why it orients better than construction 1.* The task frame projects the closing
axis perpendicular to world-up, so it can only carry a yaw. Any out-of-plane
approach tilt in the planner's grasp survives untouched, and shows up as the 4.1
to 7.5 degree orientation error of construction 1. The full grasp pose carries
all of it, leaving 0.2 to 0.9 degrees.

*Why its determinant is better.* Both cubes are the same size regardless of the
object, so the map never has to inflate or crush space to reconcile two
differently-shaped boxes — the mechanism of Experiment A.

**Construction 3, the composition, is measurably worse, and the stage columns say
exactly why.** Stage 1 is healthy in every case (0.92 to 1.00). **Stage 2 is the
damage**: 1.00, 0.08, 0.17, 0.30. And the composed map's `aim_map` rises from
0.0 mm to 2.8-13.0 mm.

*Why.* This is the original conflict reappearing in a new place. The jaw contacts
sit 29 mm from the grasp centre and want to go where the target's contacts are;
the cube's centre *is* the grasp and wants to stay. Composing does not remove
that disagreement, it just moves where it is resolved — into stage 2, which
satisfies the contacts by **dragging the cube's centre off its target** (hence
the aim error appearing) and compressing volume around them (hence `det` falling
to 0.08). The one case where stage 2 is harmless, cereal at 1.00, is the case
where the correction it was asked for was near zero.

**So the contacts are not worth adding, in either form.** In the same map they
fold it (measured earlier at 10 cells of 10 for the cloud box); as a composed
stage they cost the determinant and the aim. The grasp-pose cube already carries
the grasp point exactly and its orientation to under a degree, which is what the
contacts were for.

`fig_ablation_objects.png` and `fig_ablation_contacts.png` show the earlier
object and contact ablations from which this comparison grew;
`fig_ablation_grasp_yaw.png` shows the same map's insensitivity to rotating the
grasp's closing axis in-plane.

---

## 6. Experiment C — the orientation trade, isolated

**Question.** Constructions 1 and 2 differ only in whether the cube's corners
carry the grasp's out-of-plane tilt. What does carrying it buy, and what does it
cost?

**Method.** One object (cereal), one grasp, with the approach tilted about the
grasp's **own closing axis** by 0, 15, 30 and 45 degrees. Tilting about that axis
leaves the task frame *unchanged by construction*, so it isolates exactly the
difference between the two constructions.

**Result.**

| approach tilt | task frame: orientation error | task frame: world tilt | grasp pose: orientation error | grasp pose: world tilt |
|---|---|---|---|---|
| 0° | 1.6° | 4.8° | 1.6° | 4.8° |
| 15° | **14.5°** | 4.8° | **1.6°** | **15.7°** |
| 30° | **29.4°** | 4.8° | **1.7°** | **28.9°** |
| 45° | **44.4°** | 4.8° | **1.6°** | **41.3°** |

**Why — this is a conservation law, not a tuning choice.** `J_perp` is a single
rotation per point and Eq. 11 uses that same rotation for the gripper. So the
angle a map adds to the hand and the angle it adds to the vertical are *the same
number*. The task frame pays the entire tilt as orientation error and nothing in
world tilt; the grasp pose pays nothing in orientation and the entire tilt in
world tilt. There is no setting that buys one without the other, and a unit test
(`test_metrics_transport.py`) asserts the identity directly.

**Is the world tilt harmful?** Not established. It rotates the demonstration's
straight-up lift into a diagonal one, which is a clearance and reachability
question. But because it comes from the **affine stage** it applies *uniformly*
across the transit rather than locally, which is closer to "the object is held at
a consistent angle" than to a distortion. Experiment E measures reachability
directly and finds no penalty. It is carried as an open question, not scored as a
cost.

**At 0 degrees the two constructions are identical**, and that is a designed
internal control rather than a null result: for a straight-down approach the task
frame and the grasp pose differ by a half turn about the closing axis, which maps
a cube's corners onto themselves. **Any experiment using top-down grasps cannot
tell these two constructions apart** — which is why section 2.2's real planner
grasps are used throughout and the top-down recipe was abandoned.

---

## 7. Experiment D — why the grasp filter is not optional

**Question.** Section 2.2 filters candidates to within 45 degrees of the
demonstration's approach. What happens without it?

**Result.** Taking the planner's top-*scoring* candidate instead, 96 cells:

| construction | valid maps | `min det` (median) | orientation (median / max) | world tilt (median / max) |
|---|---|---|---|---|
| cloud box | 16/16 | 0.547 | **119.4° / 173.2°** | 3.9° / 6.4° |
| cube task | 16/16 | 0.931 | **120.1° / 174.0°** | 1.8° / 5.3° |
| cube grasp pose | **7/16** | **−0.031** | **1.3° / 2.0°** | **116.5° / 154.6°** |

**Why.** GraspGen-X scores candidates on grasp quality, with no knowledge of the
demonstration. Its best candidate is frequently a grasp approached from a
completely different direction — 119 to 174 degrees away, very nearly inverted.
The conservation law of Experiment C then bites at full strength: the grasp-pose
cube carries that orientation *faithfully* (1.3°) by rotating the world by up to
**155 degrees**, which turns the demonstrated lift into a descent, and its
determinant goes negative in 9 of 16 cells.

**The filter bounds the approach mismatch at 45 degrees, and therefore bounds the
world tilt too.** That is what makes construction 2 safe in Experiment B and
unsafe here. It is a real dependency and should be stated as one: *the grasp-pose
cube is only sound behind an approach filter.*

**Two findings about the pipeline, not the keypoints.** The **milk** had **0 of
100** candidates inside the filter (closest 49.8°), so the pipeline would reject
it before keypoints were built at all. The **lemon**'s 17-point cloud raises
`GraspGenUnavailable`. Neither is a keypoint problem and neither is addressed
here.

---

## 8. Experiment E — is a well-conditioned map actually executable?

**Question.** Everything above is map geometry. A construction that produces a
beautifully conditioned map through poses the arm cannot hold has bought nothing.

**Method.** The transported path followed **pose by pose under position control**
(`replay_labels`) — no GP policy, no attractor integration, no lag gate — so every
number is attributable to the keypoints alone. A **fresh seeded scene is built for
every replay**, and the driver asserts the seeded scene is reproducible across
rebuilds. Read a replay as an **upper bound**: a failure here is the keypoints', a
failure only under the policy is the executor's.

*Results are in `outputs/keypoint_replay/`; the run with the recipe grasps this
supersedes is kept at `outputs/keypoint_replay_recipe_grasps/`.*

### What each column means

| column | definition |
|---|---|
| reachable | fraction of waypoints IK could solve **with the commanded hand orientation** |
| track mm | mean distance between the commanded pose and the pose the arm reached |
| r / unr | that error split by whether the pose was reachable. A sharp split means the residual is **kinematic** — the arm cannot be there with its hand at that angle — and no controller closes it |
| slip mm | drift of the object-to-fingertip vector while held. The grasp is assumed rigid once closed; drift is the object sliding |
| held | control steps with genuine MuJoCo contact between hand and object. **0 means the hand never touched it** |
| place mm | final distance from the destination slot |
| ok | did the object end up in its slot |

### Result

| construction | object | reachable | track mm | r / unr | slip mm | **held** | place mm | ok |
|---|---|---|---|---|---|---|---|---|
| 0 cloud box | cereal | 72% | 11.6 | 5.8 / 26.4 | — | **0** | 232.0 | no |
| **2 cube grasp pose** | cereal | 62% | 16.6 | 5.4 / 35.0 | 16.5 | **110** | **34.5** | **yes** |
| 3 composed | cereal | 71% | 16.4 | 8.7 / 35.5 | 39.6 | 117 | 22.6 | **yes** |
| 0 cloud box | milk | 62% | 18.9 | 3.9 / 42.8 | — | **0** | 392.6 | no |
| **2 cube grasp pose** | milk | 82% | 13.9 | 12.0 / 22.7 | 48.5 | **138** | **10.9** | **yes** |
| 3 composed | milk | 82% | 13.7 | 11.7 / 22.7 | 46.9 | 130 | 13.8 | **yes** |
| 0 cloud box | can | 86% | 13.5 | 13.5 / 13.3 | 7.6 | 82 | 29.6 | **yes** |
| **2 cube grasp pose** | can | 84% | 23.9 | 24.4 / 21.3 | 44.5 | **148** | 20.6 | **yes** |
| 3 composed | can | 74% | 22.3 | 17.5 / 35.9 | 16.8 | 143 | 19.6 | **yes** |
| **0 cloud box** | bread | 100% | 23.1 | 23.1 / — | 9.6 | 110 | **31.1** | **yes** |
| 2 cube grasp pose | bread | 72% | 48.5 | 24.8 / **111.0** | 47.7 | 112 | 136.3 | no |
| 3 composed | bread | 74% | 47.9 | 27.0 / **107.2** | 12.7 | 110 | 128.8 | no |

**Succeeded: cloud box 2 of 4, grasp-pose cube 3 of 4, composed 3 of 4.**

> **Superseded as a cross-hand claim by Experiment H (§8e).** These four cells
> are one hand — a Panda, the hand the demonstration was recorded on. Run across
> **six** hands the two constructions tie: cloud box **4/23**, cube **5/23**. The
> 3-of-4 above is real for the Panda and does not generalise. What *does*
> generalise is the mechanism visible in this table: the cube acquires the object
> far more often (10 firm grips against 3 across six hands) and then loses half of
> them. Read §8e before quoting any success rate from this section.

### Results and why

**The cloud box never touches the object on cereal or milk** — `held = 0` out of
~110 possible control steps. That is not poor tracking, it is a systematic miss,
and it is exactly what Experiment B's geometry predicted: its `aim_map` on those
two objects is 65.1 mm and 70.1 mm, so the jaws close around empty air. Its two
successes are the can and the bread, its two smallest aim errors (40.2 and
26.5 mm). **The geometry tier predicted the physics tier**, which is the best
evidence available that neither instrument is lying.

**The grasp-pose cube grasps every object** — 110 to 148 contact steps — and
places three of four within 35 mm. It carries the grasp point exactly, so the
jaws close where the planner intended.

**The composition is tied with it in physics, despite being much worse in
geometry.** Its stage-2 determinant falls to 0.08-0.30 (Experiment B) and yet it
succeeds on the same three objects with comparable placement. So the determinant
penalty did **not** convert into an execution penalty at this sample size. Two
readings are available and this data cannot separate them: either a compressed
stage-2 determinant is tolerable when it is confined to the grasp's
neighbourhood, or four objects is too few to see the cost. Given the composition
also *loses* the exact aim (0.0 → 2.8-13.0 mm) and adds a second thing to tune,
there is no measured reason to prefer it.

### Why bread fails for the cube — and it is not the aiming

Bread is the one object where the cube loses, so it is worth attributing
properly rather than noting.

*Not the grasp.* `held = 112` steps: the hand closed on the bread and carried it.

*Not the map's place target.* Measured on the geometry, the cube's release error
is **0.0 mm** and its clearance error **0.0 mm** on every object including bread
— the map puts the object exactly on the shelf. (The cloud box's is 2.7-5.8 mm.
Note the `expected_clearance` figures are **not** comparable between
constructions: they come from `corresponding_point`, which normalises by the
box's half extents and therefore degenerates to a rigid offset under a fixed
cube.)

*Not the grasp's tilt.* Bread's chosen grasp is 4.7° from the demonstration's
approach, comparable to cereal's 4.1° where the cube succeeds, and its
`executable_fraction` is 77% against cereal's 58%.

*It is kinematic, during the carry.* The tracking error splits sharply: 24.8 mm
on the poses IK could solve, **111.0 mm** on the 28% it could not. A sharp split
means the arm cannot be at those poses *with the commanded hand orientation* at
all, and no controller closes that — it is the failure mode §7.25 measured. The
object was grasped and then carried along a path the arm could not hold, ending
136 mm from the slot.

**What this analysis cannot yet say is *where* along the path.** `replay_labels`
records per-waypoint reachability but the driver only surfaces the aggregate, so
attributing this to the lift, the transit or the insertion needs the positional
trace exposed. That is the obvious next instrument, and until it exists this
failure is characterised but not localised.

---

## 8c. Experiment F — stress-testing the tilt: does it alter the trajectory too much?

**Question.** Experiment C established that the grasp-pose cube buys its
orientation by rotating world-up by the same angle, and Experiment D showed the
approach filter bounds that at 45°. **But nothing had checked what the rotation
does to the trajectory itself.** A map can be perfectly conditioned and still
produce a path that goes through the table, leaves the arm's envelope, or turns
the lift into something that is no longer a lift.

**Method.** One object (cereal), one filtered GraspGen-X grasp, approach tilted
about the grasp's **own closing axis** by 0 to 90 degrees — well past the 45°
filter limit, deliberately, to find where it breaks. The grasp-pose cube's
transported path is compared **against the task-frame cube's path on the same
keypoints**, so the only difference is the corner orientation. Geometry plus IK;
no policy.

### Shape: how far the path moves

| approach tilt | `min det(J)` | world tilt | deviation from the task-frame path (max) | Fréchet | `executable_fraction` |
|---|---|---|---|---|---|
| 0° | 0.950 | 6.1° | 0.0 mm | 0.0 mm | 100% |
| 10° | 0.937 | 9.8° | 16.8 mm | 17.1 mm | 100% |
| 20° | 0.889 | 13.5° | 35.2 mm | 36.2 mm | 100% |
| 30° | 0.827 | 18.3° | 54.3 mm | 57.4 mm | 96% |
| **45°** *(filter limit)* | **0.714** | **30.2°** | **80.2 mm** | **91.2 mm** | **96%** |
| 60° | 0.578 | 43.1° | 99.4 mm | 122.9 mm | 92% |
| 75° | 0.395 | 55.6° | 108.1 mm | 147.0 mm | 96% |
| 90° | 0.134 | 67.6° | 113.7 mm | 161.9 mm | 92% |

*(the task-frame cube, for reference, holds `min det` 0.976, world tilt 2.2° and
100% executable at every tilt — it is invariant to this axis by construction)*

**The path really does move, and by a lot.** At the filter limit the grasp-pose
path deviates up to **80 mm** from the task-frame path — more than a Panda's jaw
aperture. Deviation grows at roughly 1.8 mm per degree of tilt and then saturates
near 110 mm beyond 60°. So the answer to "does it alter the trajectory" is
unambiguously yes.

### Safety: whether the moved path goes anywhere it should not

| approach tilt | lift off vertical | lift height | lowest point on path | below the table? | max lateral reach |
|---|---|---|---|---|---|
| 0° | 35.0° | 400 mm | 0.884 | no | 0.222 |
| 15° | 34.5° | 399 mm | 0.884 | no | 0.222 |
| 30° | 34.6° | 391 mm | 0.884 | no | 0.222 |
| **45°** | **35.3°** | **377 mm** | 0.884 | **no** | 0.222 |
| 60° | 36.7° | 357 mm | 0.884 | no | 0.222 |
| 90° | 41.2° | 303 mm | 0.883 | no | 0.222 |

*(table surface z = 0.800, shelf z = 1.081. The demonstration's lift segment runs
label 50 to 123 and rises 378 mm; it is already 35° off vertical because it moves
across as well as up, so 35° at zero tilt is the demonstration's own diagonal and
not an artifact.)*

### Result and why

**The tilt rotates the path about the grasp; it does not push it anywhere
dangerous.** Three independent checks agree:

- **The lift stays a lift.** Its angle from vertical moves 35.0° → 35.3° across
  the whole filter range, and only reaches 41.2° at a 90° tilt. Its height
  shrinks 400 → 377 mm at the filter limit (6%), and 303 mm at 90° (24%).
- **The path never enters the table.** Its lowest point is 0.883-0.884 m against
  a table at 0.800 m, constant to a millimetre across every tilt.
- **The lateral extent does not change at all** — 0.222 m throughout, inside the
  0.24 m at which this arm's end-effector x saturates.

The reason all three hold is that the rotation is applied *about the grasp*,
which sits above the table and inside the envelope, so rotating the path around
it moves points laterally and vertically in compensating directions rather than
translating the whole path somewhere new.

**And reachability barely suffers**: 100% at 0-20°, 96% at the 45° filter limit,
92% at 90°. This is the check that mattered most, because §7.7 measured a warped
path that was fully reachable in position and reachable with its commanded
orientation at only 3 of 20 points. That failure does **not** reproduce here.

**What does degrade is the determinant**: 0.950 → 0.714 at the filter limit →
**0.134** at 90°, a sevenfold volume compression. So the tilt is not free, and
the thing that keeps it safe is the filter. Beyond 45° the map is still valid but
increasingly distorted, which is consistent with Experiment D finding that
*unfiltered* candidates drove the determinant negative outright.

**Conclusion: the grasp-pose cube's trajectory tilt is acceptable within the
approach filter, and the filter is what makes it acceptable.** That is the same
dependency Experiment D established for the orientation, now confirmed for the
path shape. It should be recorded as a **precondition of the construction**: the
grasp-pose cube is sound behind a 45° approach filter and has not been shown
sound without one.

**Caveat.** One object, one grasp, one axis of rotation. The tilt was applied
about the grasp's closing axis, which is the axis the task frame is invariant to
and therefore the one that isolates the two constructions. Tilting about the
*approach* axis is a roll that a parallel jaw is symmetric under, and tilting
about the third axis is untested.

---

## 8d. Experiment G — does one keypoint construction hold across nine hands?

**Question.** The grasp cube is centred on the grasp point with a **fixed** 20 mm
half extent and encodes nothing whatever about the hand: not the jaw aperture,
not the fingertip depth, not the finger count. Everything so far was measured on
a Panda. If the grasp centre is a good enough representation of a grasp, the map
should come out equally valid and equally well aimed for every hand — and if it
is not, that is the single assumption the whole cross-embodiment claim rests on.

**Why this is falsifiable rather than circular.** A hand can only reach the map
through two channels, and both are *inputs* to it rather than parameters of it:

1. **Which grasp GraspGen-X returns**, since the planner conditions on the hand's
   swept volume. A Robotiq 2F-140 and a Yumi get different candidate sets on the
   same cloud.
2. **The tool offset the labels are expressed in** — `contact_offset(hand)`, the
   wrist-to-fingertip distance, which spans **24.3 mm (yumi) to 134.4 mm
   (inspire)**, a factor of 5.5.

So if the construction quietly depended on the hand, `min det(J)`, the aim or
the transported orientation would move with one of those. The test is whether
they do.

### Conditions

| held fixed | value |
|---|---|
| source demonstration | reshelving seed 0, one demonstration, converted to the tool frame by `contact_offset("panda")` |
| destination | `top_middle` shelf slot |
| scene seed | 0 |
| grasp height | mid (`height_fraction=0.5`) |
| grasp source | **real GraspGen-X candidates for each hand**, from `tpgpt.grasp.cache`, filtered to within `MAX_APPROACH_MISMATCH_DEG` = 45° of the demonstrated approach |
| cube half extent | 20 mm (`GRASP_CUBE_HALF_EXTENT`) |
| scoring | geometry only — no policy, no rollout, no physics |

| varied | values |
|---|---|
| hand | all nine registered pairs |
| object | cereal, milk, can, bread, lemon |
| construction | the four of §3 |

**A scene is rebuilt per hand**, because the mounted gripper has to be set at
construction. That has a consequence worth stating: **the same seed does not
settle the objects identically across hands.** The placement sampler is seeded
the same, but the scene's 60 settle steps then run with a different hand
attached, and the can comes to rest at `[-0.1255, -0.0683, 0.8399]` with a Panda
against `[-0.1139, -0.0777, 0.8426]` with a Robotiq 2F-140 — **14.6 mm apart**.
Every cross-hand comparison therefore carries a ~15 mm scene difference by
construction. It is recorded per row as `object_position` rather than assumed
away.

150 cells; **134 scored**, 10 skipped, 6 refused. Milliseconds per cell once the
grasp is cached.

### What each column means

| column | meaning |
|---|---|
| `tcp mm` | `‖contact_offset(hand)‖`, the wrist-to-fingertip distance the labels are converted by. The hand's only geometric entry into the map besides its grasps |
| `aper mm` | the hand's published jaw aperture, from GraspGen-X's swept volume |
| `min det` | smallest `det(J)` over the demonstration. Positive everywhere means the map is a diffeomorphism (property (ii), §1.1); larger is better conditioned |
| `aim mm` | `‖φ(source grasp TCP) − target grasp TCP‖`. **Zero by construction for the cube variants** (the cube's centre *is* that point and `φ` interpolates keypoints exactly), so it is not evidence for them — it is evidence about the cloud box |
| `orient` | angle between `J_perp · R_source_grasp` and the target grasp's true rotation, after minimising over the jaw's own 180° symmetry |
| `tilt mid` | median tilt of world-up over the middle half of the path — what the map does to the *transit*, where nothing should be rotating |
| `resid µm` | how far `φ` misses its own keypoints. Property (i) of Sec. III-D demands ~0 |

### Result, pooled over all hands and objects

| construction | n | `min det` med | `min det` min | aim med | orient med | orient max | resid med |
|---|---|---|---|---|---|---|---|
| 0 cloud box | 35 | 0.629 | 0.198 | **62.2 mm** | 6.10° | 38.2° | 0.14 µm |
| 1 task-frame cube | 35 | **0.961** | 0.549 | 0.0 mm | 4.14° | 39.4° | 0.65 µm |
| **2 grasp-pose cube** | 35 | 0.916 | 0.555 | **0.0 mm** | **0.82°** | **1.8°** | 0.53 µm |
| 3 composed | 29 | 0.456 | 0.102 | 6.8 mm | 1.11° | 10.5° | **10 864 µm** |

Validity gates, per cell:

| construction | `det(J) > 0` everywhere | residual < 1e-5 |
|---|---|---|
| 0 cloud box | 35/35 | 35/35 |
| 1 task-frame cube | 35/35 | 35/35 |
| 2 grasp-pose cube | 35/35 | 35/35 |
| 3 composed | 29/29 | **0/29** |

### Result, per hand — construction 2

Ordered by tool offset, which is the axis a hand-dependent construction would
track.

| hand | tcp mm | aper mm | n | `min det` med | `min det` min | aim | orient med | orient max | tilt mid |
|---|---|---|---|---|---|---|---|---|---|
| yumi | 24.3 | 50 | 5 | 0.955 | 0.725 | 0.0 | 0.5° | 1.8° | 10.8° |
| xarm | 26.7 | 85 | 4 | 0.917 | 0.583 | 0.0 | 0.9° | 1.6° | 9.9° |
| rethink | 35.1 | 66 | 4 | 0.958 | 0.558 | 0.0 | 0.6° | 1.6° | 8.0° |
| panda | 41.1 | 80 | 3 | 0.693 | 0.592 | 0.0 | 1.6° | 1.7° | 11.4° |
| robotiq3f | 43.6 | 110 | 4 | 0.887 | 0.786 | 0.0 | 1.2° | 1.6° | 7.0° |
| robotiq85 | 47.8 | 85 | 4 | 0.906 | 0.615 | 0.0 | 0.6° | 1.6° | 8.1° |
| robotiq140 | 60.8 | 125 | 4 | 0.889 | 0.555 | 0.0 | 1.0° | 1.7° | 5.5° |
| umi | 117.2 | 80 | 4 | 0.889 | 0.558 | 0.0 | 0.9° | 1.6° | 7.6° |
| inspire | 134.4 | 80 | 3 | 0.956 | 0.637 | 0.0 | 1.1° | 1.5° | 6.1° |

And the same for the control, construction 0:

| hand | tcp mm | `min det` med | aim max | orient med | orient max |
|---|---|---|---|---|---|
| yumi | 24.3 | 0.757 | 152.8 mm | 14.3° | 38.2° |
| xarm | 26.7 | 0.656 | 89.1 mm | 9.9° | 15.0° |
| rethink | 35.1 | 0.633 | 73.7 mm | 5.4° | 23.7° |
| panda | 41.1 | 0.565 | 62.2 mm | 6.1° | 11.7° |
| robotiq3f | 43.6 | 0.676 | 147.5 mm | 6.0° | 15.0° |
| robotiq85 | 47.8 | 0.639 | 85.3 mm | 6.6° | 25.0° |
| robotiq140 | 60.8 | 0.610 | 133.3 mm | 5.6° | 8.9° |
| umi | 117.2 | 0.616 | 115.8 mm | 7.0° | 9.2° |
| inspire | 134.4 | 0.547 | 104.5 mm | 5.9° | 8.2° |

### The answer, and how it is read

**Yes — the grasp centre is a good enough representation of a grasp, for the
grasp-pose cube. Three measurements agree.**

**1. The transported orientation is hand-independent *and* an order of magnitude
better.** Construction 2's per-hand median orientation error runs **0.5° to
1.6°** across all nine hands, with a worst single cell of **1.8°**. The cloud
box, on the same grasps and objects, runs 5.4° to 14.3° with a worst cell of
38.2°. So the cube is roughly seven times better in the median and twenty times
better at the tail, and its spread across hands (0.33° sd of the nine hand
medians) is eight times tighter than the cloud box's (2.75°).

**2. The map's conditioning does not track the tool offset.** Correlating each
hand's median `min det(J)` against its tool offset over the nine hands:

| construction | Pearson r(tool offset, median `min det`) |
|---|---|
| 0 cloud box | **−0.616** |
| 2 grasp-pose cube | **+0.135** |

The cloud box degrades as the hand gets deeper — a real hand dependence, and
mechanically sensible, since a deeper hand shifts the label path further from the
object's centroid, which is where that box is centred. The cube shows essentially
nothing across a 5.5× range of offset. The UMI at 117.2 mm scores 0.889 and the
Inspire at 134.4 mm scores 0.956, against the Panda's 0.693 at 41.1 mm — the
*source* hand is not even the best.

**3. The object moves the map more than the hand does.** Taking each hand's
median across its objects, and each object's median across its hands, and
comparing the two spreads:

| construction | metric | sd of 9 hand medians | sd of 5 object medians | ratio |
|---|---|---|---|---|
| 0 cloud box | `min det` | 0.058 | 0.208 | 3.6× |
| **2 grasp-pose cube** | `min det` | 0.077 | 0.152 | **2.0×** |
| 0 cloud box | orient | 2.749° | 4.616° | 1.7× |
| **2 grasp-pose cube** | orient | **0.332°** | **0.465°** | **1.4×** |

For construction 2 the object is the dominant source of variation, not the hand —
which is what "the hand is not a parameter of this map" should look like. The
ratio is not enormous (2.0×), so this is a supporting measurement rather than the
main one; the orientation numbers in point 1 are the decisive ones, because there
the *absolute* level differs by a factor of seven, not just the spread.

**One honest caveat.** `r(aperture, median orientation error) = +0.499` for
construction 2 — a wider jaw correlates with slightly worse orientation. But the
whole range being correlated is 0.5° to 1.6°, so the effect is real and
negligible: a 1.1° spread on a quantity the cloud box gets wrong by up to 38°.

### Construction 3 is dead, and this is what killed it

Composition **violates property (i)**. Its median keypoint residual is
**10 864 µm — 10.9 mm** — against 0.14 to 0.65 µm for the other three, and
**0 of 29 cells** meet the 1e-5 gate that every cell of the other three passes.
A map that misses its own keypoints by a centimetre is not a transportation map
in the sense Sec. III-D defines; it has no exactness guarantee left to spend.

It also refuses outright on 6 of 35 cells, with `fit_local_correction` raising
because the correction it is asked for exceeds its own 30 mm locality radius —
35.8 mm (robotiq140/cereal), 38.4 (rethink), 43.1 (umi), 49.1 (inspire), 72.7
(yumi/cereal), 30.8 (yumi/lemon). That guard is doing its job: without it those
would have been silent 30-70 mm deformations.

And it loses the aim it was built to keep: median 6.8 mm against the plain
cube's exact 0.0 mm, because stage 2 satisfies the contacts by dragging the
cube's centre off its target. There is nothing left to recommend it.

### A gap this experiment exposed: the closing budget cannot be read from a cube

`diagnose.closing_budget` derives the aiming tolerance as
`(aperture − object width) / 2`, taking the object's width from the target
keypoints. **With a fixed grasp cube that no longer works**, and the sweep shows
it plainly — the width read from the keypoints, averaged per object:

| construction | width read from the keypoints |
|---|---|
| 0 cloud box | bread 43.6, can 45.2, cereal 46.2, lemon 26.4, milk 60.4 mm |
| **2 grasp-pose cube** | **40.0 mm for every object** |

40.0 mm is the cube's own 2 × 20 mm extent. This is not a defect in the
construction — removing the object's size from the keypoints is precisely what
kills the volume scaling of Experiment A — but it means the size cannot be read
back out of them, and a budget derived from a cube would give a lemon and a milk
carton the same tolerance.

Fixed by recording `metrics["object_width_closing"]` from the **cloud**, where
the object's size is actually known, at the point in `pipeline.run` where both
the cloud and the chosen grasp are in scope. `closing_budget` prefers that and
now returns **`None`** for a cube keypoint set with no recorded cloud width,
rather than reporting the cube. A separate and more serious defect was found in
the same function and is in §9.

### Systematic skips

| skip | cells | cause |
|---|---|---|
| lemon | 8 of 9 hands | GraspGen-X itself raises `selected index k out of range`. Its cloud is 17 points against a `MIN_CLOUD_POINTS` of 40, too few for the planner's top-k. Hand-independent |
| milk | panda, inspire only | every one of 100 candidates lies beyond the 45° approach filter, closest 49.8°. **Hand-dependent** *and* **scene-dependent** — see below |

The milk result corrects an earlier reading in this document that treated the
milk as failing the filter outright, but it needs a qualification of its own that
was found while running Tier 2.

**The rejection is not a property of the milk.** Within this sweep it is
hand-dependent: seven of nine hands found a milk grasp inside the 45° filter and
two did not, because the planner conditions on each hand's swept volume. But the
Tier 2 replay, on a **Panda**, then picked and placed the milk successfully at
14.8 mm — the same hand this sweep recorded as having no admissible candidate.

The difference is the scene. Tier 1 builds a **five-object** scene
(`ABLATION_OBJECTS`, including the lemon); Tier 2 builds a **four-object** one
(`REPLAY_OBJECTS`, without it). The placement sampler is seeded identically, but
a different object set lays the scene out differently, so the milk presents a
different cloud to the cameras and the planner draws from a different candidate
set.

So the honest statement is: **whether the milk survives the approach filter
depends on the hand *and* on what else is in the scene.** Neither table alone
licenses "the milk cannot be grasped", and any future claim about an object
failing a filter has to name the scene it was measured in.

---

## 8e. Experiment H — Tier 2: does the transported plan actually execute, on six hands?

**Question.** Everything up to here is geometry. A map can be perfectly
conditioned, exact at the grasp and correctly oriented, and still produce a plan
the arm cannot follow or the hand cannot hold. This is the tier that finds out,
and it is the tier that answers the question the whole multi-gripper thread was
about: *given that no gripper dimension — jaw width, fingertip depth, hand size —
is ever fed to the keypoints, does the transported grasp still work?*

**Method.** `tpgpt/sim/replay.py:replay_labels` drives the arm pose by pose along
the transported path: for each of ~120 waypoints it solves inverse kinematics and
holds the joint target for 8 control steps under a stiff `JOINT_POSITION`
controller. **No GP policy, no attractor integration, no lag gate** — the
executor is removed on purpose, so anything that fails here is attributable to
the plan rather than to the controller. Replay is therefore an *upper bound*: if
it fails, the keypoints or the frame are at fault; if it succeeds and the policy
does not, the fault is in the dynamics thread.

### Conditions

| held fixed | value |
|---|---|
| source demonstration | reshelving seed 0, one demonstration, converted to the tool frame by `contact_offset("panda")` |
| destination | `top_middle` shelf slot |
| scene seed | 0, **rebuilt fresh for every cell** and asserted reproducible |
| grasp source | real GraspGen-X candidates **for that hand**, from the cache, filtered to within 45° of the demonstrated approach; the planner's **full 6-DoF pose is used as-is** |
| cube half extent | 20 mm |
| control | `JOINT_POSITION` via `solve_ik`, 8 settle steps per waypoint |
| tool offset | `contact_offset(<this hand>)` — per hand, not the source's |

| varied | values |
|---|---|
| hand | yumi, xarm, panda, robotiq85, robotiq140, umi |
| object | cereal, milk, can, bread |
| construction | cloud box (control), grasp-pose cube |

Six hands spanning **24.3 to 117.2 mm of tool offset** and **50 to 125 mm of jaw
aperture**. 48 cells; **46 ran**, 2 refused (yumi/cereal, no candidate inside the
approach filter). Commit `e07c738`, `reproducible: True`.

### What each column means

| column | meaning |
|---|---|
| `reach` | fraction of the ~120 waypoints for which IK converged. Below ~50% the arm is being asked to go where it cannot |
| `track` | mean distance between the commanded fingertip pose and the measured one, in mm. Large values mean the arm did not get there even when IK said it could |
| `slip` | how far the object moved **relative to the hand** after first contact. A rigid grasp holds this near zero; it is the object sliding, rolling or being squeezed out |
| `held` | control steps during which any gripper geom was in contact with the object. **The direct measure of whether anything was gripped**, and the only one of these that was never misread |
| `shut@lift` | jaw closure at the first commanded close, 0 = fully open and 1 = fully shut on air, calibrated per hand. **Descriptive only** — see the caveat below |
| `place` | final horizontal distance from the object to its slot centre, in mm |
| `minDet`, `aim`, `orient` | carried through from the geometry so the two tiers can be read on one row |

> **A caveat on `shut@lift`, stated because it cost four wrong readings.** It is a
> single sample at one waypoint, and whether the jaws have begun moving at that
> instant depends on timing as much as on what is between them. The Robotiq
> 2F-140 spans 0.00 (can) to 0.84 (cereal) on the same hand. It is reported
> because it is cheap and occasionally illuminating; **no conclusion in this
> section rests on it.** `held` carries the contact claims. See §9.

### Result: success by hand

| hand | closing angle | aperture | tool offset | success | median `min det` |
|---|---|---|---|---|---|
| **panda** *(the source hand)* | 0° | 80 mm | 41.1 mm | **5/8** | 0.932 |
| yumi | 0° | 50 mm | 24.3 mm | 2/6 | 0.706 |
| robotiq85 | 0° | 85 mm | 47.8 mm | 1/8 | 0.667 |
| robotiq140 | 0° | 125 mm | 60.8 mm | 1/8 | 0.788 |
| xarm | −90° | 85 mm | 26.7 mm | **0/8** | 0.700 |
| umi | −90° | 80 mm | 117.2 mm | **0/8** | 0.684 |

**Only the hand the demonstration was recorded on works. 4 successes in 38 cells
on every other hand.**

### Result: the two constructions are tied on success and differ completely underneath

| | cloud box | grasp-pose cube |
|---|---|---|
| **successes** | **4/23** | **5/23** |

A one-cell difference. **The cube's panda-only advantage (2/4 against 3/4) does
not generalise**, and the earlier statement in this document that it does is
superseded by this table.

But the totals hide two entirely different failure profiles:

| stage the cell died at | cloud box | cube |
|---|---|---|
| path <50% reachable | 4 | 5 |
| reachable, but **never touched the object** | **11** | **3** |
| brief contact (<100 steps), then lost it | 5 | 5 |
| firm grip (≥100 steps) | **3** | **10** |
| → of those firm grips, succeeded | **3/3** | **5/10** |

**The cube acquires the object 3.3× more often (10 firm grips against 3) and then
drops half of them.** It converts an *aiming* problem into a *holding* problem.
Overall contact rate: cube 15/23 cells, cloud box 8/23.

---

### Every cell

`slip` is `nan` where the hand never made contact — there is no first-contact
reference to measure drift against, and a zero there would read as "held
perfectly".

| hand | object | construction | reach | track | slip | held | place | min det | aim | orient | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yumi | cereal | cloud box | — | — | — | — | — | — | — | — | **refused** |
| yumi | cereal | cube | — | — | — | — | — | — | — | — | **refused** |
| yumi | milk | cloud box | 60% | 29.3 | nan | 0 | 392.7 | 0.568 | 81.4 | 3.2 | no |
| yumi | milk | cube | 74% | 11.3 | 10.7 | 112 | 32.0 | 0.997 | 0.0 | 0.6 | **yes** |
| yumi | can | cloud box | 62% | 11.5 | nan | 0 | 352.2 | 0.660 | 47.6 | 5.0 | no |
| yumi | can | cube | 80% | 13.9 | 9.7 | 6 | 360.4 | 0.968 | 0.0 | 0.9 | no |
| yumi | bread | cloud box | 80% | 12.2 | nan | 0 | 322.9 | 0.555 | 43.9 | 6.8 | no |
| yumi | bread | cube | 62% | 24.8 | 34.8 | 115 | 18.0 | 0.753 | 0.0 | 0.8 | **yes** |
| xarm | cereal | cloud box | 62% | 13.1 | nan | 0 | 218.2 | 0.824 | 73.2 | 4.6 | no |
| xarm | cereal | cube | 84% | 4.7 | nan | 0 | 218.2 | 0.832 | 0.0 | 0.6 | no |
| xarm | milk | cloud box | 60% | 29.2 | nan | 0 | 392.6 | 0.575 | 92.7 | 27.6 | no |
| xarm | milk | cube | 87% | 9.0 | 57.9 | 111 | 143.2 | 0.954 | 0.0 | 0.9 | no |
| xarm | can | cloud box | 71% | 10.0 | 19.2 | 17 | 336.5 | 0.562 | 42.1 | 6.4 | no |
| xarm | can | cube | 78% | 11.4 | 46.6 | 22 | 355.5 | 0.868 | 0.0 | 1.3 | no |
| xarm | bread | cloud box | 100% | 3.7 | 18.6 | 22 | 313.1 | 0.347 | 24.1 | 10.7 | no |
| xarm | bread | cube | 100% | 3.8 | 50.1 | 110 | 236.8 | 0.399 | 0.0 | 1.6 | no |
| panda | cereal | cloud box | 72% | 11.6 | nan | 0 | 232.0 | 0.948 | 65.1 | 4.2 | no |
| panda | cereal | cube | 62% | 16.6 | 16.5 | 110 | 34.5 | 0.970 | 0.0 | 0.2 | **yes** |
| panda | milk | cloud box | 62% | 18.9 | nan | 0 | 392.6 | 0.584 | 70.1 | 7.8 | no |
| panda | milk | cube | 82% | 13.9 | 48.1 | 137 | 14.8 | 0.996 | 0.0 | 0.7 | **yes** |
| panda | can | cloud box | 86% | 13.5 | 7.6 | 82 | 29.6 | 0.414 | 40.2 | 3.2 | **yes** |
| panda | can | cube | 84% | 24.0 | 44.4 | 148 | 20.3 | 0.996 | 0.0 | 0.4 | **yes** |
| panda | bread | cloud box | 100% | 23.1 | 9.6 | 110 | 31.0 | 0.528 | 26.5 | 8.1 | **yes** |
| panda | bread | cube | 72% | 48.5 | 47.6 | 112 | 136.3 | 0.916 | 0.0 | 0.9 | no |
| robotiq85 | cereal | cloud box | 61% | 18.6 | nan | 0 | 222.4 | 1.054 | 82.8 | 12.1 | no |
| robotiq85 | cereal | cube | 100% | 5.7 | nan | 0 | 230.2 | 0.864 | 0.0 | 0.4 | no |
| robotiq85 | milk | cloud box | 60% | 26.6 | nan | 0 | 392.6 | 0.619 | 99.7 | 13.1 | no |
| robotiq85 | milk | cube | 96% | 9.0 | 31.7 | 114 | 68.5 | 0.983 | 0.0 | 0.8 | no |
| robotiq85 | can | cloud box | 57% | 16.6 | nan | 0 | 390.2 | 0.422 | 50.2 | 10.9 | no |
| robotiq85 | can | cube | 17% | 29.4 | nan | 0 | 488.3 | 0.714 | 0.0 | 1.4 | no |
| robotiq85 | bread | cloud box | 100% | 18.0 | 20.3 | 110 | 45.2 | 0.331 | 43.8 | 9.4 | **yes** |
| robotiq85 | bread | cube | 100% | 22.0 | 55.4 | 131 | 62.2 | 0.430 | 0.0 | 1.6 | no |
| robotiq140 | cereal | cloud box | 60% | 24.8 | nan | 0 | 273.4 | 0.785 | 121.0 | 5.8 | no |
| robotiq140 | cereal | cube | 97% | 12.0 | nan | 0 | 247.6 | 0.790 | 0.0 | 0.5 | no |
| robotiq140 | milk | cloud box | 60% | 33.6 | 27.2 | 65 | 238.0 | 0.578 | 130.4 | 4.5 | no |
| robotiq140 | milk | cube | 70% | 20.6 | 6.7 | 16 | 385.0 | 0.997 | 0.0 | 0.8 | no |
| robotiq140 | can | cloud box | 61% | 22.3 | 14.9 | 110 | 43.0 | 0.630 | 87.1 | 6.0 | **yes** |
| robotiq140 | can | cube | 76% | 28.1 | 50.4 | 22 | 375.4 | 0.806 | 0.0 | 1.3 | no |
| robotiq140 | bread | cloud box | 100% | 20.1 | 33.8 | 32 | 360.7 | 0.622 | 45.6 | 11.2 | no |
| robotiq140 | bread | cube | 100% | 33.6 | 36.3 | 45 | 311.8 | 0.814 | 0.0 | 1.5 | no |
| umi | cereal | cloud box | 30% | 43.7 | nan | 0 | 323.2 | 0.655 | 87.7 | 8.5 | no |
| umi | cereal | cube | 4% | 67.9 | nan | 0 | 436.1 | 0.712 | 0.0 | 1.4 | no |
| umi | milk | cloud box | 20% | 63.0 | nan | 0 | 445.8 | 0.591 | 98.7 | 16.3 | no |
| umi | milk | cube | 0% | 87.0 | nan | 0 | 512.0 | 0.988 | 0.0 | 0.6 | no |
| umi | can | cloud box | 24% | 80.1 | nan | 0 | 352.2 | 0.570 | 108.0 | 2.6 | no |
| umi | can | cube | 6% | 164.2 | nan | 0 | 359.1 | 0.997 | 0.0 | 0.2 | no |
| umi | bread | cloud box | 30% | 62.2 | nan | 0 | 269.1 | 0.521 | 51.7 | 10.4 | no |
| umi | bread | cube | 26% | 101.7 | nan | 0 | 220.4 | 0.907 | 0.0 | 0.9 | no |

*(yumi/cereal was refused before any physics: none of the 100 candidates for that
hand in that scene lay within 45° of the demonstrated approach. It is a refusal by
the grasp pipeline, not a failure of either construction, and is excluded from
every count.)*

---

### Failure analysis: four mechanisms, with the evidence for each

#### Mechanism 1 — the path is unreachable (9 cells, 0 succeeded)

Entirely the **UMI**, which fails this way in **8 of its 8 cells**, plus
robotiq85/can under the cube.

| hand | object | variant | reach | track |
|---|---|---|---|---|
| umi | cereal | cloud box / cube | 30% / **4%** | 43.7 / 67.9 mm |
| umi | milk | cloud box / cube | 20% / **0%** | 63.0 / 87.0 mm |
| umi | can | cloud box / cube | 24% / **6%** | 80.1 / **164.2** mm |
| umi | bread | cloud box / cube | 30% / 26% | 62.2 / 101.7 mm |

**Why.** The transported labels are a *fingertip* path. To execute one, IK must
solve for the **wrist**, which sits `contact_offset` behind the fingertips along
the hand's own approach — `wrist = target − R·offset` (`replay.py:153`). The
UMI's offset is **117.2 mm**, nearly three times the Panda's 41.1 mm and the
largest in the registry. So every waypoint asks the wrist to be 117 mm further
back than the Panda arm would need, and that pushes it outside the arm's
reachable envelope. The arm never gets near the object: `held` is 0 in all eight
cells.

This is **not** a keypoint failure and not a grasp failure. It is the frame
conversion doing exactly what it should, on a hand whose geometry the arm cannot
accommodate.

**Note the direction of the effect**: the cube is *worse* than the cloud box on
UMI reachability in all four objects (30→4, 20→0, 24→6, 30→26). That is
consistent and mechanistic — the cube aims *exactly* at the planned grasp, so it
inherits the full 117 mm setback, while the cloud box's 52–108 mm aim error
happens to pull the path somewhere the arm can more nearly reach. **Being right
about the target is a disadvantage when the target is unreachable.**

#### Mechanism 2 — the hand arrives and the object is not between the fingers (14 cells, 0 succeeded)

**Eleven of the fourteen are the cloud box**, and for those the cause is
measurable: it aims wrong.

| cloud-box cells | median `aim` | range |
|---|---|---|
| never touched the object (11) | **73.2 mm** | 43.9 – 121.0 |
| did touch it (8) | **42.9 mm** | 24.1 – 130.4 |

`aim` is `‖φ(source grasp point) − target grasp point‖`: how far the transported
plan passes from the grasp the planner chose. Nothing pins it for the cloud box,
because that box is centred on the object's **centroid** while the grasp is
somewhere on its surface. A plan that passes 73 mm from the intended grasp puts
the fingers 73 mm from the object, and a Panda jaw is 80 mm wide. **This is the
non-tautological half of Experiment G showing up in physics** — the cloud box's
aim error was a real measurement, and here is what it costs.

**The three cube cells in this category are the interesting ones, and they are
all the cereal**, on three different hands:

| hand | reach | track | `aim` | `orient` | `min det` | held |
|---|---|---|---|---|---|---|
| xarm | 84% | 4.7 mm | **0.0 mm** | 0.6° | 0.832 | **0** |
| robotiq85 | **100%** | **5.7 mm** | **0.0 mm** | **0.4°** | 0.864 | **0** |
| robotiq140 | 97% | 12.0 mm | **0.0 mm** | 0.5° | 0.790 | **0** |

Read the robotiq85 row carefully: **every quantity the map controls is perfect.**
The path is 100% reachable, tracked to 5.7 mm, aimed exactly at the planned grasp
point, oriented to within 0.4°, on a well-conditioned map. And the hand never
touches the cereal.

**This is the single clearest demonstration that matching TCP and orientation is
not sufficient for a grasp.** Something about the hand's physical interaction
with this object defeats a geometrically perfect plan on three different hands.

**The mechanism is not established.** The plausible candidate is that the hand's
*body* — not its fingertips — contacts the cereal during the approach and pushes
it away before the fingers close; the cereal is the tallest object in the scene
and the three hands involved are 145–270 mm deep. Testing that needs the object's
position trace during the approach, and **this run did not persist the probe
traces** (`replay_variant` extracts summary scalars only). That is a gap in the
instrumentation, not a finding, and it is the first thing to fix before the next
run.

#### Mechanism 3 — firm grip, then the object slips out (5 cells, all the cube)

Thirteen cells achieved a firm grip (≥100 contact steps). Eight succeeded. The
five that did not are **all the grasp-pose cube**, and slip is what separates
them:

| | n | median slip | range | median track | median place |
|---|---|---|---|---|---|
| firm grip, succeeded | 8 | **18.4 mm** | 9.6 – 48.1 | 20.2 mm | 31.5 mm |
| firm grip, failed | 5 | **50.1 mm** | 31.7 – 57.9 | **9.0 mm** | 136.3 mm |

| hand | object | held | slip | place | track |
|---|---|---|---|---|---|
| xarm | milk | 111 | 57.9 mm | 143.2 mm | 9.0 mm |
| xarm | bread | 110 | 50.1 mm | 236.8 mm | 3.8 mm |
| panda | bread | 112 | 47.6 mm | 136.3 mm | 48.5 mm |
| robotiq85 | milk | 114 | 31.7 mm | 68.5 mm | 9.0 mm |
| robotiq85 | bread | 131 | 55.4 mm | 62.2 mm | 22.0 mm |

**The tracking error is *lower* in the failures than in the successes (9.0 mm
against 20.2).** So this is not the arm failing to follow the path. The arm
follows it well, holds the object for over a hundred steps, and the object
rotates or slides out of the grip anyway.

**Why the cube specifically? Not established, and the obvious explanations do not
survive the data:**

- **Not map tilt.** `r(tilt_mid_path, slip) = +0.352` over 13 cells — weak — and
  the cube's median mid-path tilt among firm grips is **6.7°** against the cloud
  box's **10.8°**. The cube tilts the transit *less*.
- **Not conditioning.** `r(min_det, slip) = +0.071`.
- **Not tracking.** `r(tracking_error, slip) = +0.026`.
- `r(orientation_error, slip) = −0.521` looks like a signal but is a confound:
  the cube has both the low orientation errors and all five slips, so this is
  "cube against cloud box" re-expressed, not a mechanism.

The hypothesis worth testing next, and it is testable: **the cloud box's aim error
may be accidentally helping.** Its three firm grips succeeded 3/3 with aim errors
of 26.5, 43.8 and 87.1 mm — it grips somewhere *other* than the planned point,
plausibly nearer the object's centre of mass, where the gravity torque about the
grip is smaller. The cube grips exactly where the planner said, which may be a
pinch on a narrow face. Comparing the grip point to the object's centroid would
settle it, and needs data this run did not save. **n = 3 on the cloud-box side,
so this is a hypothesis, not a result.**

#### Mechanism 4 — the jaw is too narrow for the object (isolated, 1 clear cell)

`yumi / can / cube`: `held` 6 steps, `slip` 9.7 mm, placed 360 mm away. The jaws
met the can, could not close around it, and it escaped.

The arithmetic: the yumi's aperture is **50 mm**; the can measures **45.2 mm**
across this grasp's closing axis (from the cloud box's own measurement, since a
fixed cube cannot report an object's width). The budget is
`(50 − 45.2) / 2 = 2.4 mm` per side. Every other hand has 10–40 mm:

| hand | aperture | bread | can | cereal | milk |
|---|---|---|---|---|---|
| **yumi** | 50 mm | **3.2** | **2.4** | **1.9** | **−5.2** |
| rethink | 66 mm | 11.2 | 10.4 | 9.9 | 2.8 |
| panda / umi / inspire | 80 mm | 18.2 | 17.4 | 16.9 | 9.8 |
| robotiq85 / xarm | 85 mm | 20.7 | 19.9 | 19.4 | 12.3 |
| robotiq3f | 110 mm | 33.2 | 32.4 | 31.9 | 24.8 |
| robotiq140 | 125 mm | 40.7 | 39.9 | 39.4 | 32.3 |

*(closing budget in mm, `(aperture − object width)/2`)*

**But aperture does not explain the run as a whole.** Success against aperture is
not monotone: the two 80 mm hands are 5/8 (panda) and 0/8 (umi). It is a real
constraint that the keypoints cannot see, and it explains this cell, and it is not
the driver of the overall result.

---

### What Tier 2 does to Experiment G's status

Experiment G is not wrong, but this tier bounds what it licenses.

**The geometry does not predict the physics.** Across the 23 cube cells:

| correlation | value |
|---|---|
| `r(min det, held steps)` | **−0.071** |
| `r(aim, held steps)` | **+0.015** |
| `r(orientation error, held steps)` | −0.027 |
| `r(tilt mid-path, held steps)` | +0.106 |

All approximately zero. The metrics that rank the constructions cleanly and
hand-independently in Tier 1 carry **no information** about whether the hand ends
up holding the object.

This also confirms a criticism of Tier 1 that should have been stated there:
**for the cube variants `aim` is ~0 and `orient` is ~0 by construction.** The
cube's centre *is* the target grasp point and `φ` interpolates keypoints exactly,
so the aim is guaranteed; the corners are laid out in the source and target grasp
frames, so `J_perp` is being asked to recover a rotation that was built into the
keypoints, and it very nearly does — that is a self-consistency check on the cube
size (which is why size moved it from 0.1° to 20.6°), not a fact about the world.

**What in Tier 1 remains genuinely informative:** `min det(J)` (nothing pins it,
and the cloud box really does fold in configurations where the cube does not),
the mid-path tilt and lift deviation (far from any keypoint), **every cloud-box
number** (nothing pins those either, which is why the aim comparison above is
meaningful), and the composed variant's 10.9 mm keypoint residual, which is a
genuine violation of property (i).

**So the correct reading of the two tiers together is:** the transportation map
is sound and hand-independent, and that is necessary but nowhere near sufficient.
Execution is limited by things the map does not model — the hand's setback from
the labels, its physical bulk during approach, its jaw width, and the stability
of the grip it achieves.

---

### What would make Experiment H's conclusions false

- **One seed, one slot, one demonstration.** Every cell is scene seed 0 placing
  into `top_middle` from a single top-down reshelving demonstration. The
  hand-versus-object confound cannot be separated without more scenes.
- **The Panda advantage may be a grasp-selection artefact.** The demonstration
  was recorded on a Panda, and candidates are filtered to within 45° of *its*
  approach. The Panda's median `min det` is 0.932 against 0.667–0.788 for every
  other hand, so the Panda may simply be getting better target grasps rather than
  executing better. Testing this needs a demonstration recorded on a different
  hand — which the project does not currently have.
- **Mechanisms 2 and 3 are unexplained.** The cereal never-touched cases and the
  cube's slips both need the per-waypoint object trace, which this run did not
  persist. Until that exists, those are described failure *modes*, not diagnosed
  *causes*.
- **`n` is small everywhere.** 3 cloud-box firm grips, 5 cube slips, 1 clear
  aperture case. None of the per-mechanism claims would survive a demand for
  statistical significance, and they are not offered as such.
- **Replay is not the policy.** These are upper bounds under position control.
  The GP policy has to follow the same paths with an impedance controller and a
  lag gate, and will do worse.

---

## 8f. Experiment I — the control Tier 2 never had: execute the grasp without transporting anything

**Question.** Every number in §8d and §8e compares one transported plan against
another. Neither says whether transportation helps or hurts, because there is no
untransported baseline. If you sampled a GraspGen-X pose and simply commanded a
robot to go there, close and lift, would it work? If not, the grasp was bad and
nothing downstream means anything.

**Method.** For each (hand, object) cell of Tier 2, take the **identical** cached
candidate — same scene, same approach filter, same rank — and drive a minimal
path through it: back off 120 mm along the grasp's **own approach axis**, descend
to the grasp over 20 waypoints with the jaws open, hold for the dwell with them
closing, lift 150 mm. A lift over 50 mm counts.

**The executor is the same one Tier 2 used** — `replay_labels`, IK per waypoint
under stiff joint-position control, the same `contact_offset` — so the *only*
difference from a Tier 2 cell is the path: the planner's grasp reached directly,
against the transported demonstration warped onto it.

> **A first version of this control was wrong and its numbers are discarded.** It
> used `verify.execute_grasp`, which drives with the `CartesianImpedanceController`
> while Tier 2 used position control. That changes two things at once, and it
> showed: `panda/can` failed the "control" while succeeding in both transported
> conditions, which is backwards.

> **The dwell is measured, not chosen.** The source demonstration holds
> *perfectly still* for exactly **15 waypoints** after its gripper command goes
> `+1` — displacement 0.0 mm fifteen times, then 6.4, 12.8, 19.2 mm — so the
> transported path carries a 15-waypoint dwell and the control carries the same
> one. This is not a free parameter: swept on the Panda, the can lifts at dwell
> 3 (130.7 mm), 5 (133.6 mm) and 10 (139.3 mm) and is **squeezed out at 25**
> (0.1 mm). Choosing 25, as the first version did, would have manufactured a
> failure.

### Result: 9 of 23 grasps lift with no transportation at all

| hand | direct lift rate | Tier 2 cloud box | Tier 2 cube |
|---|---|---|---|
| panda | 3/4 | 2/4 | 3/4 |
| robotiq85 | 3/4 | 1/4 | 0/4 |
| yumi | 2/3 | 0/3 | 2/3 |
| xarm | 1/4 | 0/4 | 0/4 |
| **robotiq140** | **0/4** | 1/4 | 0/4 |
| **umi** | **0/4** | 0/4 | 0/4 |
| **total** | **9/23** | 4/23 | 5/23 |

**Fourteen of twenty-three GraspGen-X grasps cannot be executed even when the arm
is driven straight to the pose along the grasp's own approach axis.** Every cell
was 100% reachable — the direct path is short and near the object — so this is
grasp quality alone, with no kinematics and no transport in it.

### Per-cell attribution

| reading | n | cells |
|---|---|---|
| **the grasp is bad — transport exonerated** | **12** | robotiq140 x3, umi x4, xarm x3, robotiq85/cereal, yumi/can |
| works | 6 | panda x3, robotiq85/bread, yumi x2 |
| **transport broke it** | **3** | robotiq85/can, robotiq85/milk, xarm/can |
| transport *helped* a bad grasp | 2 | panda/can, robotiq140/can |

**Of the 17 failures §8e attributed to keypoints and transport, 12 were bad
grasps and only 3 are transport's.** The Tier 2 failure analysis was measuring
grasp quality with a transportation-shaped ruler.

### The planner's own confidence is anti-predictive

| | n | score median | range |
|---|---|---|---|
| grasps that lifted | 9 | **0.577** | 0.462 – 0.867 |
| grasps that did not | 14 | **0.769** | 0.573 – 0.917 |

**`r(score, lifted) = −0.529`.** Higher confidence predicts *failure*. The
Robotiq 2F-140's four candidates score 0.871, 0.917, 0.871 and 0.573 and **none
of them lifts anything**; the Panda's score 0.867, 0.462, 0.660, 0.560 and three
of four work.

That single fact explains the Robotiq 2F-140's 0/4 here and its 1/8 in §8e with
no reference to transportation. GraspGen-X conditions on a hand's swept volume
but its discriminator was trained on its own gripper set; on this registry it
cannot be used to rank candidates. **Grasp selection currently has no working
quality signal.**

### The objects are being knocked over, and it is not the map

The failures are not gentle misses. Signed lift and total displacement:

| cell | lift | object moved | held |
|---|---|---|---|
| robotiq140 / cereal | **−71.2 mm** | 112.5 mm | 0 |
| xarm / cereal | **−67.5 mm** | 93.4 mm | **0** |
| umi / cereal | **−63.5 mm** | 114.7 mm | 16 |
| robotiq140 / milk | −42.2 mm | **270.4 mm** | 0 |
| xarm / milk | −42.7 mm | 93.3 mm | 0 |

A negative lift means the object ended up **lower than it started** — knocked off
its support. With **no transportation anywhere in the loop**, on a path that
descends along the grasp's own approach axis.

**This is the mechanism §8e could not explain, and it is not a transport
failure.** It also is not, on the face of it, a grasp-*generation* failure: a
hand descending along its own approach axis onto a correctly-placed grasp should
never strike the object. Something upstream of both is wrong.

**`xarm/cereal` is the sharpest case and it rules out the obvious explanation.**
`held = 0` — no gripper *collision* geom ever touched the cereal — while the
cereal moved 93.4 mm and fell 67.5 mm. So the object was displaced by something
that is not a gripper contact geom: an arm link, a gripper geom outside the
collision set (`contact_geoms` covers 4 of the xarm's 11 geoms, the rest being
visual), or another object toppling into it.

**The leading hypothesis is partial observability, and it is untested.** The
grasp is planned on a cloud assembled from three cameras, so it describes only
the surfaces those cameras see. `by_collision` then checks the approach corridor
against a *scene* cloud with the same limitation. A grasp whose corridor is clear
of every observed point can still be blocked by a surface nobody observed. That
would explain why `xarm/cereal` passes all seven filters and still knocks the
cereal 93 mm.

**What would test it**, cheaply and without physics: compare each object's true
extent, read from MuJoCo's own body and geom data, against the extent of the
cloud the cameras produced. A systematic shortfall on the unobserved side is the
mechanism; no shortfall falsifies it. This has not been run.

### What this does to §8d and §8e

- **§8e's absolute success rates are not measurements of transportation.** They
  are measurements of a pipeline in which most commanded grasps do not work.
- **The construction comparison survives but on a much smaller sample.** Both
  constructions received identical grasps, so cloud box against cube is still
  internally valid — but it is only interpretable on the **9 cells where the
  grasp works at all**, not 46.
- **§8e's Mechanism 2 is reassigned.** "The hand arrives and the object is not
  between the fingers" is not a keypoint or transport failure; it happens without
  either.
- **§8e's Mechanism 1 stands, and is now separable.** The UMI fails direct
  execution 0/4 *at 100% reachability*, so its bad grasps and its unreachable
  transported path are **two independent defects**, which §8e conflated.

### Consequences for what to build

**A centre-of-mass grasp filter addresses only half of one mechanism.** It scores
grip *stability*, which is the right treatment for §8e's Mechanism 3 (firm grip,
then slip). It does nothing for the twelve bad grasps here, most of which fail by
knocking the object over before any grip exists.

The ordering that follows from this table is:

1. **Test the partial-cloud hypothesis** (cheap, no physics). It is upstream of
   everything: if the clouds under-describe the objects, then the planner, the
   collision filter and the keypoint box are all working from bad geometry.
2. **Find a grasp quality signal that works.** The planner's score is
   anti-correlated with success, so candidate ranking currently has nothing
   trustworthy behind it. Direct execution is the ground truth and is cheap
   (~2 min per cell) — it can label a candidate set to test any proposed
   surrogate.
3. **Then** the stability filter, on grasps already known to be executable.

---

## 8g. Experiment J — Tier 1, re-run on the corrected scene

**Supersedes Experiment G (§8d).** Everything there was measured in a scene whose
objects were still falling and whose contact offsets were wrong by up to 30 mm;
see the banner at the top of this document and `ROBOTICS_NOTES.md` §7.32.

**What changed since §8d**, all of it upstream of the numbers:

* objects now settle before handover, so the cloud describes a pose the object
  holds — previously a 150 mm cereal box toppled 35–78 mm *after* capture;
* the arm starts clear of the sampling region, so no object is created inside it
  (interpenetration 4.85–26.77 mm on four of six hands, now 0.00 everywhere);
* the scene is **identical across grippers** — it differed by up to 154 mm;
* every `contact_offset` was re-derived, moving up to 30 mm. The Panda's is
  **11.1 mm**, not the 41.1 mm quoted throughout §§4–8f;
* the grasp cache was emptied, so every candidate is planned fresh.

`outputs/keypoints_grippers_v2/`, commit `01533e3`, `reproducible: True`.
153 cells, **136 scored**, nine hands × five objects × four constructions.

### Result

| construction | `min det` median, range across hands | orientation median | orientation **worst cell** | keypoint residual |
|---|---|---|---|---|
| 0 cloud box | 0.507 – 0.745 | 1.7 – 4.2° | **14.6°** | 0.1 µm |
| 1 task-frame cube | 0.776 – 0.970 | 2.2 – 4.1° | 15.0° | 0.9 µm |
| **2 grasp-pose cube** | **0.774 – 0.960** | **0.45 – 1.41°** | **1.7°** | 0.9 µm |
| 3 composed | 0.551 – 0.778 | 0.6 – 2.0° | 5.2° | **~10 000 µm** |

### Per hand, construction 2, ordered by tool offset

| hand | tool offset | aperture | `min det` med | `min det` min | orient med | orient max |
|---|---|---|---|---|---|---|
| rethink | 5.1 mm | 66 mm | 0.940 | 0.601 | 0.6° | 1.6° |
| yumi | 9.3 | 50 | 0.936 | 0.838 | 0.5° | 1.5° |
| panda | 11.1 | 80 | 0.774 | 0.596 | 1.3° | 1.6° |
| xarm | 19.2 | 85 | 0.892 | 0.603 | 1.1° | 1.5° |
| robotiq85 | 32.8 | 85 | 0.858 | 0.639 | 1.4° | 1.7° |
| robotiq3f | 36.4 | 110 | 0.878 | 0.658 | 1.2° | 1.7° |
| robotiq140 | 38.3 | 125 | 0.948 | 0.573 | 0.6° | 1.6° |
| inspire | 134.4 | 80 | 0.882 | 0.730 | 1.4° | 1.7° |
| umi | 146.5 | 80 | 0.960 | 0.589 | 0.9° | 1.5° |

### What it says

**The construction is hand-independent, and the conclusion survives the scene
fix.** Per-hand median orientation error runs **0.45° to 1.41°** with a worst
single cell of **1.7°**, across a tool-offset range of **5.1 to 146.5 mm** — a
29× spread, wider than §8d could test because the offsets themselves changed.
The cloud box on the same grasps and objects runs 1.7° to 4.2° and reaches
**14.6°**. Spread across hands: sd **0.338°** for the cube against **0.822°**.

**Composition is dead, and this run proves it in a way §8d could not.** Median
keypoint residual **~10 000 µm — 10 mm** — violates property (i) of Sec. III-D
outright. And `robotiq3f` records `min det = −0.045`: an actual **fold**, which
the withdrawn data never produced. A map that misses its own keypoints by a
centimetre and turns inside out has nothing left to offer.

**The cloud box's aim error is the one number here that predicts anything**:
36.9 to 94.4 mm across hands, on a scene where nothing is being disturbed.
Nothing pins it, because that box is centred on the object's centroid while the
grasp is on its surface.

### What it does *not* say — and this is the important part

`aim = 0.0 mm` for all three cube variants is **true by construction**: the
cube's centre *is* the target grasp point, and `φ` interpolates its keypoints
exactly. It cannot be evidence for the construction.

The orientation figure is **nearly** as circular. The corners are laid out in the
source grasp frame and the target grasp frame respectively, so `J_perp` is asked
to recover a rotation that was built into the keypoints. It is not exactly zero
only because `J_perp` is a *derivative*, estimated over the finite stencil the
corners provide — which is why cube size moved it from 0.1° to 20.6° (§10). It
is a self-consistency check on the cube size, not a fact about the world.

**And measured against physics, none of these metrics predicts contact.** From
the withdrawn Tier 2, `r(min det, held steps) = −0.071` and
`r(aim, held steps) = +0.015`. That correlation was computed on a broken scene
and needs re-deriving, but the direction of the concern stands: **a
well-conditioned, exactly-aimed map is necessary and nowhere near sufficient.**

So what Experiment J licenses is narrow and worth stating plainly: the map is
valid, exact at its keypoints, and does not depend on which hand it is built
for. Whether the hand then grasps, carries and places the object is a different
question, and §§8h and 8i are where it is asked.

---

## 8h. Experiment K — the grasp-quality control, on the corrected scene

**Supersedes Experiment I (§8f).** Same question, same method, same code path: take
the GraspGen-X candidate each cell would use, drive the arm straight to it along
the grasp's own approach axis, close for the demonstration's own 15-waypoint
dwell, lift 150 mm. **No map anywhere in the loop.** The only difference from
§8f is the scene, which now settles its objects and starts the arm clear of them
(`ROBOTICS_NOTES.md` §7.32), and the contact offsets, which moved up to 30 mm.

Five hands x four objects. Grasps planned fresh — the cache was emptied, so no
candidate is carried over from the broken scene.

### Result

**19 of 20 grasps lift.** The one failure is `xarm/cereal` at −63.9 mm.

| hand | lifted |
|---|---|
| yumi | 4/4 |
| xarm | 3/4 |
| panda | 4/4 |
| robotiq85 | 4/4 |
| robotiq140 | 4/4 |

Lift heights among the successes run **122 to 167 mm** against a 50 mm threshold,
so these are not marginal — the hand takes the object and raises it most of the
commanded 150 mm.

### What this retires

§8f reported **9 of 23** and concluded from it that most of GraspGen-X's grasps
could not be executed. That was wrong, and the difference is entirely the scene:

| §8f claimed | this run measures |
|---|---|
| 14 of 23 grasps unexecutable | **19 of 20 lift** |
| "the grasp is bad — transport exonerated", 12 cells | those objects were toppling or being ejected before the hand arrived |
| `r(score, lifted) = −0.529`, the planner's discriminator anti-predictive | **no variance left to correlate** — 19 of 20 lift, and the successes span scores 0.416 to 0.903 with a median of 0.621, while the single failure scores 0.604, squarely inside that range |
| a centre-of-mass filter is justified by bad grasps | **premise false** |

**GraspGen-X's grasps are sound for these hands and objects.** The entire
"grasp quality" thread of §8f was measuring scene damage.

The score result deserves a plain statement because it was a strong claim about
an external tool: **there is now no evidence that GraspGen-X's confidence is
anti-predictive, and no evidence that it is predictive either.** With one failure
in twenty there is nothing to correlate against. The honest position is that this
run cannot rank candidates by score, not that the score is bad.

### One anomaly, recorded because it changes how another metric should be read

Three cells report **0% reachable** and lift the object anyway — `panda/bread`,
`robotiq85/bread`, `robotiq140/milk`, at 19 to 25 mm of tracking error. IK
declares the direct path unreachable while the arm executes it successfully. So
`reachable_fraction` is **stricter than physical feasibility** and must not be
used as a failure predictor; it says IK did not converge to tolerance, not that
the arm cannot get there.

---

## 8i. Experiment L — Tier 2 on the corrected scene: grasped, traversed, placed

**Supersedes Experiment H (§8e).** The transported path followed pose by pose
under stiff position control — no policy, no attractor integration, no lag gate —
so a failure is the plan's rather than the executor's.

Five hands (the UMI is excluded, §7.32) x four objects x two constructions.
**All 40 cells ran**, zero precondition refusals. `outputs/keypoint_replay_v2/`,
commit `01db9d8`, `reproducible: True`.

**Measured as grasped / traversed / placed, not aim and orientation.** For the
cube constructions aim is 0 and orientation near 0 *by construction* (§8g), and
they carry no information about whether the hand ends up holding anything.

### Success

| hand | cloud box | cube |
|---|---|---|
| yumi | **4/4** | 2/4 |
| xarm | 1/4 | 1/4 |
| panda | **0/4** | **3/4** |
| robotiq85 | **4/4** | **4/4** |
| robotiq140 | 1/4 | 2/4 |
| **total** | **10/20** | **12/20** |

Against §8e's 4/23 and 5/23 on the broken scene: success roughly **doubles**,
from about 20% to 50-60%. The `robotiq85` goes **8 of 8** across both
constructions where it previously managed 1 of 8.

### The three stages

| | grasped | traversed | placed |
|---|---|---|---|
| cloud box | 12/20 | 12/20 | **10/20** |
| **grasp-pose cube** | **18/20** | **16/20** | **12/20** |

`grasped` requires contact in the closing window **and** the object rising more
than 20 mm above its resting height — the only unambiguous evidence the hand
bears its weight rather than brushing it. `traversed` requires that plus contact
through more than half the carry and no permanent loss. `placed` is the task
score.

**The cube grasps far more reliably: 18 of 20 against 12 of 20.** That is the aim
advantage of §8g arriving in physics — a plan that passes through the grasp point
puts the fingers on the object, and one that misses by 37 to 94 mm does not.

**But it converts worse.** Of the cells that grasped, the cloud box places **10
of 12 (83%)** and the cube **12 of 18 (67%)**. The cube acquires more and loses
more. That shape survived from §8e, which is notable given everything else about
§8e did not.
### Why the eight grasped-but-not-placed cells fail

> **Diagnosed after the fact, and the diagnosis is in §8j.** This section
> originally offered four "mechanisms" reasoning backwards from the symptoms.
> All four are withdrawn. The real causes were three coordinate-frame defects,
> found by working forwards from the geometry and confirmed by re-running these
> same cells with the defects fixed and the grasp selection held identical.

| hand | object | variant | held through carry | lost at | slip | placement |
|---|---|---|---|---|---|---|
| yumi | cereal | cube | **1.00** | never | 43 mm | 67 mm |
| yumi | bread | cube | **1.00** | never | 48 mm | 15 mm |
| panda | cereal | cube | **1.00** | never | 34 mm | 76 mm |
| robotiq140 | milk | cloud box | **1.00** | never | 46 mm | **344 mm** |
| robotiq140 | milk | cube | **1.00** | never | 27 mm | **275 mm** |
| robotiq140 | can | cloud box | **1.00** | never | 10 mm | **213 mm** |
| xarm | milk | cube | 0.78 | wp 135 | 29 mm | 268 mm |
| xarm | bread | cube | 0.09 | wp 60 | 3 mm | 352 mm |

**Six of eight never lose the object at all.** They hold it through the whole
carry and set it down in the wrong place. That shape is what pointed the
investigation at the *plan* rather than at the grasp.

### The three defects that caused them

Full mechanism in `ROBOTICS_NOTES.md` §7.33 and §7.34. In short, the code was
doing geometry in two different coordinate systems and treating them as one.

A **frame** here means three arrows glued to the gripper: the **approach** (the
direction the hand drives in along), the **closing axis** (the direction the two
fingers travel as they squeeze), and a third at right angles to both. Three such
systems are in play, and two of them were being confused:

* the **grasp convention**, which is GraspGen-X's: `+Z` is the approach and
  `+X` is the closing direction. It means the same thing on all nine hands,
  which is why geometry that has to work across embodiments belongs in it;
* the **wrist convention**, each gripper model's own `grip_site`. Its
  orientation relative to the fingers was chosen by whoever authored that model,
  and it is *not* the same across hands. Inverse kinematics aims this one;
* the **task frame**, support normal plus closing axis, used to fit a box to a
  point cloud and snap it onto a surface.

**Defect A — the source's own pick and place frames were 180 degrees apart.**
The closing axis has two equally valid directions, because a two-finger hand
closing left-to-right and right-to-left performs the same squeeze. `task_frame`
chose between them with a world-axis rule — keep the arrow if its `y` component
is positive, flip it if negative, and if `y` is zero decide on `x` instead — run
independently at each of four places. The demonstration **places its object
square with the shelf**, which puts the placed closing axis exactly along world
`x` at `[1, -1e-17, 0]`. Its `y` component is a rounding crumb, so the tie-break
fired and answered the opposite way to the pick, whose `y` is an unambiguous
`-0.581`.

Once the fingers close the object is rigid with the hand, so where it lands is
decided entirely by how far the hand turns between closing and opening. A half
turn taken at one end and not the other **reflects the object through the point
where the fingers grip it**: hold it dead centre and nothing moves, hold it `d`
off-centre and it lands `2d` away. Grips here sat 2 to 50 mm off centre.

**Defect B — the plan was commanded in the wrong frame.** The keypoint cubes are
built in the grasp convention; inverse kinematics aims the wrist. Neither
`alignment_rotation` nor `grasp_to_eef_pose` appeared anywhere in this driver, so
no conversion happened. The error is each hand's own offset: 0.2° for the two
Robotiqs, 90° for the XArm, 180° for the Panda, Yumi and Rethink.

**Defect C — `flip_target` was inert.** The sign resolution ran after it and put
the sign back, so both branches produced identical keypoints. Measured on the old
code across a target-yaw sweep, the two branches returned the same `min det` to
three decimals at all eight yaws. `pipeline._choose_grasp` loops over both rolls
and keeps whichever scores better; it has been scoring one option twice.

### Why nothing caught them, which matters more than the defects

Three instruments looked straight at this and read clean, and **none of them was
broken**:

1. **The map stayed a valid warp** — `min det(J)` 0.44 to 1.00, keypoint
   residual about 1e-7, property (ii) of Sec. III-D satisfied on every cell. It
   has to: a consistently mislabelled coordinate system is still a coordinate
   system.
2. **The aim was exactly 0.00 mm** at both the pick and the release. This is the
   cruel one. A half turn about the approach axis leaves the grasp point
   **fixed** — it is the one point a reflection does not move — so every aim
   metric is blind to it by construction, not by accident.
3. **`orientation_transport_error` minimises over `JAW_SYMMETRY`**, which *is*
   this half turn, so it reported 0.2 to 4.5 degrees for a plan that was 179
   degrees wrong. The default is `symmetric=True` and `transport()` calls it
   bare, so every orientation number in §§5-8g was measured with the error
   forgiven.

That third one is defensible for a *single grasp* and wrong for a *journey*. A
half turn taken at **both** ends genuinely cancels — it is the same task done
with the wrist rolled over — while one taken at a single end does not, and
minimising each end separately cannot tell those apart.
`metrics.transport.carry_orientation_error` compares the *change* in angle from
grasp to release instead, where a consistent roll cancels out of the subtraction
and an inconsistent one does not. It reads 178.5 to 180.0 degrees on these cells.

### Per cell, against the re-run

Confirmed by §8j, which re-ran these cells with the defects fixed and the grasp
selection held identical:

| hand | object | recorded | after the fixes | verdict |
|---|---|---|---|---|
| yumi | cereal | 66.9 mm ✗ | **14.4 mm ✓** | Defect A — predicted 70.4 mm before the run |
| panda | cereal | 76.1 mm ✗ | **14.2 mm ✓** | Defect A — predicted 85.7 mm before the run |
| yumi | bread | stuck in the hand ✗ | **16.4 mm ✓** | the frame defects |
| xarm | milk | 268 mm ✗ | **8.4 mm ✓** | the frame defects |
| xarm | bread | 352 mm ✗ | **56.9 mm ✓** | the frame defects |
| robotiq140 | milk | 275 mm ✗ | 110 mm ✗ | improves, still fails |
| robotiq140 | can | 200 mm ✗ | 217 mm ✗ | unexplained |
| xarm | cereal | 414 mm ✗ | 218 mm ✗ | the grasp does not hold — see below |

**`xarm/cereal` is the only one of the eight that is not a frame defect.** It
never lifts the object — 7 mm of rise, 9 control steps of contact — and it is
also Experiment K's single failure, where no map is in the loop at all. It fails
under every protocol tried, which is what a genuinely bad grasp looks like.

### Does a centre-of-mass stability filter have a target?

**No, not on this evidence.** Such a filter scores whether a grasp will *hold*
during a carry. Exactly **2 of 40 cells** lose the object mid-carry, and both are
the xarm. The dominant failure mode — six of eight — is carry-and-misplace with
the grip completely intact, which a stability criterion cannot address.

The open question this leaves is a different one: **why does a plan that grips
the object and carries it faithfully release it in the wrong place?** That is
about where the transported path *ends*, not about the grasp, and it is the
natural successor to this experiment.

---

## 8j. Experiment M — Tier 2 with the three frame defects fixed

**The before-and-after for §8i.** Same five hands, same four objects, same two
constructions, same seed, **and the same grasp selection**, so the only thing
that differs from Experiment L is the three defects. `outputs/keypoint_replay_v4/`,
commit `6fb83d4`.

Holding the grasp selection fixed is not a detail. An earlier attempt at this
run changed the filters at the same time and had to be thrown away; that is
§8k.

### What was changed

Described in full in `ROBOTICS_NOTES.md` §7.33-§7.34 and summarised in §8i. One
sentence each:

* the closing axis's sign is now **taken from the grasp** rather than re-derived
  from a world axis, and the source grasp is derived from the demonstrating
  *hand* rather than from the product's body axis, which is where the undefined
  sign entered;
* the plan is converted **out of** the grasp convention and **into** the
  executing hand's wrist convention, once at each end, by that hand's measured
  `alignment_rotation`;
* `flip_target` works, so a caller can genuinely try both wrist rolls.

One thing had to be added rather than removed. Deleting the sign resolution lost
a second job it was quietly doing — keeping the rotation between the source and
target frames small — and three of twenty maps folded as a result. `min det`
tracks that angle closely:

| source-to-target frame rotation | `min det` | `min det`, other roll |
|---|---|---|
| 3 - 19° | 0.94 - 0.99 | 0.006 - 0.08 |
| 74 - 135° | 0.33 - 0.79 | 0.46 - 0.99 |
| 149 - 179° | **-0.06 to 0.16** | 0.93 - 1.00 |

So the target's roll is now chosen **once, on the grasp**, by agreement with the
demonstration, before either target block is derived. That is a different thing
from the rule it replaces: the old one ran four times and asked "does this point
along world `+y`", a question about the object's yaw with nothing to do with the
gripper; this asks "does this grip the object the way the demonstration gripped
its object".

### Result

| | Experiment M | Experiment L | change |
|---|---|---|---|
| **grasp-pose cube placed** | **15/20** | 12/20 | **+3** |
| cube traversed | 18/20 | 16/20 | +2 |
| cube grasped | 18/20 | 18/20 | 0 |
| cloud box placed | 10/20 | 10/20 | 0 |
| cloud box grasped | 14/20 | 12/20 | +2 |

Five cube cells recovered, two regressed. Per hand, grasp-pose cube:

| hand | Experiment M | Experiment L |
|---|---|---|
| yumi | **4/4** | 2/4 |
| panda | **4/4** | 3/4 |
| xarm | **3/4** | 1/4 |
| robotiq85 | 3/4 | 4/4 |
| robotiq140 | 1/4 | 2/4 |

### The two predictions that were registered before the run

These are the only two cells where the mechanism gave a *number* in advance, so
they are the ones that carry evidential weight rather than merely being
consistent:

| cell | predicted from the defect | before | after |
|---|---|---|---|
| yumi/cereal | 70.4 mm | 66.9 mm ✗ | **14.4 mm ✓** |
| panda/cereal | 85.7 mm | 76.1 mm ✗ | **14.2 mm ✓** |

The prediction is `2 x (lateral grasp offset)`: the half turn reflects the object
through the grip point, so a grip 35.5 mm off centre on the yumi's cereal swings
the object 71 mm, past the 60 mm pass mark.

### Two claims this retires

**"The xarm's grip does not hold" is withdrawn.** The hand went from 1/4 to 3/4:
`xarm/milk` from 268 mm to **8.4 mm** and `xarm/bread` from 352 mm to 56.9 mm.
Those two failures were the frame defects.

**And the direct-execution control that appeared to corroborate it was not
independent.** It ran on the unfixed code and commanded the grasp rotation
straight to the wrist, so it contained Defect B — the very error it was being
used to test for. When it failed on those two cells it was reproducing the bug.
Only its *successes* were ever evidence.

### Nothing else moved

* the reshelving 17/20 regression gate is **bit-identical**: the same placement
  errors to 0.1 mm on all ten seeds and the same single failure on seed 8;
* the cloud box's four Panda cells are bit-identical to Experiment L, which is
  the check that paths expected to be untouched really are;
* 590 tests pass, 25 of them new.

### Why the two regressions are not a new defect

`robotiq140/bread` went from 54.3 mm to 235 mm and `robotiq85/milk` from 8.1 mm
to 62.3. Both are Robotiq hands and both are release-phase failures, and the
underlying weakness is present in **both** runs equally:

| | Experiment L | Experiment M |
|---|---|---|
| object released above the shelf board | median 44.4 mm, max 129 | median 39.6 mm, max 153 |
| tracking error at the release waypoint | median 69.4 mm, max 133 | median 51.8 mm, max 130 |

**Every placement in this experiment is a drop, not a set-down.** The arm does
not complete the descent into the shelf — it stops 13 to 130 mm short of the
commanded release pose — so the object is let go from up to 15 cm above the
board and falls. Whether it stays on the board is then decided by the bounce.

That release height does **not** separate success from failure: placed cells sit
a median 37 mm above the board and failed cells 40 mm, and `xarm/bread` was
released **153 mm** high and placed successfully. So it is a systemic weakness
with a partly random outcome rather than a cause specific to any cell.

The two regressions are that lottery resampled. `robotiq140/bread` was released
22 mm higher than before (150 against 128 mm), bounced off the board and fell
435 mm to the table. `robotiq85/milk` was released *lower* than before (21
against 52 mm) but the object moved **81.9 mm laterally after the jaws opened**,
against 40.1 mm before — and the sibling project measured exactly this
independently: a Robotiq 2F-85's pads rotate inward as they open, so an object
set down over a surface can catch on the opening fingers
(`6dof_GraspMAS/docs/simulation.md`, "A release that does not release").

**The real finding here is the drop itself**, which no previous experiment had
looked at, and it is now the largest known execution weakness.

---

## 8k. Experiment N — the full grasp funnel, and why it cannot be read

**Recorded so the mistake is not repeated, not because it measured anything.**
`outputs/keypoint_replay_v3_FULLFILTER_CONFOUNDED/`, commit `9894bd6`.

### What was asked, and what was actually changed

Experiment L applied **one** of the seven grasp filter stages — the 45 degree
approach test. The other six, including the reachability check that tests the
pick, the lift, the placement and the approach corridors into both, never ran.
So it measured candidates the pipeline itself would have thrown away. Running
the whole funnel is plainly the right thing.

The problem is that `filter_grasps` changes the **ranking** as well as the
filtering, and this was not noticed before the run:

* the approach path keeps candidates within 45 degrees of the demonstration and
  takes the **closest**;
* `filter_grasps` keeps the same candidates and returns them ranked by
  **GraspGen-X's own confidence score**.

Measured across the 20 cells:

| | approach only | full funnel |
|---|---|---|
| approach mismatch of the chosen grasp | median **3.2°** | median **13.8°**, up to 40.3 |
| grasp TCP moves by | — | median **15.7 mm**, up to 51.1 |

**39 of 40 cells executed a different grasp.** The run therefore changed the
construction and the grasp selection together and measured neither. Its numbers
(cloud box 10/20, cube 8/20) are not comparable with anything.

This is the failure mode §7.26 exists to prevent, committed while the document
that forbids it was open.

### What is worth keeping from it

Two things, neither of which needed the physics:

* **the measurement above**, which is the reason grasp selection is now an
  explicit parameter (`--filters approach|full`) with its consequences written
  into its own docstring, rather than a property of whichever code path a driver
  happened to take;
* **the observation that ranking by the planner's confidence may be actively
  worse for transport.** The method's whole job is to reproduce a demonstrated
  approach, so taking a candidate 40 degrees off instead of 6 is not a neutral
  choice — and §8h already found no evidence the planner's score predicts
  anything.

**Still open:** whether the funnel's *filtering* helps, asked with the ranking
held fixed. That needs one run varying only the filter.

---

## 8l. What GraspGen-X declares about its own grippers, and one thing we ignore

Read from the authoritative source — the gripper configs in the sibling
GraspGen-X checkout — rather than measured, so this is a contract, not a result.

Every config carries a **`symmetric`** flag, and it is **not the same for all
grippers**:

| our name | GraspGen-X name | type | `symmetric` |
|---|---|---|---|
| panda, yumi, umi | franka_panda, abb_yumi, franka_umi | `parallel_2f` | True |
| rethink, xarm, robotiq85, robotiq140 | sawyer_hand, xarm_hand, robotiq_2f_85, robotiq_2f_140 | `revolute_2f` | True |
| **robotiq3f** | robotiq_3f | `revolute_3f` | **False** |
| **inspire** | inspire_hand | `revolute_3f` | **False** |

The flag says whether rolling the hand a half turn about its approach axis gives
the *same physical grasp*. For a two-finger hand it does — the fingers swap and
the squeeze is identical. **For a three-finger hand it does not**: the fingers
land somewhere else entirely.

`metrics.transport.orientation_transport_error` applies that symmetry
unconditionally, with `symmetric=True` as its default, and `transport()` calls
it without an argument. **So every orientation figure recorded for `robotiq3f`
and `inspire` was measured with a symmetry those hands do not have** — including
their rows in Experiment J's nine-hand table (§8g).

Neither hand appears in any Tier 2 run, so no physics result is affected. What
is affected is the geometry claim that the grasp-pose cube's transported
orientation is hand-independent, which rests on that table.

**Not yet fixed**, and the fix has two parts: read the flag from the registry,
and refuse the half-turn symmetry — both in the metric and in the target-roll
choice of §8j — for any hand that declares itself asymmetric.

---

## 8z. Open items, blocked work, and grey areas

**Current as of commit `01db9d8`, after Experiments J, K and L.** Kept in this
document rather than only in a conversation so the state of the investigation
survives the people running it — the failure `ROBOTICS_NOTES.md` §7.26 is about.

### Resolved by Experiments K and L

| was | now |
|---|---|
| Are GraspGen-X's grasps any good? | **Yes. 19 of 20 lift** with no map in the loop (§8h). The 9-of-23 of §8f was scene damage |
| Is the planner's confidence usable for ranking? | **Undeterminable, and the "anti-predictive" claim is withdrawn.** With one failure in twenty there is no variance to correlate; the failure's score sits inside the successes' range |
| Does a centre-of-mass stability filter have a target? | **No.** Only **2 of 40** cells lose the object mid-carry, both on one hand. Six of the eight grasped-but-unplaced cells hold it through **100%** of the carry and misplace it (§8i) |
| Does the plan execute across hands? | Partly. **10/20 cloud box, 12/20 cube**, up from ~20% on the broken scene |

### The question those runs opened, and its answer

**Why does a plan that grips the object and carries it faithfully release it in
the wrong place?** **Answered — three coordinate-frame defects**, §8i and
`ROBOTICS_NOTES.md` §7.33-§7.34. Fixing them recovered five of the eight cells
and took the grasp-pose cube from 12/20 to 15/20 (§8j), including both cells
whose error magnitude was predicted in advance.

It also opened a new one, which is now the largest known execution weakness:
**every placement is a drop.** The arm stops 13 to 130 mm short of the commanded
release pose, so the object is let go a median 40 mm and up to 153 mm above the
shelf board and falls. It usually lands anyway — the height does not separate
success from failure — but it turns each placement into a partial lottery, and
it is what makes marginal cells flip between runs.

### Open, not started

| # | item | why it matters |
|---|---|---|
| 1 | **The arm does not complete the descent into the shelf**, so every placement is a drop from a median 40 mm and up to 153 mm (§8j) | The largest known execution weakness, and the reason marginal cells flip between runs. Tracking error at the release waypoint is 13-130 mm while the pose itself solves IK to 3-5 mm, so this is not a workspace limit |
| 1b | **The half-turn symmetry is applied to two hands that declare themselves asymmetric** (§8l) | `robotiq3f` and `inspire` are `revolute_3f` with `symmetric: False` in GraspGen-X's own config. Every orientation number recorded for them, including their rows in §8g, used a symmetry they do not have |
| 1c | **Does the full grasp funnel help?** Asked with the ranking held fixed | §8k could not answer it, because `filter_grasps` changes the ranking as well as the filtering and 39 of 40 cells got a different grasp |
| 1d | **`robotiq140` is now the weakest hand at 1/4**, having been 2/4 | Different from where the investigation started, and undiagnosed. Its two clean failures carry the object faithfully and misplace it |
| 2 | **Cross-hand closing schedule.** Jaws close at a fixed speed from different apertures, but the dwell is inherited from a Panda demonstration through the time belief | The traces from §8i record calibrated `closure` per waypoint, so this is answerable **without a new run** |
| 3 | **Contact-gated closing.** The rule must be closure-**rate** based: a box arrests the jaws dead (milk, +0.010 over the dwell) while a cylinder keeps yielding (can, +0.080) and never stalls | Needs 2. Also needs the phase to wait for the grasp event — a rollout change in the dynamics thread |
| 4 | **Per-hand dwell scaling**, the cheaper alternative to 3, needing no new sensing | Derivable from the finger-travel calibration in `gripper_frames.json` |
| 5 | **Objects are still dropped ~25 mm** at placement, landing at 0.7 m/s | Seating them at rest height was attempted and reverted: the arithmetic verified correct in isolation but produced floor-level escapes. The settle absorbs the bounce, so this is now cosmetic rather than corrupting |
| 6 | **Whether `object_keypoints` should default to the grasp cube** | Now clearly favoured: **15/20 against 10/20** on the corrected code (§8j), 18/20 grasped against 14/20, and better conditioned. The 67%-against-83% conversion gap that argued against it was the half turn, which only the cube carried. Still its own commit |
| 7 | **`prediction.reference` and `GPPolicy.attractor()` are fitted and never read; `velocity_desired` is never passed** | Pre-existing, dynamics thread, §2.7–2.8 |

### Grey areas — recorded because they are *not* solid

- ~~**The cube grasps more and places less.**~~ **Resolved.** The conversion gap was the half turn, which the cloud box's keypoints are immune to because a fitted box is symmetric under it. On the corrected code the cube leads on every stage: grasped 18/20 against 14/20, traversed 18 against 13, placed **15/20 against 10/20** (§8j).
- **`reachable_fraction` is stricter than physical feasibility, and it is a property of the whole path.** `replay_labels` seeds each IK solve from the previous one, so it answers "can the arm move *between* these poses", not "is this pose reachable". Solving each from the rest configuration instead changes the release residual on only **2 of 20** cells. Its 5 mm tolerance also flickers: successful solves land at 2.4-5.0 mm. The direct-execution control places objects at 3.8 mm while reporting 53% reachable.
- **`worst_segment` names "approach" whenever nothing was unreachable.** `unreachable_segments` takes the maximum share starting from -1, so an all-reachable path returns the first segment tested. It reads as an accusation and means the opposite.
- **Why `robotiq3f` folds under composition** (`min det = −0.045`, §8g) is unexplained — the only outright fold in 136 cells.
- **`shut@lift` is not a usable signal.** Four separate wrong readings came from the jaw channel. The traces record it; nothing should be concluded from it without a properly averaged statistic over the held interval.
- **One seed, one slot, one demonstration** everywhere. The hand-versus-object confound cannot be separated without more scenes.
- **The Panda advantage may be selection bias** — the demonstration is a Panda's and candidates are filtered to within 45° of *its* approach. Notably the Panda now scores **0/4** with the cloud box, so if the bias exists it is not simply favouring that hand.
- **Replay is not the policy.** Every number here is an upper bound under stiff position control; the impedance controller the policy uses will do worse.
- **The frame contract's acceptance test is behavioural** — "does the hand lift the object" — which is why a 30 mm error in every `contact_offset` passed the suite for the life of the project.

### Closed, listed so they are not re-opened

Gripper close-direction "sign bug" (**withdrawn** — `+1` closes all nine hands;
the *ruler* was broken); jaw units and sign not comparable across hands;
`closing_budget` returning 0.0 for every object and hand; the scene's settle,
placement interpenetration and cross-gripper divergence; all seven contact
depths; the UMI's verification status; the partial-cloud hypothesis for knocking
(**falsified** — ground-truth clouds gave identical results); "GraspGen-X's
grasps are mostly bad" (**withdrawn**, §8h); "its discriminator is
anti-predictive" (**withdrawn**, §8h); "a COM filter is justified" (**withdrawn**,
§8i); "the xarm's grip does not hold" (**withdrawn**, §8j — 1/4 to 3/4 once the
frames were fixed); the four failure "mechanisms" of §8e (**withdrawn**, §8i —
reasoned backwards from symptoms); "the vertical snap sits high because the
clouds are short" (**withdrawn** — measured, the clouds capture 99.9 to 106.1%
of true height and each object's base lands within 1.8 mm of the board).

---

## 9. Failures, and what caused each

| what failed | cause | status |
|---|---|---|
| Synthetic claim that the cloud box folds at `det = -0.63` | I placed the four pick/place keypoint blocks by hand instead of letting `place_transform` and `placement_rotation` derive the placed ones, making a configuration the real construction cannot produce | **Withdrawn.** It does not fold, to a 6.7x height ratio. The §7.21 error — an isolation study only as good as the settings held fixed around it |
| Claim that the cube is worse on orientation and mid-path tilt | measured with a **60 mm** cube half-extent, a 120 mm box larger than every object, extrapolated past the range actually swept | **Withdrawn.** At 20 mm the cube is better on both. Section 10 explains why size matters |
| First replay run, `keypoint_replay_WRIST_INVALID/` | replayed the **wrist-frame** demonstration. The cube pins the *fingertip* point, so the map carried `contact_offset` = 41.1 mm straight through: aim error was exactly 41 mm in every cell and the hand held the object for **0-1 control steps of ~119** | **Invalid, archived.** §7.20 repeating. The 41 mm had already been measured and named in this document before the driver was written |
| Second replay run, `keypoint_replay_NORESET_INVALID/` | **no environment reset between replays.** A replay leaves the object displaced and the arm parked, so each run inherited the last and *ordering* decided the result: placement errors of 354, 698 and **1170 mm**, and 0% reachability for every variant on the last object | **Invalid, archived.** `run_experiments.campaign` already builds a fresh scene per run |
| `fit_local_correction` appearing to explode — a 12 mm push producing a 336 mm excursion | **my test**, not the function: targets expressed in the source frame asked stage 2 to undo stage 1 entirely, a 411 mm correction through a 30 mm-locality stage | **Fixed, and guarded.** The function now refuses a correction larger than its own locality. It still interpolated its four points to machine precision while excursing 336 mm, so the property-(i) check gives false confidence — only the guard catches it |
| `diagnose.closing_budget` returning **0.0** for every object and every hand | `scene_keypoints` emits a pick block *and* a placed block and `_select_parts` keeps both, so the "object width" was the pick-to-place span — 239-277 mm against an aperture of at most 125 mm, which clamps `(aperture - width)/2` to zero | **Fixed.** Filtered to the `pick_*` block. Zero is worse than `None` here: indistinguishable from a genuinely impossible grasp, and the exact failure this function's `None` return was written to prevent. Invisible because every test fake carried `.points` and no `.labels`, so there was no placed block to exclude |
| The closing budget could not be derived at all for a cube construction | with `box="grasp_cube"` the keypoints are a fixed cube, so their extent is **40.0 mm for every object** — the cube, not the object | **Fixed, differently.** The width is now recorded from the **cloud** as `metrics["object_width_closing"]`, and `closing_budget` returns `None` for a cube set with no cloud width rather than reporting the cube. Not a defect in the construction: erasing the object's size is what kills the volume scaling of Experiment A |
| First nine-hand sweep, 139 of 150 cells raising `'list' object has no attribute 'orientations'` | I bound a local `labels` for a keypoint set's label list inside `gripper_sweep`, **shadowing the function's own `labels` parameter** — the source demonstration. The first cell scored and every cell after it received a list | **Fixed**, renamed to `roles`. Caught in seconds rather than after the run, because the sweep records a failure per cell instead of dropping it |
| First nine-hand sweep aborting on the first hand | `target_placement` raises `RuntimeError`, not `ValueError`, when every candidate is beyond the approach filter — which is what the milk does on a Panda. Catching only `ValueError` lost eight hands to one object | **Fixed.** Both are the pipeline legitimately refusing an object and both are recorded as a skipped cell |
| The claim that the grasp-pose cube beats the cloud box **in execution** | measured on **one hand**, the Panda the demonstration was recorded on, at 3 of 4 against 2 of 4 | **Superseded.** Across six hands they tie, 5/23 against 4/23 (§8e). What survives is the mechanism, not the score: the cube gets 10 firm grips against 3 and loses half of them |
| Reading `min det`, `aim` and `orient` as evidence a construction *works* | for the cube variants `aim` is 0 by construction and `orient` is near-0 by construction; and measured against physics, all four geometric metrics correlate with contact at `r` between −0.07 and +0.02 | **Withdrawn as a sufficiency claim.** They remain valid as necessity checks — a folded or badly-aimed map does fail — but they predict nothing about execution (§8e) |
| Four wrong readings of the jaw closure channel | each took a *summary statistic over a whole episode* and inferred a *cause at one instant*: `shutMax == 1.00` "means shut on air" (falsified: cloud box on the can, 1.00, held 82, succeeded); `shutMax > 1` "means a squeeze" (falsified: xarm, 1.04, held 0); `shut@lift` "is a per-hand baseline" (falsified: robotiq140 spans 0.00 to 0.84); and before those, the whole gripper-sign misdiagnosis | **All withdrawn.** `held_steps` is the direct contact measure and was correct throughout. Closure is now reported descriptively only; no conclusion rests on it |
| Contacts, in every form tried | see section 5: in one map they fold it; composed, stage 2 satisfies them by dragging the cube's centre off target and crushing volume around them | **Not recommended.** The grasp-pose cube already delivers what they were for |

---

## 10. Why the cube's size matters, and which metric should set it

`phi` interpolates keypoint **positions** exactly at any cube size, so `aim_map`
is 0.0 mm at 5 mm and at 60 mm alike — it cannot choose between them. But Eq. 11
reads `J_perp`, a **derivative**, and the cube's eight corners are the only thing
pinning the local rotation. Their distance from the centre is therefore the
**width of the stencil** over which that rotation is estimated, and a wide stencil
averages the grasp's own rotation together with the far field — which is the
affine stage.

Measured on real clouds, median over all objects:

| cube half extent | `min det` (median) | orientation (median) | orientation (max) |
|---|---|---|---|
| 5 mm | 0.943 | 0.0° | 0.1° |
| 10 mm | 0.941 | 0.1° | 0.4° |
| **20 mm** *(default)* | 0.943 | **0.5°** | **1.6°** |
| 30 mm | 0.943 | 1.2° | 3.6° |
| 60 mm | 0.939 | 5.1° | **14.1°** |

**So the transported orientation error is the metric that should set the size** —
the only one that varies over the usable range, and it varies for a principled
reason. There is a lower bound too: interpolation conditions *worse* as the
keypoints crowd together (3.9 um residual at 5 mm against under 1 um at 20 mm),
so smaller is not monotonically better. 20 mm sits at or below every object's own
half extent, which range 20-75 mm.

For scale, since "6 cm" sounds small and is not: a **half** extent of 60 mm
describes a **120 mm cube**, larger than every object in the scene and a third of
the 355 mm pick-to-place distance.

---

## 11. What would make these conclusions false

- **Answered on one hand, then answered again on six, and the second answer is
  different.** Experiment E gave the grasp-pose cube 3 of 4 against the cloud
  box's 2 of 4 on a Panda. Experiment H (§8e) runs the same comparison across six
  hands and the two **tie at 4/23 and 5/23**. The cube's advantage is real on the
  source hand and does not transfer. Worse, §8e shows the geometric metrics that
  rank the constructions — `min det`, `aim`, `orient` — correlate with physical
  contact at `r` between −0.07 and +0.02, i.e. not at all. **No conclusion in
  §§4–8c about which construction is better should be read as a claim about
  execution.**
- **A different grasp filter.** Construction 2 depends on the 45-degree approach
  filter, for its orientation (Experiment D) *and* for its path shape
  (Experiment F): at the limit the path already deviates 80 mm and the
  determinant has fallen from 0.950 to 0.714. Widen the filter and both degrade
  without bound — 0.134 at 90°.
- **A source demonstration that is not top-down.** Every result here uses one
  demonstration whose approach is straight down. A tilted source would change the
  relationship between the task frame and the grasp pose throughout.
- **More objects.** Four objects and one slot. The lemon is excluded because the
  pipeline rejects it — GraspGen-X itself raises on its 17-point cloud, on every
  hand. The milk fails the 45 deg approach filter **on the Panda and the Inspire
  hand only**; Experiment G found a milk grasp inside the filter for the other
  seven, because the planner conditions on the hand's swept volume. An earlier
  version of this line said the milk failed outright, which was true of the one
  hand it had been measured on.
- **A committed tree.** Until `provenance.reproducible` is `true`, no table here
  identifies the code that produced it.
