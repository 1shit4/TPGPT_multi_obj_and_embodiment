# The grasp library, and measuring coverage

*(hardware branch, 2026-09-24. Code: `tpgpt/hardware/grasp_library.py`,
`tpgpt/hardware/coverage.py`, tests `tests/unit/test_grasp_library.py`.)*

## The problem it solves

GraspGen-X's diffusion planner samples from noise, and neither the client nor the server takes a
seed (`tpgpt/grasp/cache.py` says so in as many words). **Every plan therefore asked a different
question.** Measured live on one banana, on the table, with the same scene and the same
demonstration:

| candidates asked for | admissible grasps found | planning time |
|---|---|---|
| 100 | 16, then 0, then 0 | — (two of three runs refused) |
| 400 | 5, then 38, then 38 | 271 s, 22 s, 15 s |

Three runs of the same task, three different answers. That is not a property of the task; it is the
draw. It makes a refusal uninterpretable ("is this scene impossible, or did we get a bad draw?"),
it makes a success unreproducible, and it made the number of candidates a tuning knob that decided
whether planning worked at all.

## What the library is

Since `object_model` (audit A8) the planner does not hand GraspGen-X the camera's cloud; it hands
it the object's **YCB scan**, fitted to the camera. The scan is a fixed, known shape. So the
grasps can be asked for **once, in the scan's own frame**, kept on disk, and carried into the scene
by the rigid transform the fit already provides:

```
scan (object frame) --GraspGen-X, once--> 2000 poses in the object frame   [disk]
                                                  |
                          fit (rotation, position) of the scan to the camera cloud
                                                  v
                                      2000 poses in the robot base frame
```

`placed()` is `T @ poses` with `T` the fit. It is exact, not an approximation: the fit is rigid, so
every grasp keeps its geometry relative to the surface it was generated on (pinned by
`test_placing_the_library_is_the_rigid_transform_itself`).

The cache is the **simulation's own** `tpgpt.grasp.cache` — same key (the cloud's bytes, the
gripper, the size, the planner), same store. What is new is *which cloud* is the key: the scan in
its canonical frame, sampled with a fixed seed (`SAMPLE_SEED = 1`), rather than a camera cloud that
is never twice the same.

Consequences:

* **The same grasps every run.** A difference between two runs is a difference in the task.
* **Size stops being a tuning knob.** 2000 grasps cost 4–10 s once per object and nothing
  afterwards (0.1–0.8 s from disk), against 200 drawn fresh on every plan.
* **The feasible set becomes measurable** — see *Coverage* below.

**When it is not used.** If the scan fit is refused the planner works from the camera's own cloud,
whose shape is not the scan's, so the grasps are asked for live exactly as before. Any other
failure (no server on a cold cache, a missing scan) also falls back to the live draw: a library
must never be the reason a plan is lost.

## Using it

The planner node uses the library by default when the fit is accepted:

```bash
# defaults; nothing to do to get the library
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node \
    -p grasp_library:=true -p grasp_library_size:=2000
# the old behaviour, a fresh draw on every plan
ros2 run --prefix python3 ycb_pointcloud_pipeline tpgpt_planner_node -p grasp_library:=false
```

The plan's metadata records which it was, as `grasp_candidates_from`
(`library (2000, cached)` / `graspgenx (fresh draw of 200)`).

The libraries live in `outputs/hardware/grasp_library/` (one `.npz` per object, gripper and size;
override with `TPGPT_GRASP_LIBRARY`). They are built on first use, which needs the GraspGen-X
server. To build them all ahead of time, with the server running
(`ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh`):

```bash
cd ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment
~/venvs/tpgpt/bin/python -c "
from tpgpt.hardware.grasp_library import library
for name in ('banana', 'apple', 'tomato_soup_can', 'pudding_box', 'foam_brick'):
    poses, scores = library(name)
    print(name, len(poses), 'grasps')
"
```

## Coverage: what a destination admits

With the draw gone, a question that used to be meaningless becomes a property of the task:

> of this object's 2000 grasps, how many survive each stage for this destination, and how many have
> a transported path the arm can execute?

`tpgpt/hardware/coverage.py` answers it offline — no camera, no robot — on a scene a plan recorded
earlier (`~/tpgpt_plans/<object>_<time>[_refused]/clouds.npz`). It is measured **on the
generator's own grasps**, so it carries GraspGen-X's notion of what holds; a geometric sweep of
hand poses would not.

Read it as a funnel:

* everything lost at `place_zone` or in the carry ⇒ the **shelf** blocks that destination, not the
  object;
* everything lost at `jaws` ⇒ the **hand** cannot hold that shape from those directions;
* a zero at the end over a whole library ⇒ that destination is not reachable from that pick with
  this demonstration, and no number of candidates will change it.

```bash
source ~/venvs/tpgpt/bin/activate
cd ~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment
# the capture's own object, its scan re-fitted to the capture exactly as the planner does
python -m tpgpt.hardware.coverage \
    --capture ~/tpgpt_plans/banana_20260923_231459_refused \
    --demo ~/tpgpt_demos/banana_20260923_230159/source_bundle \
    --slots top_left,top_right,bottom_left --json banana.json
# any object's scan set down in that scene: resting pose 0, handle turned 90 deg, at (0.606, 0.161)
python -m tpgpt.hardware.coverage \
    --capture ~/tpgpt_plans/banana_20260923_231459_refused \
    --demo ~/tpgpt_demos/banana_20260923_230159/source_bundle \
    --object mug --at 0.606 0.161 --rest 0 --heading 90 --slots top_right
```

About 4 minutes per slot on an idle machine (2000 screened, then 25 path checks); several run in
parallel with `OMP_NUM_THREADS=1`. It uses the planner node's selection defaults (tilt cap 90,
visibility and centred off, no grasp offset, strict reachability) and reads the **installed**
`shelf.yaml` and `camera_calibration.yaml`, which is what the planner reads (`--config-dir` to
change). Checked on 2026-09-25 against a result from the week: the mug at 90 deg into top_right
with the 23:01 demonstration, 4 admissible of 25, first at rank 4, both times.

`coverage()` also takes `cloud=` and `poses=` directly, for a capture whose fit was refused (the
planner is then on the camera's cloud) or for a set of grasps that did not come from the library.

## What was measured

All of it offline, on captures the planner recorded on the evening of 2026-09-23 (after the
recalibration), with the 23:01 banana demonstration, the Franka Hand, 2000 grasps per object and
the first 25 screened survivors planned and path-checked per destination. Sources:
`outputs/hardware/coverage_fixed/*.json`.

**The banana, on the four captures that were refused live.** Admissible of 25 planned:

| capture (fit) | top_left | top_right | bottom_left | bottom_right |
|---|---|---|---|---|
| 23:05 (0.74 explained) | 9 | 8 | **10** | 0 |
| 23:13 (0.89) | 9 | 7 | 2 | 0 |
| 23:14 (0.90) | 11 | 8 | 5 | 0 |
| 23:15 (0.84) | 11 | 8 | 0 | 0 |

Every one of those captures was refused live, with a fresh 200-grasp draw. The grasps existed; the
draw did not contain them. (The top slots additionally needed the clearance correction below; the
bottom-left numbers are unaffected by it.)

**The other objects**, at a resting pose placed where the banana stood, in the same scene
(synthetic pose: this measures the geometry, not a particular perception):

| object (resting pose, height) | top_left | top_right | bottom_left | bottom_right |
|---|---|---|---|---|
| apple (72 mm) | 9 | 6 | 2 | 0 |
| foam brick (flat, 51 mm) | 10 | 5 | 3 | 0 |
| pudding box **upright** (112 mm) | 5 | 8 | 1 | 0 |
| pudding box **lying flat** (39 mm) | 0 | 0 | 0 | 0 |
| tomato soup can (upright, 102 mm) | 0 | 0 | 0 | 0 |
| tomato soup can **lying** (68 mm) | 6 | 4 | 0 | -- |
| **mug** (upright, 81 mm, handle at 0 deg) | 3 | 14 | 3 | -- |
| **mug** (upright, handle at 90 deg) | 7 | 4 | 6 | -- |

(The lying can and the mug were measured on 2026-09-25, after `bottom_right` was abandoned, so
that column is not filled for them. An earlier file named `tomato_soup_can_lying.json` used
resting pose 1, which for this scan is a *second upright* pose -- rests 0 and 1 both stand the can
on an end, 102 mm tall; the lying poses are 2 to 11. The row above is rest 2.)

Four things this made visible that a single draw could not:

**1. The place clearance was measured against an infinite plane, and it refused every upper-shelf
grasp.** `_verify`'s `path_hits_surface` took the hand's lowest point over the whole retreat and
compared it with the destination's *height*, everywhere. The taught retreat comes back down in
front of the shelf: traced on one top_left plan, the hand leaves the board 8 mm clear, rises to
70 mm, then descends to 35-53 mm "below" the board while 215-314 mm away from it, over open floor.
`path_check`, which knows where the boards end, passed that same plan. Before the correction the
upper slots scored 0 admissible out of 25 on every capture and every object; after it, the table
above. Fixed in `planner._hand_clearance` (audit A14): the place side is measured only while the
hand is within the object's own footprint plus 50 mm of the destination.

**2. A lying pudding box cannot be picked at all, and that is the hand, not the method.** Flat it is
138 x 128 x 39 mm: both horizontal dimensions exceed the hand's 80 mm opening, and a pinch across
the 39 mm thickness has the fingertips at the table. Of 2000 grasps, 1246 survive `from_below` and
**none** survives the jaw check (892 `hits_support`, 329 `shallow`, 25 `empty`). Standing on its end
(112 mm tall, 75-79 mm across, against an 80 mm opening) the same object reaches three slots, so
**the box is always set down upright** (user, 2026-09-25). Worth knowing before the object is put on
the bench.

**3. Beyond about 45 degrees of tilt, the transport map folds.** Across 286 planned candidates on
five objects, by the grasp's tilt from straight down:

| grasp tilt | maps that fold (det(J) changes sign) |
|---|---|
| 0-15 deg | 0 of 73 (0 %) |
| 15-30 deg | 0 of 58 (0 %) |
| 30-45 deg | 1 of 24 (4 %) |
| 45-60 deg | 11 of 19 (58 %) |
| 60-75 deg | 21 of 36 (58 %) |
| 75-90 deg | 42 of 76 (55 %) |

This is the sim's own rule (`consistent_sign`), not the hardware's extra `min_determinant` gate:
the determinants read -0.34 to -0.58. It is why the **tomato soup can fails everywhere**: 68 mm
across and 102 mm tall, the only grasps it offers are around its barrel, 57 of 60 of them beyond
60 degrees, and half of those fold. Laid on its side the same can plans into both top slots (6 and
4), because a barrel lying down is gripped from above -- the tilt, not the can, was the problem. The banana offers nothing above 45 degrees and never folds.
The plausible mechanism is A12 -- the placed orientation is a yaw about the table, so a
near-horizontal pick has its keypoint cube rotated nearly into the table plane at the release --
but that is a hypothesis; what is measured is the table above. Two consequences to weigh, neither
of which loosens a check: selection could prefer the grasps the map can transport (it currently
walks the generator's score order, and tilt is not in it), and a demonstration picked at a low tilt
cannot lend its motion to an object that can only be gripped horizontally.

**4. Something stands in the bottom-right slot.** Every object scores 0 there, on
`the carried object touches seen points`: 73 obstacle points within 130 mm of that slot, 105 to
229 mm above the bottom board, 30 to 57 mm clear of the shelf model (so not the boards). Either an
object is in that slot or the camera is worse there than the 30 mm the model already absorbs.

**The slot is abandoned** (user, 2026-09-25): it is also the corner furthest from the base, and the
arm reaches it badly. The destination set is `top_left`, `top_right`, `bottom_left`, and the
bottom-right column in the tables above is kept only as the record of what was measured.

**5. A rim is what makes an upright vessel graspable, and the scan has to have one.** The can was
replaced by the **mug** on 2026-09-25 (user). Of 2000 grasps, sorted by tilt from straight down
with the object standing on the table:

| object | 0-15 | 15-30 | 30-45 | 45-60 | 60-75 | 75-90 | at or under 45 deg |
|---|---|---|---|---|---|---|---|
| tomato soup can | 176 | 62 | 70 | 77 | 195 | 626 | **308 of 2000** |
| mug | 476 | 486 | 334 | 188 | 91 | 113 | **1296 of 2000** |
| banana | 154 | 198 | 150 | 151 | 165 | 234 | 502 of 2000 |
| apple | 103 | 116 | 91 | 109 | 175 | 394 | 310 of 2000 |

The mug's 25 best-scoring survivors at `top_left` are **all rim grasps** -- 39 to 42 mm from the
cup's axis, 68 to 74 mm up a rim that stands 81 mm above the table, so the pads sit about 11 mm
down the wall, just past the 10 mm `MIN_ENGAGEMENT` that stops a grasp pivoting off a rim -- at
2.5 to 40.6 degrees of tilt, every one inside the fold limit, scoring 0.90 to 0.945. The jaw check
keeps 1500 of 1688 for the mug against 459 of 1033 for the lying can. And `plan_refused`, the fold,
drops from 17-18 candidates in 25 for the upright can to 0-4 for the mug: what kills the mug's
remaining candidates is `collision` with the shelf's side walls, which is geometry, not the map.

**The can's own rim cannot be used, and that is a modelling fact, not a physical one.** The real
can on the bench has no lid. The scan does: sampling `005_tomato_soup_can` upright puts 22 points
on the top face inside a 25 mm radius and only 3 anywhere in the interior, and 6-12 mm below the
rim its radii form one shell, not two. `025_mug` is the opposite -- 0 points on the top face, 340
in the cavity, and two shells 6-12 mm down (37-40 mm and 40-42.5 mm: a 4-5 mm wall). Since the
planner plans against the **scan**, a finger reaching down inside the real can's rim is a finger
inside solid material, and `grasp_collision` refuses it. The mug is the same task -- an upright
vessel too wide to grip around -- with a rim the model knows about.

**The mug's heading matters more than any other object's**, because it is the handle that decides
which side of the cup a grasp can approach from: at 0 degrees it scores 3 / 14 / 3 across the three
slots, at 90 degrees 7 / 4 / 6. Both reach all three; which slot is easy swaps. Worth knowing when
it is put on the table.

**6. Which demonstration** (2026-09-25). Two new banana demonstrations, 16:35 and 16:52, taught
to give more clearance, measured against the 09-23 23:01 one on the same capture and objects
(`python -m tpgpt.hardware.coverage --demo ...`; raw in `outputs/hardware/coverage_demos/`).
Admissible of 25, top_left / top_right / bottom_left:

| object | 09-23 23:01 | 16:35 | 16:52 | 18:33 | 18:39 |
|---|---|---|---|---|---|
| banana (23:14 capture) | 11/8/5 | 22/18/2 | 21/20/4 | 15/17/3 | 22/14/7 |
| apple | 9/6/2 | 6/3/2 | 10/8/1 | 8/3/4 | 9/10/3 |
| foam brick | 10/5/3 | 17/16/2 | 18/19/2 | 16/21/3 | 17/22/3 |
| pudding box upright | 5/8/1 | 11/11/1 | 10/8/**0** | 3/7/**0** | 11/11/**0** |
| mug, handle 0 deg | 3/14/3 | 9/17/**0** | 10/18/**0** | 5/15/3 | 9/17/5 |
| mug, handle 90 deg | 7/4/6 | 20/7/2 | 18/5/1 | 17/2/2 | 23/6/3 |
| **total** (top_left, top_right, bottom_left) | **110** (45, 45, 20) | **166** (85, 72, 9) | **173** (87, 78, 8) | **144** (64, 65, 15) | **192** (91, 80, 21) |

The 18:33 and 18:39 demonstrations used a different strategy (picked at 23 and 26 deg; 18:33 with
73 mm of headroom under the top board, 18:39 with 45 mm, 136 mm from the left wall and the banana
turned only 1.9 deg in the hand). **18:39 is the best measured**: the most admissible, bottom_left
back to the 23:01 level, and the top slots still doubled. The fold returns somewhat at the
intermediate tilt (`plan_refused` 46 and 51, against 10-11 at 13 deg and 54 at 41).

What changed, from the demonstrations' own geometry (all three place on the bottom board):

| | 23:01 | 16:35 | 16:52 |
|---|---|---|---|
| hand tilt at the pick | 40.6 deg | 13.4 deg | 13.4 deg |
| hand's headroom under the top board inside the slot | 60 mm | 16 mm | 37 mm |
| candidates lost to the map folding (`plan_refused`) | 54 | 11 | 10 |
| bottom_left candidates hitting the top board (carry) | 39 | 83 | 54 |

The near-vertical pick is what the new demonstrations gain: the fold, which took 54 candidates, all
but disappears, and the top slots roughly double. What they lose is **bottom_left**, their own
destination: they carry the banana higher inside the slot, so the hand rides 16-37 mm under the top
board instead of 60, and transported to taller objects it hits it. Two object-slots go to zero of
25 (the planner itself tries up to 100 survivors, and bottom_left has 40-80, so zero of 25 is not
yet zero). "More clearance" was clearance above the bottom board, paid for under the top one.

**7. Moving the bottom slot toward the shelf's front edge** (2026-09-25, 18:39 demonstration,
`python -m tpgpt.hardware.coverage --slot-depth`; raw in `outputs/hardware/coverage_depth/`). The
touched marks sit 90 mm into the 242 mm-deep shelf. Admissible at bottom_left, of those planned
(the pudding box with the planner's full budget, which at bottom_left is all 41-42 survivors):

| object | 90 mm (the marks) | 75 mm | 65 mm |
|---|---|---|---|
| banana | 7/17 | 7/20 | 10/22 |
| apple | 3/25 | 3/25 | 5/25 |
| foam brick | 3/13 | 3/13 | 4/15 |
| pudding box upright | **0/42** | 1/41 | 1/41 |
| mug, handle 0 deg | 5/25 | 6/25 | 7/25 |
| mug, handle 90 deg | 3/25 | 4/25 | 4/25 |
| **total** | 21 | 24 | 31 |

Every object does as well or better closer to the edge, but the pudding box is not rescued: its
candidates die the same way at every depth (the carried box against the bottom board's edge 15-16,
the arm or hand against the top board 11, the fold 14). It is 112 mm tall in a 238 mm slot with the
hand above it, and the banana's carry does not leave that room; the depth does not change it.
Support limits the move: half the footprint in the worst orientation is banana 99 mm, mug 62,
foam brick 51, pudding box 48, apple 42 -- at 65 mm the mug fits with its handle pointing out and
the banana overhangs by at most 34 mm with its middle well on the board. Not applied: the slots
are still the touched marks.

**8. The pudding box on its long side** (2026-09-25; the protocol pose since, `PROTOCOL_RESTS`).
Every coverage number above for the pudding box is the box **on its end** (112 mm). In live runs
that evening only the box on its long narrow side (89 mm up) planned into bottom_left. Measured on
the five live captures with it on its side, re-fitted in that pose, 18:39 demonstration, the
planner's full budget -- admissible / planned:

| capture | top_left | top_right | bottom_left |
|---|---|---|---|
| 21:25 | 28/67 | 30/53 | 1/27 |
| 21:26 | 32/66 | 33/56 | 2/28 |
| 21:33 | 32/50 | 31/51 | 0/11 |
| 21:46 | 31/69 | 29/52 | 1/30 |
| 21:47 | 31/70 | 33/71 | 3/35 |

The top slots are strong (against 11 of 25 on its end); bottom_left has 1-3 on four of five
placements and none on one -- workable, not comfortable. What still stops bottom_left is the same
as on its end, only less often: the arm or hand against the boards, the fingers below the board
surface at the set-down, the fold. Raw in `outputs/hardware/coverage_side/`.

**9. The margins behind the 15 live plans** (captures of 2026-09-26 16:42-17:38, measured
2026-09-29). Each of the fifteen plans that succeeded live (five objects, three slots) replayed on
its own capture: the 18:39 demonstration, the robot filter at 50 mm, the planner's budget of 100,
the arm starting at the joints the plan recorded (home), and the move onto the path still
rejecting (the rule until 2026-09-29; the new rule can only add grasps). Admissible / planned, and
the rank of the first admissible (0 = the first planned):

| object | top_left | top_right | bottom_left |
|---|---|---|---|
| banana | 63/100, 0 | 50/100, 0 | **2/20**, 5 |
| apple | 22/100, 6 | 31/100, 3 | **2/63**, 5 |
| mug | 56/100, 0 | 25/100, 7 | **2/59**, 3 |
| pudding box (long side) | 29/68, 0 | 32/55, 0 | 4/30, 2 |
| foam brick | 28/46, 0 | 27/39, 0 | **2/15**, 2 |

The top slots have margin: 22 to 63 executable grasps each, and the first is usually the first
planned. **bottom_left does not**: two executable grasps for four objects of five, four for the
pudding box. Two is enough to plan, and it planned live on all five, but it is one placement, one
fit or one calibration shift away from none. That is what item 8's 0/11 already showed for the
pudding box. What kills the others, over all five objects: the arm or hand against the underside
of the **top board** in carry, place and retreat (5-22 candidates each; the board has sat 10 mm
lower since 2026-09-26, 0.3211, headroom 244 mm); the carried object against the **bottom board**
during the carry (7-32; the taught carry slides in low); and for the apple and the mug, the
carried object against **seen points** (18 and 23). Those points are the layer the band
re-measure found: on the mug's capture, 27 obstacle points 31-52 mm above the bottom board, at
x 0.745-0.805, y -0.229 to -0.181. That is 60-120 mm deeper than the bottom_left mark and 50-100 mm
to its left. If something is standing there, taking it away returns those candidates. If it is
part of the shelf, the obstacle is real. Raw in `outputs/hardware/margins_20260926/`, logs in
`~/tpgpt_coverage_logs/margin_*.log`.

What would make this wrong: the counts are for these captures. A placement a few centimetres off,
or the mug's handle turned, moves them (item 1 above: 10, 2, 5 and 0 for four captures of one
motionless banana). So read "2" as "few", not as an exact number.

**Run-to-run variation is now perception, not the sampler.** The four banana captures are minutes
apart, motionless, yet the fit's heading reads 53.9, 52.2, 47.4 and 47.6 degrees, and the funnel
moves with it: 1109, 1034, 1052 and 1088 through `from_below`, 431, 375, 315 and 403 through the
jaws, and bottom_left 10, 2, 5 and 0 admissible. With the draw fixed, this is what is left.

**10. With learned labels the check is on the arm's path** (2026-09-30, audit B1 step 6). The same fifteen captures, the 18:39 demonstration as taught (`source_bundle_directional`: 372 admissible, the A3b baseline -- item 9 with the arm posed at its waypoints -- identical capture by capture) and as learned by the replay (`source_bundle_learned`: the command it learned is executed, the path the arm traced is checked with its run-to-run margin):

| object | top_left | top_right | bottom_left |
|---|---|---|---|
| banana | 63 -> 58 -> 59 | 47 -> 32 -> 33 | 1 -> 4 -> 4 |
| apple | 23 -> 17 -> 17 | 31 -> 21 -> 21 | **2 -> 0 -> 0** |
| mug | 58 -> 49 -> 49 | 24 -> 17 -> 17 | 1 -> 1 -> 1 |
| pudding box (long side) | 29 -> 28 -> 28 | 32 -> 32 -> 32 | 3 -> 2 -> 2 |
| foam brick | 29 -> 28 -> 28 | 27 -> 27 -> 27 | 2 -> 2 -> 2 |

Taught -> learned with the full margin -> learned with only the margin's vertical part against the table (the check since 2026-10-01): 372 -> 318 -> 320. One cell goes to zero, **apple bottom_left**: on the taught path both of its grasps slide the apple onto the bottom board with 0.1-0.4 mm to spare before the set-down begins; the arm's path runs about 1 mm lower there and crosses the 1 mm tolerance (one of the two also puts a pad on the apple on the way down). Sub-millimetre, on a cell item 9 already called one shift away from none. The difference is not the labels getting worse: the taught check assumed the arm is where it is commanded, which it is not (pass 0 strayed 17 mm RMS). Three things account for most of it, from a per-candidate re-check of the three worst slots: the arm comes down to the grasp **4-5 mm off across the jaws** (not learned, audit B1 step 6), which puts an open pad on an object that nearly fills the 80 mm opening (apple, mug); the **run-to-run margin against the table** in the last labels before the grasp (banana top_right: 32 with it, 33 with only its vertical part, 51 without); and after the release the arm drifting 3 mm more across the jaws than the taught hand (a pad back on the placed mug). bottom_left is unchanged at 9. What would make this wrong: the arm path is pass 3's, carried by the map; that the arm strays from a transported command as it strayed from the learned one is assumed, not measured. Raw in `outputs/hardware/learned_20260930/`, logs in `~/tpgpt_coverage_logs/learned_*.log`.

## Deviation from the simulation

The simulation caches candidates per cloud (`tpgpt/grasp/cache.py`) and re-draws whenever the cloud
changes, which in simulation it does not. On hardware the camera's cloud changes every frame, so
the same cache would never hit; keying on the **scan in its canonical frame** is what makes it hit,
and it is available only because the planner already works on the scan. The grasps themselves, the
selection rule and every filter are unchanged: the library changes *which draw* the selection sees,
not how it chooses. Recorded in `FAITHFULNESS_AUDIT.md` as A13.
