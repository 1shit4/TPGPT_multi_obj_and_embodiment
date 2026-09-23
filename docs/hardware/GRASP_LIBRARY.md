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
# one capture, all four slots (about 4 minutes per slot: 2000 screened, then 25 path checks)
~/venvs/tpgpt/bin/python - <<'PY'
import numpy as np, yaml, os
from tpgpt.hardware import coverage, shelf
from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.fr3 import HOME_Q
from tpgpt.hardware.object_model import ObjectModel, fit_object
from tpgpt.hardware.source_bundle import SourceBundle
SHARE = os.path.expanduser('~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/config')
cfg = yaml.safe_load(open(f'{SHARE}/shelf.yaml'))
cal = yaml.safe_load(open(f'{SHARE}/camera_calibration.yaml'))['camera_calibration']
camera = np.array(cal['base_to_camera']['translation'])
table = float(cfg['shelf']['anchors']['table'])
d = np.load(os.path.expanduser('~/tpgpt_plans/banana_20260923_231556_refused/clouds.npz'))
fit = fit_object(ObjectModel.load('banana'), d['observed'], table, camera)
world = shelf.ShelfWorld(model=shelf.ShelfModel.from_config(cfg), cloud=d['obstacles'],
                         table_height=table)
source = SourceBundle.load(os.path.expanduser('~/tpgpt_demos/banana_20260923_230159/source_bundle'))
for name, xyz in shelf.slots(cfg).items():
    report = coverage.coverage('banana', (fit.rotation, fit.position), world, source, xyz, xyz[2],
                               table, camera, HOME_Q, FRANKA_HAND, plan_budget=25)
    print(f'--- {name}'); print(coverage.describe(report))
PY
```

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
(112 mm tall, 75-79 mm across, against an 80 mm opening) the same object reaches three slots. Worth
knowing before the object is put on the bench.

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
60 degrees, and half of those fold. The banana offers nothing above 45 degrees and never folds.
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
Check before reading anything else into that column.

**Run-to-run variation is now perception, not the sampler.** The four banana captures are minutes
apart, motionless, yet the fit's heading reads 53.9, 52.2, 47.4 and 47.6 degrees, and the funnel
moves with it: 1109, 1034, 1052 and 1088 through `from_below`, 431, 375, 315 and 403 through the
jaws, and bottom_left 10, 2, 5 and 0 admissible. With the draw fixed, this is what is left.

## Deviation from the simulation

The simulation caches candidates per cloud (`tpgpt/grasp/cache.py`) and re-draws whenever the cloud
changes, which in simulation it does not. On hardware the camera's cloud changes every frame, so
the same cache would never hit; keying on the **scan in its canonical frame** is what makes it hit,
and it is available only because the planner already works on the scan. The grasps themselves, the
selection rule and every filter are unchanged: the library changes *which draw* the selection sees,
not how it chooses. Recorded in `FAITHFULNESS_AUDIT.md` as A13.
