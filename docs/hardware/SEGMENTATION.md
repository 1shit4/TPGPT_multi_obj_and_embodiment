# Detection: prompts, thresholds, and an object that comes and goes

`segmentation_node` is the first stage of everything: no cloud on
`/ycb_pointclouds/<object>`, no plan. Its detector is Grounding DINO (text prompt -> boxes)
followed by SAM (box -> mask), both from `config/target_objects.yaml`.

## What the detector knows, and what it does not

**Nothing in this pipeline knows what a YCB object looks like.** "YCB" is a catalogue of
physical objects with 3D scans; it is not a class list any model here was trained on. Grounding
DINO is an *open-vocabulary* detector: it was trained on ordinary web images paired with
captions, so it knows words -- "box", "brown", "banana", "foam" -- and finds the image region that
best matches a sentence. It has never seen the label `008_pudding_box`. **The prompt is the only
definition of the object the detector has**, which is why it has to describe what the thing
*looks like* in plain words.

The YCB identity enters *after* detection, and only there: the planner takes the camera cloud
under the mask and fits the object's own YCB scan to it (`tpgpt/hardware/object_model.py`). That
fit is what carries the true shape into planning, and it refuses a cloud that does not look like
the scan. So the detector's job is narrower than "recognise the pudding box": it is **put a mask
on the right patch of pixels, every frame, and on no other patch**.

That gives the prompt two opposite ways to fail:

* **too narrow** -- a wording the model half-recognises scores 0.3 on one frame and 0.5 on the
  next, and the object comes and goes. This is the reported pudding-box failure.
* **too broad** -- a wording like "brown box" scores well every frame, but on this bench two of the
  five objects are boxes, so it can land on the foam brick instead. A confident mask on the wrong
  object is worse than a missing one: the planner fits the pudding box's scan to the brick's
  cloud, and either refuses or plans a grasp on the wrong thing.

**Each object has exactly one prompt** in `target_objects.yaml`. Several wordings appear only when
*testing*: `prompt_bench` is given a handful of candidates, scores each one on saved frames, and
one is chosen. The others never go into the config.

The failure this document is about was reported on 2026-09-25: **the pudding box is detected on
some frames and not on others**, with nothing in between to explain it.

## Why a miss used to be invisible

Up to 2026-09-25 the node logged only what it published:

```
[41] inference 0.62 s, published: ['banana(1877 pts, 0.71)', 'apple(1204 pts, 0.55)']
```

An object missing from that list could be any of four different problems, and the log did not
distinguish them:

| what happened | what to do about it |
|---|---|
| the box scored **0.34** against a threshold of 0.35 | lower *that object's* threshold, or reword its prompt |
| the box scored **0.02** -- the model did not see it | reword the prompt; check lighting, occlusion, the object's pose |
| the matched phrase fit **two prompts equally** (`"box"` when two prompts contain "box") | make the prompts distinguishable |
| detected fine, but **no depth in the mask** | a depth problem, not a detection problem: surface, range, `depth_trunc` |
| found, but **on the same box as another target** | one prompt is landing on the other object; the node keeps the higher score and drops the other |

The first and the fourth look identical from the outside and have nothing in common. The last one
is the confusion between similar objects: when two targets' boxes overlap by 70 % or more the node
has always kept the higher score, because one physical object cannot be two targets -- and the
other target simply went missing that frame.

## What the node reports now

Detection runs at a **report floor** (`detection_report_floor`, 0.05) and each target is accepted
at **its own threshold** afterwards. What is accepted is exactly what the single `box_threshold`
accepted before; what is new is that everything else leaves a trace:

```
pudding_box: best box scored 0.29 as 'pudding box', below its threshold of 0.35 -- not published
foam_brick: not seen at all this frame (nothing above 0.05)
pudding_box: detected at 0.41 over 3120 pixels, but no depth in the mask (within 2.0 m) -- nothing published
a box scoring 0.44 matched the phrase 'box', which fits two prompts equally well, ...
pudding_box: found at 0.52, but on the same box as foam_brick (0.61, overlap 0.83) -- one object
  under two names, so the lower score was dropped. One of the two prompts is landing on the other object.
```

The same per-target report goes into `/ycb_detections/info` (fields `report` and `unmatched`), so
an X6 record can say afterwards why a trial's object never arrived.

## Measuring, instead of trying wordings at the robot

One run tells you nothing about a flaky detector, and the scene changes between runs. So:

**1. Keep the frames it goes wrong on.** Run a normal session with

```bash
# T3, segmentation venv
source ~/venvs/ycb/bin/activate
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false \
    save_frames:=misses save_frames_dir:=~/ycb_detection_frames
```

Every frame where a listed target reaches no topic is written out as `<cycle>_miss_<names>_rgb.png`,
its depth as `.npy`, and the report as `.json`. `save_frames:=all` keeps the good frames too, which
is what you want for a comparison: a prompt that works on the frames that already worked is not
evidence.

**2. Replay them offline**, with no camera and no robot:

```bash
source ~/venvs/ycb/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline prompt_bench \
    --frames ~/ycb_detection_frames --object pudding_box \
    --prompts "brown chocolate pudding box" "chocolate pudding box" "small dark brown box"
```

It scores each wording on every saved frame, in both caption modes, and reports two things:

* the **worst** frame's score. That is the number that decides robustness: a prompt whose floor is
  0.45 is robust, one that reaches 0.8 on its best frame and 0.1 on its worst is the flakiness you
  started with.
* **CONFUSED**, when that wording's box sits on another target's box (overlap above 0.5) -- e.g.
  the pudding-box wording finding the foam brick. A wording that is ever confused is not chosen,
  however well it scores. For this to mean anything the saved frames must have **both boxes in
  view**; the bench compares boxes, and cannot know which object is which on its own.

## The two knobs the bench chooses between

**A per-object threshold.** `target_objects.yaml`:

```yaml
  - name: pudding_box
    prompt: brown chocolate pudding box
    threshold: 0.25        # this object only; others stay at box_threshold
```

Use it when the object is consistently *seen* but consistently *low*. It does not make the detector
better, it accepts a weaker box -- so check the mask that comes with it (`/ycb_masks/pudding_box`)
before trusting the cloud.

**The caption mode.** Grounding DINO scores a phrase in the context of the whole caption, and the
node has always sent one caption containing every target:

```
banana. apple. brown chocolate pudding box. red foam brick. tomato soup can.
```

`caption_mode:=per_object` sends one caption per target instead -- one forward pass each, so the
inference time multiplies by the number of targets -- and no phrase competes with another. Whether
that helps *here* is what the bench measures; it is not assumed.

**It cuts both ways.** The competition between phrases is also what suppresses a wrong answer. On a
synthetic frame -- a brown rectangle on black, nothing else -- the joint caption returned the
pudding box at 0.55 and no banana at all; `per_object` returned the pudding box at 0.59 **and a
banana at 0.41**, because a caption saying only "banana." has nothing better to offer. The bench
recorded a similar thing live: the mug was taken out of the list on 2026-09-23 because "red mug"
matched a cup behind the robot at 0.75. So per-object captions want a tighter threshold, and the
mask (`/ycb_masks/<object>`) checked before the cloud is trusted.

```bash
ros2 launch ycb_pointcloud_pipeline ycb_pipeline.launch.py launch_camera:=false \
    caption_mode:=per_object
```

## What is measured so far

**The pudding box vanished because its score sits on the cutoff** (run of 2026-09-25 14:42, from
the node's own log, `~/.ros/log/python3_90487_*.log`). Pudding box and foam brick both on the table,
both correctly identified. Of 605 cycles:

| object | published | score when published | |
|---|---|---|---|
| pudding box | 320 (53 %) | 0.35-0.44, median 0.39 | 26 frames at 0.35-0.36 |
| foam brick | 569 (94 %) | 0.36-0.46, median 0.39 | 2 frames at 0.36 |

Histogram of the pudding box's published scores, 0.01 bins from 0.35: 26, 57, 36, 29, 28, 79, 58,
6, 1. A distribution that is still full at the cutoff continues below it, and the misses alternate
with hits frame by frame rather than in long gaps -- noise moving a score across a line, not an
object leaving view. The old log could only show the frames above 0.35; the misses' own scores
were not recorded, which is what the report now fixes.

**Fix: `threshold: 0.25` for the pudding box alone** (`config/target_objects.yaml`), leaving 0.35
for everything else. Chosen, not measured: the misses' scores were not logged, so whether 0.25
catches all of them is what the next run shows -- any frame still missing it now logs its score.

**Then the shelf became the pudding box** (run of 2026-09-25 15:42, `python3_93946_*.log`). With
the pudding box off the table, its prompt found the shelf on 36 cycles. The masks tell them apart
at a glance -- the real box 1 500-2 100 points, the shelf 71 500-72 100 -- and the scores cannot:

| what "pudding box" was | points | score |
|---|---|---|
| the real box, first placement | ~1 500 | 0.65-0.70 |
| the real box, moved | ~2 050 | 0.33-0.48 |
| the shelf | ~72 000 | 0.25-0.40 |

The shelf scored up to 0.40 and the real box as low as 0.33, so no cutoff separates them; even the
original 0.35 let the shelf through on about ten of those cycles. Lowering the pudding box to 0.25
made it more frequent, it did not cause it.

**Fix: a size check against the object's scan.** Every object in `target_objects.yaml` carries
`max_size`, its YCB scan's largest dimension (banana 0.198 m, apple 0.085, mug 0.125, pudding box
0.142, foam brick 0.102). Every candidate box is measured -- width in pixels times the median depth
in its central half, over the focal length -- and refused for a target when it is more than
`size_margin` (1.5) times that target's `max_size`. It happens **before** the best box per target is
chosen, so with both in view the real box wins even on a frame where the shelf scores higher. The
refusal is logged:

```
foam_brick: a box scoring 0.41 measured 1.54 m across, but foam_brick is at most 0.15 m -- refused, ...
```

and each published object now shows its measured size, `pudding_box(4621 pts, 0.43, 0.12 m)`, so
the margin can be checked on the real objects. Verified on synthetic frames through the node (a
small box in front of a large brown wall: the box published at 0.12 m, the foam brick's prompt on
the wall refused at 1.54 m -- at 0.41, above its own 0.35). Not yet on the bench.

**Then the prompts were replaced, from labelled frames** (2026-09-25). Six scenes recorded with
`save_frames:=all`, each with a known set of objects on the table: brick only (28 frames), pudding
box only (24), both (35), mug only (17), apple only (23), mug and apple (30) --
`~/ycb_detection_frames/<scene>/`. Looking at them explained the confusions at once: the pudding
box is a small printed box, mostly **white** with a red logo; the foam brick is **dark brown**; the
mug and the apple are both **red** and round. The old prompts ("brown chocolate pudding box", "red
foam brick") had the colours nearly swapped, and "red mug" fits the apple. Frames apple 16-19 have
no apple on the table (it was being moved) and were counted as empty.

The node's detection was replayed on every frame -- all prompts in one caption, the size check
on the saved depth, per-object thresholds, the same-box rule -- and reproduced what the node had
actually published on **157 of 157** frames. The replay is `detection_replay`; what was on the
table in each scene is in `~/ycb_detection_frames/labels.yaml`:

```bash
source ~/venvs/ycb/bin/activate
ros2 run --prefix python3 ycb_pointcloud_pipeline detection_replay --frames ~/ycb_detection_frames
# try a change without editing target_objects.yaml -- and judge it on every object:
ros2 run --prefix python3 ycb_pointcloud_pipeline detection_replay --frames ~/ycb_detection_frames \
    --prompt foam_brick="brown block" --threshold foam_brick=0.30
```

About 5 minutes on the GPU for the 157 frames. Found when present / published when absent:

| prompts | apple | mug | pudding box | brick | absent objects published |
|---|---|---|---|---|---|
| old: apple / red mug / brown chocolate pudding box (0.25) / red foam brick | 22/49 | 47/47 | 51/59 | 46/63 | **17** (mug on the apple 13, pudding box on the brick 4) |
| **new: red apple (0.41) / red cup with a handle (0.50) / small printed box (0.32) / dark brown block (0.26)** | **48/49** | 47/47 | 51/59 | 46/63 | **0** |

Each threshold sits between the object's lowest score when present and its highest when absent:
apple 0.43 / 0.39, mug 0.56 / 0.45, brick (never scored when absent). The pudding box has no gap
(0.28 present, 0.31 absent -- on a frame where a hand is placing the brick), so it is set at 0.32.
Other sets tried, and why not:

* one caption per prompt (`caption_mode:=per_object`): every pudding-box wording scored as high
  on frames without it (up to 0.66) as with it (10th percentile 0.47). A lone prompt always finds
  something.
* "brown block" for the brick found it in 62 of 63 frames -- beside "jello box", which found the
  pudding box in none. Beside "small printed box" it made the pudding-box prompt land on the brick
  (26 false pudding boxes), and the threshold that removes them halves the pudding box. **Prompts
  interact**: change one and re-run the replay.
* cropping the image to the pick area before detection (objects ~1.8x larger at the model's
  input) brought the apple to 49/49 and the brick to 63/63 but not the pudding box. Not adopted;
  the remaining lever if the brick's 73 % is not enough.

What is left: the brick is found in 46 of 63 frames and the pudding box in 51 of 59 -- misses,
not false detections, mostly the pudding box edge-on and frames with a hand in them. A miss costs
a planner one cycle; a false detection costs a grasp on the wrong object.

What size does **not** separate: objects of similar size. In the same run the apple's prompt landed
on the mug (0.40 against the mug's 0.42, the same box); the node kept the mug, which was right,
but by 0.02. That one is a prompt problem, for `prompt_bench`.

Earlier, from when the cracker box was listed: the phrase "cracker box" scored below 0.2 where
"small red box" scored 0.67, which is why prompts here describe what a thing **looks like**.

## Related

* `config/target_objects.yaml` -- the list, the prompts, the per-object thresholds.
* `CLAUDE.md`, "Object clouds are only as good as the mask" -- `mask_erosion`, and the flying
  pixels at a silhouette (`depth_gradient_max_mm_px`), which are about the *mask*, not the box.
* `docs/hardware/hardware_integration.md` section 4 -- verifying the cloud once it is published.
