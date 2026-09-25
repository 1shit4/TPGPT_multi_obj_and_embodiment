# Detection: prompts, thresholds, and an object that comes and goes

`segmentation_node` is the first stage of everything: no cloud on
`/ycb_pointclouds/<object>`, no plan. Its detector is Grounding DINO (text prompt -> boxes)
followed by SAM (box -> mask), both from `config/target_objects.yaml`.

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

The first and the last look identical from the outside and have nothing in common.

## What the node reports now

Detection runs at a **report floor** (`detection_report_floor`, 0.05) and each target is accepted
at **its own threshold** afterwards. What is accepted is exactly what the single `box_threshold`
accepted before; what is new is that everything else leaves a trace:

```
pudding_box: best box scored 0.29 as 'pudding box', below its threshold of 0.35 -- not published
foam_brick: not seen at all this frame (nothing above 0.05)
pudding_box: detected at 0.41 over 3120 pixels, but no depth in the mask (within 2.0 m) -- nothing published
a box scoring 0.44 matched the phrase 'box', which fits two prompts equally well, ...
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
    --prompts "brown chocolate pudding box" "brown box" "pudding box" "small brown carton"
```

It scores each wording on every saved frame, in both caption modes, and prints the **worst** frame
per wording. That is the number that decides: a prompt whose floor is 0.45 is robust, a prompt that
reaches 0.8 on its best frame and 0.1 on its worst is the flakiness you started with.

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

Nothing yet on this bench: the report and the frame keeping went in on 2026-09-25, before any
frames had been saved. The file `config/target_objects.yaml` records one earlier observation, from
when the cracker box was in the list -- the phrase "cracker box" scored below 0.2 where "small red
box" scored 0.67 -- which is why prompts here describe what a thing **looks like** rather than what
it is called. Fill this section in once `prompt_bench` has run on saved frames.

## Related

* `config/target_objects.yaml` -- the list, the prompts, the per-object thresholds.
* `CLAUDE.md`, "Object clouds are only as good as the mask" -- `mask_erosion`, and the flying
  pixels at a silhouette (`depth_gradient_max_mm_px`), which are about the *mask*, not the box.
* `docs/hardware/hardware_integration.md` section 4 -- verifying the cloud once it is published.
