"""Score candidate Grounding DINO prompts on frames the camera already saw.

A detector that finds an object on one frame and not the next cannot be fixed by
trying another wording at the robot: one run says nothing about a distribution,
and the scene changes between runs. So the segmentation node keeps the frames it
had trouble with (``save_frames:=misses``), and this replays them offline, with
no camera and no robot:

* how the object's **current** prompt scores on every saved frame, so a miss can
  be read as a near miss (0.34 against a threshold of 0.35) or as an object the
  model never saw (0.02);
* how **other wordings** score on those same frames;
* both **caption modes** -- the prompt sharing one caption with every other
  target, or alone -- because Grounding DINO scores a phrase in the context of
  the whole caption, and that is a competition the rare object can lose.

The answer to read off is not the best single score but the **worst** score over
the frames: what makes a prompt robust is that its floor clears the threshold.

    ros2 run --prefix python3 ycb_pointcloud_pipeline prompt_bench \
        --frames ~/ycb_detection_frames --object pudding_box \
        --prompts "brown chocolate pudding box" "brown box" "pudding box"

Run it in the segmentation venv (``~/venvs/ycb``): it loads Grounding DINO, but
not SAM -- the score is the box's, and SAM only turns a box into a mask.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def frames(directory):
    """``[(name, rgb)]`` for every ``*_rgb.png`` in ``directory``, sorted."""
    import cv2

    found = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(directory), '*_rgb.png'))):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        found.append((os.path.basename(path)[:-8], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return found


def iou(a, b):
    """Overlap of two boxes, 0 when either is missing."""
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_prompt(detector, images, targets, object_name, prompt, mode, overlap=0.5):
    """What ``prompt`` does to every target, on every image, in place of its own.

    The other targets keep their prompts, so a joint caption is the one the node
    would really send. Two numbers come back per frame, because a prompt can fail
    in two opposite ways:

    * ``score`` -- the best box for ``object_name``. Too low and the object goes
      missing, which is the failure we started from.
    * ``confused`` -- that box landing on **another target's** box (overlap above
      ``overlap``), or another target's prompt landing on this one. A wording
      broad enough to be found every time ("brown box") is also broad enough to
      be found on the wrong object, and on this bench two of the five objects are
      boxes. A prompt that scores 0.9 while sitting on the foam brick is worse
      than one that scores 0.4 on the right thing.
    """
    from PIL import Image
    from ycb_pointcloud_pipeline.ycb_detector import build_captions

    swapped = [dict(t, prompt=prompt) if t['name'] == object_name else dict(t) for t in targets]
    captions = build_captions(swapped, mode)
    scores, confused, took = [], [], []
    for _, rgb in images:
        pil = Image.fromarray(rgb)
        found = {}
        for caption, mapping in captions:
            part, _ = detector.boxes(pil, caption, mapping, rgb.shape[:2], floor=0.01)
            for name, d in part.items():
                if name not in found or d['score'] > found[name]['score']:
                    found[name] = d
        mine = found.get(object_name)
        scores.append(0.0 if mine is None else mine['score'])
        clashes = [n for n, d in found.items()
                   if n != object_name and iou(d['box'], (mine or {}).get('box')) > overlap]
        confused.append(bool(clashes))
        took.append(",".join(clashes))
    return np.asarray(scores, float), np.asarray(confused, bool), took


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--frames', required=True, help='directory of *_rgb.png (save_frames_dir)')
    ap.add_argument('--object', required=True, help='the target to score, e.g. pudding_box')
    ap.add_argument('--prompts', nargs='+', default=None,
                    help='wordings to try; default: the one in target_objects.yaml')
    ap.add_argument('--modes', nargs='+', default=['joint', 'per_object'])
    ap.add_argument('--threshold', type=float, default=0.35)
    ap.add_argument('--overlap', type=float, default=0.5,
                    help='box overlap at which two targets count as confused')
    ap.add_argument('--targets-yaml', default=None)
    ap.add_argument('--grounding-model', default='IDEA-Research/grounding-dino-base')
    args = ap.parse_args(argv)

    from ycb_pointcloud_pipeline.segmentation_node import (_default_target_objects_path,
                                                           load_target_objects)
    from ycb_pointcloud_pipeline.ycb_detector import GroundedSAMDetector

    targets = load_target_objects(args.targets_yaml or _default_target_objects_path())
    names = [t['name'] for t in targets]
    if args.object not in names:
        raise SystemExit(f"{args.object!r} is not in the target list {names}")
    images = frames(args.frames)
    if not images:
        raise SystemExit(f"no *_rgb.png in {args.frames}")
    current = next(t['prompt'] for t in targets if t['name'] == args.object)
    prompts = args.prompts or [current]

    print(f"{len(images)} frame(s) from {args.frames}")
    print(f"targets: {names}")
    print(f"{args.object}: current prompt {current!r}, threshold {args.threshold:.2f}\n")

    detector = GroundedSAMDetector(targets, grounding_model=args.grounding_model,
                                   box_threshold=args.threshold, load_sam=False)
    rows = []
    for mode in args.modes:
        for prompt in prompts:
            s, confused, clashes = score_prompt(detector, images, targets, args.object, prompt,
                                                mode, overlap=args.overlap)
            rows.append((mode, prompt, s, confused))
            note = ''
            if confused.any():
                who = sorted({c for c in clashes if c})
                note = f"  CONFUSED with {'/'.join(who)} on {int(confused.sum())} frame(s)"
            print(f"{mode:10s} {prompt!r:42s} "
                  f"worst {s.min():.2f}  median {float(np.median(s)):.2f}  best {s.max():.2f}  "
                  f"clears {int((s >= args.threshold).sum())}/{len(s)}{note}")

    print("\nper frame ('*' = the box sat on another target):")
    width = max(len(n) for n, _ in images)
    cols = [f"{m[:4]}:{p[:18]}" for m, p, _, _ in rows]
    print(f"{'frame':{width}s}  " + "  ".join(cols))
    for i, (name, _) in enumerate(images):
        cells = [f"{s[i]:.2f}{'*' if c[i] else ''}" for (_, _, s, c) in rows]
        print(f"{name:{width}s}  " + "  ".join(f"{cell:>{len(col)}s}"
                                               for cell, col in zip(cells, cols)))
    # Ranked by frames cleared, then by the worst frame -- but a wording that ever
    # sat on another object is not a candidate at all, however well it scores.
    clean = [r for r in rows if not r[3].any()] or rows
    best = max(clean, key=lambda r: (int((r[2] >= args.threshold).sum()), r[2].min()))
    print(f"\nmost robust here: {best[0]} caption, {best[1]!r} -- worst frame {best[2].min():.2f}, "
          f"clears {int((best[2] >= args.threshold).sum())} of {len(images)}"
          + ("" if not best[3].any() else "  (and it is confused on some frames: none was clean)"))
    print("A prompt is only as good as its worst frame; one good frame proves nothing.")
    print("Check the frames it is confused on before trusting any of this: the bench compares "
          "boxes, it does not know which object is which.")


if __name__ == '__main__':
    main()
