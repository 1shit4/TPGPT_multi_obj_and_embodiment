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


def score_prompt(detector, images, targets, object_name, prompt, mode):
    """Best box score for ``object_name`` on every image, with ``prompt`` in place.

    The other targets keep their own prompts, so the joint caption is the one the
    node would really send.
    """
    from PIL import Image
    from ycb_pointcloud_pipeline.ycb_detector import build_captions

    swapped = [dict(t, prompt=prompt) if t['name'] == object_name else dict(t) for t in targets]
    captions = build_captions(swapped, mode)
    scores = []
    for _, rgb in images:
        pil = Image.fromarray(rgb)
        best = 0.0
        for caption, mapping in captions:
            if object_name not in mapping.values():
                continue                       # per_object: only that object's own pass matters
            found, _ = detector.boxes(pil, caption, mapping, rgb.shape[:2], floor=0.01)
            best = max(best, found.get(object_name, {}).get('score', 0.0))
        scores.append(best)
    return np.asarray(scores, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--frames', required=True, help='directory of *_rgb.png (save_frames_dir)')
    ap.add_argument('--object', required=True, help='the target to score, e.g. pudding_box')
    ap.add_argument('--prompts', nargs='+', default=None,
                    help='wordings to try; default: the one in target_objects.yaml')
    ap.add_argument('--modes', nargs='+', default=['joint', 'per_object'])
    ap.add_argument('--threshold', type=float, default=0.35)
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
            s = score_prompt(detector, images, targets, args.object, prompt, mode)
            rows.append((mode, prompt, s))
            print(f"{mode:10s} {prompt!r:42s} "
                  f"worst {s.min():.2f}  median {float(np.median(s)):.2f}  best {s.max():.2f}  "
                  f"clears {int((s >= args.threshold).sum())}/{len(s)}")

    print("\nper frame:")
    width = max(len(n) for n, _ in images)
    header = "  ".join(f"{m[:4]}:{p[:18]}" for m, p, _ in rows)
    print(f"{'frame':{width}s}  {header}")
    for i, (name, _) in enumerate(images):
        print(f"{name:{width}s}  " + "  ".join(f"{s[i]:>{len(f'{m[:4]}:{p[:18]}')}.2f}"
                                               for m, p, s in rows))
    best = max(rows, key=lambda r: (int((r[2] >= args.threshold).sum()), r[2].min()))
    print(f"\nmost robust here: {best[0]} caption, {best[1]!r} -- worst frame {best[2].min():.2f}, "
          f"clears {int((best[2] >= args.threshold).sum())} of {len(images)}")
    print("A prompt is only as good as its worst frame; one good frame proves nothing.")


if __name__ == '__main__':
    main()
