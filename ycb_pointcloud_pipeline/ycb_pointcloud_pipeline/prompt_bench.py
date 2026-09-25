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
import json
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


# ---------------------------------------------------------------- the replay
def replay(detector, targets, scenes, box_ok_for, empty=None):
    """What the node would publish on every labelled frame, and whether it is right.

    ``scenes`` maps a frame directory's name to ``(frames, present)`` where
    ``frames`` is ``[(tag, rgb, depth)]``; ``box_ok_for(depth)`` builds the
    node's size check for one frame. Returns per-frame records and, per target,
    ``found``/``present`` counts, the frames on which it was published although
    absent, and its scores when present and when absent (the margin a
    threshold has to sit in).
    """
    from PIL import Image

    empty = empty or {}
    names = [t['name'] for t in targets]
    stats = {n: {'found': 0, 'present': 0, 'false': [], 'present_scores': [],
                 'absent_scores': []} for n in names}
    records = []
    for scene, (frames, present) in scenes.items():
        for tag, rgb, depth in frames:
            here = [] if tag in empty.get(scene, ()) else list(present)
            kept = detector.detect_and_segment(rgb, box_ok=box_ok_for(depth), segment=False)
            published = {d['label'] for d in kept}
            for n in names:
                r = detector.last_report.get(n, {})
                score = r.get('score', 0.0)
                if n in here:
                    stats[n]['present'] += 1
                    stats[n]['found'] += n in published
                    stats[n]['present_scores'].append(score)
                else:
                    stats[n]['absent_scores'].append(score)
                    if n in published:
                        stats[n]['false'].append(f'{scene}/{tag}')
            records.append({'scene': scene, 'tag': tag, 'present': here,
                            'published': sorted(published),
                            'scores': {n: round(detector.last_report[n]['score'], 3)
                                       for n in names},
                            'best': getattr(detector, 'last_best', {})})
    return records, stats


def publish(best, thresholds, overlap=0.7):
    """What the node publishes from a frame's best boxes, for any thresholds.

    The same rule as :meth:`GroundedSAMDetector.detect_and_segment`: each target's
    best box if it clears the target's threshold, then, of two targets on one box
    (overlap at or above ``overlap``), the higher score.
    """
    dets = sorted(((n, b) for n, b in best.items() if b['score'] >= thresholds[n]),
                  key=lambda nb: -nb[1]['score'])
    kept = []
    for n, b in dets:
        if all(iou(b['box'], k['box']) < overlap for _, k in kept):
            kept.append((n, b))
    return {n for n, _ in kept}


def score_thresholds(records, thresholds):
    """``(found, present, false)`` per target for ``thresholds`` over replay records."""
    names = list(thresholds)
    out = {n: [0, 0, []] for n in names}
    for r in records:
        pub = publish(r.get('best') or {}, thresholds)
        for n in names:
            if n in r['present']:
                out[n][1] += 1
                out[n][0] += n in pub
            elif n in pub:
                out[n][2].append(f"{r['scene']}/{r['tag']}")
    return out


def sweep_thresholds(records, thresholds, grid=None, rounds=3):
    """Per-target thresholds that publish nothing absent and find the most.

    Coordinate search from ``thresholds``: each target's threshold in turn over
    ``grid``, scored by false detections first (any is worse than any number of
    misses), then found; ties go to the higher threshold, the safer one. Tuned on
    the frames it is judged on, so read the margins it reports, not only the
    counts.
    """
    grid = grid or [round(0.20 + 0.01 * i, 2) for i in range(41)]
    best = dict(thresholds)

    def cost(th):
        s = score_thresholds(records, th)
        return (sum(len(v[2]) for v in s.values()), -sum(v[0] for v in s.values()))

    for _ in range(rounds):
        for n in best:
            best[n] = min(grid, key=lambda g: (cost({**best, n: g}), -g))
    return best


def sweep_main(argv=None):
    """Re-score a detection_replay record for other thresholds, with no model.

        ros2 run --prefix python3 ycb_pointcloud_pipeline detection_sweep replay.json
    """
    ap = argparse.ArgumentParser(description=sweep_main.__doc__.splitlines()[0])
    ap.add_argument('replay_json')
    ap.add_argument('--threshold', action='append', default=[], metavar='NAME=VALUE',
                    help='score these thresholds instead of searching')
    args = ap.parse_args(argv)
    d = json.load(open(os.path.expanduser(args.replay_json)))
    records, used = d['records'], d.get('thresholds') or {}
    if not records or 'best' not in records[0]:
        raise SystemExit('this replay predates the per-box record; run detection_replay again')
    names = sorted({n for r in records for n in r['scores']})
    base = {n: float(used.get(n, 0.35)) for n in names}
    fixed = dict(base)
    for item in args.threshold:
        n, v = item.split('=', 1)
        fixed[n] = float(v)
    for label, th in (('as recorded', base), ('given' if args.threshold else 'searched',
                                               fixed if args.threshold else
                                               sweep_thresholds(records, base))):
        s = score_thresholds(records, th)
        print(f"{label:12s} " + "  ".join(f"{n} {th[n]:.2f}: {v[0]}/{v[1]}" + (f" +{len(v[2])} false" if v[2] else '')
                                        for n, v in s.items() if v[1] or v[2]))


def load_scenes(root, labels):
    """``{scene: ([(tag, rgb, depth)], present)}`` from ``root/<scene>/*_rgb.png``."""
    import cv2

    scenes = {}
    for scene, present in labels['scenes'].items():
        frames = []
        for path in sorted(glob.glob(os.path.join(root, scene, '*_rgb.png'))):
            depth_path = path[:-len('_rgb.png')] + '_depth.npy'
            bgr = cv2.imread(path)
            if bgr is None or not os.path.exists(depth_path):
                continue
            frames.append((os.path.basename(path)[:5], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                           np.load(depth_path)))
        scenes[scene] = (frames, list(present))
    return scenes


def replay_main(argv=None):
    """Replay the node's detection on recorded, labelled frames.

    Frames recorded with ``save_frames:=all`` into one directory per scene, and a
    ``labels.yaml`` beside them saying what was on the table in each::

        scenes:
          brick_only: [foam_brick]
          both_mug_apple: [mug, apple]
        empty:                      # frames with nothing on the table
          apple: ['00016']

    It runs exactly what the node runs -- the prompts, per-object thresholds and
    sizes from ``target_objects.yaml``, one caption, the size check on the saved
    depth, the same-box rule -- and reports, per object, how often it was found
    when on the table and every frame on which it was published when it was
    not. ``--prompt name=text`` and ``--threshold name=0.4`` try a change
    without editing the file; prompts interact through the shared caption, so
    judge a change on every object, not only the one changed.

        ros2 run --prefix python3 ycb_pointcloud_pipeline detection_replay \\
            --frames ~/ycb_detection_frames --prompt foam_brick="brown block"
    """
    import yaml

    from ycb_pointcloud_pipeline.segmentation_node import (_default_calibration_path,
                                                           _default_target_objects_path,
                                                           box_size_m, load_target_objects)
    from ycb_pointcloud_pipeline.ycb_detector import GroundedSAMDetector

    ap = argparse.ArgumentParser(description=replay_main.__doc__.splitlines()[0])
    ap.add_argument('--frames', required=True, help='directory holding the scene directories '
                                                     'and labels.yaml')
    ap.add_argument('--labels', default=None, help='default: <frames>/labels.yaml')
    ap.add_argument('--targets-yaml', default=None)
    ap.add_argument('--prompt', action='append', default=[], metavar='NAME=TEXT')
    ap.add_argument('--threshold', action='append', default=[], metavar='NAME=VALUE')
    ap.add_argument('--box-threshold', type=float, default=0.35)
    ap.add_argument('--size-margin', type=float, default=1.5)
    ap.add_argument('--caption-mode', default='joint')
    ap.add_argument('--grounding-model', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--json', default=None, help='write every frame\'s record here')
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.frames)
    labels = yaml.safe_load(open(args.labels or os.path.join(root, 'labels.yaml')))
    targets = load_target_objects(args.targets_yaml or _default_target_objects_path())
    for item in args.prompt:
        name, text = item.split('=', 1)
        next(t for t in targets if t['name'] == name)['prompt'] = text
    for item in args.threshold:
        name, value = item.split('=', 1)
        next(t for t in targets if t['name'] == name)['threshold'] = float(value)
    cal = yaml.safe_load(open(_default_calibration_path()))['camera_calibration']['intrinsics']
    fx, fy = float(cal['fx']), float(cal['fy'])
    limits = {t['name']: t['max_size'] * args.size_margin for t in targets if 'max_size' in t}

    def box_ok_for(depth):
        def ok(name, box):
            if name not in limits:
                return True, None
            size = box_size_m(box, depth, fx, fy)
            return (size is None or size <= limits[name]), size
        return ok

    scenes = load_scenes(root, labels)
    print(f"{sum(len(f) for f, _ in scenes.values())} frames in {len(scenes)} scenes from {root}")
    detector = GroundedSAMDetector(
        targets, grounding_model=args.grounding_model, box_threshold=args.box_threshold,
        caption_mode=args.caption_mode, load_sam=False,
        thresholds={t['name']: t['threshold'] for t in targets if 'threshold' in t})
    records, stats = replay(detector, targets, scenes, box_ok_for, labels.get('empty'))

    print(f"\n{'object':12s} {'prompt':26s} {'thr':>5s}  found when present   published when absent")
    for t in targets:
        s = stats[t['name']]
        pres = sorted(x for x in s['present_scores'] if x > 0)
        absent = max(s['absent_scores'], default=0.0)
        found = f"{s['found']}/{s['present']}" if s['present'] else '-'
        print(f"{t['name']:12s} {t['prompt'][:26]:26s} {detector.thresholds[t['name']]:5.2f}  "
              f"{found:>8s}            {len(s['false']):3d}"
              + (f"   e.g. {', '.join(s['false'][:3])}" if s['false'] else ''))
        if s['present']:
            print(f"{'':12s} scores when present: lowest {pres[0] if pres else 0:.2f}, "
                  f"median {float(np.median(pres)) if pres else 0:.2f};  highest when absent "
                  f"{absent:.2f}")
    total = sum(len(s['false']) for s in stats.values())
    print(f"\n{total} absent object(s) published in all. A miss costs the planner a cycle; a "
          "false detection costs a grasp on the wrong object.")
    if args.json:
        with open(os.path.expanduser(args.json), 'w') as f:
            json.dump({'records': records, 'thresholds': detector.thresholds,
                       'stats': {n: {k: v for k, v in s.items()} for n, s in stats.items()}},
                      f, indent=1, default=float)


if __name__ == '__main__':
    main()
