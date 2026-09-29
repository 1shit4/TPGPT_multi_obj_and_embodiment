"""Replaying a taught demonstration on the robot to learn its attractor: the passes. No ROS.

Step 5 of the execution-gap plan (``docs/hardware/FAITHFULNESS_AUDIT.md`` B1). The robot
side is ``tpgpt_executor_node`` in replay mode (``replay_dir``): it streams one pass's
attractor in time, under the demonstration's own stiffness labels and at the execution
speed, acts the gripper at the teacher's marks and records where the arm was. This module
keeps the passes and computes the next attractor between them (:mod:`tpgpt.hardware.ilc`).

A replay directory, one per demonstration::

    replay.json               the bundle, the speed, the contact windows, the ILC settings
    pass_000/attractor.npy    (N, 3) what pass 0 commands: the taught path itself
    pass_000/recording.npz    what the arm did (written by the executor node, replay mode)
    pass_000/report.json      the pass's error and the update made from it
    pass_001/attractor.npy    the next attractor
    ...

Command line::

    python -m tpgpt.hardware.replay init --bundle <source_bundle> --out <dir> [--speed-scale 0.35]
    python -m tpgpt.hardware.replay update <dir> [--gain 0.5]
    python -m tpgpt.hardware.replay status <dir>

The label period of a replay is ``dt / speed_scale`` (0.05 / 0.35 = 0.143 s): the executor
must run the resulting labels at the same ``speed_scale``, since the lag it learns grows
with speed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from tpgpt.hardware.ilc import IlcSettings, frozen_mask, ilc_update, label_positions, residuals_at
from tpgpt.hardware.source_bundle import SourceBundle

#: Contact windows around the teacher's marks, in labels relative to them (20 Hz labels),
#: where the update is bridged, not learned (:mod:`tpgpt.hardware.ilc`):
#: from 0.5 s before the grasp mark to 0.5 s after it -- the final descent, where the fingers
#: may meet the object or the table, and the squeeze -- and from 0.5 s before the release mark
#: to 0.25 s after (the object meets the board). The grasp's window starts before the mark
#: because the 18:39 demonstration's grasp sits less than 17 mm above the table (the path
#: check found a pad in the table with the path 17 mm lower there): learning the descent could
#: press the fingers into the table. The cost: the grasp is corrected only by the bridge.
GRASP_WINDOW = (-10, 10)
RELEASE_WINDOW = (-10, 5)


def marks(gripper) -> tuple[int, int]:
    """``(grasp, release)``: the first and the last label with the jaws closed."""
    closed = np.flatnonzero(np.asarray(gripper) > 0)
    if not len(closed):
        raise ValueError("the demonstration never closes the gripper")
    return int(closed[0]), int(closed[-1])


def closing_width(bundle: SourceBundle) -> float:
    """Width of the object across the jaws at the taught grasp, m (for the gripper goal)."""
    placement = bundle.placement
    points = np.asarray(placement.points, dtype=float).reshape(-1, 3)
    closing = np.asarray(placement.grasp.closing, dtype=float)
    closing = closing / max(np.linalg.norm(closing), 1e-12)
    near = np.linalg.norm(points - placement.grasp.tcp, axis=1) < 0.05
    along = (points[near] if near.sum() >= 10 else points) @ closing
    return float(along.max() - along.min())


def attractor_at(attractor, s: float) -> np.ndarray:
    """The attractor at fractional label index ``s`` (linear between labels)."""
    a = np.asarray(attractor, dtype=float)
    s = float(np.clip(s, 0.0, len(a) - 1))
    i = min(int(np.floor(s)), len(a) - 2) if len(a) > 1 else 0
    w = s - i
    return a[i] if len(a) == 1 else a[i] + w * (a[i + 1] - a[i])


# ------------------------------------------------------------------------ the directory
def _pass_dir(root: Path, k: int) -> Path:
    return Path(root) / f"pass_{k:03d}"


def init_replay(bundle_dir, out_dir, speed_scale: float = 0.35,
                settings: IlcSettings = IlcSettings()) -> Path:
    """Start a replay: pass 0 commands the taught path itself."""
    bundle_dir, out = Path(bundle_dir).expanduser().resolve(), Path(out_dir).expanduser()
    if (out / "replay.json").exists():
        raise FileExistsError(f"{out} already holds a replay")
    bundle = SourceBundle.load(bundle_dir)
    labels = bundle.tool_labels
    grasp, release = marks(labels.gripper)
    n = len(labels.positions)
    windows = [(grasp + GRASP_WINDOW[0], grasp + GRASP_WINDOW[1]),
               (release + RELEASE_WINDOW[0], release + RELEASE_WINDOW[1])]
    info = {
        "bundle": str(bundle_dir), "n_labels": n,
        "dt": float(labels.metadata.get("dt", 1.0 / float(bundle.metadata.get("control_freq", 20)))),
        "speed_scale": float(speed_scale), "grasp_index": grasp, "release_index": release,
        "frozen_windows": [[int(a), int(b)] for a, b in windows],
        "object_width_closing_m": closing_width(bundle),
        "stiffness": bundle.metadata.get("stiffness", "static"),
        "ilc": dataclasses.asdict(settings),
    }
    _pass_dir(out, 0).mkdir(parents=True)
    np.save(_pass_dir(out, 0) / "attractor.npy", np.asarray(labels.positions, dtype=float))
    (out / "replay.json").write_text(json.dumps(info, indent=2))
    return out


def load_replay(out_dir) -> tuple[dict, SourceBundle]:
    out = Path(out_dir).expanduser()
    info = json.loads((out / "replay.json").read_text())
    return info, SourceBundle.load(info["bundle"])


def passes(out_dir) -> list[int]:
    return sorted(int(p.name[5:]) for p in Path(out_dir).expanduser().glob("pass_[0-9][0-9][0-9]")
                  if (p / "attractor.npy").exists())


def pending_pass(out_dir) -> int:
    """The pass to run next: the last one with an attractor and no recording."""
    ks = passes(out_dir)
    if not ks:
        raise FileNotFoundError(f"{out_dir} holds no pass; run `replay init` first")
    k = ks[-1]
    if (_pass_dir(out_dir, k) / "recording.npz").exists():
        raise FileExistsError(f"pass {k} is already recorded: run "
                              f"`python -m tpgpt.hardware.replay update {out_dir}` first")
    return k


def load_attractor(out_dir, k: int) -> np.ndarray:
    return np.load(_pass_dir(out_dir, k) / "attractor.npy")


def save_recording(out_dir, k: int, **arrays) -> Path:
    """What the arm did in pass ``k``: at least ``label_index`` and ``positions`` (grasp point)."""
    path = _pass_dir(out_dir, k) / "recording.npz"
    np.savez_compressed(path, **{name: np.asarray(v) for name, v in arrays.items()})
    return path


def pass_error(out_dir, k: int) -> dict:
    """How far pass ``k``'s arm was from the taught path, overall and at the two marks."""
    info, bundle = load_replay(out_dir)
    rec = np.load(_pass_dir(out_dir, k) / "recording.npz")
    labels = bundle.tool_labels
    measured = label_positions(rec["label_index"], rec["positions"], info["n_labels"])
    frozen = frozen_mask(info["n_labels"], info["frozen_windows"])
    e = np.linalg.norm(np.asarray(labels.positions) - measured, axis=1)
    return {
        "error_rms_mm": float(np.sqrt(np.mean(e[~frozen] ** 2)) * 1000),
        "error_max_mm": float(e[~frozen].max() * 1000),
        "at_marks": residuals_at(labels.positions, measured, labels.orientations,
                                 {"grasp": info["grasp_index"], "release": info["release_index"]}),
    }, measured


def update(out_dir, gain: float | None = None) -> dict:
    """From the last recorded pass, write the next pass's attractor and this pass's report."""
    out = Path(out_dir).expanduser()
    info, bundle = load_replay(out)
    recorded = [k for k in passes(out) if (_pass_dir(out, k) / "recording.npz").exists()]
    if not recorded:
        raise FileNotFoundError(f"no recorded pass in {out}; run the replay first")
    k = recorded[-1]
    if (_pass_dir(out, k + 1) / "attractor.npy").exists():
        raise FileExistsError(f"pass {k + 1} already has its attractor")
    settings = IlcSettings(**info["ilc"])
    if gain is not None:
        settings = dataclasses.replace(settings, gain=float(gain))
    error, measured = pass_error(out, k)
    new, report = ilc_update(load_attractor(out, k), measured, bundle.tool_labels.positions,
                             info["dt"] / info["speed_scale"],
                             frozen_mask(info["n_labels"], info["frozen_windows"]), settings)
    _pass_dir(out, k + 1).mkdir(parents=True, exist_ok=True)
    np.save(_pass_dir(out, k + 1) / "attractor.npy", new)
    report = {"pass": k, **error, "update": report, "gain": settings.gain}
    (_pass_dir(out, k) / "report.json").write_text(json.dumps(report, indent=2))
    return report


def check_pass(out_dir, k: int, shelf_config) -> tuple[str | None, dict]:
    """The pass's attractor path against the shelf model and the table: ``(verdict, check)``.

    ``verdict`` is ``None`` when admissible (``path_check.verdict``), else why not. The
    attractor, not the taught path, is what the spring pulls toward, and learning moves it
    from pass to pass, so every pass is checked before it runs. No camera cloud: the scene
    is the measured shelf model and the table (anything else in the way is the operator's).
    """
    import yaml

    from tpgpt.hardware import path_check
    from tpgpt.hardware.end_effector import end_effector_for_plan
    from tpgpt.hardware.fr3 import HOME_Q
    from tpgpt.hardware.shelf import ShelfModel, ShelfWorld

    info, bundle = load_replay(out_dir)
    labels = bundle.tool_labels
    cfg = yaml.safe_load(open(Path(shelf_config).expanduser()))
    world = ShelfWorld(model=ShelfModel.from_config(cfg), cloud=np.zeros((0, 3)),
                       table_height=float(cfg["shelf"]["anchors"]["table"]))
    hand = end_effector_for_plan(bundle.metadata)
    check = path_check.path_feasibility(
        load_attractor(out_dir, k), hand.ee_from_grasp(labels.orientations), labels.gripper, hand,
        world, bundle.placement.points, info["grasp_index"], info["release_index"],
        closed_width=float(info["object_width_closing_m"]), start_q=HOME_Q,
        destination_height=float(bundle.placement.destination_height))
    return path_check.verdict(check), check


def status(out_dir) -> list[dict]:
    out = Path(out_dir).expanduser()
    rows = []
    for k in passes(out):
        row = {"pass": k, "recorded": (_pass_dir(out, k) / "recording.npz").exists()}
        report = _pass_dir(out, k) / "report.json"
        if report.exists():
            r = json.loads(report.read_text())
            row.update({key: r[key] for key in ("error_rms_mm", "error_max_mm", "at_marks", "gain")})
        rows.append(row)
    return rows


def _print_status(out_dir):
    for row in status(out_dir):
        line = f"pass {row['pass']:3d}: {'recorded' if row['recorded'] else 'to run'}"
        if "error_rms_mm" in row:
            m = row["at_marks"]
            line += (f"; error rms {row['error_rms_mm']:.1f} mm, max {row['error_max_mm']:.1f}; "
                     f"grasp {m['grasp']['total_mm']:.1f} (closing {m['grasp']['closing_mm']:.1f}), "
                     f"release {m['release']['total_mm']:.1f} (closing "
                     f"{m['release']['closing_mm']:.1f}); next pass at gain {row['gain']}")
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay passes of a taught demonstration (ILC)")
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init", help="start a replay; pass 0 commands the taught path")
    a.add_argument("--bundle", required=True, help="the demonstration's source_bundle directory")
    a.add_argument("--out", required=True, help="the replay directory to create")
    a.add_argument("--speed-scale", type=float, default=0.35,
                   help="the executor's speed_scale; the replay learns the lag at this speed")
    u = sub.add_parser("update", help="compute the next pass's attractor from the last recording")
    u.add_argument("dir")
    u.add_argument("--gain", type=float, default=None, help="ILC gain for this update (default 0.5)")
    s = sub.add_parser("status", help="the passes and their errors")
    s.add_argument("dir")
    args = ap.parse_args(argv)
    if args.command == "init":
        out = init_replay(args.bundle, args.out, args.speed_scale)
        info, _ = load_replay(out)
        print(f"replay at {out}: {info['n_labels']} labels, grasp {info['grasp_index']}, release "
              f"{info['release_index']}, contact windows {info['frozen_windows']}, speed_scale "
              f"{info['speed_scale']} ({info['n_labels'] * info['dt'] / info['speed_scale']:.0f} s "
              "per pass). Pass 0 is ready.")
    elif args.command == "update":
        report = update(args.dir, args.gain)
        print(f"pass {report['pass']}: error rms {report['error_rms_mm']:.1f} mm; next attractor "
              f"moved up to {report['update']['update_max_mm']:.1f} mm "
              f"(capped: {report['update']['capped_update']} labels this pass, "
              f"{report['update']['capped_total']} at the 60 mm limit)")
        _print_status(args.dir)
    else:
        _print_status(args.dir)


if __name__ == "__main__":
    main()
