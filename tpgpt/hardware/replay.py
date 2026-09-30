"""Replaying a taught demonstration on the robot to learn its attractor: the passes. No ROS.

Step 5 of the execution-gap plan (``docs/hardware/FAITHFULNESS_AUDIT.md`` B1). The robot
side is ``tpgpt_executor_node`` in replay mode (``replay_dir``): it streams one pass's
attractor -- position and orientation -- in time, under the demonstration's own stiffness
labels and at the execution speed, acts the gripper at the teacher's marks and records
where the arm was. This module keeps the passes and computes the next attractor between
them (:mod:`tpgpt.hardware.ilc`).

A replay directory, one per demonstration::

    replay.json                 the bundle, the speed, the contact windows, the ILC settings,
                                how many times each pass is run
    pass_000/attractor.npy      (N, 3) what pass 0 commands: the taught path itself
    pass_000/attractor_rot.npy  (N, 3, 3) its orientations (the plan's grasp convention)
    pass_000/margin.npz         the gap the pass's path check asked for (see below)
    pass_000/recording_1.npz    what the arm did (written by the executor node, replay mode)
    pass_000/recording_2.npz    ... when the pass is run again with the same attractor
    pass_000/report.json        the pass's error and the update made from it
    pass_001/...                the next attractor, and so on

**Runs.** Pass 0 is run twice by default (``init --pass0-runs``). The two runs command the
same attractor, so how much they differ is the arm's run-to-run spread: what no learning
can remove, and so part of every later margin. The update learns from their average,
which also halves the noise it would learn. ``repeat`` adds a run to the last pass.

**The path check before every pass is on the taught path, with a margin** -- not on the
attractor. The attractor is where the spring pulls, not where the arm goes: the arm lags
it along the path and learning moves it off the path on purpose. What the arm traces is the
taught path plus how far it strays from it, so that is what is checked
(``path_check.path_feasibility`` with ``clearance`` and ``tilt``):

* **clearance**: how far sideways the arm may be off the taught path. Sideways only: an
  arm that is late but on the path is somewhere the check has already cleared.
* **tilt**: how far the hand may be turned from the taught orientation; each robot piece
  then needs the angle times its own distance from the grasp point.

Pass 0's margin comes from the stiffness checks, not from this path: 6.5 mm sideways (the
worst they saw across the motion, 5.3-6.4 mm on 2026-09-29, 4.9-5.8 on 2026-09-30) and the
tilt the rotational spring leaves at the stiffness each label runs at -- the resting tilt
after a move plus the turn rate times the turning time constant, both measured at 15, 30, 45
and 60 N m/rad on 2026-09-30 (:data:`MEASURED_ROTATION`). ``use-checks`` replaces them with
other stiffness checks. Every later pass's margin is **measured**: the previous pass's
worst sideways stray and tilt around each label (over ``MARGIN_WINDOW`` labels either
side) plus the run-to-run spread. If learning works the next pass strays less; if it
overshoots, the stray moves to the other side at about the same size. Either way the margin
covers it -- and after each pass the stray it actually had is compared with the margin it
was checked with (``pass_error``: ``margin_exceeded``), so the assumption is tested every
time. No margin is smaller, and no excess is reported, below what the check itself
resolves (``MARGIN_FLOOR``, ``TILT_FLOOR``: 1 mm and 0.6 deg).

Command line::

    python -m tpgpt.hardware.replay init --bundle <source_bundle> --out <dir> [--speed-scale 0.35]
    python -m tpgpt.hardware.replay update <dir> [--gain 1.0]
    python -m tpgpt.hardware.replay repeat <dir>
    python -m tpgpt.hardware.replay use-checks <dir> <stiffness check dir> ...
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

from tpgpt.hardware.ilc import (IlcSettings, frozen_mask, ilc_rotation_update, ilc_update,
                                label_positions, label_rotations, residuals_at, rotation_errors)
from tpgpt.hardware.source_bundle import SourceBundle

#: Contact windows around the teacher's marks, in labels relative to them (20 Hz labels),
#: where the update is bridged, not learned (:mod:`tpgpt.hardware.ilc`):
#: from 0.5 s before the grasp mark to 0.5 s after it -- the final descent, where the fingers
#: may meet the object or the table, and the squeeze -- and from 0.5 s before the release mark
#: to 0.25 s after (the object meets the board). The grasp's window starts before the mark
#: because the 18:39 demonstration's grasp sits less than 17 mm above the table (the path
#: check found a pad in the table with the path 17 mm lower there): learning the descent could
#: press the fingers into the table. The cost: the grasp is corrected only by the bridge.
#: A bridge carries what the arm needed at the window's two edges across it, so the hand
#: should be nearly still there; ``init`` reports the taught speed at the edges (the 18:39
#: demonstration: at most 4.6 mm/s at execution speed, a lag of ~1.4 mm).
GRASP_WINDOW = (-10, 10)
RELEASE_WINDOW = (-10, 5)
#: Above this taught speed at a window's edge (execution speed, m/s), ``init`` warns.
EDGE_SPEED_WARNING = 0.010

#: Pass 0's margin, before anything on this path is measured (module docstring):
#: sideways, m -- the worst the stiffness checks saw across the motion (5.3-6.4 mm,
#: ~/tpgpt_stiffness_checks/20260929_18*; 4.9-5.8 mm, 20260930_13*);
FIRST_PASS_STRAY = 0.0065
#: and per rotational stiffness (N m/rad), measured 2026-09-30 (audit B2 step 3b,
#: ~/tpgpt_stiffness_checks/20260930_13*): the resting tilt after a move (the largest of a
#: check's holds' medians) and the turning time constant (the larger of its two turns').
#: The runs asked for 80 and 100 ran at the cap, 60, and are counted there.
MEASURED_ROTATION = (
    {"k": 15.0, "tilt_deg": 2.47, "tau_s": 0.76},
    {"k": 30.0, "tilt_deg": 1.32, "tau_s": 0.60},
    {"k": 45.0, "tilt_deg": 1.03, "tau_s": 0.44},
    {"k": 60.0, "tilt_deg": 0.90, "tau_s": 0.36},
)
#: A measured margin is the worst stray within this many labels either side (0.7 s at
#: speed_scale 0.35): the arm's stray at one label is really about where it was, a little
#: earlier or later along the path.
MARGIN_WINDOW = 5
#: The variable-stiffness controller's cap on rotational stiffness, N m/rad
#: (``variable_max_stiffness.rotational``, config/tpgpt_variable_stiffness_controller.yaml).
ROTATIONAL_CAP = 60.0
#: Where along the taught path a measured pose is matched (labels before, after): the
#: arm lags, so its nearest taught point lies behind the label it was commanded.
NEAREST_WINDOW = (30, 10)
#: The smallest margin, and the smallest excess over it worth a warning: the path check
#: poses the arm only to 1 mm and 0.6 deg (``path_check.POSE_TOLERANCE``), so a smaller
#: margin claims more than the check knows, and a smaller excess is inside its error.
MARGIN_FLOOR = 0.001
TILT_FLOOR = 0.01


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


# ------------------------------------------------------------------------ strays and margins
def _running_max(x: np.ndarray, half: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(np.asarray(x, dtype=float), 2 * int(half) + 1, mode="nearest")


def strays(reference, reference_rot, measured, measured_rot,
           window: tuple[int, int] = NEAREST_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """How far the arm was off the taught path, per label: ``(sideways m, tilt rad)``.

    Sideways: the distance from the measured position to the nearest point of the taught
    path, searched ``window`` labels behind and ahead. Tilt: the angle between the measured
    orientation and the taught one *at that nearest point* -- the orientation the teacher
    had where the arm actually was.
    """
    from tpgpt.hardware.frames import slerp_rotation

    P = np.asarray(reference, dtype=float).reshape(-1, 3)
    R = np.asarray(reference_rot, dtype=float).reshape(-1, 3, 3)
    M = np.asarray(measured, dtype=float).reshape(-1, 3)
    MR = np.asarray(measured_rot, dtype=float).reshape(-1, 3, 3)
    n = len(P)
    side, tilt = np.zeros(n), np.zeros(n)
    for i in range(n):
        lo, hi = max(i - window[0], 0), min(i + window[1], n - 1)
        if hi == lo:
            j, t, d = lo, 0.0, float(np.linalg.norm(M[i] - P[lo]))
        else:
            a, b = P[lo:hi], P[lo + 1:hi + 1]
            ab = b - a
            tt = np.clip(((M[i] - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
            dist = np.linalg.norm(a + tt[:, None] * ab - M[i], axis=1)
            s = int(np.argmin(dist))
            j, t, d = lo + s, float(tt[s]), float(dist[s])
        side[i] = d
        R_near = slerp_rotation(R[j], R[min(j + 1, n - 1)], t)
        tilt[i] = float(np.linalg.norm(rotation_errors(R_near[None], MR[i][None])[0]))
    return side, tilt


def model_from_checks(check_dirs) -> dict:
    """What stiffness checks measured, as pass 0's margin needs it.

    Each ``cartesian_target_publisher`` stiffness check (``report.json``) gives, at its
    rotational stiffness: the resting tilt (the largest of its holds' median tilts), the
    turning time constant (the larger of its two turns') and the sideways stray (the
    largest across the motion of its moves). A check that was cut short is left out.
    Returns ``{"stray_m", "rotation": [{"k",
    "tilt_deg", "tau_s"}] sorted by k, "checks"}``; a check without a turn gives no tau.
    """
    rows, stray, used = {}, 0.0, []
    for d in check_dirs:
        report = json.loads((Path(d).expanduser() / "report.json").read_text())
        if report.get("aborted"):
            continue                    # cut short (a vibration, Ctrl-C): not a stiffness to use
        segs = report.get("segments", {})
        tilts = [s["tilt_median_deg"] for s in segs.values() if "tilt_median_deg" in s]
        taus = [s["tau_s"] for s in segs.values() if "tau_s" in s]
        stray = max([stray] + [s["across_max_mm"] / 1000 for s in segs.values()
                               if "across_max_mm" in s])
        if not tilts or not taus:
            continue
        # What the arm ran at: reports before 2026-09-30 13:30 named what was asked, which the
        # controller clamps to its cap (runs asked for 80 and 100 ran at 60).
        k = float(report.get("rotational_stiffness_applied",
                             min(float(report["rotational_stiffness"]), ROTATIONAL_CAP)))
        old = rows.get(k, {"tilt_deg": 0.0, "tau_s": 0.0})
        rows[k] = {"k": k, "tilt_deg": max(old["tilt_deg"], max(tilts)),
                   "tau_s": max(old["tau_s"], max(taus))}
        used.append(str(Path(d).expanduser()))
    if not rows:
        raise ValueError("none of these stiffness checks measured a tilt and a turn")
    return {"stray_m": stray or FIRST_PASS_STRAY, "rotation": [rows[k] for k in sorted(rows)],
            "checks": used}


def _tilt_and_tau(k: np.ndarray, model: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Resting tilt (rad) and turning time constant (s) at rotational stiffness ``k``.

    From ``model`` (:func:`model_from_checks`), else :data:`MEASURED_ROTATION`: interpolated
    between the stiffnesses measured; below the softest, scaled up from it (``1/k``,
    ``1/sqrt(k)``); above the stiffest, its values -- never extrapolated to less than was
    measured.
    """
    model = model or {"rotation": MEASURED_ROTATION}
    ks = np.array([r["k"] for r in model["rotation"]])
    tilt = np.radians([r["tilt_deg"] for r in model["rotation"]])
    tau = np.array([r["tau_s"] for r in model["rotation"]])
    t = np.interp(k, ks, tilt)
    s = np.interp(k, ks, tau)
    soft = k < ks[0]
    t[soft] = tilt[0] * ks[0] / k[soft]
    s[soft] = tau[0] * np.sqrt(ks[0] / k[soft])
    return t, s


def first_pass_margin(orientations, rotational, label_period: float, model: dict | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Pass 0's margin: ``(clearance m, tilt rad)`` per label.

    ``rotational``: the rotational stiffness each label runs at, N m/rad (the executor's
    schedule, soft while carrying). Tilt = the resting tilt at that stiffness plus the
    turn rate times the turning time constant (:func:`_tilt_and_tau`); sideways, the
    stray the stiffness checks measured. ``model`` is what other checks measured
    (:func:`model_from_checks`); without it, :data:`FIRST_PASS_STRAY` and
    :data:`MEASURED_ROTATION`.
    """
    from scipy.spatial.transform import Rotation

    R = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    k = np.maximum(np.asarray(rotational, dtype=float).reshape(-1), 1e-3)
    step = np.linalg.norm(Rotation.from_matrix(
        R[1:] @ np.transpose(R[:-1], (0, 2, 1))).as_rotvec(), axis=1)
    rate = _running_max(np.r_[0.0, step] / float(label_period), MARGIN_WINDOW)
    resting, tau = _tilt_and_tau(k, model)
    stray = float(model["stray_m"]) if model else FIRST_PASS_STRAY
    return np.full(len(R), stray), _running_max(resting + rate * tau, MARGIN_WINDOW)


# ------------------------------------------------------------------------ the object's place
#: How close the object must stand to where it was taught before a pass with it starts: its
#: centre in the table plane, and its heading. The jaws have about (80 - 37) / 2 = 21 mm of room
#: either side of the 18:39 banana; the scan fit itself moves a few mm between camera frames.
PLACE_TOLERANCE_M = 0.010
PLACE_TOLERANCE_DEG = 10.0


def object_name(info: dict) -> str:
    """The demonstration's object: its directory is named ``<object>_<date>_<time>``."""
    return Path(info["bundle"]).parent.name.rsplit("_", 2)[0]


def placement_offset(taught, now) -> dict:
    """How the object stands now against where it was taught, both from scan fits
    (``object_model.fit_object``): what to do to put it back.

    ``move_x``/``move_y`` (m, base frame: +x away from the robot, +y to its left) and
    ``turn_deg`` (about the vertical, counter-clockwise seen from above) take it from
    where it is to where it was; ``same_side`` is whether it rests on the same side.
    """
    move = taught.full_cloud.mean(axis=0) - now.full_cloud.mean(axis=0)
    R = np.asarray(taught.rotation) @ np.asarray(now.rotation).T
    turn = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    same_side = int(taught.rest_index) == int(now.rest_index)
    distance = float(np.hypot(move[0], move[1]))
    return {"move_x": float(move[0]), "move_y": float(move[1]), "distance_m": distance,
            "turn_deg": turn, "same_side": same_side,
            "in_place": bool(same_side and distance <= PLACE_TOLERANCE_M
                             and abs(turn) <= PLACE_TOLERANCE_DEG)}


def placement_advice(offset: dict) -> str:
    """:func:`placement_offset` in words, for the operator at the table."""
    if not offset["same_side"]:
        return "it lies on a different side than when taught: turn it over"
    words = []
    x, y, turn = offset["move_x"] * 1000, offset["move_y"] * 1000, offset["turn_deg"]
    if abs(x) >= 2:
        words.append(f"{abs(x):.0f} mm {'away from' if x > 0 else 'toward'} the robot")
    if abs(y) >= 2:
        words.append(f"{abs(y):.0f} mm to the robot's {'left' if y > 0 else 'right'}")
    move = "move it " + " and ".join(words) if words else ""
    rot = (f"turn it {abs(turn):.0f} deg {'counter-clockwise' if turn > 0 else 'clockwise'} "
           "(seen from above)") if abs(turn) >= 2 else ""
    todo = ", ".join(t for t in (move, rot) if t)
    where = f"{offset['distance_m'] * 1000:.0f} mm and {abs(turn):.0f} deg from its taught place"
    if offset["in_place"]:
        return f"in place ({where})" + (f"; closer still: {todo}" if todo else "")
    return f"{where}: {todo}"


# ------------------------------------------------------------------------ the directory
def _pass_dir(root: Path, k: int) -> Path:
    return Path(root).expanduser() / f"pass_{k:03d}"


def label_period(info: dict) -> float:
    return float(info["dt"]) / float(info["speed_scale"])


def init_replay(bundle_dir, out_dir, speed_scale: float = 0.35,
                settings: IlcSettings = IlcSettings(), pass0_runs: int = 2) -> Path:
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
    dt = float(labels.metadata.get("dt", 1.0 / float(bundle.metadata.get("control_freq", 20))))
    P = np.asarray(labels.positions, dtype=float)
    speed = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)] / (dt / float(speed_scale))
    edges = {f"{name}_{side}": float(speed[min(max(i, 0), n - 1)])
             for name, (a, b) in zip(("grasp", "release"), windows)
             for side, i in (("before", a - 1), ("after", b + 1))}
    info = {
        "bundle": str(bundle_dir), "n_labels": n, "dt": dt,
        "speed_scale": float(speed_scale), "grasp_index": grasp, "release_index": release,
        "frozen_windows": [[int(a), int(b)] for a, b in windows],
        "window_edge_speed_m_s": edges,
        "object_width_closing_m": closing_width(bundle),
        "stiffness": bundle.metadata.get("stiffness", "static"),
        "ilc": dataclasses.asdict(settings),
        "runs": {"0": int(pass0_runs)},
    }
    _pass_dir(out, 0).mkdir(parents=True)
    np.save(_pass_dir(out, 0) / "attractor.npy", P)
    np.save(_pass_dir(out, 0) / "attractor_rot.npy", np.asarray(labels.orientations, dtype=float))
    (out / "replay.json").write_text(json.dumps(info, indent=2))
    return out


def load_replay(out_dir) -> tuple[dict, SourceBundle]:
    out = Path(out_dir).expanduser()
    info = json.loads((out / "replay.json").read_text())
    return info, SourceBundle.load(info["bundle"])


def passes(out_dir) -> list[int]:
    return sorted(int(p.name[5:]) for p in Path(out_dir).expanduser().glob("pass_[0-9][0-9][0-9]")
                  if (p / "attractor.npy").exists())


def planned_runs(info: dict, k: int) -> int:
    return int(info.get("runs", {}).get(str(k), 1))


def recordings(out_dir, k: int) -> list[Path]:
    """Pass ``k``'s recordings, run 1 first."""
    d = _pass_dir(out_dir, k)
    return sorted(d.glob("recording_[0-9]*.npz"), key=lambda p: int(p.stem.split("_")[1]))


def pending_pass(out_dir) -> tuple[int, int]:
    """``(pass, run)`` to run next: the last pass, while it has fewer runs than planned."""
    info, _ = load_replay(out_dir)
    ks = passes(out_dir)
    if not ks:
        raise FileNotFoundError(f"{out_dir} holds no pass; run `replay init` first")
    k = ks[-1]
    done = len(recordings(out_dir, k))
    if done >= planned_runs(info, k):
        raise FileExistsError(f"pass {k} is recorded ({done} run{'s' * (done != 1)}): run "
                              f"`python -m tpgpt.hardware.replay update {out_dir}` first")
    return k, done + 1


def add_run(out_dir) -> tuple[int, int]:
    """Run the last recorded pass once more, with the same attractor: ``(pass, runs now)``."""
    out = Path(out_dir).expanduser()
    info, _ = load_replay(out)
    ks = passes(out)
    if not ks or not recordings(out, ks[-1]):
        raise FileNotFoundError("no recorded pass to repeat")
    k = ks[-1]
    runs = max(planned_runs(info, k), len(recordings(out, k))) + 1
    info.setdefault("runs", {})[str(k)] = runs
    (out / "replay.json").write_text(json.dumps(info, indent=2))
    return k, runs


def load_attractor(out_dir, k: int) -> np.ndarray:
    return np.load(_pass_dir(out_dir, k) / "attractor.npy")


def load_attractor_rotations(out_dir, k: int, bundle: SourceBundle | None = None) -> np.ndarray:
    """Pass ``k``'s commanded orientations (grasp convention); the taught ones if none saved."""
    path = _pass_dir(out_dir, k) / "attractor_rot.npy"
    if path.exists():
        return np.load(path)
    bundle = bundle or load_replay(out_dir)[1]
    return np.asarray(bundle.tool_labels.orientations, dtype=float)


def save_recording(out_dir, k: int, run: int, **arrays) -> Path:
    """What the arm did in run ``run`` of pass ``k``: at least ``label_index``, ``positions``
    (grasp point) and ``rotations`` (grasp convention)."""
    path = _pass_dir(out_dir, k) / f"recording_{int(run)}.npz"
    np.savez_compressed(path, **{name: np.asarray(v) for name, v in arrays.items()})
    return path


def pass_margin(out_dir, k: int, rotational=None) -> tuple[np.ndarray, np.ndarray, str]:
    """The gap pass ``k`` is checked with: ``(clearance m, tilt rad, where it came from)``.

    Pass 0: :func:`first_pass_margin` from ``rotational`` and, when ``use-checks`` has
    put them in ``replay.json``, the stiffness checks' measurements (saved, so the pass's
    own record says what it was checked with). Later passes: what ``update`` measured.
    """
    path = _pass_dir(out_dir, k) / "margin.npz"
    if k == 0 and rotational is not None:
        info, bundle = load_replay(out_dir)
        model = info.get("first_pass_model")
        clearance, tilt = first_pass_margin(bundle.tool_labels.orientations, rotational,
                                            label_period(info), model)
        source = "stiffness checks" if model else "stiffness checks of 2026-09-30"
        np.savez(path, clearance=clearance, tilt=tilt, source=source,
                 rotational=np.asarray(rotational, dtype=float))
        return clearance, tilt, source
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: pass 0 needs the rotational stiffness, later "
                                "passes the margin `update` writes")
    m = np.load(path)
    return m["clearance"], m["tilt"], str(m["source"])


def measured_runs(out_dir, k: int, info: dict | None = None):
    """Each run of pass ``k`` on the label grid: ``[(positions (N,3), rotations (N,3,3))]``."""
    info = info or load_replay(out_dir)[0]
    n = info["n_labels"]
    out = []
    for path in recordings(out_dir, k):
        rec = np.load(path)
        out.append((label_positions(rec["label_index"], rec["positions"], n),
                    label_rotations(rec["label_index"], rec["rotations"], n)))
    if not out:
        raise FileNotFoundError(f"pass {k} has no recording")
    return out


def _mean(runs):
    """The runs' average pose per label, and their spread ``(m, rad)`` (0 for one run)."""
    from scipy.spatial.transform import Rotation

    P = np.stack([p for p, _ in runs])
    R = np.stack([r for _, r in runs])
    mean_p = P.mean(axis=0)
    mean_r = np.stack([Rotation.from_matrix(R[:, i]).mean().as_matrix()
                       for i in range(R.shape[1])])
    spread_p, spread_r = np.zeros(len(mean_p)), np.zeros(len(mean_p))
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            spread_p = np.maximum(spread_p, np.linalg.norm(P[a] - P[b], axis=1))
            spread_r = np.maximum(spread_r, np.linalg.norm(rotation_errors(R[a], R[b]), axis=1))
    return mean_p, mean_r, spread_p, spread_r


def latest_spread(out_dir, upto: int, info: dict) -> tuple[np.ndarray, np.ndarray, int | None]:
    """The run-to-run spread of the latest pass up to ``upto`` that was run more than once."""
    for k in range(upto, -1, -1):
        if len(recordings(out_dir, k)) >= 2:
            _, _, sp, sr = _mean(measured_runs(out_dir, k, info))
            return sp, sr, k
    n = info["n_labels"]
    return np.zeros(n), np.zeros(n), None


def pass_error(out_dir, k: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """How far pass ``k``'s arm was from the taught path: ``(report, mean positions,
    mean rotations)``.

    The report has the error overall and at the two marks (position and tilt), the runs'
    spread, the worst sideways stray and tilt, and whether they stayed inside the margin
    the pass was checked with (``margin_exceeded``: labels where a run strayed further,
    and by how much at worst).
    """
    info, bundle = load_replay(out_dir)
    labels = bundle.tool_labels
    runs = measured_runs(out_dir, k, info)
    P, R = np.asarray(labels.positions, dtype=float), np.asarray(labels.orientations, dtype=float)
    mean_p, mean_r, spread_p, spread_r = _mean(runs)
    frozen = frozen_mask(info["n_labels"], info["frozen_windows"])
    free = ~frozen
    e = np.linalg.norm(P - mean_p, axis=1)
    tilt = np.degrees(np.linalg.norm(rotation_errors(R, mean_r), axis=1))
    side = np.zeros(len(P))
    turn = np.zeros(len(P))
    for p, r in runs:
        s, t = strays(P, R, p, r)
        side, turn = np.maximum(side, s), np.maximum(turn, t)
    report = {
        "runs": len(runs),
        "error_rms_mm": float(np.sqrt(np.mean(e[free] ** 2)) * 1000),
        "error_max_mm": float(e[free].max() * 1000),
        "tilt_rms_deg": float(np.sqrt(np.mean(tilt[free] ** 2))),
        "tilt_max_deg": float(tilt[free].max()),
        "at_marks": residuals_at(P, mean_p, R, {"grasp": info["grasp_index"],
                                                "release": info["release_index"]}, mean_r),
        "stray_max_mm": float(side.max() * 1000),
        "stray_tilt_max_deg": float(np.degrees(turn.max())),
        "spread_max_mm": float(spread_p.max() * 1000),
        "spread_tilt_max_deg": float(np.degrees(spread_r.max())),
    }
    margin = _pass_dir(out_dir, k) / "margin.npz"
    if margin.exists():
        m = np.load(margin)
        over = np.maximum(side - m["clearance"], 0.0)
        over_t = np.maximum(turn - m["tilt"], 0.0)
        bad = np.flatnonzero((over > MARGIN_FLOOR) | (over_t > TILT_FLOOR))
        report["margin_exceeded"] = {
            "labels": int(len(bad)),
            "first": int(bad[0]) if len(bad) else None,
            "worst_mm": float(over.max() * 1000), "worst_tilt_deg": float(np.degrees(over_t.max())),
            "checked_with": str(m["source"]),
        }
    return report, mean_p, mean_r


def update(out_dir, gain: float | None = None) -> dict:
    """From the last recorded pass, write the next pass's attractor and margin, and this
    pass's report."""
    out = Path(out_dir).expanduser()
    info, bundle = load_replay(out)
    recorded = [k for k in passes(out) if recordings(out, k)]
    if not recorded:
        raise FileNotFoundError(f"no recorded pass in {out}; run the replay first")
    k = recorded[-1]
    if (_pass_dir(out, k + 1) / "attractor.npy").exists():
        raise FileExistsError(f"pass {k + 1} already has its attractor")
    done, planned = len(recordings(out, k)), planned_runs(info, k)
    if done < planned:
        raise FileNotFoundError(f"pass {k} is to be run {planned} times and has {done}: run it "
                                "again first")
    settings = IlcSettings(**info["ilc"])
    if gain is not None:
        settings = dataclasses.replace(settings, gain=float(gain))
    labels = bundle.tool_labels
    error, mean_p, mean_r = pass_error(out, k)
    period = label_period(info)
    frozen = frozen_mask(info["n_labels"], info["frozen_windows"])
    new_p, rep_p = ilc_update(load_attractor(out, k), mean_p, labels.positions, period, frozen,
                              settings)
    new_r, rep_r = ilc_rotation_update(load_attractor_rotations(out, k, bundle), mean_r,
                                       labels.orientations, period, frozen, settings)
    # The next pass's margin: this pass's worst stray (over its runs) plus the spread.
    side, turn = np.zeros(info["n_labels"]), np.zeros(info["n_labels"])
    for p, r in measured_runs(out, k, info):
        s, t = strays(labels.positions, labels.orientations, p, r)
        side, turn = np.maximum(side, s), np.maximum(turn, t)
    spread_p, spread_r, spread_from = latest_spread(out, k, info)
    clearance = np.maximum(_running_max(side + spread_p, MARGIN_WINDOW), MARGIN_FLOOR)
    tilt = np.maximum(_running_max(turn + spread_r, MARGIN_WINDOW), TILT_FLOOR)
    nxt = _pass_dir(out, k + 1)
    nxt.mkdir(parents=True, exist_ok=True)
    np.save(nxt / "attractor.npy", new_p)
    np.save(nxt / "attractor_rot.npy", new_r)
    np.savez(nxt / "margin.npz", clearance=clearance, tilt=tilt, source=f"pass {k}")
    report = {"pass": k, **error, "update": rep_p, "rotation_update": rep_r,
              "gain": settings.gain, "spread_from_pass": spread_from,
              "next_margin_max_mm": float(clearance.max() * 1000),
              "next_margin_tilt_max_deg": float(np.degrees(tilt.max()))}
    if k > 0:
        before = _pass_dir(out, k - 1) / "report.json"
        if before.exists():
            prev = json.loads(before.read_text())["error_rms_mm"]
            if error["error_rms_mm"] > prev:
                report["warning"] = (f"pass {k} ({error['error_rms_mm']:.1f} mm RMS) is worse than "
                                     f"pass {k - 1} ({prev:.1f} mm): learning is not converging "
                                     f"at this gain. Delete pass_{k + 1:03d} and update again "
                                     "with --gain 0.5.")
    (_pass_dir(out, k) / "report.json").write_text(json.dumps(report, indent=2))
    return report


def use_checks(out_dir, check_dirs) -> dict:
    """Put what the stiffness checks measured into ``replay.json`` for pass 0's margin."""
    out = Path(out_dir).expanduser()
    info, _ = load_replay(out)
    info["first_pass_model"] = model_from_checks(check_dirs)
    (out / "replay.json").write_text(json.dumps(info, indent=2))
    return info["first_pass_model"]


def check_pass(out_dir, k: int, shelf_config, rotational=None, with_object: bool = True,
               start_q=None) -> tuple[str | None, dict]:
    """The taught path, with pass ``k``'s margin, against the shelf model and the table:
    ``(verdict, check)``.

    ``verdict`` is ``None`` when admissible (``path_check.verdict``), else why not. What
    the arm traces is the taught path plus its stray from it, not the attractor, so the
    taught path is checked with the margin (:func:`pass_margin`). ``rotational`` (per
    label, N m/rad) is needed for pass 0's estimate. ``with_object``: whether the object is
    carried (not when the gripper is not used). No camera cloud: the scene is the measured
    shelf model and the table (anything else in the way is the operator's).
    """
    import yaml

    from tpgpt.hardware import path_check
    from tpgpt.hardware.end_effector import end_effector_for_plan
    from tpgpt.hardware.fr3 import HOME_Q
    from tpgpt.hardware.shelf import ShelfModel, ShelfWorld

    info, bundle = load_replay(out_dir)
    labels = bundle.tool_labels
    clearance, tilt, _ = pass_margin(out_dir, k, rotational)
    cfg = yaml.safe_load(open(Path(shelf_config).expanduser()))
    world = ShelfWorld(model=ShelfModel.from_config(cfg), cloud=np.zeros((0, 3)),
                       table_height=float(cfg["shelf"]["anchors"]["table"]))
    hand = end_effector_for_plan(bundle.metadata)
    check = path_check.path_feasibility(
        labels.positions, hand.ee_from_grasp(labels.orientations), labels.gripper, hand, world,
        bundle.placement.points if with_object else np.zeros((0, 3)),
        info["grasp_index"], info["release_index"],
        closed_width=float(info["object_width_closing_m"]),
        start_q=HOME_Q if start_q is None else start_q,
        destination_height=float(bundle.placement.destination_height),
        clearance=clearance, tilt=tilt)
    return path_check.verdict(check), check


def status(out_dir) -> list[dict]:
    out = Path(out_dir).expanduser()
    info, _ = load_replay(out)
    rows = []
    for k in passes(out):
        row = {"pass": k, "runs": len(recordings(out, k)), "planned": planned_runs(info, k)}
        report = _pass_dir(out, k) / "report.json"
        if report.exists():
            row.update(json.loads(report.read_text()))
        elif row["runs"]:
            row.update(pass_error(out, k)[0])
        rows.append(row)
    return rows


def _print_status(out_dir):
    for row in status(out_dir):
        line = f"pass {row['pass']:3d}: {row['runs']} of {row['planned']} run(s) recorded"
        if "error_rms_mm" in row:
            m = row["at_marks"]
            line += (f"\n    off the taught path: {row['error_rms_mm']:.1f} mm RMS, "
                     f"{row['error_max_mm']:.1f} worst; tilt {row['tilt_rms_deg']:.2f} deg RMS, "
                     f"{row['tilt_max_deg']:.2f} worst"
                     f"\n    at the grasp {m['grasp']['total_mm']:.1f} mm (across the jaws "
                     f"{m['grasp']['closing_mm']:.1f}), tilt {m['grasp'].get('tilt_deg', 0):.2f} deg;"
                     f" at the release {m['release']['total_mm']:.1f} mm (across the jaws "
                     f"{m['release']['closing_mm']:.1f}), tilt {m['release'].get('tilt_deg', 0):.2f}"
                     f"\n    strayed sideways up to {row['stray_max_mm']:.1f} mm and "
                     f"{row['stray_tilt_max_deg']:.2f} deg")
            if row["runs"] > 1:
                line += (f"; runs differed by up to {row['spread_max_mm']:.1f} mm and "
                         f"{row['spread_tilt_max_deg']:.2f} deg")
            ex = row.get("margin_exceeded")
            if ex:
                line += ("\n    inside the margin it was checked with" if not ex["labels"] else
                         f"\n    MARGIN EXCEEDED at {ex['labels']} labels (first {ex['first']}), by "
                         f"up to {ex['worst_mm']:.1f} mm / {ex['worst_tilt_deg']:.2f} deg: the "
                         f"{ex['checked_with']} margin was too small there")
            if "gain" in row:
                line += (f"\n    next pass: attractor moved up to {row['update']['update_max_mm']:.1f}"
                         f" mm and {row['rotation_update']['update_max_deg']:.2f} deg (gain "
                         f"{row['gain']}); its margin up to {row['next_margin_max_mm']:.1f} mm and "
                         f"{row['next_margin_tilt_max_deg']:.2f} deg")
            if "warning" in row:
                line += f"\n    WARNING: {row['warning']}"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay passes of a taught demonstration (ILC)")
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init", help="start a replay; pass 0 commands the taught path")
    a.add_argument("--bundle", required=True, help="the demonstration's source_bundle directory")
    a.add_argument("--out", required=True, help="the replay directory to create")
    a.add_argument("--speed-scale", type=float, default=0.35,
                   help="the executor's speed_scale; the replay learns the lag at this speed")
    a.add_argument("--pass0-runs", type=int, default=2,
                   help="how many times pass 0 is run (2: their difference is the spread)")
    u = sub.add_parser("update", help="compute the next pass's attractor from the last pass")
    u.add_argument("dir")
    u.add_argument("--gain", type=float, default=None, help="ILC gain for this update (default 1.0)")
    r = sub.add_parser("repeat", help="run the last pass once more with the same attractor")
    r.add_argument("dir")
    c = sub.add_parser("use-checks", help="pass 0's margin from stiffness checks' measurements")
    c.add_argument("dir")
    c.add_argument("checks", nargs="+", help="stiffness check directories "
                   "(~/tpgpt_stiffness_checks/<time>_<mode>_<k>_rot<k_rot>)")
    s = sub.add_parser("status", help="the passes and their errors")
    s.add_argument("dir")
    args = ap.parse_args(argv)
    if args.command == "init":
        out = init_replay(args.bundle, args.out, args.speed_scale, pass0_runs=args.pass0_runs)
        info, _ = load_replay(out)
        print(f"replay at {out}: {info['n_labels']} labels, grasp {info['grasp_index']}, release "
              f"{info['release_index']}, contact windows {info['frozen_windows']}, speed_scale "
              f"{info['speed_scale']} ({info['n_labels'] * label_period(info):.0f} s per run). "
              f"Pass 0 is ready, to be run {info['runs']['0']} times.")
        fast = {k: v for k, v in info["window_edge_speed_m_s"].items() if v > EDGE_SPEED_WARNING}
        if fast:
            print("WARNING: the taught hand is moving at a contact window's edge ("
                  + ", ".join(f"{k} {v * 1000:.0f} mm/s" for k, v in fast.items())
                  + "): the bridge will carry that motion's lag correction across the window.")
    elif args.command == "update":
        report = update(args.dir, args.gain)
        print(f"pass {report['pass']} -> pass {report['pass'] + 1}: attractor written.")
        _print_status(args.dir)
    elif args.command == "repeat":
        k, runs = add_run(args.dir)
        print(f"pass {k} will be run {runs} times; run the replay again.")
    elif args.command == "use-checks":
        model = use_checks(args.dir, args.checks)
        print(f"pass 0's margin now comes from {len(model['checks'])} stiffness check(s): "
              f"sideways {model['stray_m'] * 1000:.1f} mm; per rotational stiffness:")
        for row in model["rotation"]:
            print(f"  {row['k']:5.1f} N m/rad: resting tilt {row['tilt_deg']:.2f} deg, turning "
                  f"time constant {row['tau_s']:.2f} s")
    else:
        _print_status(args.dir)


if __name__ == "__main__":
    main()
