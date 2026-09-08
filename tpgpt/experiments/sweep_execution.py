"""Comparing attractor laws without a robot and without a transportation map.

**What this answers.** Executing a transported policy means moving an attractor
that the impedance controller chases, and there are several defensible ways to
move it (``tpgpt.policy.rollout``). This sweeps them and reports which to use
**as a function of how aggressively the map warps space**, so the answer is a
lookup whatever the keypoint work eventually settles on.

**Why it does not use the simulator.** A real rollout costs 25-40 s. Comparing
five laws across three warp regimes with enough paired samples to separate a
2 mm difference from noise is hundreds of runs. The surrogate plant
(:func:`~tpgpt.policy.rollout.rollout_impedance`) runs the *same* gate, clamp,
law, phase update and watchdog against the arm's own steady-state equation at
about 2 ms a step. It cannot see contact, inverse-kinematic infeasibility or
orientation, so **it ranks hypotheses and does not confirm them** -- 7.26 rule 4.

**Why it does not use the real keypoints.** The keypoint/map design is being
reworked in parallel and is unresolved. Importing it would make every number
here inherit whatever that work later changes, which is exactly how three
findings in ``ROBOTICS_NOTES`` outlived the settings they were measured under.
So the warps here are built from a **lattice of our own choosing**, displaced by
a known analytic field, with the amplitude forced to zero at the grasp and the
release. Pinning the endpoints keeps the task meaningful -- the object and the
slot are still where the labels aim -- while the interior deforms as much as we
ask. The independent variable is the resulting ``min det(J)``, which is the
same conditioning number the keypoint study reports, so the two connect without
either depending on the other.

The demonstration is the **real recorded one** (``outputs/reshelving``), not a
synthetic stand-in, so the dwells, the speed profile and the segment structure
are the ones the system actually runs.

**What would refute the headline.** If the laws' paired differences are smaller
than the demonstration's own precision at the grasp (4.8 mm), the honest
conclusion is that this bed cannot rank them and the choice must be made in
physics. That is a real possible outcome and is reported as such rather than
hidden behind a p-value.

Usage::

    MUJOCO_GL=egl python -m tpgpt.experiments.sweep_execution --out outputs/dynamics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.metrics.curves import path_deviation
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.policy.rollout import (
    AnchorSchedule,
    compliance_norm,
    rollout_impedance,
)
from tpgpt.reporting.html import write_manifest
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap

#: The laws compared. Each is ``(label, kwargs for rollout_impedance)``.
#:
#: ``VR-sched`` uses a gain that is strong when the policy commands a hold and
#: weak in transit; the constant gains bracket it. ``R-m`` and ``VR-m`` query
#: the policy at the measured pose rather than at the attractor.
LAWS = {
    "V": {},
    "VR-a k=0.20": dict(attractor_law="anchor", anchor_gain=0.20),
    "VR-a k=0.50": dict(attractor_law="anchor", anchor_gain=0.50),
    "VR-sched": dict(attractor_law="anchor",
                     anchor_gain=AnchorSchedule(dwell=0.5, transit=0.05)),
    "VR-sched ungated": dict(attractor_law="anchor", anchor_gated=False,
                             anchor_gain=AnchorSchedule(dwell=0.5, transit=0.05)),
    "VR-m k=0.50": dict(attractor_law="anchor", anchor_gain=0.50, query_at="measured"),
    "R-a": dict(attractor_law="reference"),
    "R-m": dict(attractor_law="reference", query_at="measured"),
}

#: Conditioning bins. ``min det(J)`` is the smallest Jacobian determinant of the
#: map along the trajectory: 1.0 is no deformation, and 0 is a fold.
BINS = ((0.6, 1e9, "well conditioned"), (0.35, 0.6, "moderate"), (0.02, 0.35, "aggressive"))

#: Below this the map has folded or is about to; such a map is not a valid
#: transport and any number measured on it describes the fold, not the law.
MIN_VALID_DET = 0.02


def load_demonstration(path="outputs/reshelving/source_labels.npz") -> PolicyLabels:
    """The recorded source demonstration, with its measured-arm ground truth."""
    return PolicyLabels.load(path)


def grasp_frame(labels: PolicyLabels, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(closing, approach, jaw)`` unit axes of the hand at one label.

    The convention is the one in :mod:`tpgpt.grasp.grasps`: the gripper's local
    ``+X`` is the direction the jaws travel and ``+Z`` is the direction the hand
    advances. Decomposing an error into these three matters because their
    tolerances differ by an order of magnitude -- roughly 15 mm across the jaws
    against 120-135 mm along the approach -- so a straight-line distance mixes
    a budget that decides the task with one that does not (7.27).
    """
    R = np.asarray(labels.orientations[index], dtype=float)
    closing, approach = R[:, 0], R[:, 2]
    jaw = np.cross(approach, closing)
    return closing, approach, jaw / max(float(np.linalg.norm(jaw)), 1e-12)


def dwell_windows(labels: PolicyLabels, threshold=1e-3) -> list[tuple[int, int]]:
    """Contiguous spans where the demonstration commands the attractor to hold.

    These are deliberate: the teacher dwells while the jaws close and again
    while they open, which is how the arm is given time to arrive before the
    gripper acts. They are the moments where an execution error is least
    forgiven, so they get measured separately.
    """
    still = np.linalg.norm(labels.velocities, axis=1) < threshold
    spans, start = [], None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        elif not s and start is not None:
            spans.append((start, i)); start = None
    if start is not None:
        spans.append((start, len(still)))
    return [(a, b) for a, b in spans if b - a >= 5]


def pinned_warp(labels: PolicyLabels, scale: float, seed: int) -> TransportMap | None:
    """A map that deforms the interior while leaving the endpoints alone.

    A lattice spanning the trajectory's bounding box is displaced by a sum of
    Gaussian bumps of magnitude ``scale``. Each bump's amplitude is multiplied
    by how far the lattice point is from the grasp and the release, so the two
    poses that decide the task stay put and the aim error is not confounded
    with the deformation being studied.

    Returns ``None`` when the resulting map folds, which is not a failure of
    the law under test and must not be averaged in with one.
    """
    rng = np.random.default_rng(seed)
    P = labels.positions
    lo, hi = P.min(axis=0) - 0.05, P.max(axis=0) + 0.05
    grid = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 3) for i in range(3)]), -1)
    source = grid.reshape(-1, 3)

    spans = dwell_windows(labels)
    anchors = np.array([P[(a + b) // 2] for a, b in spans]) if spans else P[[0, -1]]
    # Distance to the nearest pinned pose, saturating at 15 cm: 0 at the grasp
    # and the release, 1 well away from both.
    d = np.linalg.norm(source[:, None, :] - anchors[None], axis=2).min(axis=1)
    envelope = np.clip(d / 0.15, 0.0, 1.0)

    displacement = np.zeros_like(source)
    for _ in range(3):
        centre = source[rng.integers(len(source))]
        width = 0.12
        w = np.exp(-np.linalg.norm(source - centre, axis=1) ** 2 / (2 * width ** 2))
        displacement += (w * envelope)[:, None] * rng.normal(size=3) * scale

    m = TransportMap().fit(source, source + displacement)
    det = float(m.check_diffeomorphism(P).min_determinant)
    return None if det <= MIN_VALID_DET else m


def measure(rollout, planned: np.ndarray, truth: np.ndarray, labels: PolicyLabels,
            spans, tau: float) -> dict:
    """Every number this study reports, from one rollout.

    ``drift_attractor_vs_planned`` -- how far the commanded attractor strayed
    from the path it was meant to retrace, pointwise and therefore never
    negative. The measure it replaces compared two separately minimised
    distances and could come out negative (7.26).

    ``arm_error_vs_truth`` -- the arm against ``planned - tau*v``, which is
    where the demonstrating arm actually was, warped. Comparing the arm to the
    *label* path instead would charge the impedance lag as an error, and 2.7
    shows that lag is correct behaviour.

    ``dwell_net_*`` -- net displacement of the attractor across each window
    where the labels command a hold, decomposed into the hand's own axes. Net,
    not path length: path length counts an attractor that wanders and returns,
    which is jiggle rather than drift, and reporting it alone overstates the
    problem by a factor of four.
    """
    out = {
        "steps": int(rollout.metadata["steps"]),
        "final_phase": float(rollout.metadata["final_phase"]),
        "terminated_on_phase": bool(rollout.metadata["terminated_on_phase"]),
        "blocked": bool(rollout.metadata["blocked"]),
        "budget_exhausted": bool(rollout.metadata["budget_exhausted"]),
        "clamped_fraction": float(rollout.metadata["clamped_fraction"]),
        "sag_rebaselines": int(rollout.metadata["sag_rebaselines"]),
        "gate_shut_fraction": float(np.mean(rollout.gate < 1.0)),
    }
    dev = path_deviation(rollout.attractors, planned)
    out["drift_attractor_vs_planned_max"] = dev["max"] * 1000
    out["drift_attractor_vs_planned_median"] = dev["median"] * 1000
    arm = path_deviation(rollout.positions, truth)
    out["arm_error_vs_truth_max"] = arm["max"] * 1000
    out["arm_error_vs_truth_median"] = arm["median"] * 1000

    # Excess lag: measured minus what the physics demands at that speed. Zero
    # is correct, not small-is-better.
    speeds = np.linalg.norm(rollout.velocities, axis=1)
    out["excess_lag_median"] = float(np.median(rollout.lag - tau * speeds)) * 1000

    for name, (a, b) in zip(("grasp", "release"), spans[:2]):
        t0, t1 = labels.time_belief[a], labels.time_belief[min(b, len(labels) - 1)]
        win = (rollout.time_belief >= t0) & (rollout.time_belief <= t1)
        if win.sum() < 2:
            continue
        seg = rollout.attractors[win]
        net = seg[-1] - seg[0]
        closing, approach, jaw = grasp_frame(labels, (a + b) // 2)
        out[f"dwell_{name}_net"] = float(np.linalg.norm(net)) * 1000
        out[f"dwell_{name}_path"] = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) * 1000
        out[f"dwell_{name}_closing"] = abs(float(net @ closing)) * 1000
        out[f"dwell_{name}_approach"] = abs(float(net @ approach)) * 1000
        out[f"dwell_{name}_jaw"] = abs(float(net @ jaw)) * 1000
    return out


def run(out_dir="outputs/dynamics", n_warps=8, substeps=1, seed0=100,
        load=None) -> dict:
    """Sweep every law over a range of warp aggressiveness, paired by warp.

    Args:
        load: A constant velocity the arm cannot overcome, in m/s. Without one
            the surrogate arm tracks so well that the lag gate engages on only
            2% of steps and the clamp on none, so the sweep measures the laws
            with the two mechanisms they most interact with essentially
            switched off. Passing a load -- 0.08 m/s downward shuts the gate on
            42% of steps -- is how the disturbed condition is produced without
            a robot.
    """
    labels = load_demonstration()
    dt = float(labels.metadata["dt"])
    tau = compliance_norm(labels.stiffness[0], labels.damping[0])
    spans = dwell_windows(labels)
    print(f"demonstration: {len(labels)} labels, dt={dt}, tau={tau*1000:.1f} ms, "
          f"dwells at {spans}")

    cases = [("identity", None, 1.0)]
    for i, scale in enumerate(np.concatenate([
        np.linspace(0.01, 0.06, n_warps // 3),
        np.linspace(0.06, 0.13, n_warps - n_warps // 3),
    ])):
        m = pinned_warp(labels, float(scale), seed0 + i)
        if m is None:
            print(f"  warp {scale:.3f} folded -- skipped (not a valid transport)")
            continue
        det = float(m.check_diffeomorphism(labels.positions).min_determinant)
        cases.append((f"warp{i}", m, det))

    rows = []
    for name, m, det in cases:
        planned_labels = labels if m is None else transport_labels(m, labels)
        policy = GPPolicy().fit(planned_labels)
        planned = planned_labels.positions
        truth = planned - tau * planned_labels.velocities
        for law_name, kwargs in LAWS.items():
            try:
                ro = rollout_impedance(
                    policy, dt=dt, substeps=substeps, load=load, **kwargs
                )
                rec = measure(ro, planned, truth, planned_labels, spans, tau)
                rec.update(case=name, min_det=det, law=law_name, failed=False)
            except Exception as exc:  # a law that cannot run is a result
                rec = dict(case=name, min_det=det, law=law_name, failed=True,
                           error=f"{type(exc).__name__}: {exc}")
            rows.append(rec)
        print(f"  {name:9} min det {det:6.3f}  {len(LAWS)} laws")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    write_manifest(
        out_dir,
        title="execution-law sweep",
        description=(
            "Five attractor laws on a surrogate impedance plant, over warps of "
            "increasing aggressiveness with the grasp and release poses pinned. "
            "No simulator and no keypoint design: the independent variable is "
            "min det(J), so the result is a lookup for whatever the keypoint "
            "work settles on."
        ),
        settings={
            "campaign": "sweep_execution",
            "n_settings": len(rows),
            "varied": {"law": list(LAWS), "case": [c[0] for c in cases]},
            "fixed": {"demonstration": "outputs/reshelving/source_labels.npz",
                      "dt": dt, "substeps": substeps,
                      "plant": "first-order impedance",
                      "load": None if load is None else list(np.asarray(load))},
        },
        thresholds={
            "demonstration_lag_mean_mm": float(labels.metadata["attractor_lag_mean"]) * 1000,
            "demonstration_lag_at_dwell_end_mm": 4.8,
            "closing_budget_mm": 15.0,
            "approach_tolerance_mm": 120.0,
            "min_valid_det": MIN_VALID_DET,
        },
        results={"rows": len(rows), "failed": sum(1 for r in rows if r["failed"])},
        runs=[],
    )
    print(f"\nwrote {len(rows)} rows -> {out_dir}")
    return {"rows": rows, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/dynamics")
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--load", type=float, default=0.0,
                        help="downward velocity the arm cannot overcome, m/s; "
                             "0.08 shuts the lag gate on ~42%% of steps")
    args = parser.parse_args()
    load = np.array([0.0, 0.0, -args.load]) if args.load else None
    run(args.out, n_warps=args.warps, substeps=args.substeps, load=load)
