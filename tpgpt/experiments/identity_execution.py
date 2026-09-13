"""Tier 1: can the policy reproduce the demonstration in its own scene?

**The one bed with no transportation map anywhere in it.** The policy is fitted
on the source demonstration's *own* labels and rolled out in the *same* scene
the demonstration was recorded in, at the same seed. There is no warp, no
keypoint construction, no grasp planner. So every millimetre of error belongs to
policy execution -- the attractor law, the query site, the lag gate, the
feed-forward -- and to nothing else.

That isolation is the point. ``run_keypoint_replay`` removes the *executor* and
keeps the map, which bounds what a keypoint construction can deliver. This
removes the *map* and keeps the executor, which bounds what execution can
deliver. Between them a failure has only two places left to be.

**The ground truth is exact, and it is not the labels.** ``record_demonstration``
stores ``metadata["measured_positions"]`` -- where the demonstrating arm actually
was, as opposed to where it was commanded. Those differ by the impedance lag,
``||K^-1 D|| v``, which is 22.6 mm on average in this demonstration and is
*correct* behaviour rather than error (section 2.7). Scoring the rollout against
the labels would charge that lag twice and penalise a perfect executor by the
full 22.6 mm; scoring against the measured trace asks the right question, which
is whether the arm went where the teacher's arm went.

**What this bed has that the surrogate plant does not**: real contact, real
inverse kinematics, the orientation task competing for the arm's effort, and a
real object that can be gripped, dropped or missed. The surrogate plant of
section 7.40 ranks the execution laws; this confirms or overturns that ranking
in physics.

Usage::

    MUJOCO_GL=egl python -m tpgpt.experiments.identity_execution --out outputs/identity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.metrics.curves import path_deviation
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.policy.rollout import AnchorSchedule, compliance_norm
from tpgpt.reporting.html import write_manifest
from tpgpt.sim.backend import make_reshelving_env
from tpgpt.sim.rollout import rollout_policy

#: The laws compared, keyed as in ``tpgpt.experiments.sweep_execution`` so the
#: two studies' tables line up. Trimmed to the ones the surrogate plant left
#: standing plus the incumbent: running the reference-only laws here would spend
#: simulator time confirming a result already settled at 34 stalls in 44.
LAWS = {
    "V": {},
    "VR-a k=0.20": dict(attractor_law="anchor", anchor_gain=0.20),
    "VR-sched": dict(attractor_law="anchor",
                     anchor_gain=AnchorSchedule(dwell=0.5, transit=0.05)),
}


def run_one(seed: int, law_kwargs: dict, max_steps: int = 600) -> dict:
    """Fit on the demonstration's own labels and execute in its own scene."""
    from tpgpt.experiments.reshelving_pipeline import record_source_placement

    labels, _placement, demo_ok = record_source_placement(seed)
    if not demo_ok:
        return {"failed": True, "error": "the demonstration itself failed"}

    truth = np.asarray(labels.metadata["measured_positions"], dtype=float)
    tau = compliance_norm(labels.stiffness[0], labels.damping[0])
    policy = GPPolicy().fit(labels)

    env = make_reshelving_env(seed=seed, control_freq=int(labels.metadata["control_freq"]))
    try:
        env.reset()
        rollout = rollout_policy(env, policy, max_steps=max_steps, **law_kwargs)
    finally:
        env.close()

    # The arm against where the demonstrating arm actually was. Pointwise, so it
    # cannot come out negative, and to the polyline rather than to its vertices.
    arm = path_deviation(rollout.positions, truth)
    # And against the labels, recorded only to show the gap the lag accounts
    # for -- never as the score. See the module docstring.
    versus_labels = path_deviation(rollout.positions, labels.positions)
    speeds = np.linalg.norm(rollout.velocities, axis=1)
    return {
        "failed": False,
        "seed": seed,
        "success": bool(rollout.success),
        "placement_error_xy": float(rollout.metadata["placement_error_xy"]) * 1000,
        "arm_vs_truth_max": arm["max"] * 1000,
        "arm_vs_truth_median": arm["median"] * 1000,
        "arm_vs_labels_median": versus_labels["median"] * 1000,
        "excess_lag_median": float(
            np.median(np.linalg.norm(rollout.positions - rollout.attractors, axis=1)
                      - tau * speeds)
        ) * 1000,
        "steps": int(rollout.metadata["steps"]),
        "final_phase": float(rollout.metadata["final_phase"]),
        "blocked": bool(rollout.metadata["blocked"]),
        "terminated_on_phase": bool(rollout.metadata["terminated_on_phase"]),
        "orientation_relaxed_steps": int(rollout.metadata["orientation_relaxed_steps"]),
        "skips": int(rollout.metadata["skips"]),
        "sag_rebaselines": int(rollout.metadata["sag_rebaselines"]),
    }


def run(out_dir="outputs/identity", seeds=(0, 1, 2, 3, 4), max_steps: int = 600) -> list:
    """Every law on every seed, paired -- the same scene for each law."""
    rows = []
    for law, kwargs in LAWS.items():
        for seed in seeds:
            print(f"  {law:14} seed {seed} ...", end="", flush=True)
            try:
                rec = run_one(seed, kwargs, max_steps)
            except Exception as exc:  # a law that cannot run is a result
                rec = {"failed": True, "error": f"{type(exc).__name__}: {exc}",
                       "seed": seed}
            rec["law"] = law
            rows.append(rec)
            if rec["failed"]:
                print(f" FAILED {rec.get('error', '')}")
            else:
                print(f" {'placed' if rec['success'] else 'missed'}"
                      f"  arm {rec['arm_vs_truth_max']:5.1f} mm"
                      f"  phase {rec['final_phase']:.2f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    ok = [r for r in rows if not r["failed"]]
    write_manifest(
        out_dir,
        title="identity-transport execution",
        description=(
            "The policy fitted on the source demonstration's own labels and "
            "executed in the same scene at the same seed -- no transportation "
            "map, no keypoints, no grasp planner. Ground truth is the recorded "
            "measured arm trace, so every millimetre belongs to policy "
            "execution. Confirms or overturns the surrogate-plant ranking of "
            "section 7.40 in physics."
        ),
        settings={"campaign": "identity_execution", "n_settings": len(rows),
                  "varied": {"law": list(LAWS), "seed": list(seeds)},
                  "fixed": {"transport": "identity (none)",
                            "ground_truth": "metadata['measured_positions']",
                            "max_steps": max_steps}},
        results={"rows": len(rows), "failed": len(rows) - len(ok),
                 "placed": sum(1 for r in ok if r["success"])},
        runs=[],
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/identity")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()
    run(args.out, tuple(int(s) for s in args.seeds.split(",")), args.max_steps)
