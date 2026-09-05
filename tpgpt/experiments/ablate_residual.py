"""Ablate the nonlinear stage of the transportation map (paper Sec. III-D).

    MUJOCO_GL=egl python -m tpgpt.experiments.ablate_residual

Answers the question the paper's construction implies but a pure-translation
test scene cannot: does the residual ``psi`` earn its place, or would the affine
``gamma`` alone do? Everything downstream of the map -- the labels, the policy,
the controller, the scenes -- is held fixed; only the residual regressor changes.

Reported in ROBOTICS_NOTES.md section 4.6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.experiments.reshelving_pipeline import record_source, transport_to_scene
from tpgpt.metrics import rank_methods
from tpgpt.transport.svgp import SparseGaussianProcessRegressor

#: Map variants to compare. ``None`` removes the nonlinear stage entirely.
VARIANTS = {
    "full (gamma + psi)": "default",
    "sparse (SV-GPT, M=12)": lambda: SparseGaussianProcessRegressor(n_inducing=12),
    "affine only (gamma)": None,
}


def run(
    source_seed: int = 0,
    target_seeds: tuple[int, ...] = tuple(range(1, 21)),
    out_dir: str | Path = "outputs/ablation",
    variants: dict | None = None,
) -> dict:
    variants = variants or VARIANTS
    labels, keypoints, demo_ok = record_source(source_seed)
    if not demo_ok:
        raise RuntimeError("the source demonstration failed; nothing to transport")

    results: dict[str, dict] = {}
    errors: dict[str, np.ndarray] = {}
    for name, residual in variants.items():
        successes, placement, keypoint_residual = [], [], []
        for seed in target_seeds:
            factory = residual() if callable(residual) else residual
            result = transport_to_scene(labels, keypoints, seed, residual=factory)
            successes.append(result.success)
            placement.append(result.diagnostics["placement_error_xy"])
            keypoint_residual.append(result.diagnostics["keypoint_residual_max"])

        successes = np.array(successes)
        placement = np.array(placement)
        errors[name] = placement
        results[name] = {
            "n_episodes": len(target_seeds),
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "placement_error_median": float(np.median(placement)),
            "placement_error_median_on_success": (
                float(np.median(placement[successes])) if successes.any() else None
            ),
            "keypoint_residual_max": float(np.max(keypoint_residual)),
        }
        r = results[name]
        print(
            f"{name:24s} success {r['successes']:2d}/{r['n_episodes']}  "
            f"median placement {r['placement_error_median'] * 1000:7.1f} mm  "
            f"max keypoint residual {r['keypoint_residual_max'] * 1000:8.3f} mm"
        )

    ranking = rank_methods(errors, lower_is_better=True)
    print("\nMann-Whitney U on placement error (lower is better):")
    for winner, loser, p in ranking.wins:
        print(f"  {winner} beats {loser} at p = {p:.3g}")

    report = {
        "source_seed": source_seed,
        "target_seeds": list(target_seeds),
        "variants": results,
        "ranking": {"points": ranking.points, "ranks": ranking.ranks},
        "wins": [{"winner": w, "loser": l, "p": p} for w, l, p in ranking.wins],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nreport written to {out_dir / 'ablation.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--out", default="outputs/ablation")
    args = parser.parse_args()
    run(args.source_seed, tuple(range(1, args.n_targets + 1)), args.out)
