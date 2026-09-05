"""Run the reshelving transportation experiment from a config file.

    MUJOCO_GL=egl python -m tpgpt.experiments.run_reshelving \
        --config configs/reshelving.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tpgpt.experiments.reshelving_pipeline import record_source, transport_to_scene
from tpgpt.transport.svgp import SparseGaussianProcessRegressor


def _residual(name: str):
    if name in ("none", None):
        return None
    if name == "svgp":
        return SparseGaussianProcessRegressor()
    return "default"


def run(config_path: str | Path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    experiment, scene = config["experiment"], config["scene"]
    execution = config.get("execution", {})
    out_dir = Path(experiment["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, keypoints, demo_ok = record_source(
        source_seed=experiment["source_seed"],
        embodiment=scene["embodiment"],
        control_freq=scene["control_freq"],
        keypoint_noise=scene.get("keypoint_noise", 0.0),
    )
    if not demo_ok:
        raise RuntimeError("the source demonstration failed; nothing to transport")
    labels.save(out_dir / "source_labels.npz")
    keypoints.save(out_dir / "source_keypoints.json")
    print(f"source demonstration: {len(labels)} labels, {len(keypoints)} keypoints")

    residual = _residual(config.get("transport", {}).get("residual", "gp"))
    records = []
    for seed in experiment["target_seeds"]:
        result = transport_to_scene(
            labels,
            keypoints,
            target_seed=seed,
            embodiment=scene["embodiment"],
            control_freq=scene["control_freq"],
            keypoint_noise=scene.get("keypoint_noise", 0.0),
            max_steps=execution.get("max_steps", 600),
            residual=residual,
        )
        records.append({"seed": seed, "success": result.success, **result.diagnostics})
        result.transported_labels.save(out_dir / f"transported_labels_seed{seed}.npz")
        result.target_keypoints.save(out_dir / f"target_keypoints_seed{seed}.json")
        print(f"  seed {seed:3d}: {result.summary()}")

    successes = [r["success"] for r in records]
    errors = np.array([r["placement_error_xy"] for r in records])
    summary = {
        "n_episodes": len(records),
        "success_rate": float(np.mean(successes)),
        "placement_error_xy_median": float(np.median(errors)),
        "placement_error_xy_median_on_success": (
            float(np.median(errors[np.array(successes)])) if any(successes) else None
        ),
        "keypoint_residual_max": max(r["keypoint_residual_max"] for r in records),
        "jacobian_fraction_positive_min": min(
            r["jacobian_fraction_positive"] for r in records
        ),
        "keypoint_displacement_max": max(r["keypoint_displacement_max"] for r in records),
        "affine_residual_max": max(r["affine_residual_max"] for r in records),
    }
    report = {"config": config, "summary": summary, "episodes": records}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print(
        f"\nsuccess {summary['success_rate'] * 100:.0f}% "
        f"({sum(successes)}/{len(records)}); median placement error on success "
        f"{(summary['placement_error_xy_median_on_success'] or 0) * 1000:.1f} mm"
    )
    print(f"report written to {out_dir / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reshelving.yaml")
    run(parser.parse_args().config)
