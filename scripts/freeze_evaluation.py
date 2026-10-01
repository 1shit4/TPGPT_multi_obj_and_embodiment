"""Freeze this study's prospective 120-cell local-workspace evaluation.

Does not execute simulations. Refuses to overwrite an existing evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from tpgpt.experiments.audit_artifacts import atomic_json, load_source, sha256
from tpgpt.experiments.forensics import VARIANTS


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def freeze(out):
    out = Path(out)
    development = Path("outputs/paper_development")
    # Freeze only after both predeclared development comparisons finish.
    for run in ("nominal_task_replay_55", "nominal_task_joint_55"):
        summary = json.loads((development / run / "summary.json").read_text())
        if summary["completed"] != summary["scheduled"] or summary["scheduled"] != 55:
            raise ValueError(f"development comparison is incomplete: {run}")
    source = development / "source_compact"
    load_source(source)  # Verify all source-file hashes and recorded success.
    inventories = [development / "asset_inventory.json", development / "grasp_assets_inventory.json"]
    for inventory in inventories:
        record = json.loads(inventory.read_text())
        files = record.get("files", record)
        for name, digest in files.items():
            if file_digest(name) != digest:
                raise ValueError(f"asset changed since development: {name}")
    templates = json.loads((development / "upright_objects_15.json").read_text())
    rest = {c["obj"]: c["object_rest_quat"] for c in templates}
    count, random_seed = 6, 20260930
    rng = np.random.default_rng(random_seed)
    low = np.array([-.23, -.14, 50., -.03, -.04, -.03])
    high = np.array([-.21, -.10, 65., 0., .04, 0.])
    # Stratify each dimension; never inspect outcomes when drawing conditions.
    samples = np.column_stack([(rng.permutation(count) + rng.random(count))/count
                               for _ in low]) * (high-low) + low
    conditions = [dict(id=f"E{i+1:02d}", pick_pose=v[:3].tolist(),
                       shelf_translation=v[3:].tolist()) for i, v in enumerate(samples)]
    cells = []
    for gripper in ("panda", "robotiq85", "robotiq140", "xarm", "rethink"):
        for obj in ("pudding", "gelatin", "brick", "meat"):
            for condition in conditions:
                cell = dict(gripper=gripper, obj=obj, world="ycb", seed=0,
                            pick_pose=condition["pick_pose"],
                            shelf_translation=condition["shelf_translation"],
                            shelf_variant="cubby", slot="top_middle",
                            evaluation_condition=condition["id"],
                            ik_seed_strategy="nominal", post_settle_seconds=2.,
                            max_steps=2400, rollout_kwargs={})
                if obj in rest:
                    cell["object_rest_quat"] = rest[obj]
                cells.append(cell)
    assert len(cells) == 120
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, out / "source")
    (out / "candidates").mkdir()
    atomic_json(out / "cells_replay.json", cells)
    for cell in cells:
        cell["rollout_kwargs"] = {"query_at": "attractor", "tracking_tolerance": .03}
    atomic_json(out / "cells_joint_policy.json", cells)
    for path in inventories + [development / "environment.txt"]:
        shutil.copy2(path, out / path.name)
    for root in ("tpgpt", "scripts"):
        for path in sorted(Path(root).rglob("*.py")):
            dest = out / "code" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    atomic_json(out / "protocol.json", {
        "status": "frozen_before_evaluation", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scheduled_per_executor": 120, "primary_executor": "joint_policy", "baseline_executor": "replay",
        "sampling": {"method": "six independently stratified dimensions", "seed": random_seed,
                     "low": low, "high": high, "conditions": conditions},
        "scope": "Development-selected upright packaging and foam brick in a local tabletop-to-upper-shelf workspace; five grippers mounted on one Panda arm.",
        "source": str((out / "source").resolve()), "source_manifest_sha256": sha256(out / "source/source.json"),
        "source_selection": "Two scripted one-box Panda demonstrations were compared in development; only the compact source is used in every evaluation cell.",
        "filter_variant": "task", "filter_settings": VARIANTS["task"],
        "fallback_policy": "Legacy fallbacks are retained and counted explicitly; this is not collision certification.",
        "geometry": "Full YCB target scan at simulator-derived pose; observed obstacle cloud and legacy simulator place-zone filter.",
        "executor_limits": "GP queries its reference state and phase, with measured tracking-error phase gating. Fixed joint gains do not execute transported compliance.",
        "success": "Successful completed execution under the unchanged 60 mm XY / 30 mm base-height slot criterion, then 2 s of open-jaw holding. During the final 1 s require slot occupancy, destination-board contact, no robot contact, and <=2 mm position span. Post-settling cannot rescue an execution failure.",
        "denominator": "All 120 scheduled cells per executor. Report refusals, setup errors, diagnostic errors and execution failures; no dropped or retried failures.",
        "planner": "One new draw per cell in a dedicated cache, reused by the baseline. Environment seed 0 is not a claim of seeding the external planner RNG.",
        "cells_sha256": {name: sha256(out / name) for name in ("cells_replay.json", "cells_joint_policy.json")},
        "code_sha256": {str(p.relative_to(out / "code")): sha256(p) for p in sorted((out / "code").rglob("*.py"))},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "development_evidence": [str(development / name) for name in ("nominal_task_replay_55", "nominal_task_joint_55")],
    })
    (out / "code.patch").write_text(subprocess.check_output(["git", "diff"], text=True))
    print(f"Frozen 120 fresh cells per executor in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    freeze(parser.parse_args().out)
