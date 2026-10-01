"""Export complete forensic campaigns; never pool development attempts.

PYTHONPATH=. python scripts/paper_results.py --out outputs/paper_results \
    --run replay=outputs/paper_evaluation/replay --run policy=outputs/paper_evaluation/policy
"""
import argparse
import csv
import html
import json
import math
import os
from collections import Counter
from pathlib import Path

from tpgpt.experiments.audit_artifacts import atomic_json, outcome_summary, sha256


SCENE_FIELDS = ("gripper", "obj", "world", "seed", "pick_config", "pick_pose",
                "object_rest_quat", "slot", "shelf_variant", "shelf_translation")


def identity(row):
    return json.dumps({k: row["cell"].get(k) for k in SCENE_FIELDS}, sort_keys=True)


def wilson(successes, scheduled):
    if not scheduled:
        return None
    z = 1.959963984540054
    p = successes / scheduled
    d = 1 + z*z / scheduled
    c = (p + z*z / (2*scheduled)) / d
    e = z * math.sqrt(p*(1-p)/scheduled + z*z/(4*scheduled*scheduled)) / d
    return [c-e, c+e]


def read_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if not config["cells"]:
        raise ValueError("a results campaign must schedule at least one cell")
    rows = []
    for index, cell in enumerate(config["cells"]):
        path = directory / f"cell_{index:03d}" / "result.json"
        if not path.exists():
            raise ValueError(f"incomplete campaign: missing {path}")
        row = json.loads(path.read_text())
        if row["cell"] != cell:
            raise ValueError(f"checkpoint cell differs from frozen configuration: {path}")
        rows.append(row)
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError(f"duplicate physical cells in {directory}")
    summary = outcome_summary(rows, len(config["cells"]))
    saved = json.loads((directory / "summary.json").read_text())
    if saved != summary:
        raise ValueError(f"saved summary disagrees with checkpoints: {directory}")
    summary["outcomes"] = dict(Counter(r["outcome"] for r in rows))
    summary["wilson_95_descriptive"] = wilson(summary["successes"], summary["scheduled"])
    summary["path_fallback_executions"] = sum(
        r.get("executed", False) and r.get("metrics", {}).get("path_fell_back", False) for r in rows)
    summary["funnel_fallback_executions"] = sum(
        r.get("executed", False) and bool(r.get("metrics", {}).get("funnel_fallbacks")) for r in rows)
    summary["forced_diagnostic_cells"] = sum(bool(r.get("forced_diagnostic")) for r in rows)
    return directory, config, rows, summary


def compare(left, right):
    """Keep unmatched/reproduction-mismatched cells visible."""
    a = {identity(r): r for r in left[2]}
    b = {identity(r): r for r in right[2]}
    pairs, transitions = [], Counter()
    same_source = left[1]["source_sha256"] == right[1]["source_sha256"]
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            pairs.append({"cell": json.loads(key), "matched_scene": False})
            continue
        m, n = x.get("metrics", {}), y.get("metrics", {})
        cache = m.get("candidate_cache", {}).get("candidate_sha256")
        labels = m.get("transported_labels_sha256")
        record = dict(cell=json.loads(key), matched_scene=True, same_source=same_source,
                      same_candidates=cache is not None and cache == n.get("candidate_cache", {}).get("candidate_sha256"),
                      same_labels=labels is not None and labels == n.get("transported_labels_sha256"),
                      same_effective_pose=m.get("effective_grasp_pose") is not None and m.get("effective_grasp_pose") == n.get("effective_grasp_pose"),
                      before=x["outcome"], after=y["outcome"])
        if record["same_source"] and record["same_candidates"]:
            transitions[f"{x['outcome']} -> {y['outcome']}"] += 1
        pairs.append(record)
    return {"transitions_same_source_and_candidates": dict(transitions), "pairs": pairs}


def export(runs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = {name: read_run(path) for name, path in runs.items()}
    atomic_json(out / "summaries.json", {name: run[3] for name, run in loaded.items()})
    atomic_json(out / "inputs.json", {name: {
        "directory": str(run[0].resolve()), "config_sha256": sha256(run[0] / "config.json"),
        "source_sha256": run[1]["source_sha256"], "settings": run[1]["settings"],
        "executor": run[1]["executor"],
    } for name, run in loaded.items()})
    names = list(loaded)
    if len(names) > 1:
        atomic_json(out / "paired.json", {f"{names[0]} vs {name}": compare(loaded[names[0]], loaded[name])
                                          for name in names[1:]})
    data, links = [], []
    for name, (directory, config, rows, summary) in loaded.items():
        for index, row in enumerate(rows):
            cell, metrics = row["cell"], row.get("metrics", {})
            settled = metrics.get("post_settle", {})
            error = settled.get("placement_error_xy_m", metrics.get("placement_error_xy"))
            data.append(dict(run=name, index=index, gripper=cell["gripper"], object=cell["obj"],
                             condition=cell.get("evaluation_condition", cell.get("development_condition", cell.get("pick_config", "explicit"))),
                             outcome=row["outcome"], executed=row.get("executed", False),
                             refused=row.get("refused", False),
                             pre_settle_outcome=metrics.get("pre_settle_outcome", "not measured"),
                             settled=settled.get("passed", "not measured"),
                             error_xy_mm=None if error is None else 1000*error,
                             path_fallback=metrics.get("path_fell_back", False),
                             candidate_sha256=metrics.get("candidate_cache", {}).get("candidate_sha256"),
                             labels_sha256=metrics.get("transported_labels_sha256")))
            folder = directory / f"cell_{index:03d}"
            artifacts = []
            for filename in ("result.json", "simulation_steps.json", "settling.json", "workspace.mp4", "sideview.mp4", "final_state.png", "final_sim_state.npz"):
                path = folder / filename
                if path.exists():
                    artifacts.append(f'<a href="{html.escape(os.path.relpath(path, out))}">{filename}</a>')
            links.append(f"<tr><td>{html.escape(name)}</td><td>{index}</td><td>{html.escape(cell['gripper'])}</td>"
                         f"<td>{html.escape(cell['obj'])}</td><td>{html.escape(row['outcome'])}</td><td>{' · '.join(artifacts)}</td></tr>")
    with (out / "cells.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\hline", r"Run & Scheduled & Successes & Success (\%) & Refused & Setup errors & Diagnostic errors & Execution failures \\", r"\hline"]
    for name, (_, _, _, s) in loaded.items():
        safe = name.replace("_", r"\_")
        tex.append(f"{safe} & {s['scheduled']} & {s['successes']} & {100*s['overall_success']:.1f} & {s['refusals']} & {s['setup_errors']} & {s['diagnostic_errors']} & {s['execution_failures']} " + r"\\")
    tex += [r"\hline", r"\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(tex) + "\n")
    (out / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Simulation evidence</title>"
        "<style>body{font:15px sans-serif;margin:2rem}td,th{padding:.5rem;border-bottom:1px solid #ddd}table{border-collapse:collapse}a{color:#1458a0}</style>"
        "<h1>Simulation evidence</h1><p>Every scheduled checkpoint is included. Run names and inputs determine whether these are development or evaluation results."
        " Wilson intervals are descriptive and do not account for development selection or shared conditions.</p>"
        "<table><tr><th>Run</th><th>Cell</th><th>Gripper</th><th>Object</th><th>Outcome</th><th>Artifacts</th></tr>" + "".join(links) + "</table>")
    return loaded


def plot_results(loaded, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    first = next(iter(loaded.values()))
    grippers = list(dict.fromkeys(r["cell"]["gripper"] for r in first[2]))
    objects = list(dict.fromkeys(r["cell"]["obj"] for r in first[2]))
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.5*len(loaded), 4.4),
                             squeeze=False, layout="constrained")
    for ax, (name, (_, _, rows, summary)) in zip(axes[0], loaded.items()):
        values = np.full((len(grippers), len(objects)), np.nan)
        for i, g in enumerate(grippers):
            for j, o in enumerate(objects):
                cells = [r for r in rows if r["cell"]["gripper"] == g and r["cell"]["obj"] == o]
                if cells:
                    wins = sum(r["outcome"] == "success" for r in cells)
                    values[i, j] = wins / len(cells)
                    ax.text(j, i, f"{wins}/{len(cells)}", ha="center", va="center",
                            color="white" if values[i, j] < .65 else "black")
        im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(objects)), objects, rotation=30, ha="right")
        ax.set_yticks(range(len(grippers)), grippers)
        ax.set_title(f"{name}: {summary['successes']}/{summary['scheduled']} "
                     f"({100*summary['overall_success']:.1f}%)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Success / all scheduled cells", shrink=.8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(Path(out) / f"success_matrix.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    if len(runs) != len(args.run):
        parser.error("run names must be unique")
    loaded = export(runs, args.out)
    if args.plot:
        plot_results(loaded, args.out)


if __name__ == "__main__":
    main()
