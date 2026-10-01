"""The keypoint ladder: five constructions, one grasp per cell, no physics.

    python -m tpgpt.experiments.run_ladder --cells outputs/campaigns/ycb/ycb/rows.json \\
        --out outputs/ladder

Answers the paper's first question -- what must the keypoints encode? -- by
fitting the five constructions of its Sec. III-D (``PAPER_VARIANTS``) to every
cell of a finished campaign, **with the grasp pinned to the candidate that
campaign executed** (``grasp_chosen_index``, an index into the cached
candidate set, so it is the same pose to the bit). The construction is then the
only thing that varies between the five rows of a cell, and every comparison
is paired.

Nothing is executed. Each cell rebuilds its scene, reads the cached grasp, fits
the map and measures it: milliseconds of arithmetic behind a few seconds of
scene construction, so the whole grid runs in minutes and the same cells can
be re-measured after any change to the map code.

What is measured per cell and construction, and what each means:

``aim_grasp_mm`` / ``aim_release_mm``
    Closest approach of the transported fingertip path to the target grasp
    point and to the release point. For a construction whose keypoints include
    the grasp point this is ~0 **by construction** (``phi`` interpolates its
    keypoints exactly) and is not evidence of anything; for the others it is
    the miss the executor inherits.
``orientation_error_grasp_deg`` / ``orientation_error_release_deg``
    Angular error of the transported hand orientation at the two ends, against
    the frames the keypoints were built from.
``keypoint_residual``
    Interpolation error at the keypoints, metres -- property (R1) directly.
``min_det`` / ``fraction_positive``
    ``det J`` along the executed path -- property (R2). A map with a negative
    determinant anywhere has folded space and is not a diffeomorphism.
``refused``
    The construction declined to produce a map for this cell (degenerate
    keypoints, or a fold). Counted apart from a task failure: it is a
    statement about the construction's domain, not about the task.

The paired statistics -- Wilcoxon signed-rank against the grasp cube, Holm
corrected over the four comparisons, medians with interquartile ranges -- are
computed here from the rows so the table in the manifest is the table in the
paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for
from tpgpt.experiments.run_keypoint_transport import PAPER_VARIANTS, KeypointVariant
from tpgpt.reporting.html import write_manifest

#: The reference row every other construction is tested against.
REFERENCE = "D_grasp_cube"

#: Columns the paired tests are run on, with the direction that is "better".
COMPARED = (
    ("aim_grasp_mm", "lower"),
    ("aim_release_mm", "lower"),
    ("orientation_error_grasp_deg", "lower"),
    ("orientation_error_release_deg", "lower"),
    ("keypoint_residual", "lower"),
    ("min_det", "higher"),
)


def _cells(rows_path: Path) -> list[dict]:
    rows = json.loads(Path(rows_path).read_text())
    cells = []
    for r in rows:
        if r.get("grasp_chosen_index") is None or r.get("skipped"):
            continue
        cells.append({
            "gripper": r["gripper"], "object": r["object"],
            "slot": (r.get("destination") or "top_middle").replace("_", " "),
            "pick_config": r.get("pick_config_id"), "seed": r.get("seed", 0),
            "world": r.get("world", "benchmark"),
            "grasp_index": int(r["grasp_chosen_index"]),
            "placed": r.get("outcome") == "success",
        })
    return cells


def measure_cell(cell: dict, source, variants=PAPER_VARIANTS) -> list[dict]:
    """Fit every construction to one cell's pinned grasp and measure each."""
    out = []
    env = build_scene(
        (cell["object"],), gripper=cell["gripper"], seed=cell["seed"],
        world=cell["world"], pick_config=cell["pick_config"],
    )
    try:
        for v in variants:
            row = {"variant": v.name, **{k: cell[k] for k in
                   ("gripper", "object", "slot", "pick_config", "seed", "grasp_index", "placed")}}
            try:
                result = run(
                    prompt_for(cell["object"], cell["slot"]),
                    gripper=cell["gripper"], seed=cell["seed"], source=source,
                    keypoint_parts=v.parts, keypoint_box=v.box,
                    keypoint_orientation=v.orientation,
                    cube_half_extent=v.cube_half_extent,
                    include_contacts=v.contacts, env=env, executor="none",
                    forced_grasp_index=cell["grasp_index"],
                )
            except Exception as exc:  # a refused or degenerate construction
                row.update(refused=True, reason=f"{type(exc).__name__}: {exc}")
                out.append(row)
                continue
            m = result.metrics
            row.update(
                refused=result.outcome != "not_executed",
                reason=result.detail if result.outcome != "not_executed" else "",
                n_keypoints=m.get("n_keypoints"),
                keypoint_residual=m.get("keypoint_residual"),
                min_det=m.get("min_det"),
                fraction_positive=m.get("fraction_positive"),
                folded=(m.get("fraction_positive") is not None
                        and m["fraction_positive"] < 1.0),
                aim_grasp_mm=m.get("aim_grasp_mm"),
                aim_release_mm=m.get("aim_release_mm"),
                orientation_error_grasp_deg=m.get("orientation_error_grasp_deg"),
                orientation_error_release_deg=m.get("orientation_error_release_deg"),
                det_J_at_grasp=m.get("det_J_at_grasp"),
                det_J_at_release=m.get("det_J_at_release"),
                executed_grasp_index=m.get("grasp_chosen_index"),
                seconds=sum(v_ for v_ in (result.timings or {}).values()),
            )
            out.append(row)
    finally:
        env.close()
    return out


def _holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down over a dict of p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[k] = min(1.0, running)
    return adjusted


def table(rows: list[dict], reference: str = REFERENCE) -> dict:
    """Per-construction medians, IQRs, counts, and paired tests vs the reference."""
    from scipy.stats import wilcoxon

    by_variant: dict[str, dict[tuple, dict]] = {}
    for r in rows:
        key = (r["gripper"], r["object"], r["pick_config"], r["slot"], r["seed"])
        by_variant.setdefault(r["variant"], {})[key] = r
    ref = by_variant.get(reference, {})
    summary = {}
    for name, cells in by_variant.items():
        fitted = [r for r in cells.values() if not r.get("refused")]
        # A construction that produced a map with det J changing sign is a
        # *fold*, and pipeline.run refuses to go on with one; keep the two
        # counts apart, as the paper's table does.
        folds = [r for r in cells.values() if r.get("refused") and any(
            m in (r.get("reason") or "")
            for m in ("diffeomorphism", "valid transportation map"))]
        entry = {
            "cells": len(cells),
            "refused": sum(1 for r in cells.values() if r.get("refused")) - len(folds),
            "folded": len(folds) + sum(1 for r in fitted if r.get("folded")),
        }
        for col, _ in COMPARED:
            vals = np.array([r[col] for r in fitted if r.get(col) is not None], float)
            if len(vals):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                entry[col] = {"median": float(med), "iqr": [float(q1), float(q3)],
                              "worst": float(vals.min() if col == "min_det" else vals.max()),
                              "n": int(len(vals))}
        if name != reference and ref:
            pvals = {}
            for col, _ in COMPARED:
                pairs = [(cells[k][col], ref[k][col]) for k in cells
                         if k in ref and not cells[k].get("refused")
                         and not ref[k].get("refused")
                         and cells[k].get(col) is not None and ref[k].get(col) is not None]
                if len(pairs) >= 6:
                    a, b = np.array(pairs).T
                    if np.allclose(a, b):
                        pvals[col] = 1.0
                    else:
                        pvals[col] = float(wilcoxon(a, b).pvalue)
            entry["p_vs_reference"] = pvals
            entry["p_vs_reference_holm"] = _holm(pvals) if pvals else {}
        summary[name] = entry
    return summary


def _print_table(summary: dict) -> None:
    cols = ("aim_grasp_mm", "orientation_error_grasp_deg", "keypoint_residual", "min_det")
    print(f"\n{'construction':26s} {'n':>4s} {'ref':>4s} {'fold':>4s}  "
          f"{'aim mm':>14s} {'orient deg':>14s} {'resid mm':>14s} {'min det':>16s}")
    for name, e in summary.items():
        def fmt(col, scale=1.0):
            v = e.get(col)
            if not v:
                return f"{'-':>14s}"
            return f"{v['median'] * scale:6.2f} [{v['iqr'][0] * scale:5.2f},{v['iqr'][1] * scale:5.2f}]"
        print(f"{name:26s} {e['cells']:4d} {e['refused']:4d} {e['folded']:4d}  "
              f"{fmt('aim_grasp_mm')} {fmt('orientation_error_grasp_deg')} "
              f"{fmt('keypoint_residual', 1000)} {fmt('min_det')}"
              + (f"  worst {e['min_det']['worst']:.3f}" if e.get('min_det') else ""))
        if e.get("p_vs_reference_holm"):
            print("    Holm p vs (D): " + "  ".join(f"{k}={p:.2g}" for k, p in e["p_vs_reference_holm"].items()))


def main(cells_path: str, out_dir: str, variant_names=None, limit=None) -> dict:
    from tpgpt.experiments.reshelving_pipeline import record_source_placement

    variants = tuple(v for v in PAPER_VARIANTS
                     if not variant_names or v.name in variant_names)
    cells = _cells(Path(cells_path))
    if limit:
        cells = cells[:limit]
    print(f"{len(cells)} cells x {len(variants)} constructions, grasp pinned per cell")
    labels, placement, ok = record_source_placement(0, capture=True)
    if not ok:
        raise RuntimeError("the source demonstration failed")
    rows = []
    for i, cell in enumerate(cells):
        tag = f"{cell['gripper']}/{cell['object']}/{cell['pick_config']}/{cell['slot']}"
        for r in measure_cell(cell, (labels, placement), variants):
            rows.append(r)
            flag = "REFUSED " + r.get("reason", "")[:50] if r.get("refused") else (
                f"aim {r['aim_grasp_mm']:6.1f} mm  orient {r['orientation_error_grasp_deg']:5.1f} deg  "
                f"resid {r['keypoint_residual'] * 1000:6.2f} mm  min det {r['min_det']:.3f}")
            print(f"  [{i + 1:3d}/{len(cells)}] {tag:40s} {r['variant']:24s} {flag}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    summary = table(rows)
    _print_table(summary)
    write_manifest(
        out,
        title="Keypoint ladder: the five constructions on pinned grasps",
        description=__doc__.split("\n\n")[1],
        settings={
            "varied": {"variant": [v.name for v in variants]},
            "fixed": {"cells_from": str(cells_path), "grasp": "pinned per cell to "
                      "grasp_chosen_index of the source campaign", "physics": "none",
                      "reference": REFERENCE, "correction": "Holm-Bonferroni over "
                      "the comparisons against the reference, per column"},
            "cells": len(cells),
        },
        results=summary,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", required=True, help="a campaign's rows.json")
    parser.add_argument("--out", default="outputs/ladder")
    parser.add_argument("--variants", default=None, help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="first N cells only")
    parser.add_argument("--retable", action="store_true",
                        help="recompute the table from <out>/rows.json without re-measuring")
    args = parser.parse_args()
    if args.retable:
        rows = json.loads((Path(args.out) / "rows.json").read_text())
        summary = table(rows)
        _print_table(summary)
        (Path(args.out) / "table.json").write_text(json.dumps(summary, indent=1))
    else:
        main(args.cells, args.out,
             args.variants.split(",") if args.variants else None, args.limit)
