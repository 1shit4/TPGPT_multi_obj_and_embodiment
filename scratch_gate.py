"""Reshelving regression gate: default vs anchor(0.2). Must stay ~20/20."""
import numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source, transport_to_scene

labels, keypoints, ok = record_source(source_seed=0, embodiment="panda", control_freq=20)
print(f"source demo ok={ok}", flush=True)
seeds = list(range(1, 21))

def gate(rk, tag):
    good, errs = 0, []
    for s in seeds:
        r = transport_to_scene(labels, keypoints, s, embodiment="panda",
                               control_freq=20, rollout_kwargs=rk)
        succ = bool(r.rollout.success)
        err = r.diagnostics.get("placement_error_mm") if hasattr(r, "diagnostics") else None
        good += succ
        errs.append((s, succ, err))
    print(f"[{tag}] {good}/{len(seeds)} placed", flush=True)
    fails = [s for s, su, _ in errs if not su]
    if fails: print(f"    failed seeds: {fails}", flush=True)
    return good

gate(None, "default")
gate({"attractor_law": "anchor", "anchor_gain": 0.2}, "anchor 0.2")
