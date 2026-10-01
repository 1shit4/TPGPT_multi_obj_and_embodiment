import sys, numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

gripper, obj, pick, dest = sys.argv[1:5]
labels, placement, _ = record_source_placement(0, capture=True)
env = build_scene((obj,), gripper=gripper, shelf_variant="cubby", seed=0,
                  world="ycb", pick_config=pick)
result = run(prompt_for(obj, dest.replace("_", " ")), gripper=gripper,
             shelf_variant="cubby", seed=0, source=(labels, placement),
             max_steps=600, env=env, executor="policy",
             rollout_kwargs={"release_gate": True, "close_gate": True})
ro = result.rollout; meas = result.diagnosis.measurements
tr = ro.metadata.get("trace", {})
ph = np.asarray(ro.phases, float) if getattr(ro, "phases", None) is not None else None
n = len(ro.positions); ri = min(int(meas.get("release_step") or 3*n//4), n-1)
keys = [k for k in ("gate","behind","expected","max_lag","clamped","anchor_gain") if k in tr]
print(f"\n=== {gripper}/{obj} {dest}: {result.outcome} release={ri}/{n} ===")
print("phase = policy time-belief (0..1); gate/behind/clamped from ExecutionLaw")
hdr = "step  phase  " + "  ".join(f"{k:>9}" for k in keys)
print(hdr)
for k in range(max(0, ri-27), min(n, ri+3), 3):
    row = f"{k:>4} {ph[k]:6.3f}  " if ph is not None else f"{k:>4}   --   "
    row += "  ".join(f"{np.asarray(tr[key])[k]:>9.4f}" if k < len(tr[key]) else f"{'':>9}" for key in keys)
    print(row + (" <-- release" if abs(k-ri)<3 else ""))
env.close()
