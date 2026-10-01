"""Compare the shipped integrator vs attractor_law='reference' per cell."""
import sys, numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

cells = [("panda","banana","P0","top_left"), ("xarm","hammer","P0","top_middle"),
         ("robotiq85","sugar","P0","top_middle"), ("robotiq140","mug","P0","top_middle"),
         ("panda","meat","P0","top_left")]
labels, placement, _ = record_source_placement(0, capture=True)

def run_cell(g, o, p, d, law):
    env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0,
                      world="ycb", pick_config=p)
    slot = np.asarray(env.slot_poses()[d], float)
    rk = {"release_gate": True, "close_gate": True}
    if law: rk["attractor_law"] = law
    r = run(prompt_for(o, d.replace("_"," ")), gripper=g, shelf_variant="cubby",
            seed=0, source=(labels, placement), max_steps=600, env=env,
            executor="policy", rollout_kwargs=rk)
    probe = r.rollout.metadata.get("probe", {})
    tr = np.stack([np.asarray(probe.get(f"object_{a}", []), float) for a in "xyz"], 1)
    total = np.linalg.norm((tr[-1]-slot)[:2])*1000 if len(tr) else float('nan')
    env.close()
    return r.outcome, total

for g,o,p,d in cells:
    o0, t0 = run_cell(g,o,p,d, None)
    o1, t1 = run_cell(g,o,p,d, "reference")
    print(f"{g}/{o:7s} {d:11s} | default: {o0:26s} {t0:6.1f}mm | "
          f"reference: {o1:26s} {t1:6.1f}mm", flush=True)
