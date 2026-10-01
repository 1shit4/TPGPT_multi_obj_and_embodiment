"""Test the gated anchor law at several gains vs the shipped default."""
import numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

cells = [("panda","banana","P0","top_left"), ("xarm","hammer","P0","top_middle"),
         ("robotiq85","sugar","P0","top_middle"), ("robotiq140","mug","P0","top_middle")]
labels, placement, _ = record_source_placement(0, capture=True)

def run_cell(g,o,p,d,rk):
    env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0, world="ycb", pick_config=p)
    slot = np.asarray(env.slot_poses()[d], float)
    base = {"release_gate": True, "close_gate": True}; base.update(rk)
    r = run(prompt_for(o,d.replace("_"," ")), gripper=g, shelf_variant="cubby", seed=0,
            source=(labels,placement), max_steps=600, env=env, executor="policy", rollout_kwargs=base)
    probe = r.rollout.metadata.get("probe", {})
    tr = np.stack([np.asarray(probe.get(f"object_{a}",[]),float) for a in "xyz"],1)
    tot = np.linalg.norm((tr[-1]-slot)[:2])*1000 if len(tr) else float('nan')
    env.close(); return r.outcome, tot

configs = [("default", {}),
           ("anchor g=0.2", {"attractor_law":"anchor","anchor_gain":0.2}),
           ("anchor g=0.5", {"attractor_law":"anchor","anchor_gain":0.5})]
for g,o,p,d in cells:
    line = f"{g}/{o:7s} {d:11s} |"
    for tag,rk in configs:
        oc,t = run_cell(g,o,p,d,rk)
        ok = "OK " if oc=="success" else "xx "
        line += f"  {tag}: {ok}{t:6.1f}mm |"
    print(line, flush=True)
