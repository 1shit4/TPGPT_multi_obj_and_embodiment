"""Test grasp-descent fixes: longer close-gate cap, and anchor at the pick."""
import numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

cells = [("panda","meat"), ("rethink","meat"), ("rethink","mug"),
         ("robotiq140","sugar"), ("robotiq85","meat")]
pick, dest = "P0", "top_middle"
labels, placement, _ = record_source_placement(0, capture=True)

def run_cell(g, o, rk):
    env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0, world="ycb", pick_config=pick)
    base = {"release_gate": True, "close_gate": True}; base.update(rk)
    r = run(prompt_for(o, dest.replace("_"," ")), gripper=g, shelf_variant="cubby", seed=0,
            source=(labels, placement), max_steps=700, env=env, executor="policy", rollout_kwargs=base)
    lift = next((s.value*1000 for s in r.diagnosis.stages if s.name=="lift"), float('nan'))
    env.close(); return ("OK" if r.diagnosis.succeeded else "xx"), lift

configs = [("default",{}),
           ("close300",{"close_gate_max_steps":300}),
           ("close300+anch",{"close_gate_max_steps":300,"attractor_law":"anchor","anchor_gain":0.2})]
for g,o in cells:
    line=f"{g}/{o:7s} |"
    for tag,rk in configs:
        oc,lift=run_cell(g,o,rk)
        line+=f" {tag}: {oc} lift={lift:5.0f}mm |"
    print(line, flush=True)
