"""Bucket failure causes across a representative slice of the grid."""
import numpy as np, re
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

hands = ["panda", "robotiq85", "robotiq140", "rethink"]
objs = ["meat", "banana", "sugar", "mug", "hammer"]
pick, dest = "P0", "top_middle"
labels, placement, _ = record_source_placement(0, capture=True)

def stage(d, name):
    for s in d.stages:
        if s.name == name: return s
    return None

for g in hands:
    for o in objs:
        env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0, world="ycb", pick_config=pick)
        r = run(prompt_for(o, dest.replace("_"," ")), gripper=g, shelf_variant="cubby", seed=0,
                source=(labels, placement), max_steps=600, env=env, executor="policy",
                rollout_kwargs={"release_gate": True, "close_gate": True})
        d = r.diagnosis
        first = next((s.name for s in d.stages if not s.ok), "placed")
        reach = stage(d, "reach"); grasp = stage(d, "grasp"); lift = stage(d, "lift")
        # parse approach component from reach detail
        appr = re.search(r"([\d.]+) mm along the approach", reach.detail) if reach else None
        gf = re.search(r"peak grip force was (\d+) N", grasp.detail) if grasp else None
        lifted = lift.value*1000 if lift else float('nan')
        rr = reach.value*1000 if reach else float('nan')
        print(f"{g+'/'+o:20s} {('OK' if d.succeeded else first):9s} "
              f"reach={rr:6.1f}mm appr={appr.group(1) if appr else '?':>5}mm "
              f"gripF={gf.group(1) if gf else '-':>3}N lift={lifted:6.1f}mm", flush=True)
        env.close()
