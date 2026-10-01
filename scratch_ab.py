"""A/B the gated anchor(0.2) vs default over a representative YCB subset."""
import numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

hands = ["panda", "robotiq85", "robotiq140"]
objs = ["meat", "banana", "sugar", "mug", "hammer"]
pick, dest = "P0", "top_left"
labels, placement, _ = record_source_placement(0, capture=True)

def run_cell(g, o, rk):
    env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0, world="ycb", pick_config=pick)
    slot = np.asarray(env.slot_poses()[dest], float)
    base = {"release_gate": True, "close_gate": True}; base.update(rk)
    r = run(prompt_for(o, dest.replace("_"," ")), gripper=g, shelf_variant="cubby", seed=0,
            source=(labels, placement), max_steps=600, env=env, executor="policy", rollout_kwargs=base)
    probe = r.rollout.metadata.get("probe", {})
    tr = np.stack([np.asarray(probe.get(f"object_{a}",[]),float) for a in "xyz"],1)
    tot = np.linalg.norm((tr[-1]-slot)[:2])*1000 if len(tr) else float('nan')
    env.close(); return r.outcome == "success", tot

dA = dB = 0
print(f"{'cell':22s} {'default':>16s} {'anchor0.2':>16s}")
for g in hands:
    for o in objs:
        aok, at = run_cell(g, o, {})
        bok, bt = run_cell(g, o, {"attractor_law":"anchor","anchor_gain":0.2})
        dA += aok; dB += bok
        flag = "" if aok==bok else ("  <== FIXED" if bok else "  <== BROKE")
        print(f"{g+'/'+o:22s} {('OK ' if aok else 'xx ')+f'{at:6.1f}':>16s} "
              f"{('OK ' if bok else 'xx ')+f'{bt:6.1f}':>16s}{flag}", flush=True)
print(f"\nTOTAL  default {dA}/15 (top_left)   anchor0.2 {dB}/15 (top_left)")
