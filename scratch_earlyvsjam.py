"""Near the release, is the hand still approaching the planned release pose
(early release) or stalled against something (jam)?"""
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
ro = result.rollout; m = result.metrics; meas = result.diagnosis.measurements
hand = np.asarray(ro.positions, float)
att = np.asarray(ro.attractors, float) if getattr(ro, "attractors", None) is not None else None
gr = np.asarray(m["grasp_pose_release"], float); target = gr[:3, 3]
n = len(hand); ri = min(int(meas.get("release_step") or 3*n//4), n-1)
print(f"\n=== {gripper}/{obj} {dest}: {result.outcome}  (release step {ri}/{n}) ===")
print(f"{'step':>5} {'dist_to_release_mm':>18} {'hand_speed_mm':>13} {'attr_to_release_mm':>18}")
for k in range(max(0, ri-24), min(n, ri+3), 3):
    dist = np.linalg.norm(hand[k]-target)*1000
    spd = np.linalg.norm(hand[k]-hand[k-1])*1000 if k > 0 else 0.0
    ad = np.linalg.norm(att[k]-target)*1000 if att is not None and k < len(att) else float('nan')
    mark = " <-- release" if abs(k-ri) < 3 else ""
    print(f"{k:>5} {dist:>18.1f} {spd:>13.2f} {ad:>18.1f}{mark}")
env.close()
