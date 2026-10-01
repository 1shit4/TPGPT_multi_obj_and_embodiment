"""Full decomposition of placement error:
  plan      : planned object body @release vs slot   (method quality)
  hand_miss : actual hand @release vs planned grip @release (executor tracking)
  in_grip   : object moved relative to the hand, grasp->release (grip stability)
"""
import sys, numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

gripper, obj, pick, dest = sys.argv[1:5]
labels, placement, _ = record_source_placement(0, capture=True)
env = build_scene((obj,), gripper=gripper, shelf_variant="cubby", seed=0,
                  world="ycb", pick_config=pick)
slot = np.asarray(env.slot_poses()[dest], float)
result = run(prompt_for(obj, dest.replace("_", " ")), gripper=gripper,
             shelf_variant="cubby", seed=0, source=(labels, placement),
             max_steps=600, env=env, executor="policy",
             rollout_kwargs={"release_gate": True, "close_gate": True})
ro = result.rollout; m = result.metrics; meas = result.diagnosis.measurements
probe = ro.metadata.get("probe", {})
obj_tr = np.stack([np.asarray(probe.get(f"object_{a}", []), float) for a in "xyz"], 1)
hand = np.asarray(ro.positions, float)
n = min(len(obj_tr), len(hand)); obj_tr, hand = obj_tr[:n], hand[:n]
gi = min(int(meas.get("close_step") or n//4), n-1)
ri = min(int(meas.get("release_step") or 3*n//4), n-1)

gt = np.asarray(m["grasp_pose_target"], float); gr = np.asarray(m["grasp_pose_release"], float)
grip_pick_p, R_pick = gt[:3,3], gt[:3,:3]; grip_rel_p, R_rel = gr[:3,3], gr[:3,:3]
b_local = R_pick.T @ (obj_tr[0] - grip_pick_p)
planned_body_release = grip_rel_p + R_rel @ b_local

def d2(a,b): return float(np.linalg.norm((np.asarray(a)-np.asarray(b))[:2])*1000)
plan   = d2(planned_body_release, slot)
handm  = d2(hand[ri], grip_rel_p)
ingrip = abs(np.linalg.norm(obj_tr[ri]-hand[ri]) - np.linalg.norm(obj_tr[gi]-hand[gi]))*1000
total  = d2(obj_tr[ri], slot)
print(f"{gripper}/{obj:7s} {dest:11s} {result.outcome:26s} | "
      f"plan={plan:5.1f}  hand_miss={handm:6.1f}  in_grip={ingrip:5.1f}  total={total:6.1f} mm")
env.close()
