"""Decompose placement error into PLAN error vs EXECUTION error.

plan error      = where the plan intends the object body origin at release, vs slot
execution error = actual final body origin, vs where the plan intended it
"""
import sys, numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

gripper, obj, pick, dest = sys.argv[1:5]
labels, placement, _ = record_source_placement(0, capture=True)
env = build_scene((obj,), gripper=gripper, shelf_variant="cubby", seed=0,
                  world="ycb", pick_config=pick)
slot = np.asarray(env.slot_poses()[dest], dtype=float)
result = run(prompt_for(obj, dest.replace("_", " ")), gripper=gripper,
             shelf_variant="cubby", seed=0, source=(labels, placement),
             max_steps=600, env=env, executor="policy",
             rollout_kwargs={"release_gate": True, "close_gate": True})
m = result.metrics
probe = result.rollout.metadata.get("probe", {})
ox, oy, oz = (np.asarray(probe.get(f"object_{a}", []), float) for a in "xyz")
traj = np.stack([ox, oy, oz], 1)
obj_pick = traj[0]; obj_final = traj[-1]

gt = np.asarray(m["grasp_pose_target"], float)     # grip pose @ pick (world)
gr = np.asarray(m["grasp_pose_release"], float)     # grip pose @ release (planned)
grip_pick_p, R_pick = gt[:3, 3], gt[:3, :3]
grip_rel_p,  R_rel  = gr[:3, 3], gr[:3, :3]

# object body origin relative to the grip, fixed from grasp onwards
b_local = R_pick.T @ (obj_pick - grip_pick_p)
planned_body_release = grip_rel_p + R_rel @ b_local

def mm(v): return np.round(np.asarray(v) * 1000, 1)
print(f"\n=== {gripper}/{obj} {pick} {dest} : outcome={result.outcome} ===", flush=True)
print(f"slot                     {mm(slot[:2])}")
print(f"planned body @release xy {mm(planned_body_release[:2])}  "
      f"-> PLAN error vs slot = {np.linalg.norm((planned_body_release-slot)[:2])*1000:.1f} mm")
print(f"actual final body     xy {mm(obj_final[:2])}  "
      f"-> EXEC error vs plan = {np.linalg.norm((obj_final-planned_body_release)[:2])*1000:.1f} mm")
print(f"TOTAL body vs slot xy = {np.linalg.norm((obj_final-slot)[:2])*1000:.1f} mm  "
      f"(recorded={m.get('placement_error_xy',float('nan'))*1000 if m.get('placement_error_xy') else float('nan'):.1f})")
env.close()
