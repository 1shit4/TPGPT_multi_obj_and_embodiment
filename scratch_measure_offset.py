"""For a cell, measure the horizontal offset between the chosen grip point and
the object's centroid -- the quantity place_pose_for ignores."""
import sys, numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

gripper, obj, pick, dest = sys.argv[1:5]
labels, placement, _ = record_source_placement(0, capture=True)
env = build_scene((obj,), gripper=gripper, shelf_variant="cubby", seed=0,
                  world="ycb", pick_config=pick)
# object centroid + footprint from the sim
obj_pos = np.asarray(env.object_position(obj), dtype=float)
result = run(prompt_for(obj, dest.replace("_", " ")), gripper=gripper,
             shelf_variant="cubby", seed=0, source=(labels, placement),
             max_steps=1, env=env)   # max_steps=1: we only want selection/metrics
m = result.metrics
gt = np.asarray(m["grasp_pose_target"], dtype=float)      # 4x4 grip pose at pick
grip_xy = gt[:3, 3][:2]
off = np.linalg.norm(grip_xy - obj_pos[:2]) * 1000
print(f"[measure] {gripper}/{obj}: object_centroid_xy={obj_pos[:2]}, "
      f"grip_xy={grip_xy}, horizontal grip->centroid offset = {off:.1f} mm", flush=True)
print(f"[measure] release_error_xy from run = "
      f"{1000*(result.metrics.get('release_error_xy') or float('nan')):.1f} mm", flush=True)
env.close()
