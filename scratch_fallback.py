"""Are the grasp/grip failures funnel fallbacks (poor least-bad grasps)?"""
import numpy as np
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

cells = [("panda","meat"),("rethink","meat"),("rethink","mug"),("robotiq140","sugar"),
         ("panda","hammer"),("robotiq85","meat"),("robotiq85","banana")]
pick, dest = "P0", "top_middle"
labels, placement, _ = record_source_placement(0, capture=True)
for g,o in cells:
    env = build_scene((o,), gripper=g, shelf_variant="cubby", seed=0, world="ycb", pick_config=pick)
    obj = np.asarray(env.object_position(o), float)
    r = run(prompt_for(o, dest.replace("_"," ")), gripper=g, shelf_variant="cubby", seed=0,
            source=(labels,placement), max_steps=1, env=env)
    m = r.metrics
    fell = m.get("path_fell_back"); surv = m.get("survivors"); idx = m.get("grasp_chosen_index")
    rankex = m.get("path_rank_examined")
    gt = np.asarray(m["grasp_pose_target"], float)
    grip_xy_off = np.linalg.norm((gt[:3,3]-obj)[:2])*1000
    appr = gt[:3,2]  # 3rd col = approach (z of grasp frame); dot with world -z = top-down-ness
    topdown = float(-appr[2])  # 1.0 = straight down
    print(f"{g}/{o:7s} fell_back={str(fell):5s} survivors={surv} idx={idx} rank_ex={rankex} "
          f"grip_horiz_off={grip_xy_off:5.1f}mm topdown={topdown:+.2f}", flush=True)
    env.close()
