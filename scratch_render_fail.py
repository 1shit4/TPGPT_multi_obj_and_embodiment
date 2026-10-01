"""Render a single YCB cell end-to-end to video, to see the failure physically.

Usage:
  MUJOCO_GL=egl GRASPGENX_GRIPPER_CFG_DIR=... TPGPT_GRASP_CACHE=... \
    python scratch_render_fail.py <gripper> <object> <pick> <destination> <out.mp4>
"""
import sys
import numpy as np

from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for

gripper, obj, pick, destination, out = sys.argv[1:6]

print(f"[render] {gripper}/{obj} pick={pick} dest={destination} -> {out}", flush=True)
labels, placement, ok = record_source_placement(0, capture=True)
print(f"[render] source demo recorded ok={ok}", flush=True)

env = build_scene(
    (obj,), gripper=gripper, shelf_variant="cubby", seed=0,
    world="ycb", pick_config=pick,
)
result = run(
    prompt_for(obj, destination.replace("_", " ")),
    gripper=gripper, shelf_variant="cubby", seed=0,
    source=(labels, placement),
    max_steps=600, env=env, executor="policy",
    rollout_kwargs={
        "video_path": out, "camera": "workspace",
        "release_gate": True, "close_gate": True,
    },
)
d = result.diagnosis
print(f"[render] outcome={result.outcome} detail={result.detail}", flush=True)
if d is not None:
    print(f"[render] stages={d.stages}", flush=True)
    m = d.measurements or {}
    keys = ("lift_height", "reach_error", "drift_max", "carry_fraction",
            "contact_steps", "release_error_xy", "final_error_xy")
    print("[render] " + " ".join(
        f"{k}={m[k]:.3f}" for k in keys if k in m and m[k] is not None), flush=True)
print(f"[render] video={out}", flush=True)
env.close()
