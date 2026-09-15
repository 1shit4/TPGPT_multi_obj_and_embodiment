"""Re-run every cell of the real-world campaign with a full per-step trace.

The campaign records aggregates. This records the story: when the jaws were
commanded shut, when the fingers first touched anything, whether the touch was
*opposed*, how hard they gripped, whether the object moved before they closed,
and where the hand actually was when it closed compared with where the grasp
said to be. Those are the quantities that separate "the hand never arrived" from
"it arrived and the object was not between the fingers" from "it had hold and
lost it", and no aggregate can.
"""
import json, sys, numpy as np
from pathlib import Path
from tpgpt.experiments.pipeline import build_scene, run
from tpgpt.experiments.run_experiments import prompt_for, MAX_STEPS
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.diagnose import object_probe

OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
labels, placement, ok = record_source_placement(0, capture=True)
assert ok
SOURCE = (labels, placement)

cells = json.load(open(sys.argv[1]))
done = []
for i, c in enumerate(cells):
    g, o, pick, slot, expect = c
    tag = f"{g}__{o}__{pick}__{slot}"
    if (OUT / f"{tag}.npz").exists():
        continue
    try:
        env = build_scene((o,), gripper=g, seed=0, world="real", pick_config=pick)
        r = run(prompt_for(o, slot.replace("_", " ")), gripper=g, seed=0, objects=(o,),
                source=SOURCE, max_steps=MAX_STEPS, env=env,
                rollout_kwargs={"probe": object_probe(env, o, g)})
        ro = r.rollout
        t = ro.metadata.get("probe", {})
        np.savez_compressed(
            OUT / f"{tag}.npz",
            hand=np.asarray(ro.positions, np.float32),
            attractor=np.asarray(ro.attractors, np.float32),
            grip=np.asarray(ro.gripper, np.float32),
            phase=np.asarray(ro.time_belief, np.float32),
            **{k: np.asarray(v, np.float32) for k, v in t.items()
               if k in ("object_x","object_y","object_z","held","grip_force","closure","jaw")},
            grasp=np.asarray(r.metrics.get("grasp_pose_target"), np.float32),
            release=np.asarray(r.metrics.get("grasp_pose_release"), np.float32),
            meta=np.asarray([r.metrics.get("grasp_chosen_index", -1), expect if expect is not None else -1], np.float32),
        )
        print(f"[{i+1}/{len(cells)}] {tag}  {r.outcome}  grasp#{r.metrics.get('grasp_chosen_index')}"
              f" (was #{expect})", flush=True)
        env.close()
    except Exception as e:
        print(f"[{i+1}/{len(cells)}] {tag}  EXCEPTION {type(e).__name__}: {e}", flush=True)
print("BATCH DONE")
