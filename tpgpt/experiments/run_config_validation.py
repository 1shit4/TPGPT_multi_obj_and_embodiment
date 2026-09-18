"""Authored against curated descriptions, on a fixed candidate set.

The first pass at this (``execution_driver.py``) called the planner fresh each
time. GraspGen-X's planner is an unseeded diffusion model, so each call draws a
different candidate set, and one cell moved from 0/8 to 4/9 between two
otherwise identical runs -- see ``docs/gripper_configs.md`` section 12 and
``ROBOTICS_NOTES.md`` 7.15. A comparison built on that is comparing draws.

Here every candidate set goes through ``tpgpt.grasp.cache``, so a cell's grasps
are fixed the first time it runs and identical on every rerun. The two arms of a
pair still see *different* candidates -- they must, because changing the config
is what changes what the model proposes, and that is the thing under test -- but
neither arm is a fresh coin flip, so a rerun measures the physics rather than
the planner, and the number quoted is reproducible.

n is 30 per cell rather than 12, because section 12's differences were not
resolvable at 12.
"""
import json
import pathlib
import sys

import numpy as np

from tpgpt.grasp import cache
from tpgpt.grasp import validate_config as V

CFG_ROOT = ("/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/assets/"
            "gripper_descriptions/gripper_descriptions/assets/x_grippers")

PAIRS = [
    ("panda",      "franka_panda",   "panda",      "curated"),
    ("panda",      "panda_v3",       "panda",      "authored"),
    ("robotiq140", "robotiq_2f_140", "robotiq140", "curated"),
    ("robotiq140", "robotiq140_v3",  "robotiq140", "authored"),
    ("robotiq85",  "robotiq_2f_85",  "robotiq85",  "curated"),
    ("robotiq85",  "robotiq85_v3",   "robotiq85",  "authored"),
    ("jaco3f",     "jaco3f_v3",      "jaco3f",     "authored"),
    ("inspire",    "inspire_hand",   "inspire",    "curated"),
    ("inspire",    "inspire_v3",     "inspire",    "authored"),
]
CUBE_HALF = 0.020
N_GRASPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def candidates(cloud, name):
    """Cached ``(poses, scores)`` for this cloud and this description."""
    key = cache.cache_key(cloud, name, num_grasps=200, topk=100)
    hit = cache.load(key)
    if hit is not None:
        return hit[0], hit[1], True
    poses, scores = V.propose(cloud, name, num_grasps=200, topk=100)
    cache.store(key, np.asarray(poses), np.asarray(scores))
    return np.asarray(poses), np.asarray(scores), False


rows = []
for hand, name, short, who in PAIRS:
    try:
        tcp = float(json.load(open(f"{CFG_ROOT}/{name}/config.json"))["fingertip"][-1])
        env = V._lift_scene(short, cube_half=CUBE_HALF, seed=0)
        V.reset(env, seed=0)
        centre = np.array(V._cube_state(env)[0])
        cloud = V.surface_cloud(np.full(3, CUBE_HALF), centre, seed=0)
        poses, scores, cached = candidates(cloud, name)
        order = np.argsort(-scores.reshape(-1))
        keep = [poses[i] for i in order if poses[i][:3, 2][2] < -0.5][:N_GRASPS]

        held = touched = reach = 0
        lifts = []
        for pose in keep:
            try:
                r = V.teleport_grasp(env, pose, short, tcp, seed=0)
            except Exception:                            # noqa: BLE001
                continue
            if not r.get("reachable"):
                continue
            reach += 1
            touched += bool(r.get("touching"))
            held += bool(r.get("held"))
            lifts.append(float(r.get("lift_mm", 0.0)))
        env.close()
        row = {"hand": hand, "config": name, "by": who, "cached": cached,
               "tcp_mm": round(tcp * 1000, 1), "top_down": len(keep),
               "reachable": reach, "touched": touched, "held": held,
               "held_of_attempted": f"{held}/{len(keep)}",
               "mean_lift_mm": round(float(np.mean(lifts)), 1) if lifts else 0.0}
    except Exception as exc:                             # noqa: BLE001
        row = {"hand": hand, "config": name, "by": who,
               "error": f"{type(exc).__name__}: {exc}"[:90]}
    rows.append(row)
    print(json.dumps(row), flush=True)

print()
print(f"{'hand':12s} {'by':9s} {'tcp mm':>7s} {'top':>4s} {'reach':>6s} "
      f"{'touch':>6s} {'HELD':>5s} {'of top':>7s} {'lift mm':>8s}")
for r in rows:
    if "error" in r:
        print(f"{r['hand']:12s} {r['by']:9s} ERROR {r['error']}")
        continue
    print(f"{r['hand']:12s} {r['by']:9s} {r['tcp_mm']:7.1f} {r['top_down']:4d} "
          f"{r['reachable']:6d} {r['touched']:6d} {r['held']:5d} "
          f"{r['held_of_attempted']:>7s} {r['mean_lift_mm']:8.1f}")

out = pathlib.Path("outputs/config_authoring")
out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "paired_rows.json", "w"), indent=1)
