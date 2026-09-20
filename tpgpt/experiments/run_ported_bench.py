"""Do GraspGen-X's own hands, ported into robosuite, actually hold anything?

``tpgpt.grasp.port_gripper`` converts a curated GraspGen-X gripper directory
into a robosuite gripper. That answers "does the hand mount and move". This
driver answers the separate and much harder question of whether it **grasps**,
on the same bench as ``docs/gripper_configs.md`` section 19: a Panda in a
``Lift`` env, one cube, candidates from the planner, each candidate reached by
inverse kinematics, closed on, and lifted.

**Why this is a package module and not a scratchpad script.** The
``sharpa_wave grasps: 4 of 12`` in commit ``bf489da``'s message came from a
scratchpad file with no provenance, which is ``ROBOTICS_NOTES.md`` 7.37 --
the rule that a scratch diagnostic producing a number worth quoting *is* a
campaign. ``df8e1a9`` had to relocate the config-validation driver for the same
reason earlier in the same session. That number has since been withdrawn;
section 20 of ``docs/gripper_configs.md`` records why, using this driver.

**Draws, not reruns.** GraspGen-X's planner is an unseeded diffusion model, so
"run it again" changes the candidate set and therefore changes the task
(7.15). The unit of variance here is the **candidate set**, not the grasp: a
hand that holds 4 of 12 on one draw and 0 of 10 on the next has not become
worse, it has been asked a different question. So ``--draws`` requests *k*
independent candidate sets, each keyed into ``tpgpt.grasp.cache`` by its draw
index, which makes every individual draw reproducible while still letting the
spread across draws be seen. Pooling across draws is the number to quote; a
single draw is not.

**The frame is asserted, not measured.** A ported hand needs no
``alignment_rotation`` or ``contact_offset`` pass, because the porter builds
the frame canonical: the XML's ``eef`` body sits at the description's own
``fingertip`` depth along +Z, which is the frame GraspGen-X emits poses in. So
the planned-pose to ``grip_site`` conversion is the identity rotation plus that
depth. The ``reach_mm`` column is what checks the assertion -- a frame error
puts the arm somewhere else, and shows up there rather than as a silent miss.

Usage::

    MUJOCO_GL=egl /home/ishita/mujoco_env/bin/python -m \\
        tpgpt.experiments.run_ported_bench --hands sharpa_wave --draws 3 \\
        --out outputs/ported_bench
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from tpgpt.grasp import cache
from tpgpt.grasp import validate_config as V
from tpgpt.grasp.port_gripper import CURATED, convert
from tpgpt.grasp.ported_models import PortedGripper
from tpgpt.reporting import provenance as prov
from tpgpt.sim.kinematics import solve_ik
from tpgpt.sim.replay import _joint_action

#: Candidates carried into physics per draw, after the top-down filter.
N_POSES = 20

#: How many grasps the planner is asked for, and how many it returns ranked.
NUM_GRASPS, TOPK = 200, 100

#: Approach must point downward. ``pose[:3, 2][2] < -0.5`` is roughly "within
#: 60 degrees of straight down" -- the same filter section 19's bench uses, so
#: ported hands and registered hands are asked the same question.
TOPDOWN_Z = -0.5

#: Finger count per declared family. GraspGen-X's ``type`` names the trained
#: conditioning family, **not** the number of fingers: it ships the five-finger
#: ``sharpa_wave`` as ``revolute_3f``. So this column is what the config
#: declares, and the real finger count is recorded separately where known.
FAMILY_FINGERS = {"parallel_2f": 2, "revolute_2f": 2, "revolute_3f": 3}


def _register(name: str) -> None:
    """Convert the curated directory and make robosuite able to build it."""
    from robosuite.models.grippers import GRIPPER_MAPPING

    convert(name)
    GRIPPER_MAPPING[name] = type(
        f"Ported_{name}",
        (PortedGripper,),
        {"__init__": lambda self, idn=0, n=name: PortedGripper.__init__(self, n, idn)},
    )


def _patch_cube(half: float):
    """Force the ``Lift`` cube to a fixed half-size, and prove it took.

    robosuite samples the cube's size from a range, so the only honest way to
    ask a big hand about a big object is to pin ``size_min == size_max`` before
    the env is built. Patching ``geom_size`` afterwards does **not** work:
    ``V.reset`` reloads the model and silently discards the patch, which is how
    three earlier runs came to plan for a 90 mm cube and grasp a 40 mm one.
    """
    import robosuite.environments.manipulation.lift as lift_mod

    original = lift_mod.BoxObject

    def sized(*args, **kwargs):
        kwargs["size_min"] = [half] * 3
        kwargs["size_max"] = [half] * 3
        return original(*args, **kwargs)

    lift_mod.BoxObject = sized
    return original


def _draw(env, name: str, cloud, draw: int, fresh: bool):
    """One candidate set, cached under its own draw index."""
    key = cache.cache_key(
        cloud, name, num_grasps=NUM_GRASPS, topk=TOPK, draw=draw
    )
    hit = None if fresh else cache.load(key)
    if hit is None:
        poses, scores = V.propose(cloud, name, num_grasps=NUM_GRASPS, topk=TOPK)
        poses, scores = np.asarray(poses), np.asarray(scores)
        cache.store(key, poses, scores)
    else:
        poses, scores = hit
    order = np.argsort(-np.asarray(scores).reshape(-1))
    keep = [np.asarray(poses)[i] for i in order
            if np.asarray(poses)[i][:3, 2][2] < TOPDOWN_Z][:N_POSES]
    return keep, key


def bench(name: str, half: float, draws: int, fresh: bool) -> list[dict]:
    """Grasp-close-lift a ported hand over ``draws`` candidate sets."""
    import robosuite as suite

    cfg = json.loads((CURATED / name / "config.json").read_text())
    tcp = float(cfg["fingertip"][-1])
    _register(name)

    env = suite.make(
        "Lift", robots="Panda", gripper_types=name, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, horizon=400, ignore_done=True,
    )
    rows = []
    try:
        V.reset(env, seed=0)
        gid = env.sim.model.geom_name2id("cube_g0")
        got = float(env.sim.model.geom_size[gid][0])
        assert abs(got - half) < 1e-6, f"cube is {got}, wanted {half}"
        centre = np.array(V._cube_state(env)[0])
        cloud = V.surface_cloud(np.full(3, half), centre, seed=0)

        robot, arm = env.robots[0], "right"
        qi = np.asarray(
            robot.composite_controller.part_controllers[arm].qpos_index
        )
        for draw in range(draws):
            keep, key = _draw(env, name, cloud, draw, fresh)
            reach_ok = touch = held = 0
            reaches, lifts = [], []
            for pose in keep:
                V.reset(env, seed=0)
                R = np.asarray(pose)[:3, :3]        # identity alignment, asserted
                target = np.asarray(pose)[:3, 3] + np.asarray(pose)[:3, 2] * tcp
                res = solve_ik(env, target, R, arm,
                               position_tolerance=V.IK_TOLERANCE)
                if not res.reachable:
                    continue
                reach_ok += 1
                env.sim.data.qpos[qi] = np.array(res.qpos)
                env.sim.forward()
                site = (robot.eef_site_id[arm]
                        if isinstance(robot.eef_site_id, dict)
                        else robot.eef_site_id)
                reaches.append(float(np.linalg.norm(
                    target - np.array(env.sim.data.site_xpos[site]))) * 1000)
                z0 = float(V._cube_state(env)[0][2])
                seed_q = np.array(res.qpos)
                for _ in range(40):
                    env.step(_joint_action(env, robot, arm, seed_q, 1.0))
                touch += bool(V._cube_contacts(env))
                for step in range(60):
                    t = target + np.array(
                        [0.0, 0.0, V.LIFT_TO * (step + 1) / 60])
                    seed_q = solve_ik(env, t, R, arm,
                                      position_tolerance=V.IK_TOLERANCE,
                                      seed_qpos=seed_q).qpos
                    env.step(_joint_action(env, robot, arm, seed_q, 1.0))
                dz = float(V._cube_state(env)[0][2]) - z0
                lifts.append(dz * 1000)
                held += dz > V.LIFT_THRESHOLD
            rows.append({
                "hand": name, "draw": draw, "cache_key": key,
                "family": cfg["type"],
                "declared_fingers": FAMILY_FINGERS.get(cfg["type"]),
                "tcp_mm": tcp * 1000, "cube_half_mm": half * 1000,
                "topdown": len(keep), "reachable": reach_ok,
                "reach_mm": float(np.mean(reaches)) if reaches else None,
                "touch": touch, "held": held,
                "lift_mm": float(np.mean(lifts)) if lifts else None,
            })
            print(f"  {name:16s} draw {draw}: held {held}/{reach_ok} "
                  f"(top-down {len(keep)}, touch {touch}, "
                  f"reach {np.mean(reaches) if reaches else float('nan'):.1f} mm, "
                  f"lift {np.mean(lifts) if lifts else float('nan'):.1f} mm)",
                  flush=True)
    finally:
        env.close()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--cube-half", type=float, default=0.020,
                    help="cube half-size in metres (robosuite's Lift default "
                         "is 0.020, i.e. a 40 mm cube)")
    ap.add_argument("--draws", type=int, default=3,
                    help="independent candidate sets; one draw is not a result")
    ap.add_argument("--fresh", action="store_true",
                    help="bypass the cache -- makes the run NON-reproducible, "
                         "use --draws instead")
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args(argv)

    info = prov.provenance()
    prov.warn_if_unreproducible(info, strict=args.require_clean)
    print(prov.describe(info), flush=True)

    rows = []
    for name in args.hands:
        try:
            rows += bench(name, args.cube_half, args.draws, args.fresh)
        except Exception as exc:                            # noqa: BLE001
            print(f"{name:16s} ERROR {type(exc).__name__}: {exc}"[:200],
                  flush=True)
            rows.append({"hand": name, "error": f"{type(exc).__name__}: {exc}"})

    for name in args.hands:
        mine = [r for r in rows if r.get("hand") == name and "held" in r]
        if mine:
            h = sum(r["held"] for r in mine)
            n = sum(r["reachable"] for r in mine)
            print(f"POOLED {name:16s} {h}/{n} "
                  f"({100.0 * h / n if n else 0:.0f}%) over {len(mine)} draws",
                  flush=True)

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=2))
        (out / "manifest.json").write_text(json.dumps({
            "title": "Ported GraspGen-X hands: grasp-close-lift",
            "settings": {
                "varied": {"hands": args.hands, "draws": args.draws},
                "fixed": {"cube_half_m": args.cube_half, "env": "Lift",
                          "robot": "Panda", "n_poses": N_POSES,
                          "num_grasps": NUM_GRASPS, "topk": TOPK,
                          "topdown_z": TOPDOWN_Z,
                          "lift_threshold_m": float(V.LIFT_THRESHOLD)},
            },
            "provenance": info,
        }, indent=2))
        print(f"wrote {out}/rows.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
