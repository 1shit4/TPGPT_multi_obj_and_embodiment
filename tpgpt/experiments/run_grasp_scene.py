"""Prompt to grasp candidates, across grippers, on one scene.

    MUJOCO_GL=egl python -m tpgpt.experiments.run_grasp_scene \
        --prompt "put the milk carton on the top shelf" \
        --grippers panda,robotiq85,robotiq140

Runs the first two stages of the new-scene flow:

    text -> object + destination        (deterministic, tpgpt.language)
         -> object point cloud          (simulator-native, tpgpt.perception)
         -> ranked grasps per gripper   (GraspGen-X, tpgpt.grasp)
         -> end-effector targets

Filtering, selecting a single grasp, and keypoint extraction for transport are
deliberately **not** here; that stage is still under discussion. Every candidate
the generator returned is reported, ranked by its score and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.grasp.client import GraspGenClient
from tpgpt.grasp.grasps import approach_waypoint, grasp_to_eef_pose
from tpgpt.grasp.grippers import DEFAULT_PAIRS, gripper_geometry, resolve_pair
from tpgpt.grasp.pipeline import grasps_for_object
from tpgpt.grasp.server import server_available, server_status
from tpgpt.language.parser import parse_task
from tpgpt.perception.scene_graph import build_scene_graph

CAMERAS = ("agentview", "frontview", "birdview")


def build_env(objects, seed, robot="Panda", gripper=None, resolution=256):
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    config = load_composite_controller_config(controller="BASIC", robot=robot)
    kwargs = {} if gripper is None else {"gripper_types": gripper}
    return TabletopShelf(
        robots=robot, controller_configs=config, objects=tuple(objects), seed=seed,
        control_freq=20, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=list(CAMERAS),
        camera_heights=resolution, camera_widths=resolution,
        camera_depths=True, camera_segmentations="instance", **kwargs,
    )


def run(
    prompt: str,
    grippers=DEFAULT_PAIRS,
    objects=("milk", "can", "cereal", "bread"),
    seed: int = 0,
    num_grasps: int = 200,
    topk: int = 40,
    planner: str = "diffusion",
    out_dir: str | Path = "outputs/grasp_scene",
) -> dict:
    if not server_available():
        raise RuntimeError(server_status())

    env = build_env(objects, seed)
    try:
        obs = env.reset()
        scene = build_scene_graph(env)
        print(scene.table(), "\n")

        spec = parse_task(prompt, scene)
        print(f"prompt   : {prompt!r}")
        for note in spec.rationale:
            print(f"           {note}")
        if not spec.ok:
            raise SystemExit(f"could not parse the prompt: {spec.error}")
        target = scene.by_id(spec.object_id)
        destination = scene.by_id(spec.destination_id)
        print(f"parsed   : {spec.describe()}")
        print(f"           object at {np.round(target.position, 3)}, "
              f"destination at {np.round(destination.position, 3)}\n")

        client = GraspGenClient()
        try:
            cloud, sets = grasps_for_object(
                env, target.instance, grippers, obs=obs, client=client,
                num_grasps=num_grasps, topk=topk, planner=planner,
            )
        finally:
            client.close()

        print(f"cloud    : {cloud.describe()}  per camera {cloud.points_per_camera}\n")
        report = {
            "prompt": prompt,
            "seed": seed,
            "task": {
                "object_id": spec.object_id, "object_label": spec.object_label,
                "destination_id": spec.destination_id,
                "destination_label": spec.destination_label,
                "destination_position": destination.position.tolist(),
                "rationale": spec.rationale,
            },
            "cloud": {
                "instance": cloud.instance, "n_points": len(cloud),
                "extent": cloud.extent.tolist(), "centroid": cloud.centroid.tolist(),
                "points_per_camera": cloud.points_per_camera,
            },
            "grippers": {},
        }

        for name, grasp_set in sets.items():
            pair = resolve_pair(name)
            geometry = gripper_geometry(pair.graspgen)
            print(f"  {name:11s} {grasp_set.describe()}")
            candidates = []
            for grasp in grasp_set:
                entry = {"score": grasp.score, "pose": grasp.pose.tolist()}
                if pair.frame_verified:
                    position, rotation = grasp_to_eef_pose(grasp, pair)
                    entry["eef_position"] = position.tolist()
                    entry["eef_rotation"] = rotation.tolist()
                    entry["pre_grasp_position"] = approach_waypoint(grasp, gripper=pair).tolist()
                candidates.append(entry)
            report["grippers"][name] = {
                "robosuite": pair.robosuite,
                "graspgen": pair.graspgen,
                "family": geometry.family,
                "aperture": geometry.aperture,
                "tcp_depth": geometry.tcp_depth,
                "frame_verified": pair.frame_verified,
                "n_candidates": len(grasp_set),
                "candidates": candidates,
            }
            if not pair.frame_verified:
                print(f"  {'':11s} (frame convention unmeasured; no end-effector targets)")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"seed{seed}_{spec.object_label}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {path}")
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="put the milk carton on the top shelf")
    parser.add_argument("--grippers", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--objects", default="milk,can,cereal,bread")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--planner", default="diffusion", choices=("diffusion", "graspmoe"))
    parser.add_argument("--out", default="outputs/grasp_scene")
    args = parser.parse_args()
    run(
        prompt=args.prompt,
        grippers=tuple(g.strip() for g in args.grippers.split(",") if g.strip()),
        objects=tuple(o.strip() for o in args.objects.split(",") if o.strip()),
        seed=args.seed, num_grasps=args.num_grasps, topk=args.topk,
        planner=args.planner, out_dir=args.out,
    )


if __name__ == "__main__":
    main()
