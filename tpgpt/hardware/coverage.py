"""How many of an object's grasps a destination admits, and where the rest die.

A plan needs one grasp that survives everything: the table, the jaws, the
transported path into the destination. When a plan is refused the tally says
what removed the candidates *of that draw*, which on a stochastic sampler is a
statement about the draw. With :mod:`tpgpt.hardware.grasp_library` the draw is
gone -- the object has a fixed library -- so the same question becomes a
property of the task:

    of this object's 2000 grasps, how many survive to each stage for this
    destination, and how many have a transported path that is executable?

That is measured on the generator's own grasps, so it carries its notion of
what holds; sweeping hand poses geometrically would not. It runs offline, with
no camera and no robot, on a scene captured earlier (the shelf model plus the
obstacle cloud a plan recorded).

Read it as a funnel. A destination whose survivors all die at ``place_zone`` or
in the carry is blocked by the shelf, not by the object; one that loses
everything at ``hits_support`` is blocked by the hand against the table; and a
zero at the end over a whole library is a task this hand cannot do from that
pick, which is worth knowing before another evening at the robot.
"""

from __future__ import annotations

import collections
import dataclasses

import numpy as np


def coverage(object_name, object_pose, world, source, destination, destination_height,
             support_height, camera_position, start_q, end_effector, settings=None,
             plan_budget: int = 40, gripper: str = "franka_panda", library_size=None,
             cloud=None, poses=None, scores=None, clearance=None):
    """Walk an object's whole grasp library against one destination.

    Args:
        object_pose: ``(rotation, position)`` of the object's scan in the base
            frame -- what ``object_model.fit_object`` returns for a capture.
        world: the :class:`~tpgpt.hardware.shelf.ShelfWorld` to check against.
        source: the demonstration (:class:`SourceBundle`).
        plan_budget: how many screened survivors are planned and path-checked,
            in score order. The screening is cheap and covers the library; the
            path check is ~1 s each, so this bounds the run.
        cloud, poses, scores: the object's points (base frame) and the grasps to
            walk, already placed. Given, they replace the scan and its library --
            for a capture whose fit was refused, where the planner works from the
            camera's own cloud, and for tests.
        clearance: a gap the path must keep, m (``path_check.path_feasibility``);
            None is the planner's check.

    Returns a dict: the stage tally, what the planned ones died of, and the
    first admissible grasp's rank if there is one.
    """
    from tpgpt.hardware.grasp_library import library, placed
    from tpgpt.hardware.grasp_selection import (PathContext, ScreenContext, SelectionSettings,
                                                _try_plan, screen_candidates)
    from tpgpt.hardware.object_model import FULL_CLOUD_POINTS, ObjectModel
    from tpgpt.hardware.planner import TargetScene
    from tpgpt.hardware import path_check
    from tpgpt.sim.keypoints import carry_transform

    rotation, position = object_pose if object_pose is not None else (None, None)
    if cloud is None:
        model = ObjectModel.load(object_name)
        points, _ = model.sample(FULL_CLOUD_POINTS, 1)
        cloud = points @ np.asarray(rotation).T + np.asarray(position)
    if poses is None:
        poses, scores = library(object_name, gripper=gripper,
                                **({} if library_size is None else {"size": int(library_size)}))
        poses = placed(poses, rotation, position)
    cloud, poses = np.asarray(cloud, dtype=float), np.asarray(poses, dtype=float)
    if scores is None:
        scores = np.ones(len(poses))

    s = dataclasses.replace(settings or SelectionSettings(), max_plans=max(int(plan_budget), 1))
    context = ScreenContext(camera_position=camera_position, world=world,
                            destination=np.asarray(destination)[:2],
                            destination_height=float(destination_height),
                            carry_rotation=carry_transform(source.demonstration_labels)[0])
    screened = screen_candidates(cloud, poses, scores, float(support_height), s, context=context)
    survivors = [c for c in screened if c.rejected is None]      # Screened is a list of candidates

    scene = TargetScene(points=cloud, destination=np.asarray(destination, dtype=float),
                        destination_height=float(destination_height),
                        support_height=float(support_height))
    path = PathContext(world=world, end_effector=end_effector, start_q=start_q)
    planned, admissible, first_rank, unrefined = 0, 0, None, 0
    reasons, faults, detail = collections.Counter(), collections.Counter(), collections.Counter()
    for rank, candidate in enumerate(survivors[:int(plan_budget)]):
        planned += 1
        # The selection's own plan-and-verify, so that a candidate counted here is
        # one `select_and_plan` would have taken -- the transport map, its
        # determinant, the aim at the grasp, the workspace box and the clearances.
        result = _try_plan(source, scene, candidate, s, {})
        if result is None:
            reasons[str(candidate.rejected).split(":")[0]] += 1
            continue
        report = path_check.check_plan(result, path.world, path.end_effector,
                                       start_q=path.start_q, criterion=s.kinematic_criterion,
                                       approach_rejects=s.approach_rejects, clearance=clearance,
                                       joint_margin=s.joint_margin,
                                       reach_margin=s.reach_margin,
                                       retreat_clearance=s.retreat_clearance)
        unrefined += int(report.get("unrefined", 0))
        verdict = path_check.verdict(report)
        if verdict is None:
            admissible += 1
            first_rank = rank if first_rank is None else first_rank
        else:
            reasons[verdict.split("(")[0].split(":")[0].strip()] += 1
            # What touched what, counted once per candidate (a candidate with the
            # same fault at twenty waypoints counts once): "the hand meets the
            # shelf" is a different problem from "the carried object meets the
            # table", and which it is decides what to do about it.
            detail.update(report.get("faults") or {})
            faults.update({group(fault) for fault in (report.get("faults") or {})})
    return {
        "object": object_name,
        "library": len(poses),
        "stages": screened.stages,
        "screened_survivors": len(survivors),
        "planned": planned,
        "admissible": admissible,
        "first_rank": first_rank,
        "clearance": clearance,
        "postures_not_at_their_waypoint": unrefined,
        "why_the_planned_ones_failed": dict(reasons),
        "what_they_touched": dict(faults),
        "faults_detail": dict(detail),
        "screen_rejections": dict(collections.Counter(
            (c.rejected or "").split(":")[0] for c in screened if c.rejected)),
    }


def group(fault: str) -> str:
    """One fault line, with the particular link and waypoint taken out.

    ``"carry: fr3_link6 in shelf top_board"`` and ``"carry: hand_body in shelf
    top_board"`` are the same problem -- the arm cannot get under that board --
    and reading a hundred link names hides it.
    """
    stage, _, rest = fault.partition(":")
    stage, rest = stage.replace(" (relaxed)", ""), rest.strip()
    if rest.startswith("self "):
        return f"{stage}: the arm meets itself"
    if rest.startswith("the carried object"):
        for verb in (" pressed into ", " touches ", " in "):
            if verb in rest:
                return f"{stage}: the carried object vs {rest.split(verb, 1)[1].split(' with ')[0]}"
        return f"{stage}: {rest}"
    part, _, thing = rest.partition(" in ")
    who = ("the arm" if part.startswith("fr3_link") else
           "the hand" if part.startswith("hand_") else part)
    return f"{stage}: {who} in {thing}" if thing else f"{stage}: {rest}"


def describe(report: dict) -> str:
    """One line per stage, as a funnel a person can read."""
    lines = [f"{report['object']}: {report['library']} grasps in the library"]
    for name, before, after, passthrough in report["stages"]:
        note = "  (pass-through: nothing survived)" if passthrough else ""
        lines.append(f"  {name:16s} {before:5d} -> {after:5d}{note}")
    lines.append(f"  screened survivors {report['screened_survivors']}, "
                 f"planned {report['planned']}, admissible {report['admissible']}"
                 + (f", first at rank {report['first_rank']}" if report["first_rank"] is not None
                    else ""))
    if report["why_the_planned_ones_failed"]:
        lines.append("  the planned ones died of: "
                     + ", ".join(f"{k} {v}" for k, v in
                                 sorted(report["why_the_planned_ones_failed"].items(),
                                        key=lambda kv: -kv[1])))
    if report.get("what_they_touched"):
        top = sorted(report["what_they_touched"].items(), key=lambda kv: -kv[1])[:6]
        lines.append("  what they touched (candidates): "
                     + "; ".join(f"{k} x{v}" for k, v in top))
    return "\n".join(lines)


def main(argv=None):
    """Coverage of an object's grasp library against named shelf slots, offline.

    The scene is a capture the planner recorded (``<plan dir>/clouds.npz``: the
    obstacle cloud and the object's observed points). The object is either the
    capture's own, with its scan re-fitted to the observed points exactly as
    the planner does, or -- with ``--object`` and ``--at`` -- any object's scan
    set down at a chosen spot, resting pose and heading in that scene.

        python -m tpgpt.hardware.coverage \\
            --capture ~/tpgpt_plans/banana_20260923_231459_refused \\
            --demo ~/tpgpt_demos/banana_20260923_230159/source_bundle \\
            --slots top_left,top_right,bottom_left --json out.json
    """
    import argparse
    import json
    import os
    import time

    import yaml

    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--capture", required=True, help="a plan directory holding clouds.npz")
    ap.add_argument("--demo", required=True, help="the demonstration's source_bundle directory")
    ap.add_argument("--object", default=None, help="object to place (default: the capture's own)")
    ap.add_argument("--at", nargs=2, type=float, default=None,
                    help="x y on the table: set the scan down there instead of fitting it")
    ap.add_argument("--rest", type=int, default=0, help="resting pose index (with --at)")
    ap.add_argument("--heading", type=float, default=0.0, help="yaw about the table, deg (--at)")
    ap.add_argument("--slots", default="all", help="comma-separated slot names, or 'all'")
    ap.add_argument("--budget", type=int, default=25, help="survivors planned per slot")
    ap.add_argument("--start-q", nargs=7, type=float, default=None,
                    help="the arm's joint angles when planning starts (default: home). The "
                         "move from there onto the path is checked too, so where the arm was "
                         "parked matters; a plan record keeps it as joint_states_at_plan")
    ap.add_argument("--slot-depth", type=float, default=None,
                    help="move each slot to this depth into the shelf, m from the front edge "
                         "(the touched marks sit at about 0.09), keeping its position along "
                         "the shelf and its board; nothing in the config changes")
    ap.add_argument("--size", type=int, default=2000, help="grasp library size")
    ap.add_argument("--joint-margin", type=float, default=None,
                    help="refuse paths with joints 1-6 this close to a limit, rad (default: "
                         "off; path_check.JOINT_MARGIN is CRISP's 0.1)")
    from tpgpt.hardware.path_check import REACH_MARGIN
    ap.add_argument("--reach-margin", type=float, default=REACH_MARGIN,
                    help="refuse paths whose wrist comes this close to full stretch, m (default: "
                         "path_check.REACH_MARGIN, the planner's; 0 = off)")
    ap.add_argument("--retreat-clearance", type=float, default=None,
                    help="once clear of the placed object by this much in the retreat, stay that "
                         "far from it, m (default: off)")
    ap.add_argument("--clearance", type=float, default=None,
                    help="a gap every path must keep from the table, the shelf and the seen "
                         "points, m (default: none, the planner's check)")
    ap.add_argument("--config-dir", default=os.path.expanduser(
        "~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/config"),
        help="where shelf.yaml and camera_calibration.yaml are read (the installed copy, "
             "which is what the planner reads)")
    ap.add_argument("--json", default=None, help="write the full reports here")
    a = ap.parse_args(argv)

    from tpgpt.hardware import shelf as shelf_mod
    from tpgpt.hardware.end_effector import FRANKA_HAND
    from tpgpt.hardware.fr3 import HOME_Q
    from tpgpt.hardware.grasp_selection import SelectionSettings
    from tpgpt.hardware.object_model import FULL_CLOUD_POINTS, ObjectModel, _rot_z, fit_object
    from tpgpt.hardware.shelf import ShelfModel, ShelfWorld
    from tpgpt.hardware.source_bundle import SourceBundle

    cfg = yaml.safe_load(open(os.path.join(a.config_dir, "shelf.yaml")))
    cal = yaml.safe_load(open(os.path.join(a.config_dir, "camera_calibration.yaml")))
    table = float(cfg["shelf"]["anchors"]["table"])
    slots = shelf_mod.slots(cfg)
    camera = np.array(cal["camera_calibration"]["base_to_camera"]["translation"], dtype=float)

    capture = os.path.expanduser(a.capture)
    clouds = np.load(os.path.join(capture, "clouds.npz"))
    observed, obstacles = clouds["observed"], clouds["obstacles"]
    name = a.object or os.path.basename(capture.rstrip("/")).rsplit("_2026", 1)[0]
    model = ObjectModel.load(name)
    points, _ = model.sample(FULL_CLOUD_POINTS, 1)
    if a.at is None:
        fit = fit_object(model, observed, table, camera)
        rotation, position = fit.rotation, fit.position
        how = (f"fitted to the capture ({fit.explained:.2f} explained, {fit.median_mm:.1f} mm "
               f"median, accepted={fit.accepted})")
    else:
        rotation = _rot_z(np.radians(a.heading)) @ model.rests[a.rest]
        body = points @ rotation.T
        position = np.array([a.at[0], a.at[1], table - body[:, 2].min()])
        how = f"SYNTHETIC: resting pose {a.rest}, heading {a.heading:.0f} deg, at {a.at}"

    world = ShelfWorld(model=ShelfModel.from_config(cfg), cloud=obstacles, table_height=table)
    source = SourceBundle.load(os.path.expanduser(a.demo))
    # The planner node's defaults: tilt cap 90 (A10), visibility off for a fitted scan,
    # centred off, no grasp offset (A5), strict reachability.
    settings = SelectionSettings(max_tilt_deg=90.0, check_centred=False, check_visibility=False,
                                 grasp_offset=(0.0, 0.0, 0.0), kinematic_criterion="strict",
                                 joint_margin=a.joint_margin, reach_margin=a.reach_margin,
                                 retreat_clearance=a.retreat_clearance)
    want = list(slots) if a.slots == "all" else a.slots.split(",")
    if a.slot_depth is not None:
        shelf_model = world.model
        for slot in want:
            local = (np.asarray(slots[slot], dtype=float) - shelf_model.origin) @ shelf_model.R_base_shelf
            local[1] = float(a.slot_depth)
            slots[slot] = tuple(shelf_model.origin + shelf_model.R_base_shelf @ local)
        print(f"  slots moved to {a.slot_depth:.3f} m into the shelf: "
              + ", ".join(f"{s} {np.round(slots[s], 3).tolist()}" for s in want))
    print(f"{name}: {how}")
    print(f"  scene {os.path.basename(capture)}: {len(obstacles)} obstacle points, "
          f"table {table:+.4f}")
    print(f"  demo {a.demo}, library {a.size}, plan budget {a.budget}, clearance "
          f"{'none' if a.clearance is None else f'{a.clearance * 1000:.0f} mm'}", flush=True)
    out = {}
    for slot in want:
        xyz = slots[slot]
        t0 = time.monotonic()
        start = HOME_Q if a.start_q is None else np.asarray(a.start_q, dtype=float)
        report = coverage(name, (rotation, position), world, source, xyz, xyz[2], table, camera,
                          start, FRANKA_HAND, settings=settings, plan_budget=a.budget,
                          library_size=a.size, clearance=a.clearance)
        print(f"--- {slot} [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]  "
              f"({time.monotonic() - t0:.0f} s)")
        print(describe(report), flush=True)
        out[slot] = report
    if a.json:
        with open(os.path.expanduser(a.json), "w") as f:
            json.dump({"object": name, "how": how, "capture": capture, "demo": a.demo,
                       "slots": out}, f, indent=1, default=float)
        print("wrote", a.json)


if __name__ == "__main__":
    main()


__all__ = ["coverage", "describe", "group", "main"]
