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
             cloud=None, poses=None, scores=None):
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
                            carry_rotation=carry_transform(source.tool_labels)[0])
    screened = screen_candidates(cloud, poses, scores, float(support_height), s, context=context)
    survivors = [c for c in screened if c.rejected is None]      # Screened is a list of candidates

    scene = TargetScene(points=cloud, destination=np.asarray(destination, dtype=float),
                        destination_height=float(destination_height),
                        support_height=float(support_height))
    path = PathContext(world=world, end_effector=end_effector, start_q=start_q)
    planned, admissible, first_rank = 0, 0, None
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
                                       start_q=path.start_q, criterion=s.kinematic_criterion)
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


__all__ = ["coverage", "describe", "group"]
