"""Can this hand hold this object? Grasp, close, lift, carry — nothing else.

**No transportation map, no policy, no shelf.** Every other driver in this
package measures a *transported* plan, which puts the map, the keypoints, the
grasp selection, the controller and the shelf geometry in series: when a cell
fails, six things could be responsible. This one removes all of them. It asks
the planner for grasps on the object in front of it, executes them directly, and
reports whether the object was still in the hand at the end.

That makes it the right instrument for two questions that the end-to-end
campaigns cannot answer:

**Which hand-object pairs are physically viable at all?** A fleet of two-finger
jaws measured on a box, a carton, a can and a loaf cannot show that more fingers
buy anything, because those are all things a parallel jaw is good at. Pairs that
cannot hold an object should be found here, in a minute, rather than inferred
from a failed campaign twenty minutes in.

**How far should the jaws close?** Four rules have been tried and each fails
somewhere, because the window between gripping and crushing is per-object *and*
per-hand and no constant sits inside all of them (``FINDINGS.md`` 8z). A per-pair
sweep needs a per-pair measurement, and this is it.

**One object at a time, deliberately.** ``pipeline`` lays several objects out
together and its placement sampler works **in the order it is given**, so adding
or reordering an object moves all of them -- which is why ``outputs/expR_bottle``
could not be compared cell-by-cell with the run it was meant to extend. A
single-object scene makes every cell independent. The cost is stated rather than
hidden: **bench cells are not comparable with the four-object campaigns**, and a
pair that holds an object here may still fail in a cluttered scene.

**Several grasps per pair, also deliberately.** Experiment P
(``FINDINGS.md`` 8n) measured the spread *within* a pair to be larger than the
spread *between* pairs -- the same hand and object lifting 43 mm at one grasp
pose and 408 mm at another. One grasp per pair cannot tell "this pair failed"
from "this grasp failed", so the default is five.

Usage::

    python -m tpgpt.experiments.run_grasp_bench --plan-only
    python -m tpgpt.experiments.run_grasp_bench --grippers panda,robotiq3f \\
        --objects can,hammer --ranks 5 --out outputs/bench

Writes ``manifest.json`` (with provenance, automatically) and ``rows.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from tpgpt.experiments import diagnose
from tpgpt.grasp.grasps import approach_waypoint, grasp_to_eef_pose
from tpgpt.sim.kinematics import solve_ik
from tpgpt.grasp.grippers import VERIFIED_PAIRS, resolve_pair
from tpgpt.grasp.filters import filter_grasps, offset_from_centre
from tpgpt.grasp.pipeline import grasps_for_cloud
from tpgpt.perception.cameras import object_point_cloud, scene_point_cloud
from tpgpt.reporting.html import write_manifest

#: Hands the registry says are usable. Read from the registry rather than
#: hardcoded, so a hand that gains or loses a calibrated depth enters or leaves
#: the bench without this file changing.
DEFAULT_GRIPPERS = VERIFIED_PAIRS

#: Objects to try. The four grocery meshes every campaign has used, plus the
#: shapes a multi-finger hand should be better at than a parallel jaw -- a
#: handle, a rim, a hole, an offset centre of mass. See
#: ``tabletop_shelf.OBJECT_CLASSES``.
DEFAULT_OBJECTS = (
    "can", "milk", "cereal", "bread",
    "hammer", "wrench", "pot", "mug", "nut_square", "nut_round",
)

#: Grasp candidates tried per pair, best-scoring first.
DEFAULT_RANKS = 5

#: Position tolerance for the reachability screen, in metres. Matches
#: ``solve_ik``'s own default, which sits below the impedance controller's
#: tracking error -- so a pose that passes is not one the controller will
#: miss for kinematic reasons.
IK_TOLERANCE = 5e-3

#: Below this many points the object is too thinly seen to plan on. Matches
#: ``pipeline.MIN_CLOUD_POINTS`` so the bench and the pipeline agree about what
#: "seen" means.
MIN_CLOUD_POINTS = 40

#: A lift of at least this counts as having left the table, in metres. Matches
#: ``diagnose.LIFT_HEIGHT``.
LIFT_HEIGHT = 0.02

#: How far the object is carried sideways after the lift, in metres, and back.
#: A grip that survives a straight lift can still fail under lateral
#: acceleration, which is what a carry applies and a lift does not.
CARRY_DISTANCE = 0.20

#: Control steps per phase of the motion.
STEPS = {"pre": 45, "descend": 40, "close": 25, "lift": 45, "carry": 45, "settle": 10}

#: Impedance gains. Matches ``verify.execute_grasp`` so the bench and the depth
#: calibration are driving the arm the same way.
STIFFNESS = 600.0
ROTATIONAL_STIFFNESS = 60.0
ROTATIONAL_DAMPING = 12.0

#: Height above the table the carry is performed at, in metres above the pick.
LIFT_TO = 0.15


def _scene(gripper: str, obj: str, seed: int, camera_size: int = 256,
           cameras: bool = True):
    """One object, one hand. Nothing else in the scene.

    ``cameras`` is off for the second and later grasps of a cell: the cloud is
    extracted once and every grasp is planned from it, so re-rendering three
    views per grasp would cost time and ~250 MB of resident memory for a
    picture nothing reads. Rendering does not touch the dynamics, and the
    placement is seeded, so the scene is otherwise identical -- which
    ``diagnose.replay_preconditions`` re-checks per grasp anyway.
    """
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.replay import make_position_controller_config
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    pair = resolve_pair(gripper)
    # **Position control, not Cartesian impedance.** The bench asks whether a
    # hand can hold an object, so the arm must be able to put the hand where
    # the grasp says. Under impedance the arm could not: on `inspire/can` the
    # hand ended 43 to 100 mm from poses inverse kinematics reports as
    # reachable, and a grasp attempted 100 mm away measures the controller
    # rather than the gripper. This is the same reasoning that makes the Tier 2
    # replay position-controlled -- it is a measurement, not a better robot.
    config = make_position_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    vision = dict(
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["workspace", "sideview", "birdview"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        camera_depths=True,
        camera_segmentations="instance",
    ) if cameras else dict(has_offscreen_renderer=False, use_camera_obs=False)
    env = TabletopShelf(
        robots="Panda",
        gripper_types=pair.robosuite,
        controller_configs=config,
        objects=(obj,),
        control_freq=20,
        seed=seed,
        **vision,
    )
    env.reset()
    return env


def _drive(env, position, rotation, steps, command, seed, arm="right"):
    """Move the arm so the gripper site traces a straight line to ``position``.

    Each waypoint is solved by inverse kinematics, warm-started from the
    previous solution, and commanded as an absolute joint target -- the same
    mechanism ``replay.replay_labels`` uses. Returns the final seed
    configuration so the next leg continues from it.
    """
    from tpgpt.sim.replay import _joint_action

    robot = env.robots[0]
    start = np.array(env.sim.data.site_xpos[
        robot.eef_site_id[arm] if isinstance(robot.eef_site_id, dict)
        else robot.eef_site_id])
    for step in range(steps):
        alpha = (step + 1) / steps
        target = start + alpha * (np.asarray(position) - start)
        result = solve_ik(env, target, rotation, arm,
                          position_tolerance=IK_TOLERANCE, seed_qpos=seed)
        seed = result.qpos
        env.step(_joint_action(env, robot, arm, seed, command))
    return seed


def _grip_and_carry(env, grasp, gripper, obj, probes, closure=None,
                    grasp_qpos=None):
    """Approach, close, lift, carry, and report what the hand did.

    Returns a dict of measurements. The phases are the same ones
    ``verify.execute_grasp`` uses, with a lateral carry added: a grip that
    survives a straight lift can still shed the object under lateral
    acceleration, and the carry is what a reshelving task actually asks for.

    Args:
        closure: Stop the jaws at this fraction of their travel, 0 open to 1
            shut, instead of commanding them fully closed. ``None`` is the
            plain ``+1`` every campaign so far has used and stays the default.

            This exists because four closing rules have now failed the same
            way: the window between gripping and crushing is per-object *and*
            per-hand, and no constant sits inside all of them
            (``FINDINGS.md`` 8z). A fraction is the knob a per-pair table would
            set, and ``replay.set_closure`` already commands it linearly and
            monotonically on every hand -- robosuite's own action interface
            cannot, because it takes the *sign* of a command and discards its
            size.
    """
    from tpgpt.sim.replay import closing_direction, set_closure
    from tpgpt.sim.replay import _joint_action, closing_direction, set_closure

    position, rotation = grasp_to_eef_pose(grasp, gripper)
    pre_grasp = approach_waypoint(grasp, standoff=0.12, gripper=gripper)
    arm = "right"
    robot = env.robots[0]
    site_id = (robot.eef_site_id[arm] if isinstance(robot.eef_site_id, dict)
               else robot.eef_site_id)
    seed = np.array(env.sim.data.qpos[np.asarray(
        robot.composite_controller.part_controllers[arm].qpos_index)])

    trace = {"closure": [], "force": [], "slip": [], "held": []}

    def sample():
        trace["closure"].append(float(probes["closure"]()))
        trace["force"].append(float(diagnose.grip_force(env, obj)))
        trace["slip"].append(float(probes["slip"](env)))
        trace["held"].append(bool(diagnose._gripper_touches(env, obj)))

    def hold(steps, command, record=False):
        for _ in range(steps):
            env.step(_joint_action(env, robot, arm, seed, command))
            if record:
                sample()

    def go(target, steps, command, settle=0, record=False):
        nonlocal seed
        seed = _drive(env, target, rotation, steps, command, seed, arm)
        hold(settle, command, record)

    start_z = float(env.object_position(obj)[2])

    go(pre_grasp, STEPS["pre"], -1.0, settle=STEPS["settle"])
    go(position, STEPS["descend"], -1.0, settle=STEPS["settle"])

    # NOTE: the screened inverse-kinematics solution is deliberately **not**
    # commanded directly here. ``_drive`` warm-starts each waypoint from the
    # previous one, so it can diverge and end far from a pose the screen
    # solved -- `panda/can` finishes 272 mm away on two of its grasps. Jumping
    # the command to the screened configuration instead makes it worse, not
    # better: the arm then swings toward a distant target it cannot track, and
    # `robotiq3f/can` went from 3.0-4.4 mm of reach error to 274-627 mm.
    # Measured, both ways; the honest reading is that the path is the limit,
    # and ``reach_total_mm`` reports it per grasp rather than hiding it.

    # How close the hand actually got, before the jaws move. Without this a
    # cell that failed because the arm could not reach the pose is
    # indistinguishable from one that reached it and lost its grip, and those
    # want opposite fixes. Split by axis as well as in total, because only the
    # component along the **closing** axis decides whether the object ends up
    # between the jaws -- the approach axis tolerates 120 to 135 mm on a
    # parallel jaw (ROBOTICS_NOTES 7.2, 7.27).
    reached = np.array(env.sim.data.site_xpos[site_id])
    reach_error = position - reached
    axes = {"closing": rotation[:, 0], "jaw": rotation[:, 1],
            "approach": rotation[:, 2]}
    reach = {f"reach_{k}_mm": float(reach_error @ v) * 1000 for k, v in axes.items()}
    reach["reach_total_mm"] = float(np.linalg.norm(reach_error)) * 1000

    # Close, in place. ``command`` of 0.0 means "stay where you are": robosuite's
    # gripper interface integrates the *sign* of the command, and sign(0) is 0.
    grip_model = env.robots[0].gripper
    grip_model = grip_model["right"] if isinstance(grip_model, dict) else grip_model
    if closure is None:
        command = 1.0
    else:
        set_closure(grip_model, closing_direction(grip_model), float(closure))
        command = 0.0
    hold(STEPS["close"], command, record=True)
    closure_at_close = float(probes["closure"]())
    force_at_close = float(diagnose.grip_force(env, obj))

    lifted = position + np.array([0.0, 0.0, LIFT_TO])
    go(lifted, STEPS["lift"], command, settle=STEPS["settle"], record=True)
    lift_height = float(env.object_position(obj)[2]) - start_z
    closure_at_lift = float(probes["closure"]())

    # Carry sideways and back. Both directions, so the grip is loaded each way.
    across = lifted + np.array([0.0, CARRY_DISTANCE, 0.0])
    go(across, STEPS["carry"], command, settle=STEPS["settle"], record=True)
    go(lifted, STEPS["carry"], command, settle=STEPS["settle"], record=True)

    held = trace["held"]
    carry_start = STEPS["close"]
    return {
        **reach,
        "lift_height": lift_height,
        "final_height": float(env.object_position(obj)[2]) - start_z,
        "closure_at_close": closure_at_close,
        "closure_at_lift": closure_at_lift,
        "closure_max": float(np.nanmax(trace["closure"])) if trace["closure"] else float("nan"),
        "force_at_close": force_at_close,
        "force_max": float(np.max(trace["force"])) if trace["force"] else 0.0,
        "slip_max": float(np.max(trace["slip"])) if trace["slip"] else 0.0,
        "held_fraction": float(np.mean(held[carry_start:])) if len(held) > carry_start else 0.0,
        "held_at_end": bool(held[-1]) if held else False,
    }


def run_cell(gripper: str, obj: str, *, seed: int = 0, ranks: int = DEFAULT_RANKS,
             plan_only: bool = False, camera_size: int = 256,
             closure: float | None = None) -> list[dict]:
    """Every grasp tried for one hand and one object."""
    base = {"gripper": gripper, "object": obj, "seed": seed,
            "closure_commanded": closure}
    env = _scene(gripper, obj, seed, camera_size)
    try:
        cloud = object_point_cloud(env, obj)
        base["cloud_points"] = len(cloud)
        if len(cloud) < MIN_CLOUD_POINTS:
            return [{**base, "outcome": "object_not_seen", "grasp_rank": None}]

        grasps = grasps_for_cloud(cloud, gripper)
        base["candidates"] = len(grasps.grasps)
        if not grasps.grasps:
            return [{**base, "outcome": "no_grasp_generated", "grasp_rank": None}]

        # Filter before ranking. The planner's own score does **not** order
        # candidates by executability: on `panda/can` only **25 of 100** of its
        # candidates approach from above at all, and its six top-scoring ones
        # include four that come in almost horizontally, at an end-effector
        # height of 846 to 872 mm against a table top at 840. Executed
        # unfiltered they drive the hand along the table and shove the object
        # 115 to 280 mm without ever lifting it, which measures the ranking and
        # not the hand.
        #
        # Only the pick-side stages run. `check_place_approach` is off and no
        # `place_pose` is given, because this bench has no shelf: the question
        # is whether the hand can hold the object, not where it can put it.
        scene = scene_point_cloud(env, exclude=(obj,))
        funnel = filter_grasps(
            grasps.grasps, gripper,
            target_points=cloud.points,
            scene_points=scene,
            env=env,
            target_name=obj,
            support_normal=(0.0, 0.0, 1.0),
            check_place_approach=False,
            # ``by_centre_offset`` deliberately NOT run as a rejection. The
            # funnel's own docstring calls it "a tie-break among grasps that
            # are already admissible, not a reason to reject one outright", and
            # on a hand with few options it is the stage that empties the set:
            # on `inspire/can` it cut 20 candidates to 9 and took every
            # reachable one with it. It is recorded per grasp below instead.
        )
        survivors = np.asarray(funnel.survivors, dtype=int)
        base["rejected_by"] = funnel.rejected_by
        base["funnel_flags"] = sorted(funnel.flags)
        if len(survivors) == 0:
            return [{**base, "outcome": "no_grasp_survived", "grasp_rank": None}]

        # Screen for reachability, which the funnel does not do -- its own
        # reachability stage is retired as inert (FINDINGS.md 8z item 1g).
        # Executing a pose the arm cannot hold measures the arm, not the hand:
        # on `inspire/can` **46 of 100** candidates are reachable and **0 of
        # the 8** the funnel kept were, so without this the hand is scored on
        # grasps it was never able to attempt.
        reachable, solutions = [], {}
        for i in survivors:
            position, rotation = grasp_to_eef_pose(grasps.grasps[int(i)], gripper)
            result = solve_ik(env, position, rotation, "right",
                              position_tolerance=IK_TOLERANCE)
            if result.reachable:
                reachable.append(int(i))
                solutions[int(i)] = np.array(result.qpos)
        base["survivors"] = int(len(survivors))
        base["reachable_survivors"] = len(reachable)
        # Falling back rather than failing, the way every funnel stage does:
        # a cell with nothing reachable still reports what happened when its
        # best-scoring candidate was tried, flagged so it cannot be read as a
        # clean run.
        pool = np.asarray(reachable or survivors, dtype=int)
        base["reachability_fell_back"] = not reachable

        scores = np.array([grasps.grasps[i].score for i in pool])
        order = pool[np.argsort(-scores)][:ranks]
        centre = env.object_position(obj)
        if plan_only:
            return [
                {**base, "grasp_rank": int(r), "grasp_index": int(i),
                 "score": float(grasps.grasps[i].score), "outcome": "planned"}
                for r, i in enumerate(order)
            ]

        rows = []
        for rank, index in enumerate(order):
            grasp = grasps.grasps[int(index)]
            row = {**base, "grasp_rank": rank, "grasp_index": int(index),
                   "score": float(grasp.score),
                   "offset_from_centre_mm": float(
                       offset_from_centre(grasp, centre, resolve_pair(gripper))) * 1000}
            cell = (_scene(gripper, obj, seed, camera_size, cameras=False)
                    if rank else env)
            try:
                checks = diagnose.replay_preconditions(cell, gripper, obj)
                diagnose.require(checks, context=f"{gripper}/{obj} rank {rank}")
                probes = {
                    "closure": diagnose.jaw_closure_probe(cell, gripper),
                    "slip": diagnose.slip_probe(cell, obj),
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row.update(_grip_and_carry(
                        cell, grasp, gripper, obj, probes, closure=closure,
                        grasp_qpos=solutions.get(int(index))))
                row["outcome"] = (
                    "held" if row["held_at_end"] and row["final_height"] > LIFT_HEIGHT
                    else "lifted_then_lost" if row["lift_height"] > LIFT_HEIGHT
                    else "never_lifted"
                )
            except Exception as exc:
                row["outcome"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                if rank:
                    cell.close()
            rows.append(row)
        return rows
    finally:
        env.close()


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/grasp_bench")
    parser.add_argument("--grippers", default=",".join(DEFAULT_GRIPPERS))
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--ranks", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument(
        "--cells", default=None,
        help="Subset, as 'panda/can,robotiq3f/hammer'. Recorded in the "
             "manifest, so a partial run cannot be mistaken for a full one.",
    )
    parser.add_argument(
        "--closure", type=float, default=None,
        help="Stop the jaws at this fraction of travel (0 open, 1 shut) "
             "instead of commanding them fully closed. Default is the plain "
             "+1 every campaign so far has used.",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Perception and grasp selection, no physics. Seconds per cell "
             "instead of minutes, and it answers 'is this grid even runnable' "
             "before an hour is spent finding out.",
    )
    args = parser.parse_args(argv)

    grippers = [g.strip() for g in args.grippers.split(",") if g.strip()]
    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    if args.cells:
        cells = [tuple(c.split("/")) for c in args.cells.split(",")]
    else:
        cells = [(g, o) for g in grippers for o in objects]

    rows = []
    print(f"{'hand':<11}{'object':<11}{'cloud':>7}{'surv':>6}{'rank':>5}"
          f"{'reach mm':>10}{'lift mm':>9}{'closure':>9}{'slip mm':>9}"
          f"{'held':>6}  outcome")
    for gripper, obj in cells:
        for row in run_cell(gripper, obj, seed=args.seed, ranks=args.ranks,
                            plan_only=args.plan_only,
                            camera_size=args.camera_size,
                            closure=args.closure):
            rows.append(row)
            print(
                f"{row['gripper']:<11}{row['object']:<11}"
                f"{row.get('cloud_points', 0):>7}{row.get('survivors', 0):>6}"
                f"{str(row.get('grasp_rank', '-')):>5}"
                f"{row.get('reach_total_mm', float('nan')):>10.1f}"
                f"{row.get('lift_height', float('nan')) * 1000:>9.1f}"
                f"{row.get('closure_at_lift', float('nan')):>9.2f}"
                f"{row.get('slip_max', float('nan')) * 1000:>9.1f}"
                f"{str(row.get('held_at_end', '-')):>6}  {row['outcome']}",
                flush=True,
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    physical = [r for r in rows if "held_at_end" in r]
    write_manifest(
        out,
        title="Grasp bench: can this hand hold this object?",
        description=(
            "One object on the table, one hand, several planner grasps each. "
            "Approach, close, lift, carry sideways and back. No transportation "
            "map, no policy, no shelf."
        ),
        settings={
            "varied": {"gripper": grippers, "object": objects,
                       "grasp_rank": list(range(args.ranks))},
            "fixed": {
                "scene": "TabletopShelf, ONE object, rebuilt fresh per grasp",
                "seed": args.seed,
                "camera": f"{args.camera_size}x{args.camera_size}, three views",
                "grasps": "GraspGen-X via tpgpt.grasp.cache, put through "
                          "filters.filter_grasps' pick-side stages, then "
                          "ranked by the planner's own score. Cached because "
                          "the planner is an unseeded diffusion model "
                          "(ROBOTICS_NOTES 7.15)",
                "filters": "visibility, target containment, jaw width, "
                           "support approach, collision, centre offset. No "
                           "place-side stage and no reachability stage",
                "gripper_command": (
                    "plain +1 shut; no force target, no feedback"
                    if args.closure is None else
                    f"jaws stopped at closure fraction {args.closure}"),
                "control": "Cartesian impedance, stiffness 600, same as "
                           "verify.execute_grasp",
                "carry": f"{CARRY_DISTANCE} m sideways and back, at "
                         f"{LIFT_TO} m above the pick",
                "preconditions": "gripper_mounted, scene_unstepped, "
                                 "object_placement, closure_calibrated -- "
                                 "checked per grasp, aborts before physics",
                "cells": args.cells or "all",
            },
        },
        results={
            "grasps_attempted": len(physical),
            "held": sum(1 for r in physical if r["outcome"] == "held"),
            "lifted_then_lost": sum(
                1 for r in physical if r["outcome"] == "lifted_then_lost"),
            "never_lifted": sum(
                1 for r in physical if r["outcome"] == "never_lifted"),
        },
        thresholds={"lift_height_m": LIFT_HEIGHT,
                    "min_cloud_points": MIN_CLOUD_POINTS},
    )
    print(f"\nwrote {out}/rows.json and manifest.json")
    return out


if __name__ == "__main__":
    main()
