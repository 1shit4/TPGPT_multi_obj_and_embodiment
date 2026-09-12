"""Re-measure a finished replay campaign's paths, geometrically, with no physics.

**Why this is a separate driver.** The rule this project keeps relearning is
that an end-to-end run is the worst way to find something out: it costs 12 to
25 minutes, it is stochastic, and there is a whole robot between the thing
changed and the number produced. A geometric measurement over the same cells
costs seconds, is deterministic, and can be checked against outcomes that are
already known.

That is exactly what a new path-clearance filter needs before it is trusted.
The filter claims that a transported path which sits inside scene geometry for
much of its length will fail; ``ROBOTICS_NOTES.md`` 7.38 measured that
relationship once, with a different instrument -- MuJoCo's narrowphase applied
to inverse-kinematics solutions -- and a threshold carried over from one
instrument to another is an assumption, not a measurement.

So this driver takes a campaign that has already run, reads which candidate each
cell actually executed (``grasp_chosen_index``, which indexes the *cached*
candidate set and is therefore reproducible), rebuilds that cell's scene, fits
the same map, and measures the resulting path with
:func:`~tpgpt.grasp.filters.path_clearance` and
:func:`~tpgpt.grasp.filters.path_kinematics`. Set beside the campaign's own
success column, that says whether the new instrument separates the cells that
worked from the ones that did not -- before an hour is spent on a run that
assumes it does.

Usage::

    python -m tpgpt.experiments.run_path_study --rows outputs/sel_iii/rows.json \\
        --out outputs/path_study_iii
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.experiments.pipeline import (
    SOURCE_GRIPPER,
    place_pose_for,
    _demonstrated_approach,
    _to_tool_frame,
)
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.run_keypoint_transport import (
    VARIANTS,
    _apply_grasp_offsets,
    _placement_for,
    build_scene,
    table_surface,
    transport,
)
from tpgpt.grasp.filters import path_clearance, path_kinematics
from tpgpt.grasp.grasps import contact_offset, to_grasp_convention
from tpgpt.grasp.grippers import resolve_pair
from tpgpt.grasp.pipeline import grasps_for_cloud
from tpgpt.perception.cameras import object_point_cloud
from tpgpt.perception.obstacles import scene_obstacles
from tpgpt.reporting.html import write_manifest
from tpgpt.sim.keypoints import GraspFrame, carry_indices
from tpgpt.sim.replay import make_position_controller_config
from tpgpt.transport.labels import PolicyLabels

#: Objects a replay campaign covers, in the order its rows list them.
STUDY_OBJECTS = ("cereal", "milk", "can", "bread")



def _place_zone_audit(env, grasp_set, pair, target_name, slot_pose, cloud) -> dict:
    """Ray-bundle verdict against exact-hand verdict, over every candidate.

    ``by_place_approach`` answers "is the corridor the hand sweeps into the slot
    clear" by casting rays through a cylinder that contains the hand. That is
    cheap and it is an **over-estimate of the hand's volume**, because a gripper
    is mostly air: two fingers and a wrist. The exact answer -- the hand's own
    surface sample placed at the release pose, measured against the scene's
    solid primitives -- costs about the same and is what every other check in
    the pipeline now uses.

    Returns the two counts and their disagreement, so the cost of the cheap test
    is a measured number.
    """
    from tpgpt.grasp.filters import (
        PATH_PENETRATION_TOLERANCE,
        by_place_approach,
        hand_envelope,
    )
    from tpgpt.grasp.grasps import contact_offset, grasp_to_eef_pose
    from tpgpt.grasp.grippers import gripper_geometry, gripper_points
    from tpgpt.perception.obstacles import deepest_penetration, scene_obstacles

    indices = np.arange(len(grasp_set.grasps))
    release = np.array([
        np.asarray(slot_pose[0], dtype=float)
        + grasp_to_eef_pose(grasp_set.grasps[int(i)], pair)[1] @ contact_offset(pair)
        for i in indices
    ])
    kept_rays, blocked = by_place_approach(
        grasp_set.grasps, indices, env, pair, release, exclude=(target_name,)
    )

    obstacles = scene_obstacles(env, exclude=(target_name,))
    hand = gripper_points(pair.graspgen, n=512)
    depth_tcp = float(gripper_geometry(pair.graspgen).tcp_depth)
    length, _ = hand_envelope(pair)
    kept_exact = []
    for position, i in zip(release, indices):
        grasp = grasp_set.grasps[int(i)]
        rotation = grasp.rotation
        clear = True
        # The release pose itself, and a short run back along the approach, so
        # the last stretch of the arrival is covered rather than one pose.
        for step in np.linspace(0.0, length, 5):
            base = position - rotation[:, 2] * (depth_tcp + step)
            depth, _ = deepest_penetration(base + hand @ rotation.T, obstacles)
            if depth > PATH_PENETRATION_TOLERANCE:
                clear = False
                break
        if clear:
            kept_exact.append(int(i))

    kept_rays = set(int(i) for i in kept_rays)
    kept_exact = set(kept_exact)
    return {
        "place_zone_candidates": int(len(indices)),
        "place_zone_kept_rays": len(kept_rays),
        "place_zone_kept_exact": len(kept_exact),
        "place_zone_rejected_but_clear": len(kept_exact - kept_rays),
        "place_zone_kept_but_fouling": len(kept_rays - kept_exact),
        "place_zone_blocked_by": blocked,
    }


def main(
    rows_path: str | Path,
    out_dir: str | Path = "outputs/path_study",
    ik_stride: int = 4,
    check_kinematics: bool = True,
    figures_dir: str | Path | None = None,
) -> dict:
    """Measure every cell of a finished campaign geometrically.

    Args:
        rows_path: ``rows.json`` from a replay campaign. Each row must carry
            ``gripper``, ``object``, ``variant`` and ``grasp_chosen_index``;
            rows without the last are cells the campaign skipped and are
            reported as such rather than silently dropped.
        ik_stride: Waypoint stride for the kinematic pass.
        check_kinematics: Run it at all. It is the slow half, a few seconds a
            cell against a few hundred milliseconds for the clearance sweep.
        figures_dir: Draw the demonstration and its transported counterpart for
            every cell, with the keypoint displacements between them. Reading a
            path is often faster than reading a table of its statistics, and the
            trajectory *between* the keypoints is the part no keypoint
            constrains -- which is exactly where a map that looks healthy on
            every summary number can still be wrong.
    """
    rows_path, out_dir = Path(rows_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = json.loads(rows_path.read_text())

    print("recording the source demonstration (reshelving, seed 0)", flush=True)
    labels, source_placement, ok = record_source_placement(0)
    if not ok:
        raise RuntimeError("the source demonstration failed")
    # Exactly the two conversions the replay driver makes, for exactly the
    # reasons recorded there: the keypoints pin the fingertip point, and the
    # cubes are built in the grasp convention. A study that skipped either
    # would be measuring a path the campaign never executed.
    labels = _to_tool_frame(labels, contact_offset(SOURCE_GRIPPER))
    labels = PolicyLabels(
        positions=labels.positions,
        velocities=labels.velocities,
        orientations=to_grasp_convention(labels.orientations, SOURCE_GRIPPER),
        gripper=labels.gripper,
        time_belief=labels.time_belief,
    )
    reference_approach = _demonstrated_approach(labels)
    grasp_index, release_index = carry_indices(labels)

    from robosuite.controllers import load_composite_controller_config

    config = make_position_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    variants = {v.name: v for v in VARIANTS}

    print(f"\n{'hand':11}{'object':8}{'variant':20}{'inside%':>9}{'maxmm':>8}"
          f"{'medmm':>8}{'reach':>8}{'ran ok':>8}  culprits")
    print("-" * 100, flush=True)

    out_rows = []
    for row in source_rows:
        gripper, name = row.get("gripper"), row.get("object")
        variant = variants.get(row.get("variant"))
        record = {"gripper": gripper, "object": name,
                  "variant": row.get("variant"),
                  "campaign_success": bool(row.get("success")),
                  "campaign_placed": row.get("placed"),
                  "chosen_index": row.get("grasp_chosen_index")}
        if variant is None or record["chosen_index"] is None:
            record["skipped"] = row.get("skipped") or row.get("failed") or "no grasp recorded"
            out_rows.append(record)
            print(f"  {gripper:9}{name:8}{row.get('variant','?'):20} {record['skipped'][:50]}",
                  flush=True)
            continue

        figure_state = None
        env = build_scene(objects=STUDY_OBJECTS, seed=0,
                          controller_config=config, gripper=gripper)
        try:
            obs = env._get_observations()
            cloud = object_point_cloud(env, name, obs=obs)
            grasp_set = grasps_for_cloud(cloud, gripper)
            grasp = _apply_grasp_offsets(
                GraspFrame.from_grasp(grasp_set.grasps[int(record["chosen_index"])]),
                0.0, 0.0,
            )
            placement = _placement_for(
                env, cloud, grasp, "top_middle", table_surface(env), {}
            )
            slot_pose = place_pose_for(
                env, "top_middle", grasp_set.grasps[int(record["chosen_index"])],
                gripper, float(np.ptp(cloud.points[:, 2])),
            )
            result = transport(labels, source_placement, placement, variant=variant)
            rotations = result["warped_rotations"]
            pair = resolve_pair(gripper)
            clearance = path_clearance(
                env, result["warped"], rotations, pair,
                carried_points=placement.points,
                carry_span=(grasp_index, release_index),
                obstacles=scene_obstacles(env, exclude=(name,)),
            )
            record.update({
                "inside_fraction": round(clearance["inside_fraction"], 4),
                "max_depth_mm": round(clearance["max_depth"] * 1000, 2),
                "median_inside_depth_mm": round(
                    clearance["median_inside_depth"] * 1000, 2),
                "inside_waypoints": clearance["inside_waypoints"],
                "waypoints": clearance["waypoints"],
                "culprits": clearance["culprits"],
                "object_max_depth_mm": round(
                    float(np.max(clearance["depth_object"])) * 1000, 2)
                if clearance["depth_object"] is not None else None,
                "min_det": result["min_det"],
                "aim_map": result["aim_map"],
            })
            if check_kinematics:
                record.update({
                    f"ik_{k}": v for k, v in path_kinematics(
                        env, result["warped"], rotations, pair, stride=ik_stride
                    ).items()
                })

            # --- does the plan drive the hand *through* the object it is about
            # to pick up? ---------------------------------------------------
            #
            # The transported path is the demonstration's motion, which descends
            # from above. The grasp it is built around need not be a descent: a
            # candidate approaching a cereal box from the side asks for a hand
            # held sideways, and the warp then carries that orientation down the
            # demonstrated descent. The fingers sweep downward while pointing
            # sideways, and they meet the object before the jaws are told to
            # shut.
            #
            # Measured as the deepest the hand's own surface sample gets inside
            # the **target** object over the approach, i.e. every waypoint up to
            # the close. The target is the one obstacle excluded from the scene
            # set everywhere else, precisely because the fingers are meant to
            # close around it -- but they are meant to close around it *at the
            # grasp*, not to plough through it on the way in.
            target_box = [
                o for o in scene_obstacles(env, include_movable=True)
                if o.name.startswith(f"{name}_")
            ]
            approach = path_clearance(
                env, result["warped"][: grasp_index + 1],
                rotations[: grasp_index + 1], pair, obstacles=target_box,
            )
            record.update({
                "approach_into_object_mm": round(approach["max_depth"] * 1000, 2),
                "approach_waypoints_inside": approach["inside_waypoints"],
            })

            # --- where does the plan say the object should be released, and is
            # that where it went? ------------------------------------------
            #
            # Distinguishes two failures a placement error cannot tell apart:
            # a plan that asks for the wrong place, and an arm that does not get
            # to the right one. The first would be a keypoint or map fault and
            # should be impossible -- ``phi`` interpolates the release keypoint
            # exactly -- so measuring it is how that assumption gets checked
            # rather than assumed.
            commanded = np.asarray(result["warped"][release_index], dtype=float)
            record.update({
                "release_commanded_xyz": [round(float(v), 4) for v in commanded],
                "release_commanded_lateral_mm": round(float(np.linalg.norm(
                    commanded[:2] - np.asarray(placement.destination)[:2])) * 1000, 1),
                "release_commanded_above_board_mm": round(
                    float(commanded[2] - placement.destination_height) * 1000, 1),
            })

            # --- rank of the executed candidate among everything the planner
            # proposed ------------------------------------------------------
            scores = np.array([g.score for g in grasp_set.grasps], dtype=float)
            chosen = int(record["chosen_index"])
            record.update({
                "n_candidates": int(len(scores)),
                "chosen_score": round(float(scores[chosen]), 4),
                "chosen_score_rank": int((scores > scores[chosen]).sum()) + 1,
            })

            # --- is the transported path a descent, or does it follow the
            # grasp's own approach axis? ------------------------------------
            #
            # The question behind the yumi's four failures. ``phi`` carries the
            # source keypoint cube onto the target one, and if the two differ by
            # a large rotation then Eq. 11 rotates the commanded *gripper*
            # orientation by that much. What it does **not** necessarily do is
            # bend the *path*: the positions are ``phi(x)``, which is close to
            # rigid away from the keypoints, so a demonstration that descends
            # still descends. If that is right, a side grasp gets a hand held
            # sideways travelling straight down, and the fingers sweep through
            # the object.
            #
            # Measured as the angle between the direction the path actually
            # travels over its last 10 waypoints into the grasp and the grasp's
            # own approach axis. Zero means the hand arrives the way the grasp
            # asks; 90 means it arrives sideways to it.
            span = max(grasp_index - 10, 0)
            travel = np.asarray(result["warped"][grasp_index]) - np.asarray(
                result["warped"][span])
            norm = float(np.linalg.norm(travel))
            record["path_travel_mm"] = round(norm * 1000, 2)
            if norm > 1e-6:
                record["path_vs_grasp_axis_deg"] = round(float(np.degrees(np.arccos(
                    np.clip((travel / norm) @ np.asarray(grasp.approach), -1, 1)))), 1)
                record["path_vs_vertical_deg"] = round(float(np.degrees(np.arccos(
                    np.clip((travel / norm) @ np.array([0.0, 0.0, -1.0]), -1, 1)))), 1)

            # --- how badly is the path warped? -----------------------------
            #
            # A map can be a perfectly valid diffeomorphism -- no fold anywhere,
            # ``min det`` comfortably positive -- and still produce a path no
            # arm should fly: stretched, kinked, or wandering far from the
            # rigid motion the demonstration described. "It did not fold" is a
            # validity check, not a quality one, and these are the quality ones.
            src_path = np.asarray(labels.positions, dtype=float)
            warped_path = np.asarray(result["warped"], dtype=float)
            src_len = float(np.linalg.norm(np.diff(src_path, axis=0), axis=1).sum())
            out_len = float(np.linalg.norm(np.diff(warped_path, axis=0), axis=1).sum())
            steps = np.linalg.norm(np.diff(warped_path, axis=0), axis=1)
            src_steps = np.linalg.norm(np.diff(src_path, axis=0), axis=1)
            # Best-fit rigid motion of the source onto the warped path, so
            # "how far from a rigid carry is this" is a number. A rigid carry is
            # what the demonstration is; everything beyond it is the warp.
            sc, wc = src_path.mean(0), warped_path.mean(0)
            U, _, Vt = np.linalg.svd((src_path - sc).T @ (warped_path - wc))
            R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
            rigid = (src_path - sc) @ R + wc
            record.update({
                "path_length_ratio": round(out_len / src_len, 3) if src_len else None,
                "path_max_step_ratio": round(
                    float(np.max(steps / np.maximum(src_steps, 1e-9))), 2),
                "path_deviation_from_rigid_mm": round(
                    float(np.linalg.norm(warped_path - rigid, axis=1).max()) * 1000, 1),
            })

            # --- which way is the hand turned when it enters the slot? -----
            #
            # A Panda hand is **204 mm across its jaw axis** -- measured on
            # robosuite's own model, and GraspGen's agrees to 8 mm -- while the
            # top cubby leaves about 88 mm of usable depth between the board's
            # front edge and the back panel. So the hand only fits with its wide
            # axis lying **across** the shelf. Which way it is turned is decided
            # by the grasp taken at the pick, since the object is rigid with the
            # hand from the moment the jaws close.
            shelf_x = np.array([1.0, 0.0, 0.0])
            closing_at_release = np.asarray(
                result["warped_rotations"][release_index], dtype=float)[:, 0]
            record["release_jaw_axis_vs_shelf_depth_deg"] = round(float(np.degrees(
                np.arccos(np.clip(abs(closing_at_release @ shelf_x), -1, 1)))), 1)

            # --- is the place-side zone rejecting candidates the hand would
            # actually fit through? ----------------------------------------
            #
            # ``by_place_approach`` tests a **bundle of rays** filling a cylinder
            # of the hand's own radius and length behind the release pose. A
            # gripper is not a solid cylinder -- it is two fingers and a wrist
            # with a great deal of air in between -- so a cylinder containing it
            # clips panels the hand itself misses.
            #
            # The exact test is available and costs the same: put the hand's own
            # surface sample at the release pose and measure it against the
            # scene's primitives, which is what ``path_clearance`` does at every
            # other waypoint. This compares the two verdicts over **every**
            # candidate, so "the cheap test over-rejects" is a number rather
            # than an argument.
            record.update(
                _place_zone_audit(env, grasp_set, pair, name, slot_pose, cloud)
            )

            # Kept for the figure, which is drawn after the environment closes.
            figure_state = {
                "warped": result["warped"],
                "S": result["source_keypoints"],
                "T": result["target_keypoints"],
                # The figure plots each path's height against the surface it
                # is delivering onto, so these are the two destination heights.
                "surfaces": {
                    "source": source_placement.destination_height,
                    "target": placement.destination_height,
                },
            }
        except Exception as exc:                      # noqa: BLE001 - recorded
            record["failed"] = f"{type(exc).__name__}: {exc}"
        finally:
            env.close()
        if figures_dir is not None and figure_state is not None:
            try:
                from tpgpt.viz.keypoint_figures import figure_transported_trajectory

                Path(figures_dir).mkdir(parents=True, exist_ok=True)
                gap = record.get("path_vs_vertical_deg", float("nan"))
                figure_transported_trajectory(
                    labels.positions,
                    figure_state["warped"],
                    figure_state["S"],
                    figure_state["T"],
                    figure_state["surfaces"],
                    Path(figures_dir) / f"{gripper}_{name}.png",
                    release_index=release_index,
                    title=(f"{gripper} / {name}: the plan travels {gap:.0f} deg "
                           f"from vertical into the grasp"),
                )
                record["figure"] = f"{gripper}_{name}.png"
            except Exception as exc:                  # noqa: BLE001 - recorded
                record["figure_failed"] = f"{type(exc).__name__}: {exc}"
        out_rows.append(record)
        top = sorted(record.get("culprits", {}).items(),
                     key=lambda kv: -kv[1])[:2]
        print(
            f"  {gripper:9}{name:8}{record['variant']:20}"
            f"{record.get('inside_fraction', float('nan')):9.2%}"
            f"{record.get('max_depth_mm', float('nan')):8.1f}"
            f"{record.get('median_inside_depth_mm', float('nan')):8.1f}"
            f"{record.get('ik_reachable_fraction', float('nan')):8.2f}"
            f"{('  yes' if record['campaign_success'] else '   no'):>8}  "
            + ", ".join(f"{k}:{v}" for k, v in top),
            flush=True,
        )

    manifest = write_manifest(
        out_dir,
        title="Transported paths re-measured geometrically against a finished campaign",
        description=(
            "For every cell of an existing replay campaign, the grasp it "
            "actually executed is re-transported and the resulting path is "
            "measured against the scene's solid geometry -- how much of it "
            "sits inside something, how deep, and which panel -- plus a "
            "warm-started inverse-kinematics pass down the same path. No "
            "physics is stepped. Set beside the campaign's own success column, "
            "this says whether the geometric criterion separates the cells "
            "that worked from the ones that did not, which is what has to be "
            "true before it is used to select grasps."
        ),
        settings={
            "varied": {"cell": "every row of the source campaign"},
            "source_rows": str(rows_path),
            "fixed": {
                "source": "reshelving seed 0, one demonstration",
                "slot": "top_middle",
                "seed": 0,
                "grasp": "the candidate index the campaign recorded, from the "
                         "same cached candidate set, so the grasp is identical",
                "instrument": "analytic signed distance to the scene's solid "
                              "primitives; the gripper's own published surface "
                              "sample placed at every transported waypoint",
                "ik_stride": ik_stride,
            },
        },
        rows=out_rows,
    )
    (out_dir / "rows.json").write_text(json.dumps(out_rows, indent=2, default=float))
    print(f"\nwrote:\n  {manifest}\n  {out_dir / 'rows.json'}")
    return {"rows": out_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True,
                        help="rows.json from a finished replay campaign")
    parser.add_argument("--out", default="outputs/path_study")
    parser.add_argument("--ik-stride", type=int, default=4)
    parser.add_argument("--no-kinematics", action="store_true")
    parser.add_argument("--figures", default=None,
                        help="Directory to draw per-cell trajectory figures into.")
    args = parser.parse_args()
    main(args.rows, args.out, ik_stride=args.ik_stride,
         check_kinematics=not args.no_kinematics, figures_dir=args.figures)
