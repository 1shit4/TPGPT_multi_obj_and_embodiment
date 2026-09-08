"""Tier 2: can the arm physically follow a transported path?

The keypoint sweep in :mod:`tpgpt.experiments.run_keypoint_transport` scores map
geometry -- determinants, aim, transported orientation. None of that says whether
the resulting trajectory is one a robot can execute, and a keypoint construction
that produces a beautifully conditioned map through poses the arm cannot hold has
bought nothing.

This driver answers that, and it is deliberately built to answer *only* that.
It drives the arm along the transported labels **pose by pose under position
control** (:func:`~tpgpt.sim.replay.replay_labels`) -- no GP policy, no attractor
integration, no lag gate. So every number here is attributable to the keypoints
alone, which is exactly what a comparison of keypoint constructions needs.

**Read a replay as an upper bound.** It is what a construction can deliver with
the executor removed. If a replay fails, the keypoints are at fault. If a replay
succeeds and the policy rollout of the same path does not, the fault is in policy
execution -- the attractor integration, the feed-forward, the gate -- and not
here. That split is the reason this tier exists as its own driver rather than as
a mode on the campaign runner, which is hard-wired to torque control and always
ends in ``rollout_policy``.

Four questions, and what answers each:

=====================  ==================================================
question               measure
=====================  ==================================================
is the path feasible   ``reachable_fraction``, ``unreachable_waypoints``
does it execute        ``tracking_error_mean`` / ``_max``, and the split
                       ``_reachable`` vs ``_unreachable`` -- if the two
                       halves differ sharply the residual is kinematic
                       and no controller closes it
does the grasp hold    :func:`~tpgpt.experiments.diagnose.grasp_slip`
does anything touch    ``held_steps`` from the probe trace
does the task complete ``success``, ``placement_error_xy``
=====================  ==================================================

The feasibility split is aimed straight at the grasp-pose cube. Laying the cube
out in the grasp's own pose carries the full grasp orientation -- worth 0.2 to
1.7 degrees against 1.6 to 11.7 for the cloud box on filtered candidates -- but
``J_perp`` is one rotation per point, so it tilts the world by the same angle.
Section 7.7 measured a case where every point of a warped path was reachable in
*position* and only 3 of 20 with its commanded *orientation*. Geometry cannot see
that. This can.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.experiments.diagnose import grasp_slip, object_probe
from tpgpt.experiments.pipeline import (
    SOURCE_GRIPPER,
    _demonstrated_approach,
    _to_tool_frame,
)
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.run_keypoint_transport import (
    ABLATION_OBJECTS,
    VARIANTS,
    build_scene,
    table_surface,
    target_placement,
    transport,
)
from tpgpt.grasp.grasps import contact_offset
from tpgpt.reporting.html import write_manifest
from tpgpt.sim.replay import make_position_controller_config, replay_labels
from tpgpt.sim.rollout import slot_score
from tpgpt.transport.labels import PolicyLabels

#: The constructions carried into physics. Deliberately a subset: a replay is a
#: ~25 s physics episode against a millisecond geometry evaluation, and Sec. 7.26
#: is explicit that end-to-end runs confirm a result rather than find one.
#: The control plus the two cube orientations is the smallest set that can
#: separate "the cube helps" from "the grasp pose costs reachability".
REPLAY_VARIANTS = (
    "0_cloud_box",
    "2_cube_grasp_pose",
    "3_cube_grasp_pose_composed",
)

#: Objects replayed. The lemon is excluded: its cloud is 17 points, below the
#: pipeline's own ``MIN_CLOUD_POINTS`` of 40, and GraspGen-X refuses it -- so a
#: replay would be measuring a construction the pipeline would never reach.
REPLAY_OBJECTS = ("cereal", "milk", "can", "bread")


def replay_variant(env, labels, source_placement, target, variant, gripper="panda") -> dict:
    """Transport under one construction, then follow the result under position control.

    ``labels`` **must already be in the tool frame**. See :func:`main`: passing
    the wrist-frame demonstration instead costs the grasp-cube constructions
    exactly ``contact_offset(gripper)`` -- 41 mm on a Panda -- because they pin
    the fingertip point and the trajectory would be a wrist path.
    """
    result = transport(labels, source_placement, target, variant=variant)
    warped = PolicyLabels(
        positions=result["warped"],
        velocities=labels.velocities,
        orientations=result["map"].transport_orientations(
            labels.positions, labels.orientations
        ),
        gripper=None if labels.gripper is None else labels.gripper.copy(),
        time_belief=None if labels.time_belief is None else labels.time_belief.copy(),
    )
    replay = replay_labels(
        env,
        warped,
        tool_offset=contact_offset(gripper),
        score=slot_score(target.metadata.get("object_name", ""), "top_middle")
        if target.metadata.get("object_name")
        else None,
        probe=object_probe(env, target.metadata["object_name"])
        if target.metadata.get("object_name")
        else None,
    )
    row = {
        "variant": variant.name,
        # geometry, carried through so the two tiers can be read together
        "min_det": result["min_det"],
        "aim_map": result["aim_map"],
        "orientation_error_deg": result["orientation_error_deg"],
        "tilt_mid_path": result["tilt_mid_path"],
        # physics
        **{
            k: replay.metadata[k]
            for k in (
                "reachable_fraction", "unreachable_waypoints", "waypoints",
                "tracking_error_mean", "tracking_error_max",
                "tracking_error_reachable", "tracking_error_unreachable",
                "placement_error_xy",
            )
            if k in replay.metadata
        },
        "success": bool(replay.success),
        **grasp_slip(replay),
    }
    return row


def main(
    out_dir: str | Path = "outputs/keypoint_replay",
    slot: str = "top_middle",
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("recording the source demonstration (reshelving, seed 0)", flush=True)
    labels, source_placement, ok = record_source_placement(0)
    if not ok:
        raise RuntimeError("the source demonstration failed; replaying it is meaningless")

    # **Convert to the tool frame before transporting**, exactly as
    # ``pipeline.run`` does. The keypoints are anchored where the hand *holds*
    # the object -- for the grasp-cube constructions the cube's centre is
    # literally the grasp's fingertip point -- while the demonstration is
    # recorded at ``grip_site``, the wrist. Those are ``contact_offset`` apart:
    # 41 mm on a Panda.
    #
    # Measured, ``||phi(label at the grasp index) - target grasp TCP||``:
    #
    # ===============  ==============  =============
    # construction     wrist labels    tool labels
    # ===============  ==============  =============
    # cloud box        20 - 71 mm      45 - 136 mm
    # grasp cube       41 mm (exactly) **0.0 mm**
    # ===============  ==============  =============
    #
    # The cube's 41 mm is not an aim error, it is the frame offset arriving
    # undiminished, because a near-rigid map carries it straight through. A first
    # version of this driver replayed the wrist labels and the cube held the
    # object for 0 to 1 control steps out of ~119 as a direct result. 7.20 is the
    # same mistake and its own retraction warns that a null result deserves as
    # much suspicion as a surprising one.
    labels = _to_tool_frame(labels, contact_offset(SOURCE_GRIPPER))

    # Real planner grasps, filtered exactly as ``filter_grasps`` filters them.
    # A top-down recipe makes the task frame and the full grasp pose coincide,
    # so it cannot separate the two constructions this tier exists to compare.
    reference_approach = _demonstrated_approach(labels)

    # Position control has to be chosen at construction; assigning it afterwards
    # silently does nothing.
    from robosuite.controllers import load_composite_controller_config

    config = make_position_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    rows = []
    variants = [v for v in VARIANTS if v.name in REPLAY_VARIANTS]
    print(f"\n{'variant':20}{'object':8}{'reach':>8}{'trackmm':>9}"
          f"{'r/unr mm':>12}{'slip mm':>9}{'held':>6}{'place mm':>10}{'ok':>4}")
    print("-" * 90, flush=True)

    # **A fresh scene per replay.** A replay drives the arm through a whole
    # trajectory and closes the jaws, so it leaves the object displaced and the
    # arm parked wherever the path ended. Reusing one environment makes every
    # run after the first inherit that, and the *ordering* then decides the
    # result: a first version of this driver reused one scene and produced
    # placement errors of 354, 698 and 1170 mm and a reachable fraction of 0%
    # for every variant on the last object, purely from accumulated disturbance.
    #
    # ``run_experiments.campaign`` already builds per run for the same reason.
    # Rebuilding costs a few seconds against a ~90 s replay, which is cheap for
    # the guarantee that every cell starts from the identical seeded state.
    reference_tcp = None
    for name in REPLAY_OBJECTS:
        for variant in variants:
            env = build_scene(
                objects=REPLAY_OBJECTS, seed=seed, controller_config=config
            )
            try:
                obs = env._get_observations()
                target, _ = target_placement(
                    env, name, slot, height_fraction=0.5, obs=obs,
                    grasp_source="graspgen", reference_approach=reference_approach,
                )
                target.metadata["object_name"] = name
                # The rebuild is only sound if the seeded scene is reproducible.
                # Assert it rather than assume it: an unseeded placement sampler
                # would silently turn this comparison into a comparison of
                # different scenes.
                if name == REPLAY_OBJECTS[0]:
                    if reference_tcp is None:
                        reference_tcp = np.asarray(target.grasp.tcp, dtype=float)
                    elif not np.allclose(target.grasp.tcp, reference_tcp, atol=1e-6):
                        raise RuntimeError(
                            "the seeded scene is not reproducible across rebuilds: "
                            f"{name}'s grasp TCP moved "
                            f"{np.linalg.norm(target.grasp.tcp - reference_tcp) * 1000:.1f} mm"
                        )
                row = replay_variant(env, labels, source_placement, target, variant)
            except ValueError as exc:
                row = {"object": name, "variant": variant.name, "skipped": str(exc)}
            except Exception as exc:
                row = {"object": name, "variant": variant.name,
                       "failed": f"{type(exc).__name__}: {exc}"}
            finally:
                env.close()
            row["object"] = name
            rows.append(row)
            print("  " + _replay_line(row), flush=True)

    manifest = write_manifest(
        out_dir,
        title="Keypoint constructions under position-control replay",
        description=(
            "Transported paths followed pose by pose under position control "
            "-- no policy, no attractor integration, no lag gate -- so every "
            "number is attributable to the keypoint construction alone. "
            "Answers whether a well-conditioned map is actually executable: "
            "reachability, tracking error split by whether IK could hold the "
            "pose, grasp slip, and task completion. Read as an upper bound: "
            "a failure here is the keypoints', a failure only under the "
            "policy is the executor's."
        ),
        settings={
            "varied": {"variant": list(REPLAY_VARIANTS), "object": list(REPLAY_OBJECTS)},
            "fixed": {
                "source": "reshelving seed 0, one demonstration",
                "slot": slot,
                "seed": seed,
                "scene": "rebuilt fresh for every replay, asserted reproducible",
                "control": "JOINT_POSITION via solve_ik, no policy",
                "grasp_source": "recipe (top_down_grasp) at mid height",
                "labels": "tool frame -- converted by _to_tool_frame before transport",
                "tool_offset": "contact_offset('panda')",
            },
        },
        rows=rows,
    )
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote:\n  {manifest}\n  {out_dir / 'rows.json'}")
    return {"rows": rows}


def _replay_line(row: dict) -> str:
    if "failed" in row or "skipped" in row:
        note = row.get("failed") or row.get("skipped")
        return f"{row.get('variant', '?'):18}{row.get('object', '?'):8} {note[:60]}"
    g = lambda k, d=float("nan"): row.get(k, d)
    return (
        f"{row['variant']:18}{row['object']:8}"
        f"{g('reachable_fraction'):8.0%}"
        f"{g('tracking_error_mean') * 1000:9.1f}"
        f"{g('tracking_error_reachable') * 1000:6.1f}/{g('tracking_error_unreachable') * 1000:<5.1f}"
        f"{g('slip_max') * 1000:9.1f}{g('held_steps', 0):6.0f}"
        f"{g('placement_error_xy') * 1000:10.1f}{'  yes' if row.get('success') else '   no':>4}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/keypoint_replay")
    parser.add_argument("--slot", default="top_middle")
    args = parser.parse_args()
    main(args.out, args.slot)
