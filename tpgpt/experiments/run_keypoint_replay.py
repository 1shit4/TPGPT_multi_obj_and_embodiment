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

from tpgpt.experiments.diagnose import (
    grasp_slip,
    object_probe,
    replay_preconditions,
    require,
    unreachable_segments,
)
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

#: The constructions carried into physics: the control and the winner.
#:
#: The composed variant is dropped, not overlooked. It ties the plain grasp-pose
#: cube in physics (3 of 4 each) while being worse in geometry -- stage 2's
#: determinant falls to 0.08-0.30 -- and it *loses* the exact aim, 0.0 mm rising
#: to 2.8-13.0 mm, because stage 2 satisfies the contacts by dragging the cube's
#: centre off its target. With no measured gain and an extra parameter there is
#: nothing left to test.
REPLAY_VARIANTS = ("0_cloud_box", "2_cube_grasp_pose")

#: Objects replayed. The lemon is excluded: its cloud is 17 points, below the
#: pipeline's own ``MIN_CLOUD_POINTS`` of 40, and GraspGen-X refuses it -- so a
#: replay would be measuring a construction the pipeline would never reach.
REPLAY_OBJECTS = ("cereal", "milk", "can", "bread")

#: Grippers replayed.
#:
#: **Not** limited by what the server has resident: it loads a sampler lazily on
#: the first request naming a gripper, so any registered pair works and the first
#: call for a hand costs 25 to 72 s instead of the usual 4 to 12 (7.28). An
#: earlier version of this note said otherwise and restricted the set to three
#: hands for no reason.
#:
#: The question these answer: the grasp cube is centred on the grasp TCP with a
#: **fixed** half extent and encodes nothing about the hand -- not the jaw
#: aperture, not the fingertip depth, not the finger count. If that is a good
#: enough representation, the map should behave the same across hands. The
#: geometry sweep already says it does; this asks whether the plan *executes*.
#:
#: Six hands spanning the registry's tool offsets and both kinematic families
#: that lift anything:
#:
#: =========== =========== ========== =====================================
#: hand        tool offset aperture   why it is here
#: =========== =========== ========== =====================================
#: yumi            24.3 mm    50 mm   narrowest jaw; smallest offset
#: xarm            26.7 mm    85 mm   revolute, 90 deg closing angle
#: panda           41.1 mm    80 mm   the source hand, and the control
#: robotiq85       47.8 mm    85 mm   revolute linkage
#: robotiq140      60.8 mm   125 mm   widest jaw, deepest grip_site (270 mm)
#: umi            117.2 mm    80 mm   the extreme: 15 mm depth tolerance
#: =========== =========== ========== =====================================
#:
#: ``rethink`` is omitted only to keep the run near 45 minutes; it sits at
#: 35.1 mm, between the xarm and the panda, and adds no span. ``robotiq3f`` and
#: ``inspire`` are multi-finger: the 3F lifts a can but its single closing axis
#: is an approximation (anisotropy 4.3), and the Inspire hand's fingers travel
#: 0.7 to 6.5 mm against 29 to 90 mm for every other hand, so it does not
#: actuate and cannot execute any plan (7.28).
REPLAY_GRIPPERS = ("yumi", "xarm", "panda", "robotiq85", "robotiq140", "umi")


def _closure_summary(replay) -> dict:
    """How far the jaws shut during the replay, on a cross-hand scale.

    ``0`` is fully open and ``1`` is fully closed on air, for every hand, so one
    threshold reads the same on a Panda and a Robotiq 2F-140.

    Values are **not clipped**, but a reading above 1 is *not* evidence of a
    grasp, and an early version of this docstring claimed it was. The
    calibration's "closed" spread is taken after 40 settle steps of free-air
    closing; a longer or harder press compresses the linkage further with
    nothing between the fingers at all. Measured: an XArm on the cereal reached
    ``closure_max`` **1.04** while ``held_steps`` was **0**. So above 1 means
    only "pressed harder than during calibration". Below 0 means forced wider
    than the open pose. Both are kept because they are real states of the hand,
    not because either identifies a grasp.

    **Read ``closure_at_lift``, not ``closure_max``, for whether the object was
    in the jaws.** ``closure_at_lift`` is the reading at the first commanded
    close, where the jaws have just taken hold: a low value means something is
    holding them apart, a high one means they have travelled further because
    there is less between them. ``closure_max`` runs over the *whole* replay,
    which includes the jaws shutting on air after the object is released, so a
    ``closure_max`` of 1.00 is **not** evidence of a failed grasp. Measured on
    one hand, which is what makes the point:

    ========================  =========  ==========  ======  ========  ======
    variant, object           shut@lift  closure max held    place mm  ok
    ========================  =========  ==========  ======  ========  ======
    cloud box, cereal              0.63    **1.00**       0     232.0  no
    grasp-pose cube, cereal        0.42        0.62     110      34.5  yes
    cloud box, can                 0.39    **1.00**      82      29.6  **yes**
    ========================  =========  ==========  ======  ========  ======

    The first and third rows share a ``closure_max`` of 1.00 and differ in
    outcome entirely; their ``closure_at_lift`` of 0.63 against 0.39 is what
    separates them, and it separates them the right way round -- the failing
    grasp let the jaws travel *further*, because less of the object was between
    them.

    ``held_steps`` remains the direct measure of contact. What closure adds is
    **why** a hand held nothing: jaws that never moved, against jaws that shut
    fully on empty air. That distinction is what the joint-position reading
    could not make on any hand, and it is why a Robotiq 2F-140 closing hard was
    read as its jaws flying open (7.28).

    Returns ``nan`` for an uncalibrated hand rather than a plausible zero,
    which would read as "wide open throughout" and look identical to a hand
    that never closed.
    """
    trace = (getattr(replay, "metadata", None) or {}).get("probe") or {}
    values = np.asarray(trace.get("closure", []), dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return {"closure_max": float("nan"), "closure_at_lift": float("nan")}
    grip = np.asarray(replay.gripper, dtype=float)
    shut = np.flatnonzero(grip > 0)
    # Read at the *first* commanded close, where the jaws have just taken hold,
    # rather than at the end, where the object may already have been released.
    index = int(shut[0]) if len(shut) else 0
    return {
        "closure_max": float(np.nanmax(values)),
        "closure_at_lift": float(values[min(index, len(values) - 1)]),
    }


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
        # This hand's own fingertip offset. The transported labels are a
        # fingertip path, so IK has to step back to *this* wrist -- 41.1 mm on a
        # panda, 60.8 mm on a robotiq140. Getting it from the gripper rather
        # than assuming one is the whole point of the multi-hand run.
        tool_offset=contact_offset(gripper),
        score=slot_score(target.metadata.get("object_name", ""), "top_middle")
        if target.metadata.get("object_name")
        else None,
        probe=object_probe(env, target.metadata["object_name"], gripper=gripper)
        if target.metadata.get("object_name")
        else None,
    )
    row = {
        "variant": variant.name,
        "gripper": gripper,
        "tool_offset_mm": float(np.linalg.norm(contact_offset(gripper)) * 1000),
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
        # Did the jaws actually shut, on a scale that means the same thing on
        # every hand? Without this a hand that never closed is indistinguishable
        # from one that closed and dropped the object, and the raw ``jaw``
        # channel cannot tell them apart across embodiments (7.28).
        **_closure_summary(replay),
        **grasp_slip(replay),
        # Which *segment* the arm could not hold, not just how much of the path.
        **unreachable_segments(replay, warped),
    }
    return row


def main(
    out_dir: str | Path = "outputs/keypoint_replay",
    slot: str = "top_middle",
    seed: int = 0,
    grippers=REPLAY_GRIPPERS,
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
    print(
        "  hands: " + ", ".join(
            f"{h} (offset {np.linalg.norm(contact_offset(h)) * 1000:.1f} mm)"
            for h in grippers
        ),
        flush=True,
    )
    rows = []
    variants = [v for v in VARIANTS if v.name in REPLAY_VARIANTS]
    print(f"\n{'variant':20}{'hand':11}{'object':8}{'reach':>7}{'trackmm':>9}"
          f"{'slip':>7}{'held':>6}{'shut@lift':>10}{'shutMax':>8}{'place':>8}"
          f"{'worst seg':>11}{'ok':>4}")
    print("-" * 108, flush=True)

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
    #: First-cell object placement, keyed by ``(hand, object)``.
    #:
    #: Per object because each sits in its own place in the seeded scene, and
    #: per *hand* because the same seed does not settle them identically: the
    #: placement sampler is seeded the same, but the scene's 60 settle steps
    #: then run with a different gripper attached, and the can comes to rest
    #: 14.6 mm away between a Panda and a Robotiq 2F-140. That is a real ~15 mm
    #: scene difference baked into every cross-hand comparison, and it must not
    #: be reported as a reproducibility failure.
    object_reference: dict[tuple[str, str], np.ndarray] = {}
    for gripper in grippers:
        for name in REPLAY_OBJECTS:
            for variant in variants:
                env = build_scene(
                    objects=REPLAY_OBJECTS, seed=seed, controller_config=config,
                    gripper=gripper,
                )
                try:
                    # **Before any physics.** Every check here corresponds to a
                    # bug that was found only after a 12 to 25 minute campaign
                    # had already produced plausible numbers: the wrong hand
                    # mounted, a reused scene, a moved object, an uncalibrated
                    # jaw. All four are answerable in milliseconds from a fresh
                    # environment, so the campaign refuses to start instead of
                    # discovering it in the results.
                    require(
                        replay_preconditions(
                            env, gripper, name,
                            object_reference=object_reference.get((gripper, name)),
                        ),
                        f"{variant.name}/{gripper}/{name}",
                    )
                    object_reference.setdefault(
                        (gripper, name),
                        np.asarray(env.object_position(name), dtype=float),
                    )
                    obs = env._get_observations()
                    target, _ = target_placement(
                        env, name, slot, height_fraction=0.5, obs=obs,
                        grasp_source="graspgen",
                        reference_approach=reference_approach,
                        gripper=gripper,
                    )
                    target.metadata["object_name"] = name
                    # The rebuild is only sound if the seeded scene is
                    # reproducible. Assert it rather than assume it: an unseeded
                    # placement sampler would silently turn this comparison into
                    # a comparison of different scenes. Checked on the first
                    # hand only, since a different hand legitimately gets a
                    # different grasp TCP.
                    if name == REPLAY_OBJECTS[0] and gripper == grippers[0]:
                        if reference_tcp is None:
                            reference_tcp = np.asarray(target.grasp.tcp, dtype=float)
                        elif not np.allclose(target.grasp.tcp, reference_tcp, atol=1e-6):
                            raise RuntimeError(
                                "the seeded scene is not reproducible across "
                                f"rebuilds: {name}'s grasp TCP moved "
                                f"{np.linalg.norm(target.grasp.tcp - reference_tcp) * 1000:.1f} mm"
                            )
                    row = replay_variant(
                        env, labels, source_placement, target, variant,
                        gripper=gripper,
                    )
                # **Both, and deliberately.** ``ValueError`` is "too little
                # cloud to describe the object"; ``RuntimeError`` is "every
                # candidate is beyond the approach filter". Both are the
                # pipeline legitimately refusing an object -- there is no
                # admissible grasp for this hand in this scene -- and neither is
                # a failure of the keypoint construction, so filing them under
                # the generic ``failed`` bucket understates every variant
                # equally and misattributes the cause. The Tier 1 sweep already
                # files the identical condition as a skip; these two drivers
                # must agree or the two tiers cannot be read together.
                except (ValueError, RuntimeError) as exc:
                    row = {"object": name, "variant": variant.name,
                           "skipped": str(exc)}
                except Exception as exc:
                    row = {"object": name, "variant": variant.name,
                           "failed": f"{type(exc).__name__}: {exc}"}
                finally:
                    env.close()
                row.update(object=name, gripper=gripper)
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
                "grasp_source": (
                    "graspgen -- real GraspGen-X candidates for *this* hand, "
                    "filtered to within MAX_APPROACH_MISMATCH_DEG of the "
                    "demonstrated approach. A top-down recipe would make the "
                    "task frame and the full grasp pose coincide and could not "
                    "separate the two constructions."
                ),
                "labels": "tool frame -- converted by _to_tool_frame before transport",
                "tool_offset": (
                    "contact_offset(<this hand>) -- per hand, not the source's"
                ),
                "preconditions": (
                    "gripper_mounted, scene_unstepped, object_placement, "
                    "closure_calibrated -- checked on every cell, campaign "
                    "aborts on failure"
                ),
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
        # The hand, too. With one gripper it was inferable from context; with
        # six it is not, and a refusal that does not name the hand cannot be
        # read at all -- the whole point of the run is which hand did what.
        return (f"{row.get('variant', '?'):18}{row.get('gripper', '?'):11}"
                f"{row.get('object', '?'):8} {note[:56]}")
    g = lambda k, d=float("nan"): row.get(k, d)
    return (
        f"{row['variant']:18}{row.get('gripper', '?'):11}{row['object']:8}"
        f"{g('reachable_fraction'):7.0%}"
        f"{g('tracking_error_mean') * 1000:9.1f}"
        f"{g('slip_max') * 1000:7.1f}{g('held_steps', 0):6.0f}"
        # Cross-hand, so "the jaws never shut" is distinguishable from "they
        # shut and the object came out" while the run is still going (7.28).
        f"{g('closure_at_lift'):10.2f}{g('closure_max'):8.2f}"
        f"{g('placement_error_xy') * 1000:8.1f}"
        f"{str(row.get('worst_segment', '-')):>11}"
        f"{'  yes' if row.get('success') else '   no':>4}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/keypoint_replay")
    parser.add_argument("--slot", default="top_middle")
    parser.add_argument(
        "--grippers", default=None,
        help=(
            "Comma-separated registry short names. Defaults to the six-hand "
            "set in REPLAY_GRIPPERS, which spans 24.3 to 117.2 mm of tool "
            "offset and 50 to 125 mm of aperture."
        ),
    )
    args = parser.parse_args()
    main(
        args.out,
        args.slot,
        grippers=tuple(args.grippers.split(",")) if args.grippers
        else REPLAY_GRIPPERS,
    )
