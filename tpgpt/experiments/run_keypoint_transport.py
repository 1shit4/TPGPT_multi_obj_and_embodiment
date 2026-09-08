"""Transport one demonstration onto a different object (paper Sec. III-A, V-A).

The validated reshelving demonstration moves a 5x5x9 cm box from a table to a
shelf. This experiment takes *that same demonstration* and, without recording a
new one, transports it onto a milk carton, a can, a cereal box, a loaf and a
lemon standing in a different scene with a different shelf -- objects it has
never seen, whose only description is a segmented point cloud.

What makes that possible is the keypoint recipe: a box fitted to the object's
own cloud, in a frame built from the grasp and the surface the object rests on
(:func:`~tpgpt.sim.keypoints.task_frame`). Because the recipe is shared and the
geometry is not, corner *i* means the same thing on a carton and on a lemon, and
the two sets pair elementwise as Sec. III-A requires.

**Scored geometrically, not by execution.** ``demo.py`` and ``rollout.py`` are
still specific to the reshelving scene, so executing here would mean changing the
two modules the 17/20 result depends on. What is measured instead is where the
map delivers the hand: to the grasp the target object actually needs, and to the
destination surface rather than above or through it. Those are the quantities a
rollout would consume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.perception.cameras import object_point_cloud
from tpgpt.reporting.html import write_manifest
from tpgpt.sim.keypoints import (
    GraspFrame,
    ObjectPlacement,
    GRASP_CUBE_HALF_EXTENT,
    carry_indices,
    scene_keypoints,
    top_down_grasp,
)
from tpgpt.metrics.curves import point_to_curve_distance
from tpgpt.metrics.transport import (
    lift_deviation,
    orientation_transport_error,
    tilt_profile,
)
from tpgpt.transport.maps import TransportMap
from tpgpt.viz.keypoint_figures import (
    figure_ablation,
    figure_keypoint_scene,
    figure_transported_trajectory,
)

#: Objects to transport onto: box, tall box, cylinder, irregular, near-sphere.
ABLATION_OBJECTS = ("cereal", "milk", "can", "bread", "lemon")

#: Target grasp yaw offsets, degrees from the cloud's natural narrow axis.
YAW_OFFSETS = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)

#: Below this many points a cloud is too thin to describe an object's extent.
#: Not a hard limit -- the extractor still runs, and that is the point: the
#: lemon's 12-point cloud produced a box 0.6 x 1.2 cm across a 5 cm fruit while
#: every map diagnostic stayed green. The diagnostics measure the map's
#: self-consistency, not whether the box describes the object.
SPARSE_CLOUD_POINTS = 50


@dataclass(frozen=True)
class KeypointVariant:
    """One keypoint construction to measure.

    ``parts`` names the families kept from the emitted set, exactly as
    ``pipeline._select_parts`` routes them, so a variant is data rather than a
    branch.
    """

    name: str
    box: str = "cloud"
    orientation: str = "task"
    cube_half_extent: float = GRASP_CUBE_HALF_EXTENT
    contacts: bool = False

    @property
    def aim_is_pinned(self) -> bool:
        """Whether this construction makes the grasp point a keypoint.

        When it does, ``aim_vs_planned`` is ~0 **by construction** -- the cube's
        centre *is* the target grasp TCP and ``phi`` interpolates keypoints
        exactly -- so it is not evidence of anything and must not be read as a
        win. The metrics that stay informative are the orientation error (a
        *derivative*, which no keypoint pins), the determinant, the release
        clearance and the reachability.
        """
        return self.box == "grasp_cube"


#: The constructions compared. Variant 0 is today's default and the control.
VARIANTS = (
    KeypointVariant("0_cloud_box"),
    KeypointVariant("1_cloud_box_contacts", contacts=True),
    KeypointVariant("2_cube_task", box="grasp_cube"),
    KeypointVariant("3_cube_grasp_pose", box="grasp_cube", orientation="grasp"),
    KeypointVariant("4_cube_task_contacts", box="grasp_cube", contacts=True),
    KeypointVariant(
        "5_cube_grasp_pose_contacts", box="grasp_cube", orientation="grasp", contacts=True
    ),
)

#: Cube half extents swept to confirm the flat region survives real clouds.
CUBE_SIZES = (0.005, 0.01, 0.02, 0.03, 0.06)

#: Approach tilts swept, in degrees out of the support plane.
#:
#: The axis that separates the two corner orientations. ``task_frame`` is
#: invariant to it by construction, so at 0 degrees the task-frame cube and the
#: grasp-pose cube produce the *identical* map -- a designed internal control,
#: and a measurement dead end. Everything interesting is at the other values.
TILT_OFFSETS = (0.0, 15.0, 30.0, 45.0)

#: robosuite's TableArena puts the table's *top* surface at ``table_offset``;
#: the thickness hangs below it. ``env.table_top`` adds half the thickness on
#: top of that and so reads about 23 mm above where objects actually rest.
TABLE_SURFACE_IS_OFFSET = True


def build_scene(
    objects=ABLATION_OBJECTS,
    seed: int = 0,
    camera_size: int = 256,
    controller_config: dict | None = None,
):
    """A tabletop-shelf scene with depth and instance segmentation enabled.

    Args:
        controller_config: Composite controller config. Must be supplied **at
            construction** -- assigning one afterwards does nothing. Passing the
            position-control config from
            :func:`~tpgpt.sim.replay.make_position_controller_config` is what
            makes the scene replayable.
    """
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    env = TabletopShelf(
        robots="Panda",
        objects=objects,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["workspace", "sideview", "birdview"],
        camera_heights=camera_size,
        camera_widths=camera_size,
        camera_depths=True,
        controller_configs=controller_config,
        camera_segmentations="instance",
        control_freq=20,
        seed=seed,
    )
    env.reset()
    return env


def table_surface(env) -> float:
    """World height objects rest at on the table."""
    return float(env.table_offset[2]) if TABLE_SURFACE_IS_OFFSET else float(env.table_top)


def rotate_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def target_placement(
    env,
    instance: str,
    slot: str,
    yaw_offset_deg: float = 0.0,
    height_fraction: float = 0.5,
    obs: dict | None = None,
    tilt_offset_deg: float = 0.0,
    grasp_source: str = "recipe",
    gripper: str = "panda",
    grasp_rank: int = 0,
) -> tuple[ObjectPlacement, np.ndarray]:
    """Describe an object in the scene and where the task wants it.

    Args:
        yaw_offset_deg: Rotate the grasp's closing axis away from the cloud's
            natural narrow direction. The keypoint box is built in a frame
            derived from the closing axis, so the closing *direction* changes the
            keypoints while the grasp's height does not.
        tilt_offset_deg: Tilt the grasp's **approach** out of the plane, about
            its own closing axis.

            This is the axis that separates the two corner orientations, and
            without it they cannot be told apart. ``task_frame`` projects the
            closing axis perpendicular to the support normal and keeps that
            normal as its third column, so it is *invariant* to this rotation by
            construction -- while the full grasp pose is not. With a purely
            top-down grasp the two frames differ only by a half turn about the
            closing axis, which maps a cube's corners onto themselves, so both
            modes produce the identical map. That coincidence is a useful
            internal control and a measurement dead end.
        grasp_source: ``"recipe"`` derives the grasp from the cloud with
            :func:`top_down_grasp`, which always approaches straight down.
            ``"graspgen"`` takes a real ranked candidate from GraspGen-X, which
            carries whatever tilt the planner chose -- the realistic case, and
            the only one that exercises a hand's actual approach.
        gripper: Registry short name, when ``grasp_source="graspgen"``.
        grasp_rank: Which ranked candidate to take.

    Raises:
        RuntimeError: if ``grasp_source="graspgen"`` and the server returns
            nothing. Deliberately not a silent fall back to the recipe: a run
            that quietly measured a top-down grasp while reporting a planned one
            would be worse than no run.
    """
    cloud = object_point_cloud(env, instance, obs=obs)
    support = table_surface(env)
    if grasp_source == "recipe":
        grasp = top_down_grasp(cloud.points, height_fraction=height_fraction)
    elif grasp_source == "graspgen":
        from tpgpt.grasp.pipeline import grasps_for_cloud

        grasp_set = grasps_for_cloud(cloud, gripper)
        if not len(grasp_set):
            raise RuntimeError(
                f"GraspGen-X returned no candidate for {instance!r} with "
                f"{gripper!r}; refusing to substitute a top-down recipe"
            )
        grasp = GraspFrame.from_grasp(grasp_set.grasps[min(grasp_rank, len(grasp_set) - 1)])
    else:
        raise ValueError(
            f"unknown grasp_source {grasp_source!r}; expected 'recipe' or 'graspgen'"
        )
    if yaw_offset_deg:
        rotation = rotate_about([0.0, 0.0, 1.0], np.radians(yaw_offset_deg))
        grasp = GraspFrame(grasp.tcp, grasp.approach, rotation @ grasp.closing)
    if tilt_offset_deg:
        # About the grasp's own closing axis, so the closing direction -- and
        # therefore the task frame -- is untouched and only the approach moves.
        rotation = rotate_about(grasp.closing, np.radians(tilt_offset_deg))
        grasp = GraspFrame(grasp.tcp, rotation @ grasp.approach, grasp.closing)

    destination = env.slot_poses()[slot]
    placement = ObjectPlacement(
        points=cloud.points,
        grasp=grasp,
        support_height=support,
        destination=destination,
        destination_height=float(destination[2]),
    )
    return placement, cloud.points


def corresponding_point(world_point, source_meta: dict, target_meta: dict) -> np.ndarray:
    """The materially corresponding point on the target object.

    A point is located by its position *relative to the object's own box*, in
    the object's own task frame, normalised by the box's half extents. The same
    normalised coordinates on the target box name the corresponding point.

    This is what the keypoint correspondence asks ``phi`` to do, so it is the
    right thing to score the warp against. Scoring it against a separately
    scripted grasp instead measures the gap between two grasps, not the map.
    """
    source_frame = np.array(source_meta["task_frame"])
    target_frame = np.array(target_meta["task_frame"])
    source_half = np.maximum(np.array(source_meta["box_half_extents"]), 1e-9)
    local = ((np.asarray(world_point, float) - np.array(source_meta["box_centre"]))
             @ source_frame) / source_half
    return np.array(target_meta["box_centre"]) + (
        local * np.array(target_meta["box_half_extents"])
    ) @ target_frame.T


def transport(
    source_labels, source_placement, target, include_contacts=False, variant=None
) -> dict:
    """Fit the map for one target and measure what it delivers.

    Args:
        variant: A :class:`KeypointVariant` selecting the construction. ``None``
            keeps the historical behaviour -- the cloud box, honouring
            ``include_contacts`` -- so every existing call site is unaffected.
    """
    variant = variant or KeypointVariant("0_cloud_box", contacts=include_contacts)
    S, T, diagnostics = scene_keypoints(
        source_placement,
        target,
        source_labels,
        include_contacts=include_contacts or variant.contacts,
        box=variant.box,
        orientation=variant.orientation,
        cube_half_extent=variant.cube_half_extent,
    )
    transport_map = TransportMap().fit(S.points, T.points)
    warped = transport_map.transport_positions(source_labels.positions)
    warped_rotations = transport_map.transport_orientations(
        source_labels.positions, source_labels.orientations
    )
    report = transport_map.check_diffeomorphism(source_labels.positions)
    grasp_index, release_index = carry_indices(source_labels)

    wanted_pick = corresponding_point(
        source_labels.positions[grasp_index], S.metadata["src_pick"], T.metadata["tgt_pick"]
    )
    wanted_place = corresponding_point(
        source_labels.positions[release_index], S.metadata["src_place"], T.metadata["tgt_place"]
    )
    return {
        "source_keypoints": S,
        "target_keypoints": T,
        "diagnostics": diagnostics,
        "map": transport_map,
        "warped": warped,
        "grasp_index": grasp_index,
        "release_index": release_index,
        "n_keypoints": len(S),
        "keypoint_residual": float(transport_map.keypoint_residual().max()),
        "min_det": float(report.min_determinant),
        "fraction_positive": float(report.fraction_positive),
        # Does the warp put the hand where the target object's own grasp is?
        "grasp_error": float(np.linalg.norm(warped[grasp_index] - wanted_pick)),
        "release_error": float(np.linalg.norm(warped[release_index] - wanted_place)),
        # Does it arrive at the destination surface, or above / through it?
        "release_clearance": float(warped[release_index][2] - target.destination_height),
        "expected_clearance": float(wanted_place[2] - target.destination_height),
        "lateral_error": float(
            np.linalg.norm(warped[release_index][:2] - target.destination[:2])
        ),
        # How far the demonstration's own grasp sits from the one a generator
        # would pick for this object. The warp is not wrong to differ: it
        # reproduces the demonstrated grasp, and the generated grasp is a
        # separate choice that the frame contract, not the map, must serve.
        "grasp_recipe_gap": float(np.linalg.norm(wanted_pick - target.grasp.tcp)),
        # Can the transported hand actually close on the object? Eq. 11 carries
        # the orientation; if the jaws end up across the object's long axis the
        # position can still be millimetre-perfect and the grasp impossible.
        # Compared as an axis, since a parallel jaw is symmetric under a flip.
        "closing_error_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(abs(warped_rotations[grasp_index][:, 1] @ target.grasp.closing), 0, 1)
                )
            )
        ),
        "sparse_cloud": bool(len(target.points) < SPARSE_CLOUD_POINTS),
        "cloud_points": int(len(target.points)),
        # --- the variant, and what it makes measurable ----------------------
        "variant": variant.name,
        "box": variant.box,
        "orientation": variant.orientation,
        "cube_half_extent": (
            variant.cube_half_extent if variant.box == "grasp_cube" else None
        ),
        "contacts": bool(include_contacts or variant.contacts),
        # ``aim_map`` is ~0 *by construction* when this is True, because the
        # cube's centre is the target grasp TCP and phi interpolates keypoints
        # exactly. Recorded so the number is never read as evidence.
        "aim_is_pinned": variant.aim_is_pinned,
        # **The map's own promise**: does phi carry the source grasp's fingertip
        # point onto the target grasp's? Frame-independent, because both ends are
        # fingertip points.
        "aim_map": float(
            np.linalg.norm(
                transport_map.transport_positions(source_placement.grasp.tcp[None])[0]
                - target.grasp.tcp
            )
        ),
        # The same question asked of the *trajectory label* at the grasp index.
        # This carries the wrist-to-fingertip offset as well: the demonstration
        # is recorded at ``grip_site`` while a grasp TCP is at the fingertips, so
        # on this source the two are 41 mm apart (7.20). ``pipeline.run``
        # converts its labels to the tool frame before transporting and so does
        # not pay it; this driver does not, so read ``aim_map`` for the map and
        # this only as a reminder that the frames differ.
        "aim_label": float(np.linalg.norm(warped[grasp_index] - target.grasp.tcp)),
        "aim_path_min": float(
            point_to_curve_distance(target.grasp.tcp[None], warped)[0]
        ),
        # Eq. 11's own error: no keypoint pins a *derivative*, so this stays
        # informative for every variant.
        "orientation_error_deg": float(
            orientation_transport_error(
                transport_map,
                source_placement.grasp.tcp[None],
                source_placement.grasp.rotation,
                target.grasp.rotation,
            )[0]
        ),
        # Where the minimum determinant *is* matters as much as its value: a
        # fold at the grasp is worse than one in free space.
        "det_at_grasp": float(
            np.linalg.det(
                transport_map.jacobian(source_labels.positions[grasp_index][None])[0]
            )
        ),
        **{
            k: float(v) if not k.endswith("argmax") else int(v)
            for k, v in tilt_profile(
                transport_map,
                source_labels.positions,
                grasp_index=grasp_index,
                release_index=release_index,
            ).items()
        },
        "lift_deviation_deg": float(
            lift_deviation(
                transport_map,
                source_labels.positions,
                grasp_index,
                min(grasp_index + 12, len(source_labels.positions) - 1),
            )
        ),
    }


def summarise(label: str, result: dict) -> dict:
    """The row that goes into the report and the ablation figure."""
    return {
        "label": label,
        "n_keypoints": result["n_keypoints"],
        "keypoint_residual": result["keypoint_residual"],
        "min_det": result["min_det"],
        "fraction_positive": result["fraction_positive"],
        "grasp_error": result["grasp_error"],
        "release_error": result["release_error"],
        "release_clearance": result["release_clearance"],
        "expected_clearance": result["expected_clearance"],
        "lateral_error": result["lateral_error"],
        "grasp_recipe_gap": result["grasp_recipe_gap"],
        "closing_error_deg": result["closing_error_deg"],
        "sparse_cloud": result["sparse_cloud"],
        "cloud_points": result["cloud_points"],
        **{
            k: result[k]
            for k in (
                "variant", "box", "orientation", "cube_half_extent", "contacts",
                "aim_is_pinned", "aim_map", "aim_label", "aim_path_min",
                "orientation_error_deg", "det_at_grasp", "lift_deviation_deg",
                "tilt_max", "tilt_median", "tilt_mid_path", "tilt_argmax",
                "tilt_at_grasp", "tilt_at_release",
            )
            if k in result
        },
    }


def tilt_sweep(
    labels,
    source_placement,
    env,
    objects=ABLATION_OBJECTS,
    slot: str = "top_middle",
    variants=VARIANTS,
    tilts=TILT_OFFSETS,
    grasp_source: str = "recipe",
    gripper: str = "panda",
    obs=None,
) -> list[dict]:
    """Every construction against an approach tilted out of the support plane.

    This is the only axis on which the two corner orientations differ, so it is
    the only place variants 3 and 5 can be evaluated at all. Held fixed: one
    source demonstration, one scene, one slot, the target grasp at mid height.
    Varied: the construction, the object, and the approach tilt.

    With ``grasp_source="graspgen"`` the base grasp is a real ranked candidate
    rather than a top-down recipe, so the tilt is applied on top of whatever the
    planner already chose.
    """
    rows = []
    for name in objects:
        for tilt in tilts:
            try:
                target, _ = target_placement(
                    env, name, slot, height_fraction=0.5, obs=obs,
                    tilt_offset_deg=tilt, grasp_source=grasp_source, gripper=gripper,
                )
            except (ValueError, RuntimeError) as exc:
                rows.append({"label": f"{name} tilt={tilt:.0f}", "object": name,
                             "tilt_offset_deg": tilt, "grasp_source": grasp_source,
                             "skipped": str(exc)})
                continue
            for variant in variants:
                label = f"{variant.name} {name} tilt={tilt:.0f}"
                try:
                    row = summarise(
                        label,
                        transport(labels, source_placement, target, variant=variant),
                    )
                except Exception as exc:
                    row = {"label": label, "variant": variant.name,
                           "failed": f"{type(exc).__name__}: {exc}"}
                row.update(object=name, tilt_offset_deg=tilt, grasp_source=grasp_source)
                rows.append(row)
                print("  " + _variant_line(row))
    return rows


def variant_sweep(
    labels,
    source_placement,
    env,
    objects=ABLATION_OBJECTS,
    slot: str = "top_middle",
    variants=VARIANTS,
    cube_sizes=CUBE_SIZES,
    grasp_fractions=(1.0, 0.5),
    obs=None,
) -> dict:
    """Every construction against every object, scored geometrically.

    Held fixed: one source demonstration, one scene, one slot, and -- because
    ``target_placement`` derives the grasp from the cloud by recipe -- one grasp
    rule per cell. Varied: the keypoint construction, the object, and where along
    the object the target grasp sits.

    The grasp-height axis is the one that matters most: it is what makes the
    cloud box and the grasp cube disagree at all, since the cloud box is centred
    on the object's centroid and the cube on the grasp.

    Returns:
        ``{"variants": [...], "cube_sizes": [...]}`` of flat JSON-safe rows.
    """
    variant_rows, size_rows = [], []
    for name in objects:
        for fraction in grasp_fractions:
            try:
                target, _ = target_placement(
                    env, name, slot, height_fraction=fraction, obs=obs
                )
            except ValueError as exc:  # too little cloud to describe the object
                variant_rows.append(
                    {"label": f"{name} h={fraction:.2f}", "object": name,
                     "grasp_fraction": fraction, "skipped": str(exc)}
                )
                continue
            for variant in variants:
                try:
                    row = summarise(
                        f"{variant.name} {name} h={fraction:.2f}",
                        transport(labels, source_placement, target, variant=variant),
                    )
                except Exception as exc:  # a refused or degenerate construction
                    row = {"label": f"{variant.name} {name} h={fraction:.2f}",
                           "variant": variant.name, "failed": f"{type(exc).__name__}: {exc}"}
                row.update(object=name, grasp_fraction=fraction)
                variant_rows.append(row)
                print("  " + _variant_line(row))

            # The cube-size axis, on the task-frame cube only: size sets the
            # width of the stencil J_perp is estimated over, so the orientation
            # error is what it moves. Aim is exact at every size and cannot
            # discriminate.
            for half in cube_sizes:
                v = KeypointVariant(f"cube_{half * 1000:.0f}mm", box="grasp_cube",
                                    cube_half_extent=half)
                try:
                    row = summarise(
                        f"{v.name} {name} h={fraction:.2f}",
                        transport(labels, source_placement, target, variant=v),
                    )
                except Exception as exc:
                    row = {"label": f"{v.name} {name} h={fraction:.2f}",
                           "failed": f"{type(exc).__name__}: {exc}"}
                row.update(object=name, grasp_fraction=fraction, cube_half_extent=half)
                size_rows.append(row)
    return {"variants": variant_rows, "cube_sizes": size_rows}


def _variant_line(row: dict) -> str:
    if "failed" in row or "skipped" in row:
        return f"{row['label']:<34} {row.get('failed') or row.get('skipped')}"
    pinned = " (pinned)" if row.get("aim_is_pinned") else "         "
    return (
        f"{row['label']:<34} det {row['min_det']:7.3f}"
        f"{'  FOLD' if row['min_det'] <= 0 else '    ok'}"
        f"  aim {row['aim_map'] * 1000:6.1f}mm{pinned}"
        f"  ori {row['orientation_error_deg']:5.1f}d"
        f"  tilt@mid {row['tilt_mid_path']:5.1f}d"
        f"  res {row['keypoint_residual']:.0e}"
    )


def _line(row: dict) -> str:
    folded = "" if row["min_det"] > 0 else "  FOLDED"
    return (
        f"  {row['label']:<22} n={row['n_keypoints']:>2} "
        f"grasp {row['grasp_error'] * 1000:6.1f} mm  "
        f"release {row['release_error'] * 1000:6.1f} mm  "
        f"clearance {row['release_clearance'] * 1000:6.1f} mm "
        f"(want {row['expected_clearance'] * 1000:5.1f})  "
        f"gap {row['grasp_recipe_gap'] * 1000:5.1f} mm  "
        f"jaw {row['closing_error_deg']:4.1f} deg  "
        f"det>0 {row['fraction_positive']:5.0%}{folded}"
        + ("  SPARSE" if row.get("sparse_cloud") else "")
    )


def main(out_dir: str | Path = "outputs/keypoints", slot: str = "top_middle") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("recording the source demonstration (reshelving, seed 0)")
    labels, source_placement, demo_ok = record_source_placement(0)
    if not demo_ok:
        raise RuntimeError("the source demonstration failed; transporting it is meaningless")

    env = build_scene()
    try:
        obs = env._get_observations()
        report = {"slot": slot, "source": "reshelving seed 0", "objects": {}}

        # --- ablation 1: one grasp recipe, five different objects -----------
        print(f"\nsame grasp recipe, different objects -> {slot}")
        object_rows, first = [], None
        for name in ABLATION_OBJECTS:
            try:
                target, cloud = target_placement(env, name, slot, obs=obs)
            except ValueError as exc:  # too little cloud to describe the object
                print(f"  {name:<22} skipped: {exc}")
                object_rows.append({"label": name, "skipped": str(exc)})
                continue
            result = transport(labels, source_placement, target)
            row = summarise(name, result)
            row["box_half_extents"] = result["target_keypoints"].metadata["tgt_pick"][
                "box_half_extents"
            ]
            object_rows.append(row)
            print(_line(row))
            if first is None:
                first = (target, result)
        report["objects"] = object_rows

        # --- ablation 2: one object, the grasp's closing axis rotated -------
        print("\nsame object (cereal), grasp closing axis rotated")
        yaw_rows = []
        for offset in YAW_OFFSETS:
            target, _ = target_placement(env, "cereal", slot, yaw_offset_deg=offset, obs=obs)
            row = summarise(f"yaw +{offset:.0f} deg", transport(labels, source_placement, target))
            yaw_rows.append(row)
            print(_line(row))
        report["grasp_yaw"] = yaw_rows

        # --- ablation 3: the contact keypoints, on and off ------------------
        print("\ncontact keypoints on/off, grasp height varied (cereal)")
        contact_rows = []
        for fraction in (0.5, 0.75, 1.0):
            for contacts in (False, True):
                target, _ = target_placement(
                    env, "cereal", slot, height_fraction=fraction, obs=obs
                )
                row = summarise(
                    f"h={fraction:.2f} {'contacts' if contacts else 'box'}",
                    transport(labels, source_placement, target, include_contacts=contacts),
                )
                contact_rows.append(row)
                print(_line(row))
        report["contacts"] = contact_rows

        # --- the keypoint-construction sweep --------------------------------
        print("\nkeypoint constructions, every object, two grasp heights")
        sweep = variant_sweep(
            labels, source_placement, env, objects=ABLATION_OBJECTS, slot=slot, obs=obs
        )
        report["variants"] = sweep["variants"]
        report["cube_sizes"] = sweep["cube_sizes"]

        # The tilt axis, which is the only one that separates the two corner
        # orientations. Run twice: on the top-down recipe, where the tilt is the
        # only source of out-of-plane approach, and on real GraspGen-X
        # candidates, which carry their own.
        print("\napproach tilted out of plane, recipe grasps")
        report["tilt_recipe"] = tilt_sweep(
            labels, source_placement, env, objects=ABLATION_OBJECTS, slot=slot,
            obs=obs, grasp_source="recipe",
        )
        print("\napproach tilted out of plane, real GraspGen-X grasps")
        try:
            report["tilt_graspgen"] = tilt_sweep(
                labels, source_placement, env, objects=ABLATION_OBJECTS, slot=slot,
                obs=obs, grasp_source="graspgen",
            )
        except Exception as exc:  # the server is optional; say so rather than skip silently
            print(f"  GraspGen-X unavailable: {type(exc).__name__}: {exc}")
            report["tilt_graspgen"] = [{"unavailable": f"{type(exc).__name__}: {exc}"}]

        # --- figures --------------------------------------------------------
        target, result = first
        paths = [
            figure_keypoint_scene(
                result["source_keypoints"], result["target_keypoints"],
                result["diagnostics"], source_placement, target,
                out_dir / "fig_keypoints_3d.png",
                title=f"Reshelving box -> {ABLATION_OBJECTS[0]} on the {slot} shelf",
            ),
            figure_transported_trajectory(
                labels.positions, result["warped"],
                result["source_keypoints"], result["target_keypoints"],
                {"source": source_placement.destination_height,
                 "target": target.destination_height},
                out_dir / "fig_transport_3d.png",
                release_index=result["release_index"],
                title=f"One demonstration transported onto the {ABLATION_OBJECTS[0]}",
            ),
            figure_ablation(object_rows, out_dir / "fig_ablation_objects.png",
                            title="Same grasp recipe, different objects"),
            figure_ablation(yaw_rows, out_dir / "fig_ablation_grasp_yaw.png",
                            title="Same object, grasp closing axis rotated"),
            figure_ablation(contact_rows, out_dir / "fig_ablation_contacts.png",
                            title="Contact keypoints on/off, against grasp height"),
        ]
        # Rendered larger than the observation cameras: these are for reading,
        # not for unprojection, and 256 px is too small to see the shelf.
        import imageio.v2 as imageio

        for camera in ("workspace", "sideview", "birdview"):
            image = env.sim.render(width=640, height=640, camera_name=camera)
            path = out_dir / f"scene_{camera}.png"
            imageio.imwrite(path, np.ascontiguousarray(image[::-1]))
            paths.append(path)

        (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

        # A manifest, so the sweep appears on the index and names the commit
        # that produced it. Rows go under ``rows`` rather than ``runs``:
        # ``write_index`` renders the seven-stage pick-and-place funnel from
        # ``runs``, and a geometry sweep filed there would show an all-zero
        # funnel, which is worse than showing none.
        manifest = write_manifest(
            out_dir,
            title="Keypoint construction sweep",
            description=(
                "Six keypoint constructions transported onto every tabletop "
                "object at two grasp heights, scored geometrically -- no policy, "
                "no rollout. Varies where the box comes from (the object's own "
                "cloud, or a fixed cube on the grasp) and how its corners are "
                "oriented (task frame, or the full grasp pose), with and without "
                "the jaw contacts. Answers whether centring on the grasp removes "
                "the object-size volume scaling and makes the contacts "
                "affordable, and what it costs in transported orientation."
            ),
            settings={
                "varied": {
                    "variant": [v.name for v in VARIANTS],
                    "object": list(ABLATION_OBJECTS),
                    "grasp_fraction": [1.0, 0.5],
                    "tilt_offset_deg": list(TILT_OFFSETS),
                    "grasp_source": ["recipe", "graspgen"],
                    "cube_half_extent_m": list(CUBE_SIZES),
                },
                "fixed": {
                    "source": "reshelving seed 0, one demonstration",
                    "slot": slot,
                    "default_cube_half_extent_m": GRASP_CUBE_HALF_EXTENT,
                    "camera_size": 256,
                    "residual": "GaussianProcessRegressor (defaults)",
                    "scored": "geometry only -- no rollout, no policy",
                },
            },
            thresholds={
                "min_determinant": 0.2,
                "keypoint_residual_max": 1e-5,
                "sparse_cloud_points": SPARSE_CLOUD_POINTS,
                "note": (
                    "aim_map is ~0 by construction for the cube variants "
                    "(aim_is_pinned), so it is not evidence; read the "
                    "orientation error, the determinant and the clearance."
                ),
            },
            rows=(
                report["variants"] + report["cube_sizes"]
                + report["tilt_recipe"] + report["tilt_graspgen"]
            ),
            report="fig_variants.png",
        )
        (out_dir / "rows.json").write_text(
            json.dumps(
                report["variants"] + report["cube_sizes"]
                + report["tilt_recipe"] + report["tilt_graspgen"],
                indent=2, default=float,
            )
        )
        print("\nwrote:")
        for path in [*paths, out_dir / "report.json", manifest, out_dir / "rows.json"]:
            print(f"  {path}")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/keypoints")
    parser.add_argument("--slot", default="top_middle")
    args = parser.parse_args()
    main(args.out, args.slot)
