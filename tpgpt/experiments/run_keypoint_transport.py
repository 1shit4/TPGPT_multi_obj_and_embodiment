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
from tpgpt.grasp.grippers import GRIPPER_PAIRS
from tpgpt.perception.cameras import object_point_cloud
from tpgpt.reporting.html import write_manifest
from tpgpt.sim.keypoints import (
    CORNER_NAMES,
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
from tpgpt.transport.maps import TransportMap, fit_local_correction
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
    #: Fit the jaw contacts as a **second, local stage** composed on top of the
    #: box map, instead of throwing them into the same map. Determinants
    #: multiply under composition, so each stage is verifiable on its own, and
    #: the correction's length scale confines it to the grasp's neighbourhood.
    compose_contacts: bool = False
    #: Locality radius of that second stage, in metres.
    locality: float = 0.03

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


#: The constructions compared.
#:
#: Two are controls and two are the question. The cloud box is kept because it is
#: today's default and because every claim about the cube is a *contrast* with
#: it; the task-frame cube is kept because it isolates what the grasp pose adds.
#: The two that matter are the last two: the cube laid out in the **full grasp
#: pose**, which is the only construction that transports the grasp's approach
#: tilt at all, and that same cube with the jaw contacts added as a **composed
#: local stage** rather than thrown into the same map.
VARIANTS = (
    KeypointVariant("0_cloud_box"),
    KeypointVariant("1_cube_task", box="grasp_cube"),
    KeypointVariant("2_cube_grasp_pose", box="grasp_cube", orientation="grasp"),
    KeypointVariant(
        "3_cube_grasp_pose_composed",
        box="grasp_cube",
        orientation="grasp",
        contacts=True,
        compose_contacts=True,
    ),
)

#: Kept available but not swept: contacts in the *same* map. Measured as folding
#: the cloud box in 10 cells of 10, and as costing the cube up to 63.8 degrees of
#: transported orientation, which is what motivated composing them instead.
SAME_MAP_CONTACT_VARIANTS = (
    KeypointVariant("x_cloud_box_contacts", contacts=True),
    KeypointVariant("x_cube_task_contacts", box="grasp_cube", contacts=True),
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
    gripper: str = "panda",
):
    """A tabletop-shelf scene with depth and instance segmentation enabled.

    Args:
        controller_config: Composite controller config. Must be supplied **at
            construction** -- assigning one afterwards does nothing. Passing the
            position-control config from
            :func:`~tpgpt.sim.replay.make_position_controller_config` is what
            makes the scene replayable.
        gripper: Registry short name of the hand to **mount**, resolved to its
            robosuite class through :func:`resolve_pair`.

            This has to be set at construction too, and forgetting it is silent:
            a first version of the multi-hand replay hardcoded a Panda while
            planning grasps for a Robotiq, so the only thing that actually varied
            was the tool offset handed to the replay. Every number came out
            monotonic in that offset and read convincingly as a cross-hand
            result. It was not one.
    """
    from tpgpt.grasp.grippers import resolve_pair
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    env = TabletopShelf(
        robots="Panda",
        gripper_types=resolve_pair(gripper).robosuite,
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


@dataclass(frozen=True)
class PathCheck:
    """Everything needed to judge a candidate by the trajectory it produces.

    **What this is for.** Every stage of
    :func:`~tpgpt.grasp.filters.filter_grasps` judges a grasp *pose*, because
    until a map is fitted a candidate is nothing more than a pose. The thing the
    arm actually executes is a 200-waypoint trajectory, and there is a different
    one for every candidate: the keypoints are built from that candidate, the
    demonstration is warped through them, and the result is the plan. 7.38
    measured what judging the proxy costs -- ``by_reachability`` validates 13
    poses per candidate, and a candidate that passes those 13 goes on to collide
    at **79 to 162** of the other 187.

    Checking the real thing turns out to be cheap, because **no inverse
    kinematics is needed to know where the hand is**: the transported path *is*
    the hand's pose at every waypoint. Placing the gripper's own surface sample
    at each of them and measuring how far it lies inside the scene's solid
    geometry costs about a millisecond a waypoint.

    Attributes:
        labels: The source demonstration, already in the tool frame and the
            grasp convention -- exactly the object the driver transports. Not a
            copy of it: passing anything else here checks a path that will not
            be executed.
        source: The source :class:`~tpgpt.sim.keypoints.ObjectPlacement`.
        variant: Which keypoint construction to build the map with. The map, and
            therefore the path, differs between constructions, so this has to be
            the one being run.
        max_candidates: How far down the ranked list to look before giving up.
            **The whole set, by default.** A candidate that fails costs about
            0.3 s to reject, because the sweep stops at its first fault, so
            searching a hundred is seconds rather than the minutes an earlier
            design would have taken -- and stopping at 25 was measured to leave
            several cells with nothing admissible that a deeper search finds.
        check_kinematics: Run the warm-started IK pass as well. It is the
            expensive half and it runs **only** on candidates that already
            passed the clearance sweep, which is the ordering that makes the
            pair affordable.
        ik_stride: Take every n-th waypoint for the IK pass. Consecutive
            waypoints are about 5 mm apart, so a stride of 4 still tests the
            path every 2 cm and costs a quarter of the solves.
        min_reachable_fraction: How much of the path the arm must be able to
            hold before a candidate is accepted.

            **Low on purpose, and the default was corrected by measurement.**
            The obvious reading -- demand most of the path -- is wrong here,
            because ``reachable_fraction`` is a property of the *whole path*
            rather than of any pose: each solve is seeded from the previous
            one, so it answers "can the arm move between these poses" and it
            reads low on trajectories the arm executes perfectly well. Measured
            over the twenty cells of Experiment O run iii
            (``outputs/path_study_iii``), it does **not** separate outcomes at
            all -- the eleven that placed span 0.62 to 1.00 with a median of
            0.84, and the nine that failed span 0.56 to 1.00 with the same
            median of 0.84. A bar at 0.9 would have rejected **6 of the 11
            cells that worked**.
            The only one-sided cut in the data is at 0.60: it rejects 2 of the
            9 failures and none of the 11 successes. So this stage is set to
            catch a path the arm plainly cannot follow and to keep out of the
            way otherwise.
        max_inside_fraction: How much of the path may sit inside scene geometry.
            Defaults to
            :data:`~tpgpt.grasp.filters.PATH_PENETRATION_FRACTION`; see there
            for why the criterion is *sustained* penetration and not the deepest.
    """

    labels: object
    source: ObjectPlacement
    variant: "KeypointVariant"
    max_candidates: int = 100
    check_kinematics: bool = True
    ik_stride: int = 4
    min_reachable_fraction: float = 0.60
    max_inside_fraction: float | None = None


def _select_by_path(
    env,
    grasp_set,
    order,
    instance: str,
    slot: str,
    cloud,
    gripper: str,
    support: float,
    yaw_offset_deg: float,
    tilt_offset_deg: float,
    check: PathCheck,
    funnel_flags: dict,
) -> tuple[int, dict]:
    """The first ranked candidate whose transported path is clear and reachable.

    Walks the ranked list in order, and for each candidate:

    1. builds the keypoints and fits the map, exactly as the driver will;
    2. transports the demonstration through it, giving the hand's pose at every
       waypoint with no inverse kinematics involved;
    3. measures how far inside the scene's solid geometry the hand -- and, over
       the interval it is actually held, the object -- ends up
       (:func:`~tpgpt.grasp.filters.path_clearance`);
    4. only if that passes, solves IK down the path warm-started from the
       previous waypoint (:func:`~tpgpt.grasp.filters.path_kinematics`).

    **The ranking is not touched.** Candidates are considered in the order they
    were ranked and the first admissible one wins, so this rejects rather than
    re-scores. That keeps "which candidates are allowed" and "which allowed one
    is used" separate, which is the separation `FINDINGS.md` 8k was unreadable
    for want of.

    **It falls back rather than refusing**, like every stage of the funnel: if
    no candidate inside ``max_candidates`` is admissible, the top-ranked one is
    used and a flag says so. A cell that executes under that flag is reporting
    "the least bad plan, and here is what is wrong with it" rather than "no
    grasp exists", and the flag reaches the manifest.

    Returns:
        ``(chosen index, flags)``.
    """
    from tpgpt.grasp.filters import path_feasibility_observed
    from tpgpt.grasp.grippers import resolve_pair
    from tpgpt.perception.cameras import scene_point_cloud
    from tpgpt.perception.obstacles import robot_bodies, self_filtered
    from tpgpt.sim.keypoints import carry_indices

    pair = resolve_pair(gripper)
    # Built once each. Neither the scene nor the robot's own shape changes while
    # candidates are compared, and rebuilding them per candidate would be most
    # of the cost.
    #
    # **The scene is what the cameras saw, not what the simulator knows**, and
    # the robot is subtracted from it -- a depth image of a workspace contains
    # the arm, so without that the arm collides with its own reflection. The
    # true geometry is available and is deliberately not used here: a filter
    # that reads it is not one a real system could run.
    bodies = robot_bodies(env)
    scene = self_filtered(
        env, scene_point_cloud(env, exclude=(instance,), obs=None), bodies
    )
    grasp_index, release_index = carry_indices(check.labels)

    order = [int(i) for i in order][: max(int(check.max_candidates), 1)]
    examined = []
    for candidate in order:
        grasp = _apply_grasp_offsets(
            GraspFrame.from_grasp(grasp_set.grasps[candidate]),
            yaw_offset_deg,
            tilt_offset_deg,
        )
        placement = _placement_for(env, cloud, grasp, slot, support, funnel_flags)
        try:
            result = transport(
                check.labels, check.source, placement, variant=check.variant
            )
        except (ValueError, RuntimeError) as exc:
            # A map this candidate's keypoints cannot produce -- degenerate
            # keypoints, a collapsed box. Recorded, not swallowed: a candidate
            # that cannot be transported is exactly as unusable as one that
            # collides, and the reason belongs in the report.
            examined.append({"index": candidate, "rejected": f"{type(exc).__name__}: {exc}"})
            continue
        feasible = path_feasibility_observed(
            env, result["warped"], result["warped_rotations"], pair,
            scene, placement.points, grasp_index, release_index,
            bodies=bodies, stride=check.ik_stride,
            # Selection is decided by the first fault, so the rest of the sweep
            # is wasted here. Diagnosis wants the whole tally and asks for it.
            stop_early=True,
        )
        record = {
            "index": candidate,
            "violations": feasible["violations"],
            "first_violation": feasible["first_violation"],
            "reachable_fraction": round(feasible["reachable_fraction"], 3),
            "faults": {k: v for k, v in list(feasible["faults"].items())[:4]},
        }
        if feasible["violations"]:
            record["rejected"] = "collision"
            examined.append(record)
            continue
        if feasible["reachable_fraction"] < check.min_reachable_fraction:
            record["rejected"] = "kinematics"
            examined.append(record)
            continue
        examined.append(record)
        return candidate, {
            "path_check": {
                "chosen": candidate,
                "rank_examined": len(examined),
                "fell_back": False,
                "examined": examined,
            }
        }

    # **Nothing was admissible, so take the least bad and say so.** Falling back
    # to the *top-ranked* candidate would throw away everything the check just
    # learned; the candidate that got furthest before its first fault is the one
    # whose plan is wrong latest, and on a path that ends at a shelf that is the
    # one most likely to have done the useful part of the task first. The flag
    # reaches the manifest, so a cell that runs under it reads as "no admissible
    # grasp exists here" rather than as an ordinary result.
    ranked = sorted(
        (e for e in examined if "rejected" in e),
        key=lambda e: -(e.get("first_violation") if e.get("first_violation")
                        is not None else -1),
    )
    best = ranked[0]["index"] if ranked else order[0]
    return best, {
        "path_check": {
            "chosen": best,
            "rank_examined": len(examined),
            "fell_back": True,
            "fell_back_to": "the candidate that got furthest before its first fault",
            "examined": examined,
        },
        "path_check_fell_back": True,
    }


def object_centre_of_mass(env, instance: str):
    """World centre of mass of a named object, or ``None`` if it cannot be read.

    ``None`` rather than a plausible default, on the rule that cost this project
    a 41 mm frame error: a zero or a centroid substituted silently for a missing
    measurement is indistinguishable from a real one, and the filter that
    consumes it would then be measuring nothing while reporting a number.

    In simulation this is exact -- MuJoCo carries each body's inertial frame. A
    real system has no such thing and would have to substitute the fitted box's
    centre, which is biased by the cloud being one-sided; the filter that uses
    this says so.
    """
    try:
        return np.asarray(
            env.sim.data.xipos[env.object_body_ids[instance]], dtype=float
        )
    except (KeyError, AttributeError, IndexError):
        return None


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
    reference_approach: np.ndarray | None = None,
    filters: str = "approach",
    slot_for_filters: str | None = None,
    rank_by: str = "auto",
    approach_filter: bool = True,
    path_check: "PathCheck | None" = None,
    scene_zones: bool = True,
    centre_filter: bool = True,
    carry_rotation: np.ndarray | None = None,
) -> tuple[ObjectPlacement, np.ndarray]:
    """Describe an object in the scene and where the task wants it.

    Args:
        yaw_offset_deg: Rotate the grasp's closing axis away from the cloud's
            natural narrow direction. The keypoint box is built in a frame
            derived from the closing axis, so the closing *direction* changes the
            keypoints while the grasp's height does not.
        approach_filter: Whether the 45 degree approach test is applied, both
            inline and as the funnel's "demonstrated" stage. Independent of
            ``rank_by``, so the demonstration can rank without filtering or
            filter without ranking.

            Default ``True``, which is the historical behaviour. Switch it off
            to ask what the test is worth -- see the note at the test itself for
            why that is an open question rather than a settled one.

        rank_by: Which surviving candidate is executed. Independent of
            ``filters``, because "which candidates are allowed" and "which
            allowed one is used" are separate decisions.

            ``"demonstration"`` takes the smallest approach mismatch from the
            source; ``"score"`` takes GraspGen-X's highest confidence;
            ``"auto"`` (the default) keeps the historical pairing --
            demonstration for ``filters="approach"``, score for ``"full"`` --
            so every existing caller is unaffected.

            Vary this **alone** to ask what the ranking is worth, and hold it
            fixed to ask anything else. Experiment N moved it together with the
            filter set and could measure neither (`FINDINGS.md` 8k): the chosen
            candidate's approach mismatch went from a median 3.2 to 13.8 degrees
            and its TCP moved a median 15.7 mm, so 39 of 40 cells executed a
            different grasp.

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
        grasp_source: ``"graspgen"`` takes a real candidate from
            GraspGen-X, filtered by :data:`MAX_APPROACH_MISMATCH_DEG` against
            ``reference_approach` exactly as ``filter_grasps`` does, and picks
            the best-aligned. This is the realistic case and the only one that
            exercises a hand's actual approach.

            ``"recipe"`` (**the default, for backward compatibility only**)
            derives the grasp from the cloud with :func:`top_down_grasp`, which
            always approaches straight down. That makes the task frame and the
            full grasp pose **coincide**, so it cannot tell those two
            constructions apart -- useful as an internal control, useless as a
            measurement. It is also the only path that works on a cloud too thin
            for the planner, which is what
            ``tests/integration/test_keypoints.py`` relies on when it checks that
            a sparse cloud is visible as such.

            **Every experiment in this module passes ``"graspgen"`` explicitly.**
            The default is left on the recipe so that callers testing the
            *keypoint construction* rather than the grasp source keep working
            without a running server.
        reference_approach: The demonstration's own approach direction, from
            ``pipeline._demonstrated_approach(labels)``. Required for
            ``"graspgen"``.
        filters: Which of the grasp filters to apply to the planner's output.

            ``"approach"`` (default) applies only the
            :data:`MAX_APPROACH_MISMATCH_DEG` test against
            ``reference_approach``. Cheap, needs no scene cloud, and enough to
            stop the map being handed a candidate approached from underneath --
            but it is **one** of the seven stages the pipeline actually uses.

            ``"full"`` runs :func:`~tpgpt.grasp.filters.filter_grasps`: the
            approach test plus visibility, on-target, jaw width, collision
            against the rest of the scene, **reachability** at the pick, the
            lift, the placement and the approach corridors into both, and
            duplicate suppression. This is what the pipeline runs, and running
            anything less here means an experiment measures candidates the
            pipeline would have thrown away. Requires ``slot_for_filters`` so a
            placement pose exists to test reachability against.
        slot_for_filters: Destination slot, needed by ``filters="full"`` to
            derive the placement pose the reachability stage checks.
        carry_rotation: How the demonstration turns the hand between closing the
            jaws and opening them. Both place-side filter stages need it or they
            judge the wrong pose -- 35.5 degrees wrong on this demonstration, and
            orientation is what decides whether the hand fits the slot at all.
            Available from ``carry_transform(labels)``; ``path_check`` carries
            the labels, so a caller passing one can derive it.
        scene_zones: Apply the two scene-derived no-approach zones inside the
            full funnel -- :func:`~tpgpt.grasp.filters.by_support_approach` at
            the pick and :func:`~tpgpt.grasp.filters.by_place_approach` at the
            placement. Separately switchable from ``approach_filter`` because
            they are meant to *replace* it: one is a constraint read off the
            table and the shelf, the other a resemblance to a recording, and a
            run that moved both at once could not say which mattered.
        centre_filter: Apply the loose centre-of-mass stage
            (:func:`~tpgpt.grasp.filters.by_centre_offset`). Separate again, and
            for the same reason: it rests on twenty grasps from one study and
            two of them contradict it, so its contribution has to stay
            separable from the rest.
        path_check: Run the two stages that can only be run **after** the
            transport, on the candidates the funnel has already passed. See
            :class:`PathCheck`. ``None`` leaves selection exactly as it was, so
            every existing caller is unaffected.

            These cannot live in
            :func:`~tpgpt.grasp.filters.filter_grasps`, and the reason is not
            organisational. Every stage in that funnel judges a *pose*, or a
            short corridor around one, because a pose is all a candidate is
            until a map has been fitted to it. The trajectory the arm actually
            executes does not exist until the keypoints are built from that
            candidate and the demonstration is warped through them -- a
            different trajectory for every candidate. 7.38 measured the cost of
            judging the proxy instead: ``by_reachability`` validates 13 poses
            while the replay executes 200, so a candidate passes its sample and
            then collides at 79 to 162 of the other 187.
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
    #: Empty unless the full funnel runs; only it produces flags.
    funnel_flags: dict = {}
    if grasp_source == "recipe":
        grasp = top_down_grasp(cloud.points, height_fraction=height_fraction)
    elif grasp_source == "graspgen":
        from tpgpt.experiments.pipeline import _demonstrated_approach
        from tpgpt.grasp.filters import MAX_APPROACH_MISMATCH_DEG
        from tpgpt.grasp.pipeline import grasps_for_cloud

        grasp_set = grasps_for_cloud(cloud, gripper)
        if not len(grasp_set):
            raise RuntimeError(
                f"GraspGen-X returned no candidate for {instance!r} with "
                f"{gripper!r}; refusing to substitute a top-down recipe"
            )
        # The demonstration is used for two unrelated things here, and they are
        # kept apart deliberately. It **ranks** candidates whenever
        # ``rank_by="demonstration"``, and it **filters** them only when
        # ``approach_filter`` is set. Welding those together is what stopped the
        # approach test from being measurable at all: it could not be switched
        # off without also losing the ability to rank by it.
        #
        # A reference is still required, because the target's roll is chosen by
        # agreement with the demonstration regardless (`ROBOTICS_NOTES.md`
        # 7.33), so a caller without one has a deeper problem than this filter.
        if reference_approach is None:
            raise ValueError(
                "grasp_source='graspgen' needs a reference_approach; the "
                "target's roll is chosen by agreement with it even when "
                "approach_filter is off. Pass _demonstrated_approach(labels)"
            )
        reference = np.asarray(reference_approach, dtype=float).reshape(3)
        angles = np.degrees(
            np.arccos(
                np.clip([g.approach @ reference for g in grasp_set.grasps], -1.0, 1.0)
            )
        )
        if approach_filter:
            # Turning it off is a real experiment, not a convenience. The case
            # for the filter: with it off, the planner's top-scoring candidate
            # was measured 119-174 degrees from the demonstration's grasp
            # orientation, and a construction that transports orientation
            # faithfully then rotates the world by the same angle -- turning the
            # demonstrated lift into a descent.
            #
            # The case against, which is why it is now switchable: 45 degrees
            # has no measurement behind it; the quantity that actually folds the
            # map is the full source-to-target *frame* rotation, which includes
            # a roll this test cannot see; and the figure quoted for dropping it
            # (a valid map in 7 of 16 cells) comes from FINDINGS.md 8g, whose
            # numbers are marked superseded because it predates the frame fixes.
            # Filtering target grasps to resemble the source also narrows what a
            # transport result can claim.
            keep = np.flatnonzero(angles <= MAX_APPROACH_MISMATCH_DEG)
            if not len(keep):
                raise RuntimeError(
                    f"every one of {len(grasp_set)} candidates for {instance!r} is "
                    f"beyond the {MAX_APPROACH_MISMATCH_DEG:.0f} deg approach filter "
                    f"(closest {angles.min():.1f} deg); the pipeline would reject "
                    "this object before keypoints are built"
                )
        else:
            keep = np.arange(len(grasp_set.grasps))
        if filters == "full":
            # The whole funnel, exactly as ``pipeline.run`` applies it. Ordered
            # by the planner's own score, which is what ``filter_grasps``
            # returns, rather than by approach agreement -- the approach test is
            # now a pass/fail stage inside the funnel rather than the ranking.
            from tpgpt.experiments.pipeline import place_pose_for
            from tpgpt.grasp.filters import filter_grasps
            from tpgpt.perception.cameras import scene_point_cloud

            if slot_for_filters is None:
                raise ValueError(
                    "filters='full' needs slot_for_filters: the reachability "
                    "stage tests the placement pose, and there is no placement "
                    "without a slot"
                )
            scene = scene_point_cloud(env, exclude=(instance,), obs=obs)
            provisional = place_pose_for(
                env, slot_for_filters, grasp_set.grasps[0], gripper,
                float(np.ptp(cloud.points[:, 2])),
            )
            funnel = filter_grasps(
                grasp_set.grasps, gripper, cloud.points, scene_points=scene,
                camera_positions=cloud.camera_positions, env=env,
                place_pose=provisional,
                # Read off the scene, not off the demonstration: the table's
                # own outward normal at the pick, and the shelf's own panels at
                # the placement. ``None`` switches the stage off entirely.
                support_normal=(0.0, 0.0, 1.0) if scene_zones else None,
                check_place_approach=bool(scene_zones),
                centre_of_mass=object_centre_of_mass(env, instance)
                if centre_filter else None,
                carry_rotation=carry_rotation,
                # None disables the funnel's own "demonstrated" stage, which is
                # how the approach test is switched off inside the full funnel.
                # The reference is still held above for ranking and the roll.
                reference_approach=reference if approach_filter else None,
                target_name=instance,
            )
            if not len(funnel.survivors):
                raise RuntimeError(
                    f"every candidate for {instance!r} was rejected by the grasp "
                    f"filters, mostly by {funnel.rejected_by}"
                )
            survivors = np.asarray(funnel.survivors, dtype=int)
            default_rank = "score"
            # The funnel's flags are the only record of *why* a grasp was
            # chosen, and one of them names a perception failure rather than a
            # grasping one: a cloud too thin to verify the object fits the jaws.
            # Kept so a cell can be reported as a point-cloud fault.
            funnel_flags = dict(funnel.flags)
        elif filters == "approach":
            survivors = np.asarray(keep, dtype=int)
            default_rank = "demonstration"
        else:
            raise ValueError(
                f"unknown filters {filters!r}; expected 'approach' or 'full'"
            )

        # Which candidates survive and which survivor is executed are two
        # separate decisions, and they used to be welded together: asking for
        # the light filter forced "closest to the demonstration", asking for the
        # whole funnel forced "highest score". So the combination of light
        # filtering with the planner's score -- the run that isolates the
        # *ranking* -- could not be expressed at all, and any comparison between
        # the two settings moved both decisions at once. That is what made
        # Experiment N unreadable (`FINDINGS.md` 8k).
        chosen_rank = default_rank if rank_by == "auto" else rank_by
        if chosen_rank == "demonstration":
            # A map constraint, not a grasp criterion: the warp degrades with
            # the source-to-target frame rotation. Ranking on it says "the map
            # carries this one most comfortably", never "this is a better grasp".
            key = np.asarray(angles, dtype=float)[survivors]
        elif chosen_rank == "score":
            key = np.array([-grasp_set.grasps[int(i)].score for i in survivors])
        else:
            raise ValueError(
                f"unknown rank_by {rank_by!r}; expected 'auto', "
                "'demonstration' or 'score'"
            )
        # Stable, so ties keep the planner's own order rather than falling out
        # of quicksort arbitrarily -- the previous sort was unstable and its
        # comment claimed a secondary key it did not have.
        order = survivors[np.argsort(key, kind="stable")]
        funnel_flags["ranked_by"] = chosen_rank
        if path_check is not None:
            # The two stages that need the trajectory rather than the pose. They
            # walk the ranked list in order and take the first candidate whose
            # *transported path* is clear and reachable, so the ranking is
            # untouched and only inadmissible candidates are skipped.
            chosen, path_flags = _select_by_path(
                env, grasp_set, order, instance, slot, cloud, gripper, support,
                yaw_offset_deg, tilt_offset_deg, path_check, funnel_flags,
            )
            funnel_flags.update(path_flags)
        else:
            chosen = int(order[min(grasp_rank, len(order) - 1)])
        grasp = GraspFrame.from_grasp(grasp_set.grasps[chosen])
        # Which candidate was executed, and why. Recorded because a comparison
        # of selection rules is only informative on cells where the rules
        # actually disagree: if two settings happen to pick the same candidate,
        # that cell is the same run twice and carries no information about the
        # rule. `flip_target` was inert for the life of the project for want of
        # exactly this check (`ROBOTICS_NOTES.md` 7.34).
        funnel_flags["chosen_index"] = chosen
        funnel_flags["chosen_score"] = float(grasp_set.grasps[chosen].score)
        funnel_flags["chosen_approach_mismatch_deg"] = float(angles[chosen])
        funnel_flags["n_candidates"] = int(len(grasp_set.grasps))
        funnel_flags["n_survivors"] = int(len(survivors))
        # Where this grasp sits on the object, horizontally, relative to its
        # centre of mass. Recorded so "the grasp was off-centre" can be tested
        # rather than asserted -- and it has been asserted, on evidence that
        # came from a sample of failures only.
        try:
            com = object_centre_of_mass(env, instance)
            if com is None:
                raise KeyError(instance)
            tcp = np.asarray(grasp.tcp, dtype=float)
            funnel_flags["offset_mm"] = float(np.linalg.norm((tcp - com)[:2]) * 1000)
            funnel_flags["height_mm"] = float((tcp - com)[2] * 1000)
        except (KeyError, AttributeError) as exc:  # pragma: no cover
            # Narrow, and it says what it lost. A bare `except: pass` here would
            # have hidden the fact that the offset was missing at all, which is
            # how a study measures nothing and looks fine.
            funnel_flags["offset_mm_unavailable"] = repr(exc)
    else:
        raise ValueError(
            f"unknown grasp_source {grasp_source!r}; expected 'recipe' or 'graspgen'"
        )
    grasp = _apply_grasp_offsets(grasp, yaw_offset_deg, tilt_offset_deg)
    placement = _placement_for(env, cloud, grasp, slot, support, funnel_flags)
    return placement, cloud.points


def _apply_grasp_offsets(grasp, yaw_offset_deg: float, tilt_offset_deg: float):
    """The swept yaw and tilt offsets, applied in that order.

    Factored out so the grasp whose path is checked is bit for bit the grasp
    that is executed. Applying them in only one of the two places would mean
    checking a trajectory nobody runs, which is the whole failure this checking
    exists to prevent.
    """
    if yaw_offset_deg:
        rotation = rotate_about([0.0, 0.0, 1.0], np.radians(yaw_offset_deg))
        grasp = GraspFrame(grasp.tcp, grasp.approach, rotation @ grasp.closing)
    if tilt_offset_deg:
        # About the grasp's own closing axis, so the closing direction -- and
        # therefore the task frame -- is untouched and only the approach moves.
        rotation = rotate_about(grasp.closing, np.radians(tilt_offset_deg))
        grasp = GraspFrame(grasp.tcp, rotation @ grasp.approach, grasp.closing)
    return grasp


def _placement_for(env, cloud, grasp, slot: str, support: float, funnel_flags: dict):
    """Assemble the :class:`ObjectPlacement` for one grasp on one object."""
    destination = env.slot_poses()[slot]
    return ObjectPlacement(
        points=cloud.points,
        grasp=grasp,
        support_height=support,
        destination=destination,
        destination_height=float(destination[2]),
        metadata={
            "grasp_funnel_flags": funnel_flags,
            "cloud_points": int(len(cloud.points)),
        },
    )


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
    if variant.compose_contacts:
        # Two stages: the box carries the trajectory, the contacts are pinned by
        # a local correction on top. Determinants multiply, so each stage stays
        # separately verifiable and the correction's length scale keeps it off
        # the rest of the path -- neither of which a single map can offer.
        box_roles = {"center", *CORNER_NAMES}
        is_box = [lab.split("_", 1)[1] in box_roles for lab in S.labels]
        is_contact = [lab.split("_", 1)[1].startswith("grasp_") for lab in S.labels]
        stage1 = TransportMap().fit(S.points[is_box], T.points[is_box])
        transport_map = fit_local_correction(
            stage1,
            S.points[is_contact],
            T.points[is_contact],
            locality=variant.locality,
        )
    else:
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
        # The transported orientations, in the **grasp** convention -- the same
        # one the labels arrive in. Returned rather than left to be recomputed,
        # because every caller that wants the hand's pose at a waypoint needs
        # both halves and recomputing one of them is how two callers end up
        # measuring two different trajectories.
        "warped_rotations": warped_rotations,
        "grasp_index": grasp_index,
        "release_index": release_index,
        "n_keypoints": len(S),
        "keypoint_residual": float(transport_map.keypoint_residual().max()),
        "min_det": float(report.min_determinant),
        "fraction_positive": float(report.fraction_positive),
        # The rotation the warp has to realise between the source and target
        # task frames. **This, not the approach mismatch, is what folds the
        # map**: measured across 20 cells, ``min det`` runs 0.94-0.99 at 3-19
        # degrees of it and -0.06 to 0.16 at 149-179, folding past about 145.
        # A frame rotation includes the roll about the approach, which the 45
        # degree approach test cannot see -- so recording both separates "the
        # filter let a badly-rotated frame through" from "the filter was not the
        # thing protecting the map".
        "frame_rotation_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(
                            np.asarray(diagnostics["source_frame"], dtype=float).T
                            @ np.asarray(diagnostics["target_frame"], dtype=float)
                        ) - 1.0) / 2.0,
                        -1.0, 1.0,
                    )
                )
            )
        ),
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
        "composed": bool(variant.compose_contacts),
        "locality": variant.locality if variant.compose_contacts else None,
        **(
            {
                "min_det_stage1": float(
                    transport_map.stage_reports(source_labels.positions)[0].min_determinant
                ),
                "min_det_stage2": float(
                    transport_map.stage_reports(source_labels.positions)[1].min_determinant
                ),
            }
            if variant.compose_contacts
            else {}
        ),
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
        # Against the grasp **actually executed**, which is the planner's grasp
        # after the roll choice. Comparing against the emitted one instead reads
        # 179 degrees on the 13 of 20 cells that get rolled, and the jaw-symmetry
        # minimisation was what hid that -- for the life of the project. It is
        # now off by default, and against the executed grasp it changes nothing
        # on any cell, which is how it should be. 7.33.
        "orientation_error_deg": float(
            orientation_transport_error(
                transport_map,
                source_placement.grasp.tcp[None],
                source_placement.grasp.rotation,
                diagnostics["target_grasp"].rotation,
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
                "composed", "locality", "min_det_stage1", "min_det_stage2",
                "aim_is_pinned", "aim_map", "aim_label", "aim_path_min",
                "orientation_error_deg", "det_at_grasp", "lift_deviation_deg",
                "tilt_max", "tilt_median", "tilt_mid_path", "tilt_argmax",
                "tilt_at_grasp", "tilt_at_release",
            )
            if k in result
        },
    }


#: Hands for the cross-embodiment geometry sweep.
#:
#: All nine registered pairs, including the Inspire hand that lifts nothing.
#: Geometry costs milliseconds and a hand that cannot execute a grasp can still
#: say whether the *map* holds for it, so there is no reason to exclude any --
#: and the contrast between "the geometry is fine" and "the hand cannot do it"
#: is itself the answer to whether the keypoints are the limiting factor.
SWEEP_GRIPPERS = tuple(GRIPPER_PAIRS)


def gripper_sweep(
    labels,
    source_placement,
    objects=ABLATION_OBJECTS,
    slot: str = "top_middle",
    variants=VARIANTS,
    grippers=SWEEP_GRIPPERS,
    seed: int = 0,
    height_fraction: float = 0.5,
    reference_approach=None,
    controller_config: dict | None = None,
) -> list[dict]:
    """Every construction, every object, **every hand** -- scored geometrically.

    This is the cheap half of the cross-embodiment question. The keypoint cube is
    a *fixed* 20 mm half extent and encodes nothing about the hand: not the jaw
    aperture, not the fingertip depth, not the finger count. So if the grasp
    centre is a good enough representation, the map should come out equally well
    conditioned and equally well aimed for all nine hands, and the only things a
    hand changes are

    1. **which grasp GraspGen-X returns for it**, since the planner conditions on
       the hand's swept volume, and
    2. **the tool offset** the labels are expressed against, which spans 24 to
       134 mm across the registry.

    Both are inputs to the map rather than parameters of it, which is what makes
    the prediction falsifiable: a construction that quietly depended on the hand
    would show up as ``min_det`` or ``aim_map`` moving with the hand.

    A **scene is rebuilt per hand**, because the mounted gripper has to be set at
    construction (:func:`build_scene`) and because the same seed does not settle
    the objects identically with a different hand attached -- the can comes to
    rest 14.6 mm away between a Panda and a Robotiq 2F-140. That difference is
    real and is carried in the rows as ``object_position`` rather than hidden.

    Alongside the map metrics each row carries what the *hand* brings, so the
    "we never input jaw width" question can be read straight off the table:

    ``aperture_mm``
        The hand's published jaw opening, 50 to 125 mm across the registry.
    ``object_width_mm``
        The object's extent **along this grasp's own closing axis**, from the
        target keypoints. Not an axis-aligned extent: a 30 x 100 mm box yawed 45
        degrees measures 92 x 92 and reads as ungraspable (7.18).
    ``closing_budget_mm``
        ``(aperture - object width) / 2``, the room on each side once the jaws
        meet the object. This is the tolerance the closing-axis aim error has to
        fit inside, derived per hand and per object rather than assumed.
    ``tool_offset_mm``
        ``||contact_offset(hand)||``, the wrist-to-fingertip distance the labels
        are converted by.

    Grasps come from the **cache** (:mod:`tpgpt.grasp.cache`), keyed on the cloud
    and the hand, so a re-run compares the same candidate sets. GraspGen-X's
    planner is an unseeded diffusion model and without the cache this would be
    comparing random draws, not hands (7.15).

    Returns:
        Flat JSON-safe rows, one per (hand, object, construction), plus a
        ``skipped`` or ``failed`` row wherever a cell could not be built -- never
        a silently missing cell.
    """
    from tpgpt.experiments.diagnose import closing_budget
    from tpgpt.grasp.grasps import contact_offset
    from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

    rows: list[dict] = []
    for hand in grippers:
        pair = resolve_pair(hand)
        try:
            aperture = float(gripper_geometry(pair.graspgen).aperture)
        except Exception:  # pragma: no cover - absent sibling checkout
            aperture = float("nan")
        try:
            offset_mm = float(np.linalg.norm(contact_offset(hand)) * 1000)
        except Exception:
            offset_mm = float("nan")
        env = build_scene(
            objects=objects, seed=seed, controller_config=controller_config,
            gripper=hand,
        )
        try:
            obs = env._get_observations()
            for name in objects:
                position = np.asarray(env.object_position(name), dtype=float)
                try:
                    target, _ = target_placement(
                        env, name, slot, height_fraction=height_fraction, obs=obs,
                        grasp_source="graspgen",
                        reference_approach=reference_approach, gripper=hand,
                    )
                # Both, and deliberately. ``ValueError`` is "too little cloud
                # to describe the object"; ``RuntimeError`` is "every candidate
                # is beyond the approach filter", which is what the milk does --
                # its closest of 100 is 49.8 deg against a 45 deg limit. Both
                # are the pipeline legitimately refusing an object, and both
                # have to be recorded as a skipped cell rather than aborting the
                # sweep: eight hands would otherwise be lost to one object.
                except (ValueError, RuntimeError) as exc:
                    rows.append({
                        "label": f"{hand} {name}", "gripper": hand,
                        "object": name, "skipped": str(exc),
                    })
                    print(f"  {hand:<11}{name:<8} skipped: {str(exc)[:70]}", flush=True)
                    continue
                for variant in variants:
                    label = f"{variant.name} {hand} {name}"
                    try:
                        result = transport(
                            labels, source_placement, target, variant=variant
                        )
                        row = summarise(label, result)
                        # ``.points``: ``target_keypoints`` is a KeypointSet,
                        # and ``np.asarray`` on one yields an object array whose
                        # float() raises. Measured along the grasp's **own**
                        # closing axis, not an axis-aligned extent -- a 30 x 100
                        # mm box yawed 45 degrees measures 92 x 92 and reads as
                        # ungraspable (7.18).
                        keypoints = result.get("target_keypoints")
                        points = getattr(keypoints, "points", None)
                        # ``roles``, not ``labels``: ``labels`` is this
                        # function's own parameter -- the source demonstration --
                        # and binding it here shadowed it for every later cell.
                        # The first cell scored and the other 139 raised
                        # ``'list' object has no attribute 'orientations'``.
                        roles = getattr(keypoints, "labels", None)
                        # The **pick** block only. The set also holds the placed
                        # block, and the extent over both is the pick-to-place
                        # distance -- 239 to 277 mm, against an aperture of at
                        # most 125 mm, which clamps every budget to zero.
                        if points is not None and roles is not None:
                            pick = [
                                i for i, role in enumerate(roles)
                                if role.startswith("pick_")
                            ]
                            if len(pick) >= 2:
                                points = np.asarray(points, dtype=float)[pick]
                        width = (
                            float(np.ptp(
                                np.asarray(points, dtype=float) @ target.grasp.closing
                            ))
                            if points is not None and len(points) >= 2
                            else float("nan")
                        )
                        row["object_width_mm"] = width * 1000
                        row["closing_budget_mm"] = (
                            max(0.5 * (aperture - width), 0.0) * 1000
                            if np.isfinite(width) and np.isfinite(aperture)
                            else float("nan")
                        )
                    except Exception as exc:
                        row = {"label": label, "variant": variant.name,
                               "failed": f"{type(exc).__name__}: {exc}"}
                    row.update(
                        gripper=hand,
                        object=name,
                        robosuite=pair.robosuite,
                        graspgen=pair.graspgen,
                        family=pair.robosuite,
                        aperture_mm=aperture * 1000,
                        tool_offset_mm=offset_mm,
                        object_position=position.tolist(),
                    )
                    rows.append(row)
                    print("  " + _gripper_line(row), flush=True)
        finally:
            env.close()
    return rows


def _gripper_line(row: dict) -> str:
    if "failed" in row or "skipped" in row:
        note = row.get("failed") or row.get("skipped")
        return (f"{row.get('variant', '-'):27}{row.get('gripper', '?'):11}"
                f"{row.get('object', '?'):8} {note[:52]}")
    g = lambda k, d=float("nan"): row.get(k, d)
    return (
        f"{row['variant']:27}{row['gripper']:11}{row['object']:8}"
        f"{g('min_det'):9.3f}{g('aim_map') * 1000:9.1f}"
        f"{g('orientation_error_deg'):8.1f}{g('tilt_mid_path'):8.1f}"
        f"{g('keypoint_residual') * 1000:9.4f}"
        f"{g('aperture_mm'):8.0f}{g('object_width_mm'):8.1f}"
        f"{g('closing_budget_mm'):9.1f}{g('tool_offset_mm'):8.1f}"
    )


GRIPPER_HEADER = (
    f"{'variant':27}{'hand':11}{'object':8}{'minDet':>9}{'aimmm':>9}"
    f"{'orient':>8}{'tiltMid':>8}{'residmm':>9}{'apermm':>8}{'widthmm':>8}"
    f"{'budgetmm':>9}{'tcpmm':>8}"
)


def tilt_sweep(
    labels,
    source_placement,
    env,
    objects=ABLATION_OBJECTS,
    slot: str = "top_middle",
    variants=VARIANTS,
    tilts=TILT_OFFSETS,
    grasp_source: str = "graspgen",
    gripper: str = "panda",
    obs=None,
    reference_approach=None,
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
                    reference_approach=reference_approach,
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
    reference_approach=None,
    grasp_source: str = "graspgen",
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
                    env, name, slot, height_fraction=fraction, obs=obs,
                    grasp_source=grasp_source, reference_approach=reference_approach,
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


def main_grippers(
    out_dir: str | Path = "outputs/keypoints_grippers",
    slot: str = "top_middle",
    seed: int = 0,
    grippers=SWEEP_GRIPPERS,
    objects=ABLATION_OBJECTS,
) -> dict:
    """The cross-embodiment geometry sweep, on its own.

    A separate entry point rather than another block inside :func:`main`,
    because it rebuilds a scene per hand -- nine scenes against that campaign's
    one -- and because the question it answers is independent: *does the
    construction hold across hands*, not *which construction*. Keeping it apart
    means neither run's cost is charged to the other's question.

    Reported for every (hand, object, construction): the map's validity, its
    aim, its transported orientation, and -- so the "we never input jaw width"
    question is answerable -- the hand's published aperture, the object's width
    along that grasp's own closing axis, and the closing budget those imply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("recording the source demonstration (reshelving, seed 0)", flush=True)
    labels, source_placement, demo_ok = record_source_placement(0)
    if not demo_ok:
        raise RuntimeError(
            "the source demonstration failed; transporting it is meaningless"
        )

    # The labels are the *fingertip* path the keypoints are anchored to, not the
    # wrist path the demonstration was recorded at. Converting is not optional:
    # a map is exact only at its keypoints, and the grasp cube's centre *is* the
    # fingertip point, so warping the wrist path spends that guarantee on the
    # wrong point and the offset arrives undiminished (7.20).
    from tpgpt.experiments.pipeline import (
        SOURCE_GRIPPER,
        _demonstrated_approach,
        _to_tool_frame,
    )
    from tpgpt.grasp.grasps import contact_offset

    labels = _to_tool_frame(labels, contact_offset(SOURCE_GRIPPER))
    reference_approach = _demonstrated_approach(labels)

    print(f"\nnine hands, {len(objects)} objects, {len(VARIANTS)} constructions")
    print(GRIPPER_HEADER)
    print("-" * len(GRIPPER_HEADER), flush=True)
    rows = gripper_sweep(
        labels, source_placement, objects=objects, slot=slot,
        grippers=grippers, seed=seed, reference_approach=reference_approach,
    )

    manifest = write_manifest(
        out_dir,
        title="Does one keypoint construction hold across nine hands?",
        description=(
            "Every keypoint construction transported onto every tabletop "
            "object, for every registered gripper, scored geometrically -- no "
            "policy, no rollout. The grasp cube is a fixed 20 mm half extent "
            "and encodes nothing about the hand, so a construction that holds "
            "should give the same map validity and aim for all nine. The only "
            "things a hand changes are which grasp GraspGen-X returns for it "
            "and the tool offset the labels are expressed against, both of "
            "which are inputs to the map rather than parameters of it. Each "
            "row also carries the hand's aperture, the object's width along "
            "that grasp's closing axis, and the closing budget those imply, "
            "because no gripper dimension is ever fed to the keypoints."
        ),
        settings={
            "varied": {
                "gripper": list(grippers),
                "object": list(objects),
                "variant": [v.name for v in VARIANTS],
            },
            "fixed": {
                "source": "reshelving seed 0, one demonstration",
                "slot": slot,
                "seed": seed,
                "grasp_height_fraction": 0.5,
                "grasp_source": (
                    "graspgen, from tpgpt.grasp.cache -- the planner is an "
                    "unseeded diffusion model, so without the cache this would "
                    "compare random draws rather than hands (7.15)"
                ),
                "labels": "tool frame, contact_offset(source hand)",
                "cube_half_extent_m": GRASP_CUBE_HALF_EXTENT,
                "scene": (
                    "rebuilt per hand -- the mounted gripper must be set at "
                    "construction, and the same seed settles the can 14.6 mm "
                    "differently with a different hand attached, so "
                    "object_position is carried per row rather than assumed "
                    "equal"
                ),
                "scored": "geometry only -- no rollout, no policy",
            },
        },
        thresholds={
            "min_determinant": 0.2,
            "keypoint_residual_max": 1e-5,
            "note": (
                "aim_map is ~0 by construction for the cube variants "
                "(aim_is_pinned) and is not evidence for them; what is "
                "evidence across hands is whether min_det, the orientation "
                "error and the closing budget move with the hand."
            ),
        },
        rows=rows,
    )
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote:\n  {manifest}\n  {out_dir / 'rows.json'}")
    return {"rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--slot", default="top_middle")
    parser.add_argument(
        "--sweep", choices=("constructions", "grippers"), default="constructions",
        help=(
            "'constructions' is the original campaign: six constructions, five "
            "objects, two grasp heights, the tilt axis and the cube-size axis, "
            "on one hand. 'grippers' is the cross-embodiment sweep: every "
            "construction against every object for all nine hands."
        ),
    )
    parser.add_argument(
        "--grippers", default=None,
        help="Comma-separated registry short names; defaults to all nine.",
    )
    args = parser.parse_args()
    if args.sweep == "grippers":
        main_grippers(
            args.out or "outputs/keypoints_grippers",
            args.slot,
            grippers=tuple(args.grippers.split(",")) if args.grippers
            else SWEEP_GRIPPERS,
        )
    else:
        main(args.out or "outputs/keypoints", args.slot)
