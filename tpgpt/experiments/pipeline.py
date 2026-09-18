"""The whole thing, end to end: a sentence in, a robot arm out.

::

    prompt -> object + shelf -> point clouds -> grasp candidates -> filters
           -> keypoints -> transportation map -> policy -> execute -> score

Every stage can fail, and a run is only useful if it says *which* one did. A
placement that misses by 30 cm because the prompt was misread, because no grasp
survived the collision filter, because the arm could not reach the shelf, or
because the transported policy stalled are four different problems with four
different fixes, and they are indistinguishable from the placement error alone.
So each stage records what it did and the outcome carries a single named reason.

The source demonstration is the **reshelving** one: a 5 x 5 x 9 cm box moved
from a table to a shelf, recorded once and validated at 17/20. Nothing is
re-taught for the new objects, which is the whole claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from tpgpt.experiments.diagnose import diagnose, object_probe
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.experiments.run_keypoint_transport import object_centre_of_mass
from tpgpt.grasp.filters import FilterFunnel, filter_grasps
from tpgpt.grasp.grasps import (
    Grasp6D,
    contact_offset,
    grasp_to_eef_pose,
    to_grasp_convention,
    to_wrist_convention,
)
from tpgpt.grasp.grippers import resolve_pair
from tpgpt.language.parser import parse_task
from tpgpt.perception.cameras import object_point_cloud, scene_point_cloud
from tpgpt.metrics.transport import orientation_transport_error
from tpgpt.perception.scene_graph import build_scene_graph
from tpgpt.reporting.record import Timings, fingerprint, pose_matrix
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.sim.keypoints import (
    GRASP_CUBE_HALF_EXTENT,
    GraspFrame,
    ObjectPlacement,
    carry_indices,
    carry_transform,
    scene_keypoints,
)
from tpgpt.sim.rollout import rollout_policy, slot_score
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap

#: Why a run ended. Exactly one of these is reported.
OUTCOMES = (
    "success",
    "prompt_not_understood",
    "object_not_seen",
    "no_grasp_generated",
    "no_grasp_survived",
    "keypoints_degenerate",
    "map_not_a_diffeomorphism",
    "placed_in_the_wrong_place",
    "arm_could_not_hold_the_pose",
    "policy_stalled",
)

#: Cameras used for perception; see ``tpgpt.perception.cameras.DEFAULT_CAMERAS``.
SCENE_CAMERAS = ("workspace", "sideview", "birdview")

#: Below this many points an object is too thinly seen to describe.
MIN_CLOUD_POINTS = 40

#: Keypoint families used by default.
#:
#: **The box alone**, measured across 42 end-to-end runs -- all seven
#: combinations, two objects, three seeds each:
#:
#: ========================  =======  ===========  ==========
#: keypoints                 success  min det(J)   map folded
#: ========================  =======  ===========  ==========
#: box                         4/6    0.53 - 0.79     0/6
#: box + support               3/6    0.55 - 0.88     0/6
#: contacts + support          2/6    0.27 - 0.52     0/6
#: contacts                    0/6    0.18 - 0.70     0/6
#: support                     0/6    degenerate      -
#: box + contacts              0/6    **0.0074**      5/6
#: box + contacts + support    0/6    **0.0066**      5/6
#: ========================  =======  ===========  ==========
#:
#: **Adding the jaw contacts to the box collapses the map.** ``min det(J)``
#: falls a hundredfold, to a value identical to four decimal places across
#: runs -- a deterministic geometric consequence, not noise -- and five of six
#: maps are rejected outright as non-diffeomorphic.
#:
#: The reason is redundancy. The contacts sit *inside* the box's own convex
#: hull, centimetres from corners that already pin the warp, and a map that must
#: interpolate both exactly has to bend sharply between them. That is where the
#: determinant goes to zero.
#:
#: The same redundancy makes the dedicated support point unnecessary:
#: :func:`~tpgpt.sim.keypoints.fit_aligned_box` **snaps the box's lower face to
#: the support plane**, so its four bottom corners already *are* contact points
#: on the surface the object rests on. The guarantee that the object meets the
#: shelf rather than being dropped onto it or driven into it is provided by the
#: box, four times over.
#:
#: 4/6 against 3/6 is one run and not separable on this sample; the median
#: final placement error over *all* runs is the clearer signal, 16 mm for the
#: box against 103 mm for box + support.
#:
#: This restores the conclusion of ``ROBOTICS_NOTES`` section 6.4, which was
#: overturned on two pieces of bad evidence: a *geometric* pose-alignment
#: measurement that never executed anything, and end-to-end runs which -- as
#: section 7.12 records -- all ran with the gripper shut.
DEFAULT_KEYPOINTS = ("box",)

#: The hand that recorded the source demonstration, whose contact offset defines
#: the frame the demonstration's own keypoints were anchored in.
SOURCE_GRIPPER = "panda"

#: Fraction of a transported path whose poses the arm must be able to hold.
#:
#: A candidate can be collision-free and still be unreachable for a stretch of
#: its path. 0.60 is the replay driver's value, kept so the two selection paths
#: agree rather than being tuned separately here.
MIN_REACHABLE_FRACTION = 0.60

#: Surviving grasps whose transported motion is scored before choosing one.
#:
#: Each costs a map fit and fourteen IK solves, about a tenth of a second, and
#: the search stops early on the first candidate whose whole path is executable.
MAX_CANDIDATES = 8

#: The keypoint construction. ``"cloud"``/``"task"`` is the cloud box that this
#: pipeline shipped with; ``"grasp_cube"``/``"grasp"`` is Experiment R's winner
#: (``FINDINGS.md`` §8p), which placed 15/20 against the cloud box's 10/20 under
#: replay and carries the grasp point exactly rather than to 26-70 mm.
#:
#: The cube is a fixed-size box centred on the grasp TCP, so it pins the map
#: where the jaws close instead of at the object's centroid, and takes the
#: hand's own rotation rather than a task frame that can only carry a yaw.
DEFAULT_KEYPOINT_BOX = "grasp_cube"
DEFAULT_KEYPOINT_ORIENTATION = "grasp"


@dataclass
class RunResult:
    """Everything one end-to-end run produced, for the report and the index."""

    prompt: str
    gripper: str
    shelf_variant: str
    seed: int
    outcome: str = "success"
    detail: str = ""
    assumptions: list[str] = field(default_factory=list)
    object_name: str | None = None
    slot: str | None = None
    metrics: dict = field(default_factory=dict)
    funnel: FilterFunnel | None = None
    grasp: Grasp6D | None = None
    alternatives: list = field(default_factory=list)
    source_keypoints: object = None
    target_keypoints: object = None
    transport_map: object = None
    demonstration: np.ndarray | None = None
    transported: np.ndarray | None = None
    #: The full transported label set. ``transported`` is only its
    #: positions, which is enough to plot but not to re-execute.
    transported_labels: object = None
    rollout: object = None
    diagnosis: object = None
    source_capture: dict | None = None
    seconds: float = 0.0
    #: ``phi(x) - x`` on the common grid, float32. See
    #: :mod:`tpgpt.reporting.record`. Held on the result rather than written
    #: into ``metrics`` because it is 96 kB and ``metrics`` goes into JSON.
    map_fingerprint: np.ndarray | None = None
    #: Wall-clock cost of each stage of the method, in seconds.
    timings: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    #: Outcomes in which the **construction declined to build a map**, rather
    #: than a map being built and the plan failing.
    #:
    #: The distinction is not bookkeeping. If no grasp exists for this hand on
    #: this object, there is no anchor, so no keypoints, so no map -- the cell is
    #: outside the method's domain. Counting it as a failure understates the
    #: method and misattributes the cause, and counting it as a success is
    #: obviously worse; it belongs in its own column. Before this existed a
    #: refused cell simply vanished into the failure count, silently shrinking
    #: the denominator of every rate reported.
    REFUSALS = (
        "no_object_cloud",
        "no_grasps_generated",
        "no_grasp_survived",
        "keypoints_degenerate",
        "map_not_a_diffeomorphism",
    )

    @property
    def refused(self) -> bool:
        return self.outcome in self.REFUSALS

    @property
    def refusal_reason(self) -> str:
        return self.detail if self.refused else ""

    def summary(self) -> str:
        error = self.metrics.get("placement_error_xy")
        placed = f", off by {error * 1000:.0f} mm" if error is not None else ""
        stage = f"  [{self.diagnosis.blame}]" if self.diagnosis is not None else ""
        return (
            f"{'OK  ' if self.success else 'FAIL'} {self.gripper:<11}"
            f"{str(self.object_name):<8}-> {str(self.slot):<14}"
            f"{self.outcome}{placed}{stage}"
        )


def _fail(result: RunResult, outcome: str, detail: str) -> RunResult:
    result.outcome, result.detail = outcome, detail
    return result


#: Scene object order, matching ``run_keypoint_replay.REPLAY_OBJECTS``.
#:
#: **This is not cosmetic.** The placement sampler lays objects out in the order
#: it is given, so a different order is a different scene -- different positions,
#: different occlusions, different clouds. Running the campaign in one order and
#: the replay in another compares two scenes rather than two executors.
SCENE_OBJECTS = ("cereal", "milk", "can", "bread")


def build_scene(
    objects=SCENE_OBJECTS,
    gripper: str = "panda",
    shelf_variant: str = "cubby",
    seed: int = 0,
    camera_size: int = 256,
    controller_config: dict | None = None,
    world: str = "benchmark",
    pick_config: str | None = None,
):
    """A tabletop-shelf scene with depth, segmentation and torque control.

    Args:
        world: ``"benchmark"`` for robosuite's shipped object and shelf sizes,
            which every campaign before 2026-09-15 used, or ``"real"`` for the
            objects and furniture at the size of the real articles. See
            :mod:`tpgpt.sim.objects`.
        pick_config: A key of
            :data:`~tpgpt.sim.scenes.tabletop_shelf.REAL_PICK_CONFIGS`. Places
            the scene's single object at a fixed written-down pose instead of
            sampling one, which is what makes a cell's condition identical
            across grippers and objects.
        controller_config: Override the Cartesian-impedance torque controller,
            e.g. with
            :func:`~tpgpt.sim.replay.make_position_controller_config`. The
            controller has to be chosen at construction; assigning it to a built
            environment does nothing, because robosuite instantiates the part
            controllers during ``__init__``.
    """
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.controllers.cartesian_impedance import make_torque_controller_config
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    pair = resolve_pair(gripper)
    config = controller_config or make_torque_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    env = TabletopShelf(
        robots="Panda",
        gripper_types=pair.robosuite,
        controller_configs=config,
        objects=objects,
        shelf_variant=shelf_variant,
        world=world,
        pick_config=pick_config,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(SCENE_CAMERAS),
        camera_heights=camera_size,
        camera_widths=camera_size,
        camera_depths=True,
        camera_segmentations="instance",
        control_freq=20,
        seed=seed,
    )
    env.reset()
    return env


def place_pose_for(env, slot: str, grasp: Grasp6D, gripper: str, object_height: float):
    """Where the hand must be to leave the object standing in ``slot``.

    Derived from the grasp, not chosen: the hand holds the object at a fixed
    offset from the moment the jaws close, so where the hand has to end up is
    determined by where the object has to end up.
    """
    target = env.slot_poses()[slot]
    position, rotation = grasp_to_eef_pose(grasp, gripper)
    # The hand sits this far above the object's base at the pick.
    return np.array([target[0], target[1], target[2] + object_height]), rotation


def run(
    prompt: str,
    gripper: str = "panda",
    shelf_variant: str = "cubby",
    seed: int = 0,
    objects=SCENE_OBJECTS,
    source: tuple | None = None,
    include_contacts: bool = False,
    keypoint_parts: tuple[str, ...] = DEFAULT_KEYPOINTS,
    keypoint_box: str = DEFAULT_KEYPOINT_BOX,
    keypoint_orientation: str = DEFAULT_KEYPOINT_ORIENTATION,
    cube_half_extent: float = GRASP_CUBE_HALF_EXTENT,
    approach_filter: bool = False,
    max_candidates: int = 100,
    preconditions: bool = True,
    max_steps: int = 600,
    env=None,
    rollout_kwargs: dict | None = None,
    executor: str = "policy",
) -> RunResult:
    """One end-to-end attempt.

    Args:
        source: ``(labels, placement)`` from :func:`record_source_placement`.
            Recorded once and reused across a campaign; it costs a full
            demonstration otherwise.
        keypoint_parts: Which keypoint families to use. See
            :data:`DEFAULT_KEYPOINTS`; all seven combinations are meaningful and
            two of them are degenerate.
        rollout_kwargs: Extra arguments for
            :func:`~tpgpt.sim.rollout.rollout_policy`, for sweeping a control
            parameter without editing its default.
        executor: ``"policy"`` fits the GP policy and integrates an attractor
            through the Cartesian impedance controller, which is what ships.
            ``"replay"`` drives the arm onto each transported waypoint under
            position control -- no policy, no attractor, no lag gate -- which is
            the **executor-free ceiling**, and the comparison Experiment I is.

            **Both arms of that comparison must come through this one function.**
            Running them through two drivers is what invalidated the previous
            attempt: the two select grasps through structurally different code,
            so identical settings bought nothing and the same 20-cell grid
            produced *different grasps in 10 cells* from an identical cloud and
            an identical 100-candidate set. The totals read as a clean one-cell
            difference and the truth was six cells disagreeing in both
            directions at Fisher p = 1.000 (``ROBOTICS_NOTES`` 7.41). Here
            everything up to the moment of execution is the same code on the
            same inputs, so the grasp is identical by construction rather than
            by agreement -- and ``grasp_chosen_index`` is recorded so it can be
            checked rather than assumed.

            The environment must be built with a matching controller:
            :func:`~tpgpt.sim.replay.make_position_controller_config` for
            ``"replay"``. robosuite instantiates its part controllers during
            ``__init__``, so this cannot be chosen after the fact.
    """
    if executor not in ("policy", "replay"):
        raise ValueError(
            f"unknown executor {executor!r}; expected 'policy' or 'replay'")
    started = time.time()
    result = RunResult(prompt=prompt, gripper=gripper, shelf_variant=shelf_variant, seed=seed)
    owned = env is None
    env = env or build_scene(objects, gripper, shelf_variant, seed)

    try:
        if source is None:
            labels, placement, ok = record_source_placement(0)
            if not ok:
                return _fail(result, "policy_stalled", "the source demonstration failed")
        else:
            labels, placement = source

        # --- the prompt ----------------------------------------------------
        graph = build_scene_graph(env)
        spec = parse_task(prompt, graph)
        if not spec.ok:
            return _fail(result, "prompt_not_understood", spec.error or "")
        result.assumptions = list(spec.assumptions)
        entity = graph.by_id(spec.object_id)
        result.object_name = entity.label
        result.slot = graph.by_id(spec.destination_id).metadata["slot"]

        # **Before any physics, and after the object is known.** Each check here
        # corresponds to a bug that was found only once a campaign had already
        # produced plausible numbers: the wrong hand mounted, a scene reused
        # from the previous cell, an object that had drifted since reset, an
        # uncalibrated jaw. All four are answerable in milliseconds from a fresh
        # environment, so the run refuses to start rather than discovering it in
        # the results. ``FINDINGS.md`` §8p.
        if preconditions:
            from tpgpt.experiments.diagnose import replay_preconditions, require

            require(replay_preconditions(env, gripper, result.object_name),
                    f"{gripper}/{result.object_name}")

        obs = env._get_observations()

        # --- what it can see -----------------------------------------------
        cloud = object_point_cloud(env, result.object_name, obs=obs)
        # **The object's cloud is the full scan; the scene's stays observed.**
        #
        # Two different questions, and only one of them is about perception. The
        # *object* cloud decides what grasps exist and where the keypoints sit,
        # and a camera cloud of it is one-sided and top-dominated -- which is
        # what left the narrow hands choosing grasps on rims. Replacing it with
        # the mesh removes perception from that question so the map, the grasp
        # generator and the executor can be measured without it.
        #
        # The *scene* cloud is what the collision filters test the arm against,
        # and there the whole point is that the robot knows its own links
        # exactly and knows the shelf only as a point cloud. Giving it the true
        # scene would make the filter depend on something hardware will not
        # have, so ``scene_point_cloud`` below is untouched.
        #
        # ``camera_positions`` is kept from the real capture, so the visibility
        # filter still asks which directions the object was actually seen from.
        if getattr(env, "world", None) == "ycb":
            from tpgpt.sim.ycb import world_surface_cloud

            cloud = replace(cloud, points=world_surface_cloud(env, result.object_name))
            result.metrics["object_cloud_source"] = "ycb scan"
        else:
            result.metrics["object_cloud_source"] = "cameras"
        result.metrics["cloud_points"] = len(cloud)
        if len(cloud) < MIN_CLOUD_POINTS:
            return _fail(
                result, "object_not_seen",
                f"only {len(cloud)} points on the {result.object_name}; it is "
                "hidden behind something",
            )
        scene = scene_point_cloud(env, exclude=(result.object_name,), obs=obs)

        # --- candidate grasps ----------------------------------------------
        from tpgpt.grasp.pipeline import grasps_for_cloud

        grasp_set = grasps_for_cloud(cloud, gripper)
        result.metrics["grasps_generated"] = len(grasp_set)
        if not len(grasp_set):
            return _fail(result, "no_grasp_generated", "the generator returned nothing")

        target_height = float(np.ptp(cloud.points[:, 2]))
        provisional = place_pose_for(env, result.slot, grasp_set.grasps[0], gripper,
                                     target_height)
        # Four arguments that were missing, and one of them was a live bug.
        #
        # ``carry_rotation`` is the bug. The demonstration turns the hand 35.5
        # degrees between closing the jaws and opening them. Without it both
        # place-side stages judge the hand in its *pick* orientation at the
        # release position -- and against a slot that fits a 204 mm hand one way
        # round and not the other, that inverts the answer rather than shading
        # it. ``FINDINGS.md`` §8p.
        #
        # ``centre_of_mass`` switches on a stage that otherwise never runs at
        # all. ``reference_approach`` is now off by default: the target's roll
        # is resolved inside ``scene_keypoints`` from the grasps' own closing
        # axes, not from this argument, so the 45 degree test only narrows the
        # pool. And the pool itself was capped at 8 candidates, which is below
        # what the path check needs to find an admissible one.
        funnel = filter_grasps(
            grasp_set.grasps, gripper, cloud.points, scene_points=scene,
            camera_positions=cloud.camera_positions, env=env, place_pose=provisional,
            reference_approach=(
                _demonstrated_approach(labels) if approach_filter else None
            ),
            centre_of_mass=object_centre_of_mass(env, result.object_name),
            carry_rotation=carry_transform(labels)[0],
            target_name=result.object_name,
        )
        result.funnel = funnel
        if not len(funnel.survivors):
            return _fail(result, "no_grasp_survived",
                         f"every candidate was rejected, mostly by {funnel.rejected_by}")

        # --- keypoints and transport ---------------------------------------
        def target_of(grasp):
            slot = env.slot_poses()[result.slot]
            return ObjectPlacement(
                points=cloud.points,
                grasp=GraspFrame.from_grasp(grasp),
                support_height=float(env.table_offset[2]),
                destination=slot,
                destination_height=float(slot[2]),
            )

        # Everything downstream -- keypoint pairing, the diffeomorphism check,
        # the executability score and the transport itself -- has to speak about
        # the *same* trajectory. That is the fingertip path (see below), so the
        # conversion happens here, before a grasp is chosen, rather than after.
        #
        # Scoring the wrist path and executing the tool path is a real bug and
        # it was live: the selector ranked candidates on a trajectory displaced
        # by the hand's contact depth from the one it then ran -- 41 mm on a
        # Panda, 117 mm on a UMI.
        #
        # ``carry_transform`` is unaffected by the change of frame: for a rigid
        # attachment ``T(release) . T(grasp)^-1`` is identical whichever point of
        # the hand parametrises it, since the constant offset cancels.
        source_offset = contact_offset(SOURCE_GRIPPER)
        target_offset = contact_offset(gripper)
        tool_labels = _to_tool_frame(labels, source_offset)
        # **And into the grasp convention.** ``record_demonstration`` records the
        # source hand's ``grip_site`` orientations, while every keypoint cube is
        # built from a grasp pose, which is in GraspGen-X's convention -- the one
        # frame that means the same thing on all nine hands. Working in it from
        # end to end leaves exactly one per-hand step, putting the *executing*
        # hand's alignment back on below. Without it the source Panda's own
        # 180 degree alignment rides through the map and arrives attached to
        # whichever hand is executing: 0.2 degrees of error on a Robotiq 2F-85,
        # 90 on an XArm. 7.34.
        #
        # ``replace`` rather than rebuilding field by field. Rebuilding dropped
        # **stiffness, damping and time_rate** here: only five families were
        # listed, so the policy was fitted without an impedance channel and
        # ``rollout_policy`` then dereferenced ``prediction.stiffness[0]`` on a
        # ``None``. ``PolicyLabels.replace``'s own docstring names this failure
        # -- "rebuilding the object field by field instead is how a family
        # quietly gets dropped" -- which is the argument for never doing it.
        tool_labels = tool_labels.replace(
            orientations=to_grasp_convention(tool_labels.orientations, SOURCE_GRIPPER)
        )

        candidates = [grasp_set.grasps[i] for i in funnel.survivors[:max_candidates]]
        chosen = _choose_grasp(
            env, placement, target_of, tool_labels, keypoint_parts, candidates,
            box=keypoint_box, orientation=keypoint_orientation,
            cube_half_extent=cube_half_extent,
            gripper=gripper, object_name=result.object_name,
        )
        if chosen is None:
            return _fail(result, "map_not_a_diffeomorphism",
                         "no surviving grasp produced a valid transportation map")
        result.grasp, sets, path_flags = chosen
        # The object's width along **this grasp's own** closing axis, taken from
        # the cloud. Recorded here because this is the only place both the cloud
        # and the chosen grasp are in scope, and because ``closing_budget``
        # cannot get it from the keypoints when they are a fixed grasp cube --
        # their extent is then the cube's 40.0 mm for every object alike (7.28).
        #
        # Not an axis-aligned extent: a 30 x 100 mm box yawed 45 degrees
        # measures 92 x 92 and reads as ungraspable (7.18).
        result.metrics["object_width_closing"] = float(
            np.ptp(cloud.points @ result.grasp.closing)
        )
        result.alternatives = [g for g in candidates if g is not result.grasp][:5]
        # **The whole path-check record, not a single number.** Which candidate
        # was chosen, how far down the ranking it sat, whether nothing was
        # admissible and the least-bad one had to run, and for every candidate
        # examined the waypoint of its first fault and what that fault was.
        # Without this a fallback cell is indistinguishable from an ordinary one
        # in the results, which is how a campaign comes to report numbers for a
        # scene where no admissible grasp existed.
        result.metrics.update(path_flags)
        check = path_flags.get("path_check", {})
        result.metrics["path_fell_back"] = bool(check.get("fell_back"))
        result.metrics["path_rank_examined"] = check.get("rank_examined")
        # **The index into the *raw* candidate list, not into the survivors.**
        # ``_choose_grasp`` enumerates ``candidates``, which is already filtered
        # and truncated, so its ``chosen`` is a position in the survivor list
        # and means nothing to anyone holding a different funnel.
        # ``run_keypoint_transport.target_placement``'s ``forced_index`` indexes
        # ``grasp_set.grasps`` -- all 100 of them -- and that list comes from
        # the grasp cache, so it is identical for every driver and every commit
        # that sees the same cloud. Mapping back through ``funnel.survivors``
        # here is what lets a replay execute *this* run's grasp by construction
        # rather than by re-deriving it and hoping the two agree.
        #
        # 7.41 is the cost of not having had it: two campaigns on the same grid
        # picked differently in 10 of 20 cells from an identical candidate set,
        # and the totals read as a clean one-cell difference.
        survivors = list(funnel.survivors[:max_candidates])
        rank = check.get("chosen")
        result.metrics["grasp_chosen_index"] = (
            int(survivors[rank])
            if rank is not None and 0 <= rank < len(survivors) else None
        )
        # The funnel's own stage tallies, so a rejection can be attributed to
        # the stage that made it rather than only to the funnel as a whole.
        if result.funnel is not None:
            result.metrics["funnel_stages"] = {
                stage.name: [stage.entered, stage.survived]
                for stage in result.funnel.stages
            }
            result.metrics["funnel_fallbacks"] = [
                stage.name for stage in result.funnel.stages if stage.fallback
            ]
        source_set, target_set = _select_parts(sets[0], sets[1], keypoint_parts)
        result.source_keypoints, result.target_keypoints = source_set, target_set
        result.source_capture = placement.metadata.get("capture")
        if source_set.is_degenerate() or target_set.is_degenerate():
            return _fail(result, "keypoints_degenerate",
                         f"{len(source_set)} keypoints do not span three dimensions")

        timings = Timings()
        with timings.stage("map_fit"):
            transport_map = TransportMap().fit(source_set.points, target_set.points)
        result.transport_map = transport_map
        result.timings = timings.stages
        # Checked on the positions the map is actually applied to -- the tool
        # path -- not on the wrist path it is no longer used for.
        report = transport_map.check_diffeomorphism(tool_labels.positions)
        result.metrics.update(
            keypoint_residual=float(transport_map.keypoint_residual().max()),
            min_det=float(report.min_determinant),
            fraction_positive=float(report.fraction_positive),
            n_keypoints=len(source_set),
        )
        if not report.consistent_sign:
            return _fail(result, "map_not_a_diffeomorphism",
                         f"det(J) changes sign; minimum {report.min_determinant:.3f}")

        # Transport the **fingertip** trajectory, not the wrist one.
        #
        # The keypoints are anchored where the hand holds the object, and a
        # transportation map is exact only at its keypoints. The demonstration
        # is recorded at ``grip_site``, a hand-specific lever arm away -- 41 mm
        # on the Panda that recorded it, up to 117 mm on a UMI -- so warping the
        # wrist path spends that guarantee on the wrong point. Measured over ten
        # runs, moving the map into the tool frame halves the distance between
        # the warped path and the chosen grasp, 15.6 mm to 7.2 mm, and it was
        # never worse on any of them. ``ROBOTICS_NOTES`` section 7.20.
        #
        # The source offset is the demonstrating hand's and the target offset is
        # the executing hand's, which is exactly right across embodiments: the
        # map carries contact point to contact point, and each hand steps out to
        # its own wrist from there.
        with timings.stage("label_transport"):
            transported = transport_labels(transport_map, tool_labels)
        # Back into *this* hand's wrist convention, which is what the controller
        # commands and what IK aims. See the conversion above.
        if transported.orientations is not None:
            transported.orientations = to_wrist_convention(
                transported.orientations, gripper
            )
        result.demonstration = labels.positions
        result.transported = transported.positions
        result.transported_labels = transported
        result.metrics["tool_offset_mm"] = float(np.linalg.norm(target_offset) * 1000)

        # --- what the map *is*, recorded before anything is executed --------
        #
        # A campaign has always stored ``min_det``, and ``min_det`` is one
        # scalar: two entirely different maps can share it, so it cannot support
        # a comparison *between* maps. The map is a deformation of space, so it
        # is recorded as one -- ``phi(x) - x`` on a grid identical in every cell.
        # Recorded here, at the point the map exists and before the rollout can
        # fail, because a cell that fails still has a map worth comparing.
        result.map_fingerprint = fingerprint(transport_map)
        grasp_index, release_index = carry_indices(labels)
        # ``det J`` at the two poses that decide the task, not only its minimum
        # over the path. Property (R2) and the conditioning table are about
        # these two points specifically: the grasp, where the plan has to land
        # on the object, and the release, where it has to land in the slot.
        determinants = np.linalg.det(
            transport_map.jacobian(tool_labels.positions[[grasp_index,
                                                          release_index]])
        )
        result.metrics["det_J_at_grasp"] = float(determinants[0])
        result.metrics["det_J_at_release"] = float(determinants[1])
        # Both poses as 4x4 matrices, in GraspGen-X's convention so they are
        # comparable across hands. The pick pose is the chosen grasp; the place
        # pose is where the plan commands the hand when the jaws open. Until now
        # only the *index* of the chosen candidate was stored, which identifies
        # the grasp only to someone holding the same cache.
        #
        # **The grasp as executed, not as the planner emitted it.**
        # ``scene_keypoints`` may roll the incoming candidate a half turn about
        # its approach when that agrees better with the demonstration, and its
        # own docstring warns that anything converting a grasp into a robot
        # command must be given *that* pose: the two differ by 180 degrees, and
        # once the jaws are shut a half turn at one end reflects the object
        # through the grasp point rather than leaving it where it was -- 4 to
        # 100 mm over twenty cells (7.33).
        frames = sets[2]

        def _pose(frame):
            """``(position, rotation)`` from a ``GraspFrame`` or a ``Grasp6D``.

            The two coexist on purpose -- ``GraspFrame`` is deliberately not
            ``Grasp6D`` so keypoint extraction stays usable without the gripper
            registry -- and they name the same point differently, ``tcp``
            against ``position``.
            """
            position = getattr(frame, "tcp", None)
            if position is None:
                position = frame.position
            return (np.asarray(position, dtype=float).reshape(3),
                    np.asarray(frame.rotation, dtype=float).reshape(3, 3))

        executed = frames["target_grasp"]
        placed = frames["target_place_grasp"]
        result.metrics["grasp_rolled"] = bool(frames.get("target_grasp_rolled"))
        executed_p, executed_R = _pose(executed)
        placed_p, placed_R = _pose(placed)
        result.metrics["grasp_pose_target"] = pose_matrix(executed_p, executed_R)
        result.metrics["grasp_pose_release"] = pose_matrix(placed_p, placed_R)
        # Orientation error at **both** ends, against the frames the keypoints
        # were actually built on.
        #
        # Two things this deliberately is not. It is not measured against the
        # *transported* orientation, which would compare the map's output with
        # itself and return zero by construction -- the first version of this
        # did exactly that and read 0.0 on every cell, which looked like a
        # perfect result. And the release column is not assumed to equal the
        # grasp column: a roll taken at both ends cancels, and one taken at a
        # single end does not, so the two have to be measured separately. Only
        # the grasp end has ever been recorded here.
        source_pick_p, source_pick_R = _pose(placement.grasp)
        source_place_p, source_place_R = _pose(frames["source_place_grasp"])
        errors = orientation_transport_error(
            transport_map,
            np.stack([source_pick_p, source_place_p]),
            np.stack([source_pick_R, source_place_R]),
            np.stack([executed_R, placed_R]),
        )
        result.metrics["orientation_error_grasp_deg"] = float(errors[0])
        result.metrics["orientation_error_release_deg"] = float(errors[1])

        # --- execute --------------------------------------------------------
        if executor == "replay":
            rollout = _replay_execution(env, result, transported, target_offset,
                                        gripper, rollout_kwargs or {})
            result.rollout = rollout
            result.diagnosis = diagnose(result, env)
            result.metrics.update(rollout.metadata)
            result.timings = timings.stages
            _record_errors(result, env, rollout)
            if not rollout.success:
                return _fail(
                    result, "placed_in_the_wrong_place",
                    "the position-controlled replay finished without leaving the "
                    "object in its slot",
                )
            return result

        with timings.stage("policy_refit"):
            policy = GPPolicy().fit(transported)
        # Built as a dict so ``rollout_kwargs`` can override any of it. Passing
        # these as explicit keywords made a sweep over one of them a TypeError
        # rather than a sweep, which defeats the point of the argument.
        settings = {
            "max_steps": max_steps,
            "score": slot_score(result.object_name, result.slot),
            "probe": object_probe(env, result.object_name),
            "tool_offset": target_offset,
        }
        settings.update(rollout_kwargs or {})
        rollout = rollout_policy(env, policy, **settings)
        result.rollout = rollout
        result.diagnosis = diagnose(result, env)
        result.metrics.update(rollout.metadata)
        result.timings = timings.stages
        _record_errors(result, env, rollout)
        _, release = carry_indices(labels)
        result.metrics["release_index"] = release

        if not rollout.success:
            if rollout.metadata.get("blocked"):
                return _fail(
                    result, "arm_could_not_hold_the_pose",
                    f"the arm settled {rollout.metadata['final_lag'] * 1000:.0f} mm "
                    f"behind its target after {rollout.metadata['steps']} steps and "
                    "stopped making progress",
                )
            if not rollout.metadata.get("terminated_on_phase"):
                return _fail(result, "policy_stalled",
                             f"ran {rollout.metadata['steps']} steps without finishing")
            return _fail(
                result, "placed_in_the_wrong_place",
                f"ended {rollout.metadata['placement_error_xy'] * 1000:.0f} mm from the slot",
            )
        return result
    finally:
        result.seconds = time.time() - started
        if owned:
            env.close()



def _replay_execution(env, result, transported, tool_offset, gripper, extra):
    """Drive the transported waypoints under position control.

    The **executor-free ceiling**: no policy is fitted, no attractor is
    integrated and no lag gate runs, so whatever the arm fails to do here is not
    the executor's doing.

    ``time_belief`` is attached because :func:`~tpgpt.experiments.diagnose.diagnose`
    reads it to say how far the clock got, and a replay has no clock of its own.
    Its phase is simply its progress through the waypoint list, which is exactly
    what the policy's time belief is an estimate *of*, so the two are comparable
    -- and a replay's trace is recorded per waypoint rather than per control
    step, so the phase has to be per waypoint too or the arrays disagree in
    length and every stage is attributed to the wrong moment.
    """
    from tpgpt.experiments.diagnose import object_probe
    from tpgpt.experiments.run_keypoint_replay import _fingers_gate
    from tpgpt.sim.replay import replay_labels
    from tpgpt.sim.rollout import slot_score

    settings = {
        "tool_offset": tool_offset,
        "score": slot_score(result.object_name, result.slot),
        "probe": object_probe(env, result.object_name, gripper),
        # The lift waits for the jaws rather than for a step count. Opted into
        # here, as the Tier 2 driver does, because the delay between commanding
        # a close and having hold of anything is not derivable: 0, 0, 10 and 15
        # waypoints across four objects on one hand, and not monotonic in
        # object width.
        #
        # The gate tests **opposition** -- at least one finger from each side of
        # the closing axis -- rather than counting contacts. A count is safe on a
        # parallel jaw, where two fingers is both of them, and wrong on a
        # three-finger hand, which carries two fingers on one side and one on
        # the other, so two of its fingers touching can be the pair shoving the
        # object.
        "grasp_gate": _fingers_gate(env, gripper, result.object_name),
    }
    settings.update(extra)
    replay = replay_labels(env, transported, **settings)
    replay.time_belief = np.linspace(0.0, 1.0, len(replay.positions))
    # **A position-controlled replay's attractor is its commanded pose.** The
    # diagnosis measures how far the setpoint wandered from the transported
    # path, and under position control the setpoint is exactly the waypoint --
    # there is no spring and nothing to integrate -- so the drift it reports for
    # a replay is identically zero by construction. That is the correct answer
    # and it is worth having: it is the baseline the policy's drift is read
    # against.
    replay.attractors = np.asarray(replay.targets, dtype=float)[:, :3]
    replay.metadata.setdefault("tool_offset", tool_offset)
    return replay


def _record_errors(result, env, rollout):
    """Placement and release error, in three dimensions, from the object trace."""
    # --- errors, in three dimensions and at both moments ----------------
    #
    # Only a horizontal placement error has ever been recorded, and only at
    # the end. Two things are missing from that and both are load-bearing.
    #
    # **Components, not a magnitude.** A scalar distance cannot say whether
    # the object landed short, beside, or on top of the slot, and those are
    # different faults with different causes. The vertical component in
    # particular is the one a horizontal-only ruler is blind to -- which is
    # the same blindness that made ``stage_outcome``'s ``placed_on_shelf``
    # flag pass a cell sitting 84.4 mm from its slot.
    #
    # **At the release, not only at rest.** Every placement in this system
    # is a *drop*: the arm stops short of the commanded release pose and the
    # jaws open above the board. Whether that is the mechanism behind the
    # marginal cells or merely a coincidence is unresolved **precisely
    # because one side of the comparison never recorded where the object was
    # when the jaws opened.** It does now.
    # **Where the object's origin should end up, not where the board is.**
    # ``slot_poses`` gives the board surface, and the probe watches the
    # object's body origin, which sits half the object's height above
    # whatever it rests on. Differencing those directly makes the vertical
    # component read +61.8 mm for a correctly placed can -- its own half
    # height -- so the 3-D magnitude would be dominated by a constant that
    # is not an error at all. The object's height above its support is
    # measured at the pick, where it is known to be resting properly, and
    # carried across.
    destination = np.asarray(env.slot_poses()[result.slot], dtype=float)
    probe = rollout.metadata.get("probe", {})
    traces = [np.asarray(probe.get(f"object_{a}", []), dtype=float)
              for a in "xyz"]
    positions = (np.stack(traces, axis=1)
                 if all(t.ndim == 1 and len(t) for t in traces)
                 else np.zeros((0, 3)))
    if len(positions):
        stand_off = float(positions[0][2] - env.table_offset[2])
        destination = destination + np.array([0.0, 0.0, stand_off])
        result.metrics["object_stand_off_mm"] = stand_off * 1000
        final = positions[-1]
        result.metrics["placement_error_3d_mm"] = float(
            np.linalg.norm(final - destination)) * 1000
        result.metrics["placement_error_components_mm"] = (
            (final - destination) * 1000).round(3).tolist()
        release_step = result.diagnosis.measurements.get("release_step")
        if release_step is not None:
            at_release = positions[min(int(release_step), len(positions) - 1)]
            result.metrics["release_error_3d_mm"] = float(
                np.linalg.norm(at_release - destination)) * 1000
            result.metrics["release_error_components_mm"] = (
                (at_release - destination) * 1000).round(3).tolist()
            # How far the object still had to fall when it was let go: the
            # drop itself, which is the quantity the mechanism is about.
            result.metrics["release_drop_mm"] = float(
                np.linalg.norm(final - at_release)) * 1000
    # The **first** stage that failed, which is not the same question as
    # which phase carried the most unreachable waypoints. That one reads
    # "approach" whenever nothing was unreachable at all, which is an
    # accusation rather than an attribution.
    failure = result.diagnosis.first_failure
    result.metrics["first_failure_stage"] = failure.name if failure else None


#: Extra samples taken on each side of the grasp and the release, and how far
#: either side of them, in label steps.
#:
#: A uniform sweep spends its budget on free-space transit, where the arm has
#: room and infeasibility is cheap: if the hand cannot hold its angle halfway
#: through a carry it can give up a few degrees and nothing is lost. The two
#: moments where it *cannot* give anything up are closing on the object and
#: setting it down, and those occupy a handful of labels out of two hundred.
#: Sampling uniformly is therefore sampling in the wrong place.
#: **Retired.** These three, and :func:`executable_fraction` below, were the
#: grasp selector until the whole-path check replaced them. They sampled 14
#: poses of a 200-waypoint trajectory, and ``ROBOTICS_NOTES`` §7.38 measured
#: candidates passing that sample and then colliding at 79 to 162 of the other
#: 187 -- so the proxy was answering a different question, not a coarser version
#: of the same one.
#:
#: Kept rather than deleted because §7.25's account of the infeasibility
#: fallback is written in terms of them, and because a reader comparing this
#: file against that section should find the thing it describes. Nothing calls
#: them; ``run_experiments`` reads ``executable_fraction`` out of the metrics,
#: where it is now absent and reads as ``None``.
CRITICAL_SAMPLES = 6
CRITICAL_WINDOW = 12


def executable_fraction(
    env,
    transport_map,
    labels,
    arm: str = "right",
    samples: int = 14,
    critical: bool = True,
):
    """How much of the warped path the arm can follow, hand pose included.

    The filters check the grasp and the placement and their approach corridors.
    What is actually executed is the whole warped trajectory, and the arm has to
    hold a commanded *orientation* the entire way. Those are different
    questions: measured on a failing run, every sampled point was reachable in
    position and only 3 of 20 with its orientation, because the path passed
    through places the arm can get to but cannot get to *with its hand pointing
    down*.

    Args:
        labels: The labels the map will actually be applied to. These must be in
            the **same frame as the executed trajectory** -- the tool frame,
            downstream of section 7.20 -- or this scores one path and the robot
            runs another, displaced by the hand's contact depth.
        critical: Also sample densely around the grasp and the release. Sampling
            density itself changes little -- 14 uniform samples give the same
            answer as 200 -- but *where* the samples fall changes what the number
            means, and the two moments that decide the task are narrow.

    Returns:
        The fraction of sampled poses the arm can hold, which is what a grasp
        should be chosen by.
    """
    from tpgpt.sim.keypoints import carry_indices
    from tpgpt.sim.kinematics import solve_ik

    positions = transport_map.transport_positions(labels.positions)
    rotations = transport_map.transport_orientations(labels.positions, labels.orientations)
    index = list(np.linspace(0, len(positions) - 1, samples).astype(int))

    if critical:
        try:
            grasp_index, release_index = carry_indices(labels)
        except Exception:  # pragma: no cover - labels without a gripper channel
            grasp_index = release_index = None
        for centre in (grasp_index, release_index):
            if centre is None:
                continue
            index += list(
                np.clip(
                    np.linspace(
                        centre - CRITICAL_WINDOW, centre + CRITICAL_WINDOW,
                        CRITICAL_SAMPLES,
                    ).astype(int),
                    0, len(positions) - 1,
                )
            )
    index = sorted(set(int(i) for i in index))
    solved = sum(
        solve_ik(env, positions[i], rotations[i], arm=arm, position_tolerance=0.015).reachable
        for i in index
    )
    return solved / float(len(index))


def _choose_grasp(env, placement, target_of, labels, keypoint_parts, candidates,
                  arm="right", box=DEFAULT_KEYPOINT_BOX,
                  orientation=DEFAULT_KEYPOINT_ORIENTATION,
                  cube_half_extent=GRASP_CUBE_HALF_EXTENT,
                  gripper: str = "panda", object_name: str | None = None,
                  min_reachable_fraction: float = MIN_REACHABLE_FRACTION,
                  stride: int = 4):
    """Pick the candidate whose *transported motion* the arm can actually follow.

    Each candidate defines a slightly different keypoint frame and so a slightly
    different warp, and the differences matter far more than they look: the same
    scene gives paths the arm can follow almost entirely and paths it can barely
    start.

    **The roll is no longer chosen here.** This used to try both frame signs per
    candidate and keep whichever score preferred, on the argument that a
    parallel jaw closing along ``+c`` and ``-c`` is one grasp commanded as two
    wrist angles 180 degrees apart. The argument is sound and the criterion was
    not: ``FINDINGS.md`` §8j measured it picking the *worse* roll at 2 of 8
    yaws. ``scene_keypoints`` resolves the roll once, on the grasp, by agreement
    with the demonstration.

    **And a candidate is judged on the whole path it produces.** This used to
    rank by :func:`executable_fraction`, which samples 14 poses out of 200.
    ``ROBOTICS_NOTES`` §7.38 measured candidates passing that sample and then
    colliding at **79 to 162** of the other 187 waypoints, so the proxy was not
    merely coarse -- it was answering a different question. Selection now walks
    the ranked list and takes the **first admissible** candidate, judged by
    :func:`~tpgpt.grasp.filters.path_feasibility_observed` at every waypoint.

    Two things about that check are deliberate. It sees only what the robot
    could: the arm's own convex geometry from its description file, and the
    scene as an observed cloud **with the arm subtracted from it** -- without
    that subtraction the arm collides with its own reflection, because a depth
    image of a workspace contains the arm. And faults are judged by phase at a
    zero threshold: nothing may touch during the approach, only the fingers may
    touch the object during the carry, and the object may rest on a support
    surface throughout.

    Returns ``(grasp, sets, flags)``, or ``None`` if every candidate produced a
    degenerate keypoint set. ``flags`` records each candidate examined and why
    it was rejected, so a cell that had to fall back reads as "no admissible
    grasp exists here" rather than as an ordinary result.
    """
    from tpgpt.grasp.filters import path_feasibility_observed
    from tpgpt.perception.cameras import scene_point_cloud
    from tpgpt.perception.obstacles import robot_bodies, self_filtered

    pair = resolve_pair(gripper)
    # Built once. Neither the robot's own shape nor the scene changes while
    # candidates are compared, and rebuilding them per candidate would be most
    # of the cost.
    bodies = robot_bodies(env)
    scene = self_filtered(
        env, scene_point_cloud(env, exclude=(object_name,) if object_name else (),
                               obs=None),
        bodies,
    )
    grasp_index, release_index = carry_indices(labels)

    examined, fallback, degenerate = [], None, None
    for index, grasp in enumerate(candidates):
        target = target_of(grasp)
        sets = scene_keypoints(
            placement, target, labels,
            # The support point is emitted by the same block as the jaw
            # contacts, so it has to be switched on for either of them.
            include_contacts=bool({"contacts", "support"} & set(keypoint_parts)),
            box=box, orientation=orientation, cube_half_extent=cube_half_extent,
        )
        source_set, target_set = _select_parts(sets[0], sets[1], keypoint_parts)
        if source_set.is_degenerate() or target_set.is_degenerate():
            examined.append({"index": index, "rejected": "degenerate keypoints"})
            if degenerate is None:
                degenerate = (grasp, sets)
            continue
        transport_map = TransportMap().fit(source_set.points, target_set.points)
        if not transport_map.check_diffeomorphism(labels.positions).consistent_sign:
            examined.append({"index": index, "rejected": "map folded"})
            continue

        warped = transport_map.transport_positions(labels.positions)
        rotations = transport_map.transport_orientations(
            labels.positions, labels.orientations
        )
        feasible = path_feasibility_observed(
            env, warped, rotations, pair, scene, placement.points,
            grasp_index, release_index, bodies=bodies, stride=stride,
            # Selection is decided by the first fault, so sweeping the rest is
            # wasted here. Diagnosis wants the whole tally and asks for it.
            stop_early=True,
        )
        record = {
            "index": index,
            "violations": feasible["violations"],
            "first_violation": feasible["first_violation"],
            "reachable_fraction": round(feasible["reachable_fraction"], 3),
            "faults": {k: v for k, v in list(feasible["faults"].items())[:4]},
        }
        if feasible["violations"]:
            record["rejected"] = "collision"
        elif feasible["reachable_fraction"] < min_reachable_fraction:
            record["rejected"] = "kinematics"
        else:
            examined.append(record)
            return grasp, sets, {"path_check": {
                "chosen": index, "rank_examined": len(examined),
                "fell_back": False, "examined": examined,
            }}
        examined.append(record)
        if fallback is None or _got_further(record, fallback[2]):
            fallback = (grasp, sets, record)

    # **Nothing was admissible, so take the least bad and say so.** Falling back
    # to the top-ranked candidate would throw away everything the check just
    # learned. The one that got furthest before its first fault is the one whose
    # plan is wrong latest, and on a path that ends at a shelf that is the one
    # most likely to have done the useful part of the task first.
    if fallback is not None:
        return fallback[0], fallback[1], {"path_check": {
            "chosen": fallback[2]["index"], "rank_examined": len(examined),
            "fell_back": True, "examined": examined,
        }}
    if degenerate is not None:
        return degenerate[0], degenerate[1], {"path_check": {
            "chosen": None, "fell_back": True, "examined": examined,
        }}
    return None


def _got_further(record: dict, best: dict) -> bool:
    """Whether ``record``'s first fault comes later than ``best``'s."""
    far = lambda r: (r.get("first_violation") if r.get("first_violation") is not None
                     else 10**9)
    return far(record) > far(best)


def _to_tool_frame(labels, offset: np.ndarray):
    """Re-express a label set at the point where the hand holds an object.

    A rigid change of reference point on the same body: positions move by
    ``R @ offset`` and every other label -- velocity, orientation, stiffness,
    gripper, phase -- is unchanged, since none of them depends on which point of
    a rigid body is being tracked. (The velocities of two points on a rotating
    body do differ, by ``omega x (R @ offset)``; that term is dropped here
    because the rollout also runs in this frame, so the velocity it integrates
    and the position it integrates are the same point's.)
    """
    if labels.orientations is None or not np.any(offset):
        return labels
    shifted = labels.positions + np.einsum("nij,j->ni", labels.orientations, offset)
    return labels.replace(positions=shifted)


def _demonstrated_approach(labels) -> np.ndarray:
    """The direction the demonstration's hand advanced along to grasp.

    Taken from the hand's own motion just before the jaws close, so it is
    whatever the teacher actually did rather than an assumption that it was
    top-down.
    """
    grasp_index, _ = carry_indices(labels)
    window = labels.positions[max(0, grasp_index - 12) : grasp_index + 1]
    if len(window) < 2:
        return np.array([0.0, 0.0, -1.0])
    travel = window[-1] - window[0]
    norm = float(np.linalg.norm(travel))
    return travel / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])


def _select_parts(source_set, target_set, parts: tuple[str, ...]):
    """Keep only the requested keypoint families.

    ``scene_keypoints`` emits the box always and the contacts optionally, so
    dropping the box, or keeping the support point without the contacts, is done
    here by label. This is what makes the seven-way ablation a one-line change.
    """
    from tpgpt.sim.keypoints import CORNER_NAMES, KeypointSet

    box = {"center", *CORNER_NAMES}
    wanted = []
    for label in source_set.labels:
        role = label.split("_", 1)[1]
        if role in box and "box" in parts:
            wanted.append(label)
        elif role.startswith("grasp_") and "contacts" in parts:
            wanted.append(label)
        elif role == "support" and "support" in parts:
            wanted.append(label)

    index = [source_set.labels.index(label) for label in wanted]
    return (
        KeypointSet(source_set.points[index], wanted, metadata=source_set.metadata),
        KeypointSet(target_set.points[index], wanted, metadata=target_set.metadata),
    )
