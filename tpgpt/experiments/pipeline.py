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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.experiments.diagnose import diagnose, object_probe
from tpgpt.experiments.reshelving_pipeline import record_source_placement
from tpgpt.grasp.filters import FilterFunnel, filter_grasps
from tpgpt.grasp.grasps import Grasp6D, contact_offset, grasp_to_eef_pose
from tpgpt.grasp.grippers import resolve_pair
from tpgpt.language.parser import parse_task
from tpgpt.perception.cameras import object_point_cloud, scene_point_cloud
from tpgpt.perception.scene_graph import build_scene_graph
from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.sim.keypoints import (
    GraspFrame,
    ObjectPlacement,
    carry_indices,
    scene_keypoints,
)
from tpgpt.sim.rollout import rollout_policy, slot_score
from tpgpt.transport.labels import transport_labels
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

#: Surviving grasps whose transported motion is scored before choosing one.
#:
#: Each costs a map fit and fourteen IK solves, about a tenth of a second, and
#: the search stops early on the first candidate whose whole path is executable.
MAX_CANDIDATES = 8


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

    @property
    def success(self) -> bool:
        return self.outcome == "success"

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


def build_scene(
    objects=("milk", "can", "cereal", "bread"),
    gripper: str = "panda",
    shelf_variant: str = "cubby",
    seed: int = 0,
    camera_size: int = 256,
    controller_config: dict | None = None,
):
    """A tabletop-shelf scene with depth, segmentation and torque control.

    Args:
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
    objects=("milk", "can", "cereal", "bread"),
    source: tuple | None = None,
    include_contacts: bool = False,
    keypoint_parts: tuple[str, ...] = DEFAULT_KEYPOINTS,
    max_steps: int = 600,
    env=None,
    rollout_kwargs: dict | None = None,
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
    """
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

        obs = env._get_observations()

        # --- what it can see -----------------------------------------------
        cloud = object_point_cloud(env, result.object_name, obs=obs)
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
        funnel = filter_grasps(
            grasp_set.grasps, gripper, cloud.points, scene_points=scene,
            camera_positions=cloud.camera_positions, env=env, place_pose=provisional,
            reference_approach=_demonstrated_approach(labels),
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

        candidates = [grasp_set.grasps[i] for i in funnel.survivors[:MAX_CANDIDATES]]
        chosen = _choose_grasp(
            env, placement, target_of, tool_labels, keypoint_parts, candidates
        )
        if chosen is None:
            return _fail(result, "map_not_a_diffeomorphism",
                         "no surviving grasp produced a valid transportation map")
        result.grasp, sets, executable = chosen
        result.alternatives = [g for g in candidates if g is not result.grasp][:5]
        result.metrics["executable_fraction"] = executable
        source_set, target_set = _select_parts(sets[0], sets[1], keypoint_parts)
        result.source_keypoints, result.target_keypoints = source_set, target_set
        result.source_capture = placement.metadata.get("capture")
        if source_set.is_degenerate() or target_set.is_degenerate():
            return _fail(result, "keypoints_degenerate",
                         f"{len(source_set)} keypoints do not span three dimensions")

        transport_map = TransportMap().fit(source_set.points, target_set.points)
        result.transport_map = transport_map
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
        transported = transport_labels(transport_map, tool_labels)
        result.demonstration = labels.positions
        result.transported = transported.positions
        result.transported_labels = transported
        result.metrics["tool_offset_mm"] = float(np.linalg.norm(target_offset) * 1000)

        # --- execute --------------------------------------------------------
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


#: Extra samples taken on each side of the grasp and the release, and how far
#: either side of them, in label steps.
#:
#: A uniform sweep spends its budget on free-space transit, where the arm has
#: room and infeasibility is cheap: if the hand cannot hold its angle halfway
#: through a carry it can give up a few degrees and nothing is lost. The two
#: moments where it *cannot* give anything up are closing on the object and
#: setting it down, and those occupy a handful of labels out of two hundred.
#: Sampling uniformly is therefore sampling in the wrong place.
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


def _choose_grasp(env, placement, target_of, labels, keypoint_parts, candidates, arm="right"):
    """Pick the candidate whose *transported motion* the arm can actually follow.

    Each candidate defines a slightly different keypoint frame and so a slightly
    different warp, and the differences matter far more than they look: the same
    scene gives paths the arm can follow almost entirely and paths it can barely
    start. Both frame signs are tried for each, since a parallel jaw closing
    along ``+c`` and ``-c`` is one grasp commanded as two wrist angles 180
    degrees apart, and a wrist has a limited range.

    Returns ``(grasp, sets, score)`` for the best, or ``None`` if every
    candidate produced a degenerate keypoint set.
    """
    best = None
    for grasp in candidates:
        target = target_of(grasp)
        for flip in (False, True):
            sets = scene_keypoints(
                placement, target, labels,
                # The support point is emitted by the same block as the jaw
                # contacts, so it has to be switched on for either of them.
                include_contacts=bool({"contacts", "support"} & set(keypoint_parts)),
                flip_target=flip,
            )
            source_set, target_set = _select_parts(sets[0], sets[1], keypoint_parts)
            if source_set.is_degenerate() or target_set.is_degenerate():
                if best is None:
                    best = (grasp, sets, -1.0)
                continue
            transport_map = TransportMap().fit(source_set.points, target_set.points)
            if not transport_map.check_diffeomorphism(labels.positions).consistent_sign:
                continue
            score = executable_fraction(env, transport_map, labels, arm=arm)
            if best is None or score > best[2]:
                best = (grasp, sets, score)
            if score >= 1.0:
                return best
    return best


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
