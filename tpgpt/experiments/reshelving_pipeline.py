"""End-to-end policy transportation on the reshelving task (paper Sec. V-A).

The complete pipeline, in the order the paper lays it out:

1. Record **one** demonstration in a source scene and extract its keypoints
   (Sec. III-A).
2. Observe a new scene and extract the paired target keypoints.
3. Fit the transportation map ``phi`` (Sec. III-D, III-E).
4. Transport every label family through it (Sec. III-F, III-G) and quantify the
   uncertainty (Sec. III-H).
5. Refit a policy ``g`` on the transported labels (Sec. III-B, step 2).
6. Execute that policy on the robot in the target scene.

Step 5 is the one the prototype skipped; without it this is trajectory
reshaping, which the paper explicitly rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.policy.gp_policy import GPPolicy
from tpgpt.policy.rollout import total_velocity_std
from tpgpt.sim.backend import make_reshelving_env
from tpgpt.sim.demo import GRASP_HEIGHT_OFFSET, record_demonstration
from tpgpt.sim.keypoints import (
    GraspFrame,
    KeypointSet,
    ObjectPlacement,
    keypoints_from_bodies,
    pair_keypoints,
    sample_box_surface,
    scene_keypoints,
)
from tpgpt.sim.rollout import rollout_policy
from tpgpt.sim.scenes.reshelving import PRODUCT_HALF_SIZE
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import quat_to_matrix

#: Bodies tracked as task keypoints, with the half extent of each keypoint cube.
#: Sec. V-A requires at least three non-collinear points per object to pin down
#: a 3-D pose; a cube centre plus its eight corners gives nine.
TRACKED_BODIES = {
    "product_main": np.array(PRODUCT_HALF_SIZE),
    "goal_marker": np.array(PRODUCT_HALF_SIZE),
}


@dataclass
class TransportResult:
    """Everything one transported episode produced."""

    source_labels: PolicyLabels
    transported_labels: PolicyLabels
    source_keypoints: KeypointSet
    target_keypoints: KeypointSet
    transport_map: TransportMap
    policy: GPPolicy
    rollout: object
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.rollout.success)

    def summary(self) -> str:
        d = self.diagnostics
        return (
            f"success={self.success} "
            f"placement_error={d['placement_error_xy'] * 1000:.1f} mm "
            f"keypoint_residual={d['keypoint_residual_max'] * 1000:.3f} mm "
            f"det(J)>0={d['jacobian_fraction_positive'] * 100:.0f}% "
            f"steps={d['steps']}"
        )


def extract_keypoints(env, noise_std: float = 0.0, seed: int | None = None) -> KeypointSet:
    """Keypoints for the current scene configuration."""
    return keypoints_from_bodies(
        env,
        TRACKED_BODIES,
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
    )


def record_source(
    source_seed: int = 0,
    embodiment: str = "panda",
    control_freq: int = 20,
    video_path: str | Path | None = None,
    keypoint_noise: float = 0.0,
) -> tuple[PolicyLabels, KeypointSet, bool]:
    """Record the single source demonstration and its keypoints.

    Keypoints are read **before** the demonstration runs, since they describe
    the scene the teacher acted in.
    """
    env = make_reshelving_env(
        embodiment=embodiment,
        control_freq=control_freq,
        seed=source_seed,
        offscreen=video_path is not None,
    )
    try:
        env.reset()
        keypoints = extract_keypoints(env, keypoint_noise, seed=source_seed)
        labels = record_demonstration(env, video_path=video_path)
        return labels, keypoints, bool(env._check_success())
    finally:
        env.close()


def transport_to_scene(
    labels: PolicyLabels,
    source_keypoints: KeypointSet,
    target_seed: int,
    embodiment: str = "panda",
    control_freq: int = 20,
    video_path: str | Path | None = None,
    keypoint_noise: float = 0.0,
    max_steps: int = 600,
    residual: object = "default",
    source_placement: ObjectPlacement | None = None,
) -> TransportResult:
    """Transport the source policy into a new scene and execute it.

    Args:
        labels: Labels recorded in the source scene.
        source_keypoints: Keypoints of the source scene. Ignored when
            ``source_placement`` is given.
        target_seed: Seed selecting the new object pose, shelf height and slot.
        residual: Regressor for ``psi``; ``None`` gives an affine-only map,
            which is the natural ablation for the nonlinear stage.
        source_placement: When given, keypoints come from the general
            grasp-aligned extractor instead of from the two known bodies. The
            two paths describe the same scene, so the campaign result is the
            acceptance test for the general one.
    """
    env = make_reshelving_env(
        embodiment=embodiment,
        control_freq=control_freq,
        seed=target_seed,
        offscreen=video_path is not None,
    )
    try:
        env.reset()
        if source_placement is None:
            target_keypoints = extract_keypoints(env, keypoint_noise, seed=target_seed)
            S, T = pair_keypoints(source_keypoints, target_keypoints)
        else:
            target_placement = reshelving_placement(env, seed=target_seed)
            source_keypoints, target_keypoints, _ = scene_keypoints(
                source_placement, target_placement, labels
            )
            S, T = pair_keypoints(source_keypoints, target_keypoints)

        transport_map = TransportMap(residual=residual).fit(S, T)
        transported = transport_labels(transport_map, labels)
        policy = GPPolicy().fit(transported)
        rollout = rollout_policy(env, policy, max_steps=max_steps, video_path=video_path)

        report = transport_map.check_diffeomorphism(labels.positions)
        epistemic = policy.predict(
            transported.positions, transported.time_belief
        ).velocity_std
        diagnostics = {
            "target_seed": target_seed,
            "keypoint_residual_max": float(transport_map.keypoint_residual().max()),
            "jacobian_fraction_positive": report.fraction_positive,
            "jacobian_consistent_sign": report.consistent_sign,
            "jacobian_det_min": report.min_determinant,
            "jacobian_det_max": report.max_determinant,
            "transport_velocity_std_mean": float(np.mean(transported.velocity_std)),
            "epistemic_velocity_std_mean": float(np.mean(epistemic)),
            "total_velocity_std_mean": float(
                np.mean(total_velocity_std(epistemic, transported.velocity_std))
            ),
            "keypoint_displacement_max": float(np.linalg.norm(T - S, axis=1).max()),
            "affine_residual_max": float(
                np.linalg.norm(T - transport_map.affine.predict(S), axis=1).max()
            ),
            **rollout.metadata,
        }
        return TransportResult(
            source_labels=labels,
            transported_labels=transported,
            source_keypoints=source_keypoints,
            target_keypoints=target_keypoints,
            transport_map=transport_map,
            policy=policy,
            rollout=rollout,
            diagnostics=diagnostics,
        )
    finally:
        env.close()


def run_campaign(
    source_seed: int = 0,
    target_seeds: tuple[int, ...] = tuple(range(1, 11)),
    embodiment: str = "panda",
    control_freq: int = 20,
    keypoint_noise: float = 0.0,
    residual: object = "default",
    verbose: bool = True,
    keypoints: str = "bodies",
) -> tuple[list[TransportResult], dict]:
    """Transport one demonstration into many scenes and aggregate the outcome.

    This is the experiment Sec. V-A reports: a single demonstration generalised
    across randomised object and goal configurations.

    Args:
        keypoints: ``"bodies"`` reads the two known boxes out of the simulator;
            ``"grasp_aligned"`` runs the general extractor that fits a box to
            the object's cloud in a frame derived from the grasp. The second
            must reproduce the first's result, since it generalises it.
    """
    if keypoints not in ("bodies", "grasp_aligned"):
        raise ValueError(f"unknown keypoint scheme {keypoints!r}")

    source_placement = None
    if keypoints == "grasp_aligned":
        labels, source_placement, demo_ok = record_source_placement(
            source_seed, embodiment, control_freq
        )
        source_keypoints = KeypointSet(points=np.zeros((1, 3)), labels=["unused"])
    else:
        labels, source_keypoints, demo_ok = record_source(
            source_seed, embodiment, control_freq, keypoint_noise=keypoint_noise
        )
    if not demo_ok:
        raise RuntimeError(
            f"the source demonstration itself failed at seed {source_seed}; "
            "transporting a failed demonstration is meaningless"
        )
    if verbose:
        print(f"source demo: {len(labels)} labels, {len(source_keypoints)} keypoints")

    results = []
    for seed in target_seeds:
        result = transport_to_scene(
            labels,
            source_keypoints,
            seed,
            embodiment=embodiment,
            control_freq=control_freq,
            keypoint_noise=keypoint_noise,
            residual=residual,
            source_placement=source_placement,
        )
        results.append(result)
        if verbose:
            print(f"  target seed {seed:3d}: {result.summary()}")

    successes = [r.success for r in results]
    errors = [r.diagnostics["placement_error_xy"] for r in results]
    summary = {
        "n_episodes": len(results),
        "success_rate": float(np.mean(successes)),
        "placement_error_xy_mean": float(np.mean(errors)),
        "placement_error_xy_median": float(np.median(errors)),
        "keypoint_residual_max": float(
            max(r.diagnostics["keypoint_residual_max"] for r in results)
        ),
        "jacobian_fraction_positive_min": float(
            min(r.diagnostics["jacobian_fraction_positive"] for r in results)
        ),
        "keypoint_displacement_max": float(
            max(r.diagnostics["keypoint_displacement_max"] for r in results)
        ),
    }
    if verbose:
        print(
            f"\nsuccess {summary['success_rate'] * 100:.0f}% "
            f"({sum(successes)}/{len(results)}), "
            f"median placement error {summary['placement_error_xy_median'] * 1000:.1f} mm"
        )
    return results, summary


# ---------------------------------------------------------------------------
# Grasp-aligned keypoints (the generalisation to arbitrary objects)
# ---------------------------------------------------------------------------
#
# ``TRACKED_BODIES`` above reads two known boxes out of the simulator at their
# known half extent. That cannot survive a change of object, so the same scene
# is also expressible through the general extractor, which fits its box to the
# object's cloud and derives its axes from the grasp. Running the validated
# reshelving campaign through the general path is the acceptance test for it:
# the scheme it replaces is a special case, so the result must not move.


#: Where along the product's height the source demonstration grasps, as a
#: fraction from base (0) to top (1).
#:
#: The keypoint correspondence is *relative to each object's own box*, so a
#: demonstration that grips its object at the top transports to gripping the
#: target at the top -- whatever the target's height. That is fine for the
#: 9 cm reshelving box, whose fingers then straddle more than half of it, and
#: useless for a 15.3 cm cereal box, where the hand ends up 15.5 cm above the
#: base and closes on air.
#:
#: 1.0 reproduces the demonstration behind the validated 17/20; 0.5 grips the
#: middle, which is the height that transports sensibly to an object of any
#: size.
SOURCE_GRASP_FRACTION = 1.0


def reshelving_placement(
    env, n_points: int = 1200, seed: int = 0,
    grasp_fraction: float = SOURCE_GRASP_FRACTION,
):
    """Describe the current reshelving scene as an :class:`ObjectPlacement`.

    The cloud is sampled from the product's known geometry rather than
    rendered. Perception has its own acceptance tests; what is under test here
    is the keypoint construction, and a synthetic cloud isolates it. The bottom
    face is omitted because a camera never sees it -- which exercises the
    support snap of :func:`~tpgpt.sim.keypoints.fit_aligned_box` rather than
    hiding it.
    """
    obs = env._get_observations()
    position = np.asarray(obs["product_pos"], dtype=float)
    rotation = quat_to_matrix(np.asarray(obs["product_quat"]), scalar_first=False)[0]
    goal = np.asarray(env.goal_position, dtype=float)

    points = sample_box_surface(
        position,
        rotation,
        np.array(PRODUCT_HALF_SIZE),
        n=n_points,
        rng=np.random.default_rng(seed),
        faces="visible",
    )
    # The demonstration grasps at the top of the box with its yaw aligned to the
    # product (see reshelving_waypoints), so the keypoints must describe that
    # grasp and not a generic one.
    #
    # **The grasp is taken from the demonstrating hand, not from the product.**
    #
    # This used to be ``rotation[:, 0]``, the *product's* body x axis, on the
    # reasoning that the demonstration commands ``top_down_orientation(
    # product_yaw)`` so the jaws end up along it. The direction is right and the
    # **sign is not defined**: a body axis points wherever the mesh author put
    # it, and nothing about the object says which of the two fingers is on which
    # side. (It was ``rotation[:, 1]`` before that, and the reshelving result
    # never noticed, because that product is 25 mm square in cross-section so
    # both choices fit the same box.)
    #
    # An undefined sign here is what forced every consumer downstream to invent
    # one, and ``task_frame``'s world-axis test -- since deleted -- is the
    # invention that broke: the source's own pick and place frames came out 180
    # degrees apart and reflected every carried object through its grasp point.
    # 7.33.
    #
    # The hand's pose has no such problem. ``site_rotation`` is where the wrist
    # actually is, so converting it into the grasp convention with this hand's
    # measured ``alignment_rotation`` gives a closing axis whose sign is a fact
    # about the gripper rather than about the mesh -- the same convention every
    # GraspGen-X target candidate already arrives in. 7.34.
    offset = (2.0 * grasp_fraction - 1.0) * PRODUCT_HALF_SIZE[2]
    # Anchored at the point where the hand actually *holds* the object, not at
    # the commanded grip_site. The two differ by this gripper's measured contact
    # offset -- 41 mm on the Panda -- and the target side is anchored at the
    # holding point too (``GraspFrame.from_grasp`` uses ``tcp_position``). Mixing
    # the two reference points put the warped hand 31 to 49 mm from the grasp
    # the filters had chosen and verified.
    from tpgpt.grasp.grasps import contact_offset
    from tpgpt.grasp.grippers import resolve_pair
    from tpgpt.sim.demo import top_down_orientation

    site_rotation = top_down_orientation(np.arctan2(rotation[1, 0], rotation[0, 0]))
    holding = (
        position
        + np.array([0.0, 0.0, offset])
        + site_rotation @ contact_offset(resolve_pair("panda"))
    )
    from tpgpt.grasp.grasps import to_grasp_convention

    grasp_rotation = to_grasp_convention(site_rotation, resolve_pair("panda"))
    grasp = GraspFrame(
        tcp=holding,
        approach=grasp_rotation[:, 2],
        closing=grasp_rotation[:, 0],
    )
    # The plane the product rests on, not ``env.table_top``. That property adds
    # half the table thickness to ``table_offset``, but robosuite's TableArena
    # puts the *top* surface at ``table_offset`` and hangs the thickness below
    # it, so ``table_top`` reads 23 mm above where objects actually sit. Snapping
    # the box to it shortens the box, which moves the grasp keypoints down the
    # object and makes the transported gripper close above the product.
    return ObjectPlacement(
        points=points,
        grasp=grasp,
        support_height=float(position[2] - PRODUCT_HALF_SIZE[2]),
        destination=goal,
        destination_height=float(goal[2] - PRODUCT_HALF_SIZE[2]),
    )


def record_source_placement(
    source_seed: int = 0,
    embodiment: str = "panda",
    control_freq: int = 20,
    n_points: int = 1200,
    grasp_fraction: float = SOURCE_GRASP_FRACTION,
    capture: bool = False,
    camera: str = "sideview",
    camera_size: int = 512,
):
    """Record the source demonstration and describe its scene for the extractor.

    Args:
        grasp_fraction: See :data:`SOURCE_GRASP_FRACTION`. The demonstration and
            the keypoints are built from the same value, so they cannot drift
            apart.
        capture: Also keep a picture of the source scene and the camera matrix
            that projects into it, in ``placement.metadata["capture"]``.

            A report is supposed to show the keypoints on *both* scenes, and the
            source scene is torn down here -- it lives in its own environment,
            built and closed before any target scene exists. Without capturing
            it now there is nothing left to draw on later, and the source half
            of every report is an empty frame.
    """
    from tpgpt.sim.demo import pick_place_waypoints, product_yaw, top_down_orientation

    env = make_reshelving_env(
        embodiment=embodiment, control_freq=control_freq, seed=source_seed,
        offscreen=capture, render_size=camera_size, camera=camera,
    )
    try:
        env.reset()
        placement = reshelving_placement(
            env, n_points=n_points, seed=source_seed, grasp_fraction=grasp_fraction
        )
        obs = env._get_observations()
        offset = np.array([0.0, 0.0, (2.0 * grasp_fraction - 1.0) * PRODUCT_HALF_SIZE[2]])
        waypoints = pick_place_waypoints(
            np.asarray(obs["product_pos"]) + offset,
            top_down_orientation(product_yaw(obs)),
            np.asarray(env.goal_position) + offset,
            top_down_orientation(0.0),
        )
        if capture:
            placement.metadata["capture"] = _capture_scene(env, camera, camera_size)
        labels = record_demonstration(env, waypoints=waypoints)
        return labels, placement, bool(env._check_success())
    finally:
        env.close()


def _capture_scene(env, camera: str, size: int) -> dict:
    """A picture of a scene plus the matrix that projects world points into it.

    Both are needed together and neither survives ``env.close()``, so they are
    taken in one go while the environment is still alive.
    """
    from robosuite.utils import camera_utils

    return {
        "camera": camera,
        # Stored in display orientation, matching what a viewer sees, since the
        # observation buffer is upside down relative to the camera matrix.
        "image": np.ascontiguousarray(
            env.sim.render(width=size, height=size, camera_name=camera)[::-1]
        ),
        "matrix": np.asarray(
            camera_utils.get_camera_transform_matrix(env.sim, camera, size, size)
        ),
        "height": int(size),
        "width": int(size),
    }
