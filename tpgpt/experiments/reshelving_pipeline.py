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
from tpgpt.sim.demo import record_demonstration
from tpgpt.sim.keypoints import KeypointSet, keypoints_from_bodies, pair_keypoints
from tpgpt.sim.rollout import rollout_policy
from tpgpt.sim.scenes.reshelving import PRODUCT_HALF_SIZE
from tpgpt.transport.labels import PolicyLabels, transport_labels
from tpgpt.transport.maps import TransportMap

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
) -> TransportResult:
    """Transport the source policy into a new scene and execute it.

    Args:
        labels: Labels recorded in the source scene.
        source_keypoints: Keypoints of the source scene.
        target_seed: Seed selecting the new object pose, shelf height and slot.
        residual: Regressor for ``psi``; ``None`` gives an affine-only map,
            which is the natural ablation for the nonlinear stage.
    """
    env = make_reshelving_env(
        embodiment=embodiment,
        control_freq=control_freq,
        seed=target_seed,
        offscreen=video_path is not None,
    )
    try:
        env.reset()
        target_keypoints = extract_keypoints(env, keypoint_noise, seed=target_seed)
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
) -> tuple[list[TransportResult], dict]:
    """Transport one demonstration into many scenes and aggregate the outcome.

    This is the experiment Sec. V-A reports: a single demonstration generalised
    across randomised object and goal configurations.
    """
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
