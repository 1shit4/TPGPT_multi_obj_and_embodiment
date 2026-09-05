"""End-to-end policy transportation on the reshelving task (paper Sec. V-A).

One demonstration, recorded once, transported into new scenes and executed.
These exercise the whole chain: keypoint extraction, the transportation map, all
the label families, the policy refit, and execution through the Cartesian
impedance controller.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.slow]


@pytest.fixture(scope="module")
def source():
    from tpgpt.experiments.reshelving_pipeline import record_source

    labels, keypoints, succeeded = record_source(source_seed=0)
    assert succeeded, "the scripted demonstration must succeed before it is transported"
    return labels, keypoints


class TestDemonstration:
    def test_labels_cover_every_family(self, source):
        labels, _ = source
        for name in (
            "positions", "velocities", "orientations", "stiffness",
            "damping", "gripper", "time_belief", "time_rate",
        ):
            assert name in labels.present, name

    def test_timestamps_come_from_the_control_frequency(self, source):
        """The prototype timed a replay loop and got dt ~ 1e-5 s."""
        labels, _ = source
        assert labels.metadata["dt"] == pytest.approx(1.0 / labels.metadata["control_freq"])
        expected = labels.metadata["dt"] * (len(labels) - 1)
        assert labels.metadata["duration_s"] == pytest.approx(expected)

    def test_velocity_labels_are_not_degenerate(self, source):
        """The prototype's guard zeroed almost every velocity label.

        The 15% that are zero here are the two dwell segments -- grasping and
        releasing -- where the attractor genuinely holds still.
        """
        labels, _ = source
        speed = np.linalg.norm(labels.velocities, axis=1)
        assert (speed > 1e-4).mean() > 0.8
        assert 0.02 < speed.mean() < 1.0
        # Zero-velocity labels coincide with a closed or closing gripper.
        dwell = speed <= 1e-4
        assert dwell.sum() > 0
        assert dwell.mean() < 0.25

    def test_stiffness_profile_is_part_of_the_demonstration(self, source):
        """Compliant in free space, stiff while grasping and inserting."""
        labels, _ = source
        levels = np.unique(np.round(labels.stiffness[:, 0, 0], 3))
        assert len(levels) >= 2
        assert np.linalg.eigvalsh(labels.stiffness).min() > 0

    def test_keypoints_describe_both_objects(self, source):
        _, keypoints = source
        assert len(keypoints) == 18  # centre + 8 corners, for two objects
        assert not keypoints.is_degenerate()


class TestTransport:
    def test_transported_grasp_lands_on_the_new_object(self, source):
        """The core claim: the transported policy targets the *new* object.

        Checked at the label level, independently of whether execution then
        succeeds.
        """
        from tpgpt.experiments.reshelving_pipeline import extract_keypoints
        from tpgpt.sim.backend import managed_env
        from tpgpt.sim.demo import GRASP_HEIGHT_OFFSET
        from tpgpt.sim.keypoints import pair_keypoints
        from tpgpt.transport.labels import transport_labels
        from tpgpt.transport.maps import TransportMap

        labels, source_keypoints = source
        grasp_label, place_label = 64, 174  # ends of the grasp and release segments

        for seed in (1, 3, 5):
            with managed_env(seed=seed) as env:
                obs = env.reset()
                S, T = pair_keypoints(source_keypoints, extract_keypoints(env))
                transported = transport_labels(TransportMap().fit(S, T), labels)

            offset = np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
            grasp_error = np.linalg.norm(
                transported.positions[grasp_label] - (obs["product_pos"] + offset)
            )
            place_error = np.linalg.norm(
                transported.positions[place_label] - (obs["goal_pos"] + offset)
            )
            assert grasp_error < 0.015, f"seed {seed}: grasp off by {grasp_error:.4f} m"
            assert place_error < 0.015, f"seed {seed}: place off by {place_error:.4f} m"

    def test_keypoints_are_matched_and_the_map_stays_orientation_preserving(self, source):
        from tpgpt.experiments.reshelving_pipeline import transport_to_scene

        labels, keypoints = source
        result = transport_to_scene(labels, keypoints, target_seed=3)
        # Property (i) of Sec. III-C.
        assert result.diagnostics["keypoint_residual_max"] < 1e-4
        # Property (ii).
        assert result.diagnostics["jacobian_fraction_positive"] == 1.0
        assert result.diagnostics["jacobian_consistent_sign"]

    def test_the_nonlinear_stage_does_real_work(self, source):
        """Two objects moving differently cannot be matched by one rigid motion."""
        from tpgpt.experiments.reshelving_pipeline import transport_to_scene

        labels, keypoints = source
        result = transport_to_scene(labels, keypoints, target_seed=3)
        assert result.diagnostics["affine_residual_max"] > 0.01
        assert result.diagnostics["keypoint_residual_max"] < 1e-4

    def test_uncertainty_is_reported(self, source):
        """Sec. III-H: transport, epistemic and total velocity uncertainty."""
        from tpgpt.experiments.reshelving_pipeline import transport_to_scene

        labels, keypoints = source
        d = transport_to_scene(labels, keypoints, target_seed=3).diagnostics
        assert d["transport_velocity_std_mean"] >= 0
        assert d["epistemic_velocity_std_mean"] >= 0
        # Eq. (13) is a variance sum, so the total dominates each part.
        assert d["total_velocity_std_mean"] >= d["epistemic_velocity_std_mean"] - 1e-12
        assert d["total_velocity_std_mean"] >= d["transport_velocity_std_mean"] - 1e-12


class TestPolicyConditioning:
    def test_an_under_regularised_policy_cannot_complete_the_task(self, source):
        """The finding that determines whether execution works at all.

        A policy given a transportation map's priors -- near-zero likelihood
        noise, so it interpolates -- produces an explosive field just off the
        demonstration. It fails at every belief-correction gain, so the failure
        is the conditioning, not the phase handling.
        """
        from tpgpt.policy.gp_policy import GPPolicy
        from tpgpt.policy.rollout import rollout_free

        labels, _ = source
        goal = labels.positions[-1]

        def endpoint_error(policy, correction):
            rollout = rollout_free(
                policy, labels.positions[0], dt=labels.metadata["dt"],
                n_steps=900, belief_correction=correction,
            )
            return np.linalg.norm(rollout.positions[-1] - goal)

        sharp = GPPolicy(noise_variance=1e-6).fit(labels)
        smooth = GPPolicy().fit(labels)
        assert endpoint_error(smooth, 0.2) < 0.02
        assert min(endpoint_error(sharp, c) for c in (0.0, 0.2)) > 0.1


class TestExecution:
    def test_campaign_places_the_object_across_new_scenes(self, source):
        """A single demonstration, transported into eight randomised scenes."""
        from tpgpt.experiments.reshelving_pipeline import run_campaign

        results, summary = run_campaign(
            source_seed=0, target_seeds=tuple(range(1, 9)), verbose=False
        )
        assert summary["n_episodes"] == 8
        assert summary["keypoint_residual_max"] < 1e-4
        assert summary["jacobian_fraction_positive_min"] == 1.0
        # The scenes really are different from the source.
        assert summary["keypoint_displacement_max"] > 0.1
        assert summary["success_rate"] >= 0.5, summary
        placed = [r.diagnostics["placement_error_xy"] for r in results if r.success]
        assert np.median(placed) < 0.03


class TestAblation:
    def test_removing_the_nonlinear_stage_hurts(self, source):
        """The residual must earn its place, not merely be present.

        Two objects move independently between scenes, so no single rigid
        motion can match both; the affine stage alone leaves keypoint residual
        far larger than the grasp tolerance. Three scenes only -- the full
        20-scene result is in ROBOTICS_NOTES.md section 4.6.
        """
        from tpgpt.experiments.reshelving_pipeline import transport_to_scene

        labels, keypoints = source
        seeds = (1, 3, 5)
        full = [transport_to_scene(labels, keypoints, s) for s in seeds]
        affine = [transport_to_scene(labels, keypoints, s, residual=None) for s in seeds]

        assert max(r.diagnostics["keypoint_residual_max"] for r in full) < 1e-4
        assert max(r.diagnostics["keypoint_residual_max"] for r in affine) > 0.01
        assert sum(r.success for r in full) > sum(r.success for r in affine)
