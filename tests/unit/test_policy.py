"""Policy refit on transported labels -- paper Sec. III-B step 2, Sec. V."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.policy import GPPolicy, rollout_free, total_velocity_std
from tpgpt.transport.labels import PolicyLabels

DT = 1.0 / 20.0


@pytest.fixture
def pick_and_place_labels():
    """A demo that revisits the same (x, y) at two different heights.

    A pure ``xdot = f(x)`` policy cannot represent this: the descent and the
    ascent demand opposite velocities at the same position. It is the reason
    Sec. V feeds the policy a belief of time alongside the position.
    """
    m = 120
    s = np.linspace(0, 1, m)
    x = np.zeros((m, 3))
    x[:, 0] = 0.3 + 0.2 * np.clip((s - 0.35) / 0.3, 0, 1)
    x[:, 2] = np.piecewise(
        s,
        [s < 0.2, (s >= 0.2) & (s < 0.5), s >= 0.5],
        [lambda u: 0.95 - 0.5 * u, lambda u: 0.85 + 0.5 * (u - 0.2), lambda u: 1.0 - 0.5 * (u - 0.5)],
    )
    return PolicyLabels(
        positions=x,
        velocities=np.gradient(x, DT, axis=0),
        orientations=np.stack(
            [Rotation.from_euler("xyz", [np.pi, 0, 0.3 * t]).as_matrix() for t in s]
        ),
        stiffness=np.stack([np.diag([400 + 300 * t, 400 + 300 * t, 600.0]) for t in s]),
        damping=np.stack([np.diag([2 * np.sqrt(400 + 300 * t)] * 3) for t in s]),
        gripper=np.where(s < 0.35, -1.0, np.where(s < 0.75, 1.0, -1.0)),
        time_belief=s,
        time_rate=np.full(m, 1.0 / (DT * (m - 1))),
    )


@pytest.fixture
def policy(pick_and_place_labels):
    return GPPolicy().fit(pick_and_place_labels)


def test_reproduces_every_label_family(policy, pick_and_place_labels):
    """The policy regularises rather than interpolates, so this is a fit, not
    a reproduction. Velocity error is reported as a median: the worst-case
    error sits at the path's corners, where the demonstrated velocity is
    discontinuous and no smooth field can follow it.
    """
    lb = pick_and_place_labels
    pred = policy.predict(lb.positions, lb.time_belief)
    speed = np.linalg.norm(lb.velocities, axis=1).mean()
    error = np.linalg.norm(pred.velocity - lb.velocities, axis=1)
    assert np.median(error) < 0.1 * speed
    # Relative tolerances: the policy smooths, so a few N/m on a 700 N/m
    # stiffness is expected and harmless.
    assert np.abs(pred.stiffness - lb.stiffness).max() < 0.02 * lb.stiffness.max()
    assert np.abs(pred.damping - lb.damping).max() < 0.02 * lb.damping.max()
    # The gripper label is a square wave; a smooth field cannot reproduce its
    # steps, and does not need to -- execution thresholds the sign. What must
    # hold is that the sign is right everywhere except near the transitions.
    transition = np.abs(np.diff(lb.gripper, prepend=lb.gripper[0])) > 0
    near_transition = np.convolve(transition, np.ones(9), mode="same") > 0
    assert np.sign(pred.gripper[~near_transition]).tolist() == (
        lb.gripper[~near_transition].tolist()
    )
    geodesic = np.arccos(
        np.clip((np.trace(np.einsum("nij,nkj->nik", pred.orientation, lb.orientations),
                          axis1=1, axis2=2) - 1) / 2, -1, 1)
    )
    assert geodesic.max() < 0.01  # 0.6 degrees


def test_predicted_stiffness_and_damping_stay_positive_definite(policy, rng):
    """The log-Cholesky chart must hold even far outside the data."""
    x = rng.uniform(-2, 2, (30, 3))
    pred = policy.predict(x, rng.uniform(0, 1, 30))
    assert np.linalg.eigvalsh(pred.stiffness).min() > 0
    assert np.linalg.eigvalsh(pred.damping).min() > 0


def test_velocity_has_a_zero_prior_mean_far_from_the_data(policy):
    """Appendix A: the robot must not move without evidence from the human."""
    pred = policy.predict(np.array([[30.0, 30.0, 30.0]]), np.array([0.5]))
    assert np.allclose(pred.velocity, 0.0, atol=1e-9)


def test_non_velocity_channels_fall_back_to_their_mean_not_to_zero(
    policy, pick_and_place_labels
):
    """A zero-mean stiffness prior would make the robot go limp o.o.d."""
    pred = policy.predict(np.array([[30.0, 30.0, 30.0]]), np.array([0.5]))
    assert np.linalg.eigvalsh(pred.stiffness).min() > 100.0
    expected = pick_and_place_labels.stiffness.mean(axis=0)
    assert np.abs(np.linalg.eigvalsh(pred.stiffness)[0] - np.linalg.eigvalsh(expected)).max() < 50.0


def test_epistemic_uncertainty_grows_away_from_the_labels(policy, pick_and_place_labels):
    """This std is Sigma_f_hat, the epistemic term of Eq. (13)."""
    lb = pick_and_place_labels
    on = policy.predict(lb.positions, lb.time_belief).velocity_std.max()
    off = policy.predict(np.array([[3.0, 3.0, 3.0]]), np.array([0.5])).velocity_std.max()
    # The ratio is bounded by the likelihood noise, which is deliberately large
    # for a policy, so this is a growth check rather than a vanishing check.
    assert off > 5 * on


def test_rollout_reproduces_the_demonstration(policy, pick_and_place_labels):
    lb = pick_and_place_labels
    ro = rollout_free(policy, lb.positions[0], dt=DT, n_steps=400)
    assert ro.metadata["terminated_on_phase"]
    assert np.linalg.norm(ro.positions[-1] - lb.positions[-1]) < 0.02
    deviation = np.linalg.norm(
        ro.positions[:, None, :] - lb.positions[None, :, :], axis=2
    ).min(axis=1)
    assert deviation.max() < 0.02


def test_regularisation_bounds_the_field_off_the_demonstration(pick_and_place_labels):
    """A policy must not interpolate the way a transportation map does.

    Labels a few millimetres apart along a curve, far closer than the kernel
    length scale, make the Gram matrix near-singular; the field then stays well
    behaved *on* the demonstration and blows up just off it. The effect scales
    with how densely the path is sampled, so the guard here is the mechanism --
    a near-singular Gram matrix -- rather than a specific speed. Its
    consequence for a real 200-label demonstration is asserted end to end.
    """
    lb = pick_and_place_labels
    sharp = GPPolicy(noise_variance=1e-8).fit(lb)
    smooth = GPPolicy().fit(lb)

    def condition(policy):
        gp = policy.gp
        K = gp._kernel(gp.X_train, gp.X_train)
        K[np.diag_indices_from(K)] += gp.params.noise_variance + gp._applied_jitter
        return np.linalg.cond(K)

    assert condition(sharp) > 100 * condition(smooth)
    assert condition(smooth) < 1e6


def test_belief_correction_keeps_the_phase_tied_to_progress(policy, pick_and_place_labels):
    """A stalled robot must not let the phase run away from it."""
    lb = pick_and_place_labels
    stuck = lb.positions[10]
    open_loop = policy.update_time_belief(stuck, 0.5, DT, correction=0.0)
    corrected = policy.update_time_belief(stuck, 0.5, DT, correction=0.8)
    # Label 10 of 120 sits near phase 0.08, so the correction must pull back.
    assert open_loop > 0.5
    assert corrected < 0.5


def test_rollout_carries_every_channel(policy, pick_and_place_labels):
    ro = rollout_free(policy, pick_and_place_labels.positions[0], dt=DT, n_steps=400)
    assert ro.orientations.shape[1:] == (3, 3)
    assert ro.stiffness.shape[1:] == (3, 3)
    assert ro.gripper.shape == (len(ro),)
    assert ro.velocity_std.shape == (len(ro), 3)


def test_attractor_is_the_reference_plus_the_velocity_feedforward(
    policy, pick_and_place_labels
):
    """``x_desired = reference + K^-1 D xdot`` (Sec. V).

    The reference is what gives the policy a restoring action; the feed-forward
    is the steady state of the impedance law that realises the commanded speed.
    """
    lb = pick_and_place_labels
    x, t = lb.positions[:10], lb.time_belief[:10]
    pred = policy.predict(x, t)
    feedforward = np.einsum(
        "nij,njk,nk->ni", np.linalg.inv(pred.stiffness), pred.damping, pred.velocity
    )
    assert np.allclose(policy.attractor(x, t), pred.reference + feedforward, atol=1e-9)


def test_reference_tracks_the_demonstrated_path(policy, pick_and_place_labels):
    """On the demonstration the reference is the position, so the command
    reduces to the pure velocity feed-forward."""
    lb = pick_and_place_labels
    pred = policy.predict(lb.positions, lb.time_belief)
    assert np.abs(pred.reference - lb.positions).max() < 0.01


def test_total_uncertainty_is_the_variance_sum(policy, pick_and_place_labels):
    """Eq. (13)."""
    a = np.full((5, 3), 0.03)
    b = np.full((5, 3), 0.04)
    assert np.allclose(total_velocity_std(a, b), 0.05)


def test_policy_without_a_time_belief_still_fits(pick_and_place_labels):
    p = GPPolicy(use_time_belief=False).fit(pick_and_place_labels)
    assert p.predict(pick_and_place_labels.positions).velocity.shape == (120, 3)


def test_requires_velocity_labels():
    with pytest.raises(ValueError, match="velocity labels are required"):
        GPPolicy().fit(PolicyLabels(positions=np.zeros((4, 3))))


def test_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        GPPolicy().predict(np.zeros((1, 3)), np.zeros(1))


def test_rejects_missing_time_belief_at_predict(policy):
    with pytest.raises(ValueError, match="pass time_belief"):
        policy.predict(np.zeros((1, 3)))
