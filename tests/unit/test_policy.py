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
    lb = pick_and_place_labels
    pred = policy.predict(lb.positions, lb.time_belief)
    assert np.abs(pred.velocity - lb.velocities).max() < 1e-4
    assert np.abs(pred.stiffness - lb.stiffness).max() < 1e-2
    assert np.abs(pred.damping - lb.damping).max() < 1e-2
    assert np.abs(pred.gripper - lb.gripper).max() < 1e-2
    geodesic = np.arccos(
        np.clip((np.trace(np.einsum("nij,nkj->nik", pred.orientation, lb.orientations),
                          axis1=1, axis2=2) - 1) / 2, -1, 1)
    )
    assert geodesic.max() < 1e-4


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
    assert off > 100 * on


def test_rollout_reproduces_the_demonstration(policy, pick_and_place_labels):
    lb = pick_and_place_labels
    ro = rollout_free(policy, lb.positions[0], dt=DT, n_steps=400)
    assert ro.metadata["terminated_on_phase"]
    assert np.linalg.norm(ro.positions[-1] - lb.positions[-1]) < 0.02
    deviation = np.linalg.norm(
        ro.positions[:, None, :] - lb.positions[None, :, :], axis=2
    ).min(axis=1)
    assert deviation.max() < 0.02


def test_open_loop_time_belief_stalls_the_rollout(policy, pick_and_place_labels):
    """Regression guard for a failure that is easy to reintroduce.

    Without the belief correction, integration lag pushes the query off the
    training ridge, the zero-mean prior decays the velocity to zero and the
    rollout stops short. See GPPolicy.update_time_belief.
    """
    lb = pick_and_place_labels
    open_loop = rollout_free(policy, lb.positions[0], dt=DT, n_steps=400, belief_correction=0.0)
    closed_loop = rollout_free(policy, lb.positions[0], dt=DT, n_steps=400)
    goal = lb.positions[-1]
    assert np.linalg.norm(open_loop.positions[-1] - goal) > 0.1
    assert np.linalg.norm(closed_loop.positions[-1] - goal) < 0.02


def test_rollout_carries_every_channel(policy, pick_and_place_labels):
    ro = rollout_free(policy, pick_and_place_labels.positions[0], dt=DT, n_steps=400)
    assert ro.orientations.shape[1:] == (3, 3)
    assert ro.stiffness.shape[1:] == (3, 3)
    assert ro.gripper.shape == (len(ro),)
    assert ro.velocity_std.shape == (len(ro), 3)


def test_attractor_offset_realises_the_commanded_velocity(policy, pick_and_place_labels):
    """x_desired = x + K^-1 D xdot is the steady state of the impedance law."""
    lb = pick_and_place_labels
    x, t = lb.positions[:10], lb.time_belief[:10]
    pred = policy.predict(x, t)
    offset = policy.attractor(x, t) - x
    restoring = np.einsum("nij,nj->ni", pred.stiffness, offset)
    damping_force = np.einsum("nij,nj->ni", pred.damping, pred.velocity)
    assert np.allclose(restoring, damping_force, atol=1e-6)


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
