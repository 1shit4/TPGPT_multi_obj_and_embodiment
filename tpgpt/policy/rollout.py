"""Rolling out a fitted policy (paper Sec. III-B, step 2).

Two rollouts are useful and they answer different questions.

:func:`rollout_free` integrates ``xdot = g(x, t)`` with no robot in the loop. It
answers "does the fitted policy reproduce the intended motion?" and is what the
2-D theory figures use.

Executing on the robot lives in :mod:`tpgpt.sim.rollout`, which closes the loop
through the Cartesian impedance controller and therefore also exercises the
transported stiffness and damping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.transport.uncertainty import total_variance, variance_to_std


@dataclass
class Rollout:
    """A trajectory produced by integrating a policy."""

    positions: np.ndarray                     # (T, 3)
    velocities: np.ndarray                    # (T, 3)
    time_belief: np.ndarray                   # (T,)
    velocity_std: np.ndarray                  # (T, 3) -- epistemic, Sigma_f_hat
    orientations: np.ndarray | None = None    # (T, 3, 3)
    stiffness: np.ndarray | None = None       # (T, 3, 3)
    damping: np.ndarray | None = None         # (T, 3, 3)
    gripper: np.ndarray | None = None         # (T,)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.positions.shape[0]


def rollout_free(
    policy,
    start_position: np.ndarray,
    dt: float,
    n_steps: int = 500,
    start_time_belief: float = 0.0,
    stop_time_belief: float = 1.0,
    max_speed: float | None = None,
    belief_correction: float = 0.2,
) -> Rollout:
    """Integrate ``xdot = g(x, t)`` forward with explicit Euler.

    The phase is propagated by
    :meth:`~tpgpt.policy.gp_policy.GPPolicy.update_time_belief`, which advances
    it by the policy's own predicted rate and then reconciles it with the
    position actually reached. Open-loop integration of the rate stalls the
    rollout; see that method for the failure mode.

    Args:
        policy: A fitted :class:`~tpgpt.policy.gp_policy.GPPolicy`.
        start_position: ``(3,)`` initial position.
        dt: Integration step, normally ``1 / control_freq``.
        n_steps: Maximum number of steps.
        start_time_belief: Initial phase.
        stop_time_belief: Integration stops once the phase reaches this value.
        max_speed: Optional speed clamp, a safety net against a policy queried
            far outside its data.
        belief_correction: How strongly the phase is reconciled with the
            observed position each step. ``0`` runs the phase open loop.

    Returns:
        A :class:`Rollout`.
    """
    x = np.asarray(start_position, dtype=float).reshape(1, 3).copy()
    t = float(start_time_belief)

    positions, velocities, phases, stds = [], [], [], []
    orientations, stiffness, damping, grippers = [], [], [], []

    for _ in range(n_steps):
        pred = policy.predict(x, np.array([t]) if policy.use_time_belief else None)
        v = pred.velocity[0]
        if max_speed is not None:
            speed = np.linalg.norm(v)
            if speed > max_speed:
                v = v * (max_speed / speed)

        positions.append(x[0].copy())
        velocities.append(v.copy())
        phases.append(t)
        stds.append(pred.velocity_std[0].copy())
        if pred.orientation is not None:
            orientations.append(pred.orientation[0])
        if pred.stiffness is not None:
            stiffness.append(pred.stiffness[0])
        if pred.damping is not None:
            damping.append(pred.damping[0])
        if pred.gripper is not None:
            grippers.append(float(pred.gripper[0]))

        x = x + v * dt
        t = policy.update_time_belief(x[0], t, dt, correction=belief_correction)
        if t >= stop_time_belief:
            break

    def _stack(seq):
        return np.stack(seq) if seq else None

    return Rollout(
        positions=np.stack(positions),
        velocities=np.stack(velocities),
        time_belief=np.array(phases),
        velocity_std=np.stack(stds),
        orientations=_stack(orientations),
        stiffness=_stack(stiffness),
        damping=_stack(damping),
        gripper=np.array(grippers) if grippers else None,
        metadata={"dt": dt, "terminated_on_phase": bool(t >= stop_time_belief)},
    )


def total_velocity_std(
    policy_velocity_std: np.ndarray, transport_velocity_std: np.ndarray
) -> np.ndarray:
    """Eq. (13): combine epistemic and transport uncertainty into a total.

    Args:
        policy_velocity_std: ``Sigma_f_hat`` from the refitted policy.
        transport_velocity_std: ``Sigma_x_hat`` from Eq. (12).
    """
    return variance_to_std(
        total_variance(
            np.asarray(policy_velocity_std, dtype=float) ** 2,
            np.asarray(transport_velocity_std, dtype=float) ** 2,
        )
    )
