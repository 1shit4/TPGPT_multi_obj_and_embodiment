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


# ---------------------------------------------------------------------------
# Execution arithmetic, shared between the robot and the surrogate plant.
#
# Everything below is a *verbatim* extraction from ``tpgpt.sim.rollout``. It
# lives here, in the policy layer, for one reason: this module imports only
# numpy, so a test bed that models the arm analytically can call exactly the
# same code the robot runs. The alternative -- writing the gate and the clamp a
# second time for the test bed -- is how an instrument stops matching the thing
# it measures, which cost this project 254 MB of deleted campaigns and three
# retracted findings (``ROBOTICS_NOTES`` 7.26).
#
# These are the most-debugged expressions in the project. Two of the three bugs
# recorded against them were in the *interaction* between them, not in any one
# alone: 7.14 has the clamp parking the attractor beyond the gate's own
# reopening threshold and deadlocking a run for 112 of 361 steps, and 7.16 has
# the watchdog testing ``gate == 0`` when a gate held at 0.05 was the fault. So
# the parenthesisation and the order of operations here are load-bearing, and
# the unit tests carry a transcription of the original expressions so that
# editing one of these without editing the test fails.
# ---------------------------------------------------------------------------


def compliance_norm(stiffness: np.ndarray, damping: np.ndarray) -> float:
    """Largest attractor offset per unit of commanded speed, ``||K^-1 D||``.

    Multiplying by a speed limit turns it into a displacement limit, so the
    clamp scales correctly with whatever stiffness the policy commanded.

    This is also the quantity that sets the impedance controller's steady-state
    lag. An arm tracking a setpoint that *moves* at speed ``v`` settles a
    distance ``||K^-1 D|| v`` behind it -- 24 mm at 0.25 m/s with this
    project's gains -- because the damper opposes absolute velocity rather than
    velocity error, so the spring has to stay stretched to supply the force.
    It is a property of tracking a ramp, not a settling error: a *parked*
    setpoint is reached exactly. See ``ROBOTICS_NOTES`` 2.7 and 2.8.
    """
    return float(np.linalg.norm(np.linalg.solve(stiffness, damping), ord=2))


def clamp_speed(velocity: np.ndarray, speed_limit: float) -> np.ndarray:
    """Cap a commanded speed, preserving its direction.

    Returns a **new** array. The original code mutated the prediction in place
    (``prediction.velocity[0] *= ...``) and three later reads depended on
    getting the clamped value back -- the gate's ``expected`` term, the
    attractor update, and the recorded ``velocities`` channel. A caller that
    keeps logging ``prediction.velocity[0]`` after switching to this function
    silently records the *unclamped* velocity, which is a change to recorded
    data that no success rate would reveal.
    """
    speed = float(np.linalg.norm(velocity))
    if speed > speed_limit:
        return velocity * (speed_limit / speed)
    return velocity


def lag_gate(
    position: np.ndarray,
    attractor: np.ndarray,
    expected: float,
    static_sag: float,
    lag_tolerance: float | None,
) -> tuple[float, float]:
    """How much progress the arm has earned, and how far behind it actually is.

    Returns ``(gate, behind)``. ``gate`` is 1 when the arm is keeping up and
    falls linearly to 0 once it is ``lag_tolerance`` further behind than the
    physics demands; it multiplies both the attractor's advance and the task
    clock, so a lagging arm cannot have the gripper act on a pose it has not
    reached.

    The subtlety is that it gates on **excess** lag. An impedance-controlled
    arm is *supposed* to trail its setpoint by ``expected = ||K^-1 D|| v``
    whenever it is moving, so penalising that would throttle every fast
    segment. Subtracting it means free motion runs at full speed while a dwell
    -- where the commanded velocity goes to zero, so ``expected`` does too --
    demands that the arm genuinely arrive.

    ``static_sag`` is lag the watchdog has decided cannot be closed at all: a
    held object's weight against a finite stiffness, for instance. Forgiving it
    is what stops the gate waiting forever for an arrival that is not coming.

    When ``lag_tolerance`` is falsy the gate is disabled and this returns
    ``(1.0, 0.0)``. The zero matters: the caller's watchdog computes
    ``blocked_lag - behind``, and a NaN there would silently make every
    comparison false, take the sag-rebaseline path three times, and report a
    stall where there is none.
    """
    if not lag_tolerance:
        return 1.0, 0.0
    behind = float(np.linalg.norm(position - attractor))
    excess = behind - expected - static_sag
    return float(np.clip(1.0 - excess / lag_tolerance, 0.0, 1.0)), behind


def clamp_attractor(
    attractor: np.ndarray,
    position: np.ndarray,
    compliance: float,
    speed_limit: float,
    expected: float,
    static_sag: float,
    lag_tolerance: float | None,
) -> tuple[np.ndarray, float, bool]:
    """Keep the attractor within reach of the arm.

    Returns ``(attractor, max_lag, clamped)``. In a Cartesian impedance
    controller the attractor *is* the reference, so the restoring force grows
    with the distance to it; an attractor allowed to run away from a blocked
    arm builds an unbounded force. This projects it back onto a sphere of
    radius ``max_lag`` centred on the **measured** position, which bounds the
    force while preserving the direction the policy asked for.

    **The clamp and the gate must agree on what "too far" means**, and for a
    long time they did not. The clamp allowed ``speed_limit * compliance``,
    computed from the *fastest* label, while the gate reopened only within
    ``expected + static_sag + lag_tolerance``, computed from the *current*
    speed. Whenever the first exceeded the second the clamp parked the
    attractor beyond the gate's own reopening threshold and the two locked: the
    gate shut, the frozen attractor was dragged along behind the arm instead of
    being caught up to, and the lag never shrank. Measured on a cereal-box run,
    the phase sat at 0.18 for 112 of 361 steps. Taking the smaller of the two
    makes that state unreachable by construction (7.14).

    Note that this runs **after** the attractor law, so it is part of the law's
    effective behaviour. A law that sets an absolute position -- ``reference``
    -- is therefore truncated at the arm's catch-up radius, which is close to
    the "recompute the attractor from the measured position" mode that
    ``tpgpt.sim.rollout``'s docstring records as failing. ``clamped`` is
    returned so a study can tell how much of a run was the clamp rather than
    the law.
    """
    lag = attractor - position
    max_lag = speed_limit * compliance
    if lag_tolerance:
        max_lag = min(max_lag, expected + static_sag + lag_tolerance)
    lag_norm = float(np.linalg.norm(lag))
    if lag_norm > max_lag:
        return position + lag * (max_lag / lag_norm), max_lag, True
    return attractor, max_lag, False


def steer_attractor(
    attractor: np.ndarray,
    target: np.ndarray,
    speed_limit: float,
    dt: float,
    arrival: float,
) -> tuple[np.ndarray, bool]:
    """Move the attractor toward a chosen pose instead of integrating the field.

    Returns ``(attractor, arrived)``. Used by the infeasible-pose fallback when
    the position the field is heading for does not exist for this arm: the
    attractor is walked toward the next label the arm *can* hold, capped at the
    demonstrated speed so the detour is a motion rather than a jump. The caller
    hands the task clock over on arrival, so the gripper schedule stays in step
    with where the hand actually is.
    """
    direction = target - attractor
    distance = float(np.linalg.norm(direction))
    if distance <= arrival:
        return attractor, True
    return attractor + direction / distance * min(speed_limit * dt, distance), False


def phase_epsilon_from(
    labels, dt: float, stall_patience: int, min_phase_progress: float
) -> float:
    """How much phase must be gained over a patience window to count as progress.

    Derived from the demonstration's own pace rather than fixed, so it scales
    with how long the task is instead of being a magic number: it is what the
    clock would gain over the window with the gate fully open, scaled down by
    ``min_phase_progress``.

    The reason this exists rather than a test for ``gate == 0``: a gate held
    slightly open by a constant sag is never exactly zero, but at 0.05 it
    advances the clock at a twentieth of nominal. Such a run neither deadlocks
    nor arrives -- it burns the whole step budget crawling, and three runs
    carried their object to within 5-30 mm of the slot and were scored failures
    because the clock never reached the segment that opens the fingers (7.16).
    """
    nominal_rate = (
        float(np.mean(labels.time_rate))
        if labels.time_rate is not None and len(labels.time_rate)
        else 1.0 / max(len(labels.positions), 1) / dt
    )
    return max(1e-6, min_phase_progress * nominal_rate * dt * stall_patience)


def stiffness_threshold(labels) -> float:
    """Compliance separating the demonstration's soft segments from its firm ones.

    Compliance is the inverse of stiffness, so a *large* value means the
    teacher was being soft. The midpoint of the demonstration's own range
    splits its free-space transit from its grasp and insertion without
    hard-coding a number, which is what lets the executor relax the hand's
    commanded angle in transit and hold it firmly at the two ends.
    """
    label_compliance = np.array([
        compliance_norm(K, D)
        for K, D in zip(labels.stiffness, labels.damping)
    ]) if labels.stiffness is not None else np.zeros(1)
    return float(0.5 * (label_compliance.min() + label_compliance.max()))
