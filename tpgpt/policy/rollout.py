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
    #: Populated only by :func:`rollout_impedance`, which has an arm and
    #: therefore has an attractor distinct from the arm's own position.
    #: :func:`rollout_free` leaves these ``None`` -- it integrates the policy
    #: directly, so there is nothing for the arm to lag behind.
    attractors: np.ndarray | None = None      # (T, 3)
    gate: np.ndarray | None = None            # (T,)
    lag: np.ndarray | None = None             # (T,) ||position - attractor||
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


# ---------------------------------------------------------------------------
# The attractor law: what actually moves the point the arm is chasing.
# ---------------------------------------------------------------------------

#: The three laws. ``reference`` is not a separate branch -- it is ``anchor``
#: with a gain of 1, since anchoring all the way discards the integration step.
#: Keeping the name is worth it because a manifest that says ``reference`` is
#: clearer than one that says ``anchor`` at ``k=1.0``.
LAWS = ("integrate", "anchor", "reference")

#: Where the policy is queried.
QUERY_SITES = ("attractor", "measured")


@dataclass(frozen=True)
class AnchorSchedule:
    """Anchor gain as a function of the speed the policy is commanding.

    A constant gain is the obvious thing and it is measurably the wrong thing.
    Anchoring pays where the policy commands a **dwell**: the velocity goes to
    zero, the integrator has nothing left to say, and the regressed reference is
    the only absolute statement about where the hand should be. In transit the
    reference is a smoothed version of a path the integrator is already
    following well, so pulling toward it fights the feed-forward and flattens
    the demonstrated speed profile.

    So: strong when slow, weak when moving.

        ``gain(s) = transit + (dwell - transit) * exp(-s / speed_scale)``

    ``s`` is the commanded speed as a **fraction of the demonstration's fastest
    label**, which is the same normalisation ``speed_limit_factor`` already
    uses. That makes a swept value mean the same thing on a 9 cm reshelving box
    as on a 15 cm cereal box, instead of silently changing meaning with the
    scene.

    A dataclass rather than a callable on purpose: ``ROBOTICS_NOTES`` 7.26
    rule 2 requires the complete settings dictionary in every manifest, and a
    lambda serialises as ``<function <lambda> at 0x...>``, which is how a
    campaign becomes unattributable.
    """

    dwell: float = 0.0
    transit: float = 0.0
    speed_scale: float = 0.2

    def gain(self, normalised_speed: float) -> float:
        s = max(0.0, float(normalised_speed))
        return float(
            self.transit + (self.dwell - self.transit) * np.exp(-s / self.speed_scale)
        )

    def to_dict(self) -> dict:
        return {
            "kind": "schedule",
            "dwell": self.dwell,
            "transit": self.transit,
            "speed_scale": self.speed_scale,
        }


def anchor_gain_at(gain, normalised_speed: float) -> float:
    """Resolve a gain that may be a constant or a schedule."""
    if isinstance(gain, AnchorSchedule):
        return gain.gain(normalised_speed)
    return float(gain)


def advance_attractor(
    attractor: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    gate: float,
    *,
    law: str = "integrate",
    reference: np.ndarray | None = None,
    anchor_gain: float = 0.0,
    anchor_gated: bool = True,
) -> np.ndarray:
    """Move the attractor one control step.

    ``integrate`` is the shipped behaviour: ``a + v*dt*gate``. It is a pure
    dead-reckoning of the learned velocity field, and because the field is
    fitted with a zero prior mean its arrows point *along* the demonstration
    and never back toward it -- so nothing corrects an accumulated error.

    ``anchor`` integrates and then pulls a fraction ``k`` of the way toward
    ``reference``, the policy's regressed attractor-position channel. That
    channel is the paper's own Sec. V formulation and the only restoring term
    the policy has; measured on synthetic labels it holds to 3.1 mm across a
    dwell where the integrator drifts 11.4 mm.

    ``reference`` is ``anchor`` at ``k = 1``: the integration is discarded and
    the attractor is the regressed position outright, so there is no integrator
    and therefore no integration drift by construction.

    **``anchor_gated`` is not a detail.** The velocity term is multiplied by
    ``gate``; whether the anchor term is too changes what the law means. Gated,
    the anchor goes inert exactly when the arm has fallen behind -- which is
    when a restoring term is most wanted. Ungated, the attractor keeps being
    pulled toward the reference while the gate is shut, which defeats the gate's
    purpose of holding progress until the robot catches up. Neither is
    obviously right, so it is an explicit parameter and both settings get swept.

    Args:
        reference: ``(3,)`` regressed attractor position. Required for any law
            other than ``integrate``.

    Raises:
        ValueError: on an unknown law, or on a law that needs ``reference``
            when the policy was fitted without that channel. Falling back to
            integrating would be a silent downgrade -- the same shape as the
            by-identity lookup that returned a confident zero for an unmeasured
            gripper and cost weeks (7.13, 7.27).
    """
    if law not in LAWS:
        raise ValueError(f"unknown attractor law {law!r}; known: {LAWS}")
    stepped = attractor + velocity * dt * gate
    if law == "integrate":
        return stepped
    if reference is None:
        raise ValueError(
            f"attractor law {law!r} needs the policy's `reference` channel, and "
            "this prediction has none. Fit the policy with the attractor "
            "channel, or use law='integrate' -- silently integrating instead "
            "would hide the misconfiguration."
        )
    k = 1.0 if law == "reference" else float(anchor_gain)
    if anchor_gated:
        k = k * gate
    if k == 0.0:
        return stepped
    return (1.0 - k) * stepped + k * np.asarray(reference, dtype=float)


def query_position(attractor: np.ndarray, position: np.ndarray, query_at: str) -> np.ndarray:
    """Where to evaluate the policy: the attractor, or the measured pose.

    Querying at the **attractor** keeps the query on the ridge of data the
    policy was fitted on, which matters because the velocity channel has a zero
    prior mean: far from the labels it returns *no motion at all*, so an arm
    that strays stops rather than recovering. It also keeps position and phase
    consistent -- the policy was trained only on pairs that actually co-occur,
    and feeding it a lagging position with the attractor's phase asks about a
    combination that never happened.

    Querying at the **measured** pose is the honest dynamical system: a policy
    ``xdot = g(x, t)`` is a vector field and closing the loop means evaluating
    it where you are. It is safe in proportion to how good the restoring term
    is, which is why this pairs with the anchor laws rather than standing alone.

    See ``tpgpt/sim/rollout.py``'s module docstring for the measurements behind
    the default, and 7.26 for why the alternative is selectable rather than
    argued about.
    """
    if query_at not in QUERY_SITES:
        raise ValueError(f"unknown query site {query_at!r}; known: {QUERY_SITES}")
    return attractor if query_at == "attractor" else position


def belief_position(attractor: np.ndarray, position: np.ndarray, query_at: str) -> np.ndarray:
    """Where to advance the phase from.

    Deliberately a second function rather than a reuse of
    :func:`query_position`, and the reason is subtle enough to be worth stating:
    the policy is queried with the attractor **before** the law runs, while the
    phase is advanced from the attractor **after** the law and the clamp. Those
    are one step apart. Unifying them is a behaviour change with a measured
    cost -- driving the phase from a lagging position while querying the policy
    at the attractor pairs a position with a phase that never occur together in
    the labels, and measured that way the tracking error sat at 30-60 mm
    regardless of gains and the gripper closed 2-4 cm short every time (2.8).

    So ``query_at`` selects the *kind* of point and each site evaluates that
    kind at its own moment. For ``measured`` both give the same value, because
    the arm is not re-measured inside a step.
    """
    return query_position(attractor, position, query_at)


@dataclass(frozen=True)
class ExecutionLaw:
    """Everything the gate, the clamp and the attractor law depend on.

    Built once per rollout and handed to every step, so a knob added here
    cannot be threaded into the shipped rollout and forgotten in the
    physics-free one -- which is exactly how an instrument stops measuring the
    thing it is aimed at (7.26).
    """

    dt: float
    speed_limit: float
    max_label_speed: float
    lag_tolerance: float | None = 0.035
    attractor_law: str = "integrate"
    anchor_gain: float | AnchorSchedule = 0.0
    anchor_gated: bool = True
    query_at: str = "attractor"
    skip_arrival: float = 0.02

    def __post_init__(self):
        if self.attractor_law not in LAWS:
            raise ValueError(
                f"unknown attractor law {self.attractor_law!r}; known: {LAWS}"
            )
        if self.query_at not in QUERY_SITES:
            raise ValueError(
                f"unknown query site {self.query_at!r}; known: {QUERY_SITES}"
            )

    def gain_at(self, speed: float) -> float:
        """Anchor gain for a commanded speed, normalised by the fastest label."""
        reference_speed = max(self.max_label_speed, 1e-12)
        return anchor_gain_at(self.anchor_gain, float(speed) / reference_speed)

    def to_dict(self) -> dict:
        gain = self.anchor_gain
        return {
            "attractor_law": self.attractor_law,
            "anchor_gain": gain.to_dict() if isinstance(gain, AnchorSchedule) else float(gain),
            "anchor_gated": bool(self.anchor_gated),
            "query_at": self.query_at,
            "lag_tolerance": self.lag_tolerance,
            "speed_limit": self.speed_limit,
            "max_label_speed": self.max_label_speed,
            "skip_arrival": self.skip_arrival,
        }


@dataclass(frozen=True)
class GateDecision:
    """What the gate concluded, before the attractor is moved."""

    velocity: np.ndarray
    gate: float
    behind: float
    expected: float
    compliance: float


@dataclass(frozen=True)
class AttractorStep:
    """One control step's decisions, before anything is commanded."""

    attractor: np.ndarray
    velocity: np.ndarray
    gate: float
    behind: float
    expected: float
    compliance: float
    max_lag: float
    clamped: bool
    anchor_gain: float
    skip_arrived: bool


def gate_step(
    position: np.ndarray,
    attractor: np.ndarray,
    prediction,
    law: ExecutionLaw,
    *,
    static_sag: float = 0.0,
) -> GateDecision:
    """Clamp the commanded speed and decide how much progress is allowed.

    Split from :func:`advance_step` because the robot's infeasible-pose
    fallback has to run *between* them: it asks whether the arm can hold the
    commanded orientation, and it is consulted only while the gate is already
    partly shut, so a healthy run pays nothing for it.
    """
    velocity = clamp_speed(prediction.velocity[0], law.speed_limit)
    compliance = compliance_norm(prediction.stiffness[0], prediction.damping[0])
    expected = (
        compliance * float(np.linalg.norm(velocity)) if law.lag_tolerance else 0.0
    )
    gate, behind = lag_gate(
        position, attractor, expected, static_sag, law.lag_tolerance
    )
    return GateDecision(velocity, gate, behind, expected, compliance)


def advance_step(
    position: np.ndarray,
    attractor: np.ndarray,
    decision: GateDecision,
    law: ExecutionLaw,
    prediction=None,
    *,
    static_sag: float = 0.0,
    steer_to: np.ndarray | None = None,
) -> AttractorStep:
    """Move the attractor, then clamp it back within reach of the arm.

    The clamp runs **after** the law in both branches, which is what it did
    before extraction and what keeps the 7.14 deadlock unreachable. That
    ordering also means the clamp is part of every law's effective behaviour: a
    law that sets an absolute position is truncated at the arm's catch-up
    radius, so ``clamped`` is reported for any study that needs to know whether
    it measured the law or the clamp.
    """
    reference = None
    if prediction is not None and getattr(prediction, "reference", None) is not None:
        reference = prediction.reference[0]

    gain = law.gain_at(float(np.linalg.norm(decision.velocity)))
    arrived = False
    if steer_to is not None:
        attractor, arrived = steer_attractor(
            attractor, steer_to, law.speed_limit, law.dt, law.skip_arrival
        )
    else:
        attractor = advance_attractor(
            attractor, decision.velocity, law.dt, decision.gate,
            law=law.attractor_law, reference=reference,
            anchor_gain=gain, anchor_gated=law.anchor_gated,
        )
    attractor, max_lag, clamped = clamp_attractor(
        attractor, position, decision.compliance, law.speed_limit,
        decision.expected, static_sag, law.lag_tolerance,
    )
    return AttractorStep(
        attractor=attractor, velocity=decision.velocity, gate=decision.gate,
        behind=decision.behind, expected=decision.expected,
        compliance=decision.compliance, max_lag=max_lag, clamped=clamped,
        anchor_gain=gain, skip_arrived=arrived,
    )


def attractor_step(
    position: np.ndarray,
    attractor: np.ndarray,
    prediction,
    law: ExecutionLaw,
    *,
    static_sag: float = 0.0,
    steer_to: np.ndarray | None = None,
) -> AttractorStep:
    """The whole per-step decision: gate, then law, then clamp.

    The composition exists so that a caller which does not need the
    infeasibility fallback -- the physics-free bed -- cannot get the *order*
    wrong. Building an instrument out of the same parts as the system and still
    assembling it differently would be 7.26's failure with extra steps, and
    nothing would fail to warn about it. A unit test asserts this equals
    ``advance_step(gate_step(...))`` bit-for-bit.
    """
    decision = gate_step(position, attractor, prediction, law, static_sag=static_sag)
    return advance_step(
        position, attractor, decision, law, prediction,
        static_sag=static_sag, steer_to=steer_to,
    )


@dataclass
class StallWatchdog:
    """Decides when a run has stopped making progress, and when to forgive lag.

    Progress is measured on the **phase**, never on the gate being exactly
    zero. A gate held slightly open by a constant sag is not zero, but at 0.05
    it advances the clock at a twentieth of nominal: such a run neither
    deadlocks nor arrives, it burns the whole step budget crawling. Three runs
    carried their object to within 5-30 mm of its slot and were scored failures
    because the clock never reached the segment that opens the fingers, and all
    three were reported as a stalled policy -- blaming the transport for a gate
    doing its job too well (7.16).

    ``blocked_lag`` starts at infinity so the first window is always read as
    "still catching up" and spends one more window measuring before forgiving
    any lag as sag.
    """

    stall_patience: int = 40
    sag_epsilon: float = 0.002
    max_sag_rebaselines: int = 3
    phase_epsilon: float = 1e-6
    blocked_for: int = 0
    blocked_phase: float = 0.0
    blocked_lag: float = float("inf")
    static_sag: float = 0.0
    rebaselines: int = 0
    stalled: bool = False

    def update(self, phase: float, behind: float, expected: float) -> bool:
        """``True`` when the run should stop. May move the sag baseline."""
        self.blocked_for += 1
        if phase - self.blocked_phase >= self.phase_epsilon:
            self.blocked_for, self.blocked_phase, self.blocked_lag = 0, phase, behind
        elif self.blocked_for >= self.stall_patience:
            if float(self.blocked_lag - behind) >= self.sag_epsilon:
                # Still closing the gap -- which is exactly what the gate is
                # for. Keep waiting.
                self.blocked_for, self.blocked_lag = 0, behind
            elif self.rebaselines < self.max_sag_rebaselines:
                # Not catching up: this offset is load the arm cannot pull out
                # of a finite stiffness, so stop counting it against progress.
                self.static_sag = max(0.0, behind - expected)
                self.rebaselines += 1
                self.blocked_for, self.blocked_phase = 0, phase
            else:
                self.stalled = True
        return self.stalled


def advance_phase(
    policy, point: np.ndarray, phase: float, dt: float, gate: float,
    belief_correction: float,
) -> float:
    """Advance the task clock, throttled by the same gate as the attractor.

    Both are scaled so a lagging arm cannot let the schedule run away from it:
    without that the gripper acts on time while the hand is still centimetres
    short.
    """
    if gate <= 0.0:
        return phase
    return policy.update_time_belief(
        point, phase, dt * gate, correction=belief_correction * gate
    )


def rollout_impedance(
    policy,
    dt: float,
    start_position: np.ndarray | None = None,
    n_steps: int = 400,
    belief_correction: float = 0.2,
    stop_time_belief: float = 1.0,
    speed_limit_factor: float = 3.0,
    lag_tolerance: float | None = 0.035,
    attractor_law: str = "integrate",
    anchor_gain: float | AnchorSchedule = 0.0,
    anchor_gated: bool = True,
    query_at: str = "attractor",
    stall_patience: int = 40,
    sag_epsilon: float = 0.002,
    max_sag_rebaselines: int = 3,
    min_phase_progress: float = 0.25,
    substeps: int = 1,
    load: np.ndarray | None = None,
    blocked_steps: tuple[int, int] | None = None,
) -> "Rollout":
    """Run the real execution logic against an analytic arm instead of MuJoCo.

    The third of three rollouts, and the reason it exists is cost.
    :func:`rollout_free` integrates the policy with no robot at all, and
    :func:`tpgpt.sim.rollout.rollout_policy` runs the real thing at roughly
    25-40 seconds a go. Comparing execution laws needs hundreds of runs across
    a sweep of warp aggressiveness, gains and query sites, which the real
    rollout cannot afford. This sits between them: the gate, the clamp, the
    attractor law, the phase update and the stall watchdog are the **same
    functions the robot runs**, and only the arm is replaced.

    **The arm model, and why it is the right one.** An impedance-controlled arm
    that is tracking and not in contact reduces to a first-order system,

        ``xdot = D^-1 K (a - x)``

    integrated here with explicit Euler over ``substeps`` sub-intervals. Two
    properties make it trustworthy rather than merely fast, and both are
    asserted in the unit tests:

    * A **parked** setpoint is reached exactly, which is the property
      ``ROBOTICS_NOTES`` 2.7 establishes for the real controller: the lag is a
      consequence of the setpoint *moving*, not a settling error. A bed that
      missed this would be modelling the wrong thing.
    * Following a setpoint that moves at constant speed ``v``, the lag settles
      at a value known in closed form. The attractor is held fixed for a whole
      control step -- a zero-order hold, exactly as the real robot holds an
      action while MuJoCo integrates internally -- so the gap shrinks by
      ``(1 - A dt/m)^m`` per step and then grows by ``v dt``:

          ``L* = v dt / (1 - (1 - A dt / m)^m)``

      At ``m = 1`` that reduces to ``A^-1 v = K^-1 D v``, which is **exactly
      the ``expected`` term the lag gate subtracts**. As ``m`` rises it tends
      to ``v dt / (1 - exp(-dt / tau))``.

    **``substeps`` therefore changes the steady state, not only the
    transient**, and the default of 1 is a deliberate choice rather than a
    cheap one: it makes the arm's lag match the gate's own model of it, so the
    gate behaves in the bed the way its arithmetic says it should.

    That gap is worth noticing in its own right, because it is a property of
    the **shipped** system and not of this bed. The real plant is the ``m`` to
    infinity case, so its steady lag exceeds the gate's ``expected`` by
    ``(dt/tau) / (1 - exp(-dt/tau))`` -- a factor of **1.28** at this project's
    ``dt = 50 ms`` and ``tau = 96 ms``. The gate therefore sees a standing
    excess of about 4.6 mm at full demonstrated speed, roughly an eighth of its
    35 mm tolerance, before anything has actually gone wrong. Running the sweep
    at both ``substeps=1`` and a larger value brackets that effect.

    **What this cannot see, stated so no result overreaches.** No contact, so
    nothing about grasping or collision. No inverse kinematics, so nothing
    about the roughly quarter of a trajectory that is unreachable at its
    commanded orientation (7.25). No orientation task competing for the arm's
    effort. It therefore **ranks hypotheses; it does not confirm them** --
    7.26 rule 4 read in the direction it was written.

    It also deliberately reports **no success flag and no placement error.**
    Inventing one from the distance to the last label would be exactly the
    scalar-proxy mistake that 7.27 documents at length, made on purpose.

    Args:
        policy: A fitted :class:`~tpgpt.policy.gp_policy.GPPolicy`.
        dt: Control period. Normally ``1 / control_freq``.
        start_position: Where the arm and the attractor both begin. Defaults to
            the first label, which is where the real rollout's approach phase
            puts the arm before handing over.
        substeps: Euler sub-intervals per control step. Raise to check that a
            result does not depend on the integration transient.
        load: A constant ``(3,)`` velocity the arm cannot overcome -- gravity on
            a held object, or a push. Without something like this the surrogate
            arm tracks almost perfectly, the lag never exceeds what the physics
            predicts, and **the gate and the clamp are barely exercised**: on
            the undisturbed bed they engage on 2% and 0% of steps respectively.
            Since the whole reason the anchor law is interesting is how it
            interacts with those two, a bed that cannot make them fire cannot
            answer the question. This is how to make them fire without a robot.
        blocked_steps: ``(first, last)`` control steps over which the arm does
            not move at all, whatever it is commanded -- an obstruction. This
            drives the stall watchdog and the sag re-baselining, and it is the
            configuration in which the 7.14 clamp/gate deadlock would appear if
            a new law reintroduced it.

    Returns:
        A :class:`Rollout` whose ``attractors``, ``gate`` and ``lag`` channels
        are populated, with metadata keyed the same way
        :class:`~tpgpt.sim.rollout.SimRollout`'s is so one analysis reads both.

    Raises:
        ValueError: if ``D^-1 K dt`` is too large for explicit Euler to be
            stable. Silently producing an oscillating arm would make every
            number downstream meaningless.
    """
    if policy.labels is None:
        raise ValueError("rollout_impedance needs a fitted policy")
    labels = policy.labels
    speed_limit = speed_limit_factor * float(
        np.linalg.norm(labels.velocities, axis=1).max()
    )
    law = ExecutionLaw(
        dt=dt, speed_limit=speed_limit,
        max_label_speed=speed_limit / max(speed_limit_factor, 1e-12),
        lag_tolerance=lag_tolerance, attractor_law=attractor_law,
        anchor_gain=anchor_gain, anchor_gated=anchor_gated, query_at=query_at,
    )
    watchdog = StallWatchdog(
        stall_patience=stall_patience, sag_epsilon=sag_epsilon,
        max_sag_rebaselines=max_sag_rebaselines,
        phase_epsilon=phase_epsilon_from(labels, dt, stall_patience, min_phase_progress),
    )

    start = (
        np.asarray(labels.positions[0], dtype=float)
        if start_position is None
        else np.asarray(start_position, dtype=float)
    )
    attractor, x, phase = start.copy(), start.copy(), 0.0
    positions, attractors, velocities, phases, stds = [], [], [], [], []
    orientations, stiffness, damping, grippers, gates, lags = [], [], [], [], [], []
    clamped_steps = 0
    stalled = False

    for _ in range(n_steps):
        query_phase = np.array([phase]) if policy.use_time_belief else None
        prediction = policy.predict(
            query_position(attractor, x, law.query_at)[None], query_phase
        )
        step_out = attractor_step(
            x, attractor, prediction, law, static_sag=watchdog.static_sag
        )

        K = np.atleast_2d(prediction.stiffness[0])
        D = np.atleast_2d(prediction.damping[0])
        A = np.linalg.solve(D, K)
        spectral = float(np.linalg.norm(np.eye(len(A)) - A * (dt / substeps), ord=2))
        if spectral >= 1.0:
            raise ValueError(
                f"explicit Euler is unstable here: ||I - D^-1 K dt|| = "
                f"{spectral:.3f} >= 1 at dt={dt} and substeps={substeps}. "
                "Raise substeps rather than trusting an oscillating arm."
            )

        positions.append(x.copy())
        attractors.append(step_out.attractor.copy())
        velocities.append(step_out.velocity.copy())
        phases.append(phase)
        stds.append(prediction.velocity_std[0].copy())
        gates.append(step_out.gate)
        lags.append(float(np.linalg.norm(x - step_out.attractor)))
        clamped_steps += int(step_out.clamped)
        if prediction.orientation is not None:
            orientations.append(prediction.orientation[0])
        if prediction.stiffness is not None:
            stiffness.append(prediction.stiffness[0])
        if prediction.damping is not None:
            damping.append(prediction.damping[0])
        if prediction.gripper is not None:
            grippers.append(float(prediction.gripper[0]))

        attractor = step_out.attractor
        held = (
            blocked_steps is not None
            and blocked_steps[0] <= len(positions) - 1 <= blocked_steps[1]
        )
        if not held:
            for _ in range(substeps):
                x = x + A @ (attractor - x) * (dt / substeps)
                if load is not None:
                    x = x + np.asarray(load, dtype=float) * (dt / substeps)

        phase = advance_phase(
            policy, belief_position(attractor, x, law.query_at),
            phase, dt, step_out.gate, belief_correction,
        )
        if lag_tolerance and watchdog.update(
            phase, step_out.behind, step_out.expected
        ):
            stalled = True
            break
        if phase >= stop_time_belief:
            break

    def _stack(seq):
        return np.stack(seq) if seq else None

    steps = len(positions)
    return Rollout(
        positions=np.stack(positions),
        velocities=np.stack(velocities),
        time_belief=np.array(phases),
        velocity_std=np.stack(stds),
        orientations=_stack(orientations),
        stiffness=_stack(stiffness),
        damping=_stack(damping),
        gripper=np.array(grippers) if grippers else None,
        attractors=np.stack(attractors),
        gate=np.array(gates),
        lag=np.array(lags),
        metadata={
            "steps": steps,
            "dt": dt,
            "speed_limit": speed_limit,
            "substeps": int(substeps),
            "terminated_on_phase": bool(phase >= stop_time_belief),
            "blocked": bool(stalled),
            "blocked_after_steps": steps if stalled else None,
            # The third outcome, which nothing counted before: neither finished
            # nor diagnosed as stuck, just out of budget while crawling. 7.16
            # is three runs that did this and were blamed on the transport.
            "budget_exhausted": bool(
                not stalled and phase < stop_time_belief and steps >= n_steps
            ),
            "final_phase": float(phase),
            "final_lag": lags[-1] if lags else float("nan"),
            "clamped_fraction": clamped_steps / steps if steps else float("nan"),
            "static_sag": float(watchdog.static_sag),
            "sag_rebaselines": int(watchdog.rebaselines),
            "phase_epsilon": float(watchdog.phase_epsilon),
            "stall_patience": int(stall_patience),
            "gate_shut_fraction": float(np.mean(np.array(gates) < 1.0)) if gates else 0.0,
            "load": None if load is None else np.asarray(load).tolist(),
            "blocked_steps": blocked_steps,
            **law.to_dict(),
        },
    )
