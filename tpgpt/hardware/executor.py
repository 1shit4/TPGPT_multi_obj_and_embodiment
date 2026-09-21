"""The execution loop of ``rollout_policy``, with the measured pose from a robot.

:func:`tpgpt.sim.rollout.rollout_policy` runs one control step as

    predict at the attractor -> gate_step -> advance_step -> command
    -> advance_phase from the attractor -> StallWatchdog

and every one of those is a function in :mod:`tpgpt.policy.rollout` that
imports only numpy. :class:`HardwareExecutor` calls **the same functions in the
same order**; the simulator's ``controller.eef_state()`` is replaced by the
robot's measured ``fr3_hand_tcp`` pose, and ``controller.action(...)`` by a
:class:`Command` the ROS node publishes to CRISP.

What is intentionally different on hardware, stated so nothing is silently
absorbed (``docs/sim/SIM_CLAUDE.md`` rule 3):

* **No infeasible-pose fallback.** ``rollout_policy``'s two tiers ask MuJoCo's
  ``solve_ik`` whether a pose exists for the arm. There is no simulator here,
  so the fallback is off (``infeasible_fallback=False`` is a supported setting
  of the simulated rollout too). A pose the arm cannot hold then shows up as a
  shut gate and, eventually, a stalled watchdog -- the run stops and holds.
* **Optional speed scaling.** ``speed_scale < 1`` slows the whole task
  uniformly: velocities, the speed clamp and the phase clock are all scaled,
  so the lag gate's ``expected = ||K^-1 D|| v`` stays consistent with the speed
  actually commanded. ``speed_scale = 1`` is bit-for-bit the simulated loop;
  ``tests/unit/test_hardware_executor.py`` asserts it against
  :func:`~tpgpt.policy.rollout.rollout_impedance`.
* **Stiffness is a choice.** CRISP accepts a diagonal stiffness in the
  end-effector frame, not the full transported matrix; see
  :func:`crisp_stiffness`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from tpgpt.hardware.frames import rotation_angle, slerp_rotation
from tpgpt.policy.rollout import (
    ExecutionLaw,
    StallWatchdog,
    advance_phase,
    advance_step,
    belief_position,
    gate_step,
    phase_epsilon_from,
    query_position,
    stiffness_threshold,
)


@dataclass(frozen=True)
class ExecutorSettings:
    """Every knob, with ``rollout_policy``'s defaults."""

    belief_correction: float = 0.2
    stop_time_belief: float = 1.0
    speed_limit_factor: float = 3.0
    lag_tolerance: float | None = 0.035
    stall_patience: int = 40
    sag_epsilon: float = 0.002
    max_sag_rebaselines: int = 3
    min_phase_progress: float = 0.25
    rotational_stiffness: float = 60.0
    rotational_damping: float = 12.0
    transit_rotational_scale: float = 0.25
    attractor_law: str = "integrate"
    anchor_gain: float = 0.0
    anchor_gated: bool = True
    query_at: str = "attractor"
    skip_arrival: float = 0.02
    #: Uniform slow-down, in ``(0, 1]``. See the module docstring.
    speed_scale: float = 1.0
    #: Where the grasp point sits in ``fr3_hand_tcp`` coordinates. The plan's
    #: positions are the grasp point (fingertip centre); ``fr3_hand_tcp`` is
    #: already the fingertip centre of a Franka Hand, so this is zero unless a
    #: measurement says otherwise. Same meaning as ``rollout_policy``'s
    #: ``tool_offset``.
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Command:
    """One control step's output, ready for the controller."""

    tcp_position: np.ndarray        # (3,) fr3_hand_tcp target, base frame
    tcp_rotation: np.ndarray        # (3, 3) fr3_hand_tcp target, base frame
    attractor: np.ndarray           # (3,) grasp-point attractor
    stiffness: np.ndarray           # (3, 3) transported K, base frame
    damping: np.ndarray             # (3, 3) transported D, base frame
    rotational_stiffness: float
    rotational_damping: float
    gripper: float                  # +1 close, -1 open
    phase: float                    # phase used for this step's query
    next_phase: float
    gate: float
    behind: float
    expected: float
    clamped: bool
    velocity: np.ndarray
    done: bool = False
    stalled: bool = False
    trace: dict = field(default_factory=dict)


class HardwareExecutor:
    """Runs a fitted :class:`~tpgpt.policy.gp_policy.GPPolicy` one step at a time.

    Args:
        policy: Fitted on the plan's transported labels (``fr3_hand_tcp``
            orientations, grasp-point positions, base frame).
        dt: Control period of the labels, ``1 / control_freq``. The caller must
            call :meth:`step` at this period.
    """

    def __init__(self, policy, dt: float, settings: ExecutorSettings | None = None):
        if policy.labels is None:
            raise ValueError("HardwareExecutor needs a fitted policy")
        self.policy = policy
        self.settings = settings or ExecutorSettings()
        s = float(self.settings.speed_scale)
        if not 0.0 < s <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {s}")
        self.dt = float(dt)
        self.phase_dt = self.dt * s
        labels = policy.labels
        max_label_speed = float(np.linalg.norm(labels.velocities, axis=1).max()) * s
        speed_limit = self.settings.speed_limit_factor * max_label_speed
        self.law = ExecutionLaw(
            dt=self.dt, speed_limit=speed_limit, max_label_speed=max_label_speed,
            lag_tolerance=self.settings.lag_tolerance,
            attractor_law=self.settings.attractor_law,
            anchor_gain=self.settings.anchor_gain,
            anchor_gated=self.settings.anchor_gated,
            query_at=self.settings.query_at,
            skip_arrival=self.settings.skip_arrival,
        )
        self.watchdog = StallWatchdog(
            stall_patience=self.settings.stall_patience,
            sag_epsilon=self.settings.sag_epsilon,
            max_sag_rebaselines=self.settings.max_sag_rebaselines,
            phase_epsilon=phase_epsilon_from(
                labels, self.phase_dt, self.settings.stall_patience,
                self.settings.min_phase_progress,
            ),
        )
        self.stiff_threshold = stiffness_threshold(labels)
        self.offset = np.asarray(self.settings.tool_offset, dtype=float).reshape(3)
        self.attractor = np.asarray(labels.positions[0], dtype=float).copy()
        self.phase = 0.0
        self.steps = 0
        self.finished = False
        self.stalled = False

    # ------------------------------------------------------------- frames
    def tool_position(self, tcp_position, tcp_rotation) -> np.ndarray:
        """Measured grasp point from the measured TCP (``rollout_policy.tool_state``)."""
        return np.asarray(tcp_position, dtype=float) + np.asarray(tcp_rotation) @ self.offset

    def tcp_from_tool(self, tool_position, rotation) -> np.ndarray:
        """TCP target for a grasp-point target (the wrist back-out of ``rollout_policy``)."""
        return np.asarray(tool_position, dtype=float) - np.asarray(rotation) @ self.offset

    @property
    def start_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """``fr3_hand_tcp`` pose at the first label, for the approach phase."""
        labels = self.policy.labels
        R = np.asarray(labels.orientations[0], dtype=float)
        return self.tcp_from_tool(labels.positions[0], R), R

    # --------------------------------------------------------------- step
    def _predict(self, point):
        query_phase = np.array([self.phase]) if self.policy.use_time_belief else None
        prediction = self.policy.predict(np.asarray(point)[None], query_phase)
        s = self.settings.speed_scale
        if s != 1.0:
            prediction = dataclasses.replace(prediction, velocity=prediction.velocity * s)
        return prediction

    def step(self, tcp_position, tcp_rotation) -> Command:
        """Advance one control step from a measured ``fr3_hand_tcp`` pose.

        The body is ``rollout_policy``'s loop body, line for line, minus the
        simulator calls; the comments name the lines it mirrors.
        """
        if self.finished:
            raise RuntimeError("the executor already finished; build a new one")
        cfg = self.settings
        tcp_rotation = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
        position = self.tool_position(tcp_position, tcp_rotation)

        # prediction = policy.predict(query_position(attractor, position, ...))
        prediction = self._predict(query_position(self.attractor, position, self.law.query_at))
        gripper_command = 1.0 if float(prediction.gripper[0]) > 0 else -1.0
        rotation = prediction.orientation[0] if prediction.orientation is not None else None

        # decision = gate_step(...); step_out = advance_step(...)
        decision = gate_step(
            position, self.attractor, prediction, self.law,
            static_sag=self.watchdog.static_sag,
        )
        step_out = advance_step(
            position, self.attractor, decision, self.law, prediction,
            static_sag=self.watchdog.static_sag, steer_to=None,
        )
        self.attractor = step_out.attractor

        # Firm about orientation at the grasp and the placement, compliant in
        # transit -- the same expression as rollout_policy.
        firm = decision.compliance
        carrying = gripper_command > 0 and firm > 0.0 and firm >= self.stiff_threshold
        rotational_gain = (
            cfg.rotational_stiffness * cfg.transit_rotational_scale
            if carrying else cfg.rotational_stiffness
        )
        commanded_rotation = rotation if rotation is not None else tcp_rotation
        tcp_target = self.tcp_from_tool(self.attractor, commanded_rotation)

        query_phase = self.phase
        # phase = advance_phase(policy, belief_position(attractor, position), ...)
        self.phase = advance_phase(
            self.policy, belief_position(self.attractor, position, self.law.query_at),
            self.phase, self.phase_dt, decision.gate, cfg.belief_correction,
        )
        self.steps += 1
        if cfg.lag_tolerance and self.watchdog.update(
            self.phase, decision.behind, decision.expected
        ):
            self.stalled = True
            self.finished = True
        if self.phase >= cfg.stop_time_belief:
            self.finished = True

        return Command(
            tcp_position=tcp_target,
            tcp_rotation=np.asarray(commanded_rotation, dtype=float),
            attractor=self.attractor.copy(),
            stiffness=np.asarray(prediction.stiffness[0], dtype=float),
            damping=np.asarray(prediction.damping[0], dtype=float),
            rotational_stiffness=float(rotational_gain),
            rotational_damping=float(
                cfg.rotational_damping * (rotational_gain / cfg.rotational_stiffness)
            ),
            gripper=gripper_command,
            phase=query_phase,
            next_phase=self.phase,
            gate=float(decision.gate),
            behind=float(decision.behind),
            expected=float(decision.expected),
            clamped=bool(step_out.clamped),
            velocity=np.asarray(step_out.velocity, dtype=float),
            done=self.finished and not self.stalled,
            stalled=self.stalled,
            trace={
                "max_lag": float(step_out.max_lag),
                "static_sag": float(self.watchdog.static_sag),
                "sag_rebaselines": int(self.watchdog.rebaselines),
                "carrying": bool(carrying),
            },
        )


# ---------------------------------------------------------------------------
# Hardware-only safety, kept outside the execution arithmetic
# ---------------------------------------------------------------------------


@dataclass
class SafetyMonitor:
    """Stops a run that the execution loop itself would let finish.

    **Measured: the simulated execution loop completes the task with an arm
    that never moves.** Driving ``rollout_impedance`` with
    ``blocked_steps=(0, 10000)`` on a straight-line policy, the watchdog
    forgives the standing offset as sag once, the gate reopens, the clamp keeps
    the attractor 48 mm ahead of the motionless arm, the phase is advanced from
    the attractor, and the run ends with ``terminated_on_phase=True`` after 253
    steps -- gripper schedule included. :class:`HardwareExecutor` reproduces
    that bit for bit, as it must.

    In MuJoCo the arm always moves when pulled, so this never mattered. On a
    robot an inactive controller, a protective stop or a collision is exactly
    an arm that does not move, and the jaws would close on air. These checks
    are therefore added on top, never inside, the shared loop:

    * ``max_tracking_error`` -- measured grasp point to attractor distance, in
      metres. The loop's own clamp keeps this below ``expected + sag + 35 mm``
      when the arm follows, so a larger value means it is not following.
    * ``min_progress_ratio`` over ``progress_window`` steps -- the measured
      point must travel at least this fraction of the distance the attractor
      travelled. Only judged once the attractor moved ``min_attractor_travel``.
    """

    max_tracking_error: float = 0.08
    progress_window: int = 40
    min_progress_ratio: float = 0.3
    min_attractor_travel: float = 0.03
    _arm: list = field(default_factory=list)
    _attractor: list = field(default_factory=list)

    def check(self, measured_point, attractor) -> str | None:
        """Reason to abort, or ``None``."""
        measured_point = np.asarray(measured_point, dtype=float)
        attractor = np.asarray(attractor, dtype=float)
        error = float(np.linalg.norm(measured_point - attractor))
        if self.max_tracking_error and error > self.max_tracking_error:
            return (f"tracking error {error * 1000:.0f} mm exceeds "
                    f"{self.max_tracking_error * 1000:.0f} mm")
        self._arm.append(measured_point)
        self._attractor.append(attractor)
        window = self.progress_window
        if len(self._arm) > window:
            arm = np.linalg.norm(np.diff(self._arm[-window:], axis=0), axis=1).sum()
            att = np.linalg.norm(np.diff(self._attractor[-window:], axis=0), axis=1).sum()
            if att >= self.min_attractor_travel and arm < self.min_progress_ratio * att:
                return (f"the arm moved {arm * 1000:.0f} mm while its target moved "
                        f"{att * 1000:.0f} mm over {window} steps; it is not following")
            del self._arm[0], self._attractor[0]
        return None


# ---------------------------------------------------------------------------
# The approach phase and the controller interface
# ---------------------------------------------------------------------------


def approach_trajectory(
    start_position, start_rotation, goal_position, goal_rotation, dt: float,
    max_linear_speed: float = 0.05, max_angular_speed: float = 0.25,
    min_duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Waypoints from the current TCP pose to the plan's start, at bounded speed.

    ``rollout_policy``'s ``_approach`` interpolates linearly over a fixed step
    count because the simulated arm always starts at the same home. A real arm
    starts wherever it was left, so the duration comes from the distance and
    two speed caps instead. A cosine profile starts and stops at zero velocity.

    Returns ``(positions (n, 3), rotations (n, 3, 3))`` at period ``dt``.
    """
    p0 = np.asarray(start_position, dtype=float)
    p1 = np.asarray(goal_position, dtype=float)
    distance = float(np.linalg.norm(p1 - p0))
    angle = rotation_angle(start_rotation, goal_rotation)
    # Peak speed of a cosine profile is pi/2 times its mean.
    duration = max(
        min_duration,
        (np.pi / 2) * distance / max(max_linear_speed, 1e-6),
        (np.pi / 2) * angle / max(max_angular_speed, 1e-6),
    )
    n = max(2, int(np.ceil(duration / dt)) + 1)
    s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))
    positions = p0 + s[:, None] * (p1 - p0)
    rotations = np.stack([slerp_rotation(start_rotation, goal_rotation, float(v)) for v in s])
    return positions, rotations


def crisp_stiffness(
    stiffness, tcp_rotation, rotational_stiffness: float,
    min_translational: float = 50.0, max_translational: float = 900.0,
    max_rotational: float = 60.0,
) -> np.ndarray:
    """The six values CRISP's ``target_stiffness`` topic takes.

    CRISP's ``CartesianController`` with ``use_local_jacobian: true`` (the FR3
    real configuration) computes its pose error **in the end-effector frame**
    and applies ``diag(k_pos_x, k_pos_y, k_pos_z, k_rot_x, k_rot_y, k_rot_z)``
    to it (``cartesian_controller.cpp``, ``update`` and
    ``setStiffnessAndDamping``). The transported ``K`` is a full matrix in the
    base frame (Sec. III-G). So it is rotated into the measured TCP frame,
    ``K_tcp = R^T K R``, and its **diagonal** is sent. The off-diagonal terms --
    the rotation of the stiffness ellipsoid relative to the hand -- cannot be
    expressed and are dropped; the returned values say how much was dropped in
    :func:`stiffness_projection_loss`.

    Clamped to CRISP's own ``variable_max_stiffness`` bounds, plus a floor so a
    transported near-zero gain cannot make the arm limp.
    """
    K = np.asarray(stiffness, dtype=float).reshape(3, 3)
    R = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
    local = R.T @ K @ R
    translational = np.clip(np.diag(local), min_translational, max_translational)
    rotational = np.full(3, float(np.clip(rotational_stiffness, 0.0, max_rotational)))
    return np.concatenate([translational, rotational])


def stiffness_projection_loss(stiffness, tcp_rotation) -> float:
    """Fraction of ``K``'s Frobenius norm in the dropped off-diagonal terms."""
    K = np.asarray(stiffness, dtype=float).reshape(3, 3)
    R = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
    local = R.T @ K @ R
    off = local - np.diag(np.diag(local))
    return float(np.linalg.norm(off) / max(np.linalg.norm(local), 1e-12))
