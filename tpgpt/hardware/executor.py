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

* **The infeasible-pose fallback asks the FR3 model, not MuJoCo.**
  ``rollout_policy``'s two tiers -- relax the orientation if only the hand
  angle is out of reach, else steer to the next label the arm can hold and
  jump the phase on arrival -- run here exactly as there (same stride,
  tolerance, iteration cap and look-ahead), with ``solve_ik`` replaced by
  :meth:`tpgpt.hardware.fr3.FR3.solve_ik` seeded from the **measured** joint
  angles, as the simulation seeds from the arm's current configuration. It
  needs those angles: :meth:`HardwareExecutor.step` takes them, and without
  them the fallback does not run (recorded in the trace).
* **Optional speed scaling.** ``speed_scale < 1`` slows the whole task
  uniformly: velocities, the speed clamp and the phase clock are all scaled,
  so the lag gate's ``expected = ||K^-1 D|| v`` stays consistent with the speed
  actually commanded. ``speed_scale = 1`` is bit-for-bit the simulated loop;
  ``tests/unit/test_hardware_executor.py`` asserts it against
  :func:`~tpgpt.policy.rollout.rollout_impedance`.
* **Stiffness is a choice.** Stock CRISP accepts only a diagonal stiffness in
  the end-effector frame; the patched one in ``~/THOR_DOCKER_SETUP``
  (``crisp-full-stiffness.patch``) also takes the full transported matrix. See
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
    advance_attractor,
    advance_phase,
    advance_step,
    belief_position,
    clamp_speed,
    compliance_norm,
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
    #: ``rollout_policy``'s infeasible-pose fallback (see the module docstring).
    infeasible_fallback: bool = True
    feasibility_stride: int = 5
    feasibility_tolerance: float = 0.015
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


@dataclass(frozen=True)
class ArmPathSettings:
    """How a learned plan is followed: against the path the arm really traced.

    A learned plan's labels are a command that leads the arm (audit B1 step 6), and
    the plan carries where the arm went when it was learned (``Plan.arm_path``). The
    simulation's lag gate and clamp judge "normal lag" by the spring model
    (``||K^-1 D|| v``: 2-6 mm on the 18:39 plan), while on this arm the command must
    sit 17 mm ahead on average and up to 35 (twice the model's lag plus a 10-16 mm
    static offset), so they would slow the clock (gate 0.67 median) and cut the command
    (262 of 488 labels). Measured 2026-10-01; the user's decision to replace them for
    learned plans. What replaces them, against the arm path at the current phase:

    * **Pacing**: the clock (and the command) slow when the arm is behind the path,
      along it, by more than the path's margin, and stop ``pace_span`` beyond that.
    * **Correction**: a slowly growing offset on the command
      (``correction_gain`` x the deviation beyond the margin, per second) brings an arm
      that sits *beside* the path back onto it -- what waiting cannot do for a static
      offset. Across the path while it moves, in every direction while it is still;
      at most ``correction_max``. **Held, not grown, where the hand is in contact**
      (``arm_path["contact"]``, the replay's contact windows): there the transported
      stiffness alone must set the force -- grown, it pushed through it (21 mm at the
      set-down of the 2026-10-01 mug run, 12 N at 600 N/m; the user, 2026-10-01).
    * **Bound**: the command is kept within the plan's own lead at this label plus
      ``correction_max`` plus ``bound_slack`` of the arm (the force cap the clamp
      gave, without cutting what the plan needs).
    * **A held clock hands the gap to the correction.** Waiting closes lag, not a standing
      offset: an arm resting short of its command *along* the path stays there however
      long the clock waits (2026-10-01 22:20, the mug: 18-21 mm behind at labels 27-36,
      1.5-4 mm beside, the clock stopped, "blocked" with nothing in the way). So as the
      clock slows the correction takes the along-path gap too -- all of it when the
      clock has stopped -- and the clock recovers once the arm is back.
    * **Blocked** means the arm does not move: the clock held back or the correction at its
      cap for ``blocked_steps`` steps, and the arm itself moved less than
      ``blocked_motion`` in them. Then the run stops and holds. An arm that keeps moving
      is not blocked, however far the correction falls short (2026-10-01 22:41, the mug:
      the plan needed 25-36 mm, the correction stopped at its 30 mm cap with the arm 10 mm
      off its path and moving on, and "blocked" fired). With the correction off, a gap
      behind that is not closing is forgiven as sag (as the simulation's watchdog does),
      up to ``max_forgiven`` times.

    The velocity cap (``speed_limit``), the policy query and the clock's update stay
    the simulation's. Values are starting points: how far the arm strays on a new
    scene is not measured yet (step 9).
    """

    pace_span: float = 0.02
    #: Off: the clock runs at the plan's own rate whatever the arm does (to measure what
    #: the pacing is needed for; a blocked arm is then caught only by the correction cap
    #: and the executor's SafetyMonitor).
    pacing: bool = True
    correction: bool = True
    correction_gain: float = 1.5
    correction_max: float = 0.03
    bound_slack: float = 0.01
    blocked_steps: int = 40
    blocked_progress: float = 0.001
    blocked_motion: float = 0.002
    max_forgiven: int = 3
    #: Below this path speed (m/s at execution) a label counts as still.
    still_speed: float = 0.002
    #: Labels either side of the clock's label searched for where the command is.
    search_window: int = 20


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
    #: The orientation is not commanded this step (fallback tier 1):
    #: ``tcp_rotation`` is then the measured one, so the wrist is free.
    relaxed: bool = False
    trace: dict = field(default_factory=dict)


class HardwareExecutor:
    """Runs a fitted :class:`~tpgpt.policy.gp_policy.GPPolicy` one step at a time.

    Args:
        policy: Fitted on the plan's transported labels (``fr3_hand_tcp``
            orientations, grasp-point positions, base frame).
        dt: Control period of the labels, ``1 / control_freq``. The caller must
            call :meth:`step` at this period.
    """

    def __init__(self, policy, dt: float, settings: ExecutorSettings | None = None,
                 kinematics=None, arm_path=None, arm_settings: ArmPathSettings | None = None):
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
        # Infeasible-pose fallback state, as rollout_policy keeps it.
        self._kinematics = kinematics
        # Following a learned plan's arm path (ArmPathSettings) instead of the gate/clamp.
        self.arm = None
        self.correction = np.zeros(3)
        self.blocked = False
        self.blocked_reason = ""
        self._stuck_steps, self._stuck_ref, self._stuck_from = 0, 0.0, None
        self.forgiven, self.forgivals = 0.0, 0         # standing gap forgiven (correction off)
        if arm_path is not None:
            self._init_arm_path(arm_path, arm_settings or ArmPathSettings())
        self.relax_orientation = False
        self.skip_index: int | None = None
        self.relaxed_steps = self.skipped_steps = self.skips = 0
        self.feasibility_unavailable = 0

    def _init_arm_path(self, arm_path: dict, settings: ArmPathSettings) -> None:
        labels = self.policy.labels
        path = np.asarray(arm_path["positions"], dtype=float)
        if labels.time_belief is None or len(path) != len(labels.positions):
            raise ValueError("an arm path needs one point per label and labels with a time belief")
        margin = np.asarray(arm_path.get("margin", np.zeros(len(path))), dtype=float)
        contact = np.asarray(arm_path.get("contact", np.zeros(len(path), dtype=bool)), dtype=bool)
        tangent = np.gradient(path, axis=0)
        speed = np.linalg.norm(tangent, axis=1) / self.dt * float(self.settings.speed_scale)
        self.arm = {
            "settings": settings,
            "command": np.asarray(labels.positions, dtype=float),
            "time": np.asarray(labels.time_belief, dtype=float),
            "path": path, "margin": margin, "tangent": tangent, "speed": speed,
            "contact": contact,
            # how far ahead of the arm the command is meant to be, per label
            "lead": np.linalg.norm(np.asarray(labels.positions, dtype=float) - path, axis=1),
        }

    def command_index(self, attractor, phase: float) -> float:
        """Where the command is along its labels, as a fractional label index.

        The arm path is paired with the command label by label, so this -- not the
        clock -- says where the arm should be: the shared clock advances at the policy's
        predicted rate, which drifts from the command's own progress (on a straight line
        5 % slow, 25 mm apart after 160 steps). Searched within ``search_window`` labels
        of the clock's label; along a dwell, where many labels coincide, the one nearest
        the clock's.
        """
        a = self.arm
        n = len(a["time"])
        clock = float(np.interp(phase, a["time"], np.arange(n)))
        w = int(a["settings"].search_window)
        lo, hi = max(int(clock) - w, 0), min(int(clock) + w + 1, n - 1)
        seg_a, seg_b = a["command"][lo:hi], a["command"][lo + 1:hi + 1]
        d = seg_b - seg_a
        length2 = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", np.asarray(attractor) - seg_a, d)
                    / np.maximum(length2, 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(seg_a + t[:, None] * d - attractor, axis=1)
        index = np.arange(lo, hi) + t
        near = dist <= dist.min() + 0.001
        return float(index[near][np.argmin(np.abs(index[near] - clock))])

    def arm_target(self, phase: float, attractor=None) -> dict:
        """The arm path where the command is: where the arm should be, its direction, margin."""
        a = self.arm
        i = self.command_index(self.attractor if attractor is None else attractor, phase)
        k = min(int(i), len(a["time"]) - 2)
        w = i - k
        mix = lambda v: (1.0 - w) * v[k] + w * v[k + 1]   # noqa: E731
        t = mix(a["tangent"])
        n = float(np.linalg.norm(t))
        moving = mix(a["speed"]) > a["settings"].still_speed and n > 0.0
        return {"index": i, "position": mix(a["path"]), "margin": float(mix(a["margin"])),
                "contact": bool(a["contact"][int(round(i))]),
                "lead": float(mix(a["lead"])), "direction": t / n if moving else None}

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

    @property
    def kinematics(self):
        if self._kinematics is None:
            from tpgpt.hardware.fr3 import default_model

            self._kinematics = default_model()
        return self._kinematics

    def _reachable(self, tool_position, rotation, seed) -> bool:
        """``solve_ik(..., max_iterations=40)`` of ``_feasibility_action``, on the FR3."""
        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        target = self.tcp_from_tool(tool_position, R)
        return self.kinematics.solve_ik(
            target, rotation, seed=seed, max_iterations=40,
            position_tolerance=self.settings.feasibility_tolerance).reachable

    def _feasibility_action(self, rotation, seed) -> tuple[bool, int | None]:
        """``rollout._feasibility_action``: relax the hand angle, skip ahead, or neither."""
        if self.skip_index is not None:
            return False, self.skip_index
        if self._reachable(self.attractor, rotation, seed):
            return False, None
        if self._reachable(self.attractor, None, seed):
            return True, None
        return False, self._next_feasible(seed)

    def _next_feasible(self, seed, lookahead: int = 40, stride: int = 5) -> int | None:
        """``rollout._next_feasible``: the next label ahead the arm can hold."""
        labels = self.policy.labels
        if labels.time_belief is None or len(labels.positions) == 0:
            return None
        start = int(np.searchsorted(np.asarray(labels.time_belief), self.phase))
        rotations = labels.orientations
        for i in range(start + stride, min(start + lookahead, len(labels.positions)), stride):
            rotation = rotations[i] if rotations is not None else None
            if self._reachable(labels.positions[i], rotation, seed):
                return i
        return None

    def step(self, tcp_position, tcp_rotation, joint_positions=None) -> Command:
        """Advance one control step from a measured ``fr3_hand_tcp`` pose.

        The body is ``rollout_policy``'s loop body, line for line, minus the
        simulator calls; the comments name the lines it mirrors.

        Args:
            joint_positions: The measured arm joints, for the infeasible-pose
                fallback's IK seed. ``None`` leaves the fallback off this step.
        """
        if self.finished:
            raise RuntimeError("the executor already finished; build a new one")
        cfg = self.settings
        tcp_rotation = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
        position = self.tool_position(tcp_position, tcp_rotation)
        if self.arm is not None:
            return self._follow_step(position, tcp_rotation)

        # prediction = policy.predict(query_position(attractor, position, ...))
        prediction = self._predict(query_position(self.attractor, position, self.law.query_at))
        gripper_command = 1.0 if float(prediction.gripper[0]) > 0 else -1.0
        rotation = prediction.orientation[0] if prediction.orientation is not None else None

        # decision = gate_step(...); step_out = advance_step(...)
        decision = gate_step(
            position, self.attractor, prediction, self.law,
            static_sag=self.watchdog.static_sag,
        )
        # --- infeasible-pose fallback: only while the gate is shut --------------
        if cfg.infeasible_fallback and self.law.lag_tolerance and decision.gate < 1.0:
            if joint_positions is None:
                self.feasibility_unavailable += 1
            elif self.steps % max(int(cfg.feasibility_stride), 1) == 0:
                self.relax_orientation, self.skip_index = self._feasibility_action(
                    rotation, np.asarray(joint_positions, dtype=float).reshape(7))
            if self.relax_orientation:
                self.relaxed_steps += 1
        steer_to = (np.asarray(self.policy.labels.positions[self.skip_index], dtype=float)
                    if self.skip_index is not None else None)
        step_out = advance_step(
            position, self.attractor, decision, self.law, prediction,
            static_sag=self.watchdog.static_sag, steer_to=steer_to,
        )
        self.attractor = step_out.attractor
        if self.skip_index is not None:
            if step_out.skip_arrived:
                # Arrived: hand the clock over so the gripper schedule matches
                # where the arm now is, and resume the policy.
                if self.policy.labels.time_belief is not None:
                    self.phase = float(self.policy.labels.time_belief[self.skip_index])
                self.skip_index = None
                self.skips += 1
            else:
                self.skipped_steps += 1

        # Firm about orientation at the grasp and the placement, compliant in
        # transit -- the same expression as rollout_policy.
        firm = decision.compliance
        carrying = gripper_command > 0 and firm > 0.0 and firm >= self.stiff_threshold
        rotational_gain = (
            cfg.rotational_stiffness * cfg.transit_rotational_scale
            if carrying else cfg.rotational_stiffness
        )
        # A relaxed segment commands no orientation: the measured one is sent,
        # so the wrist finds whatever angle reaches the point.
        relaxed = bool(self.relax_orientation) or rotation is None
        commanded_rotation = tcp_rotation if relaxed else rotation
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
            relaxed=bool(self.relax_orientation),
            trace={
                "skip_index": self.skip_index,
                "relaxed_steps": int(self.relaxed_steps),
                "skipped_steps": int(self.skipped_steps),
                "skips": int(self.skips),
                "feasibility_unavailable": int(self.feasibility_unavailable),
                "max_lag": float(step_out.max_lag),
                "static_sag": float(self.watchdog.static_sag),
                "sag_rebaselines": int(self.watchdog.rebaselines),
                "carrying": bool(carrying),
            },
        )


    def _follow_step(self, position, tcp_rotation) -> Command:
        """One step of a learned plan, against its arm path (:class:`ArmPathSettings`)."""
        cfg, a = self.settings, self.arm
        s = a["settings"]
        prediction = self._predict(self.attractor)                  # queried at the attractor
        gripper_command = 1.0 if float(prediction.gripper[0]) > 0 else -1.0
        rotation = prediction.orientation[0] if prediction.orientation is not None else None
        velocity = clamp_speed(prediction.velocity[0], self.law.speed_limit)
        compliance = compliance_norm(prediction.stiffness[0], prediction.damping[0])

        target = self.arm_target(self.phase)
        error = target["position"] - position                     # where it should be - where it is
        u = target["direction"]
        behind = float(error @ u) if u is not None else 0.0
        pace = (float(np.clip(1.0 - (behind - target["margin"] - self.forgiven) / s.pace_span,
                              0.0, 1.0)) if s.pacing else 1.0)

        self.attractor = advance_attractor(self.attractor, velocity, self.dt, pace,
                                           law=self.law.attractor_law)
        along = (error @ u) * u if u is not None else np.zeros(3)
        side = error - along
        size = float(np.linalg.norm(side))                           # beside the path
        if s.correction and not target["contact"]:
            # Across the path, beyond the margin. Along it, as much as the clock is held back
            # (all of it when stopped) -- no deadband there: the clock is only held once the
            # arm is beyond the margin. (With the margin on both, a half-held clock and a
            # partly closed gap balanced: 7.4 mm behind for good, the clock at 0.73.)
            drive = (1.0 - pace) * along
            if size > target["margin"]:
                drive = drive + side / size * (size - target["margin"])
            self.correction = self.correction + s.correction_gain * self.dt * drive
            c = float(np.linalg.norm(self.correction))
            if c > s.correction_max:
                self.correction *= s.correction_max / c
        point = self.attractor + self.correction
        radius = target["lead"] + s.correction_max + s.bound_slack
        reach = point - position
        bounded = float(np.linalg.norm(reach)) > radius
        if bounded:
            point = position + reach * (radius / float(np.linalg.norm(reach)))

        firm = compliance
        carrying = gripper_command > 0 and firm > 0.0 and firm >= self.stiff_threshold
        rotational_gain = (cfg.rotational_stiffness * cfg.transit_rotational_scale
                           if carrying else cfg.rotational_stiffness)
        commanded_rotation = tcp_rotation if rotation is None else rotation
        query_phase = self.phase
        self.phase = advance_phase(self.policy, self.attractor, self.phase, self.phase_dt, pace,
                                   cfg.belief_correction)
        self.steps += 1

        # Blocked: correcting at the cap and not getting closer. (Without the correction a
        # stopped clock that is not getting closer forgives the gap as sag, then blocks.)
        deviation = float(np.linalg.norm(error))
        saturated = s.correction and float(np.linalg.norm(self.correction)) >= 0.99 * s.correction_max
        stopped = (not s.correction) and pace < 0.05
        held = pace < 0.05 or saturated
        if held or stopped:
            if self._stuck_steps == 0:
                self._stuck_ref, self._stuck_from = deviation, np.asarray(position).copy()
            self._stuck_steps += 1
            if self._stuck_steps >= s.blocked_steps:
                moved = float(np.linalg.norm(np.asarray(position) - self._stuck_from))
                if self._stuck_ref - deviation < s.blocked_progress:
                    if stopped and self.forgivals < s.max_forgiven:
                        self.forgiven = max(behind - target["margin"], 0.0)
                        self.forgivals += 1
                    elif moved < s.blocked_motion:
                        self.blocked, self.finished = True, True
                        self.blocked_reason = (
                            f"the arm has moved {moved * 1000:.1f} mm in {s.blocked_steps} steps, "
                            f"{deviation * 1000:.0f} mm from its path at label "
                            f"{target['index']:.0f} ("
                            + ("correction at its cap" if saturated else "clock stopped") + ")")
                self._stuck_steps = 0
        else:
            self._stuck_steps = 0
        if self.phase >= cfg.stop_time_belief:
            self.finished = True

        return Command(
            tcp_position=self.tcp_from_tool(point, commanded_rotation),
            tcp_rotation=np.asarray(commanded_rotation, dtype=float),
            attractor=self.attractor.copy(),
            stiffness=np.asarray(prediction.stiffness[0], dtype=float),
            damping=np.asarray(prediction.damping[0], dtype=float),
            rotational_stiffness=float(rotational_gain),
            rotational_damping=float(cfg.rotational_damping * (rotational_gain / cfg.rotational_stiffness)),
            gripper=gripper_command, phase=query_phase, next_phase=self.phase,
            gate=pace, behind=behind, expected=0.0, clamped=bounded,
            velocity=np.asarray(velocity, dtype=float),
            done=self.finished and not self.blocked, stalled=self.blocked, relaxed=rotation is None,
            trace={
                "following_arm_path": True,
                "path_index": float(target["index"]),
                "deviation": deviation,
                # beside the path (across it while it moves; all of it while it is still) --
                # what can take the arm off the checked path. ``behind`` is along it.
                "across": size,
                "contact": target["contact"],
                "margin": target["margin"],
                "correction": self.correction.copy(),
                "lead": target["lead"],
                "blocked": self.blocked,
                "forgiven": self.forgiven,
                "blocked_reason": self.blocked_reason,
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


def bounded_stiffness(stiffness, minimum: float, maximum: float) -> np.ndarray:
    """The symmetric part of ``K`` with its principal values clamped to ``[minimum, maximum]``.

    CRISP's own rule for a full translational block (``crisp-full-stiffness.patch`` in
    ``~/THOR_DOCKER_SETUP``): clamping eigenvalues, not entries, keeps the matrix positive
    definite and inside the bounds in every direction. A matrix already inside is returned
    unchanged (symmetrized).
    """
    K = np.asarray(stiffness, dtype=float).reshape(3, 3)
    K = 0.5 * (K + K.T)
    values, vectors = np.linalg.eigh(K)
    clamped = np.clip(values, minimum, maximum)
    if np.array_equal(clamped, values):
        return K
    return vectors @ np.diag(clamped) @ vectors.T


def crisp_stiffness(
    stiffness, tcp_rotation, rotational_stiffness: float,
    min_translational: float = 50.0, max_translational: float = 900.0,
    max_rotational: float = 60.0, full: bool = False,
) -> np.ndarray:
    """The values CRISP's ``target_stiffness`` topic takes: 6 (diagonal) or 12 (``full``).

    CRISP's ``CartesianController`` with ``use_local_jacobian: true`` (the FR3
    real configuration) computes its pose error **in the end-effector frame**
    (``cartesian_controller.cpp``, ``update``). The transported ``K`` is a full
    matrix in the base frame (Sec. III-G), so it is rotated into the measured TCP
    frame, ``K_tcp = R^T K R``, and sent in one of two forms:

    * **diagonal** (6 values ``[k_x, k_y, k_z, k_rot x3]``): the stock message. The
      off-diagonal terms -- the rotation of the stiffness ellipsoid relative to the
      hand -- cannot be expressed and are dropped; :func:`stiffness_projection_loss`
      says how much. Entries clamped to ``[min, max]``.
    * **full** (12 values: ``K_tcp`` row-major, then ``k_rot x3``): needs the patched
      CRISP (``crisp-full-stiffness.patch``). Nothing is dropped; the principal values
      are clamped to ``[min, max]`` (:func:`bounded_stiffness`), as CRISP itself does.

    The bounds are CRISP's ``variable_max_stiffness``, plus a floor so a transported
    near-zero gain cannot make the arm limp.
    """
    K = np.asarray(stiffness, dtype=float).reshape(3, 3)
    R = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
    local = R.T @ K @ R
    rotational = np.full(3, float(np.clip(rotational_stiffness, 0.0, max_rotational)))
    if full:
        block = bounded_stiffness(local, min_translational, max_translational)
        return np.concatenate([block.ravel(), rotational])
    translational = np.clip(np.diag(local), min_translational, max_translational)
    return np.concatenate([translational, rotational])


def crisp_accepts_full_stiffness(description: str) -> bool:
    """Whether CRISP's ``variable_stiffness.topic`` description admits the 12-value message.

    The patched CRISP (``crisp-full-stiffness.patch``, ``docker-thor:local`` since
    2026-09-29) documents a "12 elements" form there; the stock one only "6 elements"
    (both read from the images on fake hardware, 2026-09-29). The stock one does not
    reject a 12-value message, it ignores it and keeps its yaml gains, so this is how
    the executor finds out before running.
    """
    return "12 elements" in (description or "")


def crisp_applied_gains(values, tcp_rotation) -> tuple[np.ndarray, np.ndarray]:
    """The translational ``(K, D)`` CRISP applies for a ``target_stiffness`` message, in the base frame.

    The inverse of :func:`crisp_stiffness`, for a model of the arm that sees what the
    controller runs rather than what the policy asked for. Damping is CRISP's automatic
    rule (``d_pos_*: -1`` in ``tpgpt_variable_stiffness_controller.yaml``): ``2 sqrt(k)``
    per axis for the 6-value form, and ``2 K^(1/2)`` for the 12-value form -- the same
    rule along the block's own principal axes.
    """
    values = np.asarray(values, dtype=float).ravel()
    R = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
    if values.size == 12:
        local = values[:9].reshape(3, 3)
    elif values.size == 6:
        local = np.diag(values[:3])
    else:
        raise ValueError(f'a stiffness message has 6 or 12 values, got {values.size}')
    principal, axes = np.linalg.eigh(0.5 * (local + local.T))
    damping_local = axes @ np.diag(2.0 * np.sqrt(np.maximum(principal, 0.0))) @ axes.T
    return R @ local @ R.T, R @ damping_local @ R.T


def logged_stiffness(values) -> tuple[np.ndarray, float]:
    """``(diagonal of K_tcp, k_rot)`` of a 6- or 12-value message, for the execution log."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 12:
        return values[[0, 4, 8]], float(values[9])
    return values[:3], float(values[3])


def stiffness_projection_loss(stiffness, tcp_rotation) -> float:
    """Fraction of ``K``'s Frobenius norm in the dropped off-diagonal terms."""
    K = np.asarray(stiffness, dtype=float).reshape(3, 3)
    R = np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
    local = R.T @ K @ R
    off = local - np.diag(np.diag(local))
    return float(np.linalg.norm(off) / max(np.linalg.norm(local), 1e-12))
