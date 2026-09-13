"""Executing a transported policy on the simulated robot (paper Sec. III-B, V).

This is where the whole pipeline is actually tested: the policy refitted on the
transported labels drives a Cartesian impedance controller, so the transported
orientation *and* the transported stiffness both have to be right for the task
to succeed. Replaying a warped trajectory would exercise neither.

**How the policy is executed.** The learned dynamical system is integrated to
produce a *moving attractor*, and the arm follows that attractor through the
Cartesian impedance. This is how the demonstration itself was generated -- its
attractor was interpolated smoothly between waypoints while the arm tracked it --
and it is the standard way to run a learned dynamical system on an
impedance-controlled robot.

The alternative, recomputing the attractor from the measured position every
step, does not work here and the reason is worth recording. In an impedance
controller the attractor *is* the reference, so the restoring speed is set by
the position error times the gain ratio ``K/D``, about 12 per second in this
task. A 24 mm deviation then commands 0.29 m/s no matter what velocity the
policy asked for. Measured that way, the arm ran at 0.37 m/s while the policy
was commanding 0.002 m/s, overshot the grasp, and lost the path entirely.

Integrating the attractor also keeps the policy query on the manifold it was
fitted on, so the zero-mean prior of Appendix A never gets a chance to decay the
command to zero. For the same reason the **time belief is advanced from the
attractor, not from the measured position**: the arm lags its attractor by
``v D / K`` -- 12 to 25 mm here -- and driving the phase from the lagging
position while querying the policy at the attractor pairs a position with a
phase that never occur together in the labels. Measured that way, the path
tracking error sat at 30 to 60 mm regardless of the gains, and the gripper
closed 2 to 4 cm short of the object every time.

The physical loop closes through the impedance instead, which is the right
division of responsibilities: contact forces act on the arm directly, and the
attractor is clamped to stay within reach of it, so a blocked arm builds a
bounded force rather than letting the attractor escape.

**The lag gate.** Progress -- both the attractor and the phase -- is held back
whenever the arm falls too far behind. An impedance-controlled arm lags its
attractor while moving, and the demonstration absorbed that by dwelling at each
waypoint until the arm settled: its grasp segment held the attractor still for
15 steps, and the lag fell to 5 mm before the fingers closed. A policy rollout
has no scripted dwells, so without a gate the gripper closes on schedule while
the arm is still 20 to 56 mm behind. Measured across eight scenes, the gripper
closed at step 50 every time while the arm did not reach the object until step
54 to 63.

Gating on the measured lag recovers the dwell from the robot's own state rather
than from a script, and it is what a belief of *progress* should mean: the task
has not advanced until the robot has.

The gate is on *excess* lag, not raw lag. An impedance controller has an
expected steady-state lag of ``||K^-1 D|| v`` whenever it is moving, and
penalising that would stall every fast segment. Subtracting it means free motion
runs at full speed while a dwell -- where the commanded velocity goes to zero,
so the expected lag does too -- demands that the arm actually arrive.

**The approach phase.** Before the policy takes over, the arm is driven to the
start of the transported demonstration. This is protocol, not a workaround, and
it follows from Appendix A: the velocity prior is zero, so a policy fitted on
demonstration velocities has *no restoring component* off the demonstration --
at an unvisited state it blends the velocities of nearby labels, which points
*along* the demonstration rather than *towards* it. That is deliberate ("the
robot does not attempt to do any movement if there is no significant evidence
from the human demonstration"), and it means the policy must be entered near its
own data.

It matters here because the arm's home pose is identical in every scene while
the transported start ``phi(x_0)`` is not: the map deforms space at the home
pose too, displacing the start of the policy by of order 0.1 m. Measured
without an approach phase, the arm drifted away from the transported labels and
stalled completely within 50 steps. On a real robot the operator hands the arm
to roughly the right pose; here a reaching primitive does the same job.
:func:`rollout_policy` reports the size of that gap as a diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tpgpt.policy.rollout import (
    AnchorSchedule,
    ExecutionLaw,
    StallWatchdog,
    advance_phase,
    advance_step,
    belief_position,
    compliance_norm,
    gate_step,
    phase_epsilon_from,
    query_position,
    stiffness_threshold,
)
from tpgpt.sim.backend import FrameWriter, observation_frame
from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController


@dataclass
class SimRollout:
    """Result of executing a policy in the environment."""

    positions: np.ndarray
    attractors: np.ndarray
    velocities: np.ndarray
    time_belief: np.ndarray
    velocity_std: np.ndarray
    gripper: np.ndarray
    contact_force: np.ndarray
    success: bool = False
    final_offset: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.positions.shape[0]


def rollout_policy(
    env,
    policy,
    max_steps: int = 400,
    belief_correction: float = 0.2,
    stop_time_belief: float = 1.0,
    video_path=None,
    camera: str = "frontview",
    rotational_stiffness: float = 60.0,
    rotational_damping: float = 12.0,
    transit_rotational_scale: float = 0.25,
    approach_steps: int = 45,
    approach_stiffness: float = 350.0,
    speed_limit_factor: float = 3.0,
    lag_tolerance: float = 0.035,
    stall_patience: int = 40,
    sag_epsilon: float = 0.002,
    max_sag_rebaselines: int = 3,
    min_phase_progress: float = 0.25,
    tool_offset=None,
    infeasible_fallback: bool = True,
    feasibility_stride: int = 5,
    feasibility_tolerance: float = 0.015,
    skip_arrival: float = 0.02,
    score: Callable | None = None,
    probe: Callable | None = None,
    attractor_law: str = "integrate",
    anchor_gain: float | AnchorSchedule = 0.0,
    anchor_gated: bool = True,
    query_at: str = "attractor",
) -> SimRollout:
    """Run a fitted policy on the robot until its phase completes.

    At every control step the policy is queried at the **attractor** by default,
    not at the measured pose, which is what keeps the query on the ridge of
    ``(position, phase)`` pairs the policy was fitted on. ``query_at="measured"``
    selects the alternative; section 7.40 measures it as clearly worse, including
    on the one comparison that isolates the query site with nothing else
    attached. The physical loop closes through the impedance instead -- see the
    module docstring.

    Args:
        env: Reshelving environment on the joint-torque controller.
        policy: A fitted :class:`~tpgpt.policy.gp_policy.GPPolicy`.
        max_steps: Hard cap on control steps.
        belief_correction: Phase-belief correction gain; see
            :meth:`~tpgpt.policy.gp_policy.GPPolicy.update_time_belief`.
        stop_time_belief: Phase at which the rollout terminates.
        video_path: Optional mp4 to stream frames to.
        rotational_stiffness: Orientation gain while the hand is grasping or
            placing.
        transit_rotational_scale: Fraction of that gain used while the object is
            simply being carried.

            The exact hand angle matters at the two ends of the task and not in
            between, and the demonstration says so itself: the teacher is
            compliant through free space and stiff at the grasp and the
            insertion. That profile is transported for translation
            (Sec. III-G) and was being ignored for rotation, which was held at
            a fixed gain the whole way.

            Holding it rigidly is not free. The warped path can pass through
            places the arm can reach but cannot reach *with the commanded hand
            pose*; fighting the orientation there costs enough position error to
            shut the lag gate, and the run deadlocks. Relaxing the gain in
            transit lets the arm give up a few degrees of wrist angle it does
            not need in order to keep following the path it does.
        approach_steps: Steps spent reaching the transported start pose before
            the policy takes over. Set to ``0`` to hand over immediately, which
            is the ablation showing why the phase is needed.
        approach_stiffness: Translational stiffness used during the approach.
        speed_limit_factor: Clamp the commanded speed, and the attractor
            offset that realises it, to this multiple of the fastest label.

            The offset clamp is not cosmetic. In a Cartesian impedance
            controller the attractor *is* the reference, so the restoring speed
            is set by the position error and the gain ratio ``K/D`` -- here
            about 12 per second. An attractor 24 mm from the end effector
            therefore commands 0.29 m/s regardless of what velocity the policy
            asked for, and a 0.1 m deviation would command 1.2 m/s. Measured
            without the clamp, the arm reached 0.37 m/s while the policy was
            commanding 0.002 m/s, overshot the grasp and lost the path.

            The demonstration never hit this because its attractor was
            interpolated smoothly between waypoints and so stayed close to the
            robot. Clamping the offset restores that property: the robot
            returns to the path at no more than the demonstrated speed.
        lag_tolerance: Arm-to-attractor distance beyond which progress is held
            back; see the lag gate above. Set to ``None`` to disable, which is
            the ablation showing why it is needed.
        min_phase_progress: Fraction of the demonstration's own phase rate below
            which the run counts as making no progress. The watchdog used to
            test the gate for being exactly zero, which a partly-open gate never
            is, so a crawl was invisible to it.
        infeasible_fallback: Handle commanded poses the arm cannot achieve.

            The controller has no inverse kinematics: it turns a pose error into
            a force and pulls. Against a pose that does not exist for this arm --
            the right point with a hand angle the elbow cannot produce -- it
            pulls forever, settling about 38 mm away with a standing error that
            no amount of stiffness removes. Measured by replaying warped paths
            pose by pose, roughly a quarter of a trajectory is like this, and it
            surfaces as a *control* symptom: the lag gate sees the residual,
            shuts, and the run is reported as a stall.

            Two tiers, cheapest first, and both only ever run when the gate is
            already partly shut -- a healthy run pays nothing:

            1. **Relax the orientation.** If the position is reachable and only
               the hand angle is not, the orientation command is dropped for as
               long as that holds. This is the common case, and it is nearly
               free: the demonstration is already compliant in transit
               (``transit_rotational_scale``), so a few degrees of wrist angle
               in free space costs nothing.
            2. **Skip to the next feasible pose.** If the position itself is out
               of reach, the attractor is steered along to the next label the
               arm *can* hold, at no more than the demonstrated speed, and the
               phase jumps with it on arrival so the gripper schedule stays in
               step with where the hand actually is.

        feasibility_stride: Control steps between feasibility checks, tested
            only while the gate is shut.
        feasibility_tolerance: Position tolerance for the check. Matched to the
            controller's own tracking, not to solver precision -- a pose the
            controller would have reached must not be called infeasible.
        skip_arrival: How close the attractor must come to a skip target before
            the phase jumps to it.
        tool_offset: Where this hand holds an object, in ``grip_site``
            coordinates. When given, the policy is taken to describe the
            **fingertip** path rather than the wrist path: the gate and the
            policy query use the measured fingertip position, and the attractor
            is converted back to a wrist target at the moment it is commanded.

            This matters because the keypoints pin the transportation map at the
            fingertip contact point while the demonstration is recorded at the
            wrist, and the two are a hand-specific lever arm apart -- 41 mm on a
            Panda, 117 mm on a UMI. A map is exact only where it was fitted, so
            warping the wrist path gives up the one guarantee the keypoints
            provide, at exactly the moment it is needed.

            ``None`` reproduces the original behaviour exactly, which is what
            the validated reshelving campaign runs.
        probe: Optional ``probe(env) -> dict`` called once per control step. Its
            values are stacked into ``metadata["probe"]``.

            This is how a failure gets attributed to a stage. The outcome alone
            cannot separate a hand that closed on nothing from one that gripped
            the object and set it down in the wrong place: both end with the
            object away from its slot. Watching the object's own height and the
            jaw opening every step tells the two apart, and nothing else does.

    Returns:
        A :class:`SimRollout` including the task outcome.
    """
    controller = CartesianImpedanceController(env)
    controller.reset()
    dt = 1.0 / env.control_freq
    offset = (
        np.zeros(3) if tool_offset is None
        else np.asarray(tool_offset, dtype=float)
    )

    def tool_state():
        """Measured fingertip position and the hand's rotation."""
        position, rotation, _, _ = controller.eef_state()
        return position + rotation @ offset, rotation

    positions, attractors, velocities = [], [], []
    phases, stds, grippers, forces = [], [], [], []
    probes: list[dict] = []
    # Per-step diagnostics for the execution study. Kept under a `trace`
    # sub-dict in the metadata, matching the `probe` convention, so a few
    # hundred floats per run do not inflate every campaign manifest.
    traces: dict[str, list] = {
        k: [] for k in
        ("gate", "behind", "expected", "max_lag", "clamped", "anchor_gain")
    }
    writer = FrameWriter(video_path, fps=env.control_freq) if video_path else None
    phase = 0.0
    obs = env._get_observations()

    speed_limit = speed_limit_factor * float(
        np.linalg.norm(policy.labels.velocities, axis=1).max()
    )
    max_label_speed = speed_limit / max(speed_limit_factor, 1e-12)
    law = ExecutionLaw(
        dt=dt, speed_limit=speed_limit, max_label_speed=max_label_speed,
        lag_tolerance=lag_tolerance, attractor_law=attractor_law,
        anchor_gain=anchor_gain, anchor_gated=anchor_gated, query_at=query_at,
        skip_arrival=skip_arrival,
    )
    start_position = policy.labels.positions[0]
    start_rotation = (
        policy.labels.orientations[0] if policy.labels.orientations is not None else None
    )
    approach_gap = float(np.linalg.norm(tool_state()[0] - start_position))
    if approach_steps > 0:
        obs = _approach(
            env,
            controller,
            start_position - (
                start_rotation @ offset if start_rotation is not None else offset
            ),
            start_rotation,
            approach_steps,
            approach_stiffness,
            rotational_stiffness,
            rotational_damping,
            writer,
            camera,
        )
    attractor = policy.labels.positions[0].copy()
    stalled = False
    relax_orientation, skip_index = False, None
    relaxed_steps, skipped_steps, skips = 0, 0, 0
    # What the clock would gain over the patience window with the gate fully
    # open, scaled down: anything slower than this fraction of nominal is a
    # crawl, not progress. Derived from the demonstration's own pace so it
    # scales with how long the task is rather than being a magic number.
    watchdog = StallWatchdog(
        stall_patience=stall_patience, sag_epsilon=sag_epsilon,
        max_sag_rebaselines=max_sag_rebaselines,
        phase_epsilon=phase_epsilon_from(
            policy.labels, dt, stall_patience, min_phase_progress
        ),
    )
    # Compliance is the inverse of stiffness, so a *large* value here means the
    # teacher was being soft. The midpoint of the demonstration's own range
    # separates its free-space segments from its grasp and insertion ones
    # without hard-coding a number.
    stiff_threshold = stiffness_threshold(policy.labels)

    for step in range(max_steps):
        # In tool-offset mode this is the fingertip, which is the frame the
        # policy and the keypoints are both expressed in.
        position, hand_rotation = tool_state()
        query_phase = np.array([phase]) if policy.use_time_belief else None
        # By default the field is integrated at the attractor rather than at
        # the measured position, which is what keeps the query on the manifold
        # the policy was fitted on. ``query_at="measured"`` closes the loop
        # through physics instead; see the module docstring for the measured
        # cost of doing that without a restoring term.
        prediction = policy.predict(
            query_position(attractor, position, law.query_at)[None], query_phase
        )

        # Lag gate: hold progress while the arm is behind its attractor, so the
        # gripper never acts on a pose the robot has not reached.
        # Read out of the prediction before anything consults them: the
        # feasibility fallback below asks whether the arm can hold this
        # orientation, so it has to exist by then.
        gripper_command = 1.0 if float(prediction.gripper[0]) > 0 else -1.0
        rotation = (
            prediction.orientation[0] if prediction.orientation is not None else None
        )

        decision = gate_step(
            position, attractor, prediction, law, static_sag=watchdog.static_sag
        )
        velocity, gate = decision.velocity, decision.gate
        # --- infeasible-pose fallback ------------------------------------
        # Only consulted while the gate is shut, so a run that is tracking
        # normally never pays for it.
        if infeasible_fallback and lag_tolerance and gate < 1.0:
            if step % feasibility_stride == 0:
                relax_orientation, skip_index = _feasibility_action(
                    env, controller, attractor, rotation, offset, policy,
                    phase, feasibility_tolerance, skip_index,
                )
            if relax_orientation:
                relaxed_steps += 1

        steer_to = (
            np.asarray(policy.labels.positions[skip_index], dtype=float)
            if skip_index is not None else None
        )
        step_out = advance_step(
            position, attractor, decision, law, prediction,
            static_sag=watchdog.static_sag, steer_to=steer_to,
        )
        attractor = step_out.attractor
        if skip_index is not None:
            if step_out.skip_arrived:
                # Arrived: hand the clock over so the gripper schedule matches
                # where the arm now is, and resume the policy.
                if policy.labels.time_belief is not None:
                    phase = float(policy.labels.time_belief[skip_index])
                skip_index = None
                skips += 1
            else:
                skipped_steps += 1
        traces["gate"].append(gate)
        traces["behind"].append(decision.behind)
        traces["expected"].append(decision.expected)
        traces["max_lag"].append(step_out.max_lag)
        traces["clamped"].append(float(step_out.clamped))
        traces["anchor_gain"].append(step_out.anchor_gain)
        # Firm about orientation where it matters -- reaching for the object and
        # setting it down -- and compliant while merely carrying it. The
        # demonstration's own translational stiffness says which is which: it
        # rises for the grasp and the insertion.
        firm = decision.compliance
        carrying = gripper_command > 0 and firm > 0.0 and firm >= stiff_threshold
        rotational_gain = (
            rotational_stiffness * transit_rotational_scale
            if carrying else rotational_stiffness
        )

        # Back out to the wrist, which is what the controller commands. The
        # desired hand rotation is used when it exists, so the conversion uses
        # the pose being asked for rather than the one the arm happens to hold.
        # A relaxed segment commands no orientation at all, so the wrist is
        # free to find whatever angle reaches the point.
        commanded_rotation = None if relax_orientation else rotation
        commanded = attractor - (
            commanded_rotation if commanded_rotation is not None else hand_rotation
        ) @ offset
        action = controller.action(
            commanded,
            prediction.stiffness[0],
            prediction.damping[0],
            gripper=gripper_command,
            rotation_desired=commanded_rotation,
            rotational_stiffness=rotational_gain,
            rotational_damping=rotational_damping * (rotational_gain / rotational_stiffness),
        )
        obs, _, _, _ = env.step(action)

        positions.append(position.copy())
        attractors.append(attractor.copy())
        velocities.append(velocity.copy())
        phases.append(phase)
        stds.append(prediction.velocity_std[0].copy())
        grippers.append(gripper_command)
        forces.append(_contact_force(env))
        if probe is not None:
            probes.append(probe(env))
        if writer is not None:
            writer.append(observation_frame(obs, camera))

        phase = advance_phase(
            policy, belief_position(attractor, position, law.query_at),
            phase, dt, gate, belief_correction,
        )

        # Progress is measured on the **phase**, not on the gate being exactly
        # zero, and the difference is not academic. A gate held at 0.05 by a
        # constant sag advances the clock at a twentieth of nominal: the run
        # neither deadlocks -- so the old ``gate == 0`` test never fired -- nor
        # arrives, and it burns the whole step budget crawling. Measured on
        # three can runs, the object was carried to within 5, 11 and 30 mm of
        # its slot and then held in the air until the budget ran out, because
        # the phase never reached the segment that opens the fingers. All three
        # were reported as a stalled policy, which blamed the transport for a
        # gate that was doing its job too well.
        if lag_tolerance and watchdog.update(phase, decision.behind, decision.expected):
            stalled = True
            break
        if phase >= stop_time_belief:
            break

    if writer is not None:
        writer.close()

    succeeded, offset = (score or _reshelving_score)(env)
    return SimRollout(
        positions=np.stack(positions),
        attractors=np.stack(attractors),
        velocities=np.stack(velocities),
        time_belief=np.array(phases),
        velocity_std=np.stack(stds),
        gripper=np.array(grippers),
        contact_force=np.array(forces),
        success=bool(succeeded),
        final_offset=offset,
        metadata={
            "steps": len(positions),
            "dt": dt,
            "approach_gap": approach_gap,
            "approach_steps": int(approach_steps),
            "speed_limit": speed_limit,
            "attractor_integrated": attractor_law == "integrate",
            "tool_offset": offset.tolist(),
            "infeasible_fallback": bool(infeasible_fallback),
            "orientation_relaxed_steps": int(relaxed_steps),
            "skip_steps": int(skipped_steps),
            "skips": int(skips),
            "lag_tolerance": lag_tolerance,
            "terminated_on_phase": bool(phase >= stop_time_belief),
            "blocked": bool(stalled),
            "blocked_after_steps": len(positions) if stalled else None,
            "final_lag": float(np.linalg.norm(positions[-1] - attractors[-1])),
            "stall_patience": int(stall_patience),
            "phase_epsilon": float(watchdog.phase_epsilon),
            "final_phase": float(phase),
            "static_sag": float(watchdog.static_sag),
            "sag_rebaselines": int(watchdog.rebaselines),
            **law.to_dict(),
            "trace": {k: np.array(v) for k, v in traces.items()},
            "placement_error_xy": float(np.linalg.norm(offset[:2])),
            "placement_error": float(np.linalg.norm(offset)),
            "video": str(writer.path) if writer and writer.n_frames else None,
            "probe": {
                key: np.array([p[key] for p in probes])
                for key in (probes[0] if probes else {})
            },
        },
    )


def _feasibility_action(
    env, controller, attractor, rotation, offset, policy, phase, tolerance, skip_index
):
    """Decide what to do about an attractor the arm may not be able to reach.

    Returns ``(relax_orientation, skip_index)``. A skip already in progress is
    left alone -- re-deciding every few steps would make the target flicker.

    The IK here is seeded from the arm's current configuration and capped at a
    few iterations: it is asking "can the arm be *there*, roughly, from here",
    not solving for a pose to command.
    """
    from tpgpt.sim.kinematics import solve_ik

    if skip_index is not None:
        return False, skip_index

    wrist = attractor - (rotation if rotation is not None else np.eye(3)) @ offset
    with_orientation = solve_ik(
        env, wrist, rotation, position_tolerance=tolerance, max_iterations=40
    )
    if with_orientation.reachable:
        return False, None

    position_only = solve_ik(
        env, wrist, None, position_tolerance=tolerance, max_iterations=40
    )
    if position_only.reachable:
        # The point is fine; the hand angle is not. Give up the angle.
        return True, None

    # Neither: look ahead for the next label the arm can hold and go there.
    return False, _next_feasible(env, policy, phase, offset, tolerance)


def _next_feasible(env, policy, phase, offset, tolerance, lookahead: int = 40, stride: int = 5):
    """Index of the next label along the path the arm can actually reach.

    Searched forward from where the clock currently is, coarsely: the point is
    to leave an unreachable region, not to find the nearest millimetre.
    ``None`` when nothing ahead is reachable either, in which case the caller
    carries on and the watchdog will end the run -- which is the honest outcome,
    since the rest of the path is not executable.
    """
    from tpgpt.sim.kinematics import solve_ik

    labels = policy.labels
    if labels.time_belief is None or len(labels.positions) == 0:
        return None
    start = int(np.searchsorted(np.asarray(labels.time_belief), phase))
    rotations = labels.orientations
    for i in range(start + stride, min(start + lookahead, len(labels.positions)), stride):
        rotation = rotations[i] if rotations is not None else None
        wrist = labels.positions[i] - (
            rotation if rotation is not None else np.eye(3)
        ) @ offset
        if solve_ik(
            env, wrist, rotation, position_tolerance=tolerance, max_iterations=40
        ).reachable:
            return i
    return None


#: Kept as an alias for one commit: the parallel keypoint session's working
#: copies may still reference the private name, and a rename would surface as an
#: ImportError in *their* tree rather than here.
_compliance_norm = compliance_norm


def _approach(
    env,
    controller,
    position,
    rotation,
    n_steps,
    stiffness,
    rotational_stiffness,
    rotational_damping,
    writer,
    camera,
):
    """Drive the arm to the transported start pose with the gripper open."""
    K = np.eye(3) * stiffness
    D = np.eye(3) * (2.0 * np.sqrt(stiffness) * 0.9)
    start, _, _, _ = controller.eef_state()
    obs = env._get_observations()
    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        action = controller.action(
            start + alpha * (position - start),
            K,
            D,
            gripper=-1.0,
            rotation_desired=rotation,
            rotational_stiffness=rotational_stiffness,
            rotational_damping=rotational_damping,
        )
        obs, _, _, _ = env.step(action)
        if writer is not None:
            writer.append(observation_frame(obs, camera))
    return obs


def _contact_force(env) -> float:
    """Magnitude of the external wrench on the end-effector body.

    Sec. V-C reads the contact force from an observer over the measured joint
    torques rather than a force/torque sensor; the same quantity is available
    directly here.

    ``cfrc_ext`` is filled by ``mj_rnePostConstraint``, which ``mj_step`` runs
    only when something in the model asks for it. With no such sensor declared
    the array stays at zero, so this read used to report **0 N throughout a
    rollout that was visibly carrying a 400 g box** -- a plausible-looking
    number for "not touching anything", which is the worst way for a diagnostic
    to fail. Computing it explicitly costs one call per control step.
    """
    try:
        import mujoco

        model, data = env.sim.model._model, env.sim.data._data
        mujoco.mj_rnePostConstraint(model, data)
        body_id = env.sim.model.body_name2id("gripper0_right_eef")
        return float(np.linalg.norm(data.cfrc_ext[body_id][3:]))
    except Exception:  # pragma: no cover - embodiment without that body name
        return float("nan")


def _reshelving_score(env) -> tuple[bool, np.ndarray]:
    """Default scoring: the reshelving task's own success test.

    Kept as the default so the validated Sec. V-A path is unchanged. Any other
    scene passes its own ``score`` to :func:`rollout_policy`; the signature is
    ``score(env) -> (succeeded, offset_vector)``, where the offset is from the
    object to where it was meant to go, and its lateral norm is reported as the
    placement error.
    """
    return bool(env._check_success()), env.product_position - env.goal_position


def slot_score(object_name: str, slot: str, tolerance: float = 0.06):
    """Scoring for a scene with named shelf slots.

    Success is the object resting in the named slot. The offset is measured to
    that slot's centre, so the placement error means the same thing it does for
    reshelving and the two are comparable.
    """

    def score(env) -> tuple[bool, np.ndarray]:
        target = env.slot_poses()[slot]
        return (
            bool(env.is_object_in_slot(object_name, slot, tolerance=tolerance)),
            env.object_position(object_name) - target,
        )

    return score
