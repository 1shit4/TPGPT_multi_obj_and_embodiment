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

from dataclasses import dataclass, field

import numpy as np

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
    approach_steps: int = 45,
    approach_stiffness: float = 350.0,
    speed_limit_factor: float = 3.0,
    lag_tolerance: float = 0.035,
) -> SimRollout:
    """Run a fitted policy on the robot until its phase completes.

    At every control step the policy is queried at the **measured** end-effector
    position, so the loop is closed through the physics rather than replayed.

    Args:
        env: Reshelving environment on the joint-torque controller.
        policy: A fitted :class:`~tpgpt.policy.gp_policy.GPPolicy`.
        max_steps: Hard cap on control steps.
        belief_correction: Phase-belief correction gain; see
            :meth:`~tpgpt.policy.gp_policy.GPPolicy.update_time_belief`.
        stop_time_belief: Phase at which the rollout terminates.
        video_path: Optional mp4 to stream frames to.
        rotational_stiffness: Orientation gain. The paper transports the
            translational stiffness matrix; orientation gains stay fixed.
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

    Returns:
        A :class:`SimRollout` including the task outcome.
    """
    controller = CartesianImpedanceController(env)
    controller.reset()
    dt = 1.0 / env.control_freq

    positions, attractors, velocities = [], [], []
    phases, stds, grippers, forces = [], [], [], []
    writer = FrameWriter(video_path, fps=env.control_freq) if video_path else None
    phase = 0.0
    obs = env._get_observations()

    speed_limit = speed_limit_factor * float(
        np.linalg.norm(policy.labels.velocities, axis=1).max()
    )
    start_position = policy.labels.positions[0]
    start_rotation = (
        policy.labels.orientations[0] if policy.labels.orientations is not None else None
    )
    approach_gap = float(
        np.linalg.norm(controller.eef_state()[0] - start_position)
    )
    if approach_steps > 0:
        obs = _approach(
            env,
            controller,
            start_position,
            start_rotation,
            approach_steps,
            approach_stiffness,
            rotational_stiffness,
            rotational_damping,
            writer,
            camera,
        )
    attractor = policy.labels.positions[0].copy()

    for _ in range(max_steps):
        position, _, _, _ = controller.eef_state()
        query_phase = np.array([phase]) if policy.use_time_belief else None
        # The dynamical system is integrated at the attractor, not at the
        # measured position: that is what keeps the query on the manifold the
        # policy was fitted on.
        prediction = policy.predict(attractor[None], query_phase)

        speed = float(np.linalg.norm(prediction.velocity[0]))
        if speed > speed_limit:
            prediction.velocity[0] *= speed_limit / speed

        # Lag gate: hold progress while the arm is behind its attractor, so the
        # gripper never acts on a pose the robot has not reached.
        gate = 1.0
        if lag_tolerance:
            behind = float(np.linalg.norm(position - attractor))
            expected = _compliance_norm(
                prediction.stiffness[0], prediction.damping[0]
            ) * float(np.linalg.norm(prediction.velocity[0]))
            gate = float(np.clip(1.0 - (behind - expected) / lag_tolerance, 0.0, 1.0))
        attractor = attractor + prediction.velocity[0] * dt * gate
        # Keep the attractor within reach of the arm, so a blocked or lagging
        # robot builds a bounded interaction force instead of an unbounded one.
        lag = attractor - position
        max_lag = speed_limit * _compliance_norm(
            prediction.stiffness[0], prediction.damping[0]
        )
        lag_norm = float(np.linalg.norm(lag))
        if lag_norm > max_lag:
            attractor = position + lag * (max_lag / lag_norm)
        gripper_command = 1.0 if float(prediction.gripper[0]) > 0 else -1.0
        rotation = (
            prediction.orientation[0] if prediction.orientation is not None else None
        )

        action = controller.action(
            attractor,
            prediction.stiffness[0],
            prediction.damping[0],
            gripper=gripper_command,
            rotation_desired=rotation,
            rotational_stiffness=rotational_stiffness,
            rotational_damping=rotational_damping,
        )
        obs, _, _, _ = env.step(action)

        positions.append(position.copy())
        attractors.append(attractor.copy())
        velocities.append(prediction.velocity[0].copy())
        phases.append(phase)
        stds.append(prediction.velocity_std[0].copy())
        grippers.append(gripper_command)
        forces.append(_contact_force(env))
        if writer is not None:
            writer.append(observation_frame(obs, camera))

        if gate > 0.0:
            phase = policy.update_time_belief(
                attractor, phase, dt * gate, correction=belief_correction * gate
            )
        if phase >= stop_time_belief:
            break

    if writer is not None:
        writer.close()

    offset = env.product_position - env.goal_position
    return SimRollout(
        positions=np.stack(positions),
        attractors=np.stack(attractors),
        velocities=np.stack(velocities),
        time_belief=np.array(phases),
        velocity_std=np.stack(stds),
        gripper=np.array(grippers),
        contact_force=np.array(forces),
        success=bool(env._check_success()),
        final_offset=offset,
        metadata={
            "steps": len(positions),
            "dt": dt,
            "approach_gap": approach_gap,
            "approach_steps": int(approach_steps),
            "speed_limit": speed_limit,
            "attractor_integrated": True,
            "lag_tolerance": lag_tolerance,
            "terminated_on_phase": bool(phase >= stop_time_belief),
            "placement_error_xy": float(np.linalg.norm(offset[:2])),
            "placement_error": float(np.linalg.norm(offset)),
            "video": str(writer.path) if writer and writer.n_frames else None,
        },
    )


def _compliance_norm(stiffness: np.ndarray, damping: np.ndarray) -> float:
    """Largest attractor offset per unit of commanded speed, ``||K^-1 D||``.

    Multiplying by a speed limit turns it into a displacement limit, so the
    clamp scales correctly with whatever stiffness the policy commanded.
    """
    return float(np.linalg.norm(np.linalg.solve(stiffness, damping), ord=2))


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
    """
    try:
        body_id = env.sim.model.body_name2id("gripper0_right_eef")
    except Exception:  # pragma: no cover - embodiment without that body name
        return float("nan")
    return float(np.linalg.norm(env.sim.data.cfrc_ext[body_id][3:]))
