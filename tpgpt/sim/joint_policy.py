"""Development executor: refitted GP references with joint-position tracking.

Unlike direct label replay, this queries the learned policy at the measured
tool state and phase. Tracking error gates its learned phase rate. Joint gains
are fixed: this executor does not claim to implement transported compliance.
"""
from __future__ import annotations

import numpy as np

from tpgpt.sim.kinematics import initial_ik_seed, solve_ik
from tpgpt.sim.replay import _joint_action, _site_position, _site_rotation
from tpgpt.sim.rollout import SimRollout


def rollout_joint_policy(env, policy, tool_offset=None, arm="right", max_steps=2400,
                         phase_speed=.25, tracking_tolerance=.015, approach_steps=100,
                         stall_steps=250, grasp_gate=None, gate_max_steps=200,
                         score=None, probe=None, settle_steps=40,
                         require_ik_reachable=False, query_at="measured"):
    if phase_speed <= 0 or tracking_tolerance <= 0 or max_steps < 1:
        raise ValueError("phase speed, tracking tolerance and step budget must be positive")
    if query_at not in ("measured", "attractor"):
        raise ValueError("query_at must be measured or attractor")
    labels = policy.labels
    offset = np.zeros(3) if tool_offset is None else np.asarray(tool_offset, float)
    robot = env.robots[0]
    controller = robot.composite_controller.part_controllers[arm]
    qindex = np.asarray(controller.qpos_index)
    seed = initial_ik_seed(env, arm)
    dt = 1. / env.control_freq

    def measured():
        R = _site_rotation(env, robot, arm)
        return _site_position(env, robot, arm) + R @ offset

    R = labels.orientations[0]
    initial = solve_ik(env, labels.positions[0] - R @ offset, R,
                       arm=arm, seed_qpos=seed, position_tolerance=.005)
    for _ in range(approach_steps):
        # Limit the connection speed rather than teleporting to an IK state.
        q = np.asarray(env.sim.data.qpos[qindex])
        target_q = q + np.clip(initial.qpos - q, -.05, .05)
        env.step(_joint_action(env, robot, arm, target_q, -1.))
        if np.linalg.norm(measured() - labels.positions[0]) < .005:
            break
    seed = np.asarray(env.sim.data.qpos[qindex]).copy()
    phase = float(labels.time_belief[0])
    end = float(labels.time_belief[-1])
    hand, targets, velocities, phases, std, commands, probes = [], [], [], [], [], [], []
    waited, stalled, unreachable, reason = 0, 0, 0, "step_budget"
    pinched = False
    grasp_wait_total = 0
    last_q = seed.copy()
    last_command = -1.
    attractor = np.asarray(labels.positions[0]).copy()
    for _ in range(max_steps):
        x = measured()
        query = x if query_at == "measured" else attractor
        pred = policy.predict(query[None], np.array([phase]))
        target = np.asarray(pred.reference[0])
        attractor = target.copy()
        rotation = pred.orientation[0]
        command = 1. if pred.gripper is not None and pred.gripper[0] > 0 else -1.
        ik = solve_ik(env, target - rotation @ offset, rotation, arm=arm,
                      seed_qpos=seed, position_tolerance=.005)
        if ik.reachable:
            seed = np.asarray(ik.qpos).copy()
        else:
            unreachable += 1
        last_q = np.asarray(ik.qpos).copy()
        last_command = command
        env.step(_joint_action(env, robot, arm, last_q, command))
        actual = measured()
        error = float(np.linalg.norm(actual - target))
        held = command > 0 and grasp_gate is not None and bool(grasp_gate(env))
        pinched = pinched or held
        waiting = command > 0 and grasp_gate is not None and not pinched
        if waiting:
            waited += 1
            grasp_wait_total += 1
        rate = float(pred.time_rate[0]) if pred.time_rate is not None else 1. / (len(labels.positions) * dt)
        # IK uses a tighter 5 mm convergence criterion than the executor's
        # measured tracking tolerance. A best-effort IK flag alone must not
        # freeze an arm that is physically tracking inside that tolerance.
        advance = ((ik.reachable or not require_ik_reachable) and
                   error <= tracking_tolerance and not waiting)
        new_phase = min(end, phase + max(0., rate) * dt * phase_speed) if advance else phase
        stalled = stalled + 1 if new_phase <= phase + 1e-8 else 0
        hand.append(actual.copy()); targets.append(target.copy())
        velocities.append(pred.velocity[0].copy()); std.append(pred.velocity_std[0].copy())
        phases.append(phase); commands.append(command)
        if probe is not None:
            probes.append(probe(env))
        phase = new_phase
        if waited >= gate_max_steps:
            reason = "grasp_not_established"
            break
        if stalled >= stall_steps:
            reason = "tracking_stalled"
            break
        if phase >= end:
            reason = "phase_complete"
            break
    for _ in range(settle_steps):
        env.step(_joint_action(env, robot, arm, last_q, last_command))
    success, final = score(env) if score is not None else (False, np.zeros(3))
    return SimRollout(
        positions=np.asarray(hand), attractors=np.asarray(targets),
        velocities=np.asarray(velocities), time_belief=np.asarray(phases),
        velocity_std=np.asarray(std), gripper=np.asarray(commands),
        contact_force=np.full(len(hand), np.nan),
        success=bool(success and reason == "phase_complete" and last_command < 0),
        final_offset=final,
        metadata={"executor": "joint_policy", "policy_query": query_at + "_and_phase",
                  "transported_compliance_executed": False, "phase_speed": phase_speed,
                  "tracking_tolerance": tracking_tolerance, "termination": reason,
                  "require_ik_reachable": require_ik_reachable,
                  "steps": len(hand), "unreachable_steps": unreachable,
                  "grasp_gate_steps": grasp_wait_total, "terminated_on_phase": reason == "phase_complete",
                  "position_control": True, "tool_offset": offset,
                  "placement_error_xy": float(np.linalg.norm(final[:2])),
                  "probe": {k: np.array([p[k] for p in probes]) for k in (probes[0] if probes else {})}},
    )
