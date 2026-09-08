"""Executing a transported path under position control, as a diagnostic ceiling.

The normal rollout (:mod:`tpgpt.sim.rollout`) does two things at once: it
integrates a learned velocity field into a moving attractor, and it tracks that
attractor with a Cartesian impedance. Both are deliberate -- the paper's policy
is a dynamical system, and compliance is what lets a hand absorb contact -- and
both introduce error that has nothing to do with the transportation map:

* the impedance lags its attractor while moving, by ``||K^-1 D|| v`` -- about
  24 mm at the demonstration's peak speed (section 2.8). That lag is a
  steady-state property of tracking a *moving* setpoint without velocity
  feed-forward, not a settling error, and section 2.7 argues it may be the
  desired behaviour rather than an error at all;
* the integrated attractor can wander from the transported labels, since
  nothing corrects it back onto them, and the jaws can close at a moment when
  the attractor is off a path that does pass through the object.

  How large either term actually is has **not** been credibly measured. The
  numbers this docstring used to quote came from campaigns withdrawn in section
  7.26, and they were measured as the difference between two separately
  minimised distances rather than pointwise, which can come out negative. This
  module is the instrument for measuring them properly.

Those obscure the question the keypoints are actually being asked: *does the
warped path go to the right place?* This module answers it by removing all
three. The transported labels are replayed **directly**, pose by pose, through
inverse kinematics and a stiff joint-position controller, with the gripper
following the labels' own schedule.

**This is a measurement, not a better robot.** A stiff controller replaying a
path with a few millimetres of error drives the object into the shelf rather
than yielding to it, and has no way to absorb the contact of closing on an
object. What it gives is an **upper bound**: the success rate the keypoints and
the map could support if the controller were perfect. A failure here is a
failure of the map; a run that succeeds here and fails under impedance is a
control problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import copy

import numpy as np

#: Joint-position gain. Well above robosuite's default of 50, since the point of
#: this mode is to track the commanded path as closely as the arm allows.
POSITION_GAIN = 300.0

#: Radians of joint travel corresponding to an action of 1.
JOINT_ACTION_SCALE = 0.5


def make_position_controller_config(base_config: dict, arm: str = "right") -> dict:
    """Rewrite a composite config to drive ``arm`` by joint position.

    Mirrors :func:`~tpgpt.sim.controllers.cartesian_impedance.make_torque_controller_config`,
    including the same trap: robosuite rescales every action from
    ``[input_min, input_max]`` onto ``[output_min, output_max]``, so the output
    range has to match the units actually being commanded.
    """
    config = copy.deepcopy(base_config)
    gripper = config["body_parts"].get(arm, {}).get("gripper", {"type": "GRIP"})
    config["body_parts"][arm] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": JOINT_ACTION_SCALE,
        "output_min": -JOINT_ACTION_SCALE,
        "kp": POSITION_GAIN,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 1000],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper,
    }
    return config


@dataclass
class ReplayResult:
    """What a position-controlled replay did."""

    positions: np.ndarray
    targets: np.ndarray
    gripper: np.ndarray
    success: bool = False
    final_offset: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.positions.shape[0]


def replay_labels(
    env,
    labels,
    tool_offset=None,
    arm: str = "right",
    settle_steps: int = 8,
    score=None,
    probe=None,
    ik_tolerance: float = 5e-3,
) -> ReplayResult:
    """Drive the arm along a label set pose by pose, under position control.

    Args:
        labels: Transported :class:`~tpgpt.transport.labels.PolicyLabels`. Its
            positions are followed directly -- no policy is fitted and no
            attractor is integrated.
        tool_offset: Where this hand holds an object, in ``grip_site``
            coordinates. When the labels describe the fingertip path (which they
            do downstream of the tool-frame change of section 7.20) this is
            subtracted to get the wrist pose IK must solve for.
        settle_steps: Control steps held at each commanded pose. The arm needs a
            few steps to converge on a joint target; commanding the next one
            immediately turns a position controller back into a velocity one.
        score: ``score(env) -> (succeeded, offset)``, as for
            :func:`~tpgpt.sim.rollout.rollout_policy`.

    Returns:
        A :class:`ReplayResult`.
    """
    from tpgpt.sim.kinematics import solve_ik

    offset = np.zeros(3) if tool_offset is None else np.asarray(tool_offset, float)
    robot = env.robots[0]
    controller = robot.composite_controller.part_controllers[arm]
    qpos_index = np.asarray(controller.qpos_index)

    positions = np.asarray(labels.positions, dtype=float)
    rotations = (
        np.asarray(labels.orientations, dtype=float)
        if labels.orientations is not None else None
    )
    grip = (
        np.asarray(labels.gripper, dtype=float)
        if labels.gripper is not None else np.full(len(positions), -1.0)
    )

    recorded, targets, commands, probes = [], [], [], []
    reachable: list[bool] = []
    unreachable = 0
    seed = np.array(env.sim.data.qpos[qpos_index])

    for i, target in enumerate(positions):
        rotation = rotations[i] if rotations is not None else None
        # The labels are a fingertip path; IK solves for the wrist.
        wrist = target - (rotation @ offset if rotation is not None else offset)
        result = solve_ik(
            env, wrist, rotation, arm=arm,
            position_tolerance=ik_tolerance, seed_qpos=seed,
        )
        reachable.append(bool(result.reachable))
        if not result.reachable:
            unreachable += 1
        # Even an unconverged solve is the closest the arm can get, so it is
        # still the right thing to command -- refusing to move would report a
        # kinematic limit as a tracking failure.
        seed = np.asarray(result.qpos, dtype=float)

        gripper_command = 1.0 if grip[i] > 0 else -1.0
        for _ in range(settle_steps):
            action = _joint_action(env, robot, arm, seed, gripper_command)
            env.step(action)

        site = _site_position(env, robot, arm)
        measured_rotation = _site_rotation(env, robot, arm)
        recorded.append(site + measured_rotation @ offset)
        targets.append(target.copy())
        commands.append(gripper_command)
        if probe is not None:
            probes.append(probe(env))

    succeeded, final = (score(env) if score is not None else (False, np.zeros(3)))
    recorded = np.stack(recorded)
    tracking = np.linalg.norm(recorded - np.stack(targets), axis=1)
    return ReplayResult(
        positions=recorded,
        targets=np.stack(targets),
        gripper=np.array(commands),
        success=bool(succeeded),
        final_offset=final,
        metadata={
            "steps": len(recorded) * settle_steps,
            "waypoints": len(recorded),
            "settle_steps": int(settle_steps),
            "unreachable_waypoints": int(unreachable),
            "tracking_error_mean": float(tracking.mean()),
            "tracking_error_max": float(tracking.max()),
            # Split by whether IK could hold the commanded pose at all. If the
            # two halves differ sharply the residual is kinematic -- the arm
            # cannot be there with its hand at that angle -- and no controller,
            # however stiff, will close it.
            "tracking_error_reachable": (
                float(tracking[np.asarray(reachable)].mean())
                if any(reachable) else float("nan")
            ),
            "tracking_error_unreachable": (
                float(tracking[~np.asarray(reachable)].mean())
                if not all(reachable) else float("nan")
            ),
            "reachable_fraction": float(np.mean(reachable)) if reachable else 0.0,
            # **Per waypoint, not just aggregated.** A reachable fraction says
            # how much of a path the arm cannot hold; it cannot say *where*, and
            # "the arm could not follow this during the carry" is a different
            # problem from "it could not reach the grasp". Both arrays are in
            # label order, so an unreachable run is localisable against
            # ``carry_indices`` -- approach, grasp, lift, transit or insertion.
            #
            # Added because a bread replay failed at 28% unreachable with a
            # 111 mm residual on those poses and the aggregate could not say
            # which segment it was.
            "reachable_per_waypoint": np.asarray(reachable, dtype=bool),
            "tracking_error_per_waypoint": tracking,
            "position_control": True,
            "tool_offset": offset.tolist(),
            "placement_error_xy": float(np.linalg.norm(final[:2])),
            "placement_error": float(np.linalg.norm(final)),
            "probe": {
                key: np.array([p[key] for p in probes])
                for key in (probes[0] if probes else {})
            },
        },
    )


def _joint_action(env, robot, arm, qpos_target, gripper_command):
    """A full action vector commanding absolute joint positions.

    The JOINT_POSITION controller takes a *delta* from the current position,
    scaled onto ``[output_min, output_max]``, so an absolute target has to be
    converted here rather than passed through.
    """
    controller = robot.composite_controller.part_controllers[arm]
    qpos_index = np.asarray(controller.qpos_index)
    current = np.asarray(env.sim.data.qpos[qpos_index], dtype=float)
    delta = np.asarray(qpos_target, dtype=float) - current
    scaled = np.clip(delta / JOINT_ACTION_SCALE, -1.0, 1.0)

    action = np.zeros(env.action_dim)
    action[: len(scaled)] = scaled
    action[len(scaled):] = gripper_command
    return action


def _site_position(env, robot, arm) -> np.ndarray:
    site_ids = robot.eef_site_id
    site_id = site_ids[arm] if isinstance(site_ids, dict) else site_ids
    return np.array(env.sim.data.site_xpos[site_id])


def _site_rotation(env, robot, arm) -> np.ndarray:
    site_ids = robot.eef_site_id
    site_id = site_ids[arm] if isinstance(site_ids, dict) else site_ids
    return np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3)
