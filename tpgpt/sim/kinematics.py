"""Inverse kinematics, for asking whether the arm can reach a pose at all.

A grasp can be perfectly formed, collision-free and still useless because the
arm cannot get to it -- or, more often, can reach the object but not the shelf
slot it has to be carried to. That second case is the one that matters here and
it is invisible to every geometric filter.

The sibling project tried a cheaper proxy and deleted it. Solving IK at 285
points on its table showed that only about 43% of the surface admitted a
solution, that unreachable points began as close as 0.31 m from the base while
reachable ones extended to 0.823 m, and that constraining placement to a disc
"excluded reachable cells, admitted unreachable ones, and turned 'the hand
stopped 6 cm short' into 'no placement at all'". So this is real IK.

Damped least squares on MuJoCo's own site Jacobian: cheap enough to run on a
few hundred candidate poses, and it uses the same model the physics does, so a
solution here is a configuration the simulator will actually accept.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    """Outcome of one IK solve."""

    reachable: bool
    qpos: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int
    #: True if the solution needed a joint pushed to its limit, which usually
    #: means the pose is on the very edge of the workspace.
    at_joint_limit: bool = False

    def describe(self) -> str:
        state = "reachable" if self.reachable else "unreachable"
        return (
            f"{state}: {self.position_error * 1000:.1f} mm, "
            f"{np.degrees(self.rotation_error):.1f} deg after {self.iterations} steps"
        )


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Axis-angle of the rotation taking ``current`` to ``target``."""
    relative = target @ current.T
    vector = np.empty(3)
    mujoco.mju_mat2Quat(quat := np.empty(4), relative.flatten())
    mujoco.mju_quat2Vel(vector, quat, 1.0)
    return vector


def solve_ik(
    env,
    position: np.ndarray,
    rotation: np.ndarray | None = None,
    arm: str = "right",
    max_iterations: int = 120,
    position_tolerance: float = 5e-3,
    rotation_tolerance: float = 0.15,
    damping: float = 0.08,
    step_scale: float = 0.6,
    seed_qpos: np.ndarray | None = None,
) -> IKResult:
    """Can the arm put its end-effector site at this pose?

    Iterates damped least squares on the arm joints only, clamping each step to
    the model's joint limits. The live simulation state is saved and restored,
    so this is a query and changes nothing.

    Args:
        position: ``(3,)`` world target for the gripper site.
        rotation: ``(3, 3)`` world target orientation, or ``None`` to solve for
            position alone. Orientation matters for a grasp -- a reachable
            position with the hand upside down is not a reachable grasp.
        position_tolerance: Metres. 5 mm, which is below the impedance
            controller's own tracking error, so a pose that passes here is not
            one the controller will miss for kinematic reasons.
        rotation_tolerance: Radians, about 8.6 degrees.
        seed_qpos: Starting configuration. Defaults to the current one, which
            makes a sequence of nearby poses much cheaper to test.

    Returns:
        An :class:`IKResult`. ``reachable`` is False when the solver ran out of
        iterations without meeting both tolerances.
    """
    sim = env.sim
    robot = env.robots[0]
    controller = robot.composite_controller.part_controllers[arm]
    qpos_index = np.asarray(controller.qpos_index)
    qvel_index = np.asarray(getattr(controller, "qvel_index", controller.joint_index))

    site_ids = robot.eef_site_id
    site_id = site_ids[arm] if isinstance(site_ids, dict) else site_ids

    saved_qpos = np.array(sim.data.qpos)
    saved_qvel = np.array(sim.data.qvel)
    if seed_qpos is not None:
        sim.data.qpos[qpos_index] = np.asarray(seed_qpos, dtype=float)

    limits = sim.model.jnt_range[sim.model.jnt_qposadr.searchsorted(qpos_index)]
    limited = sim.model.jnt_limited[sim.model.jnt_qposadr.searchsorted(qpos_index)].astype(bool)

    position = np.asarray(position, dtype=float).reshape(3)
    n = len(qvel_index)
    jac_position = np.zeros((3, sim.model.nv))
    jac_rotation = np.zeros((3, sim.model.nv))

    at_limit = False
    try:
        for iteration in range(1, max_iterations + 1):
            mujoco.mj_forward(sim.model._model, sim.data._data)
            current = np.array(sim.data.site_xpos[site_id])
            error = position - current
            residual = [error]

            mujoco.mj_jacSite(
                sim.model._model, sim.data._data, jac_position, jac_rotation, site_id
            )
            jacobian = [jac_position[:, qvel_index]]

            rotation_error = 0.0
            if rotation is not None:
                current_rotation = np.array(sim.data.site_xmat[site_id]).reshape(3, 3)
                angular = _rotation_error(current_rotation, np.asarray(rotation))
                rotation_error = float(np.linalg.norm(angular))
                residual.append(angular)
                jacobian.append(jac_rotation[:, qvel_index])

            position_error = float(np.linalg.norm(error))
            if position_error < position_tolerance and rotation_error < rotation_tolerance:
                return IKResult(
                    True, np.array(sim.data.qpos[qpos_index]),
                    position_error, rotation_error, iteration, at_limit,
                )

            J = np.vstack(jacobian)
            e = np.concatenate(residual)
            # Damped least squares: stays finite through singularities, which
            # a straight pseudo-inverse does not, and grasp poses sit near them.
            delta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(e)), e)

            q = np.array(sim.data.qpos[qpos_index]) + step_scale * delta[:n]
            clamped = np.where(limited, np.clip(q, limits[:, 0], limits[:, 1]), q)
            at_limit = at_limit or bool(np.any(np.abs(clamped - q) > 1e-9))
            sim.data.qpos[qpos_index] = clamped

        return IKResult(
            False, np.array(sim.data.qpos[qpos_index]),
            position_error, rotation_error, max_iterations, at_limit,
        )
    finally:
        sim.data.qpos[:] = saved_qpos
        sim.data.qvel[:] = saved_qvel
        mujoco.mj_forward(sim.model._model, sim.data._data)


def reachable(env, poses, arm: str = "right", **kwargs) -> list[IKResult]:
    """Solve IK for a sequence of poses, warm-starting each from the last.

    A pick-and-place needs four poses to work -- pre-grasp, grasp, place and
    retreat -- and they are close together, so seeding each solve from the
    previous solution is both faster and more honest: it asks whether the arm
    can move *between* them rather than whether each is reachable from the rest
    pose by some unrelated configuration.
    """
    results, seed = [], None
    for position, rotation in poses:
        result = solve_ik(env, position, rotation, arm=arm, seed_qpos=seed, **kwargs)
        results.append(result)
        if result.reachable:
            seed = result.qpos
    return results
