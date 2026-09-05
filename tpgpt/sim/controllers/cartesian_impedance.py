"""Cartesian impedance control with full stiffness and damping matrices.

Sec. V controls the robot with a Cartesian impedance controller, and Sec. III-G
transports the stiffness by congruence, ``K_hat = J_perp K J_perp^T``. Because
``J_perp`` is a rotation, ``K_hat`` is a **full symmetric matrix** even when the
demonstrated ``K`` was diagonal: transport rotates the stiffness ellipsoid.

robosuite's own variable-impedance OSC cannot express that. It accepts a
six-vector of gains, i.e. a diagonal stiffness in the fixed task axes, which
silently discards exactly the off-diagonal structure the transport produces.
This controller therefore drives the arm through the ``JOINT_TORQUE`` part
controller and applies the impedance law directly:

    ``tau = J_p^T [K_p (x_d - x) + D_p (v_d - v)]``
         ``+ J_r^T [K_o e_o + D_o (w_d - w)] + tau_nullspace``

Gravity and Coriolis compensation is left to robosuite, whose joint-torque
controller adds ``torque_compensation`` when ``use_torque_compensation`` is set.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices

from tpgpt.utils.rotations import quat_to_matrix

#: Action-to-torque scale, in N m. robosuite maps an action of +-1 onto
#: +-output_max, so this must cover the largest joint torque any supported robot
#: can command. It is not a safety limit: the joint-torque controller separately
#: clips to the actuator ranges declared by the model.
TORQUE_ACTION_SCALE = 150.0


def make_torque_controller_config(
    base_config: dict, arm: str = "right", torque_scale: float = TORQUE_ACTION_SCALE
) -> dict:
    """Rewrite a composite controller config to drive ``arm`` by joint torque.

    ``output_max`` matters more than it looks. robosuite rescales every action
    from ``[input_min, input_max]`` onto ``[output_min, output_max]``, so leaving
    the default of 1 silently divides every commanded torque by the ratio -- the
    arm then barely moves and the impedance law appears to be wrong when it is
    only being attenuated. :class:`CartesianImpedanceController` reads this same
    value back so the two can never disagree.

    Args:
        base_config: A composite config, e.g. from
            ``load_composite_controller_config(controller="BASIC", ...)``.
        arm: Body-part key to convert.
        torque_scale: Torque corresponding to an action of 1.
    """
    config = copy.deepcopy(base_config)
    gripper = config["body_parts"].get(arm, {}).get("gripper", {"type": "GRIP"})
    config["body_parts"][arm] = {
        "type": "JOINT_TORQUE",
        "input_max": 1,
        "input_min": -1,
        "output_max": float(torque_scale),
        "output_min": -float(torque_scale),
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper,
    }
    return config


class CartesianImpedanceController:
    """Full-matrix Cartesian impedance control over robosuite joint torques.

    Args:
        env: A robosuite environment whose arm runs the ``JOINT_TORQUE``
            controller (see :func:`make_torque_controller_config`).
        arm: Body-part key.
        nullspace_stiffness: Gain pulling the arm towards its initial posture in
            the task nullspace, which keeps the redundant joints from drifting.
        nullspace_damping: Damping of the same term.
        max_wrench: Clamp on the commanded Cartesian force and torque, in N and
            N m. A safety net for attractors far from the current pose.
    """

    def __init__(
        self,
        env,
        arm: str = "right",
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 4.0,
        max_wrench: tuple[float, float] = (80.0, 12.0),
    ):
        self.env = env
        self.arm = arm
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)
        self.max_force, self.max_torque = max_wrench

        self.robot = env.robots[0]
        self.part_controller = self.robot.composite_controller.part_controllers[arm]
        self.joint_index = np.asarray(self.part_controller.joint_index)
        self.qpos_index = np.asarray(self.part_controller.qpos_index)
        self.qvel_index = np.asarray(
            getattr(self.part_controller, "qvel_index", self.part_controller.joint_index)
        )

        site_ids = self.robot.eef_site_id
        site_id = site_ids[arm] if isinstance(site_ids, dict) else site_ids
        self.eef_site = env.sim.model.site_id2name(site_id)

        # Normalise by the controller's own output range, so the action we send
        # is rescaled back to exactly the torque we computed. The controller
        # then clips to the model's actuator limits on our behalf.
        self.torque_scale = np.abs(
            np.asarray(self.part_controller.output_max, dtype=float)
        )
        self.initial_qpos = np.array(env.sim.data.qpos[self.qpos_index])

    # ----------------------------------------------------------- kinematics
    def reset(self) -> None:
        """Re-latch the nullspace posture target to the current configuration."""
        self.initial_qpos = np.array(self.env.sim.data.qpos[self.qpos_index])

    def eef_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current ``(position, rotation, linear velocity, angular velocity)``."""
        sim = self.env.sim
        position = np.array(sim.data.site_xpos[sim.model.site_name2id(self.eef_site)])
        rotation = np.array(
            sim.data.site_xmat[sim.model.site_name2id(self.eef_site)]
        ).reshape(3, 3)
        J_pos, J_ori = self.jacobians()
        qvel = np.array(sim.data.qvel[self.qvel_index])
        return position, rotation, J_pos @ qvel, J_ori @ qvel

    def jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """Positional and rotational end-effector Jacobians for this arm."""
        sim = self.env.sim
        J_pos = sim.data.get_site_jacp(self.eef_site).reshape(3, -1)[:, self.qvel_index]
        J_ori = sim.data.get_site_jacr(self.eef_site).reshape(3, -1)[:, self.qvel_index]
        return J_pos, J_ori

    # ------------------------------------------------------------- control
    @staticmethod
    def orientation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
        """Axis-angle error of ``R_desired R_current^T``, in world axes."""
        rel = np.asarray(R_desired, dtype=float) @ np.asarray(R_current, dtype=float).T
        angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 1e-9:
            return np.zeros(3)
        axis = np.array(
            [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
        ) / (2.0 * np.sin(angle))
        return axis * angle

    def compute_torque(
        self,
        position_desired: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        rotation_desired: np.ndarray | None = None,
        rotational_stiffness: float | np.ndarray = 40.0,
        rotational_damping: float | np.ndarray = 8.0,
        velocity_desired: np.ndarray | None = None,
    ) -> np.ndarray:
        """Joint torques realising the commanded Cartesian impedance.

        Args:
            position_desired: ``(3,)`` attractor position.
            stiffness: ``(3, 3)`` translational stiffness. May be a full matrix,
                which is the point of this controller.
            damping: ``(3, 3)`` translational damping.
            rotation_desired: ``(3, 3)`` attractor orientation, or ``None`` to
                leave orientation uncontrolled.
            rotational_stiffness: Scalar or ``(3, 3)`` rotational stiffness.
            rotational_damping: Scalar or ``(3, 3)`` rotational damping.
            velocity_desired: ``(3,)`` feed-forward Cartesian velocity.

        Returns:
            ``(n_joints,)`` torques, before robosuite's gravity compensation.
        """
        position, rotation, velocity, omega = self.eef_state()
        J_pos, J_ori = self.jacobians()

        K = np.atleast_2d(np.asarray(stiffness, dtype=float))
        D = np.atleast_2d(np.asarray(damping, dtype=float))
        v_des = np.zeros(3) if velocity_desired is None else np.asarray(velocity_desired)

        force = K @ (np.asarray(position_desired, dtype=float) - position) + D @ (
            v_des - velocity
        )
        force = np.clip(force, -self.max_force, self.max_force)
        torque = J_pos.T @ force

        if rotation_desired is not None:
            K_o = _as_matrix(rotational_stiffness)
            D_o = _as_matrix(rotational_damping)
            moment = K_o @ self.orientation_error(rotation_desired, rotation) - D_o @ omega
            moment = np.clip(moment, -self.max_torque, self.max_torque)
            torque = torque + J_ori.T @ moment

        return torque + self._nullspace_torque(np.vstack([J_pos, J_ori]))

    def _nullspace_torque(self, J_full: np.ndarray) -> np.ndarray:
        """Posture term projected into the task nullspace.

        Reuses robosuite's dynamically-consistent projector so the posture
        control cannot fight the Cartesian task.
        """
        sim = self.env.sim
        mass_matrix = self._mass_matrix()
        _, _, _, nullspace_matrix = opspace_matrices(
            mass_matrix, J_full, J_full[:3], J_full[3:]
        )
        return nullspace_torques(
            mass_matrix,
            nullspace_matrix,
            self.initial_qpos,
            np.array(sim.data.qpos[self.qpos_index]),
            np.array(sim.data.qvel[self.qvel_index]),
            joint_kp=self.nullspace_stiffness,
        )

    def _mass_matrix(self) -> np.ndarray:
        """Arm-block of the joint-space inertia matrix.

        ``mj_fullM`` expands MuJoCo's sparse ``qM``; this mirrors how robosuite's
        own part controllers obtain it.
        """
        sim = self.env.sim
        n = sim.model.nv
        full = np.ndarray(shape=(n, n), dtype=np.float64, order="C")
        mujoco.mj_fullM(sim.model._model, full, sim.data.qM)
        return full.reshape(n, n)[np.ix_(self.qvel_index, self.qvel_index)]

    def action(
        self,
        position_desired: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        gripper: float = -1.0,
        **kwargs,
    ) -> np.ndarray:
        """Full environment action: normalised arm torques plus the gripper."""
        torque = self.compute_torque(position_desired, stiffness, damping, **kwargs)
        normalised = np.clip(torque / self.torque_scale, -1.0, 1.0)
        return np.concatenate([normalised, [float(gripper)]])


def _as_matrix(value: float | np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value * np.eye(3) if value.ndim == 0 else np.atleast_2d(value)
