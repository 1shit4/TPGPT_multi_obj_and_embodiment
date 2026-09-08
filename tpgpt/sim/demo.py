"""Recording a single demonstration (paper Sec. V-A).

The paper teaches one demonstration kinesthetically and transports it. Here a
scripted teacher plays the same role: it produces one demonstration carrying the
**complete** label set of Sec. III-A -- position, velocity, orientation,
stiffness, damping -- plus the gripper command and the belief of time that
Sec. V's policy takes as input.

Two things the prototype got wrong are fixed here by construction:

* **Timestamps come from the step index and the control frequency**, never from
  ``time.time()``. The prototype timed a post-hoc replay loop, measured
  ``dt ~ 1e-5 s``, and then zeroed any sample whose ``dt`` was below a
  threshold -- so almost every velocity label was exactly zero.
* **Frames are actually captured.** The prototype built its recording
  environment with offscreen rendering disabled and then read camera images
  from the observation dict, getting ``None`` every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpgpt.sim.backend import FrameWriter, observation_frame
from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController
from tpgpt.sim.scenes.reshelving import PRODUCT_HALF_SIZE
from tpgpt.transport.labels import PolicyLabels
from tpgpt.utils.rotations import quat_to_matrix

#: Height of the grip site above the product centre in a top-down grasp. The
#: Panda's finger pads sit slightly above the grip site and its fingers are
#: short, so the grasp has to be taken at the top face of the box: measured on
#: this scene, offsets below the box half-height bottom out on the shoulder of
#: the box or the table and the grasp fails outright.
GRASP_HEIGHT_OFFSET = PRODUCT_HALF_SIZE[2]


@dataclass
class Waypoint:
    """One segment of the scripted demonstration."""

    position: np.ndarray
    gripper: float
    #: Translational stiffness for this segment, in N/m.
    stiffness: float
    n_steps: int
    label: str = ""
    #: Attractor orientation for this segment. ``None`` keeps the previous one.
    rotation: np.ndarray | None = None


def _stiffness_matrices(k: float, damping_ratio: float = 0.9):
    """Critically-ish damped diagonal stiffness/damping pair."""
    K = np.eye(3) * k
    D = np.eye(3) * (2.0 * np.sqrt(k) * damping_ratio)
    return K, D


def pick_place_waypoints(
    grasp_position: np.ndarray,
    grasp_rotation: np.ndarray,
    place_position: np.ndarray,
    place_rotation: np.ndarray,
    grasp_approach: np.ndarray = (0.0, 0.0, -1.0),
    place_approach: np.ndarray | None = None,
    hover_height: float = 0.16,
    free_stiffness: float = 350.0,
    insertion_stiffness: float = 900.0,
) -> list[Waypoint]:
    """A pick-and-place plan driven by a grasp and a placement, not by a scene.

    Two things about this plan are substantive rather than incidental.

    **The gripper pose follows the grasp.** The plan takes an end-effector pose
    at each end rather than an object position and a hard-coded top-down offset,
    which is what lets a generated grasp on an arbitrary object drive it, and
    what lets the hand come in horizontally for a shelf that has a roof on it.

    **The stiffness profile is part of the demonstration**, not a controller
    setting: the teacher is compliant while moving through free space and stiff
    while grasping and inserting. Sec. III-G is what makes that profile
    transportable, and it is only observable because the controller accepts a
    full stiffness matrix.

    Args:
        grasp_approach: Direction the hand advances along at the pick. The
            standoff is taken back along this, so a horizontal approach backs
            off sideways rather than lifting straight up.
        place_approach: Same for the placement. Defaults to ``grasp_approach``.
    """
    grasp_position = np.asarray(grasp_position, dtype=float)
    place_position = np.asarray(place_position, dtype=float)
    grasp_approach = np.asarray(grasp_approach, dtype=float)
    grasp_approach = grasp_approach / np.linalg.norm(grasp_approach)
    place_approach = (
        grasp_approach if place_approach is None
        else np.asarray(place_approach, dtype=float)
        / np.linalg.norm(np.asarray(place_approach, dtype=float))
    )

    above_object = grasp_position - grasp_approach * hover_height
    above_goal = place_position - place_approach * hover_height

    return [
        Waypoint(above_object, -1.0, free_stiffness, 25, "approach", grasp_rotation),
        Waypoint(grasp_position, -1.0, free_stiffness, 25, "descend", grasp_rotation),
        Waypoint(grasp_position, 1.0, insertion_stiffness, 15, "grasp", grasp_rotation),
        Waypoint(above_object, 1.0, free_stiffness, 25, "lift", grasp_rotation),
        Waypoint(above_goal, 1.0, free_stiffness, 40, "transfer", place_rotation),
        Waypoint(place_position, 1.0, insertion_stiffness, 30, "insert", place_rotation),
        Waypoint(place_position, -1.0, insertion_stiffness, 15, "release", place_rotation),
        Waypoint(above_goal, -1.0, free_stiffness, 25, "retreat", place_rotation),
    ]


def reshelving_waypoints(
    product_position: np.ndarray,
    goal_position: np.ndarray,
    product_yaw: float = 0.0,
    hover_height: float = 0.16,
    free_stiffness: float = 350.0,
    insertion_stiffness: float = 900.0,
) -> list[Waypoint]:
    """Scripted plan for the reshelving task (paper Sec. V-A).

    A thin call into :func:`pick_place_waypoints` with a top-down grasp whose
    yaw follows the product. Object yaw is randomised over 94.6 deg (Table II)
    and the box diagonal barely clears the gripper opening, so a fixed-yaw
    top-down grasp collides with the box corners and simply fails. The
    demonstration is therefore genuinely ``SE(3)``-dependent, which is what
    makes transporting the orientation labels (Eq. 11) meaningful rather than
    decorative.
    """
    product = np.asarray(product_position, dtype=float)
    goal = np.asarray(goal_position, dtype=float)
    offset = np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    return pick_place_waypoints(
        product + offset,
        top_down_orientation(product_yaw),
        goal + offset,
        top_down_orientation(0.0),
        hover_height=hover_height,
        free_stiffness=free_stiffness,
        insertion_stiffness=insertion_stiffness,
    )


def record_demonstration(
    env,
    waypoints: list[Waypoint] | None = None,
    video_path=None,
    camera: str = "frontview",
) -> PolicyLabels:
    """Execute a scripted demonstration and return its policy labels.

    The attractor is interpolated linearly between waypoints, so the commanded
    motion is smooth and the resulting velocity labels are well defined.

    Args:
        env: A reshelving environment running the joint-torque controller.
        waypoints: Plan to execute. Defaults to
            :func:`reshelving_waypoints` for the current scene configuration.
        video_path: Optional mp4 to stream frames to. Requires the environment
            to have been built with ``offscreen=True``.
        camera: Camera to record from.

    Returns:
        A :class:`~tpgpt.transport.labels.PolicyLabels` with every family
        populated.

    Note:
        The position and orientation labels are the **commanded attractor**, not
        the measured end-effector pose. Sec. V is explicit that what the policy
        learns is "the desired attractor position, orientation, stiffness,
        damping" as a function of the current pose, and the distinction is not
        academic. An impedance-controlled arm lags its attractor by ``v D / K``
        -- measured here at 23 mm on average and 65 mm at peak speed. Labelling
        with the measured pose and then *executing* those labels as attractors
        applies that lag twice: the arm ended up 20 to 40 mm behind the
        transported path and the gripper closed on empty air every time.

        The measured trajectory is kept in ``metadata["measured_positions"]``
        for analysis, along with the lag statistics.
    """
    # The caller owns the reset. Resetting here would re-randomise the scene
    # after the caller had already read the object and goal poses from it.
    obs = env._get_observations()
    controller = CartesianImpedanceController(env)
    controller.reset()

    if waypoints is None:
        waypoints = reshelving_waypoints(
            obs["product_pos"], obs["goal_pos"], product_yaw=product_yaw(obs)
        )

    dt = 1.0 / env.control_freq
    start, orientation_desired, _, _ = controller.eef_state()

    positions, measured_positions, orientations = [], [], []
    stiffness, damping, grippers = [], [], []
    writer = FrameWriter(video_path, fps=env.control_freq) if video_path else None

    previous = start
    for waypoint in waypoints:
        K, D = _stiffness_matrices(waypoint.stiffness)
        if waypoint.rotation is not None:
            orientation_desired = waypoint.rotation
        for step in range(waypoint.n_steps):
            alpha = (step + 1) / waypoint.n_steps
            attractor = previous + alpha * (waypoint.position - previous)
            action = controller.action(
                attractor,
                K,
                D,
                gripper=waypoint.gripper,
                rotation_desired=orientation_desired,
            )
            obs, _, _, _ = env.step(action)

            position, _, _, _ = controller.eef_state()
            positions.append(attractor.copy())
            measured_positions.append(position.copy())
            orientations.append(orientation_desired.copy())
            stiffness.append(K.copy())
            damping.append(D.copy())
            grippers.append(waypoint.gripper)
            if writer is not None:
                writer.append(observation_frame(obs, camera))
        previous = waypoint.position

    if writer is not None:
        writer.close()

    positions = np.stack(positions)
    measured = np.stack(measured_positions)
    n = len(positions)
    # Velocities from the attractor path at the true control period.
    velocities = np.gradient(positions, dt, axis=0)
    phase = np.linspace(0.0, 1.0, n)

    return PolicyLabels(
        positions=positions,
        velocities=velocities,
        orientations=np.stack(orientations),
        stiffness=np.stack(stiffness),
        damping=np.stack(damping),
        gripper=np.array(grippers),
        time_belief=phase,
        time_rate=np.full(n, 1.0 / (dt * max(n - 1, 1))),
        metadata={
            "control_freq": int(env.control_freq),
            "dt": dt,
            "n_labels": n,
            "duration_s": dt * (n - 1),
            "scene": "reshelving",
            "segments": [w.label for w in waypoints],
            "product_position": np.asarray(obs["product_pos"]).tolist(),
            "goal_position": np.asarray(obs["goal_pos"]).tolist(),
            "video": str(writer.path) if writer and writer.n_frames else None,
            "measured_positions": measured.tolist(),
            "attractor_lag_mean": float(np.linalg.norm(positions - measured, axis=1).mean()),
            "attractor_lag_max": float(np.linalg.norm(positions - measured, axis=1).max()),
        },
    )


def top_down_orientation(yaw: float = 0.0) -> np.ndarray:
    """Gripper pointing straight down, rotated by ``yaw`` about the world z."""
    base = quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))[0]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ base


def product_yaw(obs: dict) -> float:
    """Yaw of the product about the world z axis, from its observed pose."""
    R = quat_to_matrix(np.asarray(obs["product_quat"]))[0]
    return float(np.arctan2(R[1, 0], R[0, 0]))
