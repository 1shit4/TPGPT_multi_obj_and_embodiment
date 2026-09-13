"""Execute a grasp in physics: the arbiter for the frame contract.

Geometry can only get the frame contract close. Three definitions of "where the
contact point is" were tried against the nine registered hands -- the config's
fingertip depth, robosuite's base-to-``grip_site`` offset, and the centroid of
the closed fingers -- and each one worked for a different six and failed the
other three. Long fingers, a flipped frame and an anthropomorphic thumb do not
share a single geometric rule.

So the contact offset is **calibrated**: sweep the depth, see which values
actually pick the object up, and keep the middle of the widest working band.
The width of that band is itself worth having -- it says how forgiving a hand
is, which is the difference between a gripper that tolerates a 2 cm perception
error and one that does not.
"""

from __future__ import annotations

import numpy as np

from tpgpt.grasp.grasps import Grasp6D, approach_waypoint, grasp_to_eef_pose
from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

#: Object used for calibration. A can is a good reference: round, so the grasp
#: yaw does not matter, and narrow enough for every jaw in the registry.
REFERENCE_OBJECT = "can"

#: A lift of at least this counts as a successful grasp, in metres.
LIFT_THRESHOLD = 0.05


def synthetic_top_down(contact: np.ndarray, yaw: float, graspgen_name: str) -> Grasp6D:
    """A grasp approaching straight down, closing along ``yaw``.

    Built in GraspGen-X's own convention: ``+Z`` approach, ``+X`` closing,
    anchored at the gripper base, which sits ``tcp_depth`` back along the
    approach axis from the contact point.
    """
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
    pose[:3, 3] = contact - approach * gripper_geometry(graspgen_name).tcp_depth
    return Grasp6D(
        pose=pose,
        score=1.0,
        gripper=graspgen_name,
        width=gripper_geometry(graspgen_name).aperture,
    )


def make_env(gripper_short: str, objects=("can", "milk"), seed: int = 2,
             offscreen: bool = False, shelf_variant: str = "cubby"):
    """A tabletop scene with the given hand mounted on a Panda."""
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.controllers.cartesian_impedance import make_torque_controller_config
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    pair = resolve_pair(gripper_short)
    config = make_torque_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    return TabletopShelf(
        robots="Panda",
        gripper_types=pair.robosuite,
        controller_configs=config,
        objects=objects,
        control_freq=20,
        seed=seed,
        shelf_variant=shelf_variant,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=False,
    )


def execute_grasp(
    env,
    controller,
    grasp: Grasp6D,
    gripper: str,
    stiffness: float = 600.0,
    extra_depth: float = 0.0,
    lift: float = 0.15,
) -> None:
    """Approach, close, lift.

    Args:
        extra_depth: Pushed along the approach axis on top of the converted
            pose. Used by the calibration sweep; zero in normal use.
    """
    position, rotation = grasp_to_eef_pose(grasp, gripper)
    position = position + grasp.approach * extra_depth
    pre_grasp = approach_waypoint(grasp, standoff=0.12, gripper=gripper)
    pre_grasp = pre_grasp + grasp.approach * extra_depth
    K = np.eye(3) * stiffness
    D = np.eye(3) * (2 * np.sqrt(stiffness) * 0.9)

    def drive(target, steps, command, settle=0):
        start = controller.eef_state()[0]
        for step in range(steps):
            alpha = (step + 1) / steps
            env.step(
                controller.action(
                    start + alpha * (target - start), K, D, gripper=command,
                    rotation_desired=rotation, rotational_stiffness=60.0,
                    rotational_damping=12.0,
                )
            )
        for _ in range(settle):
            env.step(
                controller.action(
                    target, K, D, gripper=command, rotation_desired=rotation,
                    rotational_stiffness=60.0, rotational_damping=12.0,
                )
            )

    drive(pre_grasp, 45, -1.0, settle=10)
    drive(position, 40, -1.0, settle=25)
    for _ in range(25):
        env.step(
            controller.action(
                position, K, D, gripper=1.0, rotation_desired=rotation,
                rotational_stiffness=60.0, rotational_damping=12.0,
            )
        )
    drive(position + np.array([0.0, 0.0, lift]), 45, 1.0, settle=10)


def lift_height(
    gripper_short: str,
    obj: str = REFERENCE_OBJECT,
    extra_depth: float = 0.0,
    yaw: float = 0.0,
    seed: int = 2,
) -> float:
    """Height the object gained, in metres. The frame contract's pass/fail.

    ``obj`` is put into the scene rather than assumed to be there, so a hand
    can be calibrated against an object it can actually hold. A can is 65 mm
    across and the default here, which is a fine reference for a jaw that opens
    to 80 or 125 mm and a meaningless one for a hand whose fingertips part by
    30: for that hand the sweep measures "a can does not fit", not "this hand
    cannot grasp". Two of the four hands that fail the sweep entirely are
    multi-finger hands in exactly that position.
    """
    from tpgpt.sim.controllers.cartesian_impedance import CartesianImpedanceController

    # The default scene is left exactly as it was -- ``make_env``'s own
    # ("can", "milk") -- because every calibrated depth in the registry was
    # measured in it, and the placement sampler lays objects out in the order
    # it is given, so adding or reordering one moves the others. A different
    # reference object is appended rather than substituted.
    objects = ("can", "milk")
    if obj not in objects:
        objects = objects + (obj,)
    env = make_env(gripper_short, objects=objects, seed=seed)
    try:
        env.reset()
        controller = CartesianImpedanceController(env)
        controller.reset()
        start = env.object_position(obj)[2]
        contact = env.object_position(obj) + np.array([0.0, 0.0, 0.02])
        grasp = synthetic_top_down(contact, yaw, resolve_pair(gripper_short).graspgen)
        execute_grasp(env, controller, grasp, gripper_short, extra_depth=extra_depth)
        return float(env.object_position(obj)[2] - start)
    finally:
        env.close()


def calibrate_depth(
    gripper_short: str,
    span: float = 0.09,
    steps: int = 13,
    obj: str = REFERENCE_OBJECT,
) -> dict:
    """Find the depth offset at which this hand actually grasps.

    Sweeps an extra push along the approach axis and records which values lift
    the reference object. Returns the **middle of the widest contiguous working
    band** rather than the first value that works, so the stored offset is as
    far as possible from either edge of failure.

    Returns:
        ``{"offset", "band_width", "n_working", "samples"}``. ``band_width`` is
        the size of the working window in metres -- how much depth error this
        hand tolerates before it either misses the object or drives into the
        table.
    """
    deltas = np.linspace(-span, span, steps)
    lifts = [lift_height(gripper_short, obj, extra_depth=float(d)) for d in deltas]
    working = [lift > LIFT_THRESHOLD for lift in lifts]

    best_start, best_len, start = -1, 0, None
    for i, ok in enumerate([*working, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None

    if best_len == 0:
        return {
            "offset": None,
            "band_width": 0.0,
            "n_working": 0,
            "samples": list(zip(deltas.tolist(), lifts)),
        }
    band = deltas[best_start : best_start + best_len]
    step = float(deltas[1] - deltas[0])
    return {
        "offset": float(band.mean()),
        "band_width": float(best_len * step),
        "n_working": int(sum(working)),
        "samples": list(zip(deltas.tolist(), lifts)),
    }
