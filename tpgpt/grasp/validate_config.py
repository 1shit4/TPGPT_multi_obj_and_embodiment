"""Does a gripper description actually produce grasps that work?

**The test a config has to pass.** ``tpgpt.grasp.describe`` writes a GraspGen-X
``config.json`` from a robosuite model, and the planner accepts it, the funnel
filters it and the arm flies to its grasps -- and the hand touches nothing.
Measured over four authored hands: **0 of 41 grasps held**, while the two hands
running a description GraspGen-X's own authors wrote held 15 of 30 and 7 of 15.
So "the server accepted it" is not evidence of anything, and neither is
agreement with the shipped numbers: those are round to the nearest millimetre
(80, 18, 18; 66, 10, 40) because a person dragged a box in a GUI to make them.

What decides a config is whether the grasps it yields can be executed. This
module measures that, and it is built so the answer can be attributed:

**The control is a hand that has both.** Eleven registered hands have a
description GraspGen-X wrote *and* a robosuite model. Author a second
description for one of those, put both names to the planner on the same object,
execute both, and the difference is the description and nothing else -- same
hand, same arm, same object, same physics.

**The two ways a config can be wrong are separated.** A description enters the
result twice, and they are independent:

* ``sweep_volume``, ``type`` and ``symmetric`` are what the model conditions on,
  so they decide **which poses come back**;
* ``fingertip`` is the tool-centre depth, so it decides **where the hand is
  driven** for a given pose -- it is read by ``grasp_to_eef_pose`` long after
  the planner has finished.

Passing ``tcp_depth`` explicitly here lets one be held fixed while the other
varies. Scoring a config without that separation cannot tell "it proposed bad
grasps" from "it proposed good grasps and we drove to the wrong place".

**The scene is deliberately not this project's.** A plain robosuite ``Lift``
table with one cube, and an object cloud sampled from the cube's known geometry
rather than from cameras. Perception is not under test, the shelf is not under
test, and nothing here depends on the scene modules another workstream is
changing.
"""

from __future__ import annotations

import numpy as np

from tpgpt.grasp.client import GraspGenClient
from tpgpt.grasp.grasps import Grasp6D, alignment_rotation, contact_offset
from tpgpt.grasp.grippers import resolve_pair

#: Points sampled on the object's surface for the planner. GraspGen-X caps the
#: cloud it will accept at 8192; a few thousand is plenty for a primitive.
DEFAULT_CLOUD_POINTS = 2048

#: Control steps per phase of the motion.
STEPS = {"pre": 45, "descend": 40, "close": 30, "lift": 45, "settle": 12}

#: How far the object is lifted, in metres.
LIFT_TO = 0.15

#: A lift of at least this counts as held, in metres. Matches
#: ``verify.LIFT_THRESHOLD``.
LIFT_THRESHOLD = 0.05

#: Position tolerance for the reachability screen, in metres.
IK_TOLERANCE = 5e-3


def surface_cloud(half_extents, centre, n=DEFAULT_CLOUD_POINTS, seed=0):
    """Points on the surface of a box, in world coordinates.

    Sampled from the object's **known geometry** rather than rendered, so a
    config is not being scored through the perception stack. Faces are chosen in
    proportion to their area so the sampling is uniform over the surface.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(half_extents, dtype=float)
    points = rng.uniform(-1.0, 1.0, size=(n, 3))
    areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    face = rng.choice(3, size=n, p=areas / areas.sum())
    points[np.arange(n), face] = np.sign(points[np.arange(n), face])
    return (points * half + np.asarray(centre, dtype=float)).astype(np.float32)


def propose(cloud, graspgen_name, num_grasps=200, topk=100, client=None):
    """Grasps for one description, at the gripper base in GraspGen-X's frame.

    Not cached: the point of this module is to compare two *descriptions*, and
    ``tpgpt.grasp.cache`` keys on the gripper name, so each name gets its own
    entry anyway. The planner is an unseeded diffusion model, so a comparison
    between two names is a comparison of two draws unless enough grasps are
    executed to swamp that -- which is why the caller executes several.
    """
    owned = client is None
    client = client or GraspGenClient()
    try:
        return client.infer(cloud, gripper_name=graspgen_name,
                            num_grasps=num_grasps, topk_num_grasps=topk)
    finally:
        if owned:
            client.close()


def eef_pose(pose, gripper, tcp_depth):
    """Where ``grip_site`` must be for this grasp, at a given tool depth.

    Mirrors :func:`tpgpt.grasp.grasps.grasp_to_eef_pose`, except that the tool
    depth is an **argument** rather than read from the gripper's config. That is
    the whole point: it lets the pose a description proposes be executed at
    someone else's tool depth, which separates "this description proposes bad
    poses" from "this description's ``fingertip`` is wrong".
    """
    pair = resolve_pair(gripper)
    rotation = np.asarray(pose)[:3, :3] @ alignment_rotation(pair)
    approach = np.asarray(pose)[:3, 2]
    contact = np.asarray(pose)[:3, 3] + approach * float(tcp_depth)
    return contact - rotation @ contact_offset(pair), rotation


def reset(env, seed=0, settle=40):
    """Put the scene back exactly as it was, and let the cube come to rest.

    ``Lift`` samples the cube's placement at every reset, so resetting between
    grasps silently moves the object the grasps were planned for. Seeding before
    the reset -- never after -- puts it back in the same place, and the settle
    afterwards means the first commanded motion is not racing a falling cube.
    """
    np.random.seed(seed)
    env.reset()
    robot = env.robots[0]
    arm = "right"
    qi = np.asarray(robot.composite_controller.part_controllers[arm].qpos_index)
    hold = np.array(env.sim.data.qpos[qi])
    from tpgpt.sim.replay import _joint_action
    for _ in range(settle):
        env.step(_joint_action(env, robot, arm, hold, -1.0))
    return env


def site_from_base(env, pose, gripper_short, arm="right", extra_depth=0.0):
    """Where ``grip_site`` goes if the gripper **base** is put exactly at ``pose``.

    This is the hardware-faithful placement, and it is deliberately different
    from :func:`eef_pose`. That one drives the hand using ``contact_offset`` --
    a per-hand correction this project measured in physics, containing a
    ``calibrated_depth`` fitted against the description GraspGen-X shipped. A
    config scored through it is being scored together with a correction that
    hides its errors, which is why an authored ``fingertip`` 10 mm out still
    looked workable once the correction was re-fitted around it.

    A real robot has no such correction. The planner emits a pose for the
    gripper base; you put the base there and close. So the transform used here
    is only the **constant, measurable** offset from the hand's own base frame
    to the site the simulator commands -- no calibration, nothing fitted.

    Returns ``(position, rotation)`` for ``grip_site``.
    """
    from tpgpt.grasp.grasps import alignment_rotation

    sim = env.sim
    gripper = env.robots[0].gripper
    gripper = gripper[arm] if isinstance(gripper, dict) else gripper
    root = sim.model.body_name2id(gripper.root_body)
    site = sim.model.site_name2id(gripper.important_sites["grip_site"])

    root_p = np.array(sim.data.body_xpos[root])
    site_p = np.array(sim.data.site_xpos[site])
    site_R = np.array(sim.data.site_xmat[site]).reshape(3, 3)

    # The base-to-site vector is fixed in the hand, so it only has to be
    # expressed in the frame the planner writes its poses in -- the hand's own
    # grasp convention. ``alignment`` relates that frame to ``grip_site``'s as
    # ``R_site = R_grasp @ alignment``, so the grasp frame's world rotation at
    # measurement time is ``R_site @ alignment.T``. Expressing the vector in the
    # *root body's* frame instead is the obvious mistake and it is silent: the
    # two frames differ by up to 180 degrees across this registry, and every
    # grasp then lands somewhere plausible and wrong. Measured that way,
    # GraspGen-X's own Panda description held 0 of 31.
    align = alignment_rotation(gripper_short)
    grasp_R = site_R @ align.T
    local = grasp_R.T @ (site_p - root_p)

    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] @ align
    # ``extra_depth`` pushes the hand further along its own approach axis. It
    # exists to *measure* the depth a description needs: sweeping it is
    # equivalent, on the execution side, to having authored a ``fingertip``
    # that much larger, and it costs no re-inference.
    local = local + np.array([0.0, 0.0, float(extra_depth)])
    position = pose[:3, 3] + pose[:3, :3] @ local
    return position, rotation


def _lift_scene(gripper_short, cube_half=0.020, seed=0):
    """A plain ``Lift`` table with one cube, under joint-position control.

    Position control rather than Cartesian impedance, for the reason measured in
    ``docs/gripper_diversity.md``: under impedance the hand finishes 43 to
    100 mm from poses inverse kinematics reports as reachable, which would
    swamp the difference between two descriptions.
    """
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.replay import make_position_controller_config

    pair = resolve_pair(gripper_short)
    config = make_position_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    np.random.seed(seed)
    env = suite.make(
        "Lift", robots="Panda", gripper_types=pair.robosuite,
        controller_configs=config, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, control_freq=20,
    )
    reset(env, seed=seed)
    return env


def _cube_state(env):
    """Position and half-extents of the ``Lift`` cube, read from the model."""
    sim = env.sim
    body = sim.model.body_name2id("cube_main")
    gid = [g for g in range(sim.model.ngeom)
           if (sim.model.geom_id2name(g) or "").startswith("cube")]
    half = np.array(sim.model.geom_size[gid[0]][:3])
    return np.array(sim.data.body_xpos[body]), half


def execute(env, pose, gripper_short, tcp_depth, arm="right"):
    """Drive to one grasp, close, lift, and report what happened.

    Returns ``reach_mm`` (how far the gripper site finished from the commanded
    pose before the jaws moved), ``lift_mm`` and ``held``.
    """
    from tpgpt.sim.kinematics import solve_ik
    from tpgpt.sim.replay import _joint_action

    robot = env.robots[0]
    site = (robot.eef_site_id[arm] if isinstance(robot.eef_site_id, dict)
            else robot.eef_site_id)
    qi = np.asarray(robot.composite_controller.part_controllers[arm].qpos_index)
    seed = np.array(env.sim.data.qpos[qi])

    position, rotation = eef_pose(pose, gripper_short, tcp_depth)
    approach = np.asarray(pose)[:3, 2]
    pre = position - approach * 0.10

    def drive(target, steps, command):
        nonlocal seed
        start = np.array(env.sim.data.site_xpos[site])
        for step in range(steps):
            alpha = (step + 1) / steps
            seed = solve_ik(env, start + alpha * (target - start), rotation, arm,
                            position_tolerance=IK_TOLERANCE, seed_qpos=seed).qpos
            env.step(_joint_action(env, robot, arm, seed, command))

    def hold(steps, command):
        for _ in range(steps):
            env.step(_joint_action(env, robot, arm, seed, command))

    start_z = float(_cube_state(env)[0][2])
    drive(pre, STEPS["pre"], -1.0)
    hold(STEPS["settle"], -1.0)
    drive(position, STEPS["descend"], -1.0)
    hold(STEPS["settle"], -1.0)
    reach = float(np.linalg.norm(position - np.array(env.sim.data.site_xpos[site])))
    hold(STEPS["close"], 1.0)
    drive(position + np.array([0.0, 0.0, LIFT_TO]), STEPS["lift"], 1.0)
    hold(STEPS["settle"], 1.0)
    lift = float(_cube_state(env)[0][2]) - start_z
    return {"reach_mm": reach * 1000.0, "lift_mm": lift * 1000.0,
            "held": lift > LIFT_THRESHOLD}


def teleport_grasp(env, pose, gripper_short, tcp_depth, arm="right",
                   close_steps=40, lift_steps=60, seed=0, exact_base=False,
                   extra_depth=0.0):
    """Place the hand **at** the grasp, close, lift. No approach, no path.

    What a gripper description determines is the *grasp*: where the hand sits
    relative to the object and how wide it is there. Flying to it is the arm's
    problem, and a warm-started inverse-kinematics chain can diverge and finish
    hundreds of millimetres away, which buries the thing under test. Setting the
    solved configuration directly removes that, and costs about a tenth of the
    time, so a description can be scored on scores of grasps instead of four.

    A hand teleported into the object is a **fault, not an artefact**: it means
    the description claimed a clear volume where the hand's own geometry is
    not. The interpenetration is reported rather than hidden.
    """
    from tpgpt.sim.kinematics import solve_ik
    from tpgpt.sim.replay import _joint_action

    reset(env, seed=seed)
    robot = env.robots[0]
    qi = np.asarray(robot.composite_controller.part_controllers[arm].qpos_index)

    if exact_base:
        # The description alone decides the placement: no contact offset, no
        # calibrated depth. This is what a real robot can do.
        position, rotation = site_from_base(env, pose, gripper_short, arm,
                                            extra_depth=extra_depth)
    else:
        position, rotation = eef_pose(pose, gripper_short, tcp_depth)
    result = solve_ik(env, position, rotation, arm, position_tolerance=IK_TOLERANCE)
    if not result.reachable:
        return {"reachable": False, "held": False, "lift_mm": 0.0,
                "reach_mm": float("nan"), "touching": 0}

    env.sim.data.qpos[qi] = result.qpos
    env.sim.data.qvel[qi] = 0.0
    env.sim.forward()

    site = (robot.eef_site_id[arm] if isinstance(robot.eef_site_id, dict)
            else robot.eef_site_id)
    reach = float(np.linalg.norm(position - np.array(env.sim.data.site_xpos[site])))

    start_z = float(_cube_state(env)[0][2])
    seed_q = np.array(result.qpos)
    for _ in range(close_steps):
        env.step(_joint_action(env, robot, arm, seed_q, 1.0))
    touching = _cube_contacts(env)
    for step in range(lift_steps):
        target = position + np.array([0.0, 0.0, LIFT_TO * (step + 1) / lift_steps])
        seed_q = solve_ik(env, target, rotation, arm,
                          position_tolerance=IK_TOLERANCE, seed_qpos=seed_q).qpos
        env.step(_joint_action(env, robot, arm, seed_q, 1.0))
    lift = float(_cube_state(env)[0][2]) - start_z
    return {"reachable": True, "held": lift > LIFT_THRESHOLD,
            "lift_mm": lift * 1000.0, "reach_mm": reach * 1000.0,
            "touching": touching}


def _cube_contacts(env):
    """How many gripper geoms are touching the cube."""
    sim = env.sim
    n = 0
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        a = sim.model.geom_id2name(con.geom1) or ""
        b = sim.model.geom_id2name(con.geom2) or ""
        if ("gripper0" in a and b.startswith("cube")) or \
           ("gripper0" in b and a.startswith("cube")):
            n += 1
    return n


def reachable(env, pose, gripper_short, tcp_depth, arm="right"):
    """Whether the arm can hold the pose this grasp asks for."""
    from tpgpt.sim.kinematics import solve_ik

    position, rotation = eef_pose(pose, gripper_short, tcp_depth)
    return solve_ik(env, position, rotation, arm,
                    position_tolerance=IK_TOLERANCE).reachable
