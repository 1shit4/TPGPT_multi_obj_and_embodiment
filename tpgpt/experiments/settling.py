"""Post-execution evidence, without changing the placement tolerance."""
import numpy as np


def settled_verdict(samples, window, position_span=.002):
    """Require support, release and the original slot score throughout a window.

    This additional diagnostic never rescues a failed execution. The position
    span is a stability criterion, not a larger goal tolerance.
    """
    if window < 2 or len(samples) < window:
        return {"passed": False, "reason": "insufficient_samples"}
    tail = samples[-window:]
    span = float(np.linalg.norm(np.ptp([s["position"] for s in tail], axis=0)))
    checks = {
        "in_slot": all(s["in_slot"] for s in tail),
        "supported": all(s["supported"] for s in tail),
        "released": all(not s["robot_contact"] for s in tail),
        "stable": span <= position_span,
    }
    return {"passed": all(checks.values()), **checks, "position_span_m": span,
            "position_span_limit_m": position_span, "window_steps": window}


def observe_settling(env, object_name, slot, seconds=2., arm="right"):
    """Hold the final arm configuration with open jaws for a fixed duration.

    Only position-controller executors may call this. Recompute joint deltas
    on every step: repeating a nonzero delta action would move the arm.
    """
    from tpgpt.sim.replay import _joint_action
    from tpgpt.sim.rollout import slot_score

    if not np.isfinite(seconds) or seconds < 1.:
        raise ValueError("post-execution settling must last at least one second")
    robot = env.robots[0]
    controller = robot.composite_controller.part_controllers[arm]
    target = np.array(env.sim.data.qpos[np.asarray(controller.qpos_index)], copy=True)
    score = slot_score(object_name, slot)
    obj = next(o for o in env.objects if o.name == object_name)
    # Body ancestry, rather than substring matching, identifies all object geoms.
    model, data = env.sim.model, env.sim.data
    root = model.body_name2id(obj.root_body)
    body_ids = {root}
    for body in range(root + 1, model.nbody):
        if model.body_parentid[body] in body_ids:
            body_ids.add(body)
    object_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in body_ids}
    samples = []
    for _ in range(int(np.ceil(seconds * env.control_freq))):
        env.step(_joint_action(env, robot, arm, target, -1.))
        others = []
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in object_geoms and c.geom2 not in object_geoms:
                others.append(model.geom_id2name(c.geom2) or "")
            elif c.geom2 in object_geoms and c.geom1 not in object_geoms:
                others.append(model.geom_id2name(c.geom1) or "")
        samples.append({"simulation_time": float(data.time),
                        "position": env.object_position(object_name).copy(),
                        "in_slot": bool(score(env)[0]),
                        "supported": f"shelf_{slot.split('_')[0]}_board" in others,
                        "robot_contact": any(n.startswith(("robot", "gripper")) for n in others),
                        "contacts": others})
    _, final_offset = score(env)
    return {"duration_seconds": seconds, "final_offset_m": np.asarray(final_offset),
            "placement_error_xy_m": float(np.linalg.norm(final_offset[:2])),
            **settled_verdict(samples, max(2, int(env.control_freq))), "samples": samples}
