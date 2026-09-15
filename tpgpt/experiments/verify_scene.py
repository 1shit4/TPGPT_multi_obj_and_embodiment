"""Prove the real-sized scene before anything is measured in it.

``ROBOTICS_NOTES.md`` 7.32 is the reason this file exists. The benchmark
tabletop scene handed objects over **while they were still falling, from inside
the robot**, and it did so differently for every gripper: the same seed put the
cereal box's origin anywhere from 804 to 889 mm across six hands, a spread of
154 mm, so every cross-gripper comparison was comparing different worlds. Three
separate defects caused it, and none of them raised anything -- the scene built,
the cloud came back, the grasp planner proposed candidates, the physics ran.

The lesson recorded there is sharper than "check the scene". It is that
**verification written after a change checks what its author expects rather than
what is true**: the first fix froze the objects along with the arm, and every
metric then reported perfection -- zero interpenetration, zero cross-gripper
spread, everything upright -- precisely because nothing had moved. So each check
here is written to fail on a frozen scene as loudly as on a falling one, and the
settle check asserts that the objects **did** move before they stopped.

Five checks, in the order a defect would appear:

``placement``      no object starts inside the robot.
``settling``       the scene converges, and it converges by coming to rest
                   rather than by never having moved.
``cross_gripper``  the object's rest pose is identical across all seven hands.
``upright``        the object still stands the way the pick configuration asked.
``reach``          every hand can put its fingertips where the task requires,
                   at the pick and at each destination.
``destination``    the object *fits* the destination it is sent to.

Usage::

    python -m tpgpt.experiments.verify_scene --record    # write the rest poses
    python -m tpgpt.experiments.verify_scene             # run the checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.sim.scenes.tabletop_shelf import (
    REAL_DESTINATIONS,
    REAL_PICK_CONFIGS,
    TabletopShelf,
)

#: Hands the campaign runs on. Fixed on measurement in ``ROBOTICS_NOTES`` 7.42.
FLEET = ("xarm", "robotiq3f", "robotiq140", "robotiq85", "robotiq3f_dex",
         "panda", "rethink")

#: Objects the campaign runs on, also fixed in 7.42.
OBJECTS = ("can", "cereal", "hammer", "milk", "mug")

#: The hand the rest poses are recorded with.
#:
#: Any hand would do -- the point is that **one** of them is used for all of
#: them -- but the Panda is the one the source demonstration is recorded on and
#: the lightest in the fleet, so it perturbs the settle least.
REFERENCE_HAND = "panda"

#: How far an object may move, in metres, after its recorded rest pose is
#: restored into a different gripper's model. The paper plan asks for 0.1 mm.
REST_TOLERANCE = 1e-4

#: How far an object must have moved *during* the settle for the settle to count
#: as a settle rather than as a freeze. It is dropped ~0.2 mm onto the table by
#: ``_seat_objects`` and then finds its contact, so anything at machine zero
#: means the physics never ran.
SETTLE_MIN_MOTION = 1e-5

#: Steps of free physics used to ask whether a restored rest pose is actually a
#: rest pose in *this* gripper's model. 500 steps is 1 s.
HOLD_STEPS = 500

#: Extra physics run at record time, after the scene's own settle has declared
#: convergence, before the rest pose is written down.
#:
#: **The scene's settle is not tight enough to block a condition on.** Its
#: criterion is 0.5 mm of displacement over a 50-step window, held for four
#: windows, which is ample to stop a cloud being captured mid-topple and is two
#: orders of magnitude looser than the 0.1 mm this file asks for. Measured on
#: the cereal carton at pick P1 -- a 190 x 80 x 300 box standing on its narrow
#: face at 55 degrees of yaw, the least stable placement in the grid -- the
#: scene declares convergence at 250 steps while the box is still moving
#: 0.05 mm per 500 steps, and it does not reach machine zero until about 1500:
#:
#: =========  =====================================================
#: steps      further motion in the next 500 (mm)
#: =========  =====================================================
#: 500        0.021
#: 1000       0.052
#: 1500       0.002
#: 2000+      0.000
#: =========  =====================================================
#:
#: Recorded at 250 steps that residual motion is *not* a property of the
#: recording -- it is spent later, inside whichever gripper's model restores it,
#: and the seven hands spend it differently: 0.587 mm of disagreement after one
#: second of physics. Recorded at 2250 there is nothing left to spend. Settled
#: from scratch the three hands tried agree to 0.000 mm at every checkpoint, so
#: the divergence was the unfinished settle and not the hand.
RECORD_SETTLE_STEPS = 2000


def _scene(gripper: str, obj: str, pick: str, **kwargs) -> TabletopShelf:
    from tpgpt.experiments.pipeline import build_scene

    return build_scene((obj,), gripper=gripper, seed=0, world="real",
                       pick_config=pick, **kwargs)


def _object_qpos(env, name: str) -> np.ndarray:
    return np.array(env.sim.data.get_joint_qpos(env._object_by_name(name).joints[0]))


def record(path: Path) -> dict:
    """Settle every (object, pick) once on the reference hand and write it out.

    One recording, reused by every gripper, is what makes the condition block
    exact. Settling per gripper instead is what ``ROBOTICS_NOTES`` 7.28
    measured at 14.6 mm of disagreement between a Panda and a Robotiq 2F-140
    from one seed -- a difference that then travels through every cross-gripper
    comparison as a confound.
    """
    states = json.loads(path.read_text()) if path.exists() else {}
    for obj in OBJECTS:
        for pick in REAL_PICK_CONFIGS:
            env = _scene(REFERENCE_HAND, obj, pick)
            if env.settled_state_restored:
                env.close()
                continue
            env._hold_arm()
            before = _object_qpos(env, obj)[:3]
            for _ in range(RECORD_SETTLE_STEPS):
                env.sim.step()
            after = _object_qpos(env, obj)[:3]
            states[env._state_key()] = {obj: _object_qpos(env, obj).tolist()}
            print(f"  recorded {obj:7s} {pick}  settled at {env.settle_steps_taken} "
                  f"steps, moved a further "
                  f"{np.abs(after - before).max() * 1000:.3f} mm over "
                  f"{RECORD_SETTLE_STEPS}")
            env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(states, indent=1, sort_keys=True) + "\n")
    return states


def check_placement_and_settling() -> list[dict]:
    """No object starts inside the robot, and the scene settles by resting."""
    rows = []
    for obj in OBJECTS:
        for pick in REAL_PICK_CONFIGS:
            env = _scene(REFERENCE_HAND, obj, pick)
            before = env._initial_poses()[obj][0]
            after = _object_qpos(env, obj)[:3]
            rows.append({
                "object": obj, "pick": pick,
                "penetration_mm": env.placement_penetration * 1000.0,
                "settle_steps": env.settle_steps_taken,
                "hit_cap": env.settle_steps_taken >= env.MAX_SETTLE_STEPS,
                # Seating drops the object a fraction of a millimetre onto the
                # table; if this is exactly zero the physics did not run.
                "moved_during_settle_mm": float(
                    np.linalg.norm(np.asarray(after) - np.asarray(before))) * 1000.0,
                "restored": env.settled_state_restored,
            })
            env.close()
    return rows


def check_cross_gripper() -> list[dict]:
    """The object's rest pose is the same for every hand, and stays put.

    Two questions, and the second is the one that a frozen scene would pass on
    its own. The first is whether the seven hands agree at reset, which after
    restoring a recorded pose is true by construction. The second is whether
    that pose is an **equilibrium in each hand's own model**: the scene is left
    to run for a second with the arm held clear, and an object that drifts says
    the recorded pose was recorded in a model that does not describe this one.
    """
    rows = []
    for obj in OBJECTS:
        for pick in REAL_PICK_CONFIGS:
            at_reset, after_hold = {}, {}
            for hand in FLEET:
                env = _scene(hand, obj, pick)
                at_reset[hand] = _object_qpos(env, obj)[:3]
                env._hold_arm()
                for _ in range(HOLD_STEPS):
                    env.sim.step()
                after_hold[hand] = _object_qpos(env, obj)[:3]
                env.close()
            reset = np.array([at_reset[h] for h in FLEET])
            held = np.array([after_hold[h] for h in FLEET])
            rows.append({
                "object": obj, "pick": pick,
                "spread_at_reset_mm": float(np.abs(reset - reset[0]).max()) * 1000.0,
                "spread_after_hold_mm": float(np.abs(held - held[0]).max()) * 1000.0,
                "drift_mm": float(np.abs(held - reset).max()) * 1000.0,
            })
            env = None
    return rows


def check_upright() -> list[dict]:
    """The object still sits the way the pick configuration asked it to.

    Measured as the angle between the object's own ``z`` axis as placed and as
    it ends up. A carton that has toppled reads 90 degrees; the benchmark scene
    produced exactly that, silently, for four of six hands.
    """
    from robosuite.utils.transform_utils import quat2mat

    rows = []
    for obj in OBJECTS:
        for pick in REAL_PICK_CONFIGS:
            env = _scene(REFERENCE_HAND, obj, pick)
            wanted = env._initial_poses()[obj][1]
            got = _object_qpos(env, obj)[3:]
            # ``quat2mat`` takes (x, y, z, w); both stored here as (w, x, y, z).
            axis_a = quat2mat(np.roll(np.asarray(wanted), -1))[:, 2]
            axis_b = quat2mat(np.roll(np.asarray(got), -1))[:, 2]
            rows.append({
                "object": obj, "pick": pick,
                "tilt_deg": float(np.degrees(
                    np.arccos(np.clip(axis_a @ axis_b, -1.0, 1.0)))),
            })
            env.close()
    return rows


def check_reach() -> list[dict]:
    """Every hand can put its fingertips where the task needs them.

    Three poses per cell, all top-down, and the middle one is the one that
    decides the scene's geometry:

    * the **pick**, at the object's mid-height above its table pose;
    * the **place**, at the object's mid-height above each destination board;
    * the **clearance** pose 80 mm above the place, which the approach passes
      through.

    This is a kinematics question and nothing else -- ``solve_ik`` carries no
    collision model (``ROBOTICS_NOTES`` 7.38), so a pose it reaches may still be
    buried in the shelf. Collision is the grasp funnel's job; this check exists
    to catch a shelf placed outside the arm's envelope, which is the defect that
    made the benchmark scene's 0.22 m board unusable once objects were real.
    """
    from tpgpt.grasp.grasps import contact_offset
    from tpgpt.sim.kinematics import solve_ik

    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    rows = []
    for hand in FLEET:
        offset = contact_offset(hand)
        for obj in OBJECTS:
            for pick in REAL_PICK_CONFIGS:
                env = _scene(hand, obj, pick)
                vertices = env._geom_vertices(env.object_body_ids[obj])
                base, top = vertices[:, 2].min(), vertices[:, 2].max()
                mid = (base + top) / 2.0
                position = _object_qpos(env, obj)[:3]
                targets = {"pick": np.array([position[0], position[1], mid])}
                for slot in REAL_DESTINATIONS:
                    board = env.slot_poses()[slot]
                    height = mid - base
                    targets[slot] = np.array([board[0], board[1], board[2] + height])
                    targets[f"{slot}_clear"] = targets[slot] + [0, 0, 0.08]
                for label, target in targets.items():
                    result = solve_ik(env, target - rotation @ offset, rotation,
                                      arm="right")
                    rows.append({
                        "gripper": hand, "object": obj, "pick": pick,
                        "pose": label, "reachable": bool(result.reachable),
                        "target": np.round(target, 4).tolist(),
                    })
                env.close()
    return rows


def check_destination_fit() -> list[dict]:
    """Does each object fit the slot it is sent to, lying in its own footprint?

    **The check that was missing, and it cost a 210-cell campaign nine cells.**
    Every other check here asks whether the scene hands the task over cleanly.
    None of them asks whether the task is *possible*: whether the object, once
    carried to its destination, physically fits between the panels.

    It does not always. The top cubby leaves **282 mm** of clear depth between
    the board's front edge and the back panel, and the hammer is **333 mm**
    long. Its footprint along that axis is 345 mm at 55 degrees of yaw, 330 at
    90 and 355 at 110, so for most orientations it cannot be put there at all.
    Measured against the poses actually commanded in the first real-world
    campaign, **21 of 42 hammer cells** were sent to a pose that intersects the
    shelf -- through the back panel by up to 36.7 mm and the side wall by 27.2.

    This is a cheap, exact, physics-free computation, and running it after the
    campaign rather than before is what makes those cells uninterpretable: a
    cell whose destination cannot hold the object is outside the task, not a
    failure of the method.

    The bound here is the **worst case over yaw**, because the placed
    orientation is not known until a grasp has been chosen: an object whose
    longest horizontal footprint exceeds the clear depth *may* be commanded into
    an impossible pose. An object that fits at every yaw never can be.
    """
    from tpgpt.sim.objects import make_object, real_size_mm, rest_quat
    from tpgpt.sim.scenes.tabletop_shelf import (
        REAL_CUBBY_HEIGHT,
        REAL_SHELF_BOARD_DEPTH,
        REAL_SHELF_BOARD_WIDTH,
        REAL_SHELF_LEVELS,
        REAL_SHELF_THICKNESS,
    )
    from robosuite.utils.transform_utils import quat2mat

    level = {label: (x, height) for label, x, height in REAL_SHELF_LEVELS}["top"]
    depth = REAL_SHELF_BOARD_DEPTH["top"]
    clear_x = depth - REAL_SHELF_THICKNESS
    clear_y = REAL_SHELF_BOARD_WIDTH - 2 * REAL_SHELF_THICKNESS
    rows = []
    for obj in OBJECTS:
        size = np.asarray(real_size_mm(obj), dtype=float) / 1000.0
        rest = quat2mat(np.roll(rest_quat(obj), -1))
        worst_x = worst_yaw = 0.0
        for yaw in np.linspace(0.0, np.pi, 181):
            spin = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                             [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            extent = float(np.abs(spin @ rest)[0] @ size)
            if extent > worst_x:
                worst_x, worst_yaw = extent, float(np.degrees(yaw))
        rows.append({
            "object": obj,
            "clear_depth_mm": clear_x * 1000,
            "worst_footprint_mm": worst_x * 1000,
            "worst_yaw_deg": round(worst_yaw, 1),
            "fits_at_every_yaw": bool(worst_x <= clear_x),
            "height_mm": size[2] * 1000,
            "cubby_height_mm": REAL_CUBBY_HEIGHT["top"] * 1000,
        })
    return rows


def _report(name: str, rows: list[dict], failures: list[dict]) -> None:
    mark = "PASS" if not failures else f"FAIL ({len(failures)}/{len(rows)})"
    print(f"\n### {name}: {mark}")
    for row in failures[:12]:
        print("   ", row)


def main(record_states: bool = False) -> int:
    root = Path(__file__).resolve().parents[2]
    path = root / TabletopShelf.SETTLED_STATES
    if record_states:
        print("recording rest poses on the reference hand")
        record(path)
        print(f"wrote {path}")
        return 0

    if not path.exists():
        print(f"no recorded rest poses at {path}; run with --record first")
        return 2

    failed = 0

    rows = check_placement_and_settling()
    bad = [r for r in rows
           if r["penetration_mm"] > 0.0 or r["hit_cap"]
           or (not r["restored"] and r["moved_during_settle_mm"] < SETTLE_MIN_MOTION * 1000)]
    _report("placement and settling", rows, bad)
    failed += len(bad)

    rows = check_cross_gripper()
    bad = [r for r in rows if max(r["spread_at_reset_mm"],
                                  r["spread_after_hold_mm"]) > REST_TOLERANCE * 1000]
    _report("cross-gripper rest pose", rows, bad)
    print("    worst spread after a second of physics: "
          f"{max(r['spread_after_hold_mm'] for r in rows):.4f} mm")
    failed += len(bad)

    rows = check_upright()
    bad = [r for r in rows if r["tilt_deg"] > 5.0]
    _report("upright", rows, bad)
    print(f"    worst tilt: {max(r['tilt_deg'] for r in rows):.2f} deg")
    failed += len(bad)

    rows = check_reach()
    bad = [r for r in rows if not r["reachable"]]
    _report("reach", rows, bad)
    failed += len(bad)

    rows = check_destination_fit()
    bad = [r for r in rows if not r["fits_at_every_yaw"]]
    _report("destination fit", rows, bad)
    for row in rows:
        print(f"    {row['object']:8s} worst footprint {row['worst_footprint_mm']:6.1f} mm "
              f"at {row['worst_yaw_deg']:5.1f} deg against {row['clear_depth_mm']:.0f} mm "
              f"of clear depth"
              + ("" if row["fits_at_every_yaw"] else "   <-- CAN BE COMMANDED INTO THE PANELS"))
    failed += len(bad)

    print(f"\n{'ALL CHECKS PASS' if not failed else f'{failed} CHECKS FAILED'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true",
                        help="settle each (object, pick) once on the reference "
                             "hand and write the rest poses, rather than checking")
    raise SystemExit(main(parser.parse_args().record))
