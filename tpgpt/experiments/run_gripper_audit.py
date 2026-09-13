"""What does robosuite actually give us for a hand, before anything is built on it?

Mounts a gripper on an arm, drives it fully open and fully closed, and records
what moved. Three questions, all answerable in about five seconds per hand and
none of them needing GraspGen-X, a scene, or a grasp:

**Does it mount and actuate at all?** A registry entry for a hand that cannot be
built is a campaign that dies twenty minutes in. This answers it up front, which
is the rule ``diagnose.replay_preconditions`` already applies to campaigns --
never let a run be the first test of its own harness.

**How many degrees of freedom does the gripper have, and does the command reach
all of them?** robosuite lays an action vector out as the arm's degrees of
freedom followed by the gripper's, so the jaws occupy the *last* ``dof``
entries. Two measurement sites in this package wrote ``action[-1] = 1.0`` and so
drove exactly one actuator. For the seven one-actuator hands that is the whole
gripper and nothing was wrong; for the Inspire hand it drove the thumb and
recorded the other four fingers as stationary, which is how a hand that travels
~57 mm came to be written up as one that "does not actuate". This driver
measures both commands and reports the gap, so the defect is a column in a table
rather than an argument. ``tpgpt.grasp.grippers.gripper_action`` is the repair.

**How far do the fingers travel?** The number that says whether a hand is worth
onboarding. It is *not* a jaw aperture -- the moving set includes knuckles and
outer links -- but it is monotone and per hand, which is what a closure fraction
needs and what ``sum |qpos|`` failed to be (ROBOTICS_NOTES.md section 7.28).

Run it over the registry, or over every gripper robosuite can build::

    python -m tpgpt.experiments.run_gripper_audit --out outputs/gripper_audit
    python -m tpgpt.experiments.run_gripper_audit --all-robosuite --out outputs/gripper_audit_all

Writes ``manifest.json`` (with provenance, automatically) and ``rows.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgpt.grasp.grippers import GRIPPER_PAIRS, gripper_action
from tpgpt.reporting.html import write_manifest

#: Geoms moving less than this are structure, not fingers. Matches
#: ``measure_frames.MIN_TRAVEL`` so the two agree about what a finger is.
MIN_TRAVEL = 2e-4

#: A geom past this has unambiguously travelled, rather than been nudged by a
#: neighbour. Counting these is what separates "one jaw closed" from "both did":
#: the UMI's maximum travel is identical either way because the finger the
#: scalar command happens to drive moves the same distance regardless.
CLEAR_TRAVEL = 1e-3

#: Control steps held at each of open and closed. Matches ``measure_frames``.
SETTLE_STEPS = 40

#: Arm to mount on. Only the gripper's own motion is measured, so the arm does
#: not matter beyond compatibility.
DEFAULT_ROBOT = "Panda"

#: Every gripper robosuite can build that is a *hand* -- the wiping pad and the
#: null gripper are neither, and the left-handed mirrors of hands already listed
#: add no information.
_NOT_A_HAND = ("WipingGripper", "NullGripper", "AlohaGripperBase")


def audit_gripper(robosuite_name: str, robot: str = DEFAULT_ROBOT) -> dict:
    """Mount one gripper, close it, and report what moved.

    Both commands are measured on their own freshly-built environment, seeded
    identically, so the only difference between them is the command itself.

    Args:
        robosuite_name: Gripper class name, e.g. ``"SchunkSvhRightHand"``.
        robot: Arm to mount it on.

    Returns:
        A flat, JSON-safe row. ``mounted`` is False with an ``error`` string
        rather than raising, so one unbuildable hand does not end the audit.
    """
    import robosuite as suite

    row = {"robosuite": robosuite_name, "robot": robot, "mounted": False}

    def measure(whole_block: bool):
        # Seeded before make and reset, never after. robosuite samples the
        # object placement at reset, and an unseeded sample moves the measured
        # travel by up to 0.6 um between two otherwise identical runs -- small,
        # but the whole point here is to compare two runs.
        np.random.seed(0)
        env = suite.make(
            "Lift",
            robots=robot,
            gripper_types=robosuite_name,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )
        try:
            env.reset()
            sim = env.sim
            gripper = env.robots[0].gripper
            gripper = gripper["right"] if isinstance(gripper, dict) else gripper
            geom_ids = [
                i for i, n in enumerate(sim.model.geom_names) if n and "gripper0" in n
            ]
            site = sim.model.site_name2id(gripper.important_sites["grip_site"])

            def positions():
                return np.array([sim.data.geom_xpos[i].copy() for i in geom_ids])

            def command(value):
                if whole_block:
                    return gripper_action(env, gripper, value)
                action = np.zeros(env.action_dim)
                action[-1] = value
                return action

            for _ in range(SETTLE_STEPS):
                env.step(command(-1.0))
            opened = positions()
            rotation = np.array(sim.data.site_xmat[site]).reshape(3, 3)
            for _ in range(SETTLE_STEPS):
                env.step(command(1.0))

            travel = np.linalg.norm((positions() - opened) @ rotation, axis=1)
            return {
                "dof": int(gripper.dof),
                "action_dim": int(env.action_dim),
                "geoms": len(geom_ids),
                "max_travel_mm": float(travel.max() * 1000.0),
                "moving_geoms": int((travel > MIN_TRAVEL).sum()),
                "travelled_geoms": int((travel > CLEAR_TRAVEL).sum()),
            }
        finally:
            env.close()

    try:
        scalar = measure(whole_block=False)
        full = measure(whole_block=True)
    except Exception as exc:  # reported as a row, not raised
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row.update(
        mounted=True,
        dof=full["dof"],
        action_dim=full["action_dim"],
        geoms=full["geoms"],
        # The repaired command -- what the hand actually does.
        travel_mm=full["max_travel_mm"],
        moving_geoms=full["moving_geoms"],
        travelled_geoms=full["travelled_geoms"],
        # The defective command, kept so the gap is evidence rather than claim.
        scalar_travel_mm=scalar["max_travel_mm"],
        scalar_moving_geoms=scalar["moving_geoms"],
        scalar_travelled_geoms=scalar["travelled_geoms"],
        # Positive means the scalar command under-drove this hand.
        travel_shortfall_mm=full["max_travel_mm"] - scalar["max_travel_mm"],
        fingers_lost=full["travelled_geoms"] - scalar["travelled_geoms"],
        actuates=full["max_travel_mm"] > CLEAR_TRAVEL * 1000.0,
    )
    return row


def robosuite_hands() -> list[str]:
    """Every gripper robosuite can build that is a hand, sorted."""
    from robosuite.models.grippers import GRIPPER_MAPPING

    return sorted(
        n for n in GRIPPER_MAPPING
        if n is not None and n not in _NOT_A_HAND
    )


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/gripper_audit")
    parser.add_argument(
        "--grippers", default=None,
        help="Comma-separated robosuite class names. Defaults to the registry.",
    )
    parser.add_argument(
        "--all-robosuite", action="store_true",
        help="Audit every gripper robosuite can build, registered or not.",
    )
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    args = parser.parse_args(argv)

    if args.all_robosuite:
        names = robosuite_hands()
        scope = "every gripper robosuite can build"
    elif args.grippers:
        names = [n.strip() for n in args.grippers.split(",") if n.strip()]
        scope = "named on the command line"
    else:
        names = [p.robosuite for p in GRIPPER_PAIRS.values()]
        scope = "the nine pairs in tpgpt.grasp.grippers.GRIPPER_PAIRS"

    registered = {p.robosuite: s for s, p in GRIPPER_PAIRS.items()}
    rows = []
    print(
        f"{'gripper':<36}{'dof':>4}{'travel mm':>11}{'scalar':>9}"
        f"{'short mm':>10}{'fingers lost':>14}  registry"
    )
    for name in names:
        row = audit_gripper(name, args.robot)
        row["registry_short"] = registered.get(name)
        rows.append(row)
        if not row["mounted"]:
            print(f"{name:<36}  FAILED: {row['error']}")
            continue
        print(
            f"{name:<36}{row['dof']:>4}{row['travel_mm']:>11.2f}"
            f"{row['scalar_travel_mm']:>9.2f}{row['travel_shortfall_mm']:>10.2f}"
            f"{row['fingers_lost']:>14}  {row['registry_short'] or '-'}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    write_manifest(
        out,
        title="Gripper audit: does it mount, does it actuate, is it fully commanded?",
        description=(
            "Each gripper mounted on an arm and driven fully open to fully "
            "closed, twice: once commanding only the last entry of the action "
            "vector and once commanding the whole gripper block. The gap "
            "between the two is what the scalar command was mis-measuring."
        ),
        settings={
            "varied": {"gripper": names},
            "fixed": {
                "robot": args.robot,
                "scope": scope,
                "task": "robosuite Lift, no cameras, control_freq 20",
                "seed": "np.random.seed(0) before make and reset, per run",
                "settle_steps": SETTLE_STEPS,
                "min_travel_m": MIN_TRAVEL,
                "clear_travel_m": CLEAR_TRAVEL,
            },
        },
        results={
            "audited": len(rows),
            "mounted": sum(1 for r in rows if r["mounted"]),
            "actuating": sum(1 for r in rows if r.get("actuates")),
            "under_driven_by_scalar_command": sum(
                1 for r in rows if r.get("fingers_lost", 0) > 0
                or r.get("travel_shortfall_mm", 0.0) > 1.0
            ),
        },
    )
    print(f"\nwrote {out}/rows.json and manifest.json")
    return out


if __name__ == "__main__":
    main()
