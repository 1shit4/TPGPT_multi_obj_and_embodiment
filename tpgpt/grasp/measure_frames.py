"""Measure each robosuite gripper's frame against GraspGen-X's convention.

Run as ``python -m tpgpt.grasp.measure_frames`` to regenerate
``gripper_frames.json``. Needs the simulator; nothing else in the grasp package
does, which is why this lives in its own module.

**Why a full frame and not a single number.** The original contract assumed
robosuite's ``grip_site`` sits at the fingertips with ``+Z`` along the approach
and the jaws closing along ``+X`` or ``+Y``. Measured across nine hands, all
three parts of that are false somewhere:

* ``grip_site`` sits at the *gripper base* for the UMI and Inspire hands, and
  75 mm past the config's fingertip depth for the Robotiq 2F-140.
* The UMI's fingers lie along ``grip_site -Z``: its frame is flipped end for
  end relative to every other hand.
* The Inspire hand's fingers travel 7.4 degrees off ``+X``, which neither of the
  two allowed options could express.

Each of those fails silently -- the commanded pose still looks plausible while
the hand closes on air, or drives through the table. So all of it is measured:

``alignment``
    Takes a GraspGen-X grasp rotation to a ``grip_site`` rotation. Built from
    the hand's own axes as observed in simulation, so a flip is just a
    measurement rather than a special case.
``contact_offset``
    Where the object ends up, in ``grip_site`` coordinates. Kept as a full
    3-vector because for an anthropomorphic hand the contact is not on the
    approach axis at all -- the thumb opposes the fingers off to one side.
``finger_geoms`` / ``spread_open`` / ``spread_closed``
    The calibration behind a *cross-hand* measure of how far the jaws have
    shut. See below.

**Why the jaw opening has to be calibrated too.** The obvious reading of "how
far are the jaws open" is the sum of the gripper's joint positions, and it is
wrong in four different ways across these nine hands. Measured, ``sum |qpos|``
from fully open to fully closed:

===========  ========  ========  ==================
hand         open      closed    direction on close
===========  ========  ========  ==================
panda          0.0794    0.0010  **decreases**
umi            0.0678    0.0230  **decreases**
robotiq85      0.9896    1.8056  increases
robotiq140     0.2352    1.9881  increases
xarm           0.2143    4.8980  increases
robotiq3f      1.6764    7.1434  increases
inspire        5.3075    5.8104  increases
rethink        0.0225    0.0237  **+0.0012, no signal**
yumi           0.0250    0.0250  **0.0000, no signal**
===========  ========  ========  ==================

The sign is hand-dependent, because a prismatic finger pair travels toward each
other while a revolute linkage folds inward on a rising angle. The *units* are
hand-dependent too -- metres for the two-jaw prismatic hands, radians for the
revolute ones -- so a threshold that means "shut" on a Panda is 0.001 and on an
XArm is 4.9. And for the Rethink and Yumi hands the named joints in
``gripper.joints`` barely move or do not move at all, while their fingers travel
45.9 mm and 38.3 mm respectively, so the signal is simply absent.

Reading such a number as a width **misattributed a whole multi-gripper
comparison**: a Robotiq 2F-140 closing harder (0.63 -> 1.38 radians) was read as
its jaws flying open, and the conclusion drawn was that the close command was
inverted for that hand. It was not. See ROBOTICS_NOTES.md section 7.28.

So closure is measured the same way the closing axis is: from **geom
displacement**, which needs no joint names and no per-family special case. The
spread of the moving finger geoms along the closing axis is recorded fully open
and fully closed, and at run time
:func:`~tpgpt.experiments.diagnose.jaw_closure_probe` interpolates between them
to give a fraction that means the same thing on every hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tpgpt.grasp.grippers import GRIPPER_PAIRS, gripper_action

#: Where the measured frames are cached.
FRAMES_PATH = Path(__file__).with_name("gripper_frames.json")

#: Geoms moving less than this when the hand closes are structure, not fingers.
MIN_TRAVEL = 2e-4

#: Control steps to hold each open/close command for.
SETTLE_STEPS = 40

#: Fraction of the finger's length, measured back from its tip, that counts as
#: the contact region.
#:
#: Averaging over the *whole* finger drags the contact point back along the
#: finger links, which matters for a long-fingered hand: on the Robotiq 2F-140
#: the whole-finger centroid sits 77 mm behind ``grip_site`` while its
#: fingertips sit 15 mm behind, and commanding the 77 mm version drove the hand
#: below the table and picked up nothing. A depth sweep put the working value
#: near 37 mm, which is what the distal third gives.
DISTAL_FRACTION = 0.35


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise RuntimeError("degenerate direction: the gripper did not actuate")
    return np.asarray(v, dtype=float) / n


def measure_frame(robosuite_name: str, robot: str = "Panda") -> dict:
    """Measure one gripper's frame and contact point.

    Opens the hand fully, closes it fully, and reads three things off the
    simulation, all expressed in the ``grip_site`` frame:

    * **approach** -- from the gripper's root body to the fingertip centroid.
      Not root-to-``grip_site``: that is a zero vector for the two hands whose
      site sits at the base. Not the finger centroid alone either, since a
      finger extends *backwards* from its tip toward the knuckle, which points
      the Panda's approach the wrong way.
    * **closing** -- the principal direction of the finger displacements, with
      the approach component removed.
    * **contact** -- the centroid of the fingers once closed, which is where a
      grasped object sits.

    Returns a dict ready to serialise, including ``anisotropy``: how well one
    closing axis describes the hand. Every parallel jaw scores above 500, every
    multi-finger hand below 7.
    """
    import robosuite as suite

    # Seeded before ``make`` and ``reset``, never after (CLAUDE.md). robosuite
    # samples the object's placement at reset, and an unseeded sample moves the
    # measured finger travel by a few tenths of a micrometre between two
    # otherwise identical runs -- small, but it means re-measuring a hand that
    # has not changed produces a different number, and the whole value of this
    # cache is that a changed number means something changed.
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
            i for i, name in enumerate(sim.model.geom_names) if name and "gripper0" in name
        ]
        site = sim.model.site_name2id(gripper.important_sites["grip_site"])
        root = sim.model.body_name2id(gripper.root_body)

        def geom_positions():
            return np.array([sim.data.geom_xpos[i].copy() for i in geom_ids])

        # ``gripper_action`` fills the whole gripper block. ``action[-1]`` drives
        # one degree of freedom, which measured the Inspire hand's thumb and
        # called its other four fingers stationary -- 9.07 mm of travel against
        # 58.54 mm. See the helper's docstring.
        action = gripper_action(env, gripper, -1.0)
        for _ in range(SETTLE_STEPS):
            env.step(action)
        opened = geom_positions()

        action = gripper_action(env, gripper, 1.0)
        for _ in range(SETTLE_STEPS):
            env.step(action)
        closed = geom_positions()

        rotation = np.array(sim.data.site_xmat[site]).reshape(3, 3)
        site_position = np.array(sim.data.site_xpos[site])
        root_position = np.array(sim.data.body_xpos[root])

        travel = (closed - opened) @ rotation
        moving = np.linalg.norm(travel, axis=1) > MIN_TRAVEL
        if int(moving.sum()) < 2:
            raise RuntimeError(f"{robosuite_name}: gripper did not actuate")

        # Local coordinates of the fingers once closed, and of the root body.
        fingers_local = (closed[moving] - site_position) @ rotation
        root_local = (root_position - site_position) @ rotation

        # Approach: root body -> fingertip cloud.
        approach = _unit(fingers_local.mean(axis=0) - root_local)

        # Contact region: the distal part of the fingers, where an object is
        # actually held, rather than the whole finger including its knuckle.
        along = fingers_local @ approach
        cutoff = along.max() - DISTAL_FRACTION * float(np.ptp(along) or 1.0)
        contact_local = fingers_local[along >= cutoff].mean(axis=0)

        # Closing: principal direction of travel, approach component removed.
        flat = travel[moving] - np.outer(travel[moving] @ approach, approach)
        _, singular, right = np.linalg.svd(flat, full_matrices=False)
        closing = _unit(right[0] - float(right[0] @ approach) * approach)

        basis = np.column_stack([closing, np.cross(approach, closing), approach])

        # Closure calibration. The spread of the *moving* geoms along the
        # closing axis, fully open and fully closed. Both are taken in the
        # world frame along the same world axis, because the arm is stationary
        # for this measurement and the site frame does not move between the two
        # readings -- so no re-projection is needed and none is done.
        closing_world = rotation @ closing
        finger_names = [
            sim.model.geom_names[geom_ids[i]] for i in np.flatnonzero(moving)
        ]

        def spread(points: np.ndarray) -> float:
            projected = points[moving] @ closing_world
            return float(projected.max() - projected.min())

        spread_open, spread_closed = spread(opened), spread(closed)
        return {
            "robosuite": robosuite_name,
            # R_site = R_grasp @ alignment puts the hand's own axes onto the
            # grasp's, whatever orientation the model happens to use.
            "alignment": basis.T.tolist(),
            "contact_offset": contact_local.tolist(),
            "anisotropy": float(singular[0] / max(singular[1], 1e-12)),
            "closing_angle_deg": float(
                ((np.degrees(np.arctan2(closing[1], closing[0])) + 90.0) % 180.0) - 90.0
            ),
            "approach_in_site": approach.tolist(),
            "n_moving_geoms": int(moving.sum()),
            # --- closure calibration, see the module docstring ---
            #: Closing direction in ``grip_site`` coordinates. Redundant with
            #: ``alignment`` (it is its first row) but stored plainly, because
            #: the runtime probe needs exactly this and reconstructing it from
            #: the alignment convention is the sort of step that gets a sign
            #: wrong.
            "closing_in_site": closing.tolist(),
            #: Names of the geoms that actually moved. Names rather than
            #: indices: indices depend on what else is in the scene.
            "finger_geoms": finger_names,
            #: Finger spread along the closing axis, in metres, jaws fully
            #: open and fully closed. Not a jaw aperture -- the moving set
            #: includes knuckles and outer links, so the Robotiq 2F-140 reads
            #: 165 mm open with a published 125 mm aperture. It is a
            #: *monotone, hand-calibrated* travel measure, which is what a
            #: closure fraction needs and what ``sum |qpos|`` failed to be.
            "spread_open": spread_open,
            "spread_closed": spread_closed,
            #: Whether ``+1`` shuts this hand. Measured, not assumed: the
            #: per-gripper ``format_action`` sign multipliers differ between
            #: hands (the Panda's is ``[-1, +1]``, the Robotiq 2F-140's is
            #: ``[+1, -1]``), which looks like an inverted command and is not
            #: -- they compensate for opposite joint conventions in the two
            #: models. Measured ``True`` for eight of the nine hands. The
            #: Inspire hand reads ``False`` and still shuts -- see
            #: ``finger_travel_mm`` below, where the spread definition rather
            #: than the hand is what fails.
            "plus_one_closes": bool(spread_closed < spread_open - 1e-4),
            #: How far the fingers travel, in mm: ``spread_open`` minus
            #: ``spread_closed``, so it is positive when the hand shuts.
            #:
            #: **Negative means this definition of spread does not describe
            #: this hand**, not that the hand opens on ``+1``. Spread is the
            #: extent of the moving geoms along one closing axis, which is
            #: exactly right for a jaw whose fingers travel toward each other
            #: along a line, and wrong for an anthropomorphic hand whose
            #: fingers curl through an arc: the Inspire hand's five fingers
            #: sweep *outward* in extent while closing *inward* on the thumb,
            #: and it reads -1.1 mm. ``plus_one_closes`` then comes out False
            #: for a hand that does shut, which is why nothing may threshold
            #: these two numbers without checking that flag first.
            #:
            #: This field previously read 0.7 mm for the Inspire hand, and that
            #: number is the reason it was excluded from every campaign. It was
            #: measured with one of its six actuators driven -- see
            #: ``grippers.gripper_action``. Driven fully it travels ~57 mm.
            "finger_travel_mm": (spread_open - spread_closed) * 1000.0,
        }
    finally:
        env.close()


def measure_all(pairs=None) -> dict:
    """Measure every registered gripper and return the table."""
    pairs = pairs or list(GRIPPER_PAIRS)
    frames = {}
    for short in pairs:
        pair = GRIPPER_PAIRS[short]
        robot = "Panda" if "Panda" in pair.robots else pair.robots[0]
        try:
            frames[short] = measure_frame(pair.robosuite, robot)
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"  {short:<11} FAILED: {type(exc).__name__}: {exc}")
    return frames


def main(path: str | Path = FRAMES_PATH) -> Path:
    """Re-measure every gripper and **merge** the result into the cache.

    Merged, not overwritten. ``gripper_frames.json`` also carries
    ``calibrated_depth``, which comes from a much more expensive physics sweep
    (:func:`tpgpt.grasp.verify.calibrate_depth`, 13 grasp attempts per hand),
    and which :func:`~tpgpt.grasp.grippers._physics_verified` reads to decide
    which hands campaigns may use. A plain overwrite here silently emptied that
    list, and an empty list falls back to *all* measured pairs -- so the Inspire
    hand, which lifts nothing, would quietly re-enter every campaign.
    """
    frames = measure_all()
    path = Path(path)
    existing = json.loads(path.read_text()) if path.is_file() else {}
    for short, frame in frames.items():
        merged = dict(existing.get(short, {}))
        merged.update(frame)
        existing[short] = merged
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(
        f"{'gripper':<11}{'closing':>9}{'aniso':>12}{'contact offset (mm)':>26}"
        f"{'travel mm':>11}{'+1 shuts':>10}  approach"
    )
    for short, frame in frames.items():
        offset = np.array(frame["contact_offset"]) * 1000
        print(
            f"{short:<11}{frame['closing_angle_deg']:8.1f}d{frame['anisotropy']:12.1f}"
            f"{str(np.round(offset, 1)):>26}{frame['finger_travel_mm']:11.1f}"
            f"{str(frame['plus_one_closes']):>10}  "
            f"{np.round(frame['approach_in_site'], 2)}"
        )
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    main()
