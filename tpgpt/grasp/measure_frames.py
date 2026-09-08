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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tpgpt.grasp.grippers import GRIPPER_PAIRS

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

        action = np.zeros(env.action_dim)
        action[-1] = -1.0
        for _ in range(SETTLE_STEPS):
            env.step(action)
        opened = geom_positions()

        action[-1] = 1.0
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
    frames = measure_all()
    path = Path(path)
    path.write_text(json.dumps(frames, indent=2, sort_keys=True))
    print(f"{'gripper':<11}{'closing':>9}{'aniso':>12}{'contact offset (mm)':>26}  approach")
    for short, frame in frames.items():
        offset = np.array(frame["contact_offset"]) * 1000
        print(
            f"{short:<11}{frame['closing_angle_deg']:8.1f}d{frame['anisotropy']:12.1f}"
            f"{str(np.round(offset, 1)):>26}  {np.round(frame['approach_in_site'], 2)}"
        )
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    main()
