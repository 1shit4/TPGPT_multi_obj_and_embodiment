"""What a campaign must write down while it runs, because it cannot be recovered.

Most of a result can be re-derived later: a success rate can be recounted, a
placement error re-measured from a saved trajectory, a stage re-attributed by a
better diagnostic. Three things cannot, and each of them is a whole analysis:

**Which condition a cell faced.** A row carrying only the gripper and the object
says nothing about which of the six conditions produced it, so cells cannot be
paired, and a mixed-effects model with condition as a random effect cannot be
fitted at all. :func:`condition_id` and :func:`condition_block` write it down.

**The map itself.** Comparing maps to each other is a subsection of the paper,
and no representation of a map is saved anywhere: a campaign stores ``min_det``,
which is one scalar summary, and two entirely different maps can share it. A map
is a deformation of space, so it is recorded as one --- :func:`fingerprint`
evaluates ``phi`` on a grid that is **identical in every cell** and stores the
displacement ``phi(x) - x``. Displacements rather than absolute positions
because the grid is common and subtracting it is lossless, and because the
displacements are small, so float32 costs nothing in precision and halves the
file.

**What each stage cost.** "Training-free" is rhetoric until the map fit, the
label transport and the policy refit are timed. :class:`Timings` does that.

*Wall-clock is correct here and this is not an exception to the project's rule.*
That rule --- no wall-clock in recorded data --- is about **task** data, where a
timestamp must come from the step index and the control frequency so that a
trace means the same thing on a fast machine and a slow one. This is a
measurement **of software**, and how long the software takes is the quantity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

#: The grid ``phi`` is fingerprinted on: corner, corner, and points per axis.
#:
#: Fixed, and written into every manifest, because a fingerprint is only
#: comparable against another one evaluated at the same points. It spans the
#: workspace the task occupies -- the table in front of the robot, the shelf
#: behind it, and from the table surface to above the top board -- rather than
#: the trajectory, so two cells whose paths differ are still compared on the
#: same domain.
#:
#: 20 points an axis is 8000 points, which at three float32 components is 96 kB
#: a cell and about 20 MB for a 210-cell campaign. A finer grid buys resolution
#: the map does not have: the nonlinear stage is a GP whose length scale is
#: several centimetres, and the grid pitch here is already under 4 cm.
FINGERPRINT_LOWER = (-0.35, -0.35, 0.80)
FINGERPRINT_UPPER = (0.40, 0.35, 1.35)
FINGERPRINT_POINTS = 20


def fingerprint_grid() -> np.ndarray:
    """The ``(N, 3)`` evaluation points, in a fixed order."""
    axes = [np.linspace(lo, hi, FINGERPRINT_POINTS)
            for lo, hi in zip(FINGERPRINT_LOWER, FINGERPRINT_UPPER)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def fingerprint_spec() -> dict:
    """The grid's definition, for the manifest.

    Written once per campaign rather than per cell. A fingerprint without the
    grid it was taken on is uninterpretable, and a grid repeated 210 times is
    210 chances for one of them to differ.
    """
    return {
        "lower": list(FINGERPRINT_LOWER),
        "upper": list(FINGERPRINT_UPPER),
        "points_per_axis": FINGERPRINT_POINTS,
        "order": "numpy meshgrid, indexing='ij', flattened C-order",
        "stored": "displacement phi(x) - x, float32, shape (N, 3)",
    }


def fingerprint(transport_map) -> np.ndarray:
    """``phi(x) - x`` on :func:`fingerprint_grid`, as float32.

    The displacement, not the image. The grid is common to every cell, so
    storing ``phi(x)`` would store it again in each one; and the difference
    between two maps -- which is the quantity the analysis wants -- is the
    difference of their displacements either way.
    """
    grid = fingerprint_grid()
    return (transport_map.transport_positions(grid) - grid).astype(np.float32)


def pose_matrix(position, rotation) -> list:
    """A ``4 x 4`` homogeneous pose, as nested lists for JSON.

    Stored as a matrix rather than as a position and a quaternion because a
    quaternion has a sign ambiguity and this project has already lost twenty
    cells to an undetected half turn (``ROBOTICS_NOTES`` 7.33). A matrix has
    none.
    """
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    out[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return out.round(9).tolist()


def condition_id(pick_config: str, destination: str) -> str:
    """The condition block's identifier for one cell.

    One string, so a row can be grouped on it without reconstructing it from
    two columns and hoping the reconstruction matches everyone else's.
    """
    return f"{pick_config}+{destination}"


def condition_block(pick_configs, destinations, picks: dict) -> dict:
    """The block's full definition, for the manifest.

    Written once, and it is what makes the design a *randomised block design*
    rather than a nominal one: the same six conditions are faced by every
    (gripper, object) cell, so the comparison is paired and McNemar and Wilcoxon
    apply rather than their weaker unpaired counterparts.

    The pick poses are included by value, not by name. A later reader must be
    able to see that ``P1`` was ``(-0.21, 0.02, 55 degrees)`` without having to
    find the commit the campaign ran at.
    """
    return {
        "design": "randomised block: (gripper, object) is the treatment, the "
                  "fixed condition set is the block",
        "pick_configs": {k: {"x": v[0], "y": v[1], "yaw_deg": v[2]}
                         for k, v in picks.items()},
        "destinations": list(destinations),
        "conditions": [condition_id(p, d)
                       for p in pick_configs for d in destinations],
        "note": "pick pose and destination are sampled once and the identical "
                "sample is used for every cell; a pick pose drawn per cell "
                "would confound the gripper with the difficulty of the pose it "
                "happened to draw, and no analysis afterwards separates them",
    }


@dataclass
class Timings:
    """Wall-clock cost of each stage of the method, in seconds.

    Used as a context manager per stage::

        timings = Timings()
        with timings.stage("map_fit"):
            transport_map = TransportMap().fit(S, T)
    """

    stages: dict = field(default_factory=dict)

    def stage(self, name: str):
        return _Stage(self, name)

    def as_dict(self) -> dict:
        return {f"seconds_{k}": round(v, 6) for k, v in self.stages.items()}


class _Stage:
    def __init__(self, owner: Timings, name: str):
        self.owner, self.name = owner, name

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.owner.stages[self.name] = time.perf_counter() - self.started
        return False
