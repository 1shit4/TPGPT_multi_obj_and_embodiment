"""A disk cache for grasp candidates, so an experiment can be repeated.

GraspGen-X's default planner is a **diffusion** model sampled from noise, and
neither its client nor its server exposes a seed. Two runs of the same scene
therefore get different candidate sets, and the difference is not small: the
same cereal box on the same seed was picked up on one run and missed on the
next, purely because a different grasp was drawn.

That makes an unrepeated success rate close to meaningless, and it makes a
comparison between two settings -- keypoint families, say -- a comparison of two
random draws as much as of the two settings. Caching by the *cloud the grasps
were asked for* fixes both: the same scene asks the same question and gets the
same answer back, so a difference between two runs is a difference between the
things that actually changed.

It is also 4-12 seconds a run, which is most of the wall clock in a campaign.

The key is the gripper plus the cloud itself, quantised to 0.1 mm before
hashing. The quantum is a normalisation, not a tolerance: rounding has bin
boundaries, so two clouds differing by a nanometre either side of one still
hash differently. What it buys is that a cloud is keyed by a value a human can
reason about rather than by its last floating-point bits. Reproducibility rests
on the render being deterministic on one machine, which it is; a miss only ever
costs a fresh inference.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

#: Where cached candidate sets live. Overridable so a campaign can keep its own.
DEFAULT_CACHE_DIR = Path(os.environ.get("TPGPT_GRASP_CACHE", "outputs/grasp_cache"))

#: Cloud quantisation for the key, metres. See the module docstring: this
#: normalises the value that gets hashed, it does not make the key tolerant.
QUANTUM = 1e-4


def cache_key(points: np.ndarray, gripper: str, **extra) -> str:
    """A stable identifier for "these grasps, for this cloud, for this hand"."""
    quantised = np.round(np.asarray(points, dtype=float) / QUANTUM).astype(np.int64)
    digest = hashlib.sha1()
    digest.update(gripper.encode())
    for name in sorted(extra):
        digest.update(f"{name}={extra[name]}".encode())
    digest.update(quantised.tobytes())
    digest.update(str(quantised.shape).encode())
    return digest.hexdigest()[:20]


def load(key: str, cache_dir=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Cached ``(poses, scores)``, or ``None`` if this cloud is new."""
    path = Path(cache_dir or DEFAULT_CACHE_DIR) / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return data["poses"], data["scores"]
    except Exception:  # pragma: no cover - a truncated write is not fatal
        return None


def store(key: str, poses: np.ndarray, scores: np.ndarray, cache_dir=None) -> Path:
    """Write a candidate set, atomically, so a killed run leaves no half file."""
    directory = Path(cache_dir or DEFAULT_CACHE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    # Written through a handle, not a name: ``savez_compressed`` appends ``.npz``
    # to any path that does not already end in it, so passing "<key>.npz.tmp"
    # silently produces "<key>.npz.tmp.npz" and the rename below finds nothing.
    temporary = directory / f"{key}.tmp.npz"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle, poses=np.asarray(poses), scores=np.asarray(scores)
        )
    temporary.replace(path)
    return path
