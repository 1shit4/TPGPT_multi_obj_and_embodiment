"""A fixed library of grasps per object, so a plan does not depend on a random draw.

**Why.** GraspGen-X's diffusion planner is sampled from noise and neither its
client nor its server exposes a seed (``tpgpt/grasp/cache.py``). Every call
returns a different set, so on a constrained task the same scene plans on one
call and fails on the next: measured live on a banana, 100 candidates gave 16,
0 and 0 admissible; 400 gave 5, 38 and 38. The outcome was the draw, not the
task.

The simulation solved this by caching the candidates for a cloud. On the robot
it can be done better, because since ``object_model`` the planner works on the
object's **YCB scan** rather than on the camera's cloud: the shape is fixed and
known, so the grasps can be asked for **once, in the object's own frame**, kept
on disk, and carried into the scene by the rigid transform the fit already
provides. What GraspGen-X is asked -- the object's surface -- is then identical
every time, and the answer is too.

That buys three things:

* **the same grasps every run**, so a difference between two runs is a
  difference in the task, not in the sampler;
* **as many candidates as one likes.** 1000 grasps take 5.2 s to generate once
  and nothing at plan time, against 200 drawn fresh in 2 s on every plan;
* **a feasible set that can be measured.** "Of this object's 2000 grasps, N
  survive the table, the jaws and the transported path into that slot" is a
  property of the object and the destination, computed offline. It is measured
  on the generator's own grasps, so it carries the generator's notion of what
  holds; a geometric sweep of hand poses would not.

The library is **not** used when the scan fit is refused: the planner is then
working from the camera's own cloud, whose shape is not the scan's, so the
grasps have to be asked for live as before.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: Where the libraries live. One file per (object, gripper, size).
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "TPGPT_GRASP_LIBRARY",
    str(Path(__file__).resolve().parents[2] / "outputs" / "hardware" / "grasp_library")))

#: Grasps kept per object. The generator is asked for twice this and returns its
#: best half, as the planner does live. 2000 takes about 10 s, once per object.
DEFAULT_SIZE = 2000

#: Seed of the surface sample the generator is asked about. Fixed, because the
#: cache key is that sample: a different seed is a different question.
SAMPLE_SEED = 1


def canonical_cloud(object_name: str, points: int | None = None, seed: int = SAMPLE_SEED):
    """The object's scan, sampled in its own frame -- the same points every time."""
    from tpgpt.hardware.object_model import FULL_CLOUD_POINTS, ObjectModel

    model = ObjectModel.load(object_name)
    cloud, _ = model.sample(FULL_CLOUD_POINTS if points is None else int(points), seed)
    return cloud


def library(object_name: str, gripper: str = "franka_panda", size: int = DEFAULT_SIZE,
            host: str = "127.0.0.1", port: int = 5556, planner: str = "diffusion",
            timeout_ms: int = 300000, cache_dir=None, rebuild: bool = False):
    """``(poses, scores)`` for ``object_name`` in its own frame, from disk or the server.

    ``poses`` are ``(n, 4, 4)`` in the scan's frame; put them in the scene with
    :func:`placed`. Asking the server happens once per (object, gripper, size).
    """
    from tpgpt.grasp.cache import cache_key, load, store

    cloud = canonical_cloud(object_name)
    key = cache_key(cloud, gripper, size=int(size), planner=planner)
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    if not rebuild:
        hit = load(key, cache_dir)
        if hit is not None:
            return hit
    from tpgpt.hardware.grasp_selection import request_candidates

    poses, scores = request_candidates(cloud, host=host, port=port, gripper=gripper,
                                       num_grasps=2 * int(size), topk=int(size),
                                       planner=planner, timeout_ms=timeout_ms)
    store(key, poses, scores, cache_dir)
    return np.asarray(poses, dtype=float), np.asarray(scores, dtype=float)


def placed(poses: np.ndarray, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Library poses carried into the scene by the object's fitted pose.

    The library is expressed in the scan's own frame and the fit is a rigid
    transform of that frame, so this is exact -- the grasps are the same grasps,
    on the same surface, seen from the robot's frame.
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T @ np.asarray(poses, dtype=float).reshape(-1, 4, 4)


__all__ = ["DEFAULT_CACHE_DIR", "DEFAULT_SIZE", "canonical_cloud", "library", "placed"]
