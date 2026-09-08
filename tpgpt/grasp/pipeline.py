"""From a scene and a named object to ranked grasps for one or more grippers.

This is the join between :mod:`tpgpt.perception` and :mod:`tpgpt.grasp`: segment
the object, lift it into a world-frame cloud, and ask GraspGen-X for grasps for
each hand. The cloud is world-frame, so the poses come back in world frame and
need no further transform.

**No filtering or selection happens here.** Candidates are returned in full,
sorted by the discriminator score. Deciding which one to execute -- reachability,
collision, approach direction -- is a separate question and deliberately not
answered yet.
"""

from __future__ import annotations

import numpy as np

from tpgpt.grasp.cache import cache_key, load, store
from tpgpt.grasp.client import GraspGenClient
from tpgpt.grasp.grasps import GraspSet, build_grasp_set
from tpgpt.grasp.grippers import resolve_pair
from tpgpt.perception.cameras import ObjectCloud, object_point_cloud


def grasps_for_cloud(
    cloud: ObjectCloud,
    gripper: str,
    client: GraspGenClient | None = None,
    num_grasps: int = 200,
    topk: int = 100,
    planner: str = "diffusion",
    cache_dir=None,
    use_cache: bool = True,
) -> GraspSet:
    """Ask the server for grasps on an already-extracted cloud.

    Args:
        cloud: A world-frame :class:`~tpgpt.perception.cameras.ObjectCloud`.
        gripper: Short name, robosuite name, or GraspGen-X name.
        client: Reuse an open client; one is created and closed if omitted.
        num_grasps: Samples drawn before ranking.
        topk: Cap on returned candidates.
        planner: ``"diffusion"`` or ``"graspmoe"``.
        cache_dir: Where to keep cached candidate sets; see
            :mod:`tpgpt.grasp.cache`. Defaults to ``outputs/grasp_cache``.
        use_cache: Set ``False`` to force a fresh draw. The default planner is a
            diffusion model with no seed, so without the cache the same scene
            gets a different candidate set every time and no experiment can be
            repeated.
    """
    pair = resolve_pair(gripper)
    if len(cloud) == 0:
        return GraspSet(
            grasps=[], gripper=pair.graspgen, instance=cloud.instance,
            metadata={"reason": "object not visible in any camera"},
        )

    key = cache_key(
        cloud.points, pair.graspgen,
        planner=planner, num_grasps=num_grasps, topk=topk,
    )
    cached = load(key, cache_dir) if use_cache else None
    if cached is not None:
        poses, scores = cached
    else:
        owned = client is None
        client = client or GraspGenClient()
        try:
            poses, scores = client.infer(
                cloud.points,
                gripper_name=pair.graspgen,
                num_grasps=num_grasps,
                topk_num_grasps=topk,
                planner=planner,
            )
        finally:
            if owned:
                client.close()
        if use_cache:
            store(key, poses, scores, cache_dir)

    return build_grasp_set(
        poses, scores, pair.graspgen, cloud.instance, len(cloud),
        metadata={
            "planner": planner,
            "num_grasps": num_grasps,
            "cameras": list(cloud.cameras),
            "cloud_extent": np.round(cloud.extent, 4).tolist(),
            "grasp_cache_key": key,
            "from_cache": cached is not None,
        },
    )


def grasps_for_object(
    env,
    instance: str,
    grippers,
    obs: dict | None = None,
    cameras: tuple[str, ...] | None = None,
    client: GraspGenClient | None = None,
    **infer_kwargs,
) -> tuple[ObjectCloud, dict[str, GraspSet]]:
    """Segment one object and generate grasps for several grippers.

    The cloud is extracted once and reused, so comparing hands is a comparison
    of the hands and not of two different views of the object.

    Returns:
        ``(cloud, {short_name: GraspSet})``.
    """
    cloud = object_point_cloud(env, instance, cameras=cameras, obs=obs)
    owned = client is None
    client = client or GraspGenClient()
    try:
        return cloud, {
            name: grasps_for_cloud(cloud, name, client=client, **infer_kwargs)
            for name in grippers
        }
    finally:
        if owned:
            client.close()
