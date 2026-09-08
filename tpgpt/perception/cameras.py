"""Object point clouds from simulated depth cameras.

GraspGenX consumes an ``(N, 3)`` metric point cloud of a single object and
returns grasps *in the frame of the cloud it was given*. Feeding it a
**world-frame** cloud therefore yields world-frame grasps, directly commandable
by :class:`~tpgpt.sim.controllers.cartesian_impedance.CartesianImpedanceController`
with no further transform.

The cloud is built from robosuite's own depth and instance-segmentation
cameras rather than from the object mesh, so it is a genuine partial view with
the occlusion structure a real depth sensor would produce.

Three conventions here are easy to get wrong and were pinned by measurement
against a box of known size; see :func:`unproject` and
:data:`DEFAULT_MASK_EROSION`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from robosuite.utils import camera_utils
from scipy import ndimage

#: Cameras used when none are named. Three viewpoints roughly halve the
#: unobserved back side of an object compared with a single view.
#:
#: Not ``agentview`` or ``frontview``, though both were used until the shelf
#: became solid geometry. Both sit on the far side of the shelf from the table,
#: so they now look straight into its back panel, and object clouds went from
#: 1433 points to zero. The shelf had always been there -- it lived in the
#: collision group, which robosuite does not render, so depth passed through it
#: and the occlusion was invisible. ``workspace`` is placed for this scene
#: specifically; the other two are stock cameras with a clear line to the table.
#:
#: Cameras absent from an environment are skipped, so this stays valid for the
#: reshelving scene, which has none of them.
DEFAULT_CAMERAS = ("workspace", "sideview", "birdview")

#: Erosion applied to the instance mask, in pixels.
#:
#: A segmentation mask includes the boundary pixels of the object, whose depth
#: samples land partly on whatever is *behind* it. Those few pixels project to
#: points metres away and dominate the cloud's bounding box. Measured on a
#: 5 x 5 x 9 cm box: with no erosion the fused extents came out
#: [17.3, 8.8, 9.7] cm; with one pixel of erosion, [5.8, 6.2, 8.9] cm. Two
#: pixels tightens it further but empties the mask for small or distant objects
#: (one view dropped to 5 points), so one is the default.
DEFAULT_MASK_EROSION = 1

#: Hard cap on points sent to GraspGenX.
#:
#: The server's outlier removal is ``torch.cdist(X, X)`` plus an ``N x N``
#: identity mask, so it is O(N^2) in memory. A 40k-point cloud asks for roughly
#: 6 GB and the kernel kills the process.
MAX_CLOUD_POINTS = 8192


@dataclass
class ObjectCloud:
    """A segmented, metric, world-frame point cloud of one object."""

    points: np.ndarray                      # (N, 3) world frame, metres
    instance: str                           # robosuite instance name
    cameras: tuple[str, ...] = ()
    #: World position of each contributing camera, kept for the later filtering
    #: discussion: rejecting grasps that approach from an unobserved direction
    #: needs to know where the object was seen from.
    camera_positions: dict = field(default_factory=dict)
    points_per_camera: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.points.shape[0]

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def extent(self) -> np.ndarray:
        """Axis-aligned bounding-box extents, metres."""
        return self.points.max(axis=0) - self.points.min(axis=0)

    def contains(self, point: np.ndarray, tol: float = 0.0) -> bool:
        """True if ``point`` lies inside the cloud's bounding box."""
        point = np.asarray(point, dtype=float)
        return bool(
            np.all(self.points.min(axis=0) - tol <= point)
            and np.all(point <= self.points.max(axis=0) + tol)
        )

    def describe(self) -> str:
        return (
            f"ObjectCloud({self.instance}, {len(self)} pts, "
            f"extent {np.round(self.extent * 100, 1)} cm)"
        )


def instance_names(env) -> list[str]:
    """Instance names in the order robosuite encodes them into segmentation."""
    return list(env.model.instances_to_ids.keys())


def instance_segmentation_id(env, instance: str) -> int:
    """Segmentation value for ``instance``.

    robosuite maps each geom to its instance's *index*, then adds one, so ``0``
    means background or an unmapped geom. Forgetting the ``+ 1`` silently
    selects the background instead of the first object, which looks like a
    catastrophic calibration error rather than an off-by-one.
    """
    names = instance_names(env)
    if instance not in names:
        raise KeyError(f"unknown instance {instance!r}; scene has {names}")
    return names.index(instance) + 1


def unproject(
    env,
    camera: str,
    mask: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Lift masked pixels into world-frame points.

    Both conventions below were established by measurement against an object of
    known pose and size, not read off the documentation:

    * **The images are flipped.** robosuite returns observation images bottom-up
      (``seg[::convention]``, ``depth[::convention]``) while the camera matrix
      assumes a top-down raster. Both the mask and the depth must be flipped
      back together. Skipping this put the cloud 28 cm from the object.
    * **Column before row.** The homogeneous pixel vector is
      ``[col * z, row * z, z, 1]``. This matches what robosuite's own
      :func:`~robosuite.utils.camera_utils.transform_from_pixels_to_world` does
      internally, where it swaps the two components of its ``(row, col)`` input.
      Swapping them here instead moved the cloud 26 cm the other way.

    Args:
        env: A robosuite environment.
        camera: Camera name.
        mask: ``(H, W)`` boolean mask, in observation (flipped) orientation.
        depth: ``(H, W)`` metric depth, in observation orientation.
        height, width: Camera resolution.

    Returns:
        ``(M, 3)`` world-frame points.
    """
    mask = np.asarray(mask)[::-1]
    depth = np.asarray(depth, dtype=float)[::-1]
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 3))

    z = depth[rows, cols]
    pixels_to_world = np.linalg.inv(
        camera_utils.get_camera_transform_matrix(env.sim, camera, height, width)
    )
    homogeneous = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
    return (pixels_to_world @ homogeneous.T).T[:, :3]


def object_point_cloud(
    env,
    instance: str,
    cameras: tuple[str, ...] | None = None,
    obs: dict | None = None,
    erosion: int = DEFAULT_MASK_EROSION,
    max_points: int = MAX_CLOUD_POINTS,
    rng: np.random.Generator | None = None,
) -> ObjectCloud:
    """Fuse the views of one object into a single world-frame cloud.

    Args:
        env: Environment built with ``camera_depths`` and
            ``camera_segmentations="instance"``.
        instance: robosuite instance name, e.g. ``"product"``.
        cameras: Which cameras to fuse. Defaults to every camera the
            environment renders, intersected with :data:`DEFAULT_CAMERAS`.
        obs: An observation dict to reuse; otherwise one is requested.
        erosion: Mask erosion in pixels; see :data:`DEFAULT_MASK_EROSION`.
        max_points: Downsample above this; see :data:`MAX_CLOUD_POINTS`.
        rng: Generator used for downsampling, so clouds are reproducible.

    Returns:
        An :class:`ObjectCloud`. Empty if the object is not visible.
    """
    obs = obs if obs is not None else env._get_observations()
    available = tuple(env.camera_names)
    if cameras is None:
        cameras = tuple(c for c in DEFAULT_CAMERAS if c in available) or available
    missing = [c for c in cameras if c not in available]
    if missing:
        raise ValueError(f"environment does not render {missing}; it has {available}")

    seg_id = instance_segmentation_id(env, instance)
    structure = np.ones((3, 3), dtype=bool)

    chunks, per_camera, positions = [], {}, {}
    for camera in cameras:
        seg_key, depth_key = f"{camera}_segmentation_instance", f"{camera}_depth"
        if seg_key not in obs or depth_key not in obs:
            raise KeyError(
                f"{camera} is missing depth or segmentation; build the environment "
                'with camera_depths=True and camera_segmentations="instance"'
            )
        height, width = obs[seg_key].shape[:2]
        mask = obs[seg_key][..., 0] == seg_id
        if erosion > 0 and mask.any():
            mask = ndimage.binary_erosion(mask, structure, iterations=erosion)

        depth = camera_utils.get_real_depth_map(env.sim, obs[depth_key])[..., 0]
        points = unproject(env, camera, mask, depth, height, width)
        per_camera[camera] = len(points)
        positions[camera] = camera_utils.get_camera_extrinsic_matrix(env.sim, camera)[
            :3, 3
        ]
        if len(points):
            chunks.append(points)

    points = np.vstack(chunks) if chunks else np.zeros((0, 3))
    if len(points) > max_points:
        rng = rng if rng is not None else np.random.default_rng(0)
        points = points[rng.choice(len(points), max_points, replace=False)]

    return ObjectCloud(
        points=points.astype(np.float32),
        instance=instance,
        cameras=tuple(cameras),
        camera_positions=positions,
        points_per_camera=per_camera,
    )


#: Instances never included in a scene cloud: the robot itself, and the
#: visual-only slot markers, which have ``contype=0`` and cannot collide with
#: anything. Leaving the markers in would make every shelf slot look blocked.
NON_OBSTACLE_PREFIXES = ("Panda", "Sawyer", "UR5e", "Kinova", "IIWA", "XArm", "Yumi")
NON_OBSTACLE_SUBSTRINGS = ("Mount", "Gripper", "Hand", "slot_")


def is_obstacle(instance: str) -> bool:
    """Whether an instance is something the hand could collide with."""
    return not (
        instance.startswith(NON_OBSTACLE_PREFIXES)
        or any(s in instance for s in NON_OBSTACLE_SUBSTRINGS)
    )


def scene_point_cloud(
    env,
    exclude: tuple[str, ...] = (),
    cameras: tuple[str, ...] | None = None,
    obs: dict | None = None,
    max_points: int = MAX_CLOUD_POINTS,
    max_depth: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Everything the gripper must avoid, in world coordinates.

    Built from the **full** depth image rather than from segmented instances,
    because the table is arena geometry and carries no instance id at all: a
    segmentation-driven cloud would leave out the one surface every top-down
    grasp has to clear.

    Args:
        exclude: Instance names to drop -- the object being grasped, above all.
            The hand closes around it, so leaving it in makes every grasp
            collide with the thing it is meant to pick up.

    Returns:
        ``(N, 3)`` world-frame points, capped at ``max_points``.
    """
    from robosuite.utils import camera_utils

    rng = rng if rng is not None else np.random.default_rng(0)
    obs = obs if obs is not None else env._get_observations()
    available = {k[: -len("_image")] for k in obs if k.endswith("_image")} or set()
    cameras = cameras or tuple(c for c in DEFAULT_CAMERAS if c in available) or DEFAULT_CAMERAS

    drop_ids = {
        instance_segmentation_id(env, name)
        for name in instance_names(env)
        if name in exclude or not is_obstacle(name)
    }

    clouds = []
    for camera in cameras:
        depth = obs.get(f"{camera}_depth")
        segmentation = obs.get(f"{camera}_segmentation_instance")
        if depth is None or segmentation is None:
            continue
        metric = camera_utils.get_real_depth_map(env.sim, depth)[..., 0]
        ids = np.asarray(segmentation)[..., 0]
        mask = (~np.isin(ids, list(drop_ids))) & (metric > 0.0) & (metric < max_depth)
        if not mask.any():
            continue
        clouds.append(unproject(env, camera, mask, metric, *mask.shape))

    if not clouds:
        return np.empty((0, 3))
    points = np.vstack(clouds)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points
