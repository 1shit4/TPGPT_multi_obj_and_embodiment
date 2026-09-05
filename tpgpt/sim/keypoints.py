"""Keypoint task parameterisation (paper Sec. III-A, Sec. V-A).

The task is described by an ordered set of 3-D points rather than by object
poses. Sec. V-A is specific about what that takes: "To describe the 3D pose of
an object, we need at least 2 non-parallel vectors; hence, we need at least 3
non-overlapping and non-collinear points", and "for every tag, the
transportation policy extracts a cube's center and corners with predefined side
dimensions as the markers".

Two things the prototype got wrong and this module fixes:

* Keypoints were derived from an object's **position only**, so a rotated object
  produced an identical keypoint set and its rotation was invisible to the map.
  Here they are derived from the full pose.
* Source and target sets were built by applying a hard-coded offset rather than
  read from the scene, so the target was always a pure translation -- which the
  affine stage solves exactly, leaving the nonlinear stage untested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.utils.rotations import quat_to_matrix

#: Unit-cube corner offsets, ordered so that source and target sets pair up.
CUBE_CORNERS = np.array(
    [
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ],
    dtype=float,
)

CORNER_NAMES = ("nnn", "pnn", "ppn", "npn", "nnp", "pnp", "ppp", "npp")


@dataclass
class KeypointSet:
    """An ordered set of task keypoints.

    Order is meaningful: Sec. III-A assumes the source and target sets are
    already paired elementwise, so the labels carry that pairing explicitly.
    """

    points: np.ndarray
    labels: list[str]
    frame: str = "world"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.atleast_2d(np.asarray(self.points, dtype=float))
        if len(self.labels) != self.points.shape[0]:
            raise ValueError(
                f"{len(self.labels)} labels for {self.points.shape[0]} points"
            )

    def __len__(self) -> int:
        return self.points.shape[0]

    def reordered_like(self, other: "KeypointSet") -> "KeypointSet":
        """Return a copy reordered to match ``other``'s labels.

        Guards against the source and target being extracted in different
        orders, which would silently train the map on the wrong correspondences.
        """
        if set(self.labels) != set(other.labels):
            missing = set(other.labels) ^ set(self.labels)
            raise ValueError(f"keypoint label sets differ: {sorted(missing)}")
        index = {label: i for i, label in enumerate(self.labels)}
        order = [index[label] for label in other.labels]
        return KeypointSet(
            points=self.points[order],
            labels=list(other.labels),
            frame=self.frame,
            metadata=dict(self.metadata),
        )

    def is_degenerate(self, tol: float = 1e-9) -> bool:
        """True if the points span fewer than three dimensions.

        A degenerate set leaves the affine rotation undetermined
        (Sec. III-E-a), so it is worth detecting before fitting.
        """
        centred = self.points - self.points.mean(axis=0)
        if centred.shape[0] < 3:
            return True
        singular = np.linalg.svd(centred, compute_uv=False)
        return bool(singular[-1] < tol * max(singular[0], 1e-12))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "frame": self.frame,
                    "labels": self.labels,
                    "points": self.points.tolist(),
                    "metadata": self.metadata,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "KeypointSet":
        data = json.loads(Path(path).read_text())
        return cls(
            points=np.array(data["points"]),
            labels=list(data["labels"]),
            frame=data.get("frame", "world"),
            metadata=data.get("metadata", {}),
        )


def cube_keypoints(
    position: np.ndarray,
    rotation: np.ndarray | None = None,
    half_extent: float | np.ndarray = 0.02,
    include_center: bool = True,
    name: str = "obj",
) -> KeypointSet:
    """Corner (and centre) keypoints of a box attached to a pose.

    Mirrors the fiducial-marker scheme of Sec. V-A: each detected tag yields a
    cube of predefined side dimensions whose centre and corners become the
    keypoints.

    Args:
        position: ``(3,)`` body position.
        rotation: ``(3, 3)`` body rotation. ``None`` means axis aligned. Passing
            the real rotation is what makes object orientation visible to the
            transportation map.
        half_extent: Scalar or ``(3,)`` half sizes of the box.
        include_center: Also emit the centre point.
        name: Prefix for the keypoint labels.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float).reshape(3, 3)
    half = np.broadcast_to(np.asarray(half_extent, dtype=float), (3,))

    local = CUBE_CORNERS * half
    points = position + local @ R.T
    labels = [f"{name}_{n}" for n in CORNER_NAMES]

    if include_center:
        points = np.vstack([position[None], points])
        labels = [f"{name}_center", *labels]
    return KeypointSet(points=points, labels=labels, metadata={"object": name})


def keypoints_from_bodies(
    env,
    bodies: dict[str, float | np.ndarray],
    include_center: bool = True,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> KeypointSet:
    """Extract a keypoint set from named MuJoCo bodies in a live environment.

    Args:
        env: A robosuite environment.
        bodies: Maps body name to the half extent of its keypoint cube.
        include_center: Emit the centre point alongside the corners.
        noise_std: Standard deviation of isotropic Gaussian detection noise, in
            metres. Real keypoints come from AprilTags or stereo depth, both of
            which are millimetre-scale at best; this makes that cost visible.
        rng: Random generator used for the noise.

    Returns:
        A single :class:`KeypointSet` concatenating every body, in the order the
        ``bodies`` mapping is given.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sets = []
    for body_name, half_extent in bodies.items():
        body_id = env.sim.model.body_name2id(body_name)
        position = np.array(env.sim.data.body_xpos[body_id])
        # MuJoCo stores body quaternions scalar-first.
        rotation = quat_to_matrix(
            np.array(env.sim.data.body_xquat[body_id]), scalar_first=True
        )[0]
        sets.append(
            cube_keypoints(
                position, rotation, half_extent, include_center, name=body_name
            )
        )

    points = np.vstack([s.points for s in sets])
    labels = [label for s in sets for label in s.labels]
    if noise_std > 0:
        points = points + rng.normal(0.0, noise_std, points.shape)
    return KeypointSet(
        points=points,
        labels=labels,
        frame="world",
        metadata={"bodies": list(bodies), "noise_std": noise_std},
    )


def pair_keypoints(source: KeypointSet, target: KeypointSet) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S, T)`` arrays with matching label order.

    Sec. III-A assumes source and target are already paired; this enforces it
    rather than trusting extraction order.
    """
    return source.points, target.reordered_like(source).points
