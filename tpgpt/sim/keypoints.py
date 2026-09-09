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


# ---------------------------------------------------------------------------
# Grasp-aligned keypoints for arbitrary objects (paper Sec. III-A, Sec. V-A)
# ---------------------------------------------------------------------------
#
# ``cube_keypoints`` above attaches its box to an object's *body frame* and
# needs its half extent given as a constant. That is exactly right for the
# reshelving product, which is a box of known size, and exactly wrong for a
# loaf or a lemon: an arbitrary mesh has no known extent and no task-meaningful
# body frame.
#
# Sec. III-A requires the source and target keypoint sets to be **paired**. The
# construction below produces that pairing by *recipe* rather than by matching:
# every keypoint is defined by a role -- "the corner on the negative closing
# side of the support face", "the contact on the positive closing side" --
# and those roles exist on any object held by any gripper. Two objects of
# entirely different shape therefore yield keypoint sets that correspond
# elementwise by construction.

#: Extra keypoint roles emitted alongside the eight box corners and the centre.
#:
#: Four points span the grasp plane, not two. Two contacts define a *line*, and
#: a line pins one direction: it fixes which way the jaws close and says nothing
#: about the direction the hand comes in from, because infinitely many approach
#: directions are perpendicular to a given line. Measured with only the two, the
#: warped approach sat 7.5 to 15.7 degrees off the chosen grasp's and the jaw
#: axis up to 33.4 degrees off.
#:
#: The second pair lies along ``approach x closing``, the remaining in-plane
#: direction. With all four the plane itself is pinned, and the plane's normal
#: **is** the approach vector.
CONTACT_NAMES = ("grasp_neg", "grasp_pos", "grasp_side_neg", "grasp_side_pos", "support")

#: Points closer than this to the support plane count as touching it.
SUPPORT_SLAB = 0.006

#: Half thickness of the slab of cloud taken as the grasp cross-section.
GRASP_SLAB = 0.012

#: Percentile used instead of the true extreme when locating a jaw contact, so
#: that a single stray point cannot define a keypoint.
CONTACT_PERCENTILE = 5.0

#: Half extent of the fixed cube used by the ``"grasp_cube"`` box mode, in metres.
#:
#: **What this number controls, and it is not obvious.** ``phi`` interpolates
#: keypoint *positions* exactly at any cube size, so the grasp point is carried
#: exactly whatever this is set to. What the size changes is the accuracy of the
#: transported *orientation*, because Eq. 11 uses ``J_perp`` -- a **derivative** --
#: and the cube's eight corners are the only thing pinning the local rotation.
#: Their distance from the centre is therefore the width of the stencil over
#: which that rotation gets estimated, and a wide stencil averages the grasp's
#: own rotation together with the far field, which is the affine stage.
#:
#: Measured on the synthetic scene of ``tests/unit/test_keypoint_extraction``,
#: source object 60 x 100 x 90 mm onto a 60 x 100 x 150 mm target, where the
#: rotation actually required at the grasp is 23.6 degrees:
#:
#: ==========  ==============  ==============  ===============  ==========
#: half        min ``det(J)``  aim at grasp    gripper error    tilt@grasp
#: ==========  ==============  ==============  ===============  ==========
#: 5 mm              0.839          0.0 mm            0.1 deg     0.0 deg
#: 10 mm             0.836          0.0 mm            0.3 deg     0.0 deg
#: **20 mm**         0.829          0.0 mm            1.3 deg     0.8 deg
#: 30 mm             0.815          0.0 mm            3.0 deg     2.4 deg
#: 60 mm             0.698          0.0 mm           13.1 deg     7.6 deg
#: 100 mm           -0.006 FOLD     0.0 mm           20.6 deg    14.1 deg
#: ==========  ==============  ==============  ===============  ==========
#:
#: ``tilt@grasp`` climbing toward the affine stage's own 19.5 degrees is the
#: dilution happening: at 60 mm, ``J_perp`` delivers only 15.0 of the 23.6
#: degrees required.
#:
#: **Scale, because "6 cm" sounds small and is not.** This is a *half* extent, so
#: 60 mm describes a 120 mm cube -- larger than every object in the scene, and a
#: third of the pick-to-place distance. For calibration: the five tabletop
#: objects have half extents of 20 to 75 mm (milk 25x25x75, can 25x25x40, cereal
#: 40x40x30, bread 30x30x30, lemon 37.5x37.5x20), the nine registered grippers
#: have jaw apertures of 50 to 125 mm, and pick to place is about 355 mm.
#:
#: 20 mm is chosen to sit at or below every object's own half extent and inside
#: the flat region rather than at its edge. It also keeps the jaw contacts
#: affordable: with contacts included and the target's approach tilted, a 10-20 mm
#: cube holds ``min det(J)`` at 0.91 to 1.00 through 60 degrees of tilt, while a
#: 60 mm cube collapses to 0.081 -- because a large cube reaches far enough to
#: fight the contact plane, whose normal is the grasp's approach and not the
#: support normal.
GRASP_CUBE_HALF_EXTENT = 0.02

#: Keypoints closer together than this are refused rather than fitted.
#:
#: Two coincident keypoints make the map's Gram matrix singular, and the error
#: that surfaces is ``LinAlgError("Gram matrix is not positive definite ...")``
#: from deep inside the GP -- a symptom, not a cause. Coincidence became
#: *reachable* with the grasp-centred cube, whose centre and support contact are
#: both anchored to the grasp rather than to the cloud, so it is now checked.
MIN_KEYPOINT_SEPARATION = 1e-6

_EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return v / n


@dataclass
class GraspFrame:
    """The three quantities a keypoint recipe needs from a grasp.

    Deliberately not :class:`~tpgpt.grasp.grasps.Grasp6D`. That class resolves
    its TCP depth through the gripper registry, which reads the sibling
    GraspGen-X checkout; keypoint extraction must stay usable offline and in
    unit tests. :meth:`from_grasp` bridges the two when a real candidate is in
    hand.

    Attributes:
        tcp: Fingertip centre in world coordinates -- where the jaws close, not
            the gripper base.
        approach: Unit vector the hand advances along.
        closing: Unit vector the jaws travel along.
    """

    tcp: np.ndarray
    approach: np.ndarray
    closing: np.ndarray

    def __post_init__(self):
        self.tcp = np.asarray(self.tcp, dtype=float).reshape(3)
        self.approach = _unit(self.approach)
        self.closing = _unit(self.closing)

    @classmethod
    def from_grasp(cls, grasp) -> "GraspFrame":
        """Build from a :class:`~tpgpt.grasp.grasps.Grasp6D`."""
        return cls(
            tcp=grasp.tcp_position(),
            approach=grasp.approach,
            closing=grasp.closing,
        )

    @property
    def rotation(self) -> np.ndarray:
        """The grasp's full orientation, columns ``(closing, jaw, approach)``.

        Matches the :class:`~tpgpt.grasp.grasps.Grasp6D` convention exactly:
        ``+X`` is the closing direction and ``+Z`` the approach axis, so the
        middle column is ``approach x closing``.

        **The closing axis is re-orthogonalised against the approach**, because
        :meth:`__post_init__` only unit-normalises the two independently and a
        real grasp does not arrive perfectly square: measured on a constructed
        case, ``approach . closing = -0.0497``. Building a frame from the raw
        pair would give a skewed matrix that is not a rotation, and nothing
        downstream would raise -- ``J_perp`` would simply carry the skew into
        every transported orientation, stiffness and damping. That is the
        silent-corruption failure mode of 7.13 and 7.17.
        """
        a = self.approach
        c = self.closing - float(self.closing @ a) * a
        if float(np.linalg.norm(c)) < 1e-6:
            raise ValueError(
                "the closing axis is parallel to the approach axis, so the "
                "grasp's orientation is undetermined"
            )
        c = _unit(c)
        return np.column_stack([c, np.cross(a, c), a])

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> "GraspFrame":
        """The same grasp after the object it holds is moved rigidly."""
        R = np.asarray(rotation, dtype=float).reshape(3, 3)
        t = np.asarray(translation, dtype=float).reshape(3)
        return GraspFrame(
            tcp=R @ self.tcp + t,
            approach=R @ self.approach,
            closing=R @ self.closing,
        )


def task_frame(
    support_normal: np.ndarray,
    closing_axis: np.ndarray,
) -> np.ndarray:
    """Orthonormal frame defined by the task, not by the object's mesh.

    The columns are ``(c, a, n)``:

    * ``n`` -- the support normal, the direction the object rests against.
    * ``c`` -- the grasp closing axis, projected perpendicular to ``n``.
    * ``a = n x c`` -- the remaining axis.

    Why this frame rather than the body frame or the cloud's principal axes:
    it is defined identically for a carton, a can and a lemon, and for every
    gripper, because every grasp has a closing direction and every resting
    object has a support normal. Corner *i* of a box built in this frame
    therefore means the same physical thing on two objects of different shape,
    which is what makes the source and target sets of Sec. III-A paired by
    construction. Principal axes would not do: they are shape dependent and
    their signs flip arbitrarily on near-symmetric objects, so a lemon's
    principal axes are noise.

    **The sign of ``c`` is taken from the grasp and never re-derived here.**
    A parallel jaw closing along ``+c`` and along ``-c`` grips the same object
    the same way, so it is tempting to treat the sign as free and pick one by
    convention. This function used to do exactly that, with a world-axis test
    (``c . y >= 0``, tie-broken on ``c . x``) and an optional ``reference``
    frame to make two ends of a task agree. Both are gone, and the reason is
    worth stating because the test looked harmless for a long time.

    A grasp is not an axis, it is a **pose**. ``Grasp6D`` carries a full
    rotation whose first column *is* the closing direction, with a definite
    sign fixed by the planner and by which finger of that hand is which. There
    is no ambiguity in the input, so there is nothing here to resolve -- and
    resolving it anyway threw away information and then invented a replacement
    from the object's yaw in the world, which is unrelated to the gripper.

    What that cost: the source demonstration places its object square with the
    shelf, so its placed closing axis lands on world ``x`` at ``[1, -1e-17, 0]``.
    ``c[1]`` fell inside the epsilon, the ``c[0] < 0`` tie-break decided
    instead, and it decided the opposite way to the pick, whose ``c[1]`` is an
    unambiguous ``-0.581``. The source's own two frames came out **180 degrees
    apart**, which reflects the carried object through the grasp point and lands
    it twice its lateral grasp offset away.

    The caller's obligation is therefore upstream: hand this a closing axis that
    came from a real gripper pose. :func:`~tpgpt.grasp.grasps.Grasp6D.closing`
    does; an object's body axis does not, and that is what the world-axis test
    was quietly compensating for.

    Raises:
        ValueError: if the closing axis is parallel to the support normal, which
            leaves the frame undetermined. This happens for a grasp whose jaws
            close vertically onto a horizontal surface; refusing is better than
            returning an arbitrary frame.
    """
    n = _unit(support_normal)
    c = np.asarray(closing_axis, dtype=float).reshape(3)
    c = c - float(c @ n) * n
    if float(np.linalg.norm(c)) < 1e-6:
        raise ValueError(
            "the closing axis is parallel to the support normal, so the task "
            "frame is undetermined; this grasp closes straight into the "
            "support surface"
        )
    c = _unit(c)
    a = np.cross(n, c)
    return np.column_stack([c, a, n])


def grasp_pose_frame(grasp: "GraspFrame") -> np.ndarray:
    """The grasp's own rotation, columns ``(closing, jaw, approach)``.

    A cube laid out in this frame carries the whole grasp orientation, including
    the approach tilt that :func:`task_frame` discards by projecting into the
    support plane. That is what makes Eq. 11 deliver the hand at the angle the
    grasp actually asks for.

    It is not free. ``J_perp`` is one rotation per point and Eq. 11 uses it for
    the gripper, so whatever rotation reaches the hand also reaches the vertical.
    Measured against a target grasp tilted 30 degrees out of plane: the task
    frame leaves the gripper 30 degrees wrong and the vertical untouched, this
    frame leaves the gripper exact and tilts the vertical by 30 degrees, and the
    demonstration's straight-up lift comes out 24.2 degrees off vertical. Both
    are legitimate; the trade is the point, and
    :func:`tpgpt.metrics.transport.tilt_profile` is how it is measured.

    **This used to re-derive the closing axis's sign** against
    :func:`task_frame`, and took a support normal and a reference frame in order
    to do it. Both are gone. A grasp is a pose, not an axis: the sign is already
    fixed by the planner and by which finger of the hand is which, so there is
    nothing to resolve. See :func:`task_frame` for what the resolution cost.

    Note this frame needs no support normal at all, which is the other reason to
    prefer it: a hand approaching straight down has no horizontal component to
    project, so any construction that leans on the support plane degenerates
    exactly where this one does not.
    """
    return grasp.rotation


def to_frame(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Express world points in ``frame``'s coordinates."""
    return np.atleast_2d(np.asarray(points, dtype=float)) @ frame


def from_frame(local: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Take frame coordinates back to world."""
    return np.atleast_2d(np.asarray(local, dtype=float)) @ frame.T


def fit_aligned_box(
    points: np.ndarray,
    frame: np.ndarray,
    support_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an oriented box to a cloud in the given frame.

    Args:
        points: ``(N, 3)`` world-frame cloud.
        frame: ``(3, 3)`` task frame from :func:`task_frame`.
        support_height: Height of the support plane along the frame's ``n``
            axis. When given, the box's lower face is snapped to it.

    The snap is not cosmetic. A measured cloud's lowest point sits above the
    true contact plane whenever the silhouette's lower boundary is missing:
    ``DEFAULT_MASK_EROSION`` trims a pixel off every segmentation boundary --
    including the line where the object meets the surface -- and a neighbouring
    object occludes the base before it occludes the top. Measured on the five
    tabletop objects, the fused cloud's minimum stood **0.5 to 24.5 mm** above
    the table, the worst case being the lemon, which is small enough that
    erosion removes most of it. Snapping puts the four lower corners exactly on
    the surface the object rests on, which is what lets ``phi`` pin table to
    shelf exactly (property (i) of Sec. III-D, ``T = phi(S)``).

    Note that dropping only the *bottom face* of a synthetic cloud does not
    reproduce this: the side faces still reach the contact line, so the measured
    minimum is already correct. Truncation, not invisibility of one face, is
    what the snap defends against.

    Returns:
        ``(centre, half_extents)`` with the centre in world coordinates and the
        half extents along ``(c, a, n)``.
    """
    local = to_frame(points, frame)
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    if support_height is not None:
        lo = lo.copy()
        lo[2] = float(support_height)
        hi[2] = max(hi[2], lo[2] + 1e-4)
    centre_local = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    centre = from_frame(centre_local[None], frame)[0]
    return centre, half


def grasp_contacts(
    points: np.ndarray,
    grasp: GraspFrame,
    frame: np.ndarray,
    centre: np.ndarray,
    half: np.ndarray,
    slab: float = GRASP_SLAB,
    percentile: float = CONTACT_PERCENTILE,
) -> tuple[np.ndarray, bool]:
    """Four points spanning the plane the jaws travel in.

    The grasp plane is the plane through the TCP whose normal is the approach
    axis. Its intersection with the object is the cross-section being squeezed,
    and sampling it in **two** independent in-plane directions is what pins the
    plane -- and therefore the approach, which is its normal.

    Sampling only along the closing axis gives a line. A line fixes which way
    the jaws close and leaves the approach free, so the transported hand can
    arrive at the right contact points from the wrong direction. Measured:
    approach 7.5 to 15.7 degrees off the chosen grasp, jaw axis up to 33.4.

    Working from the cloud rather than the gripper model keeps this
    gripper-agnostic: a two-finger and a three-finger hand closing across the
    same axis produce the same four keypoints.

    Returns:
        ``(points, from_cloud)`` with points ordered
        ``(-closing, +closing, -side, +side)``. ``from_cloud`` is False when the
        grasp plane caught too little cloud and the box was used instead --
        sparse clouds are real here, the can measured in ROBOTICS_NOTES 5.6 gave
        44 points.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    closing = frame[:, 0]
    # The remaining in-plane direction. Built from the approach so the pair
    # genuinely spans the grasp plane rather than the support plane.
    side = np.cross(grasp.approach, closing)
    norm = float(np.linalg.norm(side))
    side = side / norm if norm > 1e-9 else frame[:, 1]

    offset = (points - grasp.tcp) @ grasp.approach
    selected = points[np.abs(offset) <= slab]

    def spread(axis, cloud):
        """The object's extent along one in-plane axis, robustly."""
        projected = cloud @ axis
        centre_value = float(grasp.tcp @ axis)
        return (
            float(np.percentile(projected, percentile)) - centre_value,
            float(np.percentile(projected, 100.0 - percentile)) - centre_value,
        )

    if len(selected) >= 8:
        low_c, high_c = spread(closing, selected)
        low_s, high_s = spread(side, selected)
        emitted = np.array([
            grasp.tcp + closing * low_c,
            grasp.tcp + closing * high_c,
            grasp.tcp + side * low_s,
            grasp.tcp + side * high_s,
        ])
        return emitted, True

    # Fallback: the box's own half extents, in the same four directions.
    reach_c, reach_s = float(half[0]), float(half[1])
    emitted = np.array([
        grasp.tcp - closing * reach_c,
        grasp.tcp + closing * reach_c,
        grasp.tcp - side * reach_s,
        grasp.tcp + side * reach_s,
    ])
    return emitted, False


def support_contact(
    points: np.ndarray,
    frame: np.ndarray,
    support_height: float,
    centre: np.ndarray,
    slab: float = SUPPORT_SLAB,
) -> np.ndarray:
    """Centre of the object's footprint, on the support plane.

    This is where the object's weight goes through into the table or the shelf.
    Projecting it exactly onto the plane means the map is constrained to take
    one surface to the other, rather than to somewhere a few millimetres above
    or below it.

    The lateral position is the *box's* centre rather than the centroid of the
    points near the plane. That centroid is an unstable statistic -- a thin band
    of a randomly sampled cloud -- and it moved 3.3 mm between two extractions
    of the same box here. Inconsistency of that kind is not noise the map
    averages away: ``phi`` interpolates every keypoint exactly, so it becomes a
    local deformation planted directly under the object.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    local = to_frame(points, frame)
    near = local[local[:, 2] <= support_height + slab]
    if len(near) == 0:  # nothing resting: fall back to the lowest points
        near = local[np.argsort(local[:, 2])[: max(1, len(local) // 50)]]
    local_centre = to_frame(np.asarray(centre, dtype=float).reshape(1, 3), frame)[0]
    return from_frame(
        np.array([local_centre[0], local_centre[1], float(support_height)])[None], frame
    )[0]


def object_keypoints(
    points: np.ndarray,
    grasp: GraspFrame,
    support_normal: np.ndarray = (0.0, 0.0, 1.0),
    support_height: float | None = None,
    name: str = "obj",
    frame: np.ndarray | None = None,
    include_contacts: bool = False,
    box: str = "cloud",
    orientation: str = "task",
    cube_half_extent: float = GRASP_CUBE_HALF_EXTENT,
) -> KeypointSet:
    """The twelve keypoints describing one object in one configuration.

    Nine points from the grasp-aligned box: its centre and its eight corners,
    matching Sec. V-A's cube but with the extent measured from the cloud and the
    axes taken from the task rather than from the mesh. Every one has a role
    that exists on any object and for any gripper, so two such sets pair
    elementwise.

    The four lower corners lie exactly on the support plane, which is the
    property that makes ``phi`` take table to shelf rather than to somewhere
    above or below it: Sec. III-D interpolates keypoints exactly.

    Args:
        points: ``(N, 3)`` world-frame cloud of the object.
        grasp: The grasp that will be, or was, taken on it.
        support_normal: Normal of the surface it rests on.
        support_height: Height of that surface along the normal. Defaults to the
            lowest point of the cloud, which is only correct for a cloud whose
            lowest points survived segmentation.
        name: Label prefix, which is what pairs source to target.
        frame: Override the task frame. Used when a configuration must reuse the
            frame of another, rather than re-deriving it.
        include_contacts: Also emit the two jaw contacts and the support
            contact, twelve points in all. **Off by default, and the reason is
            measured**: contact keypoints pin where along the object the jaws
            sit, while the box corners simultaneously pin the object's shape.
            When the target grasp sits at a different height on its object than
            the source grasp did, those two demands contradict each other, and
            since every keypoint is interpolated *exactly* the map satisfies
            both by folding -- ``det(J)`` changes sign and property (ii) of
            Sec. III-D fails. On an otherwise identical scene the minimum
            ``det(J)`` fell from 1.04 at a matched grasp height to 0.44 at
            three-quarter height and -0.07 at mid-height.

            The grasp's position along the object is better handled where it
            already is: the grasp itself converts to an end-effector target
            through the verified frame contract in ``tpgpt.grasp.grasps``. The
            warp only needs the grasp's *orientation*, and it has that from the
            task frame. Kept available because it is the natural ablation for
            that claim.
        box: Where the nine box points come from.

            ``"cloud"`` (default, unchanged) fits a box to the object's own
            point cloud and snaps its lower face to the support plane.

            ``"grasp_cube"`` uses a **fixed-size cube centred on the grasp**.
            This is what the paper's Sec. V-A actually specifies, and what the
            validated reshelving experiment already uses through
            :func:`cube_keypoints`; the cloud-fitted box is this
            implementation's own deviation.

            The measured reason to have it: a cloud box is centred on the
            object's *centroid* while a jaw contact is anchored at the *grasp*,
            and nothing couples them, so the two disagree by ``delta`` over a
            distance ``L``. ``det(J)`` falls as ``1 - delta/L`` and folds at
            ``delta/L`` around 0.9. On a source can (58 x 90 mm) transported onto
            a carton (50 x 250 mm), ``delta ~= 53 mm`` against ``L ~= 29 mm``,
            and the map folds at ``min det(J) = -0.63`` **with no contacts
            involved** while inflating space 2.80x vertically. Centring the cube
            on the grasp makes the box centre *be* the grasp, so ``delta``
            collapses to the jaw-aperture difference (~4 mm) and the same case
            gives ``min det(J) = 0.91`` with no stretch.
        orientation: How the eight corners are laid out.

            ``"task"`` (default) uses the task frame, which keeps the support
            normal fixed and therefore carries only a *yaw*. ``"grasp"`` uses
            :func:`grasp_pose_frame`, carrying the grasp's full orientation
            including its approach tilt -- at the cost of tilting the vertical by
            the same angle, one for one. Only valid with
            ``box="grasp_cube"``; a snapped cloud box in a tilted frame is
            meaningless and is refused.
        cube_half_extent: Size of the cube, when ``box="grasp_cube"``. See
            :data:`GRASP_CUBE_HALF_EXTENT`.

    Raises:
        ValueError: for an unknown mode, for ``box="cloud"`` with
            ``orientation="grasp"``, for a cloud too sparse to fit a box, or for
            two coincident keypoints.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(points) < 4:
        raise ValueError(
            f"{name}: {len(points)} points is too few to fit an oriented box; "
            "a cloud this sparse cannot describe an object's extent"
        )
    if box not in ("cloud", "grasp_cube"):
        raise ValueError(f"unknown box mode {box!r}; expected 'cloud' or 'grasp_cube'")
    if orientation not in ("task", "grasp"):
        raise ValueError(
            f"unknown orientation {orientation!r}; expected 'task' or 'grasp'"
        )
    if box == "cloud" and orientation == "grasp":
        raise ValueError(
            "a cloud-fitted box cannot be laid out in the grasp pose: its lower "
            "face is snapped to the support plane, which it would no longer be "
            "parallel to. Use box='grasp_cube' for a tilted layout."
        )

    frame = task_frame(support_normal, grasp.closing) if frame is None else frame
    if support_height is None:
        support_height = float(to_frame(points, frame)[:, 2].min())

    # The object's own extent, measured in every mode. Under a fixed cube the
    # keypoints no longer carry the object's width, and
    # ``diagnose.closing_budget`` derives ``(aperture - width) / 2`` from the
    # keypoint spread along the closing axis -- so without this it would
    # silently start reporting the *cube's* width and hand back a confident,
    # wrong tolerance. Recorded unconditionally so the budget has an honest
    # source in both modes.
    cloud_centre, cloud_half = fit_aligned_box(points, frame, support_height)

    if box == "cloud":
        centre, half = cloud_centre, cloud_half
    else:
        centre = np.asarray(grasp.tcp, dtype=float).reshape(3)
        half = np.full(3, float(cube_half_extent))

    corner_frame = (
        frame
        if orientation == "task"
        else grasp_pose_frame(grasp)
    )
    all_points = np.vstack(
        [centre[None], centre + from_frame(CUBE_CORNERS * half, corner_frame)]
    )
    labels = [f"{name}_center", *[f"{name}_{n}" for n in CORNER_NAMES]]

    from_cloud = None
    if include_contacts:
        # Both contact families stay anchored to the *cloud*: the grasp-plane
        # fallback is "too little cloud, use the object's extent", which a fixed
        # cube cannot supply, and the support point means "the centre of the
        # object's footprint" in either mode.
        contacts, from_cloud = grasp_contacts(
            points, grasp, frame, cloud_centre, cloud_half
        )
        support = support_contact(points, frame, support_height, cloud_centre)
        all_points = np.vstack([all_points, contacts, support[None]])
        labels += [f"{name}_{n}" for n in CONTACT_NAMES]

    _refuse_coincident(all_points, labels, name)

    return KeypointSet(
        points=all_points,
        labels=labels,
        frame="world",
        metadata={
            "object": name,
            "task_frame": frame.tolist(),
            "box_centre": centre.tolist(),
            "box_half_extents": half.tolist(),
            "support_height": float(support_height),
            "contacts_from_cloud": from_cloud,
            "n_cloud_points": int(len(points)),
            # What this block was actually built from, so a reader of a saved
            # set can tell the two constructions apart, and so the figures can
            # stop claiming the lower corners are on the support plane when
            # they are not.
            "block_kind": box,
            "corner_orientation": orientation,
            "cube_half_extent": (
                float(cube_half_extent) if box == "grasp_cube" else None
            ),
            "corner_frame": corner_frame.tolist(),
            "grasp_tcp": np.asarray(grasp.tcp, dtype=float).reshape(3).tolist(),
            "cloud_centre": cloud_centre.tolist(),
            "cloud_half_extents": cloud_half.tolist(),
        },
    )


def _refuse_coincident(points: np.ndarray, labels: list[str], name: str) -> None:
    """Refuse a keypoint set with two points in the same place.

    Coincident keypoints make the map's Gram matrix singular, and the failure
    surfaces from deep inside the GP as ``LinAlgError("Gram matrix is not
    positive definite even with escalated jitter; check for duplicate training
    inputs.")`` -- which names the symptom and not the cause.

    This became reachable with the grasp-centred cube, whose centre is the grasp
    TCP rather than the cloud centroid: a grasp sitting exactly on the support
    plane would put the cube's centre on top of the support contact. Refusing
    here says which two keypoints collided.
    """
    if len(points) < 2:
        return
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    i, j = np.unravel_index(int(distance.argmin()), distance.shape)
    if distance[i, j] < MIN_KEYPOINT_SEPARATION:
        raise ValueError(
            f"{name}: keypoints {labels[i]!r} and {labels[j]!r} are "
            f"{distance[i, j]:.2e} m apart, which would make the map's Gram "
            "matrix singular"
        )


# ---------------------------------------------------------------------------
# Where the object ends up (paper Sec. III-A: the target set must describe the
# *new* scene, including the configuration the object has not reached yet)
# ---------------------------------------------------------------------------


def carry_indices(labels) -> tuple[int, int]:
    """First and last label index at which the gripper is closed.

    Raises:
        ValueError: if the demonstration never closes the gripper, in which case
            there is no carried phase to inherit.
    """
    gripper = np.asarray(labels.gripper, dtype=float).reshape(-1)
    closed = np.flatnonzero(gripper > 0.0)
    if len(closed) == 0:
        raise ValueError(
            "the demonstration never closes its gripper, so it has no carry "
            "phase and no placed configuration to inherit"
        )
    return int(closed[0]), int(closed[-1])


def carry_transform(labels, grasp_index=None, release_index=None):
    """Rigid motion the grasped object underwent between grasp and release.

    Once the jaws close the object is rigidly attached to the hand, so
    ``T_obj(t) = T_eef(t) . T_eef->obj`` with the offset frozen at closure.
    The object's motion is therefore
    ``T_eef(release) . T_eef(grasp)^-1``, read straight off the labels.

    This is why the demonstration needs no second observation of the placed
    object: the placed pose is *implied* by the grasp and the trajectory, it is
    exact, and it is consistent with the labels -- whereas an image taken at
    release sees an object half occluded by the gripper that is holding it.

    Returns:
        ``(rotation, translation)`` taking a point on the object at grasp time
        to the same material point at release time.
    """
    if labels.orientations is None:
        raise ValueError("carry transform needs orientation labels")
    g, r = carry_indices(labels) if grasp_index is None else (grasp_index, release_index)
    R_g = np.asarray(labels.orientations[g], dtype=float).reshape(3, 3)
    R_r = np.asarray(labels.orientations[r], dtype=float).reshape(3, 3)
    p_g = np.asarray(labels.positions[g], dtype=float).reshape(3)
    p_r = np.asarray(labels.positions[r], dtype=float).reshape(3)
    rotation = R_r @ R_g.T
    translation = p_r - rotation @ p_g
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    """Apply a rigid motion to a cloud."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ np.asarray(rotation, dtype=float).T + np.asarray(
        translation, dtype=float
    ).reshape(3)


def placement_rotation(
    source_place_frame: np.ndarray,
    target_pick_frame: np.ndarray,
) -> np.ndarray:
    """Rotation putting the target object into the orientation it must end in.

    What the demonstration shows is not "the object turned by 40 degrees" but
    "the object ended up square with the shelf". Those coincide only when the
    target object happens to start at the same yaw as the source one did.
    Inheriting the *carried amount* instead leaves the target sitting at the
    difference between the two starting yaws -- measured here at 8 degrees of
    residual gripper yaw, enough to catch the box corners on the jaws.

    So the placed **orientation relative to the receptacle** is inherited, in
    exactly the way the placed position already is: the target object's task
    frame at release is made equal to the source object's task frame at
    release.

    Assumes the source and target receptacles share an orientation, which holds
    for the axis-aligned shelves of both scenes here. A general receptacle
    would conjugate this by its own frame.
    """
    return np.asarray(source_place_frame, dtype=float) @ np.asarray(
        target_pick_frame, dtype=float
    ).T


def lateral_offset(
    place_points: np.ndarray,
    destination: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Where the demonstration put the object within its slot, in frame axes.

    Expressed in the task frame rather than in world axes so that it travels
    with the object's orientation: an object grasped off-centre is placed
    off-centre in the direction it is held, not in the direction of world ``x``.
    """
    centre, _ = fit_aligned_box(place_points, frame)
    local_centre = to_frame(centre[None], frame)[0]
    local_dest = to_frame(np.asarray(destination, dtype=float).reshape(1, 3), frame)[0]
    return (local_centre - local_dest)[:2]


def place_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    frame: np.ndarray,
    destination: np.ndarray,
    support_height: float,
    offset: np.ndarray = (0.0, 0.0),
    pick_support_height: float | None = None,
):
    """Rigid motion putting an object into a destination slot.

    Rotates about the cloud's own centroid, then translates so the object sits
    laterally at the destination (plus the offset inherited from the
    demonstration) and rests with its lower face on the destination surface.

    The vertical snap is the part that matters. Bread measures 4.8 cm tall and
    a cereal box 15.4 cm, so inheriting the source object's placement height
    would bury one in the shelf and leave the other hanging above it.

    ``pick_support_height`` is the plane the object rests on *now*. Passing it
    makes this fit the same corrected box that the pick configuration uses:
    without it the placement is computed from the raw cloud, whose lowest points
    are truncated, and the same cereal box measures 15.4 cm at the pick and
    14.7 cm at the place. The rotation is about the support normal, so the
    heights carry over unchanged.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    pivot = points.mean(axis=0)
    rotated = apply_transform(points - pivot, rotation, pivot)

    centre, half = fit_aligned_box(rotated, frame, pick_support_height)
    local_centre = to_frame(centre[None], frame)[0]
    local_dest = to_frame(np.asarray(destination, dtype=float).reshape(1, 3), frame)[0]
    wanted = np.array(
        [
            local_dest[0] + float(offset[0]),
            local_dest[1] + float(offset[1]),
            float(support_height) + half[2],
        ]
    )
    shift = from_frame((wanted - local_centre)[None], frame)[0]
    translation = pivot - rotation @ pivot + shift
    return rotation, translation


def sample_box_surface(
    centre: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    n: int = 1200,
    rng: np.random.Generator | None = None,
    faces: str = "all",
) -> np.ndarray:
    """Sample a synthetic cloud on the surface of a box.

    Used where the object's geometry is known exactly and the question under
    test is the keypoint construction rather than the perception in front of
    it -- the unprojection pipeline has its own acceptance tests.

    Args:
        faces: ``"all"`` for a closed box, or ``"visible"`` to drop the bottom
            face, which is what a camera above the table actually sees.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    centre = np.asarray(centre, dtype=float).reshape(3)
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    half = np.broadcast_to(np.asarray(half_extents, dtype=float), (3,))

    axes = [0, 1, 2]
    signs = [-1.0, 1.0]
    combos = [(a, s) for a in axes for s in signs if not (faces == "visible" and a == 2 and s < 0)]
    per_face = max(1, n // len(combos))

    local = []
    for axis, sign in combos:
        pts = rng.uniform(-1.0, 1.0, size=(per_face, 3))
        pts[:, axis] = sign
        local.append(pts * half)
    return centre + np.vstack(local) @ R.T


def top_down_grasp(
    points: np.ndarray,
    support_normal: np.ndarray = (0.0, 0.0, 1.0),
    height_fraction: float = 0.5,
    closing_axis: np.ndarray | None = None,
) -> GraspFrame:
    """A scripted vertical grasp on a cloud, for use before grasp selection exists.

    Grasp *selection* is a separate, undesigned stage: only about a quarter of
    GraspGen-X candidates approach from above (ROBOTICS_NOTES 5.6). Driving
    keypoint extraction from a scripted grasp keeps the two stages failing
    independently, so a bad placement here is attributable to the keypoints.

    The jaws close across the cloud's narrower horizontal direction, which is
    what a parallel gripper must do. Pass ``closing_axis`` explicitly when the
    object's cross-section is square and that choice is degenerate.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = _unit(support_normal)
    if len(points) < 4:
        raise ValueError(
            f"{len(points)} points is too few to choose a grasp direction; the "
            "object is occluded or its segmentation mask was eroded away"
        )

    if closing_axis is None:
        flat = points - np.outer(points @ n, n)
        centred = flat - flat.mean(axis=0)
        # Minor axis of the horizontal spread: the direction of least extent.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        candidates = [v for v in vt if abs(float(v @ n)) < 0.9]
        closing_axis = candidates[-1] if candidates else vt[-1]

    heights = points @ n
    low, high = float(heights.min()), float(heights.max())
    tcp_height = low + height_fraction * (high - low)
    flat_centre = points.mean(axis=0)
    tcp = flat_centre - float(flat_centre @ n) * n + tcp_height * n
    return GraspFrame(tcp=tcp, approach=-n, closing=closing_axis)


def concat_keypoints(*sets: KeypointSet) -> KeypointSet:
    """Concatenate keypoint sets, preserving order and merging metadata."""
    points = np.vstack([s.points for s in sets])
    labels = [label for s in sets for label in s.labels]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate keypoint labels would break Sec. III-A pairing")
    metadata = {}
    for s in sets:
        metadata[s.metadata.get("object", f"block{len(metadata)}")] = s.metadata
    return KeypointSet(points=points, labels=labels, frame="world", metadata=metadata)


@dataclass
class ObjectPlacement:
    """An object, the surface it rests on, and where the task wants it.

    Attributes:
        points: ``(N, 3)`` world-frame cloud of the object where it stands now.
        grasp: The grasp to be taken on it.
        support_height: World height of the surface it rests on.
        destination: World position of the destination slot. Only its lateral
            components are used; the height comes from ``destination_height``.
        destination_height: World height of the destination's support surface.
        metadata: Free-form provenance. A source placement outlives the
            environment it was measured in, so anything a later report needs
            from that scene -- a picture of it, the matrix that projects into
            that picture -- has to travel here.
    """

    points: np.ndarray
    grasp: GraspFrame
    support_height: float
    destination: np.ndarray
    destination_height: float
    metadata: dict = field(default_factory=dict)


def scene_keypoints(
    source: ObjectPlacement,
    target: ObjectPlacement,
    labels,
    support_normal: np.ndarray = (0.0, 0.0, 1.0),
    include_contacts: bool = False,
    flip_target: bool = False,
    box: str = "cloud",
    orientation: str = "task",
    cube_half_extent: float = GRASP_CUBE_HALF_EXTENT,
) -> tuple[KeypointSet, KeypointSet, dict]:
    """Paired source and target keypoint sets for a transportation map.

    Four configurations, twelve keypoints each: the source object where the
    demonstration found it and where the demonstration left it, and the target
    object where it stands now and where it must end up.

    Both ends of the task are constrained, which Sec. III-A requires: keypoints
    clustered at the pick alone would leave the placement to extrapolation.

    The target's placed configuration is the only one that is not observed, and
    it is *computed* rather than transferred:

    * its orientation inherits the demonstration's carry rotation expressed in
      each object's own task frame, so "a quarter turn about the grasp axis"
      survives a target object standing at a different yaw;
    * its lateral position inherits where the demonstration put the object
      within its slot;
    * its height comes from the destination surface, so objects of different
      height rest on the shelf rather than in it or above it.

    No transform between the source and target *objects* is fitted anywhere.
    A carton and a lemon are not related by one, so fitting it would inject an
    arbitrary rotation into ``phi``. The recipe is shared; the geometry is not.

    Args:
        box: Forwarded to :func:`object_keypoints` for all four blocks.
        orientation: Likewise. With ``"grasp"``, each block's corners are laid
            out in its own grasp's pose, taken as given -- no sign is resolved
            anywhere, so no two blocks can disagree about which corner is which.
        flip_target: Take the *other* roll about the target grasp's approach
            axis, rather than the one that agrees with the demonstration. Both
            grip the object identically and both are executable; they are
            different wrist configurations, and a wrist has a limited range, so
            a caller may legitimately try both and keep whichever the arm can
            hold. Applied to the grasp before either target block is derived, so
            the pick and the place can never disagree.
        cube_half_extent: Likewise.

    Returns:
        ``(source_set, target_set, diagnostics)``, the sets paired elementwise
        and the diagnostics carrying the synthesised placed clouds for plotting.
    """
    n = _unit(support_normal)

    source_frame = task_frame(n, source.grasp.closing)

    # --- which way round to approach the target -----------------------------
    #
    # A parallel jaw grips the same object identically with its fingers swapped,
    # so the target grasp and that grasp rolled a half turn about its approach
    # are two poses for one physical grip. Both are executable; they are
    # different wrist configurations. **Something has to choose, and the choice
    # is made here, once, on the grasp itself** -- so both target blocks inherit
    # it and cannot disagree.
    #
    # The criterion is agreement with the demonstration: take the roll whose
    # closing axis points the same way as the source's. That is a statement
    # about the task -- grip this object the way the demonstrated hand gripped
    # its object -- and it needs one dot product.
    #
    # **This is not the sign resolution that 7.33 removed, and the difference
    # matters.** That one ran independently at each of four blocks, and decided
    # by asking whether the axis pointed along world ``+y`` -- a question about
    # the object's yaw in the scene, unrelated to the gripper, which answered
    # differently at the pick and the place and planted a half turn between
    # them. This runs once, on a grasp, and compares against the demonstration.
    #
    # **Why it is not optional.** ``phi`` has to carry the source cube onto the
    # target cube, so the angle between the two frames is a rotation the map
    # must realise. Measured across 20 cells, ``min det(J)`` against that angle:
    #
    # =================  ==============  =========================
    # source-to-target   ``min det``     ``min det``, other roll
    # =================  ==============  =========================
    # 3 - 19 deg         0.94 - 0.99     0.006 - 0.08
    # 74 - 135 deg       0.33 - 0.79     0.46 - 0.99
    # 149 - 179 deg      **-0.06 - 0.16**  0.93 - 1.00
    # =================  ==============  =========================
    #
    # Three of twenty maps fold outright without it. Choosing by agreement gives
    # a worst case of 0.456 and a median of 0.93 over the same cells, against
    # 0.443 and 0.878 for the world-axis version it replaces.
    take_other_roll = float(
        np.asarray(target.grasp.closing) @ np.asarray(source.grasp.closing)
    ) < 0.0
    if flip_target:
        take_other_roll = not take_other_roll
    if take_other_roll:
        target = ObjectPlacement(
            points=target.points,
            grasp=GraspFrame(
                tcp=target.grasp.tcp,
                approach=target.grasp.approach,
                closing=-target.grasp.closing,
            ),
            support_height=target.support_height,
            destination=target.destination,
            destination_height=target.destination_height,
            metadata=target.metadata,
        )
    source_pick = object_keypoints(
        source.points, source.grasp, n, source.support_height, "src_pick",
        source_frame, include_contacts, box, orientation, cube_half_extent,
    )

    carry_rotation, carry_translation = carry_transform(labels)
    source_place_points = apply_transform(source.points, carry_rotation, carry_translation)
    source_place_grasp = source.grasp.transformed(carry_rotation, carry_translation)
    source_place_frame = task_frame(n, source_place_grasp.closing)
    source_place = object_keypoints(
        source_place_points,
        source_place_grasp,
        n,
        source.destination_height,
        "src_place",
        source_place_frame,
        include_contacts,
        box,
        orientation,
        cube_half_extent,
    )
    offset = lateral_offset(source_place_points, source.destination, source_place_frame)

    # **No reference, and none is needed.** Each frame takes its sign from its
    # own grasp, and a grasp's closing axis has a definite sign fixed by the
    # planner and by which finger of the hand is which. Two frames derived that
    # way cannot disagree; two frames that each invent a sign from the object's
    # yaw in the world can, and did, by 180 degrees (7.33).
    target_frame = task_frame(n, target.grasp.closing)
    target_pick = object_keypoints(
        target.points, target.grasp, n, target.support_height, "tgt_pick",
        target_frame, include_contacts, box, orientation, cube_half_extent,
    )

    # Rotating the target object this way sends its task frame exactly onto the
    # source object's placed frame, so the two configurations describe the same
    # relationship to the shelf and the lateral offset below is measured in the
    # same axes at both ends.
    target_carry = placement_rotation(source_place_frame, target_frame)
    target_place_frame = source_place_frame
    rotation, translation = place_transform(
        target.points,
        target_carry,
        target_place_frame,
        target.destination,
        target.destination_height,
        offset,
        target.support_height,
    )
    target_place_points = apply_transform(target.points, rotation, translation)
    target_place_grasp = target.grasp.transformed(rotation, translation)
    target_place = object_keypoints(
        target_place_points,
        target_place_grasp,
        n,
        target.destination_height,
        "tgt_place",
        target_place_frame,
        include_contacts,
        box,
        orientation,
        cube_half_extent,
    )

    source_set = concat_keypoints(source_pick, source_place)
    target_set = concat_keypoints(target_pick, target_place)
    # Pairing is by role, so the labels must correspond position by position.
    source_set.labels = [l.replace("src_", "") for l in source_set.labels]
    target_set.labels = [l.replace("tgt_", "") for l in target_set.labels]

    diagnostics = {
        "source_place_points": source_place_points,
        "target_place_points": target_place_points,
        "source_place_grasp": source_place_grasp,
        "target_place_grasp": target_place_grasp,
        "carry_rotation": carry_rotation,
        "carry_translation": carry_translation,
        "target_carry_rotation": target_carry,
        "lateral_offset": offset,
        "source_frame": source_frame,
        "source_place_frame": source_place_frame,
        "target_frame": target_frame,
        # **The grasp actually executed**, which is the incoming one rolled a
        # half turn about its approach when that agrees better with the
        # demonstration. Exposed because everything downstream that converts a
        # grasp into a robot command -- ``grasp_to_eef_pose`` above all -- has to
        # be given *this* pose and not the one the planner emitted, or it will
        # command the other roll and disagree with the plan by 180 degrees.
        "target_grasp": target.grasp,
        "target_grasp_rolled": bool(take_other_roll),
        "contacts_from_cloud": {
            "src_pick": source_pick.metadata["contacts_from_cloud"],
            "src_place": source_place.metadata["contacts_from_cloud"],
            "tgt_pick": target_pick.metadata["contacts_from_cloud"],
            "tgt_place": target_place.metadata["contacts_from_cloud"],
        },
    }
    return source_set, target_set, diagnostics
