"""The scene's solid geometry, as primitives a point can be tested against.

**Why this exists rather than another point cloud.** Everything else in this
package describes the scene the way the robot's cameras see it -- a depth image
turned into world points. That is the honest representation for *objects*, whose
shape is not known in advance. It is a poor one for the *shelf*, and
``ROBOTICS_NOTES.md`` 7.35 and 7.38 are both about collisions with the shelf:

* the cloud is **one-sided and occluded**. Measured on the default tabletop
  scene at 256 px with three cameras, the column of space directly above the
  ``top_middle`` slot -- the one every placement descends through -- holds
  **76** points out of a scene cloud capped at 8192. A hand 10 cm across can
  pass between them;
* a cloud gives **no depth of penetration**. The nearest-neighbour test in
  :func:`~tpgpt.grasp.filters.by_collision` answers "is a scene point within
  10 mm of the hand", which cannot distinguish a finger grazing a wall from a
  wrist buried 80 mm inside it. 7.38 measured that the discriminating quantity
  is *sustained* penetration depth, so a check that cannot measure depth cannot
  apply the criterion.

Both problems disappear once you notice what the obstacles actually are. Every
solid, immovable thing in this scene is a **box, a cylinder or a plane**: the
table top, the twelve shelf panels, the robot's pedestal and its controller box,
and the floor. Each has a closed-form signed distance function, so "how far is
this point inside that obstacle" is a few arithmetic operations, exact, with no
sampling density to choose and nothing hidden behind anything else.

The movable objects are meshes, and they are handled by their **oriented
bounding boxes**, which MuJoCo already stores as each mesh geom's ``size``.
That is an over-estimate of the object -- a bounding box contains the mesh --
so a report of "clear" is conservative and a report of "inside" may be a corner
of empty space. For the question this is used for, keeping a hand out of a
neighbouring carton, that is the right direction to err in.

Nothing here reads a camera, so it works with the renderer off and costs
microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Geom name prefixes belonging to the robot itself.
#:
#: The robot is not an obstacle to its own hand in the sense this module means:
#: a plan is checked for driving the hand through the *scene*, and self-collision
#: is a separate question with a separate answer (MuJoCo's own narrowphase, via
#: :func:`~tpgpt.grasp.filters.arm_collides`). The pedestal and the controller
#: box are the exception and are deliberately **not** matched by these prefixes,
#: because they are static furniture the arm can genuinely drive into -- 7.38
#: recorded the transported path hitting the pedestal.
ROBOT_PREFIXES = ("robot0", "gripper0")


@dataclass(frozen=True)
class Obstacle:
    """One solid primitive, with an exact inside/outside test.

    Attributes:
        name: The MuJoCo geom name, so a collision can be attributed to a
            specific panel rather than reported as an anonymous hit.
        kind: ``"box"``, ``"cylinder"`` or ``"plane"``.
        position: World position of the primitive's centre (a point on the
            surface, for a plane).
        rotation: ``(3, 3)`` world rotation of its local frame.
        size: MuJoCo's own size vector for that geom type -- box half extents,
            ``(radius, half height)`` for a cylinder, ignored for a plane.
        movable: Whether the geom belongs to a body with degrees of freedom.
            Recorded rather than filtered out, because a caller checking a
            *carry* wants the neighbouring objects in and a caller checking the
            static layout does not.
    """

    name: str
    kind: str
    position: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    movable: bool = False

    @property
    def bounding_radius(self) -> float:
        """Radius of a sphere about ``position`` containing the whole primitive.

        Used only to skip obstacles that cannot possibly be touched, so it has
        to be an over-estimate and never an under-estimate. A plane is
        unbounded and reports infinity, which means it is never skipped.
        """
        if self.kind == "plane":
            return float("inf")
        if self.kind == "cylinder":
            return float(np.hypot(self.size[0], self.size[1]))
        return float(np.linalg.norm(self.size[:3]))

    def penetration(self, points: np.ndarray) -> np.ndarray:
        """How far each point lies **inside** this obstacle, in metres.

        Positive is inside; zero or negative is outside, and for a point
        outside the magnitude is the distance to the surface (exact for a
        plane, and the usual slight under-estimate near a box's corner, which
        errs towards reporting *less* clearance than there is).

        Args:
            points: ``(n, 3)`` world coordinates.

        Returns:
            ``(n,)`` signed depths.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if self.kind == "plane":
            # MuJoCo's plane faces along its own +Z and is a half space below.
            normal = self.rotation[:, 2]
            return -((p - self.position) @ normal)

        local = (p - self.position) @ self.rotation
        if self.kind == "box":
            # The standard box signed distance, negated so inside is positive.
            outside = np.abs(local) - np.asarray(self.size, dtype=float)
        elif self.kind == "cylinder":
            radial = np.linalg.norm(local[:, :2], axis=1) - float(self.size[0])
            axial = np.abs(local[:, 2]) - float(self.size[1])
            outside = np.column_stack([radial, axial])
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"no penetration test for obstacle kind {self.kind!r}")

        inside = np.all(outside < 0.0, axis=1)
        depth = np.where(
            inside,
            -outside.max(axis=1),
            -np.linalg.norm(np.maximum(outside, 0.0), axis=1),
        )
        return depth


#: MuJoCo geom type ids this module can represent exactly.
#:
#: Read from :mod:`mujoco` at call time rather than hard-coded, because the
#: numeric ids are an implementation detail of the enum.
_SUPPORTED = ("plane", "box", "cylinder")


def scene_obstacles(
    env,
    exclude: tuple[str, ...] = (),
    include_movable: bool = True,
    include_floor: bool = False,
) -> list[Obstacle]:
    """Every solid thing in the scene, as primitives.

    Only **collidable** geoms are returned -- those with a non-zero ``contype``
    or ``conaffinity``. That matters more than it sounds: this scene carries a
    visual-only marker box at every shelf slot, and a naive geometric query hits
    those first. They are drawn, not solid, and a hand passes straight through
    them.

    Robot geoms are excluded by name (:data:`ROBOT_PREFIXES`), since a plan is
    being checked against the *scene*. The pedestal and controller box are
    static furniture under a different prefix and are kept.

    Args:
        env: A live robosuite environment.
        exclude: Substrings naming geoms to leave out. The object being carried
            belongs here: the hand holds it, so its geometry is not an obstacle
            to the hand.
        include_movable: Keep the movable objects (as oriented bounding boxes).
            The neighbouring cartons are real obstacles -- 7.38 recorded 98 hits
            on ``milk_g0`` along one transported path -- so this defaults on.
        include_floor: Keep the ground plane. Off by default: it is 80 cm below
            the table and nothing that reaches it is a near miss, so including
            it only adds a test that never fires.

    Returns:
        The obstacles, in MuJoCo geom order.
    """
    import mujoco

    model, data = env.sim.model, env.sim.data
    kinds = {int(getattr(mujoco.mjtGeom, f"mjGEOM_{k.upper()}")): k for k in _SUPPORTED}

    obstacles: list[Obstacle] = []
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        if name.startswith(ROBOT_PREFIXES):
            continue
        if any(token and token in name for token in exclude):
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        kind = kinds.get(int(model.geom_type[gid]))
        body = int(model.geom_bodyid[gid])
        movable = bool(model.body_dofnum[body] or model.body_jntnum[body])
        if kind is None:
            # A mesh, a capsule, an ellipsoid. Every one in this scene is a
            # movable object, and MuJoCo stores a mesh geom's bounding-box half
            # extents in ``geom_size`` -- so the box is free and already there.
            # A *static* unsupported geom would be silently dropped, which is
            # the one case worth refusing over, since it would be scenery the
            # check cannot see.
            if not movable:
                raise NotImplementedError(
                    f"static geom {name!r} is a "
                    f"{mujoco.mjtGeom(model.geom_type[gid]).name} and this "
                    "module has no exact test for it; a plan checked against "
                    "this scene would pass straight through it"
                )
            kind = "box"
        if kind == "plane" and not include_floor:
            continue
        if movable and not include_movable:
            continue
        obstacles.append(
            Obstacle(
                name=name,
                kind=kind,
                position=np.array(data.geom_xpos[gid], dtype=float),
                rotation=np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3),
                size=np.array(model.geom_size[gid], dtype=float),
                movable=movable,
            )
        )
    return obstacles


def deepest_penetration(
    points: np.ndarray, obstacles, clearance: float = 0.0
) -> tuple[float, str | None]:
    """The worst intrusion of a set of points into a set of obstacles.

    Args:
        points: ``(n, 3)`` world coordinates -- a hand's surface sample, say.
        obstacles: What to test against.
        clearance: Treat a point within this distance of a surface as already
            touching. Zero means literal geometric overlap.

    Returns:
        ``(depth, name)``: how far the deepest point is inside, in metres, and
        which obstacle it is inside. ``(0.0, None)`` when everything is clear.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    # A bounding-sphere prune. The point set is a hand -- a few hundred samples
    # inside a ball 20 cm across -- and the obstacle set is a whole room, so on
    # a typical waypoint two or three obstacles are candidates and the rest are
    # metres away. Both radii are over-estimates, so this can only skip an
    # obstacle that is genuinely out of reach; it changes the cost, never the
    # answer. Measured: about a fourfold speed-up on the tabletop scene.
    centre = p.mean(axis=0)
    reach = float(np.linalg.norm(p - centre, axis=1).max()) + clearance

    worst, culprit = 0.0, None
    for obstacle in obstacles:
        radius = obstacle.bounding_radius
        if np.isfinite(radius) and (
            float(np.linalg.norm(obstacle.position - centre)) > reach + radius
        ):
            continue
        depth = float(obstacle.penetration(p).max()) + clearance
        if depth > worst:
            worst, culprit = depth, obstacle.name
    return worst, culprit


def first_obstruction(
    env,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float = 1.0,
    exclude: tuple[str, ...] = (),
    ignore_robot: bool = True,
) -> tuple[float, str | None]:
    """Distance along a ray to the first **solid** thing, using MuJoCo's own caster.

    Wraps :func:`mujoco.mj_ray`, whose one awkwardness is that it intersects
    everything that is *drawn*, including geoms with collision switched off.
    This scene has a translucent marker box at every slot, so an unguarded cast
    downwards from a slot reports an obstruction 58 mm away that the hand would
    pass straight through. The wrapper steps past any hit it is told to ignore
    and casts again, so the answer is always the first hit that matters.

    Args:
        origin: Ray start, world coordinates.
        direction: Ray direction; need not be normalised.
        max_distance: Stop looking beyond this, in metres. A miss and a hit
            beyond the limit are both reported as a miss.
        exclude: Substrings naming geoms that do not count.
        ignore_robot: Skip the robot's own links. On when the ray is asking
            "could the hand get here", since the hand is the thing travelling.

    Returns:
        ``(distance, name)`` of the first relevant hit, or
        ``(inf, None)`` if the ray is clear out to ``max_distance``.
    """
    import mujoco

    model, data = env.sim.model, env.sim.data
    origin = np.asarray(origin, dtype=float).reshape(3).copy()
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)

    travelled = 0.0
    geomid = np.zeros(1, dtype=np.int32)
    # Bounded: each iteration steps strictly past one geom, and a ray cannot
    # meet more geoms than the model holds.
    for _ in range(model.ngeom + 1):
        distance = mujoco.mj_ray(
            model._model, data._data, origin, direction, None, 1, -1, geomid
        )
        if distance < 0.0 or travelled + distance > max_distance:
            return float("inf"), None
        gid = int(geomid[0])
        name = model.geom_id2name(gid) or ""
        solid = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
        ignored = (
            not solid
            or (ignore_robot and name.startswith(ROBOT_PREFIXES))
            or any(token and token in name for token in exclude)
        )
        if not ignored:
            return float(travelled + distance), name
        # Step just past the surface we are ignoring and look again.
        step = distance + 1e-4
        origin = origin + direction * step
        travelled += step
    return float("inf"), None  # pragma: no cover - unreachable in a finite model
