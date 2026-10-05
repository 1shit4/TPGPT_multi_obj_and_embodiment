"""The Inspire RH56DFX-2R hand's geometry, in the frame GraspGen-X plans it in.

The hand on the FR3 is an **RH56DFX-2R, a right hand** (2026-10-04). GraspGen-X
describes the same hand as ``inspire_hand`` (``gripper.urdf``, robot
``inspire_hand_right``), so its URDF is this hand's kinematics and its pose
convention is the one every planned grasp arrives in. This module turns that URDF
into what the planner and the path check need -- no ROS, no robosuite, numpy and
scipy only (``tpgpt.grasp`` is not imported: it loads robosuite).

**Frames.**

* ``W``, GraspGen-X's grasp frame, the URDF's ``world`` link: ``+Z`` approach (the
  bisector of the palm normal and the finger direction), ``+X`` closing (the thumb
  on ``+X``, the four fingers on ``-X``, moving towards each other along it).
  GraspGen-X returns poses of this frame.
* ``H``, ``hand_base_link``: the wrist face. ``T_W<-H`` is the URDF's
  ``world_joint`` (``xyz 0.065 -0.01 0``, ``rpy 1.570796 2.356194 0``). In ``H``
  the fingers run along ``-Y`` (knuckles at y = -0.137) and the palm faces ``-X``.
* ``G``, the planner's grasp frame (:class:`tpgpt.sim.keypoints.GraspFrame`): the
  axes of ``W``, the origin at the hand's **tool point**, ``tool_depth`` along
  ``+Z`` from ``W``'s origin. With the description's own depth
  (``fingertip[-1]`` = 0.15 m) that is GraspGen-X's grasp point, between thumb and
  fingers; a measured depth correction ``delta`` moves the hand ``delta`` further
  along the approach (``tool_depth = 0.15 - delta``). Everything this module
  returns is in ``G``.

**Pieces.** Each link is one convex piece: a **26-DOP** (the tightest convex body
with faces in 26 fixed directions -- the 6 axes, 12 edge and 8 corner diagonals of
the link's principal axes -- around its collision mesh), which encloses the mesh,
so it can only over-state the hand, by 2.7-4.6 mm on these links; the palm is the
URDF's own boxes, exact, and the wrist cylinder a circumscribed 12-sided prism.
Each piece has ``name``, ``corners()``, ``contains(points, margin)`` and
``edges()`` -- the interface :mod:`tpgpt.hardware.path_check` already uses for the
Franka Hand's boxes -- and a ``role``: ``body`` (wrist, palm, knuckles, the
thumb's rotating base) or ``finger`` (the links that may touch the object at the
pick). Names are ``body_*`` and ``<finger>_<link>``; the path check prefixes
``hand_``.

**Joints.** The hand has six actuators (DOF order of the serial protocol: little,
ring, middle, index, thumb bend, thumb rotation) driving the URDF's
``pinky/ring/middle/index_proximal_joint``, ``thumb_proximal_pitch_joint`` and
``thumb_proximal_yaw_joint``; the other joints follow by the URDF's mimic rules
(``*_intermediate = 1.06399 q - 0.04545``, thumb intermediate ``1.334 q``, distal
``0.667 q``). Joint vectors here are always the six driven angles in DOF order,
radians. A register value ``u`` (0-1000) maps to its joint linearly per DOF,
``q = q_at_0 + (q_at_1000 - q_at_0) u / 1000``, from the ``joint_map`` block of
``config/inspire_hand.yaml`` -- the same block and formula the hand node uses.

**Closing onto an object** (:meth:`InspireHandModel.closed_onto`): each finger
chain moves from open towards the commanded close until it holds
:data:`CONTACT_POINTS` points :data:`CONTACT_DEPTH` deep, independently, as the
hand's force threshold stops each DOF on contact. The object is assumed not to
move. This is the multi-finger analogue of the Franka Hand's "fingers closed to
the object's width while holding" (``SHELF_AND_PATH_CHECK.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation

#: GraspGen-X's description of this hand.
GRASPGEN_NAME = "inspire_hand"
#: The serial protocol's DOF order, and the URDF joint each drives.
DOF_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
DRIVEN_JOINTS = ("pinky_proximal_joint", "ring_proximal_joint", "middle_proximal_joint",
                 "index_proximal_joint", "thumb_proximal_pitch_joint",
                 "thumb_proximal_yaw_joint")
#: The bending DOFs (the ones that close onto an object) and the chain each moves.
CHAIN_OF_DOF = {"little": "pinky", "ring": "ring", "middle": "middle", "index": "index",
                "thumb_bend": "thumb"}
FINGER_LINKS = {
    "pinky": ("pinky_proximal", "pinky_intermediate"),
    "ring": ("ring_proximal", "ring_intermediate"),
    "middle": ("middle_proximal", "middle_intermediate"),
    "index": ("index_proximal", "index_intermediate"),
    "thumb": ("thumb_proximal", "thumb_intermediate", "thumb_distal"),
}
#: The thumb's rotating base sits in the palm: body, not finger.
BODY_MESH_LINKS = {"thumb_proximal_base": "body_thumb_base"}
#: hand_base_link's collision primitives, in URDF order.
BASE_PIECE_NAMES = ("body_wrist", "body_palm_lower", "body_palm", "body_palm_upper",
                    "body_knuckle_1", "body_knuckle_2", "body_knuckle_3", "body_knuckle_4")
#: Register -> joint: the ruler fit of 2026-10-05 (``config/inspire_hand.yaml``
#: ``joint_map`` explains; about +-5 mm per reading, so still flagged unverified). The
#: config's block overrides it.
DEFAULT_JOINT_MAP = {
    "little": (1.470, 0.037), "ring": (1.470, 0.031), "middle": (1.470, 0.032),
    "index": (1.470, 0.029), "thumb_bend": (0.770, 0.120), "thumb_rot": (1.308, 0.0),
}
#: A finger is in contact once it holds this many points this deep -- the path
#: check's rule for measured points (``path_check.MIN_CLOUD_CONTACT_POINTS``,
#: ``CLOUD_CONTACT_SLACK``), so a stray depth pixel does not stop a finger.
CONTACT_POINTS = 3
CONTACT_DEPTH = 0.002
#: The 26 face directions of a k-DOP, in the link's principal axes.
_DOP_DIRECTIONS = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)],
                           dtype=float) - 1.0
_DOP_DIRECTIONS /= np.linalg.norm(_DOP_DIRECTIONS, axis=1, keepdims=True)
#: Sides of the prism that circumscribes the wrist cylinder.
PRISM_SIDES = 12


# ------------------------------------------------------------------ description
def descriptions_root() -> Path:
    """GraspGen-X's gripper-descriptions checkout (``tpgpt.grasp.grippers``' rule)."""
    sibling = Path(__file__).resolve().parents[3] / "6DOF_MultiEmbod_Grasp/assets/gripper_descriptions"
    return Path(os.environ.get("GRASPGENX_GRIPPER_CFG_DIR", sibling))


def description_dir(name: str) -> Path:
    """Where description ``name`` lives: the GraspGen-X checkout, else this repo's
    ``assets/x_grippers`` (where measured descriptions are written)."""
    for base in (descriptions_root() / "gripper_descriptions/assets/x_grippers",
                 Path(__file__).resolve().parents[2] / "assets" / "x_grippers"):
        if (base / name / "config.json").is_file():
            return base / name
    raise FileNotFoundError(f"no gripper description {name!r} (GraspGen-X checkout at "
                            f"{descriptions_root()}, or assets/x_grippers)")


def _origin(element) -> np.ndarray:
    T = np.eye(4)
    if element is None:
        return T
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()     # URDF fixed-axis rpy
    T[:3, 3] = xyz
    return T


def _read_obj(path: Path) -> np.ndarray:
    """The vertices of a Wavefront OBJ, ``(n, 3)``."""
    rows = [line.split()[1:4] for line in path.read_text().splitlines()
            if line.startswith("v ")]
    return np.asarray(rows, dtype=float)


@dataclass(frozen=True)
class URDFJoint:
    name: str
    parent: str
    child: str
    origin: np.ndarray                  # (4, 4) parent -> joint frame at zero
    axis: np.ndarray                    # (3,)
    mimic: tuple | None = None          # (driver, multiplier, offset)


@dataclass(frozen=True)
class HandDescription:
    """GraspGen-X's description of the hand: its URDF, meshes and ``config.json``.

    ``geometry`` is read from ``geometry_name`` (the URDF and meshes, always
    ``inspire_hand``); ``config`` from ``config_name`` -- the shipped description or
    one measured on the real hand (Stage 5), which carries no URDF of its own.
    """

    config_name: str
    config: dict
    joints: tuple                       # URDFJoint, parents before children
    link_geometry: dict                 # link -> list of (kind, data, origin 4x4)
    world_link: str
    hashes: dict

    @property
    def fingertip_depth(self) -> float:
        return float(self.config["fingertip"][-1])

    @property
    def symmetric(self) -> bool:
        return bool(self.config.get("symmetric", False))

    def _pose(self, key: str) -> np.ndarray:
        values = self.config[key]
        return np.array([float(values.get(j, 0.0)) for j in DRIVEN_JOINTS])

    @property
    def open_q(self) -> np.ndarray:
        return self._pose("open")

    @property
    def close_q(self) -> np.ndarray:
        return self._pose("close")

    def sweep_boxes(self) -> list:
        """The two sweep boxes in ``W``, ``[(lo, hi), (lo, hi)]`` (open, mid closure)."""
        sv = self.config["sweep_volume"]
        out = []
        for ext, off in (("extents", "offset"), ("extents2", "offset2")):
            e, o = np.asarray(sv[ext], dtype=float), np.asarray(sv[off], dtype=float)
            out.append((o - e / 2.0, o + e / 2.0))
        return out


@lru_cache(maxsize=8)
def load_description(config_name: str = GRASPGEN_NAME,
                     geometry_name: str = GRASPGEN_NAME) -> HandDescription:
    """Parse the URDF, the collision geometry and the config of the hand."""
    geo_dir, cfg_dir = description_dir(geometry_name), description_dir(config_name)
    urdf_text = (geo_dir / "gripper.urdf").read_text()
    cfg_text = (cfg_dir / "config.json").read_text()
    root = ET.fromstring(urdf_text)
    if root.get("name") != "inspire_hand_right":
        raise ValueError(f"{geo_dir / 'gripper.urdf'} is {root.get('name')!r}, not the right "
                         "hand this module was written for (RH56DFX-2R)")
    joints = []
    for j in root.findall("joint"):
        mimic = j.find("mimic")
        axis = j.find("axis")
        joints.append(URDFJoint(
            name=j.get("name"), parent=j.find("parent").get("link"),
            child=j.find("child").get("link"), origin=_origin(j.find("origin")),
            axis=(np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None
                  else np.zeros(3)),
            mimic=((mimic.get("joint"), float(mimic.get("multiplier", 1.0)),
                    float(mimic.get("offset", 0.0))) if mimic is not None else None)))
    children = {j.child for j in joints}
    world = next(link.get("name") for link in root.findall("link")
                 if link.get("name") not in children)
    ordered, known = [], {world}
    while len(ordered) < len(joints):          # parents before children
        for j in joints:
            if j not in ordered and j.parent in known:
                ordered.append(j)
                known.add(j.child)
    geometry = {}
    for link in root.findall("link"):
        items = []
        for col in link.findall("collision"):
            g, origin = col.find("geometry"), _origin(col.find("origin"))
            if g.find("mesh") is not None:
                items.append(("mesh", geo_dir / g.find("mesh").get("filename"), origin))
            elif g.find("box") is not None:
                items.append(("box", np.array([float(v) for v in g.find("box").get("size").split()]),
                              origin))
            elif g.find("cylinder") is not None:
                c = g.find("cylinder")
                items.append(("cylinder", (float(c.get("radius")), float(c.get("length"))), origin))
        if items:
            geometry[link.get("name")] = items
    hashes = {"urdf_sha256": hashlib.sha256(urdf_text.encode()).hexdigest(),
              "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest()}
    return HandDescription(config_name=config_name, config=json.loads(cfg_text),
                           joints=tuple(ordered), link_geometry=geometry, world_link=world,
                           hashes=hashes)


# ------------------------------------------------------------------ pieces
def _polytope(points: np.ndarray, planes: np.ndarray | None = None):
    """Hull vertices and outward planes ``[n, d]`` (inside: ``n.x + d <= 0``)."""
    hull = ConvexHull(points)
    return points[hull.vertices], np.asarray(hull.equations), hull


def kdop(vertices: np.ndarray):
    """The 26-DOP around ``vertices``, in their principal axes: ``(vertices, planes)``.

    Encloses every input point, so it can only over-state the shape.
    """
    centre = vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(vertices - centre, full_matrices=False)
    directions = _DOP_DIRECTIONS @ axes                 # face normals in the link frame
    support = (vertices @ directions.T).max(axis=0)
    hs = HalfspaceIntersection(np.column_stack([directions, -support]), centre)
    corners = np.unique(np.round(hs.intersections, 12), axis=0)
    verts, planes, _ = _polytope(corners)
    return verts, planes


def _box_corners(size: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                    dtype=float) * (np.asarray(size) / 2.0)


def _prism_corners(radius: float, length: float, sides: int = PRISM_SIDES) -> np.ndarray:
    """A prism circumscribing a cylinder along ``z`` (its faces touch the cylinder)."""
    r = radius / np.cos(np.pi / sides)
    th = np.arange(sides) * 2.0 * np.pi / sides
    ring = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return np.vstack([np.column_stack([ring, np.full(sides, z)])
                      for z in (-length / 2.0, length / 2.0)])


def _transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class HandPiece:
    """One convex piece of the hand. Duck-types :class:`tpgpt.hardware.hand.HandBox`."""

    name: str
    role: str                           # "body" or "finger"
    chain: str                          # "body", "thumb", "index", ...
    vertices: np.ndarray                # (v, 3)
    planes: np.ndarray                  # (f, 4), inside when n.x + d <= 0

    def corners(self) -> np.ndarray:
        return self.vertices

    @property
    def centre(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Which points lie inside, the piece grown by ``margin`` (negative shrinks)."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return np.all(p @ self.planes[:, :3].T + self.planes[:, 3] <= margin, axis=1)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "HandPiece":
        n = self.planes[:, :3] @ R.T
        return HandPiece(self.name, self.role, self.chain, self.vertices @ R.T + t,
                         np.column_stack([n, self.planes[:, 3] - n @ t]))

    def edges(self) -> list:
        """Index pairs of the polytope's edges (between non-coplanar faces), for drawing."""
        hull = ConvexHull(self.vertices)
        eq = hull.equations
        out = set()
        for f, (simplex, nbrs) in enumerate(zip(hull.simplices, hull.neighbors)):
            for k in range(3):
                g = nbrs[k]
                if np.allclose(eq[f], eq[g], atol=1e-9):
                    continue                       # an edge inside a flat face
                a, b = simplex[(k + 1) % 3], simplex[(k + 2) % 3]
                out.add((min(a, b), max(a, b)))
        # hull.simplices index hull.points == self.vertices
        return sorted(out)


def _link_pieces(desc: HandDescription) -> dict:
    """Every piece in its own link's frame: ``link -> [HandPiece]``."""
    chain_of_link = {link: chain for chain, links in FINGER_LINKS.items() for link in links}
    out: dict = {}
    for link, items in desc.link_geometry.items():
        pieces = []
        for k, (kind, data, origin) in enumerate(items):
            if kind == "mesh":
                pts = _transform(_read_obj(Path(data)), origin)
                verts, planes = kdop(pts)
            elif kind == "box":
                verts, planes, _ = _polytope(_transform(_box_corners(data), origin))
            else:
                verts, planes, _ = _polytope(_transform(_prism_corners(*data), origin))
            if link == "hand_base_link":
                name, role, chain = BASE_PIECE_NAMES[k], "body", "body"
            elif link in BODY_MESH_LINKS:
                name, role, chain = BODY_MESH_LINKS[link], "body", "body"
            else:
                name, role, chain = link, "finger", chain_of_link[link]
            pieces.append(HandPiece(name, role, chain, verts, planes))
        out[link] = pieces
    return out


@lru_cache(maxsize=8)
def _cached_link_pieces(config_name: str, geometry_name: str) -> dict:
    return _link_pieces(load_description(config_name, geometry_name))


@lru_cache(maxsize=64)
def _mesh_vertices(config_name: str, link: str) -> np.ndarray:
    """A mesh link's collision-mesh vertices in its own frame (its collision origin applied)."""
    desc = load_description(config_name)
    items = [(path, origin) for kind, path, origin in desc.link_geometry.get(link, [])
             if kind == "mesh"]
    if not items:
        raise ValueError(f"{link!r} has no collision mesh")
    return np.vstack([_transform(_read_obj(Path(path)), origin) for path, origin in items])


# ------------------------------------------------------------------ joint map
def joint_map_from_config(cfg: dict | None) -> dict:
    """``{dof: (q_at_0, q_at_1000, verified)}`` from ``inspire_hand.yaml``'s
    ``joint_map`` block (the hand node's :func:`joint_map_from_config`, same rule)."""
    cfg = (cfg or {}).get("inspire_hand", cfg or {})
    block = cfg.get("joint_map", {}) or {}
    out = {}
    for name in DOF_NAMES:
        entry = block.get(name)
        if entry is None:
            q0, q1000 = DEFAULT_JOINT_MAP[name]
            out[name] = (float(q0), float(q1000), False)
        else:
            out[name] = (float(entry["q_at_0"]), float(entry["q_at_1000"]),
                         bool(entry.get("verified", False)))
    return out


# ------------------------------------------------------------------ the model
@dataclass
class ClosedState:
    """Where each chain stops closing onto an object (:meth:`InspireHandModel.closed_onto`)."""

    q: np.ndarray                       # (6,) DOF order
    contact: dict                       # chain -> bool
    fraction: dict                      # chain -> 0..1 of the way from open to close

    @property
    def fingers_in_contact(self) -> list:
        return [c for c in ("pinky", "ring", "middle", "index") if self.contact.get(c)]

    @property
    def opposed(self) -> bool:
        """The thumb and at least one finger reach the object."""
        return bool(self.contact.get("thumb")) and bool(self.fingers_in_contact)

    def as_dict(self) -> dict:
        return {"q": np.asarray(self.q).tolist(), "contact": dict(self.contact),
                "fraction": {k: float(v) for k, v in self.fraction.items()},
                "opposed": self.opposed}


@dataclass
class InspireHandModel:
    """The hand's pieces at any joint vector, in the grasp frame ``G``.

    Args:
        description: :func:`load_description`'s result.
        tool_depth: ``W``'s origin to ``G``'s along ``+Z``, m. ``None`` = the
            description's ``fingertip`` depth (0.15 m).
        joint_map: :func:`joint_map_from_config`'s result; ``None`` = the defaults.
    """

    description: HandDescription = field(default_factory=load_description)
    tool_depth: float | None = None
    joint_map: dict | None = None

    def __post_init__(self):
        if self.tool_depth is None:
            self.tool_depth = self.description.fingertip_depth
        if self.joint_map is None:
            self.joint_map = joint_map_from_config({})
        self._pieces = _cached_link_pieces(self.description.config_name, GRASPGEN_NAME)

    # ------------------------------------------------------------- joints
    @property
    def open_q(self) -> np.ndarray:
        """Where the hand's open command puts each DOF: fingers and thumb bend at 1000,
        thumb rotation at 0 (opposed) -- ``inspire_hand.OPEN_POSE`` -- through the joint
        map. GraspGen-X's own ``open`` (all zero, yaw 1.308) is
        ``description.open_q``; they differ by the 0.03 rad the fingers' 1000 is short of
        the URDF's straight, and by the thumb's 0.12 rad bend at 1000 (the ruler, 2026-10-05:
        the real thumb is never as straight as GraspGen-X's open)."""
        return self.registers_to_q([1000, 1000, 1000, 1000, 1000, 0])

    @property
    def close_q(self) -> np.ndarray:
        """Where the hand's close command drives each DOF if nothing stops it: every
        register at 0 (``inspire_hand.CLOSE_POSE``), through the joint map."""
        return self.registers_to_q([0, 0, 0, 0, 0, 0])

    def registers_to_q(self, values) -> np.ndarray:
        return np.array([self.joint_map[n][0] + (self.joint_map[n][1] - self.joint_map[n][0])
                         * float(u) / 1000.0 for n, u in zip(DOF_NAMES, values)])

    def q_to_registers(self, q) -> np.ndarray:
        out = []
        for n, value in zip(DOF_NAMES, q):
            q0, q1000 = self.joint_map[n][0], self.joint_map[n][1]
            out.append(int(round(np.clip(1000.0 * (float(value) - q0) / (q1000 - q0),
                                         0.0, 1000.0))))
        return np.array(out)

    def joint_values(self, q) -> dict:
        """Every URDF joint's angle (mimics applied) for driven angles ``q`` (DOF order)."""
        values = dict(zip(DRIVEN_JOINTS, (float(v) for v in q)))
        for j in self.description.joints:
            if j.mimic is not None:
                driver, mult, off = j.mimic
                values[j.name] = mult * values[driver] + off
        return values

    def link_poses(self, q) -> dict:
        """``link -> T_G<-link`` (4x4) for driven angles ``q`` (or ``"open"``/``"close"``)."""
        values = self.joint_values(self._resolve(q))
        T = {self.description.world_link: np.eye(4)}
        T[self.description.world_link][2, 3] = -float(self.tool_depth)
        for j in self.description.joints:
            Tj = T[j.parent] @ j.origin
            angle = values.get(j.name, 0.0)
            if angle and np.any(j.axis):
                R = np.eye(4)
                R[:3, :3] = Rotation.from_rotvec(j.axis * angle).as_matrix()
                Tj = Tj @ R
            T[j.child] = Tj
        return T

    def _resolve(self, q) -> np.ndarray:
        if isinstance(q, str):
            return {"open": self.open_q, "close": self.close_q}[q]
        return np.asarray(q, dtype=float).reshape(6)

    def pieces(self, q="open", chains=None) -> list:
        """The posed pieces in ``G``. ``q``: a joint vector, ``"open"`` or ``"close"``;
        ``chains``: only these chains (``"body"``, ``"thumb"``, ``"index"``, ...)."""
        T = self.link_poses(self._resolve(q))
        out = []
        for link, pieces in self._pieces.items():
            for p in pieces:
                if chains is not None and p.chain not in chains:
                    continue
                out.append(p.transformed(T[link][:3, :3], T[link][:3, 3]))
        return out

    def sweep_boxes(self) -> list:
        """The description's two sweep boxes in ``G``: ``[(lo, hi), (lo, hi)]``."""
        shift = np.array([0.0, 0.0, float(self.tool_depth)])
        return [(lo - shift, hi - shift) for lo, hi in self.description.sweep_boxes()]

    def fingertip_points(self, q="open") -> dict:
        """``chain -> the point of its last link furthest from that link's joint``, in G."""
        T = self.link_poses(self._resolve(q))
        out = {}
        for chain, links in FINGER_LINKS.items():
            last = links[-1]
            piece = self._pieces[last][0]
            far = piece.vertices[np.argmax(np.linalg.norm(piece.vertices, axis=1))]
            out[chain] = T[last][:3, :3] @ far + T[last][:3, 3]
        return out

    def manual_angles(self, q) -> dict:
        """The RH56 manual's angle of each DOF (section 2.4.11), degrees, at ``q``.

        * ``alpha`` (each finger): at the knuckle ``P1`` (the proximal joint), between
          the palm line towards the wrist and ``P1`` -> fingertip, both seen in the
          finger's bending plane (the manual's side view; splay is not a bend);
          176.7 straight .. 19 curled.
        * ``theta`` (thumb bend): in the thumb's bending plane, between the thumb
          rotation axis (pointing to the fingertips) and the thumb's base -> tip,
          signed; 53.6 straight .. -13 bent.
        * ``beta`` (thumb rotation): seen along the hand's axis, between the palm's
          across line and the thumb's base -> tip; 165 swung out .. 90 opposed. This
          construction reads about 6 deg above the manual's at both ends (96 / 171 at
          the URDF's limits): the manual draws it from a point the URDF does not name.

        The fingertip is :meth:`fingertip_points`' point. These are what a protractor
        on the real hand measures, so the register map can be checked against them.
        """
        q = self._resolve(q)
        T = self.link_poses(q)
        tips = self.fingertip_points(q)
        H = T["hand_base_link"][:3, :3]

        def angle(u, v):
            return float(np.degrees(np.arccos(np.clip(
                u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))

        out = {}
        for dof, chain in CHAIN_OF_DOF.items():
            if chain == "thumb":
                continue
            knuckle = T[FINGER_LINKS[chain][0]]
            hinge = knuckle[:3, 2]                  # the proximal joint's axis
            v = tips[chain] - knuckle[:3, 3]
            palm = H[:, 1]
            # In the finger's own bending plane, as the manual's side view shows it:
            # the finger's splay away from the palm line is not a bend.
            out[dof] = angle(v - hinge * (v @ hinge), palm - hinge * (palm @ hinge))
        bend = T["thumb_proximal"][:3, 2]
        axis = T["thumb_proximal_base"][:3, :3] @ np.array([0.0, 0.0, 1.0])
        v = tips["thumb"] - T["thumb_proximal"][:3, 3]
        v_in, a_in = v - bend * (v @ bend), axis - bend * (axis @ bend)
        out["thumb_bend"] = -float(np.sign(np.cross(a_in, v_in) @ bend)) * angle(v_in, a_in)
        top = v - H[:, 1] * (v @ H[:, 1])
        out["thumb_rot"] = angle(top, H[:, 2])
        return out

    def mesh_vertices(self, link: str, q="open") -> np.ndarray:
        """``link``'s **exact** collision-mesh vertices in ``G`` at ``q`` -- not its 26-DOP,
        which over-states it by up to 4.6 mm. For measuring by touch (where the hand
        touches is a vertex of the real shape)."""
        T = self.link_poses(self._resolve(q))[link]
        return _transform(_mesh_vertices(self.description.config_name, link), T)

    def lowest_z(self, R: np.ndarray, p: np.ndarray, q="open") -> float:
        """World ``z`` of the hand's lowest vertex with ``G`` at rotation ``R``, origin ``p``."""
        verts = np.vstack([piece.vertices for piece in self.pieces(q)])
        return float((verts @ np.asarray(R).T + np.asarray(p))[:, 2].min())

    # ------------------------------------------------------------- closing
    def closed_onto(self, points, target=None, steps: int = 16, refine: int = 6) -> ClosedState:
        """Close each chain from open towards ``target`` (default :attr:`close_q`) until
        it holds :data:`CONTACT_POINTS` of ``points`` (in ``G``) :data:`CONTACT_DEPTH`
        deep. The thumb rotation stays at its open value."""
        all_pts = np.atleast_2d(np.asarray(points, dtype=float))
        target = self.close_q if target is None else self._resolve(target)
        q = self.open_q.copy()
        contact, fraction = {}, {}
        open_poses = self.link_poses(self.open_q)
        for k, dof in enumerate(DOF_NAMES[:5]):
            chain = CHAIN_OF_DOF[dof]
            # The chain turns about its first joint, so nothing further from that joint
            # than its furthest vertex can ever be touched: test only the points within.
            pivot = open_poses[FINGER_LINKS[chain][0]][:3, 3]
            reach = max(float(np.linalg.norm(p.vertices - pivot, axis=1).max())
                        for p in self.pieces(self.open_q, chains=(chain,)))
            pts = all_pts[np.linalg.norm(all_pts - pivot, axis=1) <= reach] if len(all_pts) \
                else all_pts

            def touches(s, k=k, chain=chain, pts=pts):
                qq = self.open_q.copy()
                qq[k] = self.open_q[k] + s * (target[k] - self.open_q[k])
                inside = np.zeros(len(pts), dtype=bool)
                for piece in self.pieces(qq, chains=(chain,)):
                    inside |= piece.contains(pts, -CONTACT_DEPTH)
                return int(inside.sum()) >= CONTACT_POINTS

            prev, hit = 0.0, None
            if len(pts) and touches(0.0):
                hit = 0.0
            else:
                for s in np.linspace(0.0, 1.0, steps + 1)[1:]:
                    if len(pts) and touches(s):
                        hit = s
                        break
                    prev = s
            if hit is not None and hit > 0.0:
                lo, hi = prev, hit
                for _ in range(refine):
                    mid = (lo + hi) / 2.0
                    lo, hi = (lo, mid) if touches(mid) else (mid, hi)
                hit = lo                              # the last pose still clear
            s = 1.0 if hit is None else hit
            q[k] = self.open_q[k] + s * (target[k] - self.open_q[k])
            contact[chain], fraction[chain] = hit is not None, float(s)
        q[5] = self.open_q[5]
        return ClosedState(q=q, contact=contact, fraction=fraction)


@lru_cache(maxsize=4)
def default_model(tool_depth: float | None = None) -> InspireHandModel:
    return InspireHandModel(tool_depth=tool_depth)


#: The palm's surface in ``hand_base_link``: the palm box's face on the palm side
#: (``body_palm``: centre x -0.0032, 39.1 mm thick, palm facing -X).
PALM_SURFACE_X = -0.0032 - 0.0391 / 2.0


def landmarks(model: InspireHandModel, q) -> dict:
    """Distances a caliper can take on the real hand, mm, as the model predicts them.

    * ``<finger>_length``: the fingertip's distance from the wrist face along the hand's
      axis (the hand's length to that tip, from the face that bolts to the adapter);
    * ``<finger>_to_palm``: the fingertip's height above the palm surface (negative =
      curled past it);
    * ``thumb_to_<finger>``: thumb tip to fingertip, straight line;
    * ``knuckles_index_to_pinky``: the index and little fingers' knuckle axes, apart.

    The fingertip is :meth:`InspireHandModel.fingertip_points`' point -- the end of the
    last pad. Measure at the same point on the hand.
    """
    q = model._resolve(q)
    T = model.link_poses(q)
    H = T["hand_base_link"]
    tips = model.fingertip_points(q)
    local = {c: (p - H[:3, 3]) @ H[:3, :3] for c, p in tips.items()}      # in H
    out = {}
    for chain in ("index", "middle", "ring", "pinky", "thumb"):
        out[f"{chain}_length"] = -local[chain][1] * 1000.0
    for chain in ("index", "middle", "ring", "pinky", "thumb"):
        out[f"{chain}_to_palm"] = (PALM_SURFACE_X - local[chain][0]) * 1000.0
    for chain in ("index", "middle", "ring", "pinky"):
        out[f"thumb_to_{chain}"] = float(np.linalg.norm(tips["thumb"] - tips[chain])) * 1000.0
    out["knuckles_index_to_pinky"] = float(np.linalg.norm(
        T["index_proximal"][:3, 3] - T["pinky_proximal"][:3, 3])) * 1000.0
    return out


def main(argv=None) -> int:
    """``python -m tpgpt.hardware.inspire_model landmarks --registers u0 .. u5``."""
    import argparse

    ap = argparse.ArgumentParser(description="The Inspire hand model's predictions.")
    ap.add_argument("command", choices=["landmarks", "angles"])
    ap.add_argument("--registers", nargs=6, type=int, default=[1000, 1000, 1000, 1000, 1000, 0],
                    metavar="U", help="little ring middle index thumb_bend thumb_rot, 0-1000")
    ap.add_argument("--config", default=None, help="inspire_hand.yaml for the joint map "
                    "(default: the source tree's)")
    args = ap.parse_args(argv)
    import yaml

    path = Path(args.config) if args.config else (
        Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config/inspire_hand.yaml")
    cfg = yaml.safe_load(path.read_text())
    model = InspireHandModel(joint_map=joint_map_from_config(cfg))
    q = model.registers_to_q(args.registers)
    print(f"registers {args.registers} -> q {np.round(q, 3).tolist()} rad "
          f"(joint map {'verified' if all(v[2] for v in model.joint_map.values()) else 'UNVERIFIED'}"
          f", {path})")
    values = landmarks(model, q) if args.command == "landmarks" else model.manual_angles(q)
    unit = "mm" if args.command == "landmarks" else "deg"
    for name, value in values.items():
        print(f"  {name:26s} {value:8.1f} {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHAIN_OF_DOF",
    "CONTACT_DEPTH",
    "CONTACT_POINTS",
    "ClosedState",
    "DOF_NAMES",
    "DRIVEN_JOINTS",
    "FINGER_LINKS",
    "GRASPGEN_NAME",
    "HandDescription",
    "HandPiece",
    "InspireHandModel",
    "default_model",
    "description_dir",
    "joint_map_from_config",
    "kdop",
    "load_description",
]
