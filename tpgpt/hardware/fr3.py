"""The FR3 arm as a kinematic chain with collision geometry, without a simulator.

The simulation answers two questions about every transported path with MuJoCo:
*can the arm hold each pose* (``tpgpt.sim.kinematics.solve_ik``) and *does any
part of it end up inside the scene* (``tpgpt.perception.obstacles.robot_bodies``
plus a forward pass). The robot computer has no MuJoCo, so this module answers
the same two questions from the robot's own description:

* **kinematics** from ``assets/fr3/fr3.urdf`` -- the URDF the real robot
  container publishes on ``/robot_description`` (joint origins, axes and the
  FR3's own joint limits, which differ from the Panda's the simulation uses);
* **collision geometry** from the same package's collision meshes, one convex
  hull per link. That is what MuJoCo itself collides for a mesh geom, so the
  arm is exactly as conservative here as it was in simulation. Measured on
  these meshes, the hull overstates a link's volume by 1.15x (link 7) to 1.78x
  (link 5, which is L-shaped).

:meth:`FR3.solve_ik` is ``solve_ik``'s damped-least-squares iteration, line for
line, with the same defaults, so "reachable" means the same thing on both sides.
The end effector's own geometry is **not** here: it comes from
:class:`tpgpt.hardware.end_effector.EndEffector`, which already describes both
hands the robot carries, and is placed at the TCP pose this module computes.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parent / "assets" / "fr3"

#: The seven actuated joints, base to flange.
ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))

#: The frame CRISP commands and ``/current_pose`` reports.
TCP_LINK = "fr3_hand_tcp"

#: Links whose collision hulls stand for the arm. The hand is the end effector's
#: business (``EndEffector.boxes``), since the Inspire hand replaces it.
ARM_LINKS = tuple(f"fr3_link{i}" for i in range(0, 8))

#: Franka's usual ready pose. Only a seed of last resort: a real plan seeds from
#: the measured ``/joint_states``.
HOME_Q = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])


# ------------------------------------------------------------------ geometry io
def read_stl(path: str | Path) -> np.ndarray:
    """Triangles of a binary STL as ``(n, 3, 3)`` float metres."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data[84:84 + 50 * n], dtype=record)["vertices"]
    return np.asarray(triangles, dtype=float).reshape(-1, 3, 3)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def _origin(element) -> np.ndarray:
    T = np.eye(4)
    if element is None:
        return T
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
    return T


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, dtype=float) * float(angle)).as_matrix()


# ------------------------------------------------------------------ data types
@dataclass(frozen=True)
class Joint:
    name: str
    kind: str                 # "revolute", "prismatic", "fixed"
    parent: str
    child: str
    origin: np.ndarray        # (4, 4) parent -> joint frame at zero
    axis: np.ndarray          # (3,) in the joint frame
    lower: float = -np.inf
    upper: float = np.inf


@dataclass(frozen=True)
class LinkHull:
    """One link's collision mesh as a convex hull, in the link's own frame.

    ``planes`` rows are ``[nx, ny, nz, d]`` with a point inside when
    ``n . x + d <= 0`` for every row -- the convention of
    :class:`tpgpt.perception.obstacles.ConvexBody`, and of scipy's
    ``ConvexHull.equations``.
    """

    name: str
    vertices: np.ndarray      # (v, 3) hull vertices
    planes: np.ndarray        # (f, 4)
    radius: float             # bounding sphere about the link origin

    def contains(self, points: np.ndarray, T_world_link: np.ndarray,
                 tolerance: float = 0.0) -> np.ndarray:
        """Which world points lie inside this hull with the link at ``T_world_link``.

        ``tolerance`` shrinks the hull (positive) or grows it (negative).
        """
        R, t = T_world_link[:3, :3], T_world_link[:3, 3]
        local = (np.atleast_2d(points) - t) @ R
        return np.all(local @ self.planes[:, :3].T + self.planes[:, 3] <= -tolerance, axis=1)

    def world_vertices(self, T_world_link: np.ndarray) -> np.ndarray:
        return self.vertices @ T_world_link[:3, :3].T + T_world_link[:3, 3]


@dataclass
class IKResult:
    """Outcome of one IK solve; the fields of ``tpgpt.sim.kinematics.IKResult``."""

    reachable: bool
    qpos: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int
    at_joint_limit: bool = False


def hull_from_mesh(name: str, triangles: np.ndarray) -> LinkHull:
    vertices = triangles.reshape(-1, 3)
    hull = ConvexHull(vertices)
    points = vertices[hull.vertices]
    return LinkHull(name=name, vertices=points, planes=np.asarray(hull.equations),
                    radius=float(np.linalg.norm(points, axis=1).max()))


# ------------------------------------------------------------------ the model
class FR3:
    """Forward and inverse kinematics of the FR3 to ``fr3_hand_tcp``, and its link hulls."""

    def __init__(self, urdf: str | Path = ASSETS / "fr3.urdf",
                 meshes: str | Path = ASSETS / "collision"):
        root = ET.fromstring(_urdf_text(urdf))
        joints = {}
        for j in root.findall("joint"):
            axis = j.find("axis")
            limit = j.find("limit")
            joints[j.find("child").get("link")] = Joint(
                name=j.get("name"), kind=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_origin(j.find("origin")),
                axis=(np.array([float(v) for v in axis.get("xyz").split()])
                      if axis is not None else np.zeros(3)),
                lower=float(limit.get("lower", -np.inf)) if limit is not None else -np.inf,
                upper=float(limit.get("upper", np.inf)) if limit is not None else np.inf,
            )
        # The chain from the base to the TCP, walked backwards from the TCP.
        chain, link = [], TCP_LINK
        while link in joints:
            chain.append(joints[link])
            link = joints[link].parent
        self.base = link
        self.chain: list[Joint] = chain[::-1]
        arm = [j for j in self.chain if j.kind == "revolute"]
        if tuple(j.name for j in arm) != ARM_JOINTS:
            raise ValueError(f"unexpected FR3 chain: {[j.name for j in arm]}")
        self.lower = np.array([j.lower for j in arm])
        self.upper = np.array([j.upper for j in arm])
        meshes = Path(meshes)
        self.hulls = {
            name: hull_from_mesh(name, read_stl(meshes / f"link{name[-1]}.stl"))
            for name in ARM_LINKS
        }

    # ----------------------------------------------------------- forward
    def link_poses(self, q) -> dict[str, np.ndarray]:
        """World (``base``) pose of every link on the chain, and of each joint frame.

        Returns ``{link: T}``; joint frames (after the joint's fixed origin,
        before its motion) are under ``"<joint>/frame"``, which is where the
        Jacobian's axes live.
        """
        q = np.asarray(q, dtype=float).reshape(7)
        T = np.eye(4)
        poses = {self.base: T.copy()}
        k = 0
        for joint in self.chain:
            T = T @ joint.origin
            poses[f"{joint.name}/frame"] = T.copy()
            if joint.kind == "revolute":
                motion = np.eye(4)
                motion[:3, :3] = _axis_rotation(joint.axis, q[k])
                T = T @ motion
                k += 1
            poses[joint.child] = T.copy()
        return poses

    def tcp(self, q) -> np.ndarray:
        """``fr3_hand_tcp`` in ``base``, ``(4, 4)``."""
        return self.link_poses(q)[TCP_LINK]

    def jacobian(self, q, poses: dict | None = None) -> np.ndarray:
        """Geometric Jacobian of ``fr3_hand_tcp``, ``(6, 7)``: linear rows, then angular."""
        poses = poses if poses is not None else self.link_poses(q)
        p_tcp = poses[TCP_LINK][:3, 3]
        J = np.zeros((6, 7))
        for k, name in enumerate(ARM_JOINTS):
            joint = next(j for j in self.chain if j.name == name)
            frame = poses[f"{name}/frame"]
            z = frame[:3, :3] @ joint.axis
            J[:3, k] = np.cross(z, p_tcp - frame[:3, 3])
            J[3:, k] = z
        return J

    # ----------------------------------------------------------- inverse
    def solve_ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        seed=None,
        max_iterations: int = 120,
        position_tolerance: float = 5e-3,
        rotation_tolerance: float = 0.15,
        damping: float = 0.08,
        step_scale: float = 0.6,
    ) -> IKResult:
        """Can ``fr3_hand_tcp`` be put at this pose?

        ``tpgpt.sim.kinematics.solve_ik`` transcribed: damped least squares on
        the seven joints, each step clamped to the joint limits, convergence
        tested at the top of every iteration against both tolerances, and the
        same defaults. ``rotation=None`` solves for position alone.
        """
        q = np.array(HOME_Q if seed is None else seed, dtype=float).reshape(7)
        target = np.asarray(position, dtype=float).reshape(3)
        at_limit = False
        position_error = rotation_error = float("inf")
        for iteration in range(1, max_iterations + 1):
            poses = self.link_poses(q)
            T = poses[TCP_LINK]
            error = target - T[:3, 3]
            residual, rows = [error], [slice(0, 3)]
            rotation_error = 0.0
            if rotation is not None:
                angular = Rotation.from_matrix(
                    np.asarray(rotation, dtype=float) @ T[:3, :3].T).as_rotvec()
                rotation_error = float(np.linalg.norm(angular))
                residual.append(angular)
                rows.append(slice(3, 6))
            position_error = float(np.linalg.norm(error))
            if position_error < position_tolerance and rotation_error < rotation_tolerance:
                return IKResult(True, q, position_error, rotation_error, iteration, at_limit)
            J_full = self.jacobian(q, poses)
            J = np.vstack([J_full[r] for r in rows])
            e = np.concatenate(residual)
            delta = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
            stepped = q + step_scale * delta
            q = np.clip(stepped, self.lower, self.upper)
            at_limit = at_limit or bool(np.any(np.abs(q - stepped) > 1e-9))
        return IKResult(False, q, position_error, rotation_error, max_iterations, at_limit)

    # ----------------------------------------------------------- geometry
    def posed_hulls(self, q=None, poses: dict | None = None) -> list[tuple[LinkHull, np.ndarray]]:
        """``(hull, T_world_link)`` for every arm link at configuration ``q``."""
        poses = poses if poses is not None else self.link_poses(q)
        return [(self.hulls[name], poses[name]) for name in ARM_LINKS]


def _urdf_text(path: str | Path) -> str:
    """The URDF, cut at ``</robot>``: ``ros2 topic echo`` appends a ``---`` line."""
    text = Path(path).read_text()
    end = text.find("</robot>")
    return text if end < 0 else text[: end + len("</robot>")]


@lru_cache(maxsize=1)
def default_model() -> FR3:
    """The vendored FR3, built once per process (parsing and hulling take ~50 ms)."""
    return FR3()


__all__ = ["ARM_JOINTS", "ARM_LINKS", "FR3", "HOME_Q", "IKResult", "LinkHull", "TCP_LINK",
           "default_model", "read_stl"]
