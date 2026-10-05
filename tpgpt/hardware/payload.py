"""What hangs on the FR3's flange, for Desk's end-effector settings (plan M5).

Desk tells the robot the mass, centre of mass and inertia of what is bolted to the flange,
and the robot subtracts that weight everywhere: in its gravity compensation (an error makes
a held arm sag and a floating arm drift) and in its estimate of external forces (an error
reads as a constant push, which is what the collision reflexes watch). Desk is still set
for the Franka Hand (:data:`FRANKA_HAND_DESK`); this module gives the values for the hand
that is on it now, two ways:

* **from the manufacturer's data** (:func:`inspire_cad_payload`): the Inspire RH56DFX-2R's
  links as its own URDF gives them (:data:`INSPIRE_HAND_INERTIAL`, 533 g), placed by the
  measured mount, plus the printed mount as a disc of an estimated mass;
* **by weighing it with the robot** (:func:`fit_payload`): at rest in several poses the
  robot's estimate of the external joint torques, ``tau_ext``, is what its model does not
  explain -- here the payload it is not told about. Holding still, the unmodelled payload
  pushes on the flange with its weight error,

      f = (m - m0) g,       tau_F = (R s - R s0) x g,    s = m c (the first moment),

  in the base frame (``g`` = (0, 0, -9.81), ``R`` the flange's orientation, ``c`` the
  centre of mass in the flange frame, ``m0, c0`` Desk's values), and the joints see
  ``tau_ext = J_F^T [f; tau_F] + b``, with ``J_F`` the flange's geometric Jacobian and
  ``b`` a constant offset per joint (the torque sensors': ~0.2 N m was seen at rest).
  **And static friction** (the first weighing, 2026-10-05): under the soft joint
  controller every joint stops 0.05-0.08 rad short of its target, always on the side it
  came from, and friction then holds up to ~0.6 N m that the robot reads as external --
  more than the payload's whole effect. Friction pushes back along the joint's last
  motion and the payload does not care which way it came, so the fit adds ``+-f_j``
  signed by each joint's approach (``approach``: the side it stopped on, sign(target - q);
  without it the first weighing's residuals were 0.13-0.53 N m, with it 0.02-0.08). All
  linear in ``(m, s, b, f)``: 18 unknowns, 7 equations per pose. The poses must turn the
  hand through different directions relative to gravity; a dozen around home do
  (:data:`POSES`). The static data cannot give the inertia: that comes from the
  manufacturer's data either way.

The sign of the robot's external torques flips the payload part of the fit, not the
friction. **Measured** (push test, 2026-10-05, ``wrench_watch``): pushing the hand up made
``external_wrench_in_base_frame``'s Fz go down (-0.43 -> -2.38 N), so the robot reports
the force IT applies to what touches it, the reverse of the environment's push
(:data:`TAU_EXT_SIGN`). It agrees with the first weighing: the other sign put this hand's
centre of mass 7 mm from the flange, where none of its mass is.

numpy only (the FR3 model for the Jacobian); no ROS. The robot side is the node
``payload_measure``.

    python -m tpgpt.hardware.payload cad          # Desk's values from the manufacturer's data
    python -m tpgpt.hardware.payload poses        # the measuring poses, checked for collisions
    python -m tpgpt.hardware.payload fit ~/inspire_mount/payload_<time>.json
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

G = np.array([0.0, 0.0, -9.81])
#: The robot's external torques/wrench are minus the environment's push on it (the push
#: test, module docstring): +1 would mean the environment's push itself.
TAU_EXT_SIGN = -1.0


@dataclass
class Payload:
    """A rigid payload on the flange: ``mass`` kg, ``com`` m in ``fr3_link8``, ``inertia``
    kg m^2 about the centre of mass in ``fr3_link8``'s axes (3x3)."""

    mass: float
    com: tuple
    inertia: tuple = ((0.0,) * 3,) * 3

    def first_moment(self) -> np.ndarray:
        return self.mass * np.asarray(self.com, dtype=float)

    def desk_text(self) -> str:
        """What to type into Desk (Settings > End effector), with the robot's units."""
        c, i = np.asarray(self.com), np.asarray(self.inertia)
        return (f"  mass          {self.mass:.3f} kg\n"
                f"  centre of mass  x {c[0]:+.4f}  y {c[1]:+.4f}  z {c[2]:+.4f}  m (flange frame)\n"
                "  inertia (kg m^2, about the centre of mass, flange axes):\n"
                + "\n".join("    " + "  ".join(f"{v:+.2e}" for v in row) for row in i))


#: What Desk holds now: the Franka Hand (franka_description's hand, Desk's default).
FRANKA_HAND_DESK = Payload(0.73, (-0.01, 0.0, 0.03),
                           ((0.001, 0.0, 0.0), (0.0, 0.0025, 0.0), (0.0, 0.0, 0.0017)))

#: The Inspire RH56DFX-2R from the manufacturer's URDF (``urdf_right_2025_4_18``, the plain
#: no-wrist right hand; :func:`vendor_hand_inertial` recomputes it), joints at 0 (open),
#: in ``hand_base_link`` (``H``: the base ring's face, -y along the fingers, -x the palm's
#: normal). Its base frame maps to ``H`` as x_H = -x, y_H = -z, z_H = -y (the knuckles and
#: the thumb base line up so; the ring is centred on both to 0.0 mm).
INSPIRE_HAND_INERTIAL = {
    "mass": 0.5332,
    "com_H": (-0.0059, -0.0812, -0.0053),
    "inertia_H": ((1.468e-03, -2.094e-05, -7.823e-05),
                  (-2.094e-05, 4.728e-04, -9.327e-06),
                  (-7.823e-05, -9.327e-06, 1.163e-03)),
}
#: The vendor base frame -> H.
VENDOR_TO_H = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
#: The printed mount: a disc as wide as the flange, its mass a guess (PLA, partly filled,
#: plus screws and tape): the robot's weighing measures the total.
MOUNT_MASS_GUESS = 0.035
MOUNT_RADIUS = 0.0315


def vendor_hand_inertial(urdf_path) -> dict:
    """Mass, centre of mass and inertia about it, in ``H``, from the manufacturer's URDF
    with every joint at 0. Returns ``{"mass", "com_H", "inertia_H"}``."""
    import xml.etree.ElementTree as ET

    from scipy.spatial.transform import Rotation

    def T_of(o):
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = Rotation.from_euler(
                "xyz", [float(v) for v in (o.get("rpy") or "0 0 0").split()]).as_matrix()
            T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        return T

    root = ET.parse(str(Path(urdf_path).expanduser())).getroot()
    child_of = {j.find("child").get("link"): j for j in root.iter("joint")}

    def pose(name):
        if name not in child_of:
            return np.eye(4)
        j = child_of[name]
        return pose(j.find("parent").get("link")) @ T_of(j.find("origin"))

    bodies = []
    for link in root.iter("link"):
        i = link.find("inertial")
        if i is None:
            continue
        T = pose(link.get("name")) @ T_of(i.find("origin"))
        e = i.find("inertia")
        I = np.array([[float(e.get("ixx")), float(e.get("ixy")), float(e.get("ixz"))],
                      [float(e.get("ixy")), float(e.get("iyy")), float(e.get("iyz"))],
                      [float(e.get("ixz")), float(e.get("iyz")), float(e.get("izz"))]])
        bodies.append((float(i.find("mass").get("value")), T[:3, 3], T[:3, :3] @ I @ T[:3, :3].T))
    m = sum(b[0] for b in bodies)
    c = sum(b[0] * b[1] for b in bodies) / m
    I = sum(b[2] + b[0] * (np.dot(b[1] - c, b[1] - c) * np.eye(3) - np.outer(b[1] - c, b[1] - c))
            for b in bodies)
    R = VENDOR_TO_H
    return {"mass": m, "com_H": tuple((R @ c).tolist()), "inertia_H": (R @ I @ R.T).tolist()}


def _box_inertia(mass, size) -> np.ndarray:
    a, b, c = size
    return mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def inspire_cad_payload(end_effector, mount_mass: float = MOUNT_MASS_GUESS,
                        hand: dict | None = None) -> Payload:
    """The hand and its mount as Desk wants them, from the manufacturer's data placed by
    ``end_effector``'s mount (``inspire_from_config``). Without the vendor inertia, the
    hand is a uniform 55 x 218 x 81 mm box (palm normal, along, across) about its centre of
    mass: the inertia matters little at the speeds the arm moves."""
    hand = dict(hand or INSPIRE_HAND_INERTIAL)
    spec = end_effector.model_spec
    from tpgpt.hardware.end_effector import flange_from_hand_base

    T_FH = flange_from_hand_base(spec["mount_yaw_deg"], spec["adapter_thickness"],
                                 spec.get("lateral", (0.0, 0.0)))
    R, t = T_FH[:3, :3], T_FH[:3, 3]
    m_h = float(hand["mass"])
    c_h = R @ np.asarray(hand["com_H"]) + t
    I_h = R @ (np.asarray(hand["inertia_H"]) if "inertia_H" in hand
               else _box_inertia(m_h, (0.055, 0.218, 0.081))) @ R.T
    a = float(spec["adapter_thickness"])
    m_m, c_m = float(mount_mass), np.array([0.0, 0.0, a / 2.0])
    r = MOUNT_RADIUS
    I_m = m_m * np.diag([(3 * r * r + a * a) / 12.0, (3 * r * r + a * a) / 12.0, r * r / 2.0])
    m = m_h + m_m
    c = (m_h * c_h + m_m * c_m) / m

    def shift(mass, d):
        return mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    I = I_h + shift(m_h, c_h - c) + I_m + shift(m_m, c_m - c)
    return Payload(m, tuple(c.tolist()), tuple(map(tuple, I.tolist())))


# ----------------------------------------------------------------------- the poses
#: Joint offsets from home (rad, joints 1-7) the robot holds still at while it weighs: the
#: hand turned about its axis (7), tilted both ways (6, 5), the elbow and shoulder moved
#: (4, 2, 3), and combinations, so gravity pulls on the hand from many directions. Each
#: offset is within joint_target_publisher's 0.8 rad, every pose 0.35 rad inside the limits,
#: and every pose clears the table and the shelf (``poses``; a test keeps it so).
POSES = [
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0.8),
    (0, 0, 0, 0, 0, 0, -0.8),
    (0, 0, 0, 0, 0, 0.6, 0),
    (0, 0, 0, 0, 0, -0.6, 0),
    (0, 0, 0, 0, 0.6, 0, 0),
    (0, 0, 0, 0, -0.6, 0, 0),
    (0, 0, 0, 0.5, 0, -0.5, 0),
    (0, 0.4, 0, -0.3, 0, 0, 0),
    (0, 0, 0, 0, 0.6, 0, 0.8),
    (0, 0, 0, 0, 0, 0.5, -0.8),
    (0, 0, 0.5, 0, 0, 0, 0),
    # The arm itself reaching and folding: the flange's lever arm about joints 2, 4 and 6
    # changes, which is what tells the mass from a constant torque offset (+-4 g at 0.02 N m
    # of noise per joint, against +-10 without these).
    (0, 0.4, 0, 0.4, 0, -0.4, 0),
    (0, -0.4, 0, -0.3, 0, 0.4, 0),
    (0, 0.5, 0, 0, 0, -0.5, 0),
    (0, -0.4, 0, 0.5, 0, 0, 0),
    (0, 0, 0, 0.6, 0, 0, 0),
    (0, 0, 0, -0.3, 0, 0.6, 0),
]


def pose_clearances(end_effector, poses=POSES, home=None, shelf_config: dict | None = None,
                    table: float | None = None, fr3=None) -> list[dict]:
    """Per pose: the lowest robot point above the table and the nearest robot vertex
    outside the shelf model (m; the arm's hulls and the open hand)."""
    from tpgpt.hardware.fr3 import HOME_Q, default_model
    from tpgpt.hardware.path_check import robot_pieces
    from tpgpt.hardware.shelf import ShelfModel

    fr3 = fr3 or default_model()
    home = np.asarray(HOME_Q if home is None else home, dtype=float)
    shelf = ShelfModel.from_config(shelf_config) if shelf_config else None
    if table is None and shelf_config:
        table = float(shelf_config["shelf"]["anchors"]["table"])
    out = []
    for off in poses:
        q = home + np.asarray(off, dtype=float)
        pieces, _ = robot_pieces(fr3, q, end_effector, end_effector.open_state)
        # The links that move with the wrist and elbow, and the hand (the base links sit
        # at the same height in every pose).
        V = np.vstack([p.vertices for p in pieces
                       if not p.name.startswith(("fr3_link0", "fr3_link1", "fr3_link2"))])
        row = {"offsets": list(off), "q": q.tolist(),
               "above_table": float(V[:, 2].min() - (table or 0.0))}
        if shelf is not None:
            depth, _ = shelf.penetration(V)
            row["shelf_clearance"] = float(-depth.max())
        out.append(row)
    return out


# ------------------------------------------------------------------------ the fit
@dataclass
class PayloadFit:
    payload: Payload                 # with the measured sign (TAU_EXT_SIGN); inertia: given
    flipped: Payload                 # what the other sign would have said
    bias: list                       # N m per joint
    friction: list                   # N m per joint (empty when no approach was given)
    residual_rms: list               # N m per joint, after the fit
    stderr: dict                     # mass, com_x, com_y, com_z
    condition: float
    n_poses: int
    notes: list = field(default_factory=list)

    @property
    def flipped_mass(self) -> float:
        return self.flipped.mass

    def describe(self) -> str:
        def one(name, p):
            return (f"  {name:30s} mass {p.mass:.3f} kg +- {self.stderr['mass']:.3f}, centre "
                    "of mass " + " ".join(f"{k} {v * 1000:+.1f}" for k, v in zip("xyz", p.com))
                    + f" mm (+- {max(self.stderr[k] for k in ('com_x', 'com_y', 'com_z')) * 1000:.1f})")
        lines = [f"payload from {self.n_poses} poses (condition {self.condition:.0f}):",
                 one("measured:", self.payload),
                 one("(the other sign would say:", self.flipped) + ")",
                 "  joint offsets (N m)  " + " ".join(f"{b:+.2f}" for b in self.bias)]
        if self.friction:
            lines.append("  static friction (N m) " + " ".join(f"{f:.2f}" for f in self.friction))
        lines.append("  residual rms (N m)   " + " ".join(f"{r:.2f}" for r in self.residual_rms))
        return "\n".join(lines + self.notes)


def flange_jacobian(fr3, q) -> np.ndarray:
    """Geometric Jacobian of ``fr3_link8``'s origin, (6, 7): linear rows, then angular."""
    from tpgpt.hardware.fr3 import ARM_JOINTS

    poses = fr3.link_poses(q)
    p = poses["fr3_link8"][:3, 3]
    J = np.zeros((6, 7))
    for k, name in enumerate(ARM_JOINTS):
        joint = next(j for j in fr3.chain if j.name == name)
        frame = poses[f"{name}/frame"]
        z = frame[:3, :3] @ joint.axis
        J[:3, k] = np.cross(z, p - frame[:3, 3])
        J[3:, k] = z
    return J


def _regressor(fr3, q) -> np.ndarray:
    """``A`` with ``tau_ext = A [dm, ds_x, ds_y, ds_z]``: the joint torques of a mass error
    ``dm`` and first-moment error ``ds`` (flange frame) hanging on the flange."""
    J = flange_jacobian(fr3, q)
    R = fr3.link_poses(q)["fr3_link8"][:3, :3]
    A = np.zeros((7, 4))
    A[:, 0] = J[:3].T @ G                                     # force dm g
    for k in range(3):                                        # torque (R e_k) x g per unit ds_k
        A[:, 1 + k] = J[3:].T @ np.cross(R[:, k], G)
    return A


def approach_of(q, target, dead: float = 0.005) -> np.ndarray:
    """Each joint's approach, from which side of its target it stopped (+1 below, -1 above,
    0 within ``dead`` rad): friction stops a joint short in the direction it moved (every
    joint, every pose, in the first weighing)."""
    e = np.asarray(target, dtype=float) - np.asarray(q, dtype=float)
    return np.where(np.abs(e) > dead, np.sign(e), 0.0)


def approach_history(poses, first) -> list[np.ndarray]:
    """Each sample's approach for ``payload_measure``'s sequence -- home (reached from the
    start: ``first``, per joint), then home -> pose -> home for each further pose: a joint
    a pose moves approaches it from that side, and the others keep their last motion's."""
    last = np.asarray(first, dtype=float).copy()
    out = []
    for k, off in enumerate(np.asarray(poses, dtype=float)):
        if k:
            last = np.where(off != 0, np.sign(off), last)
        out.append(last.copy())
        if k:
            last = np.where(off != 0, -np.sign(off), last)       # back home
    return out


def fit_payload(samples, modelled: Payload = FRANKA_HAND_DESK, inertia=None, fr3=None) -> PayloadFit:
    """The payload from ``samples`` -- ``[(q, tau_ext), ...]`` or ``[(q, tau_ext, approach),
    ...]``, each averaged at rest -- given what the robot ``modelled`` (Desk's values when
    they were taken). See the module docstring. With ``approach`` (:func:`approach_of`) a
    static friction per joint is fitted too. ``inertia``: reported with the result (default:
    ``modelled``'s)."""
    from tpgpt.hardware.fr3 import default_model

    fr3 = fr3 or default_model()
    friction = all(len(s) > 2 and s[2] is not None for s in samples)
    # A joint that came from the same side every time: its friction is a constant, which
    # the offset already is -- fit it only where the approach changes.
    varies = (np.ptp(np.array([s[2] for s in samples], dtype=float), axis=0) > 0
              if friction else np.zeros(7, dtype=bool))
    rows, rhs = [], []
    for sample in samples:
        q, tau = sample[0], sample[1]
        A = TAU_EXT_SIGN * _regressor(fr3, np.asarray(q, dtype=float))
        cols = [A, np.eye(7)]
        if friction:
            cols.append(np.diag(np.asarray(sample[2], dtype=float))[:, varies])
        rows.append(np.hstack(cols))
        rhs.append(np.asarray(tau, dtype=float))
    X, y = np.vstack(rows), np.concatenate(rhs)
    notes = []
    if len(samples) < 3:
        raise ValueError(f"{len(samples)} poses: weigh in at least 3 (a dozen is better)")
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ sol
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.pinv(X.T @ X) * float(r @ r) / dof
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sv = np.linalg.svd(X / np.maximum(np.abs(X).max(axis=0), 1e-12), compute_uv=False)
    condition = float(sv[0] / max(sv[-1], 1e-15))
    dm, ds = sol[0], sol[1:4]
    m = modelled.mass + dm
    s = modelled.first_moment() + ds
    c = s / m
    # Error propagation for c = s / m (first order).
    se_c = np.sqrt((se[1:4] / m) ** 2 + (s * se[0] / m ** 2) ** 2)
    m_f = modelled.mass - dm
    c_f = (modelled.first_moment() - ds) / m_f if m_f > 0 else np.full(3, np.nan)
    if not 0.2 < m < 3.0 and not 0.2 < m_f < 3.0:
        notes.append(f"IMPLAUSIBLE mass ({m:.3f} / {m_f:.3f} kg): check the poses")
    if condition > 1e3:
        notes.append(f"poorly conditioned ({condition:.0f}): turn the hand through more "
                     "directions")
    rms = np.sqrt(np.mean(r.reshape(-1, 7) ** 2, axis=0))
    if max(rms) > 0.15:
        notes.append(f"residuals up to {max(rms):.2f} N m: something the model lacks "
                     "(friction without approach directions? a cable pulling?)")
    I = inertia if inertia is not None else modelled.inertia
    return PayloadFit(Payload(float(m), tuple(c.tolist()), I),
                      Payload(float(m_f), tuple(np.asarray(c_f).tolist()), I),
                      sol[4:11].tolist(), _spread(sol[11:], varies) if friction else [],
                      rms.tolist(),
                      {"mass": float(se[0]), "com_x": float(se_c[0]), "com_y": float(se_c[1]),
                       "com_z": float(se_c[2])}, condition, len(samples), notes)


def _spread(values, mask) -> list:
    out, it = [], iter(values)
    for m in mask:
        out.append(float(next(it)) if m else 0.0)
    return out


def load_samples(path) -> tuple[list, Payload]:
    """``(samples, modelled)`` from a ``payload_measure`` record, with each sample's approach
    (recorded, or from its target: home plus the pose's offsets)."""
    from tpgpt.hardware.fr3 import HOME_Q

    d = json.loads(Path(path).expanduser().read_text())
    samples = d["samples"]
    if all("approach" in s for s in samples):
        approach = [s["approach"] for s in samples]
    else:                         # a record from before they were kept: the fixed sequence,
        first = approach_of(samples[0]["q"], HOME_Q)      # home reached from the start's side
        approach = approach_history(d["poses"], first)
    samples = [(s["q"], s["tau_ext"], a) for s, a in zip(samples, approach)]
    m = d.get("modelled") or asdict(FRANKA_HAND_DESK)
    return samples, Payload(m["mass"], tuple(m["com"]), tuple(map(tuple, m["inertia"])))


def _hand_from_config(config):
    import yaml

    from tpgpt.hardware.end_effector import inspire_from_config

    path = Path(config).expanduser() if config else (
        Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config/inspire_hand.yaml")
    return inspire_from_config(yaml.safe_load(path.read_text()))


def main(argv=None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="The FR3's payload for Desk (plan M5).")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cad", help="Desk's values from the manufacturer's data and the mount")
    c.add_argument("--config", default=None, help="inspire_hand.yaml (the measured mount)")
    c.add_argument("--mount-mass", type=float, default=MOUNT_MASS_GUESS)
    c.add_argument("--vendor-urdf", default=None, help="recompute the hand from this URDF")
    p = sub.add_parser("poses", help="the weighing poses, checked against the table and shelf")
    p.add_argument("--config", default=None)
    f = sub.add_parser("fit", help="fit a payload_measure record")
    f.add_argument("record")
    f.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    ee = _hand_from_config(a.config)
    if a.command == "cad":
        hand = vendor_hand_inertial(a.vendor_urdf) if a.vendor_urdf else None
        cad = inspire_cad_payload(ee, a.mount_mass, hand)
        print(f"the Inspire hand (manufacturer's data) and its mount ({a.mount_mass * 1000:.0f} g "
              "guessed), for Desk:\n" + cad.desk_text())
        print("\nDesk now (the Franka Hand):\n" + FRANKA_HAND_DESK.desk_text())
        return 0
    if a.command == "poses":
        root = Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config"
        shelf = yaml.safe_load((root / "shelf.yaml").read_text())
        for k, row in enumerate(pose_clearances(ee, shelf_config=shelf)):
            print(f"  pose {k:2d} offsets {row['offsets']}: {row['above_table'] * 1000:6.0f} mm "
                  f"above the table, {row['shelf_clearance'] * 1000:6.0f} mm from the shelf")
        return 0
    samples, modelled = load_samples(a.record)
    cad = inspire_cad_payload(ee)
    fit = fit_payload(samples, modelled, inertia=cad.inertia)
    print(fit.describe())
    print(f"\nthe manufacturer's data says {cad.mass:.3f} kg at "
          + " ".join(f"{v * 1000:+.1f}" for v in cad.com) + " mm")
    print("\nfor Desk (Settings > End effector; mass, centre of mass and inertia only -- keep its\n"
          "flange-to-end-effector transform):\n" + fit.payload.desk_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
