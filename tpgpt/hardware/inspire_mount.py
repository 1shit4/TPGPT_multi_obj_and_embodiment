"""Measuring how the Inspire hand is mounted on the FR3, by touching (plan M3).

The mount is four numbers of :func:`tpgpt.hardware.end_effector.inspire_end_effector`:
the hand's **yaw** about the flange axis (``mount_yaw_deg``), the **adapter thickness**
(flange face -> the hand's wrist face) and the wrist face's **lateral** offset from
the flange axis, ``(x, y)`` in ``fr3_link8``. A caliper gets the thickness and the
eye the yaw to a few millimetres and degrees; every millimetre lands on the grasp,
so they are fitted to touches, the way the camera was calibrated
(``docs/hardware/CALIBRATION.md``):

* the hand is put in a pointing pose (:data:`POINT_POSES`) -- the index or the little
  finger straight, the others curled, the thumb swung out of the way -- and hand-guided
  (``free_drive``) until that fingertip rests on something whose position is known;
* each touch records ``/current_pose`` (``fr3_hand_tcp``) and the hand's registers;
* a **table** touch says: the lowest point of the index finger's last link is at the
  table's touched height (one equation); a **point** touch says: its tip is on a known
  point, a touched slot mark (three);
* :func:`fit_mount` solves the four numbers by least squares from the hand model's
  **exact** collision-mesh vertices (:meth:`InspireHandModel.mesh_vertices`), not its
  conservative pieces.

**One fingertip cannot give all four.** Touches with one finger measure where that one
point is in the flange's frame -- three numbers, however the wrist is tilted -- and the
mount has four: turning the hand about the flange axis and shifting it sideways both just
move that point (the first live fit, 2026-10-05, index only: yaw +-405 deg, offset +-160
mm, while the thickness came to +-0.8 mm). Tilts help only through which vertex of the
rounded tip is lowest, which is weak. So **touch with both the index and the little
finger** (68 mm apart across the palm): then all four are determined. With one finger's
touches only, the fit holds the sideways offset at its start (0: the hand centred on the
flange) and says so. Tilt the hand 15-30 deg in several directions all the same: it
separates the thickness from the rest. The fit reports each number's standard error and
the residual of every touch; a touch that disagrees by more than a few millimetres is a
bad touch (or a bad model) and is shown as such rather than averaged in.

What it assumes: the hand's own kinematics (GraspGen-X's URDF, checked against the
manual: 217.9 mm against 217.8 to the middle fingertip) and the register map at
:data:`POINT_POSE` (the index straight, q = 0.029 rad: 2.6 mm at the tip per 0.03 rad
of error -- M4 checks it). numpy and scipy; no ROS. The recorder is the ROS node
``inspire_touch``.

    python -m tpgpt.hardware.inspire_mount fit ~/inspire_mount/touches.json
    python -m tpgpt.hardware.inspire_mount offsets            # index tip in fr3_hand_tcp
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Registers (little, ring, middle, index, thumb bend, thumb rotation) of the pointing
#: poses the touches are made in: one finger straight, the other three curled, the thumb
#: straight and swung out flat beside the palm (beta 165 deg), out of the way
#: (``inspire_hand.POINT_POSES``, the same values).
POINT_POSES = {"index": (0, 0, 0, 1000, 1000, 1000), "little": (1000, 0, 0, 0, 1000, 1000)}
POINT_POSE = POINT_POSES["index"]
#: The link that touches, per pointing finger: that finger's last.
TOUCH_LINKS = {"index": "index_intermediate", "little": "pinky_intermediate"}
TOUCH_LINK = TOUCH_LINKS["index"]
#: Register units a pointing pose may be off and still count as that pose.
POSE_TOLERANCE = 40
#: A touch whose residual is longer than this is flagged. The hand rocks on its ring by
#: ~+-2.5 mm at the fingertips (INSPIRE_HAND.md, M2b), so 3-4 mm is an ordinary touch.
DISAGREES_MM = 5.0
#: The four fitted numbers, in ``inspire_end_effector``'s units (deg, m, m, m).
PARAMS = ("mount_yaw_deg", "adapter_thickness", "lateral_x", "lateral_y")


@dataclass
class Touch:
    """One touch: what was touched, and the arm and hand when it was."""

    kind: str                       # "table" (target [z]) or "point" (target [x, y, z])
    target: list
    tcp_position: list              # fr3_hand_tcp in the base frame, m
    tcp_rotation: list              # 3x3
    registers: list                 # the hand's six ANGLE_ACT values
    label: str = ""
    link: str = TOUCH_LINK

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Touch":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def pointing_finger(registers) -> str | None:
    """Which of :data:`POINT_POSES` the hand is in (within :data:`POSE_TOLERANCE`), else None."""
    for finger, pose in POINT_POSES.items():
        if max(abs(int(a) - int(b)) for a, b in zip(registers, pose)) <= POSE_TOLERANCE:
            return finger
    return None


def _mount(params) -> dict:
    yaw, a, lx, ly = (float(v) for v in params)
    return {"mount_yaw_deg": yaw, "adapter_thickness": a, "lateral": (lx, ly)}


def _tip_index(model, link: str) -> int:
    """The link's tip: its mesh vertex furthest from the link's own joint."""
    from tpgpt.hardware.inspire_model import _mesh_vertices

    return int(np.argmax(np.linalg.norm(_mesh_vertices(model.description.config_name, link),
                                        axis=1)))


def touch_points(touch: Touch, params, base: dict | None = None) -> np.ndarray:
    """The touching link's exact vertices in the base frame, with the mount ``params``."""
    from tpgpt.hardware.end_effector import inspire_end_effector

    ee = inspire_end_effector(**{**(base or {}), **_mount(params)})
    model = ee.model
    V = model.mesh_vertices(touch.link, model.registers_to_q(touch.registers))
    R_E = np.asarray(touch.tcp_rotation, dtype=float).reshape(3, 3)
    p_E = np.asarray(touch.tcp_position, dtype=float).reshape(3)
    R_g = ee.grasp_from_ee(R_E)
    p_g = p_E + R_E @ ee.tool_offset
    return V @ R_g.T + p_g


def touch_residual(touch: Touch, params, base: dict | None = None) -> np.ndarray:
    """Predicted minus touched, m: ``[dz]`` for a table touch, ``[dx, dy, dz]`` for a point."""
    from tpgpt.hardware.end_effector import inspire_end_effector

    W = touch_points(touch, params, base)
    if touch.kind == "table":
        return np.array([W[:, 2].min() - float(touch.target[0])])
    if touch.kind == "point":
        model = inspire_end_effector(**{**(base or {}), **_mount(params)}).model
        return W[_tip_index(model, touch.link)] - np.asarray(touch.target, dtype=float)
    raise ValueError(f"touch kind must be 'table' or 'point', not {touch.kind!r}")


@dataclass
class MountFit:
    """What :func:`fit_mount` found."""

    params: dict                                   # PARAMS -> value
    stderr: dict                                   # PARAMS -> standard error
    residuals_mm: list                             # per touch: its residual vector, mm
    rms_mm: dict                                   # "table", "point", "all"
    start: dict
    n_equations: int
    condition: float
    notes: list = field(default_factory=list)
    held: list = field(default_factory=list)       # PARAMS kept at their start
    fingers: list = field(default_factory=list)    # the touching links

    def yaml_block(self) -> str:
        p = self.params
        return ("  mount:\n"
                "    controller_frame: fr3_hand_tcp\n"
                f"    yaw_deg: {p['mount_yaw_deg']:.2f}\n"
                f"    adapter_thickness_m: {p['adapter_thickness']:.4f}\n"
                f"    lateral_m: [{p['lateral_x']:.4f}, {p['lateral_y']:.4f}]\n"
                + ("    measured: true" if not self.held else
                   "    measured: false   # the offset was held: touch with both fingers"))

    def describe(self, touches) -> str:
        lines = [f"fit from {len(touches)} touches ({self.n_equations} equations), condition "
                 f"{self.condition:.0f}:"]
        units = {"mount_yaw_deg": ("deg", 1.0), "adapter_thickness": ("mm", 1000.0),
                 "lateral_x": ("mm", 1000.0), "lateral_y": ("mm", 1000.0)}
        for k in PARAMS:
            u, s = units[k]
            spread = "HELD" if k in self.held else f"+- {self.stderr[k] * s:.2f}"
            lines.append(f"  {k:18s} {self.params[k] * s:9.2f} {u}  {spread}"
                         f"   (start {self.start[k] * s:.2f})")
        lines.append("  rms: " + ", ".join(f"{k} {v:.2f} mm" for k, v in self.rms_mm.items()))
        for t, r in zip(touches, self.residuals_mm):
            flag = "   <-- disagrees" if np.linalg.norm(r) > DISAGREES_MM else ""
            finger = "little" if t.link == TOUCH_LINKS["little"] else "index"
            lines.append(f"    {t.kind:5s} {t.label:12s} {finger:6s} {np.round(r, 2).tolist()} mm{flag}")
        lines.extend(self.notes)
        return "\n".join(lines)


def fit_mount(touches: list[Touch], start: dict | None = None, base: dict | None = None,
              hold: tuple | None = None) -> MountFit:
    """Least-squares mount from touches (module docstring).

    Args:
        start: initial :data:`PARAMS` (default: -45 deg, 12 mm, 0, 0 -- the first estimate;
            the robot's tools pass the config's, since 2026-10-05 135 deg and 16 mm).
        base: the other ``inspire_end_effector`` arguments (description, joint map).
        hold: :data:`PARAMS` kept at their start values. ``None`` = the lateral offset when
            every touch is with the same finger (it cannot be told from the yaw then),
            nothing otherwise; ``()`` = nothing.
    """
    from scipy.optimize import least_squares

    s0 = {"mount_yaw_deg": -45.0, "adapter_thickness": 0.012, "lateral_x": 0.0,
          "lateral_y": 0.0, **(start or {})}
    x0 = np.array([s0[k] for k in PARAMS], dtype=float)
    notes = []
    fingers = sorted({t.link for t in touches})
    if hold is None:
        hold = ("lateral_x", "lateral_y") if len(fingers) < 2 else ()
        if hold:
            notes.append(f"every touch is with {fingers[0]}: one fingertip cannot tell the yaw "
                         f"from a sideways offset, so the offset is HELD at "
                         f"({s0['lateral_x'] * 1000:.1f}, {s0['lateral_y'] * 1000:.1f}) mm, its "
                         "start (the config's). Touch with the "
                         + ("index finger too (/inspire_hand/point)"
                            if fingers[0] == TOUCH_LINKS["little"] else
                            "little finger too (/inspire_hand/point_little)")
                         + " to measure it; fit both files together.")
    free = [k for k in PARAMS if k not in hold]
    # Scale so a degree and a millimetre weigh alike in the solver.
    scale = np.array([{"mount_yaw_deg": 1.0}.get(k, 0.001) for k in free])

    def full(x):
        v = x0.copy()
        for k, value in zip(free, x):
            v[PARAMS.index(k)] = value
        return v

    def residuals(x):
        return np.concatenate([touch_residual(t, full(x), base) for t in touches])

    n = sum(1 if t.kind == "table" else 3 for t in touches)
    if n < len(free):
        raise ValueError(f"{n} equations for {len(free)} unknowns: touch more")
    sol = least_squares(residuals, x0[[PARAMS.index(k) for k in free]], x_scale=scale,
                        method="lm" if n >= len(free) else "trf")
    J = sol.jac
    dof = max(n - len(free), 1)
    sigma2 = float(sol.fun @ sol.fun) / dof
    stderr_free = np.full(len(free), np.inf)
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        stderr_free = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        notes.append("the touches do not determine every number (singular): vary the wrist")
    stderr = np.zeros(len(PARAMS))
    for k, e in zip(free, stderr_free):
        stderr[PARAMS.index(k)] = e
    Js = J * scale
    sv = np.linalg.svd(Js, compute_uv=False)
    condition = float(sv[0] / max(sv[-1], 1e-15))
    if condition > 1e3:
        notes.append(f"poorly conditioned ({condition:.0f}): tilt the hand more between touches, "
                     "and touch with both fingers")
    yaw, a, lx, ly = full(sol.x)
    if not 0.0 <= a <= 0.06 or np.hypot(lx, ly) > 0.010 or abs(yaw - x0[0]) > 30.0:
        notes.append(f"IMPLAUSIBLE mount (adapter {a * 1000:.1f} mm, lateral "
                     f"{np.hypot(lx, ly) * 1000:.1f} mm, yaw {yaw - x0[0]:+.1f} deg from the "
                     "estimate): check the touches before using it")
    if n <= len(PARAMS) + 2:
        notes.append(f"only {n} equations for {len(PARAMS)} numbers: the residuals cannot show "
                     "a bad touch; touch more")
    per = []
    k = 0
    for t in touches:
        m = 1 if t.kind == "table" else 3
        per.append((sol.fun[k:k + m] * 1000.0).tolist())
        k += m
    kinds = {"table": [r for t, r in zip(touches, per) if t.kind == "table"],
             "point": [r for t, r in zip(touches, per) if t.kind == "point"]}
    rms = {name: float(np.sqrt(np.mean(np.concatenate([np.square(r) for r in rows]))))
           for name, rows in kinds.items() if rows}
    rms["all"] = float(np.sqrt(np.mean(np.square(sol.fun)))) * 1000.0
    return MountFit(params=dict(zip(PARAMS, full(sol.x).tolist())),
                    stderr=dict(zip(PARAMS, stderr.tolist())),
                    residuals_mm=per, rms_mm=rms, start={k: s0[k] for k in PARAMS},
                    n_equations=n, condition=condition, notes=notes, held=list(hold),
                    fingers=fingers)


def load_touches(path) -> list[Touch]:
    data = json.loads(Path(path).expanduser().read_text())
    return [Touch.from_dict(d) for d in data.get("touches", data)]


def _base_from_config(config_path) -> tuple[dict, dict]:
    """(``base`` kwargs, start params) from an ``inspire_hand.yaml``."""
    import yaml

    from tpgpt.hardware.inspire_model import joint_map_from_config

    cfg = yaml.safe_load(Path(config_path).expanduser().read_text())["inspire_hand"]
    mount, gg = cfg.get("mount", {}), cfg.get("graspgenx", {})
    base = {"description": gg.get("gripper", "inspire_hand"),
            "joint_map": joint_map_from_config(cfg),
            "adapter_radius": mount.get("adapter_radius_m", 0.0315)}
    lateral = mount.get("lateral_m", (0.0, 0.0))
    start = {"mount_yaw_deg": float(mount.get("yaw_deg", 135.0)),
             "adapter_thickness": float(mount.get("adapter_thickness_m", 0.012)),
             "lateral_x": float(lateral[0]), "lateral_y": float(lateral[1])}
    return base, start


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config/inspire_hand.yaml"


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fit the Inspire hand's mount to touches.")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="fit the mount to touches.json files (inspire_touch writes them)")
    f.add_argument("touches", nargs="+")
    f.add_argument("--free", action="store_true",
                   help="fit all four numbers even if every touch is with one finger")
    f.add_argument("--config", default=None, help="inspire_hand.yaml (start values, joint map)")
    o = sub.add_parser("offsets", help="the index tip in fr3_hand_tcp, in POINT_POSE")
    o.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    base, start = _base_from_config(a.config or default_config_path())
    if a.command == "offsets":
        from tpgpt.hardware.end_effector import inspire_end_effector

        ee = inspire_end_effector(**{**base, **_mount([start[k] for k in PARAMS])})
        model = ee.model
        V = model.mesh_vertices(TOUCH_LINK, model.registers_to_q(POINT_POSE))
        tip = V[_tip_index(model, TOUCH_LINK)]
        in_tcp = ee.grasp_to_ee.T @ (tip) + ee.tool_offset     # G -> controller frame
        print(f"index tip in fr3_hand_tcp (mount {start}): {np.round(in_tcp, 4).tolist()} m")
        print("  for calibration_check_node: -p probe_offset:="
              f"\"[{in_tcp[0]:.4f}, {in_tcp[1]:.4f}, {in_tcp[2]:.4f}]\"")
        return 0
    touches = [t for path in a.touches for t in load_touches(path)]
    result = fit_mount(touches, start, base, hold=() if a.free else None)
    print(result.describe(touches))
    print("\nfor config/inspire_hand.yaml (then colcon build):\n" + result.yaml_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
