"""The shelf, as the one thing on the real bench the robot can hit.

**What the simulation does, and why the robot does something else here.** In
``tpgpt.experiments.pipeline.run`` the collision filters test the arm against
an *observed* scene cloud (``scene_point_cloud`` with the robot subtracted), on
the argument that "the robot knows its own links exactly and knows the shelf
only as a point cloud" -- giving the filter the true geometry would make it
depend on something hardware would not have (``path_feasibility_observed``'s
docstring).

On this bench that argument turns round, for two measured reasons:

* **The camera does not see the whole shelf, nor the whole workspace.** Its
  view covers part of the shelf and misses much of the space the arm moves
  through. A cloud-only check is therefore blind exactly where the arm may meet
  the unseen part of the shelf; and treating everything unseen as occupied
  would reject nearly every path, since most of the arm's motion is out of view.
* **The shelf's geometry *is* known**: its boards were touched with the
  fingertip in the base frame (``docs/hardware/CALIBRATION.md``, repeatable to
  about 1 mm), which is independent of the camera calibration.

What the arm can hit, as the user stated it (2026-09-22):

1. **everything the camera sees** -- the observed scene cloud, over the whole of
   the arm's reach, with the target, the robot itself (FR3 hulls and hand boxes
   at the joint angles of the capture) and the table removed;
2. **where the camera cannot see, only the shelf** -- the shelf model, boxes built
   from the measured dimensions, is the authority everywhere, seen or unseen;
   unseen space away from the shelf is free, not assumed occupied;
3. **its own body**, anywhere -- checked in ``tpgpt.hardware.path_check``, not here.

Without the joint angles of the capture the robot cannot be removed from the
cloud, and the arm would collide with its own image, so the cloud is then kept
only inside the shelf's region of interest (its box plus a margin), and the
report says so (``scope``).

**Exact geometry replaces the camera wherever it exists** (user, 2026-09-22):
the table is an exact plane at the touched table height (``table_height``), and
the shelf is its model. Camera points on either are removed -- points within
:data:`TABLE_BAND` above the table, and points within :data:`EXPLAINED_DISTANCE`
of a shelf panel -- because they duplicate the exact geometry and would only
add its depth noise (+/-5 mm) as false contacts. What remains in the cloud is
everything else: other objects, and anything the model gets wrong -- a panel
mis-measured by more than 15 mm keeps its real points as obstacles.

The inside-test machinery is the simulation's: shelf panels are
:class:`tpgpt.perception.obstacles.Obstacle` boxes with its exact signed
distance, and the model's faces are also sampled into points so the same
"scene points inside a robot body" test the simulation applies to its cloud
applies to the model too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Scene points below ``support + TABLE_BAND`` are the table, metres. The cloud's
#: table reads 4.9 mm from the touched height with a 0.61 deg tilt
#: (CALIBRATION.md), about 4 mm across 0.4 m, so table points lie within about
#: 10 mm of the plane over the reach. Cost: anything flatter than 15 mm lying on
#: the table is not an obstacle.
TABLE_BAND = 0.015

#: Points further than this from the shoulder (joint 2, 0.333 m above the base)
#: cannot be touched: the FR3 reaches 0.855 m to the wrist, plus the hand.
REACH_RADIUS = 1.15
SHOULDER = (0.0, 0.0, 0.333)

#: Region-of-interest margin around the shelf model, metres. Cloud points this
#: far outside the model still count as "on the shelf" (an object standing at a
#: slot's front edge, a panel a centimetre off).
ROI_MARGIN = 0.03

#: How far the region reaches above each board objects stand on, metres. An
#: open-top shelf's model ends at its top board, but whatever stands *on* that
#: board -- an object placed earlier -- is exactly what the model cannot know and
#: the cloud must keep. 0.30 m covers the tallest objects used (a cereal box).
ROI_ABOVE_SUPPORT = 0.30

#: How far above the shelf's own bottom the region starts, metres, so the table
#: surface around the shelf's foot is not mistaken for a shelf obstacle. The
#: table is the planner's support plane.
ROI_FLOOR_CLEARANCE = 0.01

#: Spacing of the points sampled on the model's faces, metres. Below the
#: smallest robot piece that can cross a panel face (a finger pad, 15 mm), so a
#: pad pushed through a board face always contains a sample.
SURFACE_SPACING = 0.01

#: Observed points within this of the target's cloud are the target, not an
#: obstacle, metres (three planner voxels plus depth noise).
TARGET_EXCLUSION = 0.015

#: Growth of every robot piece when removing the robot from the cloud, metres:
#: a surface point sits *on* the body, and depth is noisy
#: (``perception.obstacles.self_filtered``'s margin, doubled for a real sensor).
SELF_FILTER_MARGIN = 0.01

#: Observed shelf points within this of the model's surface count as explained
#: by it, metres. The calibration's own accuracy is +/-5 mm.
EXPLAINED_DISTANCE = 0.015


#: The tape measurements ``resolve_tape`` needs, in millimetres, keyed as in
#: ``docs/hardware/shelf_measurements.html`` (M1..M14).
TAPE_KEYS = {
    "M1": "slot_left_to_left_panel",
    "M2": "slot_right_to_right_panel",
    "M3": "slot_left_to_front_edge",
    "M4": "slot_right_to_front_edge",
    "M5": "inner_width",
    "M6": "left_panel_thickness",
    "M7": "right_panel_thickness",
    "M8": "panel_height",
    "M9": "panel_depth",
    "M10": "panel_front_ahead_of_boards",
    "M11": "board_depth",
    "M12": "topmost_above_top_board",
    "M13": "board_thickness",
    "M14": "apron_height",
}

#: Width by the slots and by the tape must agree to this, metres, or the anchoring
#: is wrong (a slot mark re-found in the wrong place, or a misread tape).
TAPE_WIDTH_TOLERANCE = 0.006

# ---- fit_to_cloud: the shelf's placement corrected from the camera -------------
#: Cloud points further than this from the tape-placed model are not the shelf,
#: metres (first round; the second uses half). Above the ~10 mm the tape's
#: least certain readings (M1-M4) could be out, below the gap to the nearest
#: other thing standing near a panel.
FIT_CAPTURE = 0.04

#: Residual scale of the robust (Cauchy) loss, metres: the cloud's depth noise on
#: a surface at 1 m (CALIBRATION.md: surfaces within ~5 mm).
FIT_NOISE = 0.004

#: Points on faces square to a direction needed to fit a shift along it; fewer,
#: and that shift is left at 0 (the tape's value) and the report says so.
FIT_MIN_FACE_POINTS = 150

#: The front faces seen must span this much along the front edge to fit the yaw,
#: metres (a 1 deg yaw moves the ends of a 0.58 m shelf by 5 mm).
FIT_MIN_YAW_SPAN = 0.15

#: A correction larger than this is refused, not applied (metres, degrees): it
#: is beyond any plausible tape error, so something else is wrong -- the
#: calibration, an object or the arm against a panel, or a mis-keyed reading.
FIT_MAX_SHIFT = 0.03
FIT_MAX_YAW_DEG = 3.0

#: After the fit, the median distance of the shelf's points to the model must be
#: under this, metres, or the model does not describe what is seen (a dimension
#: wrong, a correction beyond the capture distance). Depth noise alone gives ~2-3 mm.
FIT_MAX_MEDIAN = 0.006

#: After the fit, every well-seen side or front face (at least
#: FIT_MIN_FACE_POINTS points) must sit within this of its points, metres.
#: Depth noise averages out over that many points; a face further off means the
#: model's dimensions and the cloud disagree, which no placement fixes.
FIT_FACE_TOLERANCE = 0.003


def resolve_tape(c: dict) -> dict:
    """The shelf's geometry from tape measurements anchored to touched points.

    The robot cannot touch the side panels' tops or the topmost board squarely,
    so the shelf's own dimensions come from a tape (``tape_mm``, relative to the
    shelf) and the shelf is placed in the base frame by the points already
    touched (``anchors``: the two slot marks on the top board and the board
    heights, ``CALIBRATION.md``). The slot marks are also measured *to* by tape
    (M1-M4), which ties the two together:

    * the front edge is found by stepping back from each slot mark, square to
      the slot line, by M3 / M4, then the two ends by M1 / M2 along it;
    * the width that gives is checked against the tape's own inner width (M5),
      and M5 is the width used.

    What the tape cannot fix well (M1-M4, the slot marks to the panel and the
    front edge) sets the shelf's x/y and yaw; :func:`fit_to_cloud` measures a
    correction for those from the camera, stored as ``cloud_fit``.

    Returns the config with the builder's keys filled in, and ``derived``
    holding the checks. Raises if the two widths disagree by more than
    :data:`TAPE_WIDTH_TOLERANCE`.
    """
    c = dict(c)
    t = {k: float(v) / 1000.0 for k, v in c["tape_mm"].items() if v is not None}
    missing = [f"{m} ({key})" for m, key in TAPE_KEYS.items()
               if key not in t and m not in ("M10", "M14")]
    if missing:
        raise ValueError(f"shelf tape measurements missing: {', '.join(missing)}")
    a = c["anchors"]
    L = np.asarray(a["left_slot"], dtype=float)[:2]
    R = np.asarray(a["right_slot"], dtype=float)[:2]
    s = (R - L) / np.linalg.norm(R - L)
    y = np.array([-s[1], s[0]])                     # z x slot line: into the shelf
    if float(y @ ((L + R) / 2.0)) <= 0.0:
        raise ValueError("anchors: left_slot/right_slot are swapped (left = the robot's left "
                         "as it faces the shelf's open front)")
    front_l = L - y * t["slot_left_to_front_edge"]
    front_r = R - y * t["slot_right_to_front_edge"]
    x = (front_r - front_l) / np.linalg.norm(front_r - front_l)
    width_slots = float(np.linalg.norm(front_r - front_l)
                        + t["slot_left_to_left_panel"] + t["slot_right_to_right_panel"])
    mismatch = width_slots - t["inner_width"]
    # The width is the one direct reading across the shelf (M5); M1 and M2 are the
    # less certain ones (hard to hold square), so their disagreement with it is
    # split between the two ends. ``cloud_fit`` then moves the whole shelf.
    corner_l = front_l - x * (t["slot_left_to_left_panel"] - mismatch / 2.0)
    corner_r = front_r + x * (t["slot_right_to_right_panel"] - mismatch / 2.0)
    if abs(mismatch) > TAPE_WIDTH_TOLERANCE:
        raise ValueError(
            f"shelf tape: the width from the slot marks (M1 + slot spacing + M2 = "
            f"{width_slots * 1000:.0f} mm) disagrees with the inner width M5 "
            f"({t['inner_width'] * 1000:.0f} mm) by {mismatch * 1000:+.0f} mm; re-check M1, M2, "
            "M5 and that the slot marks were re-found where they were touched")
    table = float(a["table"])
    top_board = float(a["top_board_top"])
    thick = t["board_thickness"]
    c.update({
        "front_left": corner_l.tolist(),
        "front_right": corner_r.tolist(),
        "depth": t["board_depth"],
        "bottom": table,
        "top": table + t["panel_height"],
        "wall_thickness_left": t["left_panel_thickness"],
        "wall_thickness_right": t["right_panel_thickness"],
        "back_thickness": 0.0,
        "side_front": t.get("panel_front_ahead_of_boards", 0.0),
        "side_depth": t["panel_depth"],
        "boards": [
            {"name": "bottom_board", "top": float(a["bottom_board_top"]), "thickness": thick},
            {"name": "top_board", "top": top_board, "thickness": thick},
            {"name": "topmost_board", "top": top_board + t["topmost_above_top_board"],
             "thickness": thick},
        ],
    })
    apron = t.get("apron_height", 0.0)
    if apron > 0.0:
        # A front panel under the bottom board, between the sides; its thickness
        # is not measured, so it is taken as the boards'.
        c.setdefault("extra_boxes", [])
        c["extra_boxes"] = list(c["extra_boxes"]) + [{
            "name": "apron", "lo": [0.0, 0.0, table],
            "hi": [t["inner_width"], thick, table + apron]}]
    c["derived"] = {"width_from_slots_mm": width_slots * 1000.0,
                    "width_mismatch_mm": mismatch * 1000.0,
                    "front_edge_yaw_deg": float(np.degrees(np.arctan2(x[1], x[0])))}
    return c


def _box(name: str, lo, hi, R_base_shelf: np.ndarray, origin: np.ndarray):
    """An :class:`Obstacle` box spanning ``lo..hi`` in the shelf frame."""
    from tpgpt.perception.obstacles import Obstacle

    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if np.any(hi <= lo):
        raise ValueError(f"shelf box {name!r} is empty: lo {lo.tolist()}, hi {hi.tolist()}")
    centre = (lo + hi) / 2.0
    return Obstacle(name=name, kind="box", position=origin + R_base_shelf @ centre,
                    rotation=R_base_shelf.copy(), size=(hi - lo) / 2.0)


@dataclass
class ShelfModel:
    """The shelf as boxes in the base frame.

    Attributes:
        boxes: One :class:`tpgpt.perception.obstacles.Obstacle` per panel.
        supports: Names of the boards an object may be set down on.
        R_base_shelf, origin: The shelf frame (``+x`` along the front edge from
            the left wall to the right, ``+y`` into the shelf, ``+z`` up).
        measured: ``False`` while any dimension in the config is an estimate.
        config: The configuration it was built from, for the plan record.
    """

    boxes: list
    supports: tuple[str, ...]
    R_base_shelf: np.ndarray
    origin: np.ndarray
    measured: bool
    config: dict = field(default_factory=dict)
    #: Inner width: the front edge runs from ``origin`` this far along ``+x``.
    width: float = 0.0

    # ------------------------------------------------------------ building
    @classmethod
    def from_config(cls, config: dict) -> "ShelfModel":
        """Build from the ``shelf:`` block of ``config/shelf.yaml``.

        A bookcase is described by the numbers one can touch: the front edge at
        each wall's inner face, the depth to the back panel, the heights of the
        board top surfaces, and the panel thicknesses. Anything else is an
        ``extra_boxes`` entry.
        """
        c = dict(config.get("shelf", config))
        # An all-empty tape block (not yet measured) leaves the estimate in use; a
        # partly filled one is resolved, and names what is missing.
        if any(v is not None for v in (c.get("tape_mm") or {}).values()):
            c = resolve_tape(c)
        left = np.asarray(c["front_left"], dtype=float)[:2]
        right = np.asarray(c["front_right"], dtype=float)[:2]
        if c.get("cloud_fit"):
            left, right = apply_cloud_fit(left, right, c["cloud_fit"])
        width = float(np.linalg.norm(right - left))
        if width < 0.05:
            raise ValueError("front_left and front_right must be the two ends of the front edge")
        x = np.array([*(right - left) / width, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        bottom, top = float(c["bottom"]), float(c["top"])
        origin = np.array([left[0], left[1], 0.0])
        # +y must point away from the robot, into the shelf; otherwise left and
        # right were swapped and every panel would be mirrored through the front.
        centre_xy = (left + right) / 2.0
        if float(y[:2] @ centre_xy) <= 0.0:
            raise ValueError(
                "front_left/front_right are swapped: with the robot looking at the shelf's "
                "open front, front_left must be the end on the robot's left")
        R = np.column_stack([x, y, z])
        depth = float(c["depth"])
        wall = float(c.get("wall_thickness", 0.018))
        wall_l = float(c.get("wall_thickness_left", wall))
        wall_r = float(c.get("wall_thickness_right", wall))
        back = float(c.get("back_thickness", 0.0))
        # The side panels may stand proud of the boards' front edge (side_front > 0)
        # and be deeper or shallower than the boards (side_depth).
        side_front = float(c.get("side_front", 0.0))
        side_depth = float(c.get("side_depth", depth + back))
        side_lo, side_hi = -side_front, -side_front + side_depth

        boxes, supports = [], []
        if wall_l > 0:
            boxes.append(_box("left_wall", (-wall_l, side_lo, bottom), (0.0, side_hi, top),
                              R, origin))
        if wall_r > 0:
            boxes.append(_box("right_wall", (width, side_lo, bottom), (width + wall_r, side_hi, top),
                              R, origin))
        if back > 0:
            boxes.append(_box("back_panel", (-wall_l, depth, bottom),
                              (width + wall_r, depth + back, top), R, origin))
        for board in c.get("boards", []):
            name = str(board["name"])
            surface = float(board["top"])
            thickness = float(board.get("thickness", 0.018))
            board_depth = float(board.get("depth", depth))
            boxes.append(_box(name, (0.0, 0.0, surface - thickness), (width, board_depth, surface),
                              R, origin))
            if board.get("support", True):
                supports.append(name)
        for extra in c.get("extra_boxes", []):
            lo = np.asarray(extra["lo"], dtype=float)
            hi = np.asarray(extra["hi"], dtype=float)
            if extra.get("frame", "shelf") == "base":
                boxes.append(_box(str(extra["name"]), lo, hi, np.eye(3), np.zeros(3)))
            else:
                boxes.append(_box(str(extra["name"]), lo, hi, R, origin))
            if extra.get("support", False):
                supports.append(str(extra["name"]))
        return cls(boxes=boxes, supports=tuple(supports), R_base_shelf=R, origin=origin,
                   measured=bool(c.get("measured", False)), config=c, width=width)

    # ------------------------------------------------------------ queries
    def penetration(self, points: np.ndarray) -> tuple[np.ndarray, list]:
        """Deepest penetration of each point into any panel, and which panel.

        Returns ``(depth (n,), names)``; depth > 0 is inside.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        depth = np.full(len(p), -np.inf)
        names = [None] * len(p)
        for box in self.boxes:
            d = box.penetration(p)
            deeper = d > depth
            depth = np.where(deeper, d, depth)
            for i in np.flatnonzero(deeper):
                names[i] = box.name
        return depth, names

    def roi_bounds(self, margin: float = ROI_MARGIN,
                   above_support: float = ROI_ABOVE_SUPPORT) -> tuple[np.ndarray, np.ndarray]:
        """The shelf's region of interest in the shelf frame.

        Its bounding box grown by ``margin``, reaching at least ``above_support``
        above the highest board objects stand on, with the floor lifted
        :data:`ROI_FLOOR_CLEARANCE` above the shelf's bottom so the table around
        the shelf's foot stays out.
        """
        corners = np.vstack([self._corners(b) for b in self.boxes])
        local = (corners - self.origin) @ self.R_base_shelf
        lo, hi = local.min(axis=0) - margin, local.max(axis=0) + margin
        lo[2] = local[:, 2].min() + ROI_FLOOR_CLEARANCE
        tops = [float(b.position[2] + b.size[2]) for b in self.boxes if b.name in self.supports]
        if tops:
            hi[2] = max(hi[2], max(tops) + above_support)
        return lo, hi

    def in_roi(self, points: np.ndarray, margin: float = ROI_MARGIN,
               above_support: float = ROI_ABOVE_SUPPORT) -> np.ndarray:
        lo, hi = self.roi_bounds(margin, above_support)
        local = (np.atleast_2d(points) - self.origin) @ self.R_base_shelf
        return np.all((local >= lo) & (local <= hi), axis=1)

    def near(self, points: np.ndarray, reach: float) -> bool:
        """Whether any of ``points`` is within ``reach`` of the shelf's region."""
        lo, hi = self.roi_bounds(reach, above_support=0.0)
        local = (np.atleast_2d(points) - self.origin) @ self.R_base_shelf
        return bool(np.any(np.all((local >= lo) & (local <= hi), axis=1)))

    def surface_points(self, spacing: float = SURFACE_SPACING) -> tuple[np.ndarray, np.ndarray]:
        """Points on every panel face, and the index of the panel each lies on."""
        pts, owner = [], []
        for k, box in enumerate(self.boxes):
            s = box_surface(box.size, spacing)
            pts.append(box.position + s @ box.rotation.T)
            owner.append(np.full(len(s), k))
        return np.vstack(pts), np.concatenate(owner)

    def explained(self, points: np.ndarray, distance: float = EXPLAINED_DISTANCE) -> np.ndarray:
        """Which observed points lie within ``distance`` of the model's surface."""
        depth, _ = self.penetration(points)
        return np.abs(depth) <= distance

    @staticmethod
    def _corners(box) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return box.position + (signs * box.size) @ box.rotation.T

    def to_dict(self) -> dict:
        return {
            "measured": self.measured,
            "supports": list(self.supports),
            "origin": self.origin.tolist(),
            "R_base_shelf": self.R_base_shelf.tolist(),
            "boxes": [{"name": b.name, "centre": np.asarray(b.position).tolist(),
                       "half_extents": np.asarray(b.size).tolist()} for b in self.boxes],
            "config": self.config,
        }


def box_surface(half: np.ndarray, spacing: float) -> np.ndarray:
    """Points on the six faces of a box of half extents ``half``, box frame."""
    half = np.asarray(half, dtype=float)
    out = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        nu = max(int(np.ceil(2 * half[u] / spacing)) + 1, 2)
        nv = max(int(np.ceil(2 * half[v] / spacing)) + 1, 2)
        gu, gv = np.meshgrid(np.linspace(-half[u], half[u], nu),
                             np.linspace(-half[v], half[v], nv), indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((gu.size, 3))
            face[:, u], face[:, v], face[:, axis] = gu.ravel(), gv.ravel(), sign * half[axis]
            out.append(face)
    return np.vstack(out)


# ----------------------------------------------------------------- the cloud
def robot_mask(points: np.ndarray, q, fr3, end_effector=None, opening: float = 0.08,
               margin: float = SELF_FILTER_MARGIN) -> np.ndarray:
    """Which observed points are the robot itself, at joint configuration ``q``.

    ``perception.obstacles.self_filtered`` for the real arm: each link's hull and
    the hand's boxes, grown by ``margin``. It must be the configuration the cloud
    was captured in.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    hit = np.zeros(len(points), dtype=bool)
    if not len(points):
        return hit
    poses = fr3.link_poses(q)
    for hull, T in fr3.posed_hulls(poses=poses):
        hit |= hull.contains(points, T, tolerance=-abs(margin))
    if end_effector is not None:
        from tpgpt.hardware.fr3 import TCP_LINK

        T = poses[TCP_LINK]
        g_R = end_effector.grasp_from_ee(T[:3, :3])
        g_p = T[:3, 3] + T[:3, :3] @ end_effector.tool_offset
        local = (points - g_p) @ g_R
        for box in end_effector.boxes(opening):
            hit |= box.contains(local, margin=abs(margin))
    return hit


@dataclass
class ShelfWorld:
    """Everything a path is checked against: the shelf model and what is seen on it.

    Attributes:
        model: The shelf. ``None`` checks the cloud alone.
        cloud: Observed obstacle points inside the shelf's region, target and
            robot removed. ``(0, 3)`` when there is no scene cloud.
        cloud_report: What was kept and removed, and how much of the observed
            shelf the model explains.
    """

    model: ShelfModel | None
    cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cloud_report: dict = field(default_factory=dict)
    #: The table's top surface, an exact horizontal plane (``None``: no table).
    table_height: float | None = None
    _surface: tuple | None = None
    _trees: dict = field(default_factory=dict)

    @classmethod
    def build(cls, model: ShelfModel | None, scene_points: np.ndarray | None = None,
              target_points: np.ndarray | None = None, robot_q=None, fr3=None,
              end_effector=None, margin: float = ROI_MARGIN,
              above_support: float = ROI_ABOVE_SUPPORT,
              target_exclusion: float = TARGET_EXCLUSION, scope: str = "visible",
              support_height: float | None = None,
              table_band: float = TABLE_BAND) -> "ShelfWorld":
        """Turn an observed scene cloud into obstacle points.

        Args:
            scene_points: The whole observed scene, base frame, or ``None``.
            target_points: The object being picked; points near it are removed.
            robot_q: Joint angles when the scene was captured. ``None`` means the
                robot cannot be removed, so ``scope`` falls back to ``"shelf"``.
            scope: ``"visible"`` keeps everything seen within reach (module
                docstring); ``"shelf"`` only what lies in the shelf's region.
            support_height: The table's height; points within ``table_band``
                above it (and anything below it) are the table and removed.
        """
        from scipy.spatial import cKDTree

        if scope not in ("visible", "shelf"):
            raise ValueError(f"scope must be 'visible' or 'shelf', not {scope!r}")
        report: dict = {"model": model is not None,
                        "model_measured": bool(model.measured) if model else False}
        pts = np.zeros((0, 3))
        if scene_points is not None and len(scene_points):
            pts = np.atleast_2d(np.asarray(scene_points, dtype=float))
            report["scene_points"] = int(len(pts))
            if scope == "visible" and (robot_q is None or fr3 is None):
                scope = "shelf"
                report["scope_fallback"] = ("no joint angles at capture time: the robot cannot "
                                            "be removed, so only the shelf's region is kept")
            report["scope"] = scope
            if scope == "shelf" and model is not None:
                pts = pts[model.in_roi(pts, margin, above_support)]
                report["in_shelf_region"] = int(len(pts))
            elif scope == "visible":
                reach = np.linalg.norm(pts - np.asarray(SHOULDER), axis=1) <= REACH_RADIUS
                report["outside_reach"] = int((~reach).sum())
                pts = pts[reach]
            if support_height is not None and len(pts):
                table = pts[:, 2] < float(support_height) + float(table_band)
                report["removed_as_table"] = int(table.sum())
                pts = pts[~table]
            if target_points is not None and len(target_points) and len(pts):
                d, _ = cKDTree(np.asarray(target_points, dtype=float)).query(pts)
                keep = d > target_exclusion
                report["removed_as_target"] = int((~keep).sum())
                pts = pts[keep]
            if robot_q is not None and fr3 is not None and len(pts):
                mine = robot_mask(pts, robot_q, fr3, end_effector)
                report["removed_as_robot"] = int(mine.sum())
                pts = pts[~mine]
            elif len(pts):
                report["removed_as_robot"] = None
                report["robot_not_removed"] = "no joint angles at capture time"
            if model is not None and len(pts):
                on_shelf = model.in_roi(pts, margin, above_support)
                explained = model.explained(pts)
                if on_shelf.any():
                    report["explained_by_model"] = float(explained[on_shelf].mean())
                    report["unexplained_points"] = int((on_shelf & ~explained).sum())
                # The shelf is the model's job: its own points are removed.
                report["removed_as_shelf"] = int(explained.sum())
                pts = pts[~explained]
        report["obstacle_points"] = int(len(pts))
        return cls(model=model, cloud=pts, cloud_report=report,
                   table_height=None if support_height is None else float(support_height))

    # ------------------------------------------------------------ cached pieces
    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        if self._surface is None:
            self._surface = (self.model.surface_points() if self.model is not None
                             else (np.zeros((0, 3)), np.zeros(0, dtype=int)))
        return self._surface

    def tree(self, which: str):
        """KD-trees over the model surface (``"surface"``) or the cloud (``"cloud"``)."""
        from scipy.spatial import cKDTree

        if which not in self._trees:
            pts = self.surface()[0] if which == "surface" else self.cloud
            self._trees[which] = cKDTree(pts) if len(pts) else None
        return self._trees[which]

    def near(self, points: np.ndarray, reach: float) -> bool:
        """Cheap prune: could anything in ``points`` (plus ``reach``) touch this world?"""
        if self.table_height is not None and \
                float(np.min(np.atleast_2d(points)[:, 2])) <= self.table_height + reach:
            return True
        if self.model is not None and self.model.near(points, reach):
            return True
        tree = self.tree("cloud")
        if tree is None:
            return False
        d, _ = tree.query(np.atleast_2d(points))
        return bool(np.min(d) <= reach)

    def to_dict(self) -> dict:
        return {"model": self.model.to_dict() if self.model is not None else None,
                "table_height": self.table_height, "cloud_report": self.cloud_report}


__all__ = ["TAPE_KEYS", "resolve_tape", "REACH_RADIUS", "TABLE_BAND", "ROI_ABOVE_SUPPORT", "ROI_MARGIN", "ShelfModel", "ShelfWorld", "box_surface", "robot_mask"]


# ------------------------------------------------ the placement, from the cloud
def apply_cloud_fit(left, right, fit: dict):
    """The front edge's two ends moved by a ``cloud_fit`` correction.

    The correction is expressed in the shelf frame as the tape and touched
    points place it: a yaw about the front edge's midpoint, then a shift along
    the front edge (``shift_along_front_mm``, towards ``front_right``) and into
    the shelf (``shift_into_shelf_mm``). Heights are never moved -- they were
    touched.
    """
    left, right = np.asarray(left, dtype=float)[:2], np.asarray(right, dtype=float)[:2]
    x = (right - left) / np.linalg.norm(right - left)
    y = np.array([-x[1], x[0]])
    pivot = (left + right) / 2.0
    a = np.radians(float(fit.get("yaw_deg", 0.0)))
    Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shift = (x * float(fit.get("shift_along_front_mm", 0.0))
             + y * float(fit.get("shift_into_shelf_mm", 0.0))) / 1000.0
    move = lambda p: pivot + Rz @ (p - pivot) + shift  # noqa: E731
    return move(left), move(right)


def _faces(model: ShelfModel, points: np.ndarray, inset: float = 0.0, camera=None):
    """Each point's signed distance to the model and the face it is nearest.

    Returns ``(distance (n,), box index (n,), normal in the shelf frame (n, 3))``;
    distance > 0 is on the outer side of the face. Exact for boxes, the only kind
    the model builds.

    With ``camera`` (base frame), only faces turned towards it are candidates: a
    point the camera saw lies on a face it can see. Without this an 18 mm panel
    can be matched from either side, and a fit may slide it through the points,
    matching its outer face's points with its hidden inner face (it did, on the
    first live run, 2026-09-22).

    With ``inset`` > 0 the normal is zero for points that do not project onto
    their face at least that far inside its edges: a point just past a board's
    front edge is nearest the front face whether the board is really there or
    6 mm further back, so it tells nothing about which.
    """
    n_pts = len(points)
    best = np.full(n_pts, np.inf)
    box_of = np.zeros(n_pts, dtype=int)
    normal = np.zeros((n_pts, 3))
    for k, box in enumerate(model.boxes):
        local = (points - box.position) @ box.rotation
        half = np.asarray(box.size)
        for axis in range(3):
            others = [u for u in range(3) if u != axis]
            for sign in (-1.0, 1.0):
                outward = box.rotation[:, axis] * sign
                if camera is not None:
                    centre = box.position + outward * half[axis]
                    if float(outward @ (np.asarray(camera, dtype=float) - centre)) <= 0.0:
                        continue
                plane = sign * local[:, axis] - half[axis]
                past = np.abs(local[:, others]) - half[others]
                d = np.sqrt(plane ** 2 + (np.maximum(past, 0.0) ** 2).sum(axis=1))
                d = np.where(plane < 0.0, -d, d)
                closer = np.abs(d) < np.abs(best)
                best = np.where(closer, d, best)
                box_of = np.where(closer, k, box_of)
                n = outward @ model.R_base_shelf
                square = past.max(axis=1) <= -inset if inset > 0.0 else np.ones(n_pts, bool)
                normal[closer] = 0.0
                normal[closer & square] = n
    return best, box_of, normal


_FACE_WORDS = {(0, 1): "right face", (0, -1): "left face", (1, 1): "back face",
               (1, -1): "front face", (2, 1): "top", (2, -1): "underside"}
_WALL_WORDS = {("left_wall", "right face"): "inner face", ("left_wall", "left face"): "outer face",
               ("right_wall", "left face"): "inner face", ("right_wall", "right face"): "outer face"}


def _face_table(model: ShelfModel, points: np.ndarray, capture: float, camera=None) -> dict:
    """Per face: how many points lie on it, and their median signed offset."""
    d, box_of, n = _faces(model, points, camera=camera)
    near = np.abs(d) <= capture
    axis = np.abs(n).argmax(axis=1)
    sign = np.sign(n[np.arange(len(n)), axis]).astype(int)
    out = {}
    for k, box in enumerate(model.boxes):
        for (ax, sg), word in _FACE_WORDS.items():
            on = near & (box_of == k) & (axis == ax) & (sign == sg) & (np.abs(n).max(axis=1) > 0.5)
            if on.sum():
                word = _WALL_WORDS.get((box.name, word), word)
                out[f"{box.name} {word}"] = (int(on.sum()), float(np.median(d[on])))
    return out


def _implied_tape(model: ShelfModel) -> dict:
    """M1-M4 as this model places the shelf relative to the touched slot marks, mm."""
    a = model.config["anchors"]
    width = model.width
    local = lambda p: (np.array([*p[:2], 0.0]) - model.origin) @ model.R_base_shelf  # noqa: E731
    L, R = local(a["left_slot"]), local(a["right_slot"])
    return {"M1": L[0] * 1000.0, "M2": (width - R[0]) * 1000.0,
            "M3": L[1] * 1000.0, "M4": R[1] * 1000.0}


def fit_to_cloud(config: dict, points: np.ndarray, camera=None) -> dict:
    """Measure the shelf's x/y placement and yaw against the observed cloud.

    The shelf's dimensions come from the tape and its heights from touching, and
    both are trusted. Its **placement** in x/y and its yaw rest on the four tape
    readings from the slot marks (M1-M4), the hardest to hold square. The camera
    sees the shelf's faces directly, so this finds the rigid correction -- a yaw
    about the front edge's midpoint and a shift in the floor plane, never a
    height -- that puts the model's faces on the observed points: robust least
    squares (Cauchy, scale :data:`FIT_NOISE`) of each point's distance to the
    nearest face, over the points within :data:`FIT_CAPTURE` of the model, then
    again within half of it.

    A direction is fitted only if the cloud constrains it (the board tops say
    nothing about x/y): the shift along the front edge needs
    :data:`FIT_MIN_FACE_POINTS` on faces square to it (the panels' sides), the
    shift into the shelf as many on front faces (board edges, the apron, the
    panels' front edges), and the yaw front faces spread over
    :data:`FIT_MIN_YAW_SPAN`. The rest stay at the tape's value.

    What would make the result wrong: the robot or an object against a panel
    within the capture distance (use an empty shelf, arm out of view); a
    calibration error, which moves the cloud and so the shelf with it (the
    calibration was validated to ~5 mm on these boards, CALIBRATION.md).

    Args:
        config: The ``shelf:`` block (any ``cloud_fit`` in it is ignored: the fit
            is made from the tape and touched placement, so it is absolute, not
            incremental).
        points: The observed scene, base frame, robot removed.
        camera: The camera's position, base frame. Points are then matched only
            to faces turned towards it (:func:`_faces`). Always pass it on
            real data.

    Returns:
        A report: ``cloud_fit`` (the block to store in ``shelf.yaml``, or
        ``None`` if refused), ``refused``, which directions were fitted, the
        per-face offsets before and after, and M1-M4 as the fit implies them.
    """
    from scipy.optimize import least_squares

    c = dict(config.get("shelf", config))
    previous = c.pop("cloud_fit", None)
    model = ShelfModel.from_config({"shelf": c})
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    table = float(model.config["bottom"])
    pts = pts[pts[:, 2] > table + TABLE_BAND]
    pts = pts[model.in_roi(pts, FIT_CAPTURE, above_support=0.0)]
    local = (pts - model.origin) @ model.R_base_shelf
    pivot = np.array([model.width / 2.0, 0.0])

    def moved(theta):
        """Points in the frame of the model moved by theta = (along, into, yaw):
        moving the model by T is moving the points by T^-1."""
        a, b, yaw = theta
        q = local.copy()
        c_, s_ = np.cos(-yaw), np.sin(-yaw)
        xy = q[:, :2] - pivot - np.array([a, b])
        q[:, :2] = pivot + xy @ np.array([[c_, -s_], [s_, c_]]).T
        return model.origin + q @ model.R_base_shelf.T

    d0, _, n0 = _faces(model, pts, inset=FIT_NOISE, camera=camera)
    on = np.abs(d0) <= FIT_CAPTURE
    axis = np.where(np.abs(n0).max(axis=1) > 0.5, np.abs(n0).argmax(axis=1), -1)
    side_pts = int((on & (axis == 0)).sum())
    front = on & (axis == 1)
    front_pts = int(front.sum())
    span = float(np.ptp(local[front, 0])) if front.any() else 0.0
    free = np.array([side_pts >= FIT_MIN_FACE_POINTS, front_pts >= FIT_MIN_FACE_POINTS,
                     front_pts >= FIT_MIN_FACE_POINTS and span >= FIT_MIN_YAW_SPAN])
    report = {"points_in_region": int(len(pts)), "points_on_model": int(on.sum()),
              "side_face_points": side_pts, "front_face_points": front_pts,
              "front_span_mm": span * 1000.0,
              "fitted": {"along_front": bool(free[0]), "into_shelf": bool(free[1]),
                         "yaw": bool(free[2])},
              "previous_cloud_fit": previous, "faces_before": _face_table(model, pts, FIT_CAPTURE, camera),
              "implied_tape_before_mm": _implied_tape(model)}
    theta = np.zeros(3)
    if free.any():
        for capture in (FIT_CAPTURE, FIT_CAPTURE / 2.0):
            d, _, _ = _faces(model, moved(theta), camera=camera)
            use = np.abs(d) <= capture
            if use.sum() < FIT_MIN_FACE_POINTS:
                break

            def residual(v, use=use):
                full = theta.copy()
                full[free] = v
                q = moved(full)[use]
                return _faces(model, q, camera=camera)[0]

            sol = least_squares(residual, theta[free], loss="cauchy", f_scale=FIT_NOISE,
                                x_scale=np.array([0.01, 0.01, 0.01])[free], diff_step=1e-3)
            theta[free] = sol.x
    fit = {"shift_along_front_mm": round(float(theta[0]) * 1000.0, 1),
           "shift_into_shelf_mm": round(float(theta[1]) * 1000.0, 1),
           "yaw_deg": round(float(np.degrees(theta[2])), 2)}
    shift = float(np.hypot(theta[0], theta[1]))
    fitted = ShelfModel.from_config({"shelf": {**c, "cloud_fit": fit}})
    for key, m in (("median_abs_before_mm", model), ("median_abs_after_mm", fitted)):
        d = np.abs(_faces(m, pts, camera=camera)[0])
        near = d <= FIT_CAPTURE / 2.0
        report[key] = float(np.median(d[near]) * 1000.0) if near.any() else None
    refused = None
    after = report["median_abs_after_mm"]
    faces_after = _face_table(fitted, pts, FIT_CAPTURE, camera)
    # The faces that decide the placement (sides and fronts, not the board tops,
    # whose heights were touched) must each land on their points.
    disagree = {k: v[1] * 1000.0 for k, v in faces_after.items()
                if v[0] >= FIT_MIN_FACE_POINTS and ("face" in k)
                and abs(v[1]) > FIT_FACE_TOLERANCE}
    if shift > FIT_MAX_SHIFT or abs(np.degrees(theta[2])) > FIT_MAX_YAW_DEG:
        refused = (f"correction of {shift * 1000:.0f} mm / {np.degrees(theta[2]):+.1f} deg is beyond "
                   f"any tape error ({FIT_MAX_SHIFT * 1000:.0f} mm / {FIT_MAX_YAW_DEG} deg): check "
                   "the calibration, the arm or an object against a panel, and the tape keys")
    elif not free.any():
        refused = "the cloud does not constrain the shelf's placement (too few points on its faces)"
    elif disagree:
        refused = ("after the fit these faces are still off (median, mm): "
                   + ", ".join(f"{k} {v:+.1f}" for k, v in disagree.items())
                   + ". One rigid move cannot put every face on its points, so the tape's "
                   "dimensions and the cloud disagree (or the camera is off at the shelf): "
                   "find out which before storing anything")
    elif after is None or after > FIT_MAX_MEDIAN * 1000.0:
        refused = (f"the fitted model still sits {after} mm (median) from the shelf's points, more "
                   f"than {FIT_MAX_MEDIAN * 1000:.0f} mm: it does not match what the camera sees")
    report.update({"refused": refused, "cloud_fit": None if refused else fit, "fit": fit,
                   "faces_after": faces_after,
                   "implied_tape_after_mm": _implied_tape(fitted)})
    return report


def describe_fit(report: dict) -> str:
    """The report of :func:`fit_to_cloud` as lines a person can check."""
    f, lines = report["fit"], []
    lines.append(f"points: {report['points_in_region']} in the shelf region, "
                 f"{report['points_on_model']} within {FIT_CAPTURE * 1000:.0f} mm of the model "
                 f"({report['side_face_points']} on side faces, {report['front_face_points']} on "
                 f"front faces spanning {report['front_span_mm']:.0f} mm)")
    fixed = [k for k, v in report["fitted"].items() if not v]
    lines.append(f"correction: {f['shift_along_front_mm']:+.1f} mm along the front edge, "
                 f"{f['shift_into_shelf_mm']:+.1f} mm into the shelf, {f['yaw_deg']:+.2f} deg"
                 + (f"  (not fitted, kept at the tape's: {', '.join(fixed)})" if fixed else ""))
    lines.append(f"median |distance| of the shelf's points to the model: "
                 f"{report['median_abs_before_mm']:.1f} mm -> {report['median_abs_after_mm']:.1f} mm")
    lines.append("M1-M4 (tape as used -> as the cloud places the shelf), mm:")
    for k in ("M1", "M2", "M3", "M4"):
        lines.append(f"  {k}: {report['implied_tape_before_mm'][k]:6.1f} -> "
                     f"{report['implied_tape_after_mm'][k]:6.1f}")
    lines.append("faces seen (points, median signed offset mm, + = cloud outside the panel): "
                 "before -> after")
    before, after = report["faces_before"], report["faces_after"]
    for face in sorted(set(before) | set(after), key=lambda k: -before.get(k, (0, 0))[0])[:15]:
        b, a = before.get(face, (0, float("nan"))), after.get(face, (0, float("nan")))
        if max(b[0], a[0]) >= 30:
            lines.append(f"  {face:28s} {b[0]:6d} {b[1] * 1000:+6.1f}  ->  {a[0]:6d} "
                         f"{a[1] * 1000:+6.1f}")
    lines.append("REFUSED: " + report["refused"] if report["refused"] else
                 "store as shelf.cloud_fit in config/shelf.yaml: " + str(report["cloud_fit"]))
    return "\n".join(lines)
