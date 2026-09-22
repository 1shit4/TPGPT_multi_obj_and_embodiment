"""The full object cloud, from the object's YCB scan fitted to what the camera sees.

**Why.** The simulation's YCB experiment gives the method the object's *full*
scan at its true pose instead of the cameras' one-sided cloud
(``tpgpt.experiments.pipeline.run``: "The object's cloud is the full scan; the
scene's stays observed"). The object cloud decides which grasps exist; a
one-sided, top-dominated cloud pushed hands onto rims and edges, and replacing
it halved the runs that never lifted the object (``docs/sim/ycb_grid.md``).
The robot does not know the true pose, so it is estimated here by fitting the
scan to the camera's partial cloud. The scene cloud (what the arm is checked
against) stays observed, as in the simulation.

**How.** The object rests on the table, so its pose is a *resting orientation*
(which face is down) plus a position on the plane and a heading. The candidate
resting orientations are the scan's own frame (the YCB scans are taken standing
on a turntable, so it is a real resting pose) and every stable face of the
scan's convex hull (one whose centre of mass projects inside it). For each, a
grid of headings is scored, the best few are refined by robust ICP in
``(x, y, z, heading)``, and the best refined fit is kept. Its residuals are
measured two ways:

* the camera's points must lie on the model (``median_mm``, ``explained``);
* the model must not stand in space the camera measured as empty
  (``in_front``): along each ray the camera's own surface is the nearest thing
  there is, so a model point closer than it, by more than the depth noise, is
  where the camera saw nothing. Rays the camera has no reading for (a mug's
  dark interior) say nothing and are skipped; ``seen`` reports the fraction of
  the model's visible surface that the camera did measure, as information
  rather than as a test.

  Both use a z-buffer -- the nearest model point along each ray from the
  camera, binned at the depth camera's own pixel -- rather than a normal test,
  because a mug's far wall faces the camera and is hidden behind its near
  wall.

A fit that fails any threshold is **refused** and the planner falls back to
the camera cloud, saying so in the plan.

**Height.** ``z`` is fitted so the camera's height error (the table reads
4.9 mm low, ``CALIBRATION.md``) does not bias the heading and position, then
the model is set down on the touched support height, where the real object is.
``dz_mm`` records the difference.

What this cannot fix: the camera's calibration error moves the fitted object
with it (about 10 mm sideways near the shelf, ``CALIBRATION.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Planner object name -> YCB directory under :data:`SCAN_ROOT`.
YCB_SCANS = {
    "banana": "011_banana",
    "apple": "013_apple",
    "mug": "025_mug",
    "tomato_soup_can": "005_tomato_soup_can",
}

#: The simulation's download location (``tpgpt.sim.ycb.YCB_ROOT``), resolved
#: against the repository rather than the working directory.
SCAN_ROOT = Path(__file__).resolve().parents[2] / "assets" / "ycb"

#: Points in the returned full cloud: the simulation's ``surface_cloud`` count.
FULL_CLOUD_POINTS = 4096

#: Model points used while fitting.
FIT_POINTS = 2000

#: Headings scored per resting orientation, and how many of the best
#: (orientation, heading) starts are refined.
HEADINGS = 24
REFINE_STARTS = 6
ICP_ITERATIONS = 30

#: Robust scale of the fit, metres: the camera's depth noise on an object at
#: about 0.8 m.
NOISE = 0.004

#: Angular bin of the visibility z-buffer, degrees: the D435's pixel at
#: 848 x 480 over a 69 deg field of view.
VISIBILITY_BIN_DEG = 0.12

#: A resting face is a distinct one if its normal differs from every kept face's
#: by more than this, degrees; at most :data:`MAX_RESTS` are tried.
REST_SEPARATION_DEG = 20.0
MAX_RESTS = 12

#: Acceptance. A fit is refused if the camera's points sit further from the
#: model than ``MAX_MEDIAN`` (median), if fewer than ``MIN_EXPLAINED`` of them
#: are within ``EXPLAINED_DISTANCE``, or if more than ``MAX_IN_FRONT`` of the
#: model's visible surface stands in front of what the camera measured along
#: the same ray (by more than ``SEEN_DISTANCE``).
#:
#: **Set from measurements, not from synthetic noise.** On simulated views
#: (3-5 mm noise) a correct fit reads about 3 mm, 0.93-0.99 explained and
#: 0.81-1.00 seen. The first real object (a YCB banana, 2026-09-22) read
#: **5.0 mm, 0.78, 0.82**, with the fitted scan sitting on its silhouette in
#: the camera image: the camera's depth of the object's own surface disagrees
#: with the scan resting on the touched table by about 5 mm, which no pose
#: fixes. The wrong object, measured on every pair of these four scans, is
#: caught by the fractions rather than the median: its best case (a mug's scan
#: on an apple) reads 3.9 mm but only 0.50 seen, and the next (an apple's scan
#: on a can) 0.62 explained. So the median is generous and the fractions do the
#: rejecting.
MAX_MEDIAN = 0.007
EXPLAINED_DISTANCE = 0.008
MIN_EXPLAINED = 0.75
SEEN_DISTANCE = 0.010
MAX_IN_FRONT = 0.15


@dataclass
class ObjectModel:
    """One YCB scan: its surface, stable resting orientations and centre of mass."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    centre_of_mass: np.ndarray
    rests: list = field(default_factory=list)       # (3, 3) rotations, face down on z=0

    @classmethod
    def load(cls, name: str, root: Path | str = SCAN_ROOT) -> "ObjectModel":
        from tpgpt.sim.ycb import read_stl_faces

        if name not in YCB_SCANS:
            raise KeyError(f"no YCB scan for {name!r}; known: {sorted(YCB_SCANS)}")
        path = Path(root) / YCB_SCANS[name] / "google_16k" / "nontextured.stl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: download {YCB_SCANS[name]}_google_16k.tgz from the YCB "
                "benchmark site into assets/ycb/")
        vertices, faces = read_stl_faces(path)
        model = cls(name=name, vertices=vertices, faces=faces,
                    centre_of_mass=_volume_centroid(vertices, faces))
        model.rests = model._resting_orientations()
        return model

    def sample(self, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Points uniform by area over the scan, and their outward normals.

        The same sampling as ``tpgpt.sim.ycb.surface_cloud`` (without its
        centring, which only moves the body origin).
        """
        tri = self.vertices[self.faces]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        cross = np.cross(b - a, c - a)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(self.faces), size=count, p=areas / areas.sum())
        u, v = rng.random((count, 1)), rng.random((count, 1))
        over = (u + v) > 1.0
        u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
        points = a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])
        normals = cross[pick] / np.maximum(np.linalg.norm(cross[pick], axis=1, keepdims=True),
                                           1e-12)
        return points, normals

    def _resting_orientations(self) -> list:
        """The scan's own frame, then each stable face of its convex hull."""
        from scipy.spatial import ConvexHull

        rests = [np.eye(3)]
        hull = ConvexHull(self.vertices)
        # Group hull facets into planes (coplanar triangles share an equation).
        planes: dict = {}
        for eq, simplex in zip(hull.equations, hull.simplices):
            key = tuple(np.round(eq, 4))
            planes.setdefault(key, set()).update(simplex.tolist())
        candidates = []
        for key, idx in planes.items():
            n, off = np.array(key[:3]), key[3]
            pts = self.vertices[sorted(idx)]
            # The centre of mass must project inside the face for it to rest there.
            R = _rotation_taking(n, np.array([0.0, 0.0, -1.0]))
            flat = (pts @ R.T)[:, :2]
            com = (R @ self.centre_of_mass)[:2]
            if len(flat) < 3 or not _inside_convex(flat, com):
                continue
            area = _polygon_area(flat)
            candidates.append((area, R, n))
        for _, R, n in sorted(candidates, key=lambda t: -t[0]):
            if len(rests) >= MAX_RESTS:
                break
            down = [r.T @ np.array([0.0, 0.0, -1.0]) for r in rests]
            if all(np.degrees(np.arccos(np.clip(n @ d, -1, 1))) > REST_SEPARATION_DEG
                   for d in down):
                rests.append(R)
        return rests


@dataclass
class ObjectFit:
    """A fitted scan and how well it matches the camera."""

    name: str
    accepted: bool
    reason: str | None
    rotation: np.ndarray
    position: np.ndarray
    rest_index: int
    heading_deg: float
    median_mm: float
    explained: float
    seen: float
    in_front: float
    dz_mm: float
    observed_points: int
    dropped_below_table: int = 0
    full_cloud: np.ndarray | None = None

    def summary(self) -> dict:
        return {"object": self.name, "accepted": self.accepted, "reason": self.reason,
                "rest_index": self.rest_index, "heading_deg": round(self.heading_deg, 1),
                "position": np.round(self.position, 4).tolist(),
                "median_mm": round(self.median_mm, 2), "explained": round(self.explained, 3),
                "seen": round(self.seen, 3), "in_front": round(self.in_front, 3),
                "dz_mm": round(self.dz_mm, 1),
                "observed_points": self.observed_points,
                "dropped_below_table": self.dropped_below_table}


def fit_object(model: ObjectModel, observed: np.ndarray, support_height: float,
               camera, seed: int = 0) -> ObjectFit:
    """Fit ``model`` to the camera's ``observed`` points (base frame).

    ``camera`` is the camera's position (base frame), for the facing test.
    Returns the fit with ``full_cloud`` (:data:`FULL_CLOUD_POINTS` points on the
    scan, set down on ``support_height``) whether or not it was accepted, so the
    caller can show a refused fit.
    """
    from scipy.spatial import cKDTree

    obs = np.atleast_2d(np.asarray(observed, dtype=float))
    # Points below the table are the mask's edge landing on the table beyond the
    # object, plus depth noise. The object cannot be there, and they drag the fit
    # down (16 of 1393 on the first real banana).
    below = obs[:, 2] < float(support_height) - 0.002
    obs = obs[~below]
    camera = np.asarray(camera, dtype=float)
    tree_obs = cKDTree(obs)
    pts, normals = model.sample(FIT_POINTS, seed)
    centre_obs = np.median(obs, axis=0)

    def placed(rest, heading, t):
        R = _rot_z(heading) @ rest
        body = pts @ R.T
        return body, normals @ R.T, R

    def score(body, nrm, t):
        world = body + t
        d_obs = cKDTree(world).query(obs)[0]
        facing = _visible(world, camera)
        d_face = tree_obs.query(world[facing])[0] if facing.any() else np.zeros(1)
        rho = lambda d: np.log1p((d / NOISE) ** 2)  # noqa: E731  Cauchy
        return float(rho(d_obs).mean() + 0.5 * rho(np.minimum(d_face, 0.03)).mean())

    # ---- coarse: every resting orientation x heading, centred on the cloud ----
    starts = []
    for k, rest in enumerate(model.rests):
        for heading in np.linspace(0.0, 2 * np.pi, HEADINGS, endpoint=False):
            body, nrm, _ = placed(rest, heading, None)
            t = centre_obs - np.median(body, axis=0)
            t[2] = support_height - body[:, 2].min()
            starts.append((score(body, nrm, t), k, heading, t))
    starts.sort(key=lambda s: s[0])

    # ---- refine the best few with robust ICP in (x, y, z, heading) ----
    best = None
    for _, k, heading, t in starts[:REFINE_STARTS]:
        for _ in range(ICP_ITERATIONS):
            body, nrm, _ = placed(model.rests[k], heading, t)
            world = body + t
            d, j = cKDTree(world).query(obs)
            w = 1.0 / (1.0 + (d / NOISE) ** 2)
            src, dst = world[j], obs               # move model points onto observed
            facing = _visible(world, camera)
            if facing.any():
                d2, j2 = tree_obs.query(world[facing])
                keep = d2 < 0.03
                src = np.vstack([src, world[facing][keep]])
                dst = np.vstack([dst, obs[j2[keep]]])
                w = np.concatenate([w, 0.5 / (1.0 + (d2[keep] / NOISE) ** 2)])
            dtheta, dxy, dz = _planar_step(src, dst, w, pivot=t)
            heading += dtheta
            t = t + np.array([dxy[0], dxy[1], dz])
            if abs(dtheta) < 1e-4 and np.linalg.norm(dxy) < 1e-5 and abs(dz) < 1e-5:
                break
        body, nrm, _ = placed(model.rests[k], heading, t)
        s = score(body, nrm, t)
        if best is None or s < best[0]:
            best = (s, k, heading, t)

    _, k, heading, t = best
    body, nrm, R = placed(model.rests[k], heading, t)
    world = body + t
    d_obs = cKDTree(world).query(obs)[0]
    facing = _visible(world, camera)
    d_face = tree_obs.query(world[facing])[0] if facing.any() else np.zeros(0)
    median = float(np.median(d_obs))
    explained = float((d_obs <= EXPLAINED_DISTANCE).mean())
    seen, in_front = _free_space(world[facing], obs, camera)
    # Set it down where the real object is: on the touched support.
    resting_z = support_height - body[:, 2].min()
    dz = float(t[2] - resting_z)
    position = np.array([t[0], t[1], resting_z])

    reason = None
    if in_front > MAX_IN_FRONT:
        reason = (f"{in_front:.0%} of the scan's visible surface stands in front of what the "
                  f"camera measured along the same ray (max {MAX_IN_FRONT:.0%}): the scan is too "
                  "big, or placed through space the camera saw as empty")
    elif median > MAX_MEDIAN:
        reason = f"camera points sit {median * 1000:.1f} mm from the scan (median; max {MAX_MEDIAN * 1000:.0f})"
    elif explained < MIN_EXPLAINED:
        reason = (f"only {explained:.0%} of the camera points lie within "
                  f"{EXPLAINED_DISTANCE * 1000:.0f} mm of the scan (min {MIN_EXPLAINED:.0%})")
    full, _ = model.sample(FULL_CLOUD_POINTS, seed + 1)
    return ObjectFit(name=model.name, accepted=reason is None, reason=reason, rotation=R,
                     position=position, rest_index=k,
                     heading_deg=float(np.degrees(heading) % 360.0), median_mm=median * 1000.0,
                     explained=explained, seen=seen, in_front=in_front, dz_mm=dz * 1000.0,
                     observed_points=int(len(obs)), dropped_below_table=int(below.sum()), full_cloud=full @ R.T + position)


def _free_space(model_visible: np.ndarray, observed: np.ndarray, camera: np.ndarray,
                bin_deg: float = VISIBILITY_BIN_DEG, tolerance: float = SEEN_DISTANCE):
    """How much of the model's visible surface the camera measured, and how much
    of it stands in front of that measurement.

    Returns ``(seen, in_front)`` as fractions of the visible surface. A ray with
    no camera reading counts in neither: it is unknown, not empty.
    """
    if not len(model_visible) or not len(observed):
        return 0.0, 0.0

    def rays(p):
        d = p - camera
        r = np.linalg.norm(d, axis=1)
        return d / np.maximum(r, 1e-9)[:, None], r

    u_m, r_m = rays(model_visible)
    u_o, r_o = rays(observed)
    fwd = u_m.mean(axis=0)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd)
    step = np.radians(bin_deg)

    def key(u):
        return (np.round(np.arctan2(u @ right, u @ fwd) / step).astype(np.int64) * 1_000_003
                + np.round(np.arctan2(u @ up, u @ fwd) / step).astype(np.int64))

    k_m, k_o = key(u_m), key(u_o)
    # Nearest camera reading in each ray bin.
    order = np.argsort(k_o)
    k_sorted, r_sorted = k_o[order], r_o[order]
    edges = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    bins = k_sorted[edges]
    nearest = np.minimum.reduceat(r_sorted, edges)
    idx = np.searchsorted(bins, k_m)
    idx = np.clip(idx, 0, len(bins) - 1)
    has = bins[idx] == k_m
    delta = r_m - nearest[idx]                     # < 0: the model is in front
    seen = float((has & (np.abs(delta) <= tolerance)).mean())
    in_front = float((has & (delta < -tolerance)).mean())
    return seen, in_front


def _visible(points: np.ndarray, camera: np.ndarray,
             bin_deg: float = VISIBILITY_BIN_DEG) -> np.ndarray:
    """Which of ``points`` the camera could see: a z-buffer over angular bins.

    The nearest point along each ray wins; everything behind it is hidden by
    the object itself. Bins are the depth camera's own pixel, so the test has
    the resolution of the measurement it will be compared against.
    """
    ray = points - camera
    dist = np.linalg.norm(ray, axis=1)
    u = ray / np.maximum(dist, 1e-9)[:, None]
    # Two angles about the mean viewing direction are enough: the object
    # subtends a few degrees.
    fwd = u.mean(axis=0)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd)
    step = np.radians(bin_deg)
    key = (np.round(np.arctan2(u @ right, u @ fwd) / step).astype(np.int64) * 1_000_003
           + np.round(np.arctan2(u @ up, u @ fwd) / step).astype(np.int64))
    order = np.lexsort((dist, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    seen = np.zeros(len(points), dtype=bool)
    seen[order[first]] = True
    return seen


# ------------------------------------------------------------------ geometry
def _planar_step(src, dst, w, pivot):
    """Weighted best rotation about z (about ``pivot``), xy shift and z shift
    taking ``src`` onto ``dst``."""
    w = w / w.sum()
    ps, pd = (src - pivot)[:, :2], (dst - pivot)[:, :2]
    ms, md = w @ ps, w @ pd
    a, b = ps - ms, pd - md
    num = float((w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum())
    den = float((w * (a * b).sum(axis=1)).sum())
    theta = np.arctan2(num, den)
    c, s = np.cos(theta), np.sin(theta)
    shift = md - np.array([[c, -s], [s, c]]) @ ms
    dz = float(w @ (dst[:, 2] - src[:, 2]))
    # The rotation is about the pivot, which the caller keeps as the body's
    # translation: rotating the body about it and shifting is exactly what
    # ``heading += theta, t += shift`` does, since the body is centred on t.
    return theta, shift, dz


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_taking(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1.0 + 1e-9:                        # opposite: half turn about any perpendicular
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _volume_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Centre of mass of the closed mesh (uniform density), by signed tetrahedra."""
    tri = vertices[faces]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    if abs(vol.sum()) < 1e-12:
        return vertices.mean(axis=0)
    return (vol[:, None] * tri.sum(axis=1) / 4.0).sum(axis=0) / vol.sum()


def _inside_convex(poly: np.ndarray, p: np.ndarray) -> bool:
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(poly)
    except Exception:  # degenerate (collinear) face
        return False
    return bool(np.all(hull.equations[:, :2] @ p + hull.equations[:, 2] <= 1e-9))


def _polygon_area(poly: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    try:
        return float(ConvexHull(poly).volume)
    except Exception:
        return 0.0
