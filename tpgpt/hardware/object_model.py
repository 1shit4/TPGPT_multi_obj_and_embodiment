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
  wall. The comparison is made only where it means something: away from the
  model's own silhouettes and rims, and on surface facing the camera within
  :data:`MAX_INCIDENCE_DEG`.

  **A thin-walled concave object may never pass**, and the YCB mug does not:
  its correct fit (3.8 mm from a circle fitted to its wall, and a better
  median and explained fraction than any placement made by hand) still reads
  0.42, because rays through its opening meet its own inner and outer walls a
  few millimetres apart. Refusing such a fit costs little -- the planner falls
  back to the camera cloud and still plans -- while accepting a wrong one is
  expensive, so the test stays strict.

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
from scipy.spatial import cKDTree

#: Planner object name -> YCB directory under :data:`SCAN_ROOT`.
YCB_SCANS = {
    "banana": "011_banana",
    "apple": "013_apple",
    "mug": "025_mug",
    "tomato_soup_can": "005_tomato_soup_can",
    "pudding_box": "008_pudding_box",       # Jell-O pudding; scan 112 x 90 x 39, ours 88 x 70 x 35
    "foam_brick": "061_foam_brick",         # 78 x 53 x 51 mm
}

#: The bench's own object where it is not the size of its YCB scan: its edges, metres,
#: longest first, measured by tape. The scan is scaled along its own box edges to match
#: (:func:`_scaled_to_edges`), so the fit, the grasp library (whose cache key is the
#: scaled surface) and everything downstream see the object on the table. For boxes only.
#: The pudding box (user, 2026-10-03) is a smaller pack of the YCB one: 88 x 70 x 35 mm
#: against the scan's 112.5 x 89.7 x 38.9. With the scan, the camera's box was 30-40 mm
#: shorter than the model on every capture (80-84 mm long at mid-height), so the fit had
#: to add that length at one end or the other and could not tell which; on 17:32 it chose
#: the end facing the camera, where the camera saw nothing, the plan put the jaws 20 mm
#: inside that end, and the jaws closed on air beside the real box.
MEASURED_EDGES = {"pudding_box": (0.088, 0.070, 0.035)}
#: The same for an object without flat faces: its extents along the scan's own x, y and z,
#: metres, scaled about the middle of the scan's box. The apple (user, 2026-10-03) is about
#: 70 mm across and 72 mm tall against the YCB scan's 75.4 x 74.9 x 71.9; z is the scan's
#: stem axis (5-6 mm dimples at both ends of it, none along x or y). The camera agreed
#: before the user's careful measurement did: on its 15 captures a 70 x 67 apple explained a
#: median 0.76 of the cloud against 0.60 for the scan, and an 80 x 75 one 0.57. The YCB
#: apple left 2.5 mm each side in the 80 mm open hand, so grasps across its middle were
#: refused as too wide (32 of the 352 refused ahead of 10-03 20:24's chosen grasp, which
#: closed 25 mm off the apple's centre, on its cap, and pushed it away).
MEASURED_EXTENTS = {"apple": (0.070, 0.070, 0.072)}

#: Resting poses the experiment's protocol allows (user, 2026-09-25): the mug is set
#: down opening-up (rest 0; rest 2 is the same shape upside down, and from this
#: camera the two differ only inside the cup); the pudding box lies on its long narrow
#: side, its 88 x 35 mm face, 70 mm tall (rests 1 and 3 of the measured box,
#: :data:`MEASURED_EDGES`; the camera's box is 80-84 x 25-40 mm at mid-height on every
#: capture since). The banana lies with its stem on the table (rest 0; user, 2026-10-03):
#: flipped (rest 1) the stem end is 9.4 mm up, the middle on the table either way, and
#: the camera cannot tell the two apart -- on 42 of its 51 captures the two fits explain
#: within 0.03 of each other, at headings up to 41 deg apart (19:56: 238 vs 197 deg, 0.92
#: each), so a banana lying the other way had its jaws planned across it 40 deg off.
#: An object not listed may rest any way. Indices are into
#: :meth:`ObjectModel._resting_orientations`, which is deterministic for a given scan;
#: ``test_object_model`` pins them.
PROTOCOL_RESTS = {"mug": (0,), "pudding_box": (1, 3), "banana": (0,)}
#: A protocol object whose camera cloud reaches this far above the fitted pose's top
#: is refused: the fit may only use the protocol poses, so it cannot refuse the
#: wrong one on its own, and a box stood on its end would be planned as a box on
#: its side. Measured 2026-09-25 (98th percentile of the cloud against the fitted
#: top): the upright mug -24 to +16 mm (stray points at the rim). Not for other
#: objects: the banana carries a specular layer ~20 mm above itself.
MAX_HEIGHT_EXCESS = 0.020
#: Per object, where the camera reads its top high. The pudding box's silver face reads
#: 14-33 mm above the box when it is up (a band of points continuous with the box, over
#: its footprint, 20-35 % of its cloud), -3 to +8 mm when it is down. Measured
#: 2026-10-03 on every capture against the measured box lying 70 mm tall: lying, 29
#: captures, -2.7 to +33.3 mm; stood on its end (the protocol before 09-25 evening, and
#: 10-03 18:14) +43.8 to +49; a cloud with something else in it +90 to +135. Between
#: 33 and 44, 38. The banana's skin reads -20 to +19 mm (median +7) on its 52 captures on
#: the table, stem down; its two faces differ by 1.6 mm in height, so the test cannot tell
#: them apart and only guards against something else in the cloud: 30. A box stood on its end with its silver face down would read about
#: +19 and is not caught. The layer misled the pose: read as the box's height it made the
#: box look 90 mm tall, which the full-size scan on its side matched (89.7) and the
#: measured box matched standing (86-87) -- the latter briefly the protocol here (10-03,
#: 2a196a4), until the user's 18:14 run.
HEIGHT_EXCESS = {"pudding_box": 0.038, "banana": 0.030}

#: Camera points further than this *behind* the fitted surface, along their own
#: ray, are refused as impossible and the pose is fitted again without them. The
#: camera cannot see through an opaque object; what reads as if it had is a depth
#: artefact. Measured 2026-09-25 on a glossy red mug: 15-30 % of its cloud sat
#: 70-200 mm from its axis, 7-21 mm above the table and ~80 mm beyond the mug
#: body along the ray -- the infrared pattern bouncing inside the cup -- and the fit
#: turned the handle to point away from the camera, onto those points, on 50 of
#: 68 fits, and tilted the upright mug 13 deg.
OCCLUSION_TOLERANCE = 0.020
#: Open vessels: the occlusion pass runs for them. Only an open vessel shows the
#: artefact, and only there is the pass measured to help: on the day's captures of
#: the banana, apple and can it moved validated fits with nothing to say whether for
#: the better.
OPEN_VESSELS = ("mug",)
#: Objects that rest on a flat face, fitted without the rest tilt: on a flat table
#: they cannot lean. Measured 2026-09-25 on the planner's own live fits: the foam
#: brick leaned 19.5 and 19.3 deg on two of three runs (resting on an edge, as it
#: looked), the upright pudding box 11.7 and 12.0 deg on both of its plans into
#: bottom_left, and the mug 11-22 deg. The tilt stays for what can really rest
#: rolled or leaning: the apple, the can. (On a synthetic box with no artefacts the
#: tilt search was worth 2.5 mm of fit against 5.5 -- the live fits are the evidence
#: that counts.) **The banana since 2026-10-03.** Rolled about its long axis it lifted
#: its middle off the table, which a banana on its side cannot do: on its 49 recorded
#: captures the tilted fit rolled it up to 29 deg (most 5-21) and lifted the middle
#: 5-15 mm, chasing the band the camera reads over its skin, while fitted flat it
#: explains as much or more on 48 (median 0.68 -> 0.81, the other 0.73 vs 0.72). On
#: 10-03 19:33 the plan centred the jaws on the lifted middle, 10 mm above the real
#: banana, and they closed on air with the arm at the planned grasp to 2 mm.
FLAT_RESTING = ("mug", "pudding_box", "foam_brick", "banana")
#: The occlusion test is a depth image of the placed scan: :data:`OCCLUSION_POINTS`
#: on its surface, the nearest per angular bin of :data:`OCCLUSION_BIN_DEG` (1.5 mm
#: at 1.1 m), each camera point compared with its own bin only. A neighbourhood
#: of bins is wrong here: looking into a cup it always contains the rim, so the
#: cup's real interior reads as "behind" (2105 of 3018 points on the first try).
OCCLUSION_BIN_DEG = 0.08
OCCLUSION_POINTS = 40000

#: The simulation's download location (``tpgpt.sim.ycb.YCB_ROOT``), resolved
#: against the repository rather than the working directory.
SCAN_ROOT = Path(__file__).resolve().parents[2] / "assets" / "ycb"

#: Points in the returned full cloud: the simulation's ``surface_cloud`` count.
FULL_CLOUD_POINTS = 4096

#: Model points used while fitting.
FIT_POINTS = 2000

#: Headings scored per resting orientation, and how many of the best
#: (orientation, heading) starts are refined -- per centring (:data:`FACING_CENTRING`).
HEADINGS = 24
REFINE_STARTS = 6
ICP_ITERATIONS = 30

#: Each coarse start is placed twice: the scan's whole median on the cloud's median, and
#: (shifted this many times) the median of the scan's camera-facing surface on it. The
#: camera sees only the near side, so the cloud's median lies toward the camera from the
#: object's centre; centred by its whole median, a scan at the right heading sits off the
#: seen faces and can score worse than a wrong heading whose corner covers them, and only
#: the best starts are refined. Measured 2026-10-02: the pudding box on its long side was
#: fitted ~50 deg off on all four captures of 10-01/10-02 (median 5.9-6.0 mm, 64-66 %
#: explained, accepted), so the plan's jaws met the real box across a corner; started from
#: the facing surface too, 4.1-4.5 mm and 81-84 %. The best :data:`REFINE_STARTS` of each
#: centring are refined, each centring's winner is finished (rest tilt included) and the
#: fit's own score picks between the two. On the 45 recorded captures that score picked the
#: better-explained of the two on 14 of the 15 where they differ (the other: 0.87 vs 0.88);
#: banana, mug and brick fits unchanged. Synthetic views of the box at 36 placements: 10
#: off by 4.1-7.3 mm (surface gap) before, none after.
FACING_CENTRING = 2

#: Robust scale of the fit, metres: the camera's depth noise on an object at
#: about 0.8 m.
NOISE = 0.004

#: Angular bin of the visibility z-buffer, degrees: the D435's pixel at
#: 848 x 480 over a 69 deg field of view.
VISIBILITY_BIN_DEG = 0.12

#: How far the object may be tilted off a resting face of its hull, degrees.
#: A banana rests on its two ends and can be rolled about its long axis; a can
#: can lean. The hull's faces are a discrete set and the real rest is a
#: continuum, so the fit is allowed this much freedom, always keeping the
#: object down on the table (:func:`_rest_tilt`).
MAX_REST_TILT_DEG = 25.0

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
#:
#: ``MIN_EXPLAINED`` is 0.60 because that is what this camera delivers, and how
#: much it varies. Two captures of the same motionless banana, fifteen minutes
#: apart, put 11% and 25% of their points off the object (a layer about 20 mm
#: above it, dense, and therefore not removable by any outlier filter -- both a
#: radius and a statistical filter leave it untouched). Correct fits accordingly
#: read 0.68 to 0.91. Refusing the low ones is worse than accepting them,
#: because the fallback is the one-sided cloud that carries the same layer *and*
#: has no back: on the live banana the fitted scan planned at rank 3 while the
#: camera cloud produced nothing. The other two numbers still bound the pose:
#: the median must be within 7 mm and the scan must not stand in measured free
#: space. These test consistency, not exactness -- a 10 mm shift *along* a
#: banana's axis is not distinguishable by any of them, and does not matter,
#: since the camera itself is good to about 8 mm here.
#:
#: Measured on seven captures of four objects, every one a correct fit, judged
#: at the set-down pose: median 4.3-7.3 mm, explained 0.53-0.79, in front
#: 0.011-0.035. The limits (8 mm, 0.50, 0.25) sit just outside that envelope.
#: The same fits displaced 15 mm read 6.7-11.7 mm and 0.32-0.59, which the
#: limits refuse for the can, apple and mug. They do not refuse it for the
#: banana: a shift along a banana's axis leaves the cloud almost unchanged, so
#: nothing here can see it.
#:
#: **``SEEN_DISTANCE`` and ``MAX_IN_FRONT`` are an error budget, not a preference.**
#: On simulated views a correct fit puts *nothing* in front (0.00). On this
#: bench it cannot, even where the comparison is well conditioned (see
#: :data:`MAX_INCIDENCE_DEG`): the camera reads a surface about 5 mm out
#: (``CALIBRATION.md``), and two independent estimators of the same can's axis
#: -- this fit and a circle fitted to its side wall -- disagree by 7.6 mm, with
#: the fit's own cost preferring its answer. So a ray tolerance below about
#: 15 mm measures the camera, not the pose. Measured on the captures of
#: 2026-09-23 at a 15 mm tolerance: correct fits read 0.04 (banana), 0.05
#: (apple) and 0.13 (can), while a mug fitted into a tilted resting pose read
#: 0.49 and the synthetic wrong-object pairs a median of 0.24. Hence 0.25.
#:
#: **What actually catches a bad fit is the pair (median, explained), and the
#: failure that matters is a wrong pose of the right scan** -- the object's name
#: comes from segmentation, so fitting the wrong scan is not a real case.
#: Measured on the day's four captures: displacing a correct fit by 10 mm takes
#: the median from 4.5-4.9 to 5.3-7.3 mm and the explained fraction from
#: 0.76-0.84 to 0.55-0.74, which is refused on every object, while ``in_front``
#: barely moves (0.00-0.03 to 0.01-0.05). So ``in_front`` is the weak member of
#: the three and is kept only as a guard against a scan standing in open space.
MAX_MEDIAN = 0.008
EXPLAINED_DISTANCE = 0.008
MIN_EXPLAINED = 0.50
SEEN_DISTANCE = 0.015
MAX_IN_FRONT = 0.25


@dataclass
class ObjectModel:
    """One YCB scan: its surface, stable resting orientations and centre of mass."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    centre_of_mass: np.ndarray
    rests: list = field(default_factory=list)       # (3, 3) rotations, face down on z=0
    #: Indices into ``rests`` the fit may use; ``None`` = all (:data:`PROTOCOL_RESTS`).
    allowed_rests: tuple | None = None

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
        if name in MEASURED_EDGES:
            vertices = _scaled_to_edges(vertices, faces, MEASURED_EDGES[name])
        if name in MEASURED_EXTENTS:
            middle = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
            vertices = middle + (vertices - middle) * (np.asarray(MEASURED_EXTENTS[name])
                                                       / np.ptp(vertices, axis=0))
        model = cls(name=name, vertices=vertices, faces=faces,
                    centre_of_mass=_volume_centroid(vertices, faces))
        model.rests = model._resting_orientations()
        model.allowed_rests = PROTOCOL_RESTS.get(name)
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
    #: Camera points refused as lying behind the fitted surface (:data:`OCCLUSION_TOLERANCE`).
    dropped_occluded: int = 0

    def summary(self) -> dict:
        return {"object": self.name, "accepted": self.accepted, "reason": self.reason,
                "rest_index": self.rest_index, "heading_deg": round(self.heading_deg, 1),
                "position": np.round(self.position, 4).tolist(),
                "median_mm": round(self.median_mm, 2), "explained": round(self.explained, 3),
                "seen": round(self.seen, 3), "in_front": round(self.in_front, 3),
                "dz_mm": round(self.dz_mm, 1),
                "observed_points": self.observed_points,
                "dropped_below_table": self.dropped_below_table,
                "dropped_occluded": self.dropped_occluded}


def fit_object(model: ObjectModel, observed: np.ndarray, support_height: float,
               camera, seed: int = 0) -> ObjectFit:
    """Fit ``model`` to the camera's ``observed`` points (base frame).

    For the objects in :data:`OPEN_VESSELS`, fitted twice: once on every
    point, then again without the points that lie
    behind that first fit's surface along their own camera ray
    (:data:`OCCLUSION_TOLERANCE`). A camera cannot see through an opaque object,
    so such a point is a depth artefact -- on a glossy cup, a large share of the
    cloud -- and left in, it drags the pose (see the constant). The second fit's
    ``dropped_occluded`` says how many were refused. Only the resting poses in
    ``model.allowed_rests`` are tried (:data:`PROTOCOL_RESTS`).

    ``camera`` is the camera's position (base frame), for the facing test.
    Returns the fit with ``full_cloud`` (:data:`FULL_CLOUD_POINTS` points on the
    scan, set down on ``support_height``) whether or not it was accepted, so the
    caller can show a refused fit.
    """
    first = _fit_once(model, observed, support_height, camera, seed)
    if model.name not in OPEN_VESSELS:
        return _check_height(model, np.atleast_2d(np.asarray(observed, dtype=float)), first)
    obs = np.atleast_2d(np.asarray(observed, dtype=float))
    dense, _ = model.sample(OCCLUSION_POINTS, seed + 2)
    placed_scan = dense @ first.rotation.T + first.position
    behind = _occluded(placed_scan, obs, np.asarray(camera, dtype=float))
    if not behind.any() or behind.sum() > 0.5 * len(obs):
        return _check_height(model, obs, first)
    second = _fit_once(model, obs[~behind], support_height, camera, seed)
    second.dropped_occluded = int(behind.sum())
    return _check_height(model, obs, second)


def _check_height(model: ObjectModel, observed: np.ndarray, fit: ObjectFit) -> ObjectFit:
    """Refuse a protocol object whose cloud stands taller than its fitted pose."""
    if getattr(model, "allowed_rests", None) is None or not fit.accepted or not len(observed):
        return fit
    excess = float(np.percentile(observed[:, 2], 98) - fit.full_cloud[:, 2].max())
    if excess > HEIGHT_EXCESS.get(model.name, MAX_HEIGHT_EXCESS):
        fit.accepted = False
        fit.reason = (f"the camera's points reach {excess * 1000:.0f} mm above the fitted "
                      f"{model.name}, which may only rest as the protocol sets it down "
                      f"(rests {model.allowed_rests}): is it standing another way?")
    return fit


def _occluded(model_world: np.ndarray, observed: np.ndarray, camera: np.ndarray,
              bin_deg: float = OCCLUSION_BIN_DEG,
              tolerance: float = OCCLUSION_TOLERANCE) -> np.ndarray:
    """Which ``observed`` points lie more than ``tolerance`` behind ``model_world``.

    Per ray bin, the nearest model point is the surface the camera would have hit
    first -- a depth image of the placed scan. An observation further along the
    same ray than that, by more than ``tolerance``, was hidden by the object.
    Bins with no model point say nothing: the point is kept.
    """
    if not len(model_world) or not len(observed):
        return np.zeros(len(observed), dtype=bool)

    def rays(p):
        d = p - camera
        r = np.linalg.norm(d, axis=1)
        return d / np.maximum(r, 1e-9)[:, None], r

    u_m, r_m = rays(model_world)
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
    order = np.argsort(k_m)
    ks, rs = k_m[order], r_m[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    bins, nearest = ks[edges], np.minimum.reduceat(rs, edges)
    idx = np.clip(np.searchsorted(bins, k_o), 0, len(bins) - 1)
    front = np.where(bins[idx] == k_o, nearest[idx], np.inf)
    return r_o > front + tolerance


def _fit_once(model: ObjectModel, observed: np.ndarray, support_height: float,
              camera, seed: int = 0) -> ObjectFit:
    """One fit of ``model`` to ``observed`` (see :func:`fit_object`).

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

    # ---- coarse: every resting orientation x heading, centred on the cloud twice ----
    # (by the scan's whole median and by its camera-facing surface: FACING_CENTRING)
    whole, facing_starts = [], []
    allowed = getattr(model, "allowed_rests", None)
    for k, rest in enumerate(model.rests):
        if allowed is not None and k not in allowed:
            continue
        for heading in np.linspace(0.0, 2 * np.pi, HEADINGS, endpoint=False):
            body, nrm, _ = placed(rest, heading, None)
            t = centre_obs - np.median(body, axis=0)
            t[2] = support_height - body[:, 2].min()
            whole.append((score(body, nrm, t), k, heading, t))
            tf = t.copy()
            for _ in range(FACING_CENTRING):
                seen_now = _visible(body + tf, camera)
                if seen_now.any():
                    tf[:2] += centre_obs[:2] - np.median((body + tf)[seen_now], axis=0)[:2]
            facing_starts.append((score(body, nrm, tf), k, heading, tf))

    # ---- refine the best few of each with robust ICP in (x, y, z, heading) ----
    finalists = []
    for group in (whole, facing_starts):
        group.sort(key=lambda s: s[0])
        finalists.append(_refine(group[:REFINE_STARTS], model, obs, tree_obs, camera, placed,
                                 score))

    # ---- finish each centring's best (the rest tilt included), keep the lower score ----
    # (chosen after the tilt: chosen before it, an apple's other start won on its untilted
    # score and ended worse, 55 % explained against 60 %)
    best = None
    for _, k, heading, t in finalists:
        if model.name in FLAT_RESTING:
            # On a flat face on a flat table the object cannot lean; the tilt search
            # there only fitted depth artefacts (see FLAT_RESTING).
            R_k, t_k = _rot_z(heading) @ model.rests[k], t
        else:
            R_k, t_k = _rest_tilt(model, obs, _rot_z(heading) @ model.rests[k], t,
                                  support_height)
        s = score(pts @ R_k.T, normals @ R_k.T, t_k)
        if best is None or s < best[0]:
            best = (s, k, R_k, t_k)
    _, k, R, t = best
    body = pts @ R.T
    nrm = normals @ R.T
    heading = float(np.arctan2(R[1, 0], R[0, 0]))
    world = body + t
    d_obs = cKDTree(world).query(obs)[0]
    facing = _visible(world, camera)
    d_face = tree_obs.query(world[facing])[0] if facing.any() else np.zeros(0)
    median = float(np.median(d_obs))
    explained = float((d_obs <= EXPLAINED_DISTANCE).mean())
    seen, in_front = _free_space(world[facing], nrm[facing], obs, camera)
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


def _refine(starts, model, obs, tree_obs, camera, placed, score):
    """Robust ICP in (x, y, z, heading) from each of ``starts`` (``(score, rest, heading,
    t)``); the lowest-scoring result as ``(score, rest, heading, t)``."""
    from scipy.spatial import cKDTree

    best = None
    for _, k, heading, t in starts:
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
    return best


#: Where the free-space test is trustworthy. A model point seen edge-on, or one
#: sitting on the model's own silhouette or a rim, tells nothing: a millimetre of
#: lateral error there changes which surface is nearest by centimetres. Measured
#: 2026-09-23 on the day's captures: dropping those took a correctly fitted can
#: from 0.11 to 0.07 and an apple from 0.17 to 0.05, while a wrong scan stayed
#: at 0.14-0.83.
MAX_INCIDENCE_DEG = 60.0
MAX_EDGE_JUMP = 0.010


def _rest_tilt(model, observed, R, t, support_height, max_tilt_deg=MAX_REST_TILT_DEG):
    """Let the object tilt a little, staying on the table, and keep it if it fits better.

    The resting orientations are the scan's own frame and its hull's stable
    faces, which are a discrete set; a real object rests in a continuum. A
    banana supported at its two ends can be rolled about its long axis by tens
    of degrees, and a can can lean. Measured 2026-09-23: this took a can from
    0.79 to 0.85 explained and an apple from 0.87 to 0.89, and it is the only
    thing that explains the points a rolled banana puts above its scan's
    bounding height.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    pts, _ = model.sample(FIT_POINTS, 0)
    tree = cKDTree(observed)

    def place(x):
        turned = Rotation.from_rotvec(x[:3]).as_matrix() @ R
        body = pts @ turned.T
        return turned, body, np.array([t[0] + x[3], t[1] + x[4],
                                       float(support_height) - body[:, 2].min()])

    def residual(x):
        _, body, shift = place(x)
        over = max(0.0, np.degrees(np.linalg.norm(x[:3])) - max_tilt_deg)
        return np.r_[tree.query(body + shift)[0], 0.05 * over]

    before = float(np.median(tree.query(pts @ R.T + t)[0]))
    sol = least_squares(residual, np.zeros(5), loss="cauchy", f_scale=NOISE,
                        x_scale=[0.1, 0.1, 0.1, 0.01, 0.01], diff_step=1e-3)
    turned, body, shift = place(sol.x)
    after = float(np.median(tree.query(body + shift)[0]))
    return (turned, shift) if after <= before else (R, t)


def _free_space(model_visible: np.ndarray, normals_visible: np.ndarray, observed: np.ndarray,
                camera: np.ndarray, bin_deg: float = VISIBILITY_BIN_DEG,
                tolerance: float = SEEN_DISTANCE):
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

    # Nearest camera reading in each ray bin, then over a 3 x 3 block of bins.
    #
    # The block matters: one bin is 1.7 mm across at 0.8 m, and on a surface seen
    # edge-on (a can's or a mug's wall) a lateral millimetre is centimetres of
    # depth, so a single-bin comparison called correctly fitted cylinders "in
    # front" (17-30% on the first live runs) while reading 1-6% on the rounder
    # apple and banana. A model point is in front only when every camera reading
    # around it is clearly further away.
    k_m, k_o = key(u_m), key(u_o)
    # The model's own depth spread around each point: large at a silhouette or a rim.
    om = np.argsort(k_m)
    kms, rms = k_m[om], r_m[om]
    em = np.flatnonzero(np.r_[True, kms[1:] != kms[:-1]])
    mbins, mnear = kms[em], np.minimum.reduceat(rms, em)
    incidence = np.degrees(np.arccos(np.clip(-(u_m * normals_visible).sum(axis=1), -1.0, 1.0)))
    spread = np.zeros(len(k_m))
    order = np.argsort(k_o)
    k_sorted, r_sorted = k_o[order], r_o[order]
    edges = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    bins = k_sorted[edges]
    nearest_in_bin = np.minimum.reduceat(r_sorted, edges)
    best = np.full(len(k_m), np.inf)
    has = np.zeros(len(k_m), dtype=bool)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            probe = k_m + du * 1_000_003 + dv
            idx = np.clip(np.searchsorted(bins, probe), 0, len(bins) - 1)
            hit = bins[idx] == probe
            has |= hit
            best = np.where(hit, np.minimum(best, nearest_in_bin[idx]), best)
            j = np.clip(np.searchsorted(mbins, probe), 0, len(mbins) - 1)
            mine = mbins[j] == probe
            spread = np.where(mine, np.maximum(spread, np.abs(r_m - mnear[j])), spread)
    has &= (incidence <= MAX_INCIDENCE_DEG) & (spread <= MAX_EDGE_JUMP)
    delta = r_m - np.where(has, best, r_m)         # < 0: the model is in front
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


def _scaled_to_edges(vertices: np.ndarray, faces: np.ndarray, edges) -> np.ndarray:
    """The scan scaled along its own box edges so that they measure ``edges``, longest
    first (:data:`MEASURED_EDGES`), about the middle of its extent; its frame is kept.

    The edge directions are the eigenvectors of the hull's area-weighted normals,
    ``sum a_i n_i n_i^T``: a box's hull is six faces, two per edge direction, so its
    face normals are the edges whatever bulges and bevels the scan has.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(vertices)
    tri = vertices[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    normals = hull.equations[:, :3]
    _, axes = np.linalg.eigh((normals * area[:, None]).T @ normals)
    along = vertices @ axes
    lo, hi = along.min(axis=0), along.max(axis=0)
    order = np.argsort(-(hi - lo))
    scale = np.empty(3)
    scale[order] = np.asarray(edges, dtype=float) / (hi - lo)[order]
    centre = (lo + hi) / 2.0
    return ((along - centre) * scale + centre) @ axes.T


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
