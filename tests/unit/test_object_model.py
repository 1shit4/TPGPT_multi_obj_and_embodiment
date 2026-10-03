"""The full object cloud: a YCB scan fitted to a one-sided camera view.

The scans are downloaded, not tracked (assets/ycb/ is gitignored), so these
skip without them.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tpgpt.hardware.object_model import (PROTOCOL_RESTS, SCAN_ROOT, YCB_SCANS, ObjectModel,
                                         _rot_z, fit_object)

pytestmark = pytest.mark.skipif(
    not all((SCAN_ROOT / d / "google_16k" / "nontextured.stl").exists() for d in YCB_SCANS.values()),
    reason="YCB scans not downloaded into assets/ycb/")

CAMERA = np.array([1.196, 0.505, 0.860])      # the real camera (camera_calibration.yaml)
TABLE = -0.0136


def _view(model, R, t, noise=0.003, seed=0):
    """What the camera sees: the nearest surface along each ray (a z-buffer over
    0.12 deg bins, about the D435's pixel at 848 px), with depth noise."""
    pts, _ = model.sample(60000, seed)
    world = pts @ R.T + t
    ray = world - CAMERA
    dist = np.linalg.norm(ray, axis=1)
    u = ray / dist[:, None]
    fwd = np.array([0.55, 0.25, 0.0]) - CAMERA
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    px = np.round(np.arctan2(u @ right, u @ fwd) / np.radians(0.12)).astype(int)
    py = np.round(np.arctan2(u @ up, u @ fwd) / np.radians(0.12)).astype(int)
    key = px * 100000 + py
    order = np.lexsort((dist, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    noisy = dist[sel] + np.random.default_rng(seed + 7).normal(0.0, noise, len(sel))
    return CAMERA + u[sel] * noisy[:, None]


def _placed(model, rest, heading, xy):
    R = _rot_z(heading) @ model.rests[rest]
    body, _ = model.sample(2000)
    t = np.array([xy[0], xy[1], TABLE - (body @ R.T)[:, 2].min()])
    return R, t


def _surface_gap(model, R, t, fit):
    """Mean distance between the true and fitted surfaces, both ways: blind to
    the turn of a symmetric object about its own axis."""
    P, _ = model.sample(4000, 5)
    true, fitted = P @ R.T + t, P @ fit.rotation.T + fit.position
    return float(np.concatenate([cKDTree(fitted).query(true)[0],
                                 cKDTree(true).query(fitted)[0]]).mean())


@pytest.mark.parametrize("name,rest,heading", [
    ("banana", 0, 0.7), ("banana", 0, 2.3), ("apple", 0, 1.1), ("mug", 0, 2.0),
    ("tomato_soup_can", 0, 0.4), ("pudding_box", 1, 4.0), ("foam_brick", 0, 2.7)])
def test_the_scan_lands_on_the_object(name, rest, heading):
    model = ObjectModel.load(name)
    R, t = _placed(model, rest, heading, (0.52, 0.18))
    fit = fit_object(model, _view(model, R, t), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert _surface_gap(model, R, t, fit) < 0.004
    assert len(fit.full_cloud) == 4096
    # Set down on the table, where the real object is.
    assert fit.full_cloud[:, 2].min() == pytest.approx(TABLE, abs=5e-4)


@pytest.mark.parametrize("rest,heading_deg", [(1, 0), (1, 170), (3, 0), (3, 10)])
def test_a_box_seen_from_its_near_side_is_not_turned_onto_its_corner(rest, heading_deg):
    """The camera sees two faces of the box: the cloud's median sits toward the camera,
    and centred there by its whole median the right heading scored worse than a wrong one
    (2026-10-02: the pudding box ~50 deg off on four real captures). Before the search also
    started from the camera-facing surface, these placements fitted 5.9-7.3 mm off."""
    model = ObjectModel.load("pudding_box")
    R, t = _placed(model, rest, np.radians(heading_deg), (0.50, 0.13))
    fit = fit_object(model, _view(model, R, t), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert _surface_gap(model, R, t, fit) < 0.004


def test_the_can_on_its_side_is_found_on_its_side():
    model = ObjectModel.load("tomato_soup_can")
    lying = next(k for k, r in enumerate(model.rests)
                 if abs((r @ [0.0, 0.0, 1.0])[2]) < 0.2)       # axis horizontal
    R, t = _placed(model, lying, 0.9, (0.55, 0.15))
    fit = fit_object(model, _view(model, R, t), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert abs((fit.rotation @ [0.0, 0.0, 1.0])[2]) < 0.2
    assert _surface_gap(model, R, t, fit) < 0.004


def test_a_different_object_is_refused():
    """The camera sees a mug; the banana's scan cannot explain it."""
    mug, banana = ObjectModel.load("mug"), ObjectModel.load("banana")
    R, t = _placed(mug, 0, 1.0, (0.5, 0.2))
    fit = fit_object(banana, _view(mug, R, t), TABLE, CAMERA)
    assert not fit.accepted and fit.reason


def test_a_camera_height_error_is_measured_and_the_object_still_rests_on_the_table():
    """The camera reads the table ~5 mm low (CALIBRATION.md)."""
    model = ObjectModel.load("tomato_soup_can")
    R, t = _placed(model, 0, 0.3, (0.5, 0.2))
    seen = _view(model, R, t) - [0.0, 0.0, 0.005]
    fit = fit_object(model, seen, TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert fit.dz_mm == pytest.approx(-5.0, abs=1.5)
    assert fit.position[2] == pytest.approx(t[2], abs=5e-4)



# ------------------------------------------------ protocol poses and phantoms
def _yaw_error_deg(fit, R):
    d = fit.rotation @ R.T
    return abs(np.degrees(np.arctan2(d[1, 0], d[0, 0])))


def test_the_protocol_poses_are_the_ones_the_protocol_means():
    """Rest indices are deterministic for a scan; pin what they are."""
    mug = ObjectModel.load("mug")
    pts, _ = mug.sample(4096, 1)
    b = pts @ np.asarray(mug.rests[PROTOCOL_RESTS["mug"][0]]).T
    b[:, 2] -= b[:, 2].min()
    axis = b[b[:, 2] < 0.004][:, :2].mean(axis=0)
    r = np.linalg.norm(b[:, :2] - axis, axis=1)
    top = int(((b[:, 2] > b[:, 2].max() - 0.004) & (r < 0.025)).sum())
    bottom = int(((b[:, 2] < 0.004) & (r < 0.025)).sum())
    assert top == 0 and bottom > 100              # opening up: nothing across the top
    box = ObjectModel.load("pudding_box")
    pts, _ = box.sample(4096, 1)
    for k in PROTOCOL_RESTS["pudding_box"]:       # lying on its 88 x 35 face, 70 mm tall
        assert np.ptp((pts @ np.asarray(box.rests[k]).T)[:, 2]) == pytest.approx(0.069, abs=0.003)


def test_the_mug_is_only_fitted_upright_and_does_not_lean():
    mug = ObjectModel.load("mug")
    R, t = _placed(mug, 0, 2.0, (0.52, 0.18))
    fit = fit_object(mug, _view(mug, R, t), TABLE, CAMERA)
    assert fit.rest_index == 0
    assert fit.rotation[:, 2] @ np.asarray(mug.rests[0])[:, 2] == pytest.approx(1.0, abs=1e-6) or \
        (fit.rotation @ np.asarray(mug.rests[0]).T)[2, 2] == pytest.approx(1.0, abs=1e-6)


def test_points_behind_the_mug_are_dropped_and_the_handle_is_still_found():
    """The glossy-cup artefact (2026-09-25): a quarter of the cloud read ~80 mm too far."""
    mug = ObjectModel.load("mug")
    R, t = _placed(mug, 0, 2.0, (0.52, 0.18))
    view = _view(mug, R, t)
    rng = np.random.default_rng(3)
    phantom = rng.random(len(view)) < 0.25
    ray = view[phantom] - CAMERA
    view[phantom] += ray / np.linalg.norm(ray, axis=1)[:, None] * 0.08
    fit = fit_object(mug, view, TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert fit.dropped_occluded >= 0.8 * phantom.sum()
    assert _yaw_error_deg(fit, R) < 20.0
    assert _surface_gap(mug, R, t, fit) < 0.004


def test_objects_other_than_the_mug_are_fitted_once():
    banana = ObjectModel.load("banana")
    R, t = _placed(banana, 0, 0.7, (0.52, 0.18))
    assert fit_object(banana, _view(banana, R, t), TABLE, CAMERA).dropped_occluded == 0



def test_a_box_seen_almost_face_on_is_placed_flat_but_can_slide_along_that_face():
    """A known limit, kept visible (2026-09-25). With one flat face nearly all the
    camera sees, nothing fixes where the box sits along it: the fit is upright and
    unleaned, but its centre can be 15-20 mm off along the face. The rest tilt
    used to hide this by accident; on live clouds it leaned boxes 12-20 deg."""
    box = ObjectModel.load("pudding_box")
    R, t = _placed(box, PROTOCOL_RESTS["pudding_box"][0], 0.3, (0.52, 0.18))
    fit = fit_object(box, _view(box, R, t), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert fit.rest_index in PROTOCOL_RESTS["pudding_box"]
    lean = fit.rotation @ np.asarray(box.rests[fit.rest_index]).T
    assert lean[2, 2] == pytest.approx(1.0, abs=1e-9)          # flat on the table
    P, _ = box.sample(4000, 5)                             # the box's centre, not the scan
    centre = (P @ fit.rotation.T + fit.position).mean(axis=0)   # origin, which a flip moves
    assert np.linalg.norm(centre - (P @ R.T + t).mean(axis=0)) < 0.025



def _with_silver_top(view, rise=0.025, share=0.25, seed=0):
    """The pudding box's silver face as the camera reads it (2026-10-03): a band of points
    from its top up to ``rise`` above it, over the box, ``share`` of the cloud."""
    rng = np.random.default_rng(seed)
    top = view[view[:, 2] > view[:, 2].max() - 0.004]
    band = top[rng.integers(0, len(top), int(share * len(view)))].copy()
    band[:, 2] += rng.uniform(0.0, rise, len(band))
    return np.vstack([view, band])


def test_a_box_stood_on_its_end_is_refused_not_squeezed_into_the_protocol_pose():
    """Lying on its long narrow side (70 mm) is the protocol. Its silver face reads 14-33 mm
    high, so the box's allowance is 38 mm; stood on its end (88 mm) with the same band it
    reads ~44 and is refused. (With the silver face down it would read ~19 and pass:
    a known gap, HEIGHT_EXCESS.)"""
    box = ObjectModel.load("pudding_box")
    pts = box.sample(2000)[0]
    on_end = next(k for k, r in enumerate(box.rests) if np.ptp((pts @ np.asarray(r).T)[:, 2]) > 0.08)
    R, t = _placed(box, on_end, 0.8, (0.52, 0.18))
    fit = fit_object(box, _with_silver_top(_view(box, R, t)), TABLE, CAMERA)
    assert not fit.accepted
    assert "standing another way" in fit.reason
    R, t = _placed(box, PROTOCOL_RESTS["pudding_box"][0], 0.8, (0.52, 0.18))
    fit = fit_object(box, _with_silver_top(_view(box, R, t)), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert _surface_gap(box, R, t, fit) < 0.006


def test_the_pudding_box_is_the_one_on_the_bench_not_the_ycb_scan():
    """88 x 70 x 35 mm by tape (2026-10-03); the YCB scan is 112.5 x 89.7 x 38.9."""
    box = ObjectModel.load("pudding_box")
    pts = box.sample(4000, 1)[0]
    heights = sorted(np.ptp((pts @ np.asarray(r).T)[:, 2]) for r in box.rests)
    assert heights[0] == pytest.approx(0.035, abs=0.002)
    assert heights[len(heights) // 2] == pytest.approx(0.070, abs=0.003)
    assert heights[-1] == pytest.approx(0.088, abs=0.003)


def test_the_real_box_of_17_32_is_fitted_where_the_camera_saw_it():
    """The capture of 2026-10-03 17:32 (pudding box, top_left). With the full-size scan the
    camera's box was 30-40 mm shorter than the model; the fit put the extra length on the
    end facing the camera, where the camera saw nothing, the jaws were planned 20 mm inside
    it and closed on air. With the measured box the model ends where the camera's box ends."""
    from pathlib import Path
    rec = np.load(Path(__file__).resolve().parents[1] / "data" / "pudding_box_20261003_173210.npz")
    obs = rec["observed"].astype(float)
    box = ObjectModel.load("pudding_box")
    fit = fit_object(box, obs, float(rec["table"]), rec["camera"])
    assert fit.accepted, fit.reason
    assert fit.rest_index in PROTOCOL_RESTS["pudding_box"]
    assert fit.explained > 0.80 and fit.median_mm < 4.5        # 0.88, 3.8; full-size scan 0.75, 4.5
    # Along the box's length (horizontal, through both clouds), the model ends within
    # 10 mm of where the camera's box ends, at both ends.
    model = fit.full_cloud
    flat = model[:, :2] - model[:, :2].mean(axis=0)
    axis = np.linalg.eigh(flat.T @ flat)[1][:, -1]
    m_lo, m_hi = np.percentile(flat @ axis, [1, 99])
    o = (obs[:, :2] - model[:, :2].mean(axis=0)) @ axis
    o_lo, o_hi = np.percentile(o, [1, 99])
    assert abs(m_lo - o_lo) < 0.010 and abs(m_hi - o_hi) < 0.010


def test_the_banana_of_19_33_lies_on_the_table():
    """The capture of 2026-10-03 19:33 (banana, top_left). With the rest tilt the fit rolled
    the banana 14.5 deg about its long axis, after the band the camera reads over its skin,
    lifting its middle 11 mm off the table; the jaws, centred on it, closed 10 mm above the
    real banana. Fitted flat (FLAT_RESTING) it explains more (0.78 against 0.72)."""
    from pathlib import Path
    rec = np.load(Path(__file__).resolve().parents[1] / "data" / "banana_20261003_193336.npz")
    banana = ObjectModel.load("banana")
    fit = fit_object(banana, rec["observed"].astype(float), float(rec["table"]), rec["camera"])
    assert fit.accepted, fit.reason
    assert fit.explained > 0.75
    model = fit.full_cloud
    middle = np.linalg.norm(model[:, :2] - model[:, :2].mean(axis=0), axis=1) < 0.02
    assert model[middle, 2].min() - float(rec["table"]) < 0.002          # its middle on the table


def test_the_banana_lies_stem_down():
    """The protocol (user, 2026-10-03): stem on the table. Of the banana's two lying faces
    only the stem end moves -- on the table in one, 9.4 mm up in the other -- so that end
    is the stem, and the protocol rest has it down."""
    banana = ObjectModel.load("banana")
    pts = banana.sample(20000, 1)[0]
    c = pts - pts.mean(axis=0)
    along = c @ np.linalg.eigh(c.T @ c)[1][:, -1]
    def ends(rest):
        b = pts @ np.asarray(banana.rests[rest]).T
        b[:, 2] -= b[:, 2].min()
        return b[along < along.min() + 0.01, 2].min(), b[along > along.max() - 0.01, 2].min()
    (a0, b0), (a1, b1) = ends(PROTOCOL_RESTS["banana"][0]), ends(1)
    moved = 0 if abs(a1 - a0) > abs(b1 - b0) else 1      # the stem: the end that moves
    assert (a0, b0)[moved] < 0.001                        # on the table in the protocol rest
    assert (a1, b1)[moved] > 0.008                        # in the air when flipped


def test_the_apple_is_the_one_on_the_bench():
    """About 70 mm across and 72 tall (user, 2026-10-03); the YCB scan is 75.4 x 74.9 x 71.9.
    z is the scan's stem axis: its two ends have the dimples."""
    apple = ObjectModel.load("apple")
    assert np.ptp(apple.vertices, axis=0) == pytest.approx([0.070, 0.070, 0.072], abs=1e-4)
    pts = apple.sample(40000, 1)[0]
    q = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    on_axis = np.linalg.norm(q[:, :2], axis=1) < 0.006
    ring = (np.linalg.norm(q[:, :2], axis=1) > 0.014) & (np.linalg.norm(q[:, :2], axis=1) < 0.020)
    for sign in (1.0, -1.0):          # the surface dips toward the stem at both ends of z
        assert np.max(sign * q[ring, 2]) - np.max(sign * q[on_axis & (sign * q[:, 2] > 0), 2]) > 0.003
