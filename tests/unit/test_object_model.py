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
    ("banana", 0, 0.7), ("banana", 1, 2.3), ("apple", 0, 1.1), ("mug", 0, 2.0),
    ("tomato_soup_can", 0, 0.4), ("pudding_box", 2, 1.3), ("foam_brick", 0, 2.7)])
def test_the_scan_lands_on_the_object(name, rest, heading):
    model = ObjectModel.load(name)
    R, t = _placed(model, rest, heading, (0.52, 0.18))
    fit = fit_object(model, _view(model, R, t), TABLE, CAMERA)
    assert fit.accepted, fit.reason
    assert _surface_gap(model, R, t, fit) < 0.004
    assert len(fit.full_cloud) == 4096
    # Set down on the table, where the real object is.
    assert fit.full_cloud[:, 2].min() == pytest.approx(TABLE, abs=5e-4)


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
    for k in PROTOCOL_RESTS["pudding_box"]:       # standing on an end
        assert np.ptp((pts @ np.asarray(box.rests[k]).T)[:, 2]) == pytest.approx(0.112, abs=0.004)


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
