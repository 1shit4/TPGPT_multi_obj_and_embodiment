"""Fitting the Inspire hand's mount to touches: it recovers a known mount.

Touches are synthesised from a "true" mount different from the design values, at
wrist orientations tilted in several directions, then fitted from the design values.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.hardware import inspire_mount as im

TRUE = (-41.5, 0.0153, 0.0021, -0.0012)
TABLE = -0.0132


@pytest.fixture(scope="module", autouse=True)
def _needs_description():
    from tpgpt.hardware.inspire_model import load_description
    try:
        load_description()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _orientations(n, seed=0, tilt_deg=25.0):
    """fr3_hand_tcp pointing down (home), tilted up to ``tilt_deg`` about random axes."""
    rng = np.random.default_rng(seed)
    down = np.diag([1.0, -1.0, -1.0])
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis[2] = 0.0
        axis /= np.linalg.norm(axis)
        tilt = Rotation.from_rotvec(axis * np.radians(rng.uniform(10.0, tilt_deg)))
        yaw = Rotation.from_rotvec([0.0, 0.0, rng.uniform(-0.6, 0.6)])
        out.append((tilt * yaw).as_matrix() @ down)
    return out


def _table_touch(R, xy, label=""):
    probe = im.Touch("table", [TABLE], [0.0, 0.0, 0.0], R.tolist(), list(im.POINT_POSE), label)
    zmin = im.touch_points(probe, TRUE)[:, 2].min()
    return im.Touch("table", [TABLE], [xy[0], xy[1], TABLE - zmin], R.tolist(),
                    list(im.POINT_POSE), label)


def _point_touch(R, target, label=""):
    probe = im.Touch("point", list(target), [0.0, 0.0, 0.0], R.tolist(), list(im.POINT_POSE), label)
    tip = im.touch_residual(probe, TRUE) + np.asarray(target)        # tip with the TCP at 0
    return im.Touch("point", list(target), (np.asarray(target) - tip).tolist(), R.tolist(),
                    list(im.POINT_POSE), label)


def _touches():
    Rs = _orientations(8)
    table = [_table_touch(R, (0.45 + 0.02 * k, -0.1 + 0.03 * k), f"t{k}")
             for k, R in enumerate(Rs[:6])]
    points = [_point_touch(Rs[6], (0.55, 0.20, 0.33), "top_left"),
              _point_touch(Rs[7], (0.60, -0.20, 0.33), "top_right")]
    return table + points


def test_the_touches_are_made_as_the_true_mount_says():
    for t in _touches():
        assert np.abs(im.touch_residual(t, TRUE)).max() < 1e-9


def test_the_fit_recovers_the_mount_from_the_design_values():
    fit = im.fit_mount(_touches())
    p = fit.params
    assert p["mount_yaw_deg"] == pytest.approx(TRUE[0], abs=0.05)
    assert p["adapter_thickness"] == pytest.approx(TRUE[1], abs=5e-5)
    assert p["lateral_x"] == pytest.approx(TRUE[2], abs=1e-4)
    assert p["lateral_y"] == pytest.approx(TRUE[3], abs=1e-4)
    assert fit.rms_mm["all"] < 0.01
    assert fit.start["mount_yaw_deg"] == -45.0
    assert "measured: true" in fit.yaml_block()


def test_a_bad_touch_is_shown_not_hidden():
    touches = _touches()
    bad = touches[2]
    touches[2] = im.Touch(bad.kind, [bad.target[0] + 0.008], bad.tcp_position, bad.tcp_rotation,
                          bad.registers, "bad")
    fit = im.fit_mount(touches)
    text = fit.describe(touches)
    assert "disagrees" in text.split("bad")[1].splitlines()[0]


def test_touches_straight_down_cannot_tell_yaw_from_lateral():
    """All touches with the same orientation leave the mount undetermined: the fit says so."""
    R = np.diag([1.0, -1.0, -1.0])
    touches = [_table_touch(R, (0.45 + 0.03 * k, 0.0)) for k in range(6)]
    fit = im.fit_mount(touches)
    assert fit.notes                                   # singular or poorly conditioned


def test_touches_round_trip_through_json(tmp_path):
    import json

    touches = _touches()
    (tmp_path / "t.json").write_text(json.dumps({"touches": [t.as_dict() for t in touches]}))
    back = im.load_touches(tmp_path / "t.json")
    assert [t.as_dict() for t in back] == [t.as_dict() for t in touches]
