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


def _table_touch(R, xy, label="", finger="index", true=TRUE):
    pose, link = list(im.POINT_POSES[finger]), im.TOUCH_LINKS[finger]
    probe = im.Touch("table", [TABLE], [0.0, 0.0, 0.0], R.tolist(), pose, label, link=link)
    zmin = im.touch_points(probe, true)[:, 2].min()
    return im.Touch("table", [TABLE], [xy[0], xy[1], TABLE - zmin], R.tolist(), pose, label,
                    link=link)


def _point_touch(R, target, label="", finger="index", true=TRUE):
    pose, link = list(im.POINT_POSES[finger]), im.TOUCH_LINKS[finger]
    probe = im.Touch("point", list(target), [0.0, 0.0, 0.0], R.tolist(), pose, label, link=link)
    tip = im.touch_residual(probe, true) + np.asarray(target)        # tip with the TCP at 0
    return im.Touch("point", list(target), (np.asarray(target) - tip).tolist(), R.tolist(),
                    pose, label, link=link)


def _touches(fingers=("index", "little"), true=TRUE):
    """Six table touches per finger, tilted, and the two slot marks with the first finger."""
    Rs = _orientations(14)
    table = [_table_touch(Rs[6 * j + k], (0.45 + 0.02 * k, -0.1 + 0.03 * k), f"t{k}{f[0]}",
                          finger=f, true=true)
             for j, f in enumerate(fingers) for k in range(6)]
    points = [_point_touch(Rs[12], (0.55, 0.20, 0.33), "top_left", fingers[0], true),
              _point_touch(Rs[13], (0.60, -0.20, 0.33), "top_right", fingers[0], true)]
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
    assert fit.held == [] and len(fit.fingers) == 2      # both fingers: nothing held
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


def test_one_finger_cannot_tell_yaw_from_lateral_so_the_offset_is_held():
    """The live fit of 2026-10-05 (index only): yaw +-405 deg, offset +-160 mm. With one
    finger the fit holds the offset at its start and says so; it then recovers the yaw and
    the thickness of a hand that is centred."""
    centred = (TRUE[0], TRUE[1], 0.0, 0.0)
    touches = _touches(fingers=("index",), true=centred)
    fit = im.fit_mount(touches)
    assert fit.held == ["lateral_x", "lateral_y"] and fit.fingers == ["index_intermediate"]
    assert any("HELD" in n for n in fit.notes)
    assert fit.params["mount_yaw_deg"] == pytest.approx(TRUE[0], abs=0.05)
    assert fit.params["adapter_thickness"] == pytest.approx(TRUE[1], abs=5e-5)
    assert "HELD" in fit.describe(touches)
    # Freed, the four are not all determined: the yaw's spread is wide.
    free = im.fit_mount(touches, hold=())
    assert not free.held and free.stderr["mount_yaw_deg"] > 1.0 or free.condition > 100


def test_the_pointing_finger_is_read_from_the_registers():
    assert im.pointing_finger([0, 0, 0, 998, 1000, 989]) == "index"     # a live touch
    assert im.pointing_finger([1000, 0, 0, 0, 1000, 1000]) == "little"
    assert im.pointing_finger([1000, 1000, 1000, 1000, 1000, 0]) is None


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
