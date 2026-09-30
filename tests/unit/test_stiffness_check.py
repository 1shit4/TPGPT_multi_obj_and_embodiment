"""tpgpt.hardware.stiffness_check: jitter and lag measures, on signals with known answers."""

import numpy as np
import pytest

from tpgpt.hardware.stiffness_check import (OscillationWatch, hold_jitter, move_lag,
                                            recent_oscillation, recent_p2p)
from tpgpt.hardware.teach import directional_matrices


def _hold(freq=8.0, amp=0.001, seconds=5.0, rate=250.0, drift=0.002):
    t = np.arange(0.0, seconds, 1.0 / rate)
    x = np.zeros((len(t), 3))
    x[:, 1] = amp * np.sin(2 * np.pi * freq * t) + drift * t / seconds   # vibration + slow drift
    return t, x


def test_hold_jitter_finds_the_vibration_and_ignores_the_drift():
    t, x = _hold()
    j = hold_jitter(t, x, skip=1.0)
    assert j["peak_hz"] == pytest.approx(8.0, abs=0.3)
    assert j["peak_mm"] == pytest.approx(1.0, rel=0.1)
    assert j["p2p_mm"] == pytest.approx(2.0, rel=0.05)      # 2 x amplitude, drift removed
    quiet = hold_jitter(*_hold(amp=0.0))
    assert quiet["p2p_mm"] < 0.01


def test_recent_p2p_ignores_a_smooth_move_and_sees_an_oscillation():
    t = np.arange(0.0, 1.0, 1 / 250)
    move = np.stack([0.05 * t + 0.02 * t ** 2, 0 * t, 0 * t], axis=1)
    assert recent_p2p(t, move) < 1e-6
    shaky = move.copy()
    shaky[:, 2] += 0.002 * np.sin(2 * np.pi * 12 * t)
    # the quadratic fitted over the half second takes in a little of the sine: 4.4 mm read
    # for 4.0 -- close enough for a safety cut
    assert recent_p2p(t, shaky) == pytest.approx(0.004, rel=0.15)


def test_recent_oscillation_tells_a_vibration_from_a_lag_building_up():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 1.0, 1 / 250)
    # a move whose lag builds up as friction lets go (still, then a ramp: a bend of several
    # mm about any quadratic), with 5 um of sensor noise
    bend = np.stack([np.where(t < 0.4, 0.0, 0.08 * (t - 0.4) ** 2 / 0.6), 0 * t, 0 * t], axis=1)
    bend += rng.normal(scale=5e-6, size=bend.shape)
    size, _ = recent_oscillation(t, bend)
    assert size < 0.001                                     # well under the 4 mm cut
    for freq in (3.0, 8.0, 20.0):
        shaky = bend.copy()
        shaky[:, 2] += 0.002 * np.sin(2 * np.pi * freq * t + 0.3)   # 4 mm peak to peak
        size, hz = recent_oscillation(t, shaky)
        assert size == pytest.approx(0.004, rel=0.15) and hz == pytest.approx(freq, abs=1.0)


def test_the_cut_needs_the_oscillation_to_persist():
    rate = 250
    t = np.arange(0.0, 6.0, 1 / rate)
    # end of a move: 30 mm of lag closed with a 0.1 s time constant at t = 2 s, then still
    catch_up = np.stack([np.where(t < 2.0, -0.03, -0.03 * np.exp(-(t - 2.0) / 0.1)),
                         0 * t, 0 * t], axis=1)
    vibrating = catch_up.copy()
    vibrating[:, 2] += np.where(t > 3.0, 0.003 * np.sin(2 * np.pi * 8 * t), 0.0)

    def first_cut(x):
        watch = OscillationWatch(0.004, persist=1.5)
        for i in range(rate, len(t), 25):                      # a check every 0.1 s
            size, _ = recent_oscillation(t[i - rate:i], x[i - rate:i])
            if watch.update(t[i], size):
                return t[i]
        return None

    assert first_cut(catch_up) is None
    cut = first_cut(vibrating)
    # it reads above 4 mm once it fills ~0.6 s of the 1 s window, then must persist 1.5 s:
    # cut ~2.1 s after it starts (measured 5.1 s for a start at 3 s)
    assert cut is not None and 4.5 <= cut <= 5.5


def test_move_lag_splits_along_and_across():
    u = np.array([1.0, 0.0, 0.0])
    cmd = np.stack([np.linspace(0, 0.1, 50), np.zeros(50), np.zeros(50)], axis=1)
    meas = cmd - [0.020, 0.0, 0.0] + [0.0, 0.003, 0.0]      # 20 mm behind, 3 mm to the side
    lag = move_lag(cmd, meas, u, np.full(50, 0.05))
    assert lag["along_median_mm"] == pytest.approx(20.0)
    assert lag["across_median_mm"] == pytest.approx(3.0)
    assert move_lag(cmd, meas, u, np.zeros(50)) == {}


def test_directional_matrix_is_the_rule():
    K = directional_matrices([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [0.1, 0.0], 350.0, 600.0)
    assert np.allclose(K[0], np.diag([600.0, 600.0, 350.0]))
    assert np.allclose(K[1], 600.0 * np.eye(3))              # still: stiff in every direction


def test_hold_tilt_is_the_angle_between_commanded_and_measured():
    from scipy.spatial.transform import Rotation

    t = np.arange(0.0, 5.0, 0.004)
    cmd = np.repeat(Rotation.from_euler("x", 180, degrees=True).as_quat()[None], len(t), axis=0)
    meas = (Rotation.from_euler("y", 1.5, degrees=True)
            * Rotation.from_quat(cmd)).as_quat()
    meas[:100] = cmd[:100]                                   # settling: skipped
    from tpgpt.hardware.stiffness_check import hold_tilt

    out = hold_tilt(t, cmd, meas)
    assert out["tilt_median_deg"] == pytest.approx(1.5, abs=1e-6)


def test_turn_lag_recovers_the_time_constant():
    from scipy.spatial.transform import Rotation

    from tpgpt.hardware.stiffness_check import turn_lag

    dt, tau, rest = 0.004, 0.4, np.radians(1.0)
    t = np.arange(0.0, 4.0, dt)
    angle = np.radians(15.0) * (0.5 - 0.5 * np.cos(np.pi * t / 4.0))      # the commanded turn
    rate = np.gradient(angle, dt)
    measured, x = [], 0.0
    for a in angle:                                           # a first-order hand, 1 deg short
        measured.append(x)
        x += (a - rest - x) * dt / tau
    base = Rotation.from_euler("x", 180, degrees=True)
    cmd = (Rotation.from_rotvec(np.outer(angle, [0, 0, 1])) * base).as_quat()
    meas = (Rotation.from_rotvec(np.outer(measured, [0, 0, 1])) * base).as_quat()
    out = turn_lag(cmd, meas, [0, 0, 1], rate)
    assert out["tau_s"] == pytest.approx(tau, rel=0.1)
    assert out["offset_deg"] == pytest.approx(1.0, abs=0.2)
    assert out["off_axis_max_deg"] < 1e-6
