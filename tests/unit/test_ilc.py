"""tpgpt.hardware.ilc on a simulated arm with the lag measured on the FR3 (2026-09-29)."""

import numpy as np
import pytest

from tpgpt.hardware.ilc import (IlcSettings, frozen_mask, ilc_update, label_positions,
                                residuals_at)

LABEL_PERIOD = 0.05 / 0.35        # a 20 Hz label replayed at speed_scale 0.35, s
TAU = 0.27                        # the arm's time constant along the motion at 350 N/m, s
SAG = np.array([0.0, 0.0, -0.003])  # a static offset the arm keeps (gravity model, stiction)


def _segment(a, b, n):
    s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n + 1)[1:])
    return np.asarray(a) + s[:, None] * (np.asarray(b) - np.asarray(a))


def _taught_path():
    """Approach, 15-label dwell at the grasp, lift, carry, 15-label dwell at the release, retreat."""
    start, grasp, above = [0.30, 0.0, 0.30], [0.50, 0.10, 0.02], [0.50, 0.10, 0.15]
    over, release, back = [0.45, -0.30, 0.40], [0.45, -0.30, 0.34], [0.35, -0.20, 0.45]
    parts = [np.array([start]), _segment(start, grasp, 30), np.repeat([grasp], 15, axis=0),
             _segment(grasp, above, 15), _segment(above, over, 30), _segment(over, release, 12),
             np.repeat([release], 15, axis=0), _segment(release, back, 20)]
    path = np.concatenate(parts)
    grasp_i = 1 + 30 + 14
    release_i = 1 + 30 + 15 + 15 + 30 + 12 + 14
    return path, grasp_i, release_i


def _replay(attractor, floor=None, noise=0.0, rng=None, substeps=20):
    """The arm on the label grid: first-order toward attractor + SAG; optional board below it."""
    h = LABEL_PERIOD / substeps
    x = attractor[0] + SAG
    measured = np.empty_like(attractor)
    for i, a in enumerate(attractor):
        measured[i] = x + (rng.normal(scale=noise, size=3) if noise else 0.0)
        for _ in range(substeps):
            x = x + (a + SAG - x) * (h / TAU)
            if floor is not None and floor[i] is not None:
                x[2] = max(x[2], floor[i])
    return measured


def _passes(reference, n, frozen=None, floor=None, noise=0.0, settings=IlcSettings()):
    rng = np.random.default_rng(3)
    a = reference.copy()
    history = []
    for _ in range(n):
        m = _replay(a, floor, noise, rng)
        history.append((a.copy(), m))
        a, report = ilc_update(a, m, reference, LABEL_PERIOD, frozen, settings)
    return a, history


def test_label_positions_takes_the_moment_each_label_was_reached():
    # label 0 at t 0..1, label 1 reached at sample 2 and held (a pause), label 2 at sample 5
    idx = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 2.0])
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9], [9, 9, 9], [3, 0, 0]], float)
    out = label_positions(idx, pos, 3)
    assert np.allclose(out[1], [2, 0, 0])                    # not the pause's samples
    assert np.allclose(out[2], [3, 0, 0])


def test_the_attractor_learned_makes_the_arm_retrace_the_taught_path():
    reference, grasp, release = _taught_path()
    _, history = _passes(reference, 6)
    rms = [np.sqrt(np.mean(np.sum((reference - m) ** 2, axis=1))) for _, m in history]
    assert max(np.linalg.norm(reference - history[0][1], axis=1)) > 0.015   # replaying it as is
    assert all(b < a for a, b in zip(rms, rms[1:]))                      # every pass better
    assert rms[-1] < rms[0] / 5
    last = history[-1][1]
    for i in (grasp, release):
        assert np.linalg.norm(reference[i] - last[i]) < 0.001, i       # < 1 mm where it counts


def test_contact_is_bridged_not_learned():
    reference, grasp, release = _taught_path()
    # the board is 5 mm higher than the taught path thinks around the set-down
    window = (release - 8, release + 4)
    floor = [None] * len(reference)
    for i in range(window[0], window[1] + 1):
        floor[i] = reference[i][2] + 0.005
    frozen = frozen_mask(len(reference), [window])
    free_a, _ = _passes(reference, 6, floor=floor)
    held_a, _ = _passes(reference, 6, frozen=frozen, floor=floor)
    pushed_free = (reference[release][2] - free_a[release][2]) * 1000
    pushed_held = (reference[release][2] - held_a[release][2]) * 1000
    assert pushed_free > 8.0                      # learning the contact presses into the board
    assert pushed_held < 5.0                      # bridged: only the sag correction carries over
    steps = np.linalg.norm(np.diff(held_a, axis=0), axis=1)
    assert steps.max() < np.linalg.norm(np.diff(reference, axis=0), axis=1).max() + 0.003


def test_a_pass_is_capped():
    reference, _, _ = _taught_path()
    far = reference + [0.2, 0.0, 0.0]                       # the arm nowhere near
    s = IlcSettings(gain=1.0, max_update=0.02, max_correction=0.06)
    a = reference.copy()
    for _ in range(5):
        new, report = ilc_update(a, far, reference, LABEL_PERIOD, settings=s)
        assert np.linalg.norm(new - a, axis=1).max() <= 0.02 + 1e-12
        a = new
    assert np.linalg.norm(a - reference, axis=1).max() <= 0.06 + 1e-12
    assert report["capped_total"] > 0


def test_noise_is_not_learned_into_a_blow_up():
    reference, grasp, release = _taught_path()
    _, history = _passes(reference, 10, noise=0.0003)
    rms = [np.sqrt(np.mean(np.sum((reference - m) ** 2, axis=1))) for _, m in history]
    assert rms[-1] < 0.0015
    assert max(rms[4:]) < rms[0] / 4


def test_residuals_are_split_into_the_hands_axes():
    R = np.eye(3)[None]                                      # closing x, approach z
    out = residuals_at([[0.0, 0.0, 0.0]], [[-0.003, 0.004, -0.010]], R, {"grasp": 0})
    assert out["grasp"]["closing_mm"] == pytest.approx(3.0)
    assert out["grasp"]["approach_mm"] == pytest.approx(10.0)
    assert out["grasp"]["total_mm"] == pytest.approx(np.sqrt(9 + 16 + 100))
