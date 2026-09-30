"""tpgpt.hardware.ilc on a simulated arm with the lag measured on the FR3 (2026-09-29)."""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from tpgpt.hardware.ilc import (IlcSettings, frozen_mask, ilc_rotation_update, ilc_update,
                                label_positions, label_rotations, residuals_at, rotation_errors)

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
    # the board is 5 mm higher than the taught path thinks around the set-down; the
    # window ends at the release, where the hand is still (as in the taught demonstrations)
    window = (release - 8, release)
    floor = [None] * len(reference)
    for i in range(window[0], window[1] + 1):
        floor[i] = reference[i][2] + 0.005
    frozen = frozen_mask(len(reference), [window])
    free_a, _ = _passes(reference, 6, floor=floor)
    held_a, _ = _passes(reference, 6, frozen=frozen, floor=floor)
    pushed_free = (reference[release][2] - free_a[release][2]) * 1000
    pushed_held = (reference[release][2] - held_a[release][2]) * 1000
    assert pushed_free > 8.0                      # learning the contact presses into the board
    # bridged: only the sag correction carries over -- the attractor 3 mm above the taught
    # point, so the arm, sagging 3 mm, sits on it
    assert abs(pushed_held - (-SAG[2] * -1000)) < 1.0
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


# --- orientation --------------------------------------------------------------------------
TAU_ROT = 0.35                                    # s, CRISP's nominal at 30 N m/rad
TILT = Rotation.from_rotvec(np.radians(1.5) * np.array([0.6, 0.8, 0.0])).as_matrix()   # resting


def _taught_turn(n=150):
    """The hand turning 40 deg about the vertical and tipping 15 deg, slowly, then back."""
    s = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, n))
    return Rotation.from_rotvec(np.outer(s, np.radians([0.0, 15.0, 40.0]))).as_matrix()


def _replay_rotation(attractor, tilt=TILT, substeps=20):
    """The hand on the label grid: first-order toward ``tilt @ attractor``."""
    h = LABEL_PERIOD / substeps
    R = tilt @ attractor[0]
    out = np.empty_like(attractor)
    for i, A in enumerate(attractor):
        out[i] = R
        for _ in range(substeps):
            step = Rotation.from_matrix(tilt @ A @ R.T).as_rotvec() * (h / TAU_ROT)
            R = Rotation.from_rotvec(step).as_matrix() @ R
    return out


def test_label_rotations_takes_the_moment_each_label_was_reached():
    Rz = lambda deg: Rotation.from_euler("z", deg, degrees=True).as_matrix()   # noqa: E731
    idx = np.array([0.0, 0.5, 1.0, 1.0, 2.0])
    rots = np.stack([Rz(0), Rz(10), Rz(20), Rz(90), Rz(30)])
    out = label_rotations(idx, rots, 3)
    assert np.allclose(out[1], Rz(20)) and np.allclose(out[2], Rz(30))
    half = label_rotations(np.array([0.0, 2.0]), np.stack([Rz(0), Rz(40)]), 3)
    assert np.allclose(half[1], Rz(20))                     # slerped between the samples


def test_the_orientation_learned_removes_the_tilt_and_the_lag():
    reference = _taught_turn()
    a = reference.copy()
    errors = []
    for _ in range(4):
        measured = _replay_rotation(a)
        errors.append(np.degrees(np.linalg.norm(rotation_errors(reference, measured), axis=1)))
        a, report = ilc_rotation_update(a, measured, reference, LABEL_PERIOD)
    assert errors[0].max() > 1.5                           # the resting tilt, plus the lag
    assert errors[-1].max() < 0.1
    assert all(np.sqrt(np.mean(b ** 2)) < np.sqrt(np.mean(e ** 2)) for e, b in zip(errors, errors[1:]))


def test_the_orientation_is_bridged_in_contact_and_capped():
    reference = _taught_turn()
    frozen = frozen_mask(len(reference), [(60, 80)])
    measured = _replay_rotation(reference)
    wild = measured.copy()
    wild[60:81] = Rotation.from_euler("x", 30, degrees=True).as_matrix() @ wild[60:81]  # a wrench
    held, _ = ilc_rotation_update(reference, wild, reference, LABEL_PERIOD, frozen)
    free, _ = ilc_rotation_update(reference, measured, reference, LABEL_PERIOD, frozen)
    assert np.allclose(held, free)                         # nothing learned from inside the window
    s = IlcSettings(max_rotation_update_deg=3.0, max_rotation_correction_deg=10.0)
    far = Rotation.from_euler("x", 45, degrees=True).as_matrix() @ reference
    a = reference.copy()
    for _ in range(6):
        new, report = ilc_rotation_update(a, far, reference, LABEL_PERIOD, settings=s)
        step = np.degrees(np.linalg.norm(rotation_errors(new, a), axis=1))
        assert step.max() <= 3.0 + 1e-6
        a = new
    assert np.degrees(np.linalg.norm(rotation_errors(a, reference), axis=1)).max() <= 10.0 + 1e-6
    assert report["capped_total"] > 0


def test_residuals_report_the_tilt():
    R = np.eye(3)[None]
    tilted = Rotation.from_euler("y", 2.0, degrees=True).as_matrix()[None]
    out = residuals_at([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], R, {"grasp": 0}, tilted)
    assert out["grasp"]["tilt_deg"] == pytest.approx(2.0)
