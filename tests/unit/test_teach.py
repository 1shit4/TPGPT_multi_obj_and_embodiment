"""Stiffness labels of a taught demonstration (tpgpt.hardware.teach): directional and static."""

import numpy as np
import pytest

from tpgpt.hardware.teach import (
    DIRECTIONAL_ACROSS,
    DIRECTIONAL_ALONG,
    Recording,
    directional_stiffness,
    labels_from_recording,
    log_euclidean_smooth,
    motion_directions,
)


def _recording(rate=1000.0):
    """Still 1 s, 0.1 m along +x at 5 cm/s, still 1 s (grasp), 0.1 m along +y, still 1 s."""
    pieces, position = [], np.zeros(3)
    for direction, seconds in ((None, 1.0), ((1, 0, 0), 2.0), (None, 1.0), ((0, 1, 0), 2.0),
                               (None, 1.0)):
        n = int(seconds * rate)
        step = np.zeros(3) if direction is None else np.asarray(direction, float) * 0.05 / rate
        pieces.append(position + np.outer(np.arange(1, n + 1), step))
        position = pieces[-1][-1]
    positions = np.concatenate(pieces)
    stamps = np.arange(len(positions)) / rate
    down = np.diag([1.0, -1.0, -1.0])
    return Recording(stamps=stamps, positions=positions,
                     orientations=np.repeat(down[None], len(positions), axis=0),
                     grasp_stamp=3.5, release_stamp=5.5,
                     object_points=np.zeros((50, 3)), support_height=0.0)


def test_motion_direction_is_the_chord_of_the_last_stretch():
    t = np.linspace(0.0, 1.0, 101)
    line = np.stack([t * 0.1, np.zeros_like(t), np.zeros_like(t)], axis=1)
    u = motion_directions(line, tangent_length=0.01)
    assert np.allclose(u, [1.0, 0.0, 0.0])
    assert np.allclose(motion_directions(np.zeros((5, 3))), 0.0)


def test_log_euclidean_smoothing_keeps_constant_and_spd():
    K = np.repeat((np.diag([350.0, 600.0, 600.0]))[None], 20, axis=0)
    assert np.allclose(log_euclidean_smooth(K, 5), K)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3, 3))
    spd = np.einsum('nij,nkj->nik', A, A) + 50.0 * np.eye(3)
    out = log_euclidean_smooth(spd, 4)
    assert np.all(np.linalg.eigvalsh(out) > 0)
    assert np.allclose(out, np.transpose(out, (0, 2, 1)))
    assert np.allclose(log_euclidean_smooth(spd, 0), spd)


def test_directional_is_soft_along_the_motion_and_stiff_across_and_when_still():
    n = 200
    x = np.linspace(0.0, 0.5, n)
    positions = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    velocities = np.tile([0.1, 0.0, 0.0], (n, 1))              # 10 cm/s, above the ramp
    velocities[:20] = 0.0                                      # still at the start
    K, D = directional_stiffness(positions, velocities)
    mid = n // 2
    assert np.allclose(K[mid] @ [1, 0, 0], [DIRECTIONAL_ALONG, 0, 0], atol=1e-6)
    assert np.allclose(K[mid] @ [0, 1, 0], [0, DIRECTIONAL_ACROSS, 0], atol=1e-6)
    assert np.allclose(K[0], DIRECTIONAL_ACROSS * np.eye(3), atol=1e-6)
    w = np.linalg.eigvalsh(K)
    assert w.min() >= DIRECTIONAL_ALONG - 1e-6 and w.max() <= DIRECTIONAL_ACROSS + 1e-6
    assert np.allclose(np.einsum('nij,njk->nik', D, D), 4.0 * K, rtol=1e-9)   # D = 2 K^(1/2)


def test_taught_labels_directional_by_default_static_on_request():
    rec = _recording()
    directional = labels_from_recording(rec)
    assert directional.metadata['stiffness_profile'] == 'directional'
    w = np.linalg.eigvalsh(directional.stiffness)
    assert w.min() == pytest.approx(DIRECTIONAL_ALONG, abs=1.0)
    assert w.max() == pytest.approx(DIRECTIONAL_ACROSS, abs=1e-6)
    grasp = int(np.flatnonzero(directional.gripper > 0)[0])
    assert np.allclose(directional.stiffness[grasp], DIRECTIONAL_ACROSS * np.eye(3), atol=1.0)

    static = labels_from_recording(rec, stiffness_profile='static')
    assert np.allclose(static.stiffness, 400.0 * np.eye(3))
    assert np.allclose(static.damping, 40.0 * np.eye(3))
    assert static.metadata['static_stiffness'] == 400.0
    # the stiffness is the only thing the profile changes
    assert np.array_equal(static.positions, directional.positions)
    assert np.array_equal(static.gripper, directional.gripper)
    with pytest.raises(ValueError):
        labels_from_recording(rec, stiffness_profile='soft')
