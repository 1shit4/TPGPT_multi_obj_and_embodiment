"""Regression-safe charts for SO(3) and SPD matrices."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.policy.orientation import rotation_from_6d, rotation_to_6d
from tpgpt.policy.spd import log_cholesky_to_spd, spd_to_log_cholesky
from tpgpt.utils.rotations import is_rotation


class TestSPDChart:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(20, 3, 3))
        M = np.einsum("nij,nkj->nik", A, A) + 0.1 * np.eye(3)
        assert np.allclose(log_cholesky_to_spd(spd_to_log_cholesky(M)), M, atol=1e-10)

    def test_diagonal_stiffness_roundtrip(self):
        K = np.stack([np.diag([600.0, 400.0, 200.0])] * 3)
        assert np.allclose(log_cholesky_to_spd(spd_to_log_cholesky(K)), K, atol=1e-8)

    def test_regressor_scale_vectors_map_to_positive_definite_matrices(self):
        """This is the whole point: a regressor can emit anything."""
        v = np.random.default_rng(1).normal(0, 1, (200, 6))
        M = log_cholesky_to_spd(v)
        assert np.allclose(M, M.transpose(0, 2, 1))
        assert np.linalg.eigvalsh(M).min() > 0

    def test_extreme_vectors_have_no_negative_stiffness_direction(self):
        """At sigma=5 the diagonal spans exp(+-15), so the condition number
        reaches ~1e13 and eigvalsh round-off alone can show a tiny negative
        value. What must hold physically is that no eigenvalue is negative
        relative to the scale of the matrix.
        """
        v = np.random.default_rng(1).normal(0, 5, (100, 6))
        eig = np.linalg.eigvalsh(log_cholesky_to_spd(v))
        assert (eig.min(axis=1) > -1e-12 * eig.max(axis=1)).all()

    def test_handles_singular_positive_semidefinite_input(self):
        K = np.stack([np.diag([500.0, 300.0, 0.0])])
        assert np.linalg.eigvalsh(log_cholesky_to_spd(spd_to_log_cholesky(K))).min() >= 0

    def test_chart_is_six_dimensional(self):
        assert spd_to_log_cholesky(np.eye(3)[None]).shape == (1, 6)


class TestRotationChart:
    def test_roundtrip(self):
        R = Rotation.random(20, random_state=1).as_matrix()
        assert np.allclose(rotation_from_6d(rotation_to_6d(R)), R, atol=1e-12)

    def test_noisy_representation_still_yields_rotations(self):
        R = Rotation.random(20, random_state=2).as_matrix()
        noisy = rotation_to_6d(R) + np.random.default_rng(3).normal(0, 0.3, (20, 6))
        assert all(is_rotation(r, atol=1e-9) for r in rotation_from_6d(noisy))

    def test_degenerate_input_yields_a_valid_rotation(self):
        """A zero prediction must not produce a NaN or a singular frame."""
        assert is_rotation(rotation_from_6d(np.zeros((1, 6)))[0], atol=1e-9)

    def test_representation_is_six_dimensional(self):
        assert rotation_to_6d(np.eye(3)[None]).shape == (1, 6)
