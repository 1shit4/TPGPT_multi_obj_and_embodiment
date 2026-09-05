"""Shared fixtures. Keeps the simulator out of the fast unit tests."""

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# The affine map legitimately warns on rank-deficient keypoint sets; several
# tests exercise that path on purpose.
logging.getLogger("tpgpt.transport.affine").setLevel(logging.ERROR)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def rigid_keypoints():
    """Source/target keypoints related by an exact rigid transform."""
    rng = np.random.default_rng(1)
    S = rng.uniform(-0.2, 0.2, (12, 3))
    A = Rotation.random(random_state=3).as_matrix()
    t = np.array([0.4, -0.3, 0.2])
    return S, S @ A.T + t, A, t


@pytest.fixture
def curved_keypoints():
    """A flat 5x5 sheet mapped onto a parabolic surface (paper Fig. 2 style)."""
    g = np.stack(
        np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5)), -1
    ).reshape(-1, 2)
    S = np.c_[g, np.zeros(len(g))]
    T = np.c_[g, 0.8 * g[:, 0] ** 2]
    return S, T
