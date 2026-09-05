"""Sparse variational GP -- paper Appendix A (SV-GPT)."""

import numpy as np
import pytest

from tpgpt.transport.gp import GaussianProcessRegressor
from tpgpt.transport.svgp import SparseGaussianProcessRegressor, farthest_point_sample


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (60, 3))
    Y = np.c_[np.sin(2 * X[:, 0]), np.cos(X[:, 1]), X[:, 2] ** 2] * 0.1
    return X, Y


def _fixed(cls, **kw):
    return cls(optimize=False, length_scale=0.6, signal_variance=0.02,
               noise_variance=1e-4, **kw)


def test_matches_exact_gp_when_all_points_are_inducing(data):
    """M = N removes the approximation, so the two must agree to machine precision."""
    X, Y = data
    exact = _fixed(GaussianProcessRegressor).fit(X, Y)
    sparse = _fixed(SparseGaussianProcessRegressor, n_inducing=len(X)).fit(X, Y)
    Xs = np.random.default_rng(1).uniform(-1, 1, (10, 3))

    assert np.allclose(exact.predict(Xs), sparse.predict(Xs), atol=1e-9)
    assert np.allclose(
        exact.predict(Xs, return_std=True)[1],
        sparse.predict(Xs, return_std=True)[1],
        atol=1e-9,
    )
    Je, Se = exact.predict_gradient(Xs, return_std=True)
    Js, Ss = sparse.predict_gradient(Xs, return_std=True)
    assert np.allclose(Je, Js, atol=1e-8)
    assert np.allclose(Se, Ss, atol=1e-8)


def test_derivative_matches_finite_differences(data):
    X, Y = data
    gp = SparseGaussianProcessRegressor(n_inducing=20).fit(X, Y)
    Xs = np.random.default_rng(1).uniform(-1, 1, (8, 3))
    J = gp.predict_gradient(Xs)
    h = 1e-5
    J_fd = np.zeros_like(J)
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        J_fd[:, :, d] = (gp.predict(Xs + e) - gp.predict(Xs - e)) / (2 * h)
    assert np.allclose(J, J_fd, atol=1e-5)


def test_converges_to_the_exact_gp_as_inducing_points_are_added(data):
    """The variational approximation must improve monotonically towards M = N.

    Asserting convergence is stronger than asserting an arbitrary error
    threshold, which depends on how informative the kernel is for the data.
    """
    X, Y = data
    exact = _fixed(GaussianProcessRegressor).fit(X, Y)
    Xs = np.random.default_rng(4).uniform(-1, 1, (10, 3))
    reference = exact.predict(Xs)

    errors = []
    for m in (10, 20, 30, 50, 60):
        sparse = _fixed(SparseGaussianProcessRegressor, n_inducing=m).fit(X, Y)
        errors.append(np.abs(reference - sparse.predict(Xs)).max())

    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-8  # M = N recovers the exact posterior


def test_zero_mean_prior_far_from_data(data):
    X, Y = data
    gp = SparseGaussianProcessRegressor(n_inducing=20).fit(X, Y)
    assert np.allclose(gp.predict(np.array([[500.0, 500.0, 500.0]])), 0.0, atol=1e-12)


def test_farthest_point_sampling_is_deterministic_and_spread(data):
    X, _ = data
    a = farthest_point_sample(X, 10, seed=3)
    assert np.array_equal(a, farthest_point_sample(X, 10, seed=3))
    assert a.shape == (10, 3)
    # Spread: minimum inter-point distance should beat a random subset's.
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(0)
    random_subset = X[rng.choice(len(X), 10, replace=False)]
    assert pdist(a).min() > pdist(random_subset).min()


def test_returns_all_points_when_inducing_exceeds_dataset(data):
    X, _ = data
    assert farthest_point_sample(X, 999).shape == X.shape


def test_scales_to_the_papers_cleaning_point_cloud():
    """Sec. V-C: 400 source/target points approximated with 100 inducing points."""
    rng = np.random.default_rng(0)
    g = np.stack(
        np.meshgrid(np.linspace(-0.15, 0.15, 20), np.linspace(-0.15, 0.15, 20)), -1
    ).reshape(-1, 2)
    X = np.c_[g, np.zeros(400)]
    Y = np.c_[np.zeros((400, 2)), 0.03 * np.sin(6 * g[:, 0])]
    gp = SparseGaussianProcessRegressor(n_inducing=100).fit(X, Y)
    assert gp.Z.shape == (100, 3)
    assert np.abs(gp.predict(X) - Y).max() < 5e-3


def test_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        SparseGaussianProcessRegressor().predict(np.zeros((1, 3)))
