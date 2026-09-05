"""Zero-mean multi-output GP -- paper Appendix A, Eqs. (14)-(16)."""

import numpy as np
import pytest

from tpgpt.transport.gp import (
    GaussianProcessRegressor,
    KernelHyperparameters,
    geometric_length_scale_bounds,
    residual_signal_variance_bounds,
)


@pytest.fixture
def smooth_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (25, 3))
    Y = np.c_[np.sin(2 * X[:, 0]), np.cos(X[:, 1]), X[:, 2] ** 2] * 0.1
    return X, Y


def _finite_difference_jacobian(gp, X, h=1e-5):
    J = np.zeros((X.shape[0], gp.Y_train.shape[1], X.shape[1]))
    for d in range(X.shape[1]):
        e = np.zeros(X.shape[1])
        e[d] = h
        J[:, :, d] = (gp.predict(X + e) - gp.predict(X - e)) / (2 * h)
    return J


def test_log_marginal_likelihood_gradient_matches_numerical(smooth_data):
    X, Y = smooth_data
    gp = GaussianProcessRegressor(optimize=False, noise_variance=1e-6)
    gp.X_train, gp.Y_train = X, Y
    params = KernelHyperparameters(0.5, 0.02, 1e-6)
    _, grad = gp.log_marginal_likelihood(params, eval_gradient=True)

    theta, h = params.to_log_theta(), 1e-6
    numeric = np.zeros(3)
    for i in range(3):
        up, dn = theta.copy(), theta.copy()
        up[i] += h
        dn[i] -= h
        numeric[i] = (
            gp.log_marginal_likelihood(KernelHyperparameters.from_log_theta(up))
            - gp.log_marginal_likelihood(KernelHyperparameters.from_log_theta(dn))
        ) / (2 * h)
    assert np.allclose(grad, numeric, rtol=1e-5)


def test_interpolates_training_data(smooth_data):
    """With negligible noise the posterior mean must pass through the data."""
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    assert np.abs(gp.predict(X) - Y).max() < 1e-5


def test_zero_mean_prior_far_from_data(smooth_data):
    """Sec. III-E-b relies on the residual decaying to zero out of distribution."""
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    far = np.array([[500.0, 500.0, 500.0]])
    assert np.allclose(gp.predict(far), 0.0, atol=1e-12)
    _, std = gp.predict(far, return_std=True)
    assert std[0] == pytest.approx(np.sqrt(gp.params.signal_variance), rel=1e-6)


def test_predictive_variance_vanishes_at_training_points(smooth_data):
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    _, std = gp.predict(X, return_std=True)
    assert std.max() < 1e-4 * np.sqrt(gp.params.signal_variance)


def test_derivative_mean_matches_finite_differences(smooth_data):
    """Eq. (16), first line."""
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    Xs = np.random.default_rng(1).uniform(-1, 1, (7, 3))
    assert np.abs(gp.predict_gradient(Xs) - _finite_difference_jacobian(gp, Xs)).max() < 1e-7


def test_derivative_variance_matches_numerical_limit(smooth_data):
    """Eq. (16), second line.

    ``Var[(f(x+h) - f(x-h)) / 2h]`` computed exactly from the posterior
    covariance converges to the derivative variance as ``h -> 0``.
    """
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    Xs = np.random.default_rng(1).uniform(-1, 1, (7, 3))
    _, std = gp.predict_gradient(Xs, return_std=True)

    h = 1e-4
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        A, B = Xs + e, Xs - e
        var = (
            np.diag(gp.posterior_covariance(A, A))
            + np.diag(gp.posterior_covariance(B, B))
            - 2 * np.diag(gp.posterior_covariance(A, B))
        ) / (4 * h * h)
        assert np.allclose(np.sqrt(np.maximum(var, 0)), std[:, 0, d], rtol=1e-4, atol=1e-9)


def test_shared_kernel_gives_identical_variance_per_output(smooth_data):
    """Appendix A: one kernel and one Cholesky are shared across outputs."""
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    Xs = np.random.default_rng(2).uniform(-1, 1, (5, 3))
    mean, std = gp.predict(Xs, return_std=True)
    assert mean.shape == (5, 3) and std.shape == (5,)
    _, jstd = gp.predict_gradient(Xs, return_std=True)
    assert np.allclose(jstd[:, 0, :], jstd[:, 2, :])


def test_geometric_length_scale_bounds_track_keypoint_spacing():
    X = np.array([[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0]])
    lo, hi = geometric_length_scale_bounds(X)
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(2.0)


def test_length_scale_does_not_collapse_on_sparse_keypoints():
    """The prototype's failure mode: MLE drove the length scale to 1e-5.

    With the scale pinned to its lower geometric bound the residual field is a
    spike train that is identically zero at every trajectory point.
    """
    c = 0.02
    offsets = np.array([[-c, c, c], [c, c, c], [c, -c, c], [-c, -c, c]])
    S = np.vstack([np.array([0.07, 0.0, 0.83]) + offsets,
                   np.array([0.06, 0.07, 0.835]) + offsets])
    residual = np.zeros_like(S)
    residual[4:] = [0.02, -0.03, 0.01]
    gp = GaussianProcessRegressor().fit(S, residual)
    assert gp.params.length_scale > 0.5 * 0.04  # half the closest corner spacing


def test_signal_variance_stays_near_the_residual_scale():
    """The opposite failure: amplitude blowing up and extrapolating wildly."""
    g = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5)), -1)
    X = np.c_[g.reshape(-1, 2), np.zeros(25)]
    Y = np.c_[np.zeros((25, 2)), 0.8 * X[:, 0] ** 2]
    gp = GaussianProcessRegressor().fit(X, Y)
    lo, hi = residual_signal_variance_bounds(Y)
    assert lo <= gp.params.signal_variance <= hi
    # Far from the data the map must decay, not diverge.
    assert np.abs(gp.predict(np.array([[5.0, 5.0, 5.0]]))).max() < 1e-9


def test_rigid_residual_collapses_amplitude_to_zero():
    """A residual that is identically zero should switch the nonlinearity off."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.2, 0.2, (12, 3))
    gp = GaussianProcessRegressor().fit(X, np.zeros_like(X))
    assert np.allclose(gp.predict(rng.uniform(-1, 1, (5, 3))), 0.0)


def test_sample_y_shape_and_determinism(smooth_data):
    X, Y = smooth_data
    gp = GaussianProcessRegressor().fit(X, Y)
    Xs = np.random.default_rng(3).uniform(-1, 1, (6, 3))
    s1 = gp.sample_y(Xs, 4, random_state=0)
    assert s1.shape == (4, 6, 3)
    assert np.allclose(s1, gp.sample_y(Xs, 4, random_state=0))


def test_accepts_single_output_targets():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (10, 2))
    gp = GaussianProcessRegressor().fit(X, np.sin(X[:, 0]))
    assert gp.predict(X).shape == (10, 1)
    assert gp.predict_gradient(X).shape == (10, 1, 2)


def test_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        GaussianProcessRegressor().predict(np.zeros((1, 3)))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="disagree on N"):
        GaussianProcessRegressor().fit(np.zeros((4, 3)), np.zeros((5, 3)))
