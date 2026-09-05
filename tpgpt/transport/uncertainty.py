"""Uncertainty propagation through the transportation map (paper Sec. III-H).

Two distinct sources are combined in Eq. (13):

* **Transport (aleatoric) uncertainty** -- from the finite set of keypoints. It
  grows as the query moves away from the source/target distribution, because the
  residual map is less certain there (paper Figs. 4, 5 left).
* **Epistemic uncertainty** -- of the policy ``g`` refitted on the transported
  labels. It grows away from the transported demonstration (Fig. 5 centre).

Their variance sum is the total predictive uncertainty of the transported policy
(Fig. 5 right).
"""

from __future__ import annotations

import numpy as np


def propagate_velocity_variance(
    jacobian_std: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    """Eq. (12): variance of the transported velocity labels.

    ``xdot_hat = J xdot`` with a deterministic ``xdot`` and an uncertain ``J``,
    so by the weighted-sum-of-Gaussians rule (ref. [49] in the paper)
    ``Var[xdot_hat_i] = sum_d Var[J_id] * xdot_d^2``.

    The Jacobian entries within a row are treated as independent. That matches
    the paper's use of the weighted-sum rule; the posterior derivative
    components of a squared-exponential GP are exactly independent under the
    prior and only weakly correlated under the posterior.

    Args:
        jacobian_std: ``(n, D, d)`` standard deviation of the Jacobian entries.
        velocities: ``(n, d)`` source velocity labels.

    Returns:
        ``(n, D)`` variance of the transported velocity labels.
    """
    jacobian_std = np.asarray(jacobian_std, dtype=float)
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    if jacobian_std.ndim != 3:
        raise ValueError(f"jacobian_std must be (n, D, d), got {jacobian_std.shape}")
    if velocities.shape[0] != jacobian_std.shape[0]:
        raise ValueError(
            f"batch mismatch: {velocities.shape[0]} velocities vs "
            f"{jacobian_std.shape[0]} Jacobians"
        )
    return np.einsum("nod,nd->no", jacobian_std**2, velocities**2)


def total_variance(epistemic_var: np.ndarray, transport_var: np.ndarray) -> np.ndarray:
    """Eq. (13): ``Sigma_xdot_hat = Sigma_f_hat + Sigma_x_hat``.

    Args:
        epistemic_var: Variance of the refitted policy ``g``.
        transport_var: Variance contributed by the transportation map,
            i.e. the output of :func:`propagate_velocity_variance`.
    """
    epistemic_var = np.asarray(epistemic_var, dtype=float)
    transport_var = np.asarray(transport_var, dtype=float)
    return epistemic_var + transport_var


def variance_to_std(var: np.ndarray) -> np.ndarray:
    """Numerically safe ``sqrt`` for variances that may be a hair below zero."""
    return np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0))
