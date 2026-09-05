"""Figures reproducing the paper's theory illustrations (Figs. 2, 4, 5).

These are not decoration. They are the practical way to check three properties
that are hard to see in a scalar diagnostic:

* **Fig. 2** -- a grid of the original space deformed by ``gamma`` alone and then
  by the full ``phi``, showing that the nonlinearity is a local correction on top
  of a global rigid motion, and that far from the keypoints ``phi`` relaxes back
  to ``gamma``.
* **Fig. 4** -- the demonstration and its transported counterpart, with the
  fitted vector fields coloured by predictive uncertainty.
* **Fig. 5** -- transport, epistemic, and total uncertainty over the velocity
  field (Eqs. 12, 13), showing that the first grows away from the *keypoints*
  and the second away from the *demonstration*.

All in 2-D, as in the paper.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tpgpt.transport.maps import TransportMap


def _grid(limits, n=17):
    (x0, x1), (y0, y1) = limits
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    return xs, ys


def _draw_deformed_grid(ax, mapping, limits, n=17, color="#5b8ff9", lw=0.8):
    """Draw a grid of the source space after passing through ``mapping``."""
    xs, ys = _grid(limits, n)
    dense = np.linspace(limits[0][0], limits[0][1], 80)
    for y in ys:
        pts = mapping(np.c_[dense, np.full_like(dense, y)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=0.7)
    dense_y = np.linspace(limits[1][0], limits[1][1], 80)
    for x in xs:
        pts = mapping(np.c_[np.full_like(dense_y, x), dense_y])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=0.7)


def figure_space_deformation(
    source: np.ndarray,
    target: np.ndarray,
    path: str | Path,
    limits=((-40, 30), (-30, 45)),
) -> Path:
    """Fig. 2: distribution match, affine transformation, full transportation."""
    affine_only = TransportMap(residual=None).fit(source, target)
    full = TransportMap().fit(source, target)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].set_title("Distribution match")
    for s, t in zip(source, target):
        axes[0].plot([s[0], t[0]], [s[1], t[1]], color="0.75", lw=0.6, zorder=1)
    axes[0].scatter(*source.T, s=22, c="#2ca02c", label="source", zorder=2)
    axes[0].scatter(*target.T, s=22, c="#1f77b4", label="target", zorder=2)
    axes[0].legend(fontsize=8)

    for ax, mapping, title in (
        (axes[1], affine_only.transport_positions, r"Affine transformation $\gamma$"),
        (axes[2], full.transport_positions, r"Full transportation $\phi$"),
    ):
        ax.set_title(title)
        _draw_deformed_grid(ax, mapping, limits)
        ax.scatter(*target.T, s=22, c="#1f77b4", zorder=3)
        ax.scatter(*mapping(source).T, s=14, c="#2ca02c", zorder=3)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(
        "Policy transportation deforms the space (paper Fig. 2)", y=1.0, fontsize=12
    )
    return _save(fig, path)


def figure_transported_field(
    demonstration: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    policy_factory,
    path: str | Path,
) -> Path:
    """Fig. 4: the demonstration and its transported policy, with uncertainty."""
    from tpgpt.transport.labels import PolicyLabels, transport_labels

    dt = 10.0 / max(len(demonstration) - 1, 1)
    labels = PolicyLabels(
        positions=np.c_[demonstration, np.zeros(len(demonstration))],
        velocities=np.c_[np.gradient(demonstration, dt, axis=0), np.zeros(len(demonstration))],
        time_belief=np.linspace(0, 1, len(demonstration)),
        time_rate=np.full(len(demonstration), 0.1),
    )
    tm = TransportMap().fit(
        np.c_[source, np.zeros(len(source))], np.c_[target, np.zeros(len(target))]
    )
    transported = transport_labels(tm, labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, pts, kp, title in (
        (axes[0], labels.positions, source, "Demonstration and source keypoints"),
        (axes[1], transported.positions, target, "Transported demonstration"),
    ):
        ax.plot(pts[:, 0], pts[:, 1], color="#d62728", lw=2, label="demonstration")
        ax.scatter(kp[:, 0], kp[:, 1], s=18, c="#2ca02c", label="keypoints")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    band = transported.position_std
    axes[1].fill_between(
        transported.positions[:, 0],
        transported.positions[:, 1] - 2 * band,
        transported.positions[:, 1] + 2 * band,
        color="#ff7f0e", alpha=0.25, label="2 sigma transport uncertainty",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("Transporting a demonstration (paper Fig. 4)", y=1.0, fontsize=12)
    return _save(fig, path)


def figure_uncertainty_fields(
    demonstration: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    path: str | Path,
    limits=((-45, 40), (-45, 55)),
    n=55,
    duration: float = 10.0,
) -> Path:
    """Fig. 5: transport, epistemic and total uncertainty on the velocity field.

    Both fields are drawn on the **same warped mesh**. The transport uncertainty
    of Eq. (12) is a function of the *source* position -- ``phi`` maps source to
    target -- while the policy's epistemic uncertainty is a function of the
    *target* position. Sampling a regular grid in source space and plotting it at
    ``phi(grid)`` puts the two in one frame; plotting the source-space field
    against target-space coordinates would put the low-uncertainty band in the
    wrong place entirely.
    """
    from tpgpt.policy.gp_policy import GPPolicy
    from tpgpt.transport.labels import PolicyLabels, transport_labels
    from tpgpt.transport.uncertainty import total_variance

    dt = duration / max(len(demonstration) - 1, 1)
    labels = PolicyLabels(
        positions=np.c_[demonstration, np.zeros(len(demonstration))],
        velocities=np.c_[np.gradient(demonstration, dt, axis=0), np.zeros(len(demonstration))],
        time_belief=np.linspace(0, 1, len(demonstration)),
        time_rate=np.full(len(demonstration), 1.0 / duration),
    )
    tm = TransportMap().fit(
        np.c_[source, np.zeros(len(source))], np.c_[target, np.zeros(len(target))]
    )
    transported = transport_labels(tm, labels)
    policy = GPPolicy(use_time_belief=False).fit(transported)

    X, Y = np.meshgrid(np.linspace(*limits[0], n), np.linspace(*limits[1], n))
    source_query = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    target_query = tm.transport_positions(source_query)
    Xw = target_query[:, 0].reshape(X.shape)
    Yw = target_query[:, 1].reshape(Y.shape)

    reference_speed = np.tile(np.abs(labels.velocities).mean(axis=0), (len(source_query), 1))
    _, transport_std = tm.transport_velocities(
        source_query, reference_speed, return_std=True
    )
    transport_var = (transport_std**2).sum(axis=1)
    epistemic_var = (policy.predict(target_query).velocity_std ** 2).sum(axis=1)
    total = total_variance(epistemic_var, transport_var)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, field, title in (
        (axes[0], transport_var, "Transport uncertainty (Eq. 12)"),
        (axes[1], epistemic_var, r"Epistemic uncertainty $\Sigma_{\hat f}$"),
        (axes[2], total, "Total uncertainty (Eq. 13)"),
    ):
        with warnings.catch_warnings():
            # The mesh is deliberately warped by phi, so it is not monotonic.
            warnings.filterwarnings("ignore", message=".*monotonically increasing.*")
            image = ax.pcolormesh(
                Xw, Yw, np.sqrt(field).reshape(X.shape), cmap="magma", shading="auto"
            )
        fig.colorbar(image, ax=ax, label="std [m/s]")
        ax.plot(
            transported.positions[:, 0], transported.positions[:, 1],
            color="#00d0ff", lw=1.8, label="transported demo",
        )
        ax.scatter(target[:, 0], target[:, 1], s=10, c="#39ff14", label="target keypoints")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-45, 55)
        ax.set_ylim(-45, 55)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Transport uncertainty grows away from the keypoints; epistemic "
        "uncertainty away from the demonstration (paper Fig. 5)",
        y=1.02, fontsize=11,
    )
    return _save(fig, path)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
