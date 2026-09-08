"""3-D figures for grasp-aligned keypoint extraction and transport.

The 2-D figures in :mod:`tpgpt.viz.figures` reproduce the paper's theory plots.
These answer a different question: what did the extractor actually do to *this*
scene, and is the warped trajectory something a robot should execute.

Three things are hard to see in a scalar diagnostic and obvious in a picture:

* whether the fitted box actually encloses the object and sits square with the
  grasp, rather than being a plausible box in the wrong frame;
* whether the transported trajectory arrives at the destination surface or
  hovers above it -- ``keypoint_residual`` is near zero either way, because it
  measures the keypoints and not the path between them;
* where the object is released, which is the difference between placing it and
  dropping it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tpgpt.sim.keypoints import CORNER_NAMES
from tpgpt.viz.figures import _save

#: Edges of the unit cube, indexing :data:`~tpgpt.sim.keypoints.CUBE_CORNERS`.
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

SOURCE_COLOUR = "#5b8ff9"
TARGET_COLOUR = "#ff7f0e"
SUPPORT_COLOUR = "#8c8c8c"


def _equal_aspect(ax, points: np.ndarray) -> None:
    """Equal axis scaling, without which a warp looks like a shear."""
    points = np.atleast_2d(points)
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    radius = 0.5 * float(np.ptp(points, axis=0).max()) or 0.05
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - radius, c + radius)


def _draw_box(ax, keypoints, colour, label=None):
    """Wireframe of the fitted grasp-aligned box."""
    corners = np.stack(
        [keypoints.points[keypoints.labels.index(f"{n}")] for n in _corner_labels(keypoints)]
    )
    for i, j in BOX_EDGES:
        ax.plot(*zip(corners[i], corners[j]), color=colour, lw=1.1, alpha=0.85)
    if label:
        ax.plot([], [], color=colour, lw=1.1, label=label)
    return corners


def _corner_labels(keypoints) -> list[str]:
    prefix = keypoints.labels[0].rsplit("_", 1)[0]
    return [f"{prefix}_{n}" for n in CORNER_NAMES]


def _draw_support_plane(ax, height, extent, colour=SUPPORT_COLOUR):
    """The surface the object rests on, so clearance is visible."""
    (x0, x1), (y0, y1) = extent
    xx, yy = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, height), color=colour, alpha=0.16, shade=False)


def _block(keypoints, prefix):
    """The sub-set of a keypoint set belonging to one configuration.

    ``scene_keypoints`` strips the ``src_``/``tgt_`` prefix from the labels so
    the two sets pair, but keys the metadata by the unstripped block name, so
    the metadata is looked up by suffix.
    """
    index = [i for i, l in enumerate(keypoints.labels) if l.startswith(f"{prefix}_")]
    metadata = next(
        (v for k, v in keypoints.metadata.items() if k.endswith(prefix)), {}
    )
    from tpgpt.sim.keypoints import KeypointSet

    return KeypointSet(
        points=keypoints.points[index],
        labels=[keypoints.labels[i] for i in index],
        metadata=metadata,
    )


def figure_keypoint_scene(
    source_keypoints,
    target_keypoints,
    diagnostics: dict,
    source_placement,
    target_placement,
    path: str | Path,
    title: str = "",
) -> Path:
    """The four configurations: both objects, where they start and where they go.

    Each panel shows the object's cloud, the box the extractor fitted to it, the
    keypoints it emitted, and the surface it rests on. If the box is in the
    wrong frame or floating above its surface, it is visible here and nowhere
    else.
    """
    fig = plt.figure(figsize=(13, 4.6))
    panels = [
        ("source, pick", source_keypoints, "pick", source_placement.points,
         source_placement.support_height, SOURCE_COLOUR),
        ("source, place", source_keypoints, "place", diagnostics["source_place_points"],
         source_placement.destination_height, SOURCE_COLOUR),
        ("target, pick", target_keypoints, "pick", target_placement.points,
         target_placement.support_height, TARGET_COLOUR),
        ("target, place", target_keypoints, "place", diagnostics["target_place_points"],
         target_placement.destination_height, TARGET_COLOUR),
    ]
    for i, (name, keypoints, prefix, cloud, height, colour) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        block = _block(keypoints, prefix)
        ax.scatter(*cloud.T, s=1.2, c="#bbbbbb", alpha=0.45, depthshade=False)
        corners = _draw_box(ax, block, colour)
        ax.scatter(*block.points[1:5].T, s=34, c="#2ca02c", depthshade=False,
                   label="on support plane")
        ax.scatter(*block.points[5:9].T, s=22, c=colour, depthshade=False)
        ax.scatter(*block.points[:1].T, s=40, c="k", marker="x", depthshade=False)
        span = np.vstack([cloud, corners])
        _draw_support_plane(
            ax, height,
            ((span[:, 0].min() - 0.03, span[:, 0].max() + 0.03),
             (span[:, 1].min() - 0.03, span[:, 1].max() + 0.03)),
        )
        _equal_aspect(ax, span)
        size = np.round(np.asarray(block.metadata.get("box_half_extents", [0, 0, 0])) * 200, 1)
        ax.set_title(
            f"{name}\n{len(cloud)} pts,  {size[0]} x {size[1]} x {size[2]} cm", fontsize=9
        )
        ax.view_init(elev=18, azim=-62)
        ax.tick_params(labelsize=5, pad=-2)
        if i == 1:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title or "Grasp-aligned keypoints", fontsize=11, y=0.99)
    return _save(fig, path)


def figure_transported_trajectory(
    demonstration: np.ndarray,
    transported: np.ndarray,
    source_keypoints,
    target_keypoints,
    surfaces: dict,
    path: str | Path,
    release_index: int | None = None,
    title: str = "",
) -> Path:
    """Demonstration and its transported counterpart, against the real surfaces.

    The keypoint displacement arrows show what the map was asked to do; the two
    paths show what it did with the trajectory in between, which is *not*
    constrained by the keypoints and is where a plausible-looking map goes
    wrong.
    """
    fig = plt.figure(figsize=(12, 5.4))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*demonstration.T, color=SOURCE_COLOUR, lw=1.6, label="demonstration")
    ax.plot(*transported.T, color=TARGET_COLOUR, lw=1.6, label="transported")
    S, T = source_keypoints.points, target_keypoints.points
    ax.quiver(*S.T, *(T - S).T, color="#999999", lw=0.6, arrow_length_ratio=0.08, alpha=0.75)
    ax.scatter(*S.T, s=14, c=SOURCE_COLOUR, depthshade=False)
    ax.scatter(*T.T, s=14, c=TARGET_COLOUR, depthshade=False)
    if release_index is not None:
        ax.scatter(*transported[release_index], s=70, facecolors="none",
                   edgecolors="k", linewidths=1.4, label="release")
    _equal_aspect(ax, np.vstack([demonstration, transported, S, T]))
    ax.view_init(elev=20, azim=-64)
    ax.set_title("keypoint displacements and the warped path", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(1, 2, 2)
    for name, curve, colour, surface in (
        ("demonstration", demonstration, SOURCE_COLOUR, surfaces["source"]),
        ("transported", transported, TARGET_COLOUR, surfaces["target"]),
    ):
        ax.plot(np.arange(len(curve)), (curve[:, 2] - surface) * 100, color=colour, label=name)
    ax.axhline(0.0, color=SUPPORT_COLOUR, lw=1.2, ls="--", label="destination surface")
    if release_index is not None:
        clearance = (transported[release_index, 2] - surfaces["target"]) * 100
        ax.scatter([release_index], [clearance], s=60, facecolors="none",
                   edgecolors="k", linewidths=1.4, zorder=5)
        ax.annotate(f"release at {clearance:.1f} cm", (release_index, clearance),
                    textcoords="offset points", xytext=(-8, 12), fontsize=8, ha="right")
    ax.set_xlabel("control step")
    ax.set_ylabel("height above destination surface (cm)")
    ax.set_title("does it place, or drop?", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(title or "Transported trajectory", fontsize=11)
    return _save(fig, path)


def figure_ablation(records: list[dict], path: str | Path, title: str = "") -> Path:
    """Where the extractor holds and where it breaks.

    ``det(J)`` is plotted because a map can interpolate every keypoint perfectly
    and still fold space between them, which is a violation of property (ii) of
    Sec. III-D and shows up as nothing else.
    """
    records = [r for r in records if "skipped" not in r]
    labels = [r["label"] for r in records]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    axes[0].bar(x, [r["release_clearance"] * 1000 for r in records], color=TARGET_COLOUR)
    axes[0].axhline(0.0, color="k", lw=1.0)
    axes[0].set_ylabel("release clearance (mm)")
    axes[0].set_title("above the destination surface", fontsize=9)

    axes[1].bar(x, [r["lateral_error"] * 1000 for r in records], color=SOURCE_COLOUR)
    axes[1].set_ylabel("lateral offset from slot (mm)")
    axes[1].set_title("placement in the slot", fontsize=9)

    colours = ["#2ca02c" if r["min_det"] > 0 else "#d62728" for r in records]
    axes[2].bar(x, [r["min_det"] for r in records], color=colours)
    axes[2].axhline(0.0, color="k", lw=1.0)
    axes[2].set_ylabel("min det(J) on the path")
    axes[2].set_title("diffeomorphism (red = folded)", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title or "Keypoint extraction ablations", fontsize=11)
    return _save(fig, path)
