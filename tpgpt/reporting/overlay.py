"""Draw 3-D things onto the camera images they came from.

A grasp pose printed as sixteen numbers is unreadable; the same pose drawn on a
picture of the object is obvious. These overlays exist so a report can be
checked by looking at it -- if the jaws straddle the object, the frame contract
held for that hand, and no amount of tabulated millimetres says that as
directly.

Projection reuses robosuite's own camera matrix, inverted the other way round
from :mod:`tpgpt.perception.cameras`: that module takes pixels to world, this
one takes world to pixels. The same two conventions bite here as there -- the
observation image is stored upside down relative to the matrix, and the pixel
vector is ``[col, row]`` rather than ``[row, col]``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tpgpt.grasp.grasps import contact_offset, grasp_to_eef_pose
from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

#: Colours shared with the 3-D figures so a reader can move between them.
SOURCE_COLOUR = "#5b8ff9"
TARGET_COLOUR = "#ff7f0e"
CONTACT_COLOUR = "#2ca02c"
GRASP_COLOUR = "#d62728"


def project_to_pixels(env, camera: str, points: np.ndarray, height: int, width: int):
    """World points to ``(column, row)`` pixels, plus a visibility mask.

    Returns pixels in *display* orientation, i.e. matching the image as it is
    rendered rather than as robosuite stores it.
    """
    from robosuite.utils import camera_utils

    points = np.atleast_2d(np.asarray(points, dtype=float))
    matrix = camera_utils.get_camera_transform_matrix(env.sim, camera, height, width)
    homogeneous = np.c_[points, np.ones(len(points))]
    projected = homogeneous @ matrix.T
    depth = projected[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    pixels = projected[:, :2] / safe[:, None]
    columns, rows = pixels[:, 0], pixels[:, 1]
    # No vertical flip here, and that is the whole subtlety. robosuite stores
    # observation images bottom-up while the camera matrix assumes a top-down
    # raster, so :mod:`tpgpt.perception.cameras` flips the depth and mask
    # buffers *before* unprojecting -- which means the matrix's row convention
    # is the flipped, human-readable one already. ``_frame`` flips the picture
    # for the same reason, so drawing matrix rows onto it needs no correction.
    #
    # Flipping a second time here was silent and looked plausible: the columns
    # stayed correct, so keypoints landed on the right part of the scene
    # left-to-right and merely at the wrong height, which reads as a keypoint
    # that has drifted rather than a projection that is wrong. Measured by
    # round-tripping each object's own cloud back through this function, 88 to
    # 100 per cent of points land inside that object's mask unflipped, against
    # 0 to 63 per cent flipped.
    visible = (
        (depth > 0)
        & (columns >= 0) & (columns < width)
        & (rows >= 0) & (rows < height)
    )
    return np.column_stack([columns, rows]), visible


def gripper_outline(grasp, gripper) -> dict:
    """A schematic of where the hand would be, in world coordinates.

    Three segments a reader can interpret without knowing the frame contract:
    the approach the hand travels along, the line the jaws close on, and the
    point where the object ends up.
    """
    pair = resolve_pair(gripper) if isinstance(gripper, str) else gripper
    position, rotation = grasp_to_eef_pose(grasp, pair)
    held = position + rotation @ contact_offset(pair)
    half = gripper_geometry(pair.graspgen).aperture / 2.0
    return {
        "approach": np.array([held - grasp.approach * 0.08, held]),
        "jaws": np.array([held - grasp.closing * half, held + grasp.closing * half]),
        "held": held,
        "site": position,
    }


def _frame(env, camera: str, obs: dict | None, size: int):
    """The camera image, in display orientation."""
    if obs is not None and f"{camera}_image" in obs:
        return np.ascontiguousarray(np.asarray(obs[f"{camera}_image"])[::-1])
    return np.ascontiguousarray(
        env.sim.render(width=size, height=size, camera_name=camera)[::-1]
    )


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def draw_grasp(
    env,
    grasp,
    gripper: str,
    path: str | Path,
    camera: str = "sideview",
    size: int = 512,
    obs: dict | None = None,
    title: str = "",
    alternatives: list | None = None,
) -> Path:
    """The chosen grasp drawn on a picture of the scene.

    Args:
        alternatives: Other surviving candidates, drawn faintly, so the reader
            can see whether the chosen one was picked from a rich set or was the
            only thing left.
    """
    image = _frame(env, camera, obs, size)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)

    for other in alternatives or []:
        outline = gripper_outline(other, gripper)
        pixels, visible = project_to_pixels(env, camera, outline["jaws"], height, width)
        if visible.all():
            ax.plot(pixels[:, 0], pixels[:, 1], color="#bbbbbb", lw=1.2, alpha=0.65)

    outline = gripper_outline(grasp, gripper)
    for key, colour, lw, label in (
        ("approach", GRASP_COLOUR, 2.0, "approach direction"),
        ("jaws", CONTACT_COLOUR, 3.0, "where the jaws close"),
    ):
        pixels, _ = project_to_pixels(env, camera, outline[key], height, width)
        ax.plot(pixels[:, 0], pixels[:, 1], color=colour, lw=lw, label=label)
    held, _ = project_to_pixels(env, camera, outline["held"][None], height, width)
    ax.scatter(held[:, 0], held[:, 1], s=70, facecolors="none",
               edgecolors=CONTACT_COLOUR, linewidths=2.0, label="where the object is held")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=10)
    return _save(fig, path)


def draw_keypoints(
    env,
    keypoint_sets: dict,
    path: str | Path,
    camera: str = "sideview",
    size: int = 512,
    obs: dict | None = None,
    title: str = "",
) -> Path:
    """Keypoints drawn on the scene they describe.

    Args:
        keypoint_sets: ``{label: (KeypointSet, colour)}``. Usually the pick and
            place blocks, so a reader can see both ends of the task at once.
    """
    image = _frame(env, camera, obs, size)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)

    for label, (keypoints, colour) in keypoint_sets.items():
        pixels, visible = project_to_pixels(env, camera, keypoints.points, height, width)
        ax.scatter(
            pixels[visible, 0], pixels[visible, 1], s=34, c=colour,
            edgecolors="k", linewidths=0.5, label=label, zorder=3,
        )
        # Mark the ones resting on a surface: those are the contact guarantee.
        on_surface = [
            i for i, name in enumerate(keypoints.labels)
            if name.endswith(("nnn", "pnn", "ppn", "npn", "support"))
        ]
        if on_surface:
            mask = visible[on_surface]
            ax.scatter(
                pixels[on_surface][mask, 0], pixels[on_surface][mask, 1],
                s=90, facecolors="none", edgecolors=CONTACT_COLOUR,
                linewidths=1.4, zorder=4,
            )

    ax.scatter([], [], s=90, facecolors="none", edgecolors=CONTACT_COLOUR,
               linewidths=1.4, label="touching a surface")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=10)
    return _save(fig, path)


def draw_trajectory_overlay(
    env,
    paths: dict,
    path: str | Path,
    camera: str = "sideview",
    size: int = 512,
    obs: dict | None = None,
    title: str = "",
) -> Path:
    """Trajectories drawn on the scene, so a warp can be seen against real walls.

    Args:
        paths: ``{label: (points, colour)}``.
    """
    image = _frame(env, camera, obs, size)
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)
    for label, (points, colour) in paths.items():
        pixels, visible = project_to_pixels(env, camera, points, height, width)
        segment = np.where(visible, 1, np.nan)
        ax.plot(pixels[:, 0] * segment, pixels[:, 1] * segment,
                color=colour, lw=2.0, label=label)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=10)
    return _save(fig, path)


def draw_keypoints_on_capture(capture: dict, keypoint_sets: dict, path, title: str = ""):
    """Keypoints drawn on a scene that no longer exists.

    The source demonstration runs in its own environment, which is closed before
    any target scene is built, so its keypoints cannot be projected live. A
    capture carries the picture and its projection matrix together (see
    :func:`tpgpt.experiments.reshelving_pipeline.record_source_placement`).
    """
    image = np.asarray(capture["image"])
    height, width = int(capture["height"]), int(capture["width"])
    matrix = np.asarray(capture["matrix"])

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)
    for label, (keypoints, colour) in keypoint_sets.items():
        points = np.atleast_2d(np.asarray(keypoints.points, dtype=float))
        projected = np.c_[points, np.ones(len(points))] @ matrix.T
        depth = projected[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        columns = projected[:, 0] / safe
        # No flip: see :func:`project_to_pixels` for why the matrix's rows are
        # already in the orientation the captured picture is stored in.
        rows = projected[:, 1] / safe
        visible = (
            (depth > 0) & (columns >= 0) & (columns < width)
            & (rows >= 0) & (rows < height)
        )
        ax.scatter(columns[visible], rows[visible], s=34, c=colour,
                   edgecolors="k", linewidths=0.5, label=label, zorder=3)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=10)
    return _save(fig, path)


def build_figures(result, env, out_dir, obs: dict | None = None, camera: str = "workspace"):
    """Every picture one run's report needs.

    Returns a dict of paths keyed as the report expects. Anything that could not
    be drawn -- a run that failed before it got that far -- is simply absent, and
    the report omits that section rather than showing an empty frame.
    """
    from tpgpt.viz.keypoint_figures import figure_transported_trajectory

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    if result.grasp is not None:
        figures["grasp"] = draw_grasp(
            env, result.grasp, result.gripper, out_dir / "grasp.png",
            camera=camera, obs=obs, alternatives=result.alternatives,
            title=f"chosen grasp on the {result.object_name}",
        )

    capture = getattr(result, "source_capture", None)
    if capture and result.source_keypoints is not None:
        blocks = {}
        for prefix, colour in (("pick", SOURCE_COLOUR), ("place", CONTACT_COLOUR)):
            block = _keypoint_block(result.source_keypoints, prefix)
            if block is not None:
                blocks[f"{prefix} (the demonstration)"] = (block, colour)
        if blocks:
            figures["keypoints_source"] = draw_keypoints_on_capture(
                capture, blocks, out_dir / "keypoints_source.png",
                title="keypoints in the scene the demonstration was recorded in",
            )

    if result.target_keypoints is not None:
        blocks = {}
        for prefix, colour in (("pick", TARGET_COLOUR), ("place", CONTACT_COLOUR)):
            block = _keypoint_block(result.target_keypoints, prefix)
            if block is not None:
                blocks[f"{prefix} (where it {'starts' if prefix == 'pick' else 'must end'})"] = (
                    block, colour,
                )
        if blocks:
            figures["keypoints_target"] = draw_keypoints(
                env, blocks, out_dir / "keypoints_target.png", camera=camera, obs=obs,
                title=f"keypoints on the {result.object_name} and its destination",
            )

    if result.demonstration is not None and result.transported is not None:
        surfaces = {
            "source": float(result.source_keypoints.points[:, 2].min()),
            "target": float(env.slot_poses()[result.slot][2]),
        }
        figures["trajectory"] = figure_transported_trajectory(
            result.demonstration, result.transported,
            result.source_keypoints, result.target_keypoints, surfaces,
            out_dir / "trajectory.png",
            release_index=result.metrics.get("release_index"),
            title=f"one demonstration, transported onto the {result.object_name}",
        )
        figures["trajectory_scene"] = draw_trajectory_overlay(
            env,
            {"transported motion": (result.transported, TARGET_COLOUR)},
            out_dir / "trajectory_scene.png", camera=camera, obs=obs,
            title="the transported path, against the real shelf",
        )
    return figures


def _keypoint_block(keypoints, prefix):
    """The pick or place half of a paired keypoint set."""
    from tpgpt.sim.keypoints import KeypointSet

    index = [i for i, label in enumerate(keypoints.labels) if label.startswith(f"{prefix}_")]
    if not index:
        return None
    return KeypointSet(
        points=keypoints.points[index],
        labels=[keypoints.labels[i] for i in index],
        metadata=keypoints.metadata,
    )
