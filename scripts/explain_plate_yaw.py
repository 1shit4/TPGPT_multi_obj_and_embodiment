#!/usr/bin/env python3
"""Pictures of why a plate on the robot's left folds the transport map, and why turning the
placed yaw with the carry direction fixes it (2026-10-08). One figure per file.

    ~/venvs/tpgpt/bin/python scripts/explain_plate_yaw.py [--grasp 373]

Writes docs/hardware/figures/plate_yaw/*.png; docs/hardware/PLATE_YAW_EXPLAINED.md shows them in
order. Case "plate, fixed" turns the placed yaw inside this script only (it replaces
tpgpt.sim.keypoints.placement_rotation for that run); the planner is unchanged.
"""

import argparse
import json
import os

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Circle, Polygon                     # noqa: E402

import tpgpt.hardware.planner as planner_mod                        # noqa: E402
import tpgpt.sim.keypoints as kp                                    # noqa: E402
from tpgpt.hardware.end_effector import inspire_from_config         # noqa: E402
from tpgpt.hardware.grasp_library import library, placed            # noqa: E402
from tpgpt.hardware.grasp_selection import (SelectionSettings, ScreenContext,  # noqa: E402
                                            _try_plan, screen_candidates)
from tpgpt.hardware.object_model import FULL_CLOUD_POINTS, ObjectModel, fit_object  # noqa: E402
from tpgpt.hardware.planner import TargetScene                      # noqa: E402
from tpgpt.hardware.shelf import ShelfModel, ShelfWorld, is_fixture, slots  # noqa: E402
from tpgpt.hardware.source_bundle import SourceBundle               # noqa: E402

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#fcfcfb"
DEMO, TARGET, STEP1, FOLD = "#8a8984", "#008300", "#4a3aa7", "#eb6834"
GRID = "#6f6e69"
MAGNIFY = 5.0                    # cubes are drawn this many times larger, about their centres
# the four corners of each cube's top face (its own +z), numbered 1-4 at the pick, 5-8 at the place
TOP = ("nnp", "pnp", "ppp", "npp")


class RecordingMap(planner_mod.TransportMap):
    last = None

    def fit(self, S, T):
        RecordingMap.last = (self, np.asarray(S, float).copy(), np.asarray(T, float).copy())
        return super().fit(S, T)


# ------------------------------------------------------------------------- planning
def setup(a):
    cfgdir = os.path.expanduser(a.config_dir)
    hand = inspire_from_config(yaml.safe_load(open(f"{cfgdir}/inspire_hand.yaml")))
    cfg = yaml.safe_load(open(f"{cfgdir}/shelf.yaml"))
    table = float(cfg["shelf"]["anchors"]["table"])
    cap = os.path.expanduser(a.capture)
    cam = np.asarray(json.load(open(cap + "/plan.json"))["metadata"]["T_base_camera"])[:3, 3]
    clouds = np.load(cap + "/clouds.npz")
    model = ObjectModel.load(a.object)
    fit = fit_object(model, clouds["observed"], table, cam)
    pts, _ = model.sample(FULL_CLOUD_POINTS, 1)
    cloud = pts @ fit.rotation.T + fit.position
    shelf = ShelfModel.from_config(cfg)
    world = ShelfWorld(model=shelf, cloud=clouds["obstacles"], table_height=table)
    source = SourceBundle.load(os.path.expanduser(a.demo))
    poses, scores = library(a.object, gripper=hand.graspgen)
    poses = placed(poses, fit.rotation, fit.position)
    settings = SelectionSettings(max_tilt_deg=90.0, check_centred=False, check_visibility=False,
                                 grasp_offset=(0, 0, 0), kinematic_criterion="strict",
                                 max_finger_lead_s=0.3, max_plans=25)
    return dict(hand=hand, cfg=cfg, table=table, cam=cam, cloud=cloud, world=world, shelf=shelf,
                source=source, poses=poses, scores=scores, settings=settings, fit=fit)


def candidate(env, dest, index):
    ctx = ScreenContext(camera_position=env["cam"], world=env["world"], destination=dest[:2],
                        destination_height=float(dest[2]),
                        carry_rotation=kp.carry_transform(env["source"].demonstration_labels)[0])
    for c in screen_candidates(env["cloud"], env["poses"], env["scores"], env["table"],
                               env["settings"], context=ctx, end_effector=env["hand"]):
        if c.index == index:
            return c
    raise SystemExit(f"grasp #{index} is not a candidate for {dest}")


def plan_case(env, dest, cand, turn):
    original = kp.placement_rotation
    Rz = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
    kp.placement_rotation = (lambda a, b: Rz @ original(a, b)) if turn else original
    try:
        cand.rejected = None
        scene = TargetScene(points=env["cloud"], destination=dest, destination_height=float(dest[2]),
                            support_height=env["table"])
        result = _try_plan(env["source"], scene, cand, env["settings"], {"end_effector": env["hand"]})
    finally:
        kp.placement_rotation = original
    tmap, S, T = RecordingMap.last
    X = np.asarray(env["source"].tool_labels.positions, float)
    return dict(map=tmap, S=S, T=T, step1=tmap.affine.predict(S), X=X, phi=tmap.transport_positions(X),
                det=tmap.check_diffeomorphism(X).determinants, planned=result is not None)


# ------------------------------------------------------------------------- drawing
def cubes(points):
    """{'pick': (centre, top 4 corners), 'place': (...)} from the 18 keypoints."""
    p = np.asarray(points)
    idx = [kp.CORNER_NAMES.index(n) for n in TOP]
    return {"pick": (p[0], p[1:9][idx]), "place": (p[9], p[10:18][idx])}


def draw_cubes(ax, points, colour, style="-", numbers=True, label=None, lw=2.2, alpha=1.0):
    for k, name in enumerate(("pick", "place")):
        c, top = cubes(points)[name]
        big = c[:2] + MAGNIFY * (top[:, :2] - c[:2])
        loop = np.vstack([big, big[:1]])
        ax.plot(loop[:, 0], loop[:, 1], style, color=colour, lw=lw, alpha=alpha,
                label=label if k == 0 else None)
        ax.fill(big[:, 0], big[:, 1], color=colour, alpha=0.08 * alpha)
        if numbers:
            for j, corner in enumerate(big):
                n = 4 * k + j + 1
                out = corner + 0.22 * (corner - c[:2])
                ax.plot(*corner, "o", color=colour, ms=7, mec=SURFACE, mew=1.2, alpha=alpha)
                ax.text(*out, str(n), color=colour, fontsize=11, fontweight="bold",
                        ha="center", va="center", alpha=alpha)


def scene_extras(ax, env, plate=True, shelf=True):
    if shelf:
        for b in env["shelf"].boxes:
            if is_fixture(b.name) or b.position[2] + b.size[2] < env["table"] + 0.02:
                continue
            corners = np.array([[sx, sy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            xy = b.position[:2] + (corners * b.size[:2]) @ b.rotation[:2, :2].T
            ax.add_patch(Polygon(xy, closed=True, fc="#e7e6e2", ec=MUTED, lw=0.8, zorder=0))
    if plate:
        p = env["cfg"]["shelf"]["plate"]
        ax.add_patch(Circle(p["centre"], p["outer_radius"], fc="#efeeea", ec=MUTED, lw=1, zorder=0))


def finish(ax, fig, title, how, path, xlim=None, ylim=None):
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.set_xlabel("forward from the robot (m)", color=INK2)
    ax.set_ylabel("to the robot's left (m)", color=INK2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=15, color=INK, y=0.985)
    fig.text(0.02, 0.935, how, ha="left", va="top", fontsize=10.5, color=INK2, wrap=True)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.97)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def new_fig():
    fig, ax = plt.subplots(figsize=(10, 9.5), facecolor=SURFACE)
    return fig, ax


def magnified(points):
    """Every drawn (magnified) corner of both cubes, for sizing a view."""
    out = []
    for name in ("pick", "place"):
        c, top = cubes(points)[name]
        out.append(c[:2] + MAGNIFY * (top[:, :2] - c[:2]))
    return np.vstack(out)


def view_of(*arrays, margin=0.12):
    xy = np.vstack([np.asarray(a)[:, :2] for a in arrays])
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    c, half = (lo + hi) / 2, max(hi - lo) / 2
    return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half)


def graph_paper(ax, case, xlim, ylim, z, n=46):
    """A square grid in the demonstration's space at height z, carried by the full map; the
    cells it turns inside out (det J <= 0) filled orange."""
    xs, ys = np.linspace(*xlim, n), np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    P = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)])
    M = case["map"].transport_positions(P).reshape(n, n, 3)
    det = case["map"].check_diffeomorphism(P).determinants.reshape(n, n)
    for i in range(n - 1):
        for j in range(n - 1):
            if min(det[i, j], det[i + 1, j], det[i, j + 1], det[i + 1, j + 1]) <= 0:
                quad = M[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j], :2]
                ax.add_patch(Polygon(quad, closed=True, fc=FOLD, ec="none", alpha=0.55, zorder=1))
    for i in range(n):
        ax.plot(M[i, :, 0], M[i, :, 1], "-", color=GRID, lw=0.5, zorder=2)
        ax.plot(M[:, i, 0], M[:, i, 1], "-", color=GRID, lw=0.5, zorder=2)
    return M


# ------------------------------------------------------------------------- figures
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--object", default="apple")
    ap.add_argument("--grasp", type=int, default=373)
    ap.add_argument("--bad-grasp", type=int, default=165, help="a grasp the fix does not save (pick half-turned)")
    ap.add_argument("--capture", default="~/tpgpt_plans/apple_20261007_014529_grasptest")
    ap.add_argument("--demo", default="~/tpgpt_demos/banana_20260925_183904/source_bundle_learned")
    ap.add_argument("--config-dir", default="~/thor_ws/install/ycb_pointcloud_pipeline/share/"
                                            "ycb_pointcloud_pipeline/config")
    ap.add_argument("--out", default="docs/hardware/figures/plate_yaw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        if f.endswith(".png"):
            os.remove(os.path.join(a.out, f))
    planner_mod.TransportMap = RecordingMap
    env = setup(a)
    sl = slots(env["cfg"])
    shelf_dest, plate_dest = np.asarray(sl["top_left"]), np.asarray(sl["plate"])
    L = env["source"].demonstration_labels
    g, r = kp.carry_indices(L)
    d_s = np.asarray(L.positions[r]) - np.asarray(L.positions[g])
    d_t = plate_dest - env["fit"].position
    turn = np.arctan2(d_t[1], d_t[0]) - np.arctan2(d_s[1], d_s[0])
    A = plan_case(env, shelf_dest, candidate(env, shelf_dest, a.grasp), 0.0)
    B = plan_case(env, plate_dest, candidate(env, plate_dest, a.grasp), 0.0)
    C = plan_case(env, plate_dest, candidate(env, plate_dest, a.grasp), turn)
    print(f"grasp #{a.grasp}: min det shelf {A['det'].min():.2f}, plate {B['det'].min():.2f}, "
          f"plate fixed {C['det'].min():.2f}")
    out = lambda name: os.path.join(a.out, name)  # noqa: E731
    mag = f"Cubes drawn {MAGNIFY:.0f}x their real 40 mm, about their own centres (positions and turns exact)."

    # 1. the scene
    fig, ax = new_fig()
    scene_extras(ax, env)
    ax.plot(*env["cloud"][:, :2].T, ".", color="#c9c8c2", ms=1)
    P = A["X"]
    ax.plot(P[:, 0], P[:, 1], "-", color=DEMO, lw=2, label="the demonstration's hand path")
    pick = env["fit"].position[:2]
    for dest, colour, text in ((shelf_dest, TARGET, "to the shelf (works)"),
                               (plate_dest, FOLD, "to the plate (fails)")):
        ax.annotate("", xy=dest[:2], xytext=pick, arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.5))
        ax.text(*(0.55 * pick + 0.45 * dest[:2] + np.array([0.02, 0.0])), text, color=colour, fontsize=11,
                fontweight="bold")
    ax.annotate("", xy=P[r, :2], xytext=P[g, :2], arrowprops=dict(arrowstyle="-|>", color=DEMO, lw=2.5, ls="--"))
    ax.text(*(P[g, :2] * 0.5 + P[r, :2] * 0.5 + np.array([0.03, 0.05])), "demonstration:\npick -> shelf",
            color=INK2, fontsize=10)
    ax.text(*(np.asarray(env["cfg"]["shelf"]["plate"]["centre"]) + np.array([-0.06, -0.16])), "plate", color=INK2, fontsize=11)
    ax.text(0.60, -0.38, "shelf", color=INK2, fontsize=11)
    ax.text(*(pick + np.array([0.03, 0.03])), "object to pick", color=INK2, fontsize=10)
    ax.plot(0, 0, "s", color=INK, ms=10)
    ax.text(0.02, -0.04, "robot base", color=INK, fontsize=10)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=MUTED)
    finish(ax, fig, "1. The scene, from above",
           "The demonstration carried the banana from the table to the shelf, to the robot's right. "
           "The shelf lies the same way, so the demonstration bends a little to reach it. "
           "The plate lies the other way, to the robot's left.", out("1_scene.png"),
           (-0.05, 0.85), (-0.45, 0.70))

    # 2. the demonstration's cubes
    xl, yl = view_of(magnified(A["S"]), A["X"], margin=0.08)
    fig, ax = new_fig()
    ax.plot(A["X"][:, 0], A["X"][:, 1], "-", color=DEMO, lw=1.5, alpha=0.6)
    draw_cubes(ax, A["S"], DEMO, label="the demonstration's two cubes")
    for name, c in (("pick", A["S"][0]), ("place", A["S"][9])):
        right = magnified(A["S"])[:, 0].max()
        ax.text(right + 0.03, c[1], f"{name} cube", color=INK2, fontsize=12, va="center")
    finish(ax, fig, "2. The demonstration's keypoints: a cube at the pick (1-4), a cube at the place (5-8)",
           f"Only each cube's top four corners are drawn and numbered; the other four sit under them. "
           f"{mag} Grey line: the hand's path between them.", out("2_demonstration_cubes.png"), xl, yl)

    # 3-5. what each target needs (green) vs where step 1 puts the demonstration (violet)
    for case, name, title, how in (
            (A, "3_shelf_step1.png", "3. Shelf: the first, rigid step lines both cubes up",
             "Green: where the cubes must end up (the new scene). Violet: the demonstration's cubes after "
             "the map's first step -- one rotation and shift for everything. Every violet number lands "
             "near its green partner, so the second step only nudges them."),
            (B, "4_plate_step1.png", "4. Plate as now: the first step turns everything round, but the plate's cube is not turned",
             "The first step turns the demonstration round to point at the plate, so the violet cubes come "
             "out turned half a turn. The green plate cube was built facing the demonstration's way, so at "
             "the plate every orange arrow has to cut through the middle of the square to the far corner. "
             "At the pick the arrows only run round the edge: a smaller turn, which bends but does not fold."),
            (C, "5_plate_fixed_step1.png", "5. Plate, fixed: build the plate's cube turned the way the first step turns everything",
             "Same first step. The green plate cube is now turned half a turn too (a round plate does not "
             "care which way the object faces), so every violet number lands near its green partner again.")):
        fig, ax = new_fig()
        scene_extras(ax, env, plate=case is not A)
        draw_cubes(ax, case["T"], TARGET, label="where the cubes must go")
        draw_cubes(ax, case["step1"], STEP1, style="--", label="the demonstration's cubes after the first step")
        for k in range(8):
            cube_name = "pick" if k < 4 else "place"
            c_t, top_t = cubes(case["T"])[cube_name]
            c_v, top_v = cubes(case["step1"])[cube_name]
            p_t = c_t[:2] + MAGNIFY * (top_t[k % 4, :2] - c_t[:2])
            p_v = c_v[:2] + MAGNIFY * (top_v[k % 4, :2] - c_v[:2])
            ax.annotate("", xy=p_t, xytext=p_v, arrowprops=dict(arrowstyle="-|>", color=FOLD, lw=1.4,
                                                                shrinkA=4, shrinkB=4))
        ax.plot([], [], "-", color=FOLD, lw=1.4, label="what the second step must still move (violet -> green)")
        xl, yl = view_of(magnified(case["T"]), magnified(case["step1"]), margin=0.06)
        ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=MUTED)
        finish(ax, fig, title, how + " " + mag, out(name), xl, yl)

    # 6-8. graph paper: the space carried by the full map, at the height the hand carries
    z = float(np.median(A["X"][g:r + 1, 2]))
    xs, ys = view_of(A["X"], A["S"], margin=0.08)
    for case, name, title, how in (
            (A, "6_shelf_space.png", "6. Shelf: the space bends but nowhere folds",
             "A square grid laid over the demonstration's space (at the carrying height), carried into the "
             "new scene by the full map. The lines bend smoothly and never cross. Black: the hand's new path."),
            (B, "7_plate_space.png", "7. Plate as now: around the plate the space is crumpled over itself -- and the path runs through it",
             "To send every plate-cube corner across to the far side while leaving the space further away "
             "alone, the map crumples the space around the plate over itself. Orange: grid cells turned "
             "inside out. Thick orange path: where the hand would move through that crumpled space, so the "
             "plan is refused."),
            (C, "8_plate_fixed_space.png", "8. Plate, fixed: the space bends again, no fold on the path",
             "With the plate's cube built turned, nothing has to be swapped: the grid bends smoothly and the "
             "hand's path stays in unfolded space, so the plan is accepted.")):
        fig, ax = new_fig()
        scene_extras(ax, env, plate=case is not A)
        M = graph_paper(ax, case, xs, ys, z)
        bad = case["det"] <= 0.1
        ax.plot(case["phi"][:, 0], case["phi"][:, 1], "-", color=INK, lw=2.4, zorder=4, label="the hand's new path")
        if bad.any():
            seg = np.where(bad, case["phi"][:, 1], np.nan)
            ax.plot(case["phi"][:, 0], seg, "-", color=FOLD, lw=4.5, zorder=5, label="path through folded space")
        for nm, c in (("pick", case["T"][0]), ("place", case["T"][9])):
            ax.plot(*c[:2], "o", color=TARGET, ms=10, mec=SURFACE, mew=1.5, zorder=6)
            ax.text(c[0] + 0.02, c[1] + 0.02, nm, color=TARGET, fontsize=11, fontweight="bold", zorder=6)
        ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=MUTED)
        xl, yl = view_of(M.reshape(-1, 3), case["phi"], margin=0.05)
        finish(ax, fig, title, how, out(name), xl, yl)
    # 9-10. what the fix does not change: a grasp whose own pick cube needs half a turn
    D = plan_case(env, plate_dest, candidate(env, plate_dest, a.bad_grasp), turn)
    print(f"grasp #{a.bad_grasp}, plate fixed: min det {D['det'].min():.2f}")
    fig, ax = new_fig()
    scene_extras(ax, env)
    draw_cubes(ax, D["T"], TARGET, label="where the cubes must go")
    draw_cubes(ax, D["step1"], STEP1, style="--", label="the demonstration's cubes after the first step")
    for k in range(8):
        cube_name = "pick" if k < 4 else "place"
        c_t, top_t = cubes(D["T"])[cube_name]
        c_v, top_v = cubes(D["step1"])[cube_name]
        ax.annotate("", xy=c_t[:2] + MAGNIFY * (top_t[k % 4, :2] - c_t[:2]),
                    xytext=c_v[:2] + MAGNIFY * (top_v[k % 4, :2] - c_v[:2]),
                    arrowprops=dict(arrowstyle="-|>", color=FOLD, lw=1.4, shrinkA=4, shrinkB=4))
    ax.plot([], [], "-", color=FOLD, lw=1.4, label="what the second step must still move (violet -> green)")
    xl, yl = view_of(magnified(D["T"]), magnified(D["step1"]), margin=0.06)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=MUTED)
    finish(ax, fig, "9. What the fix cannot change: a grasp whose own pick cube needs half a turn",
           "Another grasp of the same apple, plate fixed. The plate's cube now lines up, but this grasp holds "
           "the apple from the opposite side to the demonstration, so after the first step its pick cube is "
           "half a turn off: at the pick the arrows now cut through the middle. " + mag, out("9_other_grasp_step1.png"), xl, yl)
    fig, ax = new_fig()
    scene_extras(ax, env)
    M = graph_paper(ax, D, xs, ys, z)
    bad = D["det"] <= 0.1
    ax.plot(D["phi"][:, 0], D["phi"][:, 1], "-", color=INK, lw=2.4, zorder=4, label="the hand's new path")
    if bad.any():
        ax.plot(D["phi"][:, 0], np.where(bad, D["phi"][:, 1], np.nan), "-", color=FOLD, lw=4.5, zorder=5,
                label="path through folded space")
    for nm, c in (("pick", D["T"][0]), ("place", D["T"][9])):
        ax.plot(*c[:2], "o", color=TARGET, ms=10, mec=SURFACE, mew=1.5, zorder=6)
        ax.text(c[0] + 0.02, c[1] + 0.02, nm, color=TARGET, fontsize=11, fontweight="bold", zorder=6)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=MUTED)
    xl, yl = view_of(M.reshape(-1, 3), D["phi"], margin=0.05)
    finish(ax, fig, "10. ...so that grasp crumples the space around the pick instead, and is refused",
           "The same crumpling as figure 7, now at the pick. The planner refuses this grasp and tries the next "
           "one; grasps that hold the apple roughly the way the demonstration held the banana (turned with "
           "the task) pass. That is why only some grasps reach the plate.", out("10_other_grasp_space.png"), xl, yl)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
