#!/usr/bin/env python3
"""Why a destination opposite the demonstration folds the transport map, and why turning the
placed yaw with the carry direction unfolds it (2026-10-08, the plate).

One apple grasp from a recorded capture is planned three ways with the 18:39 learned bundle:

  A  top_left            the shelf, the demonstration's direction
  B  plate               the robot's left, the opposite direction, placed yaw = the demonstration's
  C  plate, yaw turned   the same, placed yaw turned by the change in carry direction

and for each the figure shows the keypoint cubes, the path after the affine part only (gamma,
a rigid rotation + translation fitted to all keypoints) and after the full map
(phi = gamma + psi.gamma), det(J) along the path, a horizontal slice of det(J) at the height of
the path's worst point with the fold (det <= 0) outlined, and the numbers behind it. A first
figure shows the demonstration alone.

    ~/venvs/tpgpt/bin/python scripts/plot_plate_yaw.py [--out docs/hardware/figures/plate_yaw]

Case C is produced by turning the placed yaw inside this script only (it replaces
tpgpt.sim.keypoints.placement_rotation for the run); the planner is unchanged.
"""

import argparse
import json
import os

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import tpgpt.hardware.planner as planner_mod                        # noqa: E402
import tpgpt.sim.keypoints as kp                                    # noqa: E402
from tpgpt.hardware.end_effector import inspire_from_config         # noqa: E402
from tpgpt.hardware.grasp_library import library, placed            # noqa: E402
from tpgpt.hardware.grasp_selection import (SelectionSettings, ScreenContext,  # noqa: E402
                                            _try_plan, screen_candidates)
from tpgpt.hardware.object_model import FULL_CLOUD_POINTS, ObjectModel, fit_object  # noqa: E402
from tpgpt.hardware.planner import TargetScene                      # noqa: E402
from tpgpt.hardware.shelf import ShelfModel, ShelfWorld, slots      # noqa: E402
from tpgpt.hardware.source_bundle import SourceBundle               # noqa: E402

# the reference palette (dataviz skill): text, surfaces, categorical slots
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29d"
SURFACE = "#fcfcfb"
TARGET = "#008300"      # green: the target scene's keypoints
AFFINE = "#4a3aa7"      # violet: after gamma only
FINAL = "#0b0b0b"       # ink: after the full map
SOURCE = "#8a8984"      # grey: the demonstration
FOLD = "#eb6834"        # orange: det(J) <= 0.1 (and the negative side of the slice)
POS = "#2a78d6"         # blue: the positive side of the slice
DET_MIN = 0.1           # the planner's min_determinant

CORNERS = list(kp.CORNER_NAMES)                    # nnn pnn ppn npn nnp pnp ppp npp
TOP = [CORNERS.index(c) for c in ("nnp", "pnp", "ppp", "npp")]


class RecordingMap(planner_mod.TransportMap):
    """The planner's map, remembered with the keypoints it was fitted to."""
    last = None

    def fit(self, S, T):
        RecordingMap.last = (self, np.asarray(S, float).copy(), np.asarray(T, float).copy())
        return super().fit(S, T)


def blocks(points):
    """The 18 keypoints as two cubes: (centre, 8 corners) for the pick and the place."""
    p = np.asarray(points)
    return {"pick": (p[0], p[1:9]), "place": (p[9], p[10:18])}


def cube_rotation(a, b):
    """Rotation angle (deg) taking cube corners ``a`` onto ``b`` (each centred), and its yaw."""
    A, B = a - a.mean(0), b - b.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return angle, np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def plan_once(source, scene, candidate, settings, hand):
    RecordingMap.last = None
    candidate.rejected = None
    result = _try_plan(source, scene, candidate, settings, {"end_effector": hand})
    return result, RecordingMap.last, candidate.rejected


def phase_of(k, g, r):
    return "approach" if k < g else ("carry" if k <= r else "retreat")


def setup(args):
    cfgdir = os.path.expanduser(args.config_dir)
    hand = inspire_from_config(yaml.safe_load(open(f"{cfgdir}/inspire_hand.yaml")))
    cfg = yaml.safe_load(open(f"{cfgdir}/shelf.yaml"))
    table = float(cfg["shelf"]["anchors"]["table"])
    cap = os.path.expanduser(args.capture)
    cam = np.asarray(json.load(open(cap + "/plan.json"))["metadata"]["T_base_camera"])[:3, 3]
    clouds = np.load(cap + "/clouds.npz")
    model = ObjectModel.load(args.object)
    fit = fit_object(model, clouds["observed"], table, cam)
    pts, _ = model.sample(FULL_CLOUD_POINTS, 1)
    cloud = pts @ fit.rotation.T + fit.position
    world = ShelfWorld(model=ShelfModel.from_config(cfg), cloud=clouds["obstacles"], table_height=table)
    source = SourceBundle.load(os.path.expanduser(args.demo))
    poses, scores = library(args.object, gripper=hand.graspgen)
    poses = placed(poses, fit.rotation, fit.position)
    settings = SelectionSettings(max_tilt_deg=90.0, check_centred=False, check_visibility=False,
                                 grasp_offset=(0, 0, 0), kinematic_criterion="strict",
                                 max_finger_lead_s=0.3, max_plans=25)
    return dict(hand=hand, cfg=cfg, table=table, cam=cam, cloud=cloud, world=world,
                source=source, poses=poses, scores=scores, settings=settings, fit=fit)


def survivors(env, dest):
    ctx = ScreenContext(camera_position=env["cam"], world=env["world"], destination=dest[:2],
                        destination_height=float(dest[2]),
                        carry_rotation=kp.carry_transform(env["source"].demonstration_labels)[0])
    return {c.index: c for c in screen_candidates(env["cloud"], env["poses"], env["scores"],
                                                  env["table"], env["settings"], context=ctx,
                                                  end_effector=env["hand"]) if c.rejected is None}


def run_case(env, dest, candidate, turn):
    """Plan one case; ``turn`` (rad) turns the placed yaw about the vertical, 0 = as now."""
    original = kp.placement_rotation
    Rz = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
    kp.placement_rotation = (lambda a, b: Rz @ original(a, b)) if turn else original
    try:
        scene = TargetScene(points=env["cloud"], destination=dest, destination_height=float(dest[2]),
                            support_height=env["table"])
        result, rec, why = plan_once(env["source"], scene, candidate, env["settings"], env["hand"])
    finally:
        kp.placement_rotation = original
    tmap, S, T = rec
    X = np.asarray(env["source"].tool_labels.positions, float)
    gam = tmap.affine.predict(X)
    phi = tmap.transport_positions(X)
    det = tmap.check_diffeomorphism(X).determinants
    return dict(map=tmap, S=S, T=T, X=X, gamma=gam, phi=phi, det=det, result=result, why=why,
                gS=tmap.affine.predict(S))


def analyse(case, labels):
    g, r = kp.carry_indices(labels)
    A = case["map"].affine.A
    yaw = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    tilt = np.degrees(np.arccos(np.clip(A[2, 2], -1, 1)))
    out = {"gamma_yaw": yaw, "gamma_tilt": tilt}
    bt, bg = blocks(case["T"]), blocks(case["gS"])
    for name in ("pick", "place"):
        ang, cyaw = cube_rotation(bg[name][1], bt[name][1])
        out[f"{name}_mismatch"] = ang
        out[f"{name}_mismatch_yaw"] = cyaw
        out[f"{name}_corner_move_mm"] = float(np.linalg.norm(bt[name][1] - bg[name][1], axis=1).max() * 1000)
        out[f"{name}_centre_move_mm"] = float(np.linalg.norm(bt[name][0] - bg[name][0]) * 1000)
    det = case["det"]
    bad = np.flatnonzero(det <= DET_MIN)
    out["min_det"], out["min_k"] = float(det.min()), int(det.argmin())
    out["min_phase"] = phase_of(out["min_k"], g, r)
    runs = []
    for k in bad:
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    out["fold_runs"] = [(a, b, phase_of(a, g, r), phase_of(b, g, r)) for a, b in runs]
    out["g"], out["r"] = g, r
    return out


def det_column(case, xlim, ylim, zlim, n=56, nz=18):
    """det J on a grid over the scene; from above, each spot's worst value over the heights
    ``zlim``, the height where it is worst, and the folded volume (det <= 0)."""
    xs, ys, zs = np.linspace(*xlim, n), np.linspace(*ylim, n), np.linspace(*zlim, nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    P = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    d = case["map"].check_diffeomorphism(P).determinants.reshape(XX.shape)
    worst = d.min(axis=2)
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])
    k = np.unravel_index(int(d.argmin()), d.shape)
    return dict(X=XX[:, :, 0], Y=YY[:, :, 0], worst=worst, volume_cm3=float((d <= 0).sum() * cell * 1e6),
                deepest=(float(d[k]), P[np.ravel_multi_index(k, d.shape)]))


def draw_cube(ax, centre, corners, colour, style, label=None, axis_colour=None):
    top = corners[TOP][:, :2]
    loop = np.vstack([top, top[:1]])
    ax.plot(loop[:, 0], loop[:, 1], style, color=colour, lw=2, label=label)
    # the cube's +x (the grasp's closing direction): from the centre to the mid of its +x face
    plus_x = corners[[CORNERS.index(c) for c in ("pnn", "ppn", "pnp", "ppp")]].mean(0)
    ax.annotate("", xy=plus_x[:2], xytext=centre[:2],
                arrowprops=dict(arrowstyle="-|>", color=axis_colour or colour, lw=1.6))
    k = CORNERS.index("ppp")
    ax.plot(*corners[k, :2], "o", color=colour, ms=8, mec=SURFACE, mew=1.5)


def style_axes(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.grid(color="#e7e6e2", lw=0.6)


def figure_demo(env, out):
    L = env["source"].demonstration_labels
    T = env["source"].tool_labels
    g, r = kp.carry_indices(L)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 1.3]})
    case = env["demo_case"]
    bs = blocks(case["S"])
    ax = axes[0]
    P = np.asarray(T.positions)
    ax.plot(P[:, 0], P[:, 1], "-", color=SOURCE, lw=2, label="demonstration's tool path (learned command)")
    ax.plot(*P[g, :2], "s", color=INK, ms=7, label=f"grasp (label {g})")
    ax.plot(*P[r, :2], "D", color=INK, ms=7, label=f"release (label {r})")
    for name in ("pick", "place"):
        draw_cube(ax, bs[name][0], bs[name][1], SOURCE, "-")
        ax.annotate(f"{name} cube", bs[name][0][:2] + np.array([0.03, 0.02]), color=INK2, fontsize=9)
    d = P[r] - P[g]
    ax.annotate("", xy=P[r, :2], xytext=P[g, :2],
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, ls="--"))
    ax.text(*(P[g, :2] + d[:2] * 0.45 + np.array([0.02, 0])),
            f"carry direction {np.degrees(np.arctan2(d[1], d[0])):.0f}°\n{np.linalg.norm(d[:2]) * 1000:.0f} mm across, "
            f"{d[2] * 1000:+.0f} mm up", fontsize=9, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, robot base)", color=INK2)
    ax.set_ylabel("y (m, robot base: + = robot's left)", color=INK2)
    style_axes(ax, "The demonstration from above\n(banana: picked on the table, set on the shelf)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
              facecolor=SURFACE, edgecolor=MUTED)
    ax = axes[1]
    k = np.arange(len(P))
    ax.plot(k, P[:, 2] * 1000, color=SOURCE, lw=2)
    ax.axvspan(g, r, color="#e7e6e2", label="carried (hand closed)")
    ax.set_xlabel("label (time step of the demonstration)", color=INK2)
    ax.set_ylabel("tool height (mm, robot base)", color=INK2)
    style_axes(ax, "Tool height along the demonstration")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Each cube is 40 mm, centred on the grasp point, laid out in that grasp's pose; the arrow "
             "is its +x (the closing direction), the dot its 'ppp' corner.\nThe corner named 'ppp' here is "
             "paired with 'ppp' in every target cube, and so on for all eight.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "0_demonstration.png"), dpi=130)
    plt.close(fig)


def figure_case(env, case, info, name, title, out):
    S, T = blocks(case["S"]), blocks(case["T"])
    G = blocks(case["gS"])
    fig = plt.figure(figsize=(18, 11.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.5, 1, 1])
    # ---- the whole task, from above, over det(J): the worst over every height the path uses
    ax = fig.add_subplot(gs[0, 0])
    allxy = np.vstack([case["X"][:, :2], case["phi"][:, :2], case["gamma"][:, :2], case["T"][:, :2]])
    lo, hi = allxy.min(0) - 0.06, allxy.max(0) + 0.06
    zlim = (float(case["phi"][:, 2].min()) - 0.02, float(case["phi"][:, 2].max()) + 0.02)
    col = det_column(case, (lo[0], hi[0]), (lo[1], hi[1]), zlim)
    info["fold_volume_cm3"], info["deepest"] = col["volume_cm3"], col["deepest"]
    cmap = LinearSegmentedColormap.from_list("det", [FOLD, "#f6f5f2", POS])
    norm = TwoSlopeNorm(vmin=min(-0.5, float(col["worst"].min())), vcenter=0.0,
                        vmax=max(1.5, float(col["worst"].max())))
    im = ax.pcolormesh(col["X"], col["Y"], col["worst"], cmap=cmap, norm=norm, shading="auto", alpha=0.85)
    if (col["worst"] <= 0).any():
        ax.contour(col["X"], col["Y"], col["worst"], levels=[0.0], colors=[FOLD], linewidths=2)
        dv, dp = col["deepest"]
        ax.plot(*dp[:2], "*", color=INK, ms=13, mec=SURFACE, mew=1, zorder=6)
        ax.annotate(f"deepest fold: det {dv:.2f}\nat z = {dp[2] * 1000:+.0f} mm",
                    xy=dp[:2], xytext=(-130 if dp[0] > (lo[0] + hi[0]) / 2 else 12, -34),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor=MUTED, boxstyle="round,pad=0.2"))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"worst det J over z = {zlim[0] * 1000:+.0f}..{zlim[1] * 1000:+.0f} mm  (≤ 0: folded)",
                 color=INK2, fontsize=8.5)
    ax.plot(case["X"][:, 0], case["X"][:, 1], "-", color=SOURCE, lw=1.5, label="demonstration's path")
    ax.plot(case["gamma"][:, 0], case["gamma"][:, 1], "--", color=AFFINE, lw=2,
            label="after the affine part γ (rigid)")
    ax.plot(case["phi"][:, 0], case["phi"][:, 1], "-", color=FINAL, lw=2, label="after the full map φ")
    bad = case["det"] <= DET_MIN
    if bad.any():
        ax.plot(case["phi"][bad, 0], case["phi"][bad, 1], "o", color=FOLD, ms=3.5, mec="none",
                label=f"path points with det J ≤ {DET_MIN}")
    for nm in ("pick", "place"):
        ax.plot(*T[nm][0][:2], "s", color=TARGET, ms=9, mec=SURFACE, zorder=5)
        right = T[nm][0][0] > lo[0] + 0.7 * (hi[0] - lo[0])
        ax.annotate(f"target {nm}", T[nm][0][:2], xytext=(-78 if right else 8, 6), textcoords="offset points",
                    color=TARGET, fontsize=9, fontweight="bold")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("y (m, + = robot's left)", color=INK2)
    style_axes(ax, "The whole task from above\norange outline: where space folds (any height)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    ax.legend(fontsize=8, loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=MUTED)
    # ---- the two cubes, zoomed
    for col, nm in ((1, "pick"), (2, "place")):
        ax = fig.add_subplot(gs[0, col])
        draw_cube(ax, T[nm][0], T[nm][1], TARGET, "-", label=f"target {nm} cube")
        draw_cube(ax, G[nm][0], G[nm][1], AFFINE, "--", label=f"demonstration's {nm} cube after γ")
        k = CORNERS.index("ppp")
        ax.annotate("", xy=T[nm][1][k, :2], xytext=G[nm][1][k, :2],
                    arrowprops=dict(arrowstyle="-|>", color=FOLD, lw=1.6))
        ang = info[f"{nm}_mismatch"]
        ax.text(0.02, 0.02, f"γ leaves this cube turned {ang:.0f}° from the target\n"
                f"(yaw {info[f'{nm}_mismatch_yaw']:+.0f}°): ψ must move its corners up to "
                f"{info[f'{nm}_corner_move_mm']:.0f} mm\n(orange: corner 'ppp' -> where it belongs)",
                transform=ax.transAxes, fontsize=9, color=INK, va="bottom",
                bbox=dict(facecolor=SURFACE, edgecolor=MUTED, boxstyle="round,pad=0.3"))
        both = np.vstack([T[nm][1][:, :2], G[nm][1][:, :2]])
        c = (both.min(0) + both.max(0)) / 2.0
        span = float(np.max(both.max(0) - both.min(0))) / 2.0 + 0.012
        ax.set_xlim(c[0] - span, c[0] + span); ax.set_ylim(c[1] - span, c[1] + span)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)", color=INK2)
        style_axes(ax, f"The {nm} cube, from above (top face; arrow = +x)")
        ax.legend(fontsize=8, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=MUTED)
    # ---- det(J) along the path
    ax = fig.add_subplot(gs[1, 0:2])
    k = np.arange(len(case["det"]))
    ax.axvspan(info["g"], info["r"], color="#efeeea", label="carried (hand closed)")
    ax.axhline(DET_MIN, color=INK2, lw=1, ls="--", label=f"the planner's minimum, {DET_MIN}")
    ax.axhline(0.0, color=MUTED, lw=1)
    ax.plot(k, case["det"], "-", color=FINAL, lw=2, label="det J of φ along the demonstration's path")
    if bad.any():
        ax.fill_between(k, case["det"], DET_MIN, where=bad, color=FOLD, alpha=0.35,
                        label="below the minimum")
        ax.annotate(f"worst on the path: det J {info['min_det']:.2f}\nat label {info['min_k']} ({info['min_phase']})",
                    xy=(info["min_k"], info["min_det"]), xytext=(-150, 10), textcoords="offset points",
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2),
                    bbox=dict(facecolor=SURFACE, edgecolor=MUTED, boxstyle="round,pad=0.2"))
    ax.set_xlabel("label (time step)", color=INK2); ax.set_ylabel("det J", color=INK2)
    style_axes(ax, "det J along the path (1 = no stretch, 0 = squashed flat, < 0 = turned inside out)")
    ax.legend(fontsize=8, loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=MUTED, ncol=2)
    # ---- the numbers, and why
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    runs = info["fold_runs"]
    if runs:
        where = "\n".join(f"  labels {a}-{b} ({pa}{'' if pa == pb else '-' + pb})" for a, b, pa, pb in runs[:4])
        verdict = (f"FOLDS on the path: det J ≤ {DET_MIN} on {int(bad.sum())}\nof {len(bad)} labels:\n{where}\n"
                   f"  worst {info['min_det']:.2f} at label {info['min_k']} ({info['min_phase']})")
    else:
        verdict = f"no fold on the path:\n  det J ≥ {case['det'].min():.2f} on every label"
    dv, dp = info["deepest"]
    verdict += (f"\nin space: {info['fold_volume_cm3']:.0f} cm³ folded (det ≤ 0)" if info["fold_volume_cm3"] > 0
                else "\nin space: nothing folded")
    if info["fold_volume_cm3"] > 0:
        verdict += f",\n  deepest {dv:.2f} at ({dp[0]:.2f}, {dp[1]:.2f}, {dp[2]:.2f}) m"
    text = (f"{verdict}\n\n"
            f"γ (rigid, fitted to all 18 keypoints):\n  turns everything {info['gamma_yaw']:+.0f}° about the vertical"
            f" (tilt {info['gamma_tilt']:.0f}°)\n\n"
            f"what is left for ψ (a smooth correction):\n"
            f"  pick cube:  turned {info['pick_mismatch']:.0f}°, corners up to {info['pick_corner_move_mm']:.0f} mm\n"
            f"  place cube: turned {info['place_mismatch']:.0f}°, corners up to {info['place_corner_move_mm']:.0f} mm\n\n"
            "How a fold comes about: ψ must turn a cube's corners\n"
            "round inside a 40 mm cube while being ~0 a few cm\naway. "
            "Turning a region round by a smooth stretch passes\n"
            "through 'squashed flat' (det 0) and, for an uneven\n"
            "turn, 'inside out' (det < 0) around the cube.\n"
            f"\nPlanner: {'PLANNED' if case['result'] is not None else 'refused, ' + str(case['why']).split(':')[1].strip()}")
    ax.text(0.0, 1.0, text, va="top", fontsize=8.8, color=INK, family="monospace",
            transform=ax.transAxes)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--object", default="apple")
    ap.add_argument("--capture", default="~/tpgpt_plans/apple_20261007_014529_grasptest")
    ap.add_argument("--demo", default="~/tpgpt_demos/banana_20260925_183904/source_bundle_learned")
    ap.add_argument("--config-dir", default="~/thor_ws/install/ycb_pointcloud_pipeline/share/"
                                            "ycb_pointcloud_pipeline/config")
    ap.add_argument("--out", default="docs/hardware/figures/plate_yaw")
    ap.add_argument("--index", type=int, default=None, help="the library grasp to use")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    planner_mod.TransportMap = RecordingMap
    env = setup(a)
    sl = slots(env["cfg"])
    dest_shelf, dest_plate = np.asarray(sl["top_left"]), np.asarray(sl["plate"])
    L = env["source"].demonstration_labels
    g, r = kp.carry_indices(L)
    d_s = np.asarray(L.positions[r]) - np.asarray(L.positions[g])
    d_t = dest_plate - env["fit"].position
    turn = np.arctan2(d_t[1], d_t[0]) - np.arctan2(d_s[1], d_s[0])
    sa, sp = survivors(env, dest_shelf), survivors(env, dest_plate)
    common = [i for i in sp if i in sa]
    chosen = None
    for i in ([a.index] if a.index is not None else common[:40]):
        cA = run_case(env, dest_shelf, sa[i], 0.0)
        cB = run_case(env, dest_plate, sp[i], 0.0)
        cC = run_case(env, dest_plate, sp[i], turn)
        ok = cA["det"].min() > DET_MIN and cB["det"].min() <= DET_MIN and cC["det"].min() > DET_MIN
        print(f"grasp #{i}: min det A {cA['det'].min():.2f}  B {cB['det'].min():.2f}  C {cC['det'].min():.2f}"
              + ("  <- used" if ok or a.index is not None else ""), flush=True)
        if ok or a.index is not None:
            chosen = (i, cA, cB, cC)
            break
    if chosen is None:
        raise SystemExit("no grasp in the first 40 shows all three cases; pass --index")
    i, cA, cB, cC = chosen
    env["demo_case"] = cA
    figure_demo(env, a.out)
    titles = {"A_shelf_same_direction": (cA, "A  top_left: the demonstration's direction"),
              "B_plate_opposite": (cB, f"B  plate, opposite direction ({np.degrees(turn):.0f}° turn), placed yaw = the demonstration's"),
              "C_plate_yaw_turned": (cC, f"C  plate, placed yaw turned {np.degrees(turn):.0f}° with the carry direction")}
    for name, (case, title) in titles.items():
        info = analyse(case, L)
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if k != "fold_runs"},
              info["fold_runs"][:4])
        figure_case(env, case, info, name, f"{title}  (apple, library grasp #{i})", a.out)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
