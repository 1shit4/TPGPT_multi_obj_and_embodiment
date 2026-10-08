#!/usr/bin/env python3
"""Pictures: how the pick grasp's yaw and the destination's direction together decide whether the
transport map folds (2026-10-08). Companion to explain_plate_yaw.py, same drawing.

Four test grasps on the apple: the demonstration's own grasp (tilted 26 deg, as it held the
banana), moved to the apple and turned about the vertical by 0, 90, 180 and 270 deg, through seven
cases: the shelf and the plate as now, each with no twin rule and with the planner's; and the plate
with its placed yaw turned, with no twin rule, the planner's, and the rule applied against the
demonstration turned the same way. The twin rule: a grasp more than a quarter turn from the
demonstration's is replaced by its half-turned twin (the same grip, the fingers' roles swapped).

    ~/venvs/tpgpt/bin/python scripts/explain_grasp_yaw.py

Writes docs/hardware/figures/grasp_yaw/*.png; docs/hardware/GRASP_YAW_EXPLAINED.md orders them.
The two plate variants are made inside this script only; the planner is unchanged.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_plate_yaw as E                                      # noqa: E402
import tpgpt.sim.keypoints as kp                                   # noqa: E402
from tpgpt.hardware.planner import KEYPOINT_BOX, KEYPOINT_ORIENTATION  # noqa: E402
from tpgpt.hardware.shelf import slots                             # noqa: E402
from tpgpt.sim.keypoints import GRASP_CUBE_HALF_EXTENT, GraspFrame, ObjectPlacement  # noqa: E402
from tpgpt.transport.maps import TransportMap                      # noqa: E402

YAWS = (0, 90, 180, 270)
# key: (label, destination, placed yaw turned, twin rule) -- the twin rule is "off" (the grasp as it
# is), "planner" (the planner's: twin if its closing is more than a quarter turn from the
# demonstration's) or "turned" (the same, against the demonstration turned by the carry turn)
CASES = {
    "shelf_off": ("Shelf, no twin rule", "shelf", False, "off"),
    "shelf_planner": ("Shelf, the planner's twin rule", "shelf", False, "planner"),
    "plate_off": ("Plate as now, no twin rule", "plate", False, "off"),
    "plate_planner": ("Plate as now, the planner's twin rule", "plate", False, "planner"),
    "plateyaw_off": ("Plate, placed yaw turned, no twin rule", "plate", True, "off"),
    "plateyaw_planner": ("Plate, placed yaw turned, the planner's twin rule", "plate", True, "planner"),
    "plateyaw_turned": ("Plate, placed yaw turned, twin rule against the turned demonstration", "plate", True, "turned"),
}
DET_MIN = 0.1


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def carry_turn(env, dest):
    L = env["source"].demonstration_labels
    g, r = kp.carry_indices(L)
    d_s = np.asarray(L.positions[r]) - np.asarray(L.positions[g])
    d_t = np.asarray(dest) - env["fit"].position
    return float(np.arctan2(d_t[1], d_t[0]) - np.arctan2(d_s[1], d_s[0]))


def run(env, dest, yaw_deg, turn_place, twin):
    """The planner's keypoints and map for the demonstration's grasp turned by ``yaw_deg``."""
    src = env["source"]
    g0 = src.placement.grasp
    R = Rz(np.radians(yaw_deg))
    grasp = GraspFrame(tcp=np.median(env["cloud"], axis=0), approach=R @ np.asarray(g0.approach),
                       closing=R @ np.asarray(g0.closing))
    target = ObjectPlacement(points=env["cloud"], grasp=grasp, support_height=env["table"],
                             destination=np.asarray(dest, float), destination_height=float(dest[2]))
    turn = carry_turn(env, dest)
    # scene_keypoints applies the planner's rule itself; flip_target cancels or redirects it
    now = float(np.asarray(grasp.closing) @ np.asarray(g0.closing)) < 0.0
    if twin == "planner":
        flip = False
    elif twin == "off":
        flip = now
    else:
        want = float(np.asarray(grasp.closing) @ (Rz(turn) @ np.asarray(g0.closing))) < 0.0
        flip = want != now
    original = kp.placement_rotation
    if turn_place:
        kp.placement_rotation = lambda a, b: Rz(turn) @ original(a, b)
    try:
        S, T, diag = kp.scene_keypoints(src.placement, target, src.demonstration_labels,
                                        include_contacts=False, box=KEYPOINT_BOX,
                                        orientation=KEYPOINT_ORIENTATION,
                                        cube_half_extent=GRASP_CUBE_HALF_EXTENT, flip_target=flip)
    finally:
        kp.placement_rotation = original
    tmap = TransportMap().fit(S.points, T.points)
    X = np.asarray(src.tool_labels.positions, float)
    return dict(map=tmap, S=S.points, T=T.points, step1=tmap.affine.predict(S.points), X=X,
                phi=tmap.transport_positions(X), det=tmap.check_diffeomorphism(X).determinants,
                rolled=bool(diag["target_grasp_rolled"]), turn=turn)


def arrows(ax, case):
    for k in range(8):
        name = "pick" if k < 4 else "place"
        c_t, top_t = E.cubes(case["T"])[name]
        c_v, top_v = E.cubes(case["step1"])[name]
        ax.annotate("", xy=c_t[:2] + E.MAGNIFY * (top_t[k % 4, :2] - c_t[:2]),
                    xytext=c_v[:2] + E.MAGNIFY * (top_v[k % 4, :2] - c_v[:2]),
                    arrowprops=dict(arrowstyle="-|>", color=E.FOLD, lw=1.4, shrinkA=4, shrinkB=4))
    ax.plot([], [], "-", color=E.FOLD, lw=1.4, label="what the second step must still move (violet -> green)")


def verdict(case):
    return "folds" if case["det"].min() <= DET_MIN else "bends, no fold"


def where_it_folds(case):
    """Which end the worst fold is nearer to, in the new scene."""
    k = int(case["det"].argmin())
    p = case["phi"][k]
    dp, dq = np.linalg.norm(p - case["T"][0]), np.linalg.norm(p - case["T"][9])
    return "the pick" if dp < dq else "the place"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--object", default="apple")
    ap.add_argument("--capture", default="~/tpgpt_plans/apple_20261007_014529_grasptest")
    ap.add_argument("--demo", default="~/tpgpt_demos/banana_20260925_183904/source_bundle_learned")
    ap.add_argument("--config-dir", default="~/thor_ws/install/ycb_pointcloud_pipeline/share/"
                                            "ycb_pointcloud_pipeline/config")
    ap.add_argument("--out", default="docs/hardware/figures/grasp_yaw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        if f.endswith(".png"):
            os.remove(os.path.join(a.out, f))
    env = E.setup(a)
    sl = slots(env["cfg"])
    dests = {"shelf": np.asarray(sl["top_left"]), "plate": np.asarray(sl["plate"])}
    out = lambda name: os.path.join(a.out, name)  # noqa: E731
    mag = f"Cubes drawn {E.MAGNIFY:.0f}x their real 40 mm, about their own centres (positions and turns exact)."

    # ---- 1. the demonstration's grasp; 2. the four test grasps
    src = env["source"]
    g0 = src.placement.grasp
    fig, ax = E.new_fig()
    pts = np.asarray(src.placement.points)
    ax.plot(pts[:, 0], pts[:, 1], ".", color="#c9c8c2", ms=2)
    c = np.asarray(g0.tcp)[:2]
    d = np.asarray(g0.closing)[:2] / np.linalg.norm(np.asarray(g0.closing)[:2]) * 0.06
    ax.annotate("", xy=c + d, xytext=c - d, arrowprops=dict(arrowstyle="-|>", color=E.INK, lw=3))
    ax.text(*(c + d * 1.25), "fingers close\nthis way", color=E.INK, fontsize=11, ha="center")
    ax.text(*(c + np.array([-0.02, -0.09])), "banana (the demonstration)", color=E.INK2, fontsize=11)
    E.finish(ax, fig, "1. The demonstration's grasp, from above",
             "The arrow shows which way the hand's closing direction points as it grasps the banana "
             "(the hand is tilted 26 degrees from vertical). The test grasps below are this same grasp, "
             "turned about the vertical.", out("01_demo_grasp.png"), (c[0] - 0.14, c[0] + 0.14), (c[1] - 0.14, c[1] + 0.14))

    fig, ax = E.new_fig()
    cl = env["cloud"]
    ax.plot(cl[:, 0], cl[:, 1], ".", color="#c9c8c2", ms=2)
    c = np.median(cl, axis=0)[:2]
    for yaw, colour in zip(YAWS, ("#2a78d6", "#1baf7a", "#e87ba4", "#eda100")):
        dd = (Rz(np.radians(yaw)) @ np.asarray(g0.closing))[:2]
        dd = dd / np.linalg.norm(dd) * 0.055
        ax.annotate("", xy=c + dd, xytext=c, arrowprops=dict(arrowstyle="-|>", color=colour, lw=3))
        ax.text(*(c + dd * 1.35), f"{yaw}°", color=colour, fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(*(c + np.array([-0.03, -0.075])), "apple", color=E.INK2, fontsize=11)
    E.finish(ax, fig, "2. Four test grasps on the apple: the demonstration's grasp turned 0, 90, 180, 270 degrees",
             "Each arrow is one test grasp's closing direction (same tilt as the demonstration). 0 deg grips the "
             "apple exactly the way the banana was gripped; 180 deg grips it from the opposite side.",
             out("02_test_grasps.png"), (c[0] - 0.12, c[0] + 0.12), (c[1] - 0.12, c[1] + 0.12))

    # ---- 3. the planner's twin rule
    fig, ax = E.new_fig()
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "-", color=E.MUTED, lw=1)
    for deg in range(0, 360, 15):
        t = np.radians(deg)
        tip = np.array([np.cos(t), np.sin(t)])
        swapped = np.cos(t) < -1e-9
        colour = E.FOLD if swapped else E.TARGET
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6, alpha=0.9))
        if deg in (120, 180, 240):          # three examples of where a swapped grasp goes
            ax.annotate("", xy=-tip * 0.9, xytext=tip * 0.9,
                        arrowprops=dict(arrowstyle="-|>", color=E.FOLD, lw=1.6, ls=":",
                                        connectionstyle="arc3,rad=0.35"))
        ax.text(*(tip * 1.13), f"{deg}°", ha="center", va="center", fontsize=8.5, color=E.INK2)
    ax.axvline(0, color=E.INK2, lw=1, ls="--")
    ax.text(0.04, 1.32, "quarter turn", color=E.INK2, fontsize=9)
    ax.text(0.55, -1.42, "kept as they are\n(within a quarter turn of the\ndemonstration's direction, 0°)",
            color=E.TARGET, fontsize=10, ha="center")
    ax.text(-0.55, -1.42, "swapped for their twin, which\npoints the opposite way\n(dotted: where three of them go)",
            color=E.FOLD, fontsize=10, ha="center")
    ax.plot([], [], "-", color=E.TARGET, lw=1.6, label="a grasp's closing direction, kept")
    ax.plot([], [], "-", color=E.FOLD, lw=1.6, label="a grasp's closing direction, swapped for its twin")
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor=E.SURFACE, edgecolor=E.MUTED)
    ax.set_axis_off()
    E.finish(ax, fig, "3. The planner's twin rule, in yaw",
             "Each arrow is a grasp's closing direction, measured from the demonstration's (0°). A grasp is the same "
             "grip as its half-turned twin (the fingers' roles swapped), and the twin points the opposite way. Any grasp "
             "more than a quarter turn from the demonstration's is replaced by its twin: 135° becomes -45°, 180° becomes "
             "0°. So in yaw no grasp is left more than a quarter turn off. (A grasp tilted very differently can still "
             "leave its cube further off.)", out("03_twin_rule.png"), (-1.7, 1.7), (-1.75, 1.6))

    # ---- the grid of cases
    results = {}
    n = 4
    for key, (label, where, turn_place, twin_rule) in CASES.items():
        dest = dests[where]
        for yaw in YAWS:
            case = run(env, dest, yaw, turn_place, twin_rule)
            results[(key, yaw)] = case
            twin = (" Here the grasp is replaced by its half-turned twin (the same grip with the fingers' roles "
                    "swapped), so its green pick cube is the twin's." if case["rolled"] else
                    " The grasp is used as it is (no twin).")
            v = verdict(case)
            why = (f"The map folds near {where_it_folds(case)}." if v == "folds" else "Nothing has to cross; the map only bends.")
            # cause
            fig, ax = E.new_fig()
            E.scene_extras(ax, env, plate=where == "plate")
            E.draw_cubes(ax, case["T"], E.TARGET, label="where the cubes must go")
            E.draw_cubes(ax, case["step1"], E.STEP1, style="--", label="the demonstration's cubes after the first step")
            arrows(ax, case)
            xl, yl = E.view_of(E.magnified(case["T"]), E.magnified(case["step1"]), margin=0.06)
            yl = (yl[0] - 0.13, yl[1])          # room under the cubes for the legend
            ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=E.SURFACE, edgecolor=E.MUTED)
            E.finish(ax, fig, f"{n}. {label}, grasp turned {yaw}°: {v}",
                     f"An arrow that cuts through the middle of its square means that corner must cross to the "
                     f"far side: that is what folds the map. {why}{twin} {mag}",
                     out(f"{n:02d}_{key}_{yaw:03d}_cubes.png"), xl, yl)
            n += 1
            # effect
            fig, ax = E.new_fig()
            E.scene_extras(ax, env, plate=where == "plate")
            z = float(np.median(case["X"][:, 2]))
            xs, ys = E.view_of(case["X"], case["S"], margin=0.08)
            M = E.graph_paper(ax, case, xs, ys, z)
            bad = case["det"] <= DET_MIN
            ax.plot(case["phi"][:, 0], case["phi"][:, 1], "-", color=E.INK, lw=2.4, zorder=4, label="the hand's new path")
            if bad.any():
                ax.plot(case["phi"][:, 0], np.where(bad, case["phi"][:, 1], np.nan), "-", color=E.FOLD, lw=4.5,
                        zorder=5, label="path through folded space")
            for nm, cc in (("pick", case["T"][0]), ("place", case["T"][9])):
                ax.plot(*cc[:2], "o", color=E.TARGET, ms=10, mec=E.SURFACE, mew=1.5, zorder=6)
                ax.text(cc[0] + 0.02, cc[1] + 0.02, nm, color=E.TARGET, fontsize=11, fontweight="bold", zorder=6)
            ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=E.SURFACE, edgecolor=E.MUTED)
            xl, yl = E.view_of(M.reshape(-1, 3), case["phi"], margin=0.05)
            E.finish(ax, fig, f"{n}. {label}, grasp turned {yaw}°: the space {'folds' if v == 'folds' else 'only bends'}",
                     "Graph paper laid over the demonstration's space and carried by the whole map. Orange cells: "
                     "space turned inside out. Thick orange path: the hand moving through it (the plan would be refused).",
                     out(f"{n:02d}_{key}_{yaw:03d}_space.png"), xl, yl)
            n += 1
            print(f"{key:<17} {yaw:3d} deg: {v:<15} min det {case['det'].min():+.2f}  twin {case['rolled']}", flush=True)

    # ---- the summary
    fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=E.SURFACE)
    ax.set_axis_off()
    labels = [CASES[k][0] for k in CASES]
    for i, key in enumerate(CASES):
        for j, yaw in enumerate(YAWS):
            case = results[(key, yaw)]
            folds = case["det"].min() <= DET_MIN
            ax.add_patch(plt.Rectangle((j, -i), 0.96, 0.9, fc="#f6c9b3" if folds else "#cfe8cf", ec=E.SURFACE))
            ax.text(j + 0.48, -i + 0.52, "folds" if folds else "works", ha="center", va="center",
                    fontsize=13, color=E.INK, fontweight="bold")
            if folds:
                ax.text(j + 0.48, -i + 0.25, f"at {where_it_folds(case)}", ha="center", va="center",
                        fontsize=10, color=E.INK2)
    for j, yaw in enumerate(YAWS):
        ax.text(j + 0.48, 1.15, f"grasp turned {yaw}°", ha="center", fontsize=11, color=E.INK)
    for i, lab in enumerate(labels):
        ax.text(-0.08, -i + 0.45, lab.replace(", twin rule against", ",\ntwin rule against").replace(", the planner", ",\nthe planner").replace(", no twin", ",\nno twin"),
                ha="right", va="center", fontsize=10.5, color=E.INK)
    ax.set_xlim(-3.0, 4.05); ax.set_ylim(-len(CASES) + 0.8, 1.5)
    fig.suptitle(f"{n}. Summary: which grasp works with which destination", x=0.02, ha="left", fontsize=15, color=E.INK)
    fig.text(0.02, 0.92, "A grasp works if, after the map's first step, both of its cubes are turned less than about a "
             "quarter turn from where they must go.\n90 and 270 deg sit exactly on the twin rule's boundary (a quarter turn).",
             fontsize=10.5, color=E.INK2)
    fig.subplots_adjust(top=0.85, bottom=0.03, left=0.02, right=0.98)
    fig.savefig(out(f"{n:02d}_summary.png"), dpi=130, facecolor=E.SURFACE)
    plt.close(fig)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
