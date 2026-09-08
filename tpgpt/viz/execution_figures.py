"""Figures for the execution-law study (``tpgpt.experiments.sweep_execution``).

Same house style as :mod:`tpgpt.viz.figures` and
:mod:`tpgpt.viz.keypoint_figures`: the ``Agg`` backend chosen at import, the
shared ``_save`` helper, the shared colour constants, and a ``Path`` returned so
a report can link the file it just made.

Every figure here answers one question and says so in its title, because a plot
whose point has to be reconstructed from the axes is the visual form of the
tables that made ``outputs/index.html`` unreadable (``ROBOTICS_NOTES`` 7.26).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tpgpt.viz.figures import _save  # noqa: E402

#: Shared with the keypoint figures and the report overlays so a reader can move
#: between them without relearning the palette.
BASELINE_COLOUR = "#5b8ff9"   # blue -- the shipped law
ANCHOR_COLOUR = "#2ca02c"     # green -- the anchor family
REFERENCE_COLOUR = "#d62728"  # red -- the reference-only family
MUTED = "#999999"

#: Tolerances the numbers are judged against, from measurements elsewhere in the
#: project rather than chosen here.
CLOSING_BUDGET_MM = 15.0      # (aperture - object width) / 2, Panda on this box
APPROACH_TOLERANCE_MM = 120.0  # section 7.2, parallel jaws
DWELL_PRECISION_MM = 4.8      # the demonstration's own lag at the dwell's end


def _law_colour(name: str) -> str:
    if name.startswith("R-"):
        return REFERENCE_COLOUR
    if name.startswith("VR"):
        return ANCHOR_COLOUR
    return BASELINE_COLOUR


def figure_dwell_axes(rows: list[dict], path, title: str = "") -> Path:
    """Why the dwell creep is not the defect it looked like.

    Three panels, all for the identity-transport case (no map at all):

    * **left** -- for each law, the attractor's movement across the grasp dwell
      measured two ways. ``path`` is total distance travelled; ``net`` is how
      far it actually ended up from where it started. They differ by a factor
      of roughly two, which is the attractor wandering and coming back.
    * **middle** -- the net displacement split into the hand's own axes. The
      closing axis is the one that decides whether the object ends up between
      the fingers, and it has roughly 15 mm of room; the approach axis tolerates
      120-135 mm. Almost all of the movement is on the forgiving one.
    * **right** -- the same for the release dwell, which had never been measured.

    Read it as: the bar that matters is the closing-axis one, and it is
    invisible against its own budget.
    """
    ident = [r for r in rows if r.get("case") == "identity" and not r.get("failed")]
    laws = [r["law"] for r in ident]
    x = np.arange(len(laws))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].set_title("Grasp dwell: distance travelled vs distance moved", fontsize=10)
    axes[0].bar(x - 0.2, [r.get("dwell_grasp_path", np.nan) for r in ident], 0.4,
                label="path length (includes jiggle)", color=MUTED)
    axes[0].bar(x + 0.2, [r.get("dwell_grasp_net", np.nan) for r in ident], 0.4,
                label="net displacement", color=BASELINE_COLOUR)
    axes[0].set_ylabel("mm")
    axes[0].legend(fontsize=7)

    for ax, phase, name in ((axes[1], "grasp", "Grasp dwell"),
                            (axes[2], "release", "Release dwell")):
        ax.set_title(f"{name}: net displacement by hand axis", fontsize=10)
        ax.bar(x - 0.2, [r.get(f"dwell_{phase}_closing", np.nan) for r in ident], 0.4,
               label="closing axis (~15 mm budget)", color=REFERENCE_COLOUR)
        ax.bar(x + 0.2, [r.get(f"dwell_{phase}_approach", np.nan) for r in ident], 0.4,
               label="approach axis (120-135 mm)", color=ANCHOR_COLOUR)
        ax.axhline(DWELL_PRECISION_MM, ls="--", lw=0.9, color=MUTED,
                   label="demonstration's own precision, 4.8 mm")
        # Headroom so the reference line and the panel title do not collide.
        ax.set_ylim(0, max(DWELL_PRECISION_MM * 1.35, ax.get_ylim()[1] * 1.15))
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("mm")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(laws, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        title or "The dwell creep is real, and lands almost entirely on the axis "
                 "that tolerates it",
        fontsize=11, y=1.0,
    )
    return _save(fig, path)


def figure_law_vs_conditioning(rows: list[dict], path, title: str = "") -> Path:
    """How each law behaves as the map's deformation gets more aggressive.

    ``min det(J)`` on the x-axis is the smallest Jacobian determinant of the
    transportation map along the trajectory: **1.0 means no deformation and 0
    means the map has folded**, so the axis runs from easy on the right to hard
    on the left. Each faint dot is one warp; the heavy markers are the median
    within a conditioning bin, which is what the statistics are computed on.

    * **left** -- how far the commanded attractor strayed from the path it was
      meant to retrace (integration drift, measured pointwise).
    * **right** -- how far the *arm* ended up from where the demonstrating arm
      actually was, which is what matters for the task.

    Lower is better on both, and the vertical scale is logarithmic because the
    families are an order of magnitude apart. The thing to look at is that the
    shipped law sits at the bottom across the whole range, and that the two
    laws which query the policy at the **measured** position sit an order of
    magnitude above everything else.
    """
    laws = list(dict.fromkeys(r["law"] for r in rows if not r.get("failed")))
    edges = [(0.6, 9.9), (0.35, 0.6), (0.0, 0.35)]
    styles = {}
    for law in laws:
        if law == "V":
            styles[law] = dict(color=BASELINE_COLOUR, marker="o", lw=2.4, zorder=5)
        elif law.startswith("R-"):
            styles[law] = dict(color=REFERENCE_COLOUR, marker="^", lw=1.3,
                               ls="--" if law.endswith("-m") else "-", zorder=3)
        else:
            styles[law] = dict(color=ANCHOR_COLOUR, marker="s", lw=1.3,
                               ls="--" if "-m" in law else "-", zorder=4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, name in (
        (axes[0], "drift_attractor_vs_planned_max", "Attractor drift from its planned path"),
        (axes[1], "arm_error_vs_truth_max", "Arm error against the demonstrating arm"),
    ):
        for law in laws:
            sel = [r for r in rows if r["law"] == law and not r.get("failed")
                   and r.get("case") != "identity"]
            if not sel:
                continue
            st = dict(styles[law])
            ax.scatter([r["min_det"] for r in sel], [r[key] for r in sel],
                       s=7, alpha=0.22, color=st["color"], zorder=1)
            xs, ys = [], []
            for lo, hi in edges:
                b = [r[key] for r in sel if lo <= r["min_det"] < hi]
                if b:
                    xs.append(np.median([r["min_det"] for r in sel
                                         if lo <= r["min_det"] < hi]))
                    ys.append(np.median(b))
            ax.plot(xs, ys, ms=6, label=law, **st)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("min det(J)   (1.0 = undeformed, 0 = folded)")
        ax.set_ylabel("mm, worst over the run")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(
        title or "The shipped law is the best law: every alternative is equal or "
                 "worse, and querying at the measured pose is far worse",
        fontsize=11, y=1.0,
    )
    return _save(fig, path)


def figure_lag_model(path, tau: float = 0.0962, dt: float = 0.05,
                     speed: float = 0.168, title: str = "") -> Path:
    """Why the arm lags more than the gate's own model of it expects.

    The lag gate subtracts ``expected = ||K^-1 D|| v`` before deciding the arm
    is behind. But the attractor is held fixed for a whole control step -- a
    zero-order hold, because the robot holds one action while the simulator
    integrates internally -- so the true settled lag is

        ``L* = v dt / (1 - (1 - A dt / m)^m)``,  ``A = D^-1 K``

    which at ``m = 1`` is exactly the gate's model and rises to
    ``v dt / (1 - exp(-dt/tau))`` as the sub-stepping gets finer. **The real
    plant is the fine limit**, so the gate carries a standing excess before
    anything has gone wrong: the gap between the two dashed lines.

    Left: the lag against sub-stepping. Right: the ratio of true lag to the
    gate's model, against the control period -- showing the effect vanishes
    only as ``dt`` shrinks well below the impedance time constant.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    A = 1.0 / tau
    m = np.arange(1, 65)
    lag = speed * dt / (1 - (1 - A * dt / m) ** m)
    axes[0].plot(m, lag * 1000, color=BASELINE_COLOUR, lw=1.6,
                 label="settled lag, zero-order hold")
    axes[0].axhline(tau * speed * 1000, ls="--", color=MUTED, lw=1.0,
                    label=r"gate's model, $\|K^{-1}D\|v$")
    axes[0].axhline(speed * dt / (1 - np.exp(-dt / tau)) * 1000, ls="--",
                    color=REFERENCE_COLOUR, lw=1.0, label="real plant (fine limit)")
    axes[0].set_xlabel("Euler sub-steps per control step, m")
    axes[0].set_ylabel("settled lag (mm)")
    axes[0].set_title("The bed's arm, as sub-stepping is refined", fontsize=10)
    axes[0].legend(fontsize=7)

    dts = np.linspace(0.005, 0.10, 200)
    ratio = (dts / tau) / (1 - np.exp(-dts / tau))
    axes[1].plot(dts * 1000, ratio, color=REFERENCE_COLOUR, lw=1.6)
    axes[1].axvline(dt * 1000, ls=":", color=MUTED)
    axes[1].annotate(f"this project\n{dt*1000:.0f} ms  ->  {(dt/tau)/(1-np.exp(-dt/tau)):.2f}x",
                     xy=(dt * 1000, (dt / tau) / (1 - np.exp(-dt / tau))),
                     xytext=(dt * 1000 - 34, 1.18), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    axes[1].set_xlabel("control period dt (ms)")
    axes[1].set_ylabel("true lag / gate's model")
    axes[1].set_title(r"The gap closes only when $dt \ll \tau$", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        title or "The lag gate's model of the lag is 1.28x optimistic at this control rate",
        fontsize=11, y=1.0,
    )
    return _save(fig, path)


def figure_quiet_vs_loaded(quiet: list[dict], loaded: list[dict], path,
                           title: str = "") -> Path:
    """Why the law answer depends on whether the arm is being held back.

    Each panel is one conditioning bin. Bars are the **paired** difference in
    attractor drift against the shipped law, median over the warps in that bin:
    **below zero means better than what ships, above means worse.** Blue is the
    undisturbed arm, orange is the same arm under a 0.08 m/s load it cannot
    overcome — which shuts the lag gate on 42% of steps instead of 2.5%.

    The reference-only laws are omitted; they lose by 3–60 mm in both conditions
    and would flatten the scale. The dashed line is 4.8 mm, the demonstration's
    own precision at the end of its dwell, and it is the reason the conclusion
    is "have the switch" rather than "flip the default": every bar that changes
    sign does so well inside that band.
    """
    import numpy as np

    bins = [(0.6, 9.9, "well conditioned"), (0.35, 0.6, "moderate"),
            (0.0, 0.35, "aggressive")]
    shown = [law for law in dict.fromkeys(r["law"] for r in quiet)
             if law.startswith("VR-a") or law.startswith("VR-sched")]
    key = "drift_attractor_vs_planned_max"

    def paired(rows, law, lo, hi):
        ok = [r for r in rows if not r.get("failed") and r.get("case") != "identity"]
        base = {r["case"]: r[key] for r in ok if r["law"] == "V"}
        vals = {r["case"]: r[key] for r in ok if r["law"] == law}
        d = [vals[c] - base[c] for c in base
             if c in vals and any(lo <= r["min_det"] < hi
                                  for r in ok if r["case"] == c)]
        return float(np.median(d)) if d else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(shown))
    for ax, (lo, hi, name) in zip(axes, bins):
        ax.bar(x - 0.2, [paired(quiet, law, lo, hi) for law in shown], 0.4,
               label="undisturbed (gate shut 2.5%)", color=BASELINE_COLOUR)
        ax.bar(x + 0.2, [paired(loaded, law, lo, hi) for law in shown], 0.4,
               label="loaded (gate shut 42%)", color="#ff7f0e")
        ax.axhline(0.0, color="#333333", lw=1.0)
        for s in (+DWELL_PRECISION_MM, -DWELL_PRECISION_MM):
            ax.axhline(s, ls="--", lw=0.8, color=MUTED)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(shown, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("paired drift difference vs the shipped law (mm)\n"
                       "below 0 = better", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[2].annotate("+-4.8 mm: the demonstration's own precision",
                     xy=(0.02, 0.93), xycoords="axes fraction", fontsize=7,
                     color=MUTED)
    fig.suptitle(
        title or "The anchor loses when the arm tracks freely and wins when the "
                 "lag gate engages -- by less than the demonstration's own precision",
        fontsize=11, y=1.0,
    )
    return _save(fig, path)
