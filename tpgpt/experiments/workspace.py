"""Where can the arm actually work, and where can it only reach?

There are two workspaces and the difference is the whole point of this
measurement. The **reachable** workspace is every point the end-effector can be
put. The **dexterous** workspace is every point it can be put *while holding a
particular hand orientation* -- and for a task built on top-down grasps, only
the second one matters.

They are not close. Measured on the Panda in this arena, a point can be
comfortably reachable and completely unusable: the arm gets its fingertips
there and cannot turn the hand to point down, because the elbow would have to
fold backwards.

That distinction was invisible until an end-to-end run stalled. Every sampled
point of a warped trajectory was reachable in position; three of twenty were
reachable with the orientation the task needed. The run reported a stalled
policy, which blamed the transport for a property of the arm.

Run as ``python -m tpgpt.experiments.workspace``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: Hand pointing straight down, which is what every grasp here assumes.
TOP_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

#: Matched to the controller's own tracking, not to solver precision -- see
#: ``tpgpt.grasp.filters.REACH_TOLERANCE``.
TOLERANCE = 0.015


def measure(env, x_range=(-0.30, 0.35), z_range=(0.82, 1.50), y=0.0, n=26) -> dict:
    """Sample a vertical slice of the workspace, with and without orientation.

    ``y`` is the lateral offset; the slice through ``y = 0`` is the one the
    shelf slots sit on.
    """
    from tpgpt.sim.kinematics import solve_ik

    xs = np.linspace(*x_range, n)
    zs = np.linspace(*z_range, n)
    reachable = np.zeros((len(zs), len(xs)), dtype=bool)
    dexterous = np.zeros_like(reachable)

    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            point = np.array([x, y, z])
            reachable[i, j] = solve_ik(
                env, point, None, position_tolerance=TOLERANCE
            ).reachable
            dexterous[i, j] = solve_ik(
                env, point, TOP_DOWN, position_tolerance=TOLERANCE
            ).reachable
    return {"xs": xs, "zs": zs, "y": y, "reachable": reachable, "dexterous": dexterous}


def slot_verdicts(env, clearance: float = 0.16) -> list[dict]:
    """For each shelf slot: can the arm place there, and how much headroom.

    ``clearance`` is how far above the slot the hand must also be able to hold
    a top-down pose, since a placement is approached from above and retreats the
    same way. A slot whose surface is dexterous but whose approach is not cannot
    be used.
    """
    from tpgpt.sim.kinematics import solve_ik

    verdicts = []
    for name, position in env.slot_poses().items():
        heights = []
        for lift in np.arange(0.0, clearance + 0.001, 0.02):
            point = position + np.array([0.0, 0.0, lift])
            if solve_ik(env, point, TOP_DOWN, position_tolerance=TOLERANCE).reachable:
                heights.append(float(lift))
            else:
                break
        verdicts.append({
            "slot": name,
            "surface_ok": bool(heights),
            "headroom": max(heights) if heights else 0.0,
            "needed": clearance,
            "usable": bool(heights) and max(heights) >= clearance - 1e-9,
        })
    return verdicts


def figure(slice_data: dict, verdicts: list[dict], env, path) -> Path:
    """The two workspaces, with the shelf slots drawn on top."""
    xs, zs = slice_data["xs"], slice_data["zs"]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    extent = [xs[0], xs[-1], zs[0], zs[-1]]

    ax.imshow(slice_data["reachable"], origin="lower", extent=extent, aspect="auto",
              cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    ax.imshow(np.where(slice_data["dexterous"], 1.0, np.nan), origin="lower",
              extent=extent, aspect="auto", cmap="Greens", alpha=0.55, vmin=0, vmax=1)

    for verdict in verdicts:
        position = env.slot_poses()[verdict["slot"]]
        if abs(position[1] - slice_data["y"]) > 0.02:
            continue
        colour = "#137a3f" if verdict["usable"] else "#b3261e"
        ax.plot([position[0]], [position[2]], "o", color=colour, ms=9,
                markeredgecolor="k", markeredgewidth=0.6)
        ax.annotate(
            f"{verdict['slot']}\n{verdict['headroom'] * 100:.0f} cm headroom",
            (position[0], position[2]), textcoords="offset points",
            xytext=(8, 6), fontsize=8, color=colour,
        )

    ax.plot([], [], "s", color="#9ecae1", ms=10, label="reachable (any hand pose)")
    ax.plot([], [], "s", color="#a1d99b", ms=10, label="dexterous (hand pointing down)")
    ax.set_xlabel("x, metres from the robot base")
    ax.set_ylabel("height, metres")
    ax.set_title(
        f"Panda workspace, slice at y = {slice_data['y']:.2f} m\n"
        "green is where a top-down grasp is possible", fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(out_dir="outputs/workspace", shelf_variant: str = "cubby") -> dict:
    from tpgpt.reporting.html import write_manifest
    from tpgpt.sim.scenes.tabletop_shelf import TabletopShelf

    from robosuite.controllers import load_composite_controller_config

    from tpgpt.sim.controllers.cartesian_impedance import make_torque_controller_config

    out_dir = Path(out_dir)
    config = make_torque_controller_config(
        load_composite_controller_config(controller="BASIC", robot="Panda")
    )
    env = TabletopShelf(
        robots="Panda", controller_configs=config, objects=("can",),
        shelf_variant=shelf_variant, control_freq=20, seed=0,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    try:
        env.reset()
        slice_data = measure(env)
        verdicts = slot_verdicts(env)
        path = figure(slice_data, verdicts, env, out_dir / "workspace.png")

        dexterous = slice_data["dexterous"].mean()
        reachable = slice_data["reachable"].mean()
        print(f"{'slot':<16}{'surface':>9}{'headroom':>10}{'usable':>8}")
        for verdict in verdicts:
            print(f"{verdict['slot']:<16}{str(verdict['surface_ok']):>9}"
                  f"{verdict['headroom'] * 100:9.0f}cm{str(verdict['usable']):>8}")
        print(f"\nslice at y=0: {reachable:.0%} reachable, {dexterous:.0%} dexterous")
        print(f"wrote {path}")

        summary = {
            "slots": verdicts,
            "reachable_fraction": float(reachable),
            "dexterous_fraction": float(dexterous),
        }
        (out_dir / "workspace.json").write_text(json.dumps(summary, indent=2))
        write_manifest(
            out_dir,
            title="Panda workspace",
            description="Where the arm can reach, and where it can reach with the "
                        "hand pointing down. Only the second is usable for a "
                        "top-down grasp, and the two differ a great deal.",
            settings={"shelf": shelf_variant, "tolerance_mm": TOLERANCE * 1000},
            report="workspace.png",
        )
        return summary
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/workspace")
    parser.add_argument("--shelf", default="cubby")
    main(parser.parse_args().out, parser.parse_args().shelf)
