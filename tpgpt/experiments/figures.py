"""Regenerate the paper's theory figures (Figs. 2, 4, 5).

Run with::

    MUJOCO_GL=egl python -m tpgpt.experiments.figures --out outputs/figures

No simulator is needed; these are the 2-D illustrations of Sec. III.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tpgpt.viz.figures import (
    figure_space_deformation,
    figure_transported_field,
    figure_uncertainty_fields,
)


def flat_to_curved_surface(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """The paper's running 2-D example: a flat surface bent into a curve.

    Fig. 2 shows exactly this -- a line of source points along a flat surface,
    matched to a curved target -- which is what makes the nonlinear stage
    necessary rather than decorative.
    """
    x = np.linspace(-25.0, 20.0, n)
    source = np.c_[x, np.full(n, -20.0)]
    curve = 22.0 * np.tanh((x + 5.0) / 12.0)
    target = np.c_[x * 0.55 + 10.0, curve - 8.0]
    return source, target


def cyclic_demonstration(n: int = 160) -> np.ndarray:
    """A cyclic approach-and-retreat demonstration, as in Figs. 4 and 5."""
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[18.0 * np.cos(t) - 5.0, 22.0 * np.sin(t) + 5.0]


def main(out_dir: str | Path = "outputs/figures") -> list[Path]:
    out_dir = Path(out_dir)
    source, target = flat_to_curved_surface()
    demonstration = cyclic_demonstration()

    paths = [
        figure_space_deformation(source, target, out_dir / "fig2_space_deformation.png"),
        figure_transported_field(
            demonstration, source, target, None, out_dir / "fig4_transported_field.png"
        ),
        figure_uncertainty_fields(
            demonstration, source, target, out_dir / "fig5_uncertainty_fields.png"
        ),
    ]
    for path in paths:
        print(f"  wrote {path}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/figures")
    main(parser.parse_args().out)
