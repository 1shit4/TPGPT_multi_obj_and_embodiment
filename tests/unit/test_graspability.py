"""The screen that decides which (gripper, object) cells are inside the domain."""

import numpy as np

from tpgpt.experiments.graspability import _best_over_pad, _slabs


def test_a_single_convex_geom_is_one_slab_however_often_the_ray_crosses_it():
    """The defect this screen was built wrong with, stated as a test.

    MuJoCo ray-casts a mesh against its triangles while it *collides* that mesh
    as its convex hull, and robosuite's grocery meshes are hollow shells. Read
    crossing by crossing, a sealed milk carton therefore looks like a thin near
    wall with a cavity behind it -- it measured **8.9 mm**, graspable by every
    hand in the fleet, and no finger can enter a sealed carton. Grouping the
    crossings by the geom they hit is what makes the answer the object's true
    70 mm.
    """
    shell = [(0.100, 3), (0.106, 3), (0.164, 3), (0.170, 3)]
    slabs = _slabs(shell)
    assert len(slabs) == 1
    start, end, before, after = slabs[0]
    assert end - start == np.float64(0.07).round(9) or abs((end - start) - 0.07) < 1e-9
    assert before == np.inf and after == np.inf


def test_two_separate_geoms_leave_a_gap_a_finger_can_use():
    """The mug: robosuite builds its wall from sixteen separate boxes.

    So the cavity between the near wall and the far one is a gap the *physics*
    has too, unlike the shell above, and a narrow jaw can pinch one wall with a
    finger inside the tube.
    """
    tube = [(0.100, 3), (0.106, 3), (0.182, 11), (0.188, 11)]
    slabs = _slabs(tube)
    assert len(slabs) == 2
    assert abs((slabs[0][1] - slabs[0][0]) - 0.006) < 1e-9
    assert abs(slabs[0][3] - 0.076) < 1e-9, "the cavity beside the near wall"
    assert slabs[0][2] == np.inf, "open space in front of it"


def test_a_ray_that_only_grazes_a_geom_yields_no_slab():
    """One crossing is a tangent, not a thickness."""
    assert _slabs([(0.1, 3)]) == []
    assert _slabs([]) == []


def test_a_slab_must_hold_its_thickness_across_a_finger_pad():
    """Every convex body has arbitrarily thin slabs at its silhouette.

    Without the pad requirement the minimum width of a can is zero, measured at
    the one grazing ray at its edge, and the screen admits everything.
    """
    widths = np.full((10, 10), 0.060)
    widths[0, 0] = 0.001          # a single grazing ray
    assert _best_over_pad(widths) == 0.060


def test_a_patch_with_a_hole_in_it_is_not_a_pad():
    """A finger cannot sit on a patch part of which is empty space."""
    widths = np.full((10, 10), 0.060)
    widths[5, 5] = np.inf
    # Windows away from the hole still qualify, so the answer is finite...
    assert np.isfinite(_best_over_pad(widths))
    # ...but a grid that is *all* hole has no answer at all.
    assert not np.isfinite(_best_over_pad(np.full((10, 10), np.inf)))
