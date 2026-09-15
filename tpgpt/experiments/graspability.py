"""Which (gripper, object) pairs admit a grasp at all, decided before the run.

**A pair with no feasible grasp is outside the method's domain, not a failure of
it.** If an object is wider than a hand's jaws in every direction, no grasp
exists, so there is no anchor, so the transportation map is never fitted. That is
categorically different from a map that was fitted and a plan that failed, and
pooling the two understates the method and misattributes the cause. So the grid
is defined first, from geometry, and the excluded pairs are published with their
reasons rather than quietly counted as failures.

**This is the one place true geometry is allowed**, and the distinction must not
blur. Deciding which cells to run is a statement about the experimental design,
and using the simulator's own meshes for it is sound. What must never use true
geometry is anything *inside* the method: the grasp screen in
:mod:`tpgpt.grasp.filters` sees only the observed point cloud, and it stays that
way.

Why not compare bounding boxes
------------------------------

Because a bounding box is the wrong measurement twice over, and this project has
made both mistakes:

* **It over-reads a rotated object.** A 30 x 100 mm box yawed 45 degrees has a
  92 x 92 mm axis-aligned box, so comparing that to a jaw aperture calls a
  perfectly graspable object impossible (``ROBOTICS_NOTES`` 7.18).
* **It cannot see a thin feature.** The mug is a tube: 90 mm across the
  outside, which only the two widest hands can span, and a **6 mm wall** any
  hand in the fleet can pinch with one finger inside it. A box test reads 90 mm
  and excludes five hands that can in fact hold it.

So the object is measured the way a jaw actually meets it. Along a candidate
closing direction, rays are cast through the object and every surface crossing
collected, which splits the line into **slabs** of material separated by gaps.
A jaw grips one slab, with a pad on each side of it, so the quantity that has to
fit the aperture is a *slab thickness* -- not the object's extent. The mug's
wall is a 6 mm slab with a 78 mm cavity beside it, which is exactly why a narrow
hand can hold it.

**A slab is one collision geom, not one pair of surface crossings, and the
difference decides four of the five objects.** MuJoCo ray-casts a mesh against
its actual triangles while it *collides* that mesh as its **convex hull**, and
robosuite's grocery meshes are hollow shells. Taken at face value the rays
therefore report a cereal carton as a 35.3 mm near wall with a cavity behind it,
a milk carton as **8.9 mm**, and a sandwich loaf as 35.6 -- all of them
graspable by every hand in the fleet, none of them true, because no finger can
enter a sealed carton and the physics would present a solid box anyway. Grouping
the crossings by the geom they hit fixes it exactly: a convex geom is entered
once and left once, so its interval is its full extent along the ray, and a gap
survives only where there genuinely are two separate collision bodies -- which
is the case for the mug, whose wall robosuite builds from sixteen separate
boxes.

Three requirements make a slab grippable, and each is there because dropping it
admits something false:

``MIN_SLAB``     the slab must be thicker than 3 mm, or the "grasp" is the
                 silhouette edge of any object at all -- every convex body has
                 arbitrarily thin slabs at its outline, so without this the
                 minimum width of a can is zero.
``PAD``          the slab must hold its thickness across a 20 x 20 mm patch, a
                 finger pad. Without this a single grazing ray decides the
                 answer, which is the same defect in a subtler form.
``FINGER_GAP``   whichever side of the slab is not open space must offer at
                 least 12 mm for a finger to enter. This is what distinguishes
                 the mug's wall, with a 78 mm cavity beside it, from a slab in
                 the middle of a solid body that no finger can reach.

The closing axis is taken **horizontal**. Both the source demonstration and the
scene's support geometry constrain the approach to come from above, so a jaw
closes in the horizontal plane; a vertical closing axis would ask the hand to
grip an object between the table and the air above it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Thinnest slab a jaw is credited with being able to hold, in metres.
MIN_SLAB = 0.003

#: Side of the square patch over which a slab must hold its thickness, in
#: metres -- a finger pad.
PAD = 0.020

#: Clearance a finger needs to enter a gap beside a slab, in metres.
FINGER_GAP = 0.012

#: Closing directions tried, as azimuths in the horizontal plane. The closing
#: axis is undirected, so half a turn covers it.
N_DIRECTIONS = 36

#: Ray grid spacing in the plane perpendicular to the closing axis, in metres.
GRID = 0.005

#: Rays are cast from this far outside the object's bounding box.
STANDOFF = 1.0


def _object_model(name: str, world: str):
    """Compile ``name`` on its own, in the orientation it rests in.

    Alone, in an empty world, for a reason: MuJoCo's ``mj_ray`` intersects
    everything *drawn*, and the scene contains a translucent marker box at every
    shelf slot with collision switched off. An unguarded cast in the full scene
    reports an obstruction 58 mm away that the hand passes straight through --
    a plausible number for a surface that is not there (``ROBOTICS_NOTES`` 7.39).
    A model holding one object cannot make that mistake.
    """
    import mujoco
    from robosuite.models.world import MujocoWorldBase

    from tpgpt.sim.objects import make_object, rest_quat

    obj = make_object(name, world, rng=np.random.default_rng(0))
    world_base = MujocoWorldBase()
    world_base.merge_assets(obj)
    body = obj.get_obj()
    quat = rest_quat(name)
    body.set("quat", " ".join(f"{v:.9f}" for v in quat))
    world_base.worldbody.append(body)
    model = world_base.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _crossings(model, data, origin, direction) -> list[tuple[float, int]]:
    """``(distance, geom)`` for every surface crossing along the ray.

    ``mj_ray`` returns only the nearest hit, so the ray is re-cast from just past
    each hit until it leaves the model. Visual duplicates are skipped: robosuite
    builds these assets with ``duplicate_collision_geoms``, so each collision
    geom has a twin that the hand cannot feel, and counting it would halve every
    reported gap.
    """
    import mujoco

    epsilon = 1e-5
    out, travelled, geomid = [], 0.0, np.zeros(1, dtype=np.int32)
    for _ in range(128):
        point = origin + direction * travelled
        distance = mujoco.mj_ray(model, data, point, direction,
                                 None, 1, -1, geomid)
        if distance < 0 or geomid[0] < 0:
            break
        travelled += distance + epsilon
        geom = int(geomid[0])
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            out.append((travelled, geom))
    return out


def _slabs(crossings: list[tuple[float, int]]):
    """``(start, end, gap_before, gap_after)`` for each slab of material.

    One interval per collision geom, from its first crossing to its last, then
    overlapping intervals merged. That is what makes this measure the geometry
    the *hand* meets rather than the geometry the renderer draws: MuJoCo
    collides a mesh as its convex hull, which is entered once and left once, so
    a hollow shell's interior is not a gap a finger can use. See the module
    docstring -- read the other way, the milk carton measures 8.9 mm.
    """
    spans: dict[int, list[float]] = {}
    for distance, geom in crossings:
        spans.setdefault(geom, []).append(distance)
    intervals = sorted((min(v), max(v)) for v in spans.values() if len(v) >= 2)
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    slabs = []
    for i, (start, end) in enumerate(merged):
        before = start - merged[i - 1][1] if i else np.inf
        after = merged[i + 1][0] - end if i + 1 < len(merged) else np.inf
        slabs.append((start, end, before, after))
    return slabs


def minimum_grip_width(name: str, world: str = "real") -> dict:
    """The thinnest slab of ``name`` a jaw could actually close on, in metres.

    Returns the width, the azimuth it was found at, and the per-direction
    minima, so a pair excluded by this screen can be checked rather than taken
    on trust.
    """
    model, data = _object_model(name, world)
    lower = model.stat.center - model.stat.extent
    upper = model.stat.center + model.stat.extent
    centre = (lower + upper) / 2.0
    reach = float(np.max(upper - lower)) / 2.0 + 0.01

    best, best_azimuth, per_direction = np.inf, None, {}
    for azimuth in np.linspace(0.0, np.pi, N_DIRECTIONS, endpoint=False):
        direction = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        across = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        offsets = np.arange(-reach, reach + GRID, GRID)
        # width[i, j] is the thinnest grippable slab on the ray through
        # (across * offsets[i], up * offsets[j]); inf where there is none.
        widths = np.full((len(offsets), len(offsets)), np.inf)
        for i, a in enumerate(offsets):
            for j, b in enumerate(offsets):
                origin = centre + across * a + up * b - direction * STANDOFF
                for start, end, before, after in _slabs(
                        _crossings(model, data, origin, direction)):
                    thickness = end - start
                    if thickness < MIN_SLAB:
                        continue
                    if min(before, after) < FINGER_GAP:
                        continue
                    widths[i, j] = min(widths[i, j], thickness)
        direction_best = _best_over_pad(widths)
        per_direction[float(np.degrees(azimuth))] = direction_best
        if direction_best < best:
            best, best_azimuth = direction_best, float(np.degrees(azimuth))
    return {"object": name, "width_m": float(best), "azimuth_deg": best_azimuth,
            "per_direction": per_direction}


def _best_over_pad(widths: np.ndarray) -> float:
    """Thinnest slab that holds across a finger pad.

    A single ray is not a grasp. The requirement is a ``PAD``-sized square in
    which every ray found a slab no thicker than the answer, which is the
    smallest patch a finger could sit on.
    """
    side = max(1, int(round(PAD / GRID)))
    if widths.shape[0] < side:
        return float(np.min(widths))
    best = np.inf
    for i in range(widths.shape[0] - side + 1):
        for j in range(widths.shape[1] - side + 1):
            window = widths[i:i + side, j:j + side]
            if not np.isfinite(window).all():
                continue
            best = min(best, float(window.max()))
    return best


def matrix(fleet, objects, world: str = "real") -> dict:
    """The graspability matrix: which cells run, and why the rest do not."""
    from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

    widths = {obj: minimum_grip_width(obj, world) for obj in objects}
    rows = []
    for hand in fleet:
        aperture = gripper_geometry(resolve_pair(hand).graspgen).aperture
        for obj in objects:
            width = widths[obj]["width_m"]
            feasible = np.isfinite(width) and width <= aperture
            rows.append({
                "gripper": hand, "object": obj,
                "aperture_mm": round(aperture * 1000, 1),
                "min_grip_width_mm": (round(width * 1000, 1)
                                      if np.isfinite(width) else None),
                "closing_azimuth_deg": widths[obj]["azimuth_deg"],
                "feasible": bool(feasible),
                "reason": "" if feasible else (
                    "no slab of this object is thin enough for any jaw"
                    if not np.isfinite(width) else
                    f"thinnest grippable slab is {width * 1000:.1f} mm against "
                    f"a {aperture * 1000:.1f} mm aperture"),
            })
    return {"world": world, "objects": widths, "cells": rows,
            "thresholds": {"min_slab_m": MIN_SLAB, "pad_m": PAD,
                           "finger_gap_m": FINGER_GAP,
                           "n_directions": N_DIRECTIONS, "grid_m": GRID}}


def main(out: str = "outputs/graspability.json") -> int:
    from tpgpt.experiments.verify_scene import FLEET, OBJECTS

    candidates = tuple(OBJECTS) + ("bottle", "lemon", "bread")
    result = matrix(FLEET, candidates)
    print(f"{'object':9s} {'thinnest grippable slab':>24s}  {'azimuth':>8s}")
    for name, info in result["objects"].items():
        width = info["width_m"]
        shown = f"{width * 1000:.1f} mm" if np.isfinite(width) else "none"
        print(f"{name:9s} {shown:>24s}  {info['azimuth_deg']!s:>8s}")
    print()
    print(f"{'gripper':14s} {'aperture':>9s}  " +
          " ".join(f"{o[:8]:>8s}" for o in candidates))
    by = {(r["gripper"], r["object"]): r for r in result["cells"]}
    for hand in FLEET:
        cells = " ".join(
            f"{'Y' if by[(hand, o)]['feasible'] else '.':>8s}" for o in candidates)
        print(f"{hand:14s} {by[(hand, candidates[0])]['aperture_mm']:7.1f}mm  {cells}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
