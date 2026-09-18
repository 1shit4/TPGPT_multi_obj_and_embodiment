"""Author the sweep-volume box GraspGen-X conditions a new gripper on.

GraspGen-X (arXiv:2606.00998) plans grasps for a hand it has never seen by
conditioning on a *description* of that hand rather than on trained weights.
Thirteen numbers carry the description -- two boxes of three extents and three
offsets each, plus the fingertip depth -- and `graspgenx/x_grippers.py` builds
its conditioning vector from exactly those. Everything else a shipped gripper
directory contains (point clouds, TSDF grids, a PointNet-VAE embedding) is
unused by the released `sweep_volume_v2` checkpoint, which is why
`make_sweep_volume_gripper_info` fills them with zeros.

So authoring a description is a geometry problem: **where is the pocket the
object goes into, and how big is it.** GraspGen-X asks a person, through the
six-panel GUI in `scripts/gripper_config_wizard.py`. This module computes it.

Why not just use the wizard's own initial guess
----------------------------------------------
That guess is `estimate_inner_sweep_volume`, and measured against the 26
descriptions NVIDIA's curators actually shipped -- running their estimator on
their own URDFs -- it does not reproduce them: median aperture error 19.9 mm
with only 11 of 26 within 10 mm, the depth systematically **2.67x** too large,
and the offset out by a median 26.1 mm. It degenerates outright on two hands
(`bd_spot` finds one moving link and reports a 0 mm aperture; `wuji_hand` picks
the wrong closing axis). Split by family it is 9.5 mm median on two-finger jaws
and **58.8 mm** on the multi-finger hands. The wizard is interactive because
that guess is a starting point for a human, not an answer.

The two corrections this module makes
-------------------------------------
Both come from measuring what a curated box *is*, over all 26 hands:

1. **The box stops at the fingertips.** `offset[2] + extents[2] / 2` against the
   finger geometry's own maximum reach along the approach axis has a median
   difference of **-1.5 mm**, and 21 of 26 hands sit within 8.5 mm of zero.
   The wizard's guess instead spans the whole finger, which is the entire
   source of the 2.67x depth inflation.

2. **The pocket is where the fingers face each other**, not the full span of
   the finger links. Sliced along the approach axis, the pocket is the run of
   slices -- ending at the fingertips -- in which the two sides are separated
   by a real gap. Below that run the fingers meet the palm and there is no
   pocket to describe.

What this module does not claim
-------------------------------
`extents2`/`offset2` (the half-closed box) are not derived here. The wizard does
not derive them either: it seeds them as `[extents[0] * 0.5, extents[1],
extents[2]]` with the offset unchanged, and 17 of 26 curated hands have an
`offset2` a person then moved -- forward for jaws, backward for hands that curl.
Pass the half-closed geometry to :func:`estimate_sweep_box` a second time to
measure it instead of assuming it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_sweep_box", "closing_axis_of", "SweepBox"]

#: How many slices the finger geometry is cut into along the approach axis.
#: 40 puts a slice every 2-4 mm on the hands in the curated set -- fine enough
#: to find where the pocket ends, coarse enough that each slice holds points.
N_SLICES = 40

#: How many consecutive unusable slices the pocket may bridge before it is
#: judged to have ended. Fingertips taper, so the slice at the very tip
#: often holds points from only one side.
MAX_SKIP = 3

#: How far a slice's gap may sit from the gap at the fingertips and still
#: count as the same pocket, as a fraction of it.
GAP_TOL = 0.25

#: A slice counts as part of the pocket if the two sides are separated by at
#: least this much. Not zero: two fingers whose bounding geometry just touches
#: enclose nothing, and a tolerance of exactly zero lets one stray vertex end
#: the pocket several centimetres early.
MIN_GAP = 0.002


class SweepBox(tuple):
    """``(extents, offset, closing_axis, diagnostics)``."""

    @property
    def extents(self):
        return self[0]

    @property
    def offset(self):
        return self[1]

    @property
    def closing_axis(self):
        return self[2]

    @property
    def diagnostics(self):
        return self[3]


def closing_axis_of(fingers):
    """The axis the fingers are most spread along.

    GraspGen-X's own rule. In a frame that is already canonical this must come
    out 0, and if it does not, the frame and the geometry disagree about the
    hand -- which is worth surfacing rather than silently accepting, because it
    is how `wuji_hand` gets described along the wrong axis.
    """
    centroids = np.array([f.mean(axis=0) for f in fingers])
    spread = centroids.max(axis=0) - centroids.min(axis=0)
    return int(np.argmax(spread))


def _sides(fingers, axis):
    """Split the fingers into the two opposing groups.

    Sided about the midpoint of the two **extreme** fingers rather than about
    the origin. A hand whose fingers curl does not straddle its own origin --
    splitting on the sign of the coordinate put every pad of the Inspire,
    SchunkSvh and Ability hands into one group and left the other empty, which
    reports a hand that cannot open.

    Every finger is assigned, not just the extreme two: the thumb opposes
    *four* fingers on an anthropomorphic hand, and the pocket's width across
    the non-closing axis is the span of all four, not of whichever one happens
    to sit furthest out.
    """
    centroids = np.array([f.mean(axis=0)[axis] for f in fingers])
    lo, hi = centroids.min(), centroids.max()
    mid = (lo + hi) / 2.0
    near = [f for f, c in zip(fingers, centroids) if c <= mid]
    far = [f for f, c in zip(fingers, centroids) if c > mid]
    return near, far


def estimate_sweep_box(fingers, closing_axis=None, n_slices=N_SLICES,
                       min_gap=MIN_GAP, approach_axis=2, max_skip=MAX_SKIP,
                       gap_tol=GAP_TOL):
    """Fit the pocket between a hand's fingers.

    Parameters
    ----------
    fingers:
        One point array ``(N_i, 3)`` per finger link, in the **canonical**
        frame -- ``+Z`` along the approach, ``+X`` along the closing direction.
        Surface points, not centroids: the pocket's width across the closing
        axis is a gap between two *surfaces*, and a box fitted to geom centres
        comes out zero-thickness in the perpendicular direction.
    closing_axis:
        Force the axis rather than deriving it. Pass 0 when the frame is known
        to be canonical; leave ``None`` to derive and check.

    Returns
    -------
    SweepBox
        ``extents`` and ``offset`` ready for ``config["sweep_volume"]``, the
        closing axis actually used, and a diagnostics dict -- which carries
        ``pocket_slices`` and ``tip``, because an estimate resting on two
        slices is not the same claim as one resting on twenty and the caller
        should be able to tell.
    """
    fingers = [np.asarray(f, dtype=float) for f in fingers if len(f)]
    if len(fingers) < 2:
        raise ValueError(
            f"need at least two finger geometries to find a pocket, got "
            f"{len(fingers)}. GraspGen-X's own estimator does not check this "
            f"and reports a 0 mm aperture for bd_spot because of it."
        )

    derived = closing_axis_of(fingers)
    axis = derived if closing_axis is None else int(closing_axis)
    near, far = _sides(fingers, axis)
    if not near or not far:
        raise ValueError("every finger fell on one side of the closing axis")

    near_p, far_p = np.vstack(near), np.vstack(far)
    allp = np.vstack(fingers)

    tip = float(allp[:, approach_axis].max())
    root = float(allp[:, approach_axis].min())
    edges = np.linspace(root, tip, n_slices + 1)

    # Slice along the approach axis and record where the two sides are
    # genuinely separated. A slice can fail for two different reasons and they
    # must not be treated alike: it can be *empty* on one side, which happens
    # at a tapered fingertip and means nothing, or it can be *closed*, which is
    # where the pocket really ends. Breaking on the first failure of either
    # kind ended the pocket after a single slice on seven of the 26 curated
    # hands and put their depth out by 33 to 94 mm.
    gap_at = {}
    for i in range(n_slices):
        z0, z1 = edges[i], edges[i + 1]
        n_in = near_p[(near_p[:, approach_axis] >= z0) & (near_p[:, approach_axis] < z1)]
        f_in = far_p[(far_p[:, approach_axis] >= z0) & (far_p[:, approach_axis] < z1)]
        if len(n_in) == 0 or len(f_in) == 0:
            continue
        gap = float(f_in[:, axis].min() - n_in[:, axis].max())
        if gap >= min_gap:
            gap_at[i] = gap

    if not gap_at:
        raise ValueError(
            "no slice along the approach axis has the two finger groups "
            "separated -- the hand encloses no pocket at this pose"
        )

    # Walk down from the topmost separated slice. Two things end the pocket,
    # and using only the first of them is what made every earlier attempt fail
    # in one direction or the other:
    #
    # * the fingers **meet** -- no gap at all. Stopping only on this ran the
    #   pocket the whole length of the finger on hands whose linkage stays open
    #   behind the pads: the OnRobot RG2's box spanned 39 of 40 slices and its
    #   aperture came out 75 mm short, because the median was taken over a
    #   region most of which is not a pocket.
    # * the gap stops being **the aperture**. The pocket is the prismatic space
    #   between the pads; where a finger flares into its knuckle or its palm
    #   the gap changes and that space has ended. This is the criterion that
    #   matters, and it is relative because the quantity it is judging is a
    #   fraction of an aperture that runs from 48 to 160 mm across this set.
    #
    # Tolerating a short run of unusable slices on top of that covers a tapered
    # fingertip, where a slice legitimately holds points from only one side.
    top = max(gap_at)
    reference_gap = float(np.median(
        [gap_at[i] for i in sorted(gap_at, reverse=True)[:max(1, n_slices // 20)]]
    ))
    lo_ok, hi_ok = reference_gap * (1.0 - gap_tol), reference_gap * (1.0 + gap_tol)

    run, misses = [top], 0
    for i in range(top - 1, -1, -1):
        gap = gap_at.get(i)
        if gap is not None and lo_ok <= gap <= hi_ok:
            run.append(i)
            misses = 0
        else:
            misses += 1
            if misses > max_skip:
                break
    gaps = [gap_at[i] for i in run]

    # The top of the box is the fingertips, not the top of the deepest slice
    # that happened to register. Measured over all 26 curated descriptions,
    # ``offset[2] + extents[2] / 2`` sits a median **-1.5 mm** from the finger
    # geometry's own maximum reach, with 21 of 26 inside 8.5 mm -- so this is
    # the curators' own convention, and anchoring here removes a whole slice
    # of quantisation error from both the depth and the offset.
    z_hi = tip
    z_lo = float(edges[min(run)])

    other = [a for a in (0, 1, 2) if a not in (axis, approach_axis)][0]
    in_pocket = allp[(allp[:, approach_axis] >= z_lo)
                     & (allp[:, approach_axis] <= z_hi)]

    extents = np.zeros(3)
    offset = np.zeros(3)
    # The aperture is the pocket's **narrowest** slice. An object has to pass
    # through every slice to reach the pocket, so the narrowest one is the
    # width the hand can actually admit -- the others describe places the
    # object cannot get to.
    #
    # Measured over the 26 curated descriptions, against the alternatives:
    # median 4.8 mm / 51.3 mm (two-finger / multi-finger error), mean 6.0 /
    # 50.9, 25th percentile 2.4 / 43.2, tenth percentile 2.0 / 36.2, and the
    # minimum **2.5 / 31.0**. It barely matters on a parallel jaw, where every
    # slice of a prismatic pocket reads the same, and it is the difference
    # between a usable description and a useless one on a hand whose fingers
    # splay at the open pose: the Barrett hand's fingertips are 279 mm apart
    # and it is declared at 160.
    #
    # This statistic was chosen **on** these 26 hands, so the figures above are
    # in-sample. The out-of-sample test is execution, not a lower number here.
    extents[axis] = float(np.min(gaps))
    extents[other] = float(np.ptp(in_pocket[:, other])) if len(in_pocket) else 0.0
    extents[approach_axis] = float(z_hi - z_lo)
    # x and y are pinned to the approach axis: all 26 curated descriptions have
    # ``offset = [0, 0, z]`` exactly, so the box is centred on the approach axis
    # by convention and a model whose geoms sit a millimetre off-centre should
    # not be described as gripping off-axis.
    offset[approach_axis] = float((z_lo + z_hi) / 2.0)

    return SweepBox((extents, offset, axis, {
        "pocket_slices": len(gaps),
        "slices_spanned": top - min(run) + 1,
        "tip": tip,
        "derived_closing_axis": derived,
        "closing_axis_forced": closing_axis is not None and derived != axis,
        "reference_gap": reference_gap,
        "_gaps": list(map(float, gaps)),
        "gap_min": float(np.min(gaps)),
        "gap_max": float(np.max(gaps)),
        "n_fingers": len(fingers),
    }))
