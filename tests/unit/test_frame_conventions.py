"""One coordinate convention, fixed once, used from end to end.

Three frames are in play and two of them used to be conflated:

* the **grasp convention** -- GraspGen-X's, ``+Z`` approach and ``+X`` closing.
  It is the same on all nine hands, so it is where geometry that must hold
  across embodiments belongs, and it is what every keypoint cube is built in;
* the **wrist convention** -- each gripper model's own ``grip_site``, whose
  orientation relative to the fingers was chosen by whoever authored that model.
  Inverse kinematics aims this one and the controller commands it. It differs
  from the grasp convention by 0.2 degrees on a Robotiq 2F-85 and by 90 on an
  XArm;
* the **task frame** -- support normal plus closing axis, used to fit a box to a
  cloud and to snap it onto a surface.

Two defects came out of confusing them, and this file pins both fixes.

**Defect A** (section 7.33). ``task_frame`` re-derived the closing axis's sign
from a world-axis test rather than taking it from the grasp. Run independently
at four blocks it answered differently at the source's pick and its place -- the
demonstration sets its object square with the shelf, so the placed closing axis
landed on world ``x`` at ``[1, -1e-17, 0]`` and the tie-break decided -- leaving
the source's own two frames **180 degrees apart**. That reflects every carried
object through its grasp point.

**Defect B** (section 7.34). The transported plan was computed in the grasp
convention and handed to inverse kinematics as though it were a wrist pose, with
no ``alignment_rotation`` anywhere in the path. The commanded orientation came
out 0.6 to 91 degrees from what ``grasp_to_eef_pose`` gives for the same grasp.

Neither showed up in any existing measure. The map stayed a valid
diffeomorphism, the keypoint residual stayed at 1e-7, and the aim stayed at
**0.00 mm** -- a half turn about the approach leaves the grasp point fixed,
being the one point a reflection cannot move.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.grasp.grasps import (
    alignment_rotation,
    to_grasp_convention,
    to_wrist_convention,
)
from tpgpt.metrics.transport import JAW_SYMMETRY, carry_orientation_error
from tpgpt.sim.keypoints import (
    GraspFrame,
    ObjectPlacement,
    apply_transform,
    carry_indices,
    carry_transform,
    scene_keypoints,
    task_frame,
    top_down_grasp,
)
from tpgpt.transport.maps import TransportMap

from tests.unit.test_keypoint_extraction import (
    HALF,
    TABLE,
    box_cloud,
    demo_labels,
    yaw_matrix,
)

#: Yaw of the source grasp, in radians.
#:
#: ``-0.62`` is the reshelving demonstration's own value to three decimals, and
#: it is half of what triggered Defect A: the closing axis comes out at
#: ``[0.814, -0.581, 0]``, whose ``c[1]`` is unambiguously negative, so the old
#: world-axis test flipped it. The existing scene builder grasps at ``+0.4``,
#: where ``c[1] = +0.389``, nothing flips, and the defect is invisible.
SOURCE_YAW = -0.62

#: Yaw the demonstration leaves the object at: square with the shelf.
#:
#: The other half of the trigger. Placing square puts the placed closing axis on
#: world ``x`` exactly, so ``c[1]`` is a rounding artefact near ``1e-17`` --
#: below the epsilon -- and the old tie-break on ``c[0]`` decided instead, the
#: opposite way to the pick.
PLACE_YAW = 0.0

ang = lambda a, b: float(  # noqa: E731
    np.degrees(np.arccos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2, -1, 1)))
)


def scene(target_yaw: float = -1.1, grasp_offset: float = 0.025):
    """Source and target for one carry, at the yaws that trigger Defect A.

    Built here rather than borrowed from ``TestSceneKeypoints``: that builder
    grasps at ``+0.4``, where nothing flips, so every assertion below would pass
    on a scene that cannot fail.

    ``grasp_offset`` puts the target grasp off the object's centre along its
    closing axis. A half turn about the approach reflects the object *through
    the grasp point*, so a centred grasp is reflected onto itself and the whole
    effect vanishes. Real GraspGen-X candidates sit 2 to 50 mm off centre.
    """
    shelf_source, shelf_target = 0.95, 1.02
    source_centre = np.array([-0.10, 0.05, TABLE + HALF[2]])
    source_dest = np.array([0.20, -0.13, shelf_source])
    source_grasp = GraspFrame(
        tcp=source_centre + [0, 0, HALF[2]],
        approach=[0, 0, -1],
        closing=yaw_matrix(SOURCE_YAW)[:, 0],
    )
    labels = demo_labels(
        source_grasp.tcp,
        [source_dest[0], source_dest[1], shelf_source + 2 * HALF[2]],
        grasp_yaw=SOURCE_YAW,
        place_yaw=PLACE_YAW,
    )
    source = ObjectPlacement(
        box_cloud(source_centre, angle=SOURCE_YAW, faces="visible"),
        source_grasp, TABLE, source_dest, shelf_source,
    )
    target_half = np.array([0.03, 0.05, 0.075])
    target_centre = np.array([-0.16, -0.06, TABLE + target_half[2]])
    target_points = box_cloud(
        target_centre, angle=target_yaw, half=target_half, seed=3, faces="visible"
    )
    centred = top_down_grasp(target_points, height_fraction=1.0)
    target = ObjectPlacement(
        target_points,
        GraspFrame(
            tcp=centred.tcp + grasp_offset * np.asarray(centred.closing, float),
            approach=centred.approach,
            closing=centred.closing,
        ),
        TABLE, np.array([0.20, 0.13, shelf_target]), shelf_target,
    )
    return source, target, labels


def kabsch(before: np.ndarray, after: np.ndarray):
    """The rigid motion taking one cloud onto the other, point for point."""
    a, b = before - before.mean(0), after - after.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = (u @ vt).T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = (u @ vt).T
    return r, after.mean(0) - r @ before.mean(0)


class TestTheTriggerIsStillPresent:
    """Guard the guard.

    Without these, a later change to ``demo_labels`` or to the scene could move
    it off the boundary and leave every test below passing on a case that was
    never at risk. Section 7.27's rule: assert the instrument reads something
    before believing what it says.
    """

    def test_the_pick_axis_has_an_unambiguously_negative_y(self):
        source, _, _ = scene()
        assert np.asarray(source.grasp.closing)[1] < -0.1

    def test_the_placed_axis_lands_on_the_old_tie_break(self):
        source, _, labels = scene()
        rotation, translation = carry_transform(labels)
        placed = source.grasp.transformed(rotation, translation).closing
        assert abs(placed[1]) < 1e-9, (
            f"the placed closing axis must sit on the tie-break, not at {placed[1]:.2e}"
        )


class TestDefectATheSourceFramesAgree:
    def test_the_placed_frame_is_the_pick_frame_carried_forward(self):
        """The cheapest statement of the invariant, and the one that broke.

        The object is rigidly held between grasp and release, so its placed task
        frame *is* its picked task frame moved by the demonstration's own rigid
        motion. Measured at 180.0 degrees before the fix.
        """
        source, target, labels = scene()
        _, _, d = scene_keypoints(source, target, labels)
        carried = d["carry_rotation"] @ d["source_frame"]
        assert ang(d["source_place_frame"], carried) < 1e-6

    def test_task_frame_no_longer_invents_a_sign(self):
        """Reversing the closing axis must reverse the frame, not preserve it.

        A grasp is a pose, not an axis: ``Grasp6D`` carries a rotation whose
        first column *is* the closing direction, with a sign fixed by the planner
        and by which finger of that hand is which. There is nothing here to
        resolve.
        """
        a = task_frame([0, 0, 1], [0.6, 0.8, 0.0])
        b = task_frame([0, 0, 1], [-0.6, -0.8, 0.0])
        assert np.allclose(a[:, 0], -b[:, 0])
        assert np.allclose(a[:, 2], b[:, 2]), "the support normal is untouched"


class TestTheTargetRollIsChosenOnce:
    """A parallel jaw grips identically either way round. Something must choose.

    The choice is legitimate and it is *not* the sign resolution that Defect A
    removed. That one ran independently at four blocks and decided by asking
    whether an axis pointed along world ``+y`` -- a question about the object's
    yaw, unrelated to the gripper. This runs once, on the grasp, and asks
    whether it agrees with the demonstration.

    It is also not optional. ``phi`` must carry the source cube onto the target
    cube, so the angle between their frames is a rotation the map has to
    realise, and past about 145 degrees it folds: measured across 20 real cells,
    three of twenty maps had ``min det(J)`` at -0.06 to -0.01 without this, and
    a worst case of 0.456 with it.
    """

    def test_both_target_blocks_take_the_same_roll(self):
        """The property that makes this safe where the old resolution was not."""
        source, target, labels = scene()
        S, _, d = scene_keypoints(source, target, labels, box="grasp_cube",
                                  orientation="grasp")
        picked = np.array([S.points[S.labels.index(f"pick_{n}")]
                           for n in ("center", "nnn", "pnn", "ppn", "npn",
                                     "nnp", "pnp", "ppp", "npp")])
        placed = np.array([S.points[S.labels.index(f"place_{n}")]
                           for n in ("center", "nnn", "pnn", "ppn", "npn",
                                     "nnp", "pnp", "ppp", "npp")])
        carried = apply_transform(picked, d["carry_rotation"], d["carry_translation"])
        assert np.allclose(carried, placed, atol=1e-9), (
            "the source's placed cube is not its picked cube carried forward; "
            f"worst corner off by {np.abs(carried - placed).max() * 1000:.2f} mm"
        )

    def test_the_chosen_grasp_is_reported(self):
        """Downstream has to convert the grasp *as executed*, not as planned.

        ``grasp_to_eef_pose`` given the un-rolled grasp would command the other
        roll and disagree with the plan by 180 degrees, so the diagnostics carry
        the pose actually used.
        """
        source, target, labels = scene()
        _, _, d = scene_keypoints(source, target, labels)
        assert "target_grasp" in d and "target_grasp_rolled" in d
        expected = (-1.0 if d["target_grasp_rolled"] else 1.0) * np.asarray(
            target.grasp.closing, float
        )
        assert np.allclose(d["target_grasp"].closing, expected)

    def test_flip_target_takes_the_other_one(self):
        source, target, labels = scene()
        _, _, a = scene_keypoints(source, target, labels)
        _, _, b = scene_keypoints(source, target, labels, flip_target=True)
        assert a["target_grasp_rolled"] is not b["target_grasp_rolled"]
        assert np.allclose(a["target_grasp"].closing, -np.asarray(b["target_grasp"].closing))

    def test_flip_target_actually_does_something(self):
        """It used to be a no-op, and that hid a whole search.

        The old sign resolution ran *after* ``flip_target`` negated the closing
        axis and put the sign back, so both branches produced byte-identical
        keypoints. Measured across a target-yaw sweep on the old code, the two
        branches returned the same ``min det`` to three decimals at all eight
        yaws -- including the two where both were negative.

        ``pipeline._choose_grasp`` loops ``for flip in (False, True)`` and keeps
        whichever scores better on reachability. That search was scoring one
        option twice for the life of the project.
        """
        source, target, labels = scene()
        out = []
        for flip in (False, True):
            S, T, _ = scene_keypoints(source, target, labels, box="grasp_cube",
                                      orientation="grasp", flip_target=flip)
            out.append(TransportMap().fit(S.points, T.points))
        a, b = (m.check_diffeomorphism(labels.positions).min_determinant for m in out)
        assert abs(a - b) > 1e-6, f"flip_target changed nothing: {a:.6f} vs {b:.6f}"

    @pytest.mark.parametrize("target_yaw", [-2.6, -1.8, -1.1, -0.4, 0.3, 1.2, 2.4, 3.0])
    def test_some_roll_always_gives_a_usable_map(self, target_yaw):
        """A target at any yaw must be servable by *one* of the two rolls.

        This is deliberately not "the default roll always works". It does not:
        at two of these eight yaws the cheap criterion -- take the roll whose
        closing axis agrees with the demonstration's -- picks the worse one, and
        the map folds at ``min det`` -0.38 and -0.44.

        **That is not a regression.** The original code folds at the same two
        yaws, at -0.43 and -0.39, and could not be rescued because
        ``flip_target`` was inert. What changed is that the other roll is now
        reachable, and at both of those yaws it gives 0.87 and 0.97.

        So the guarantee the construction can honestly offer is this one, and a
        caller that needs a conditioned map must try both and check -- which is
        what ``pipeline._choose_grasp`` already believes it is doing.
        """
        source, target, labels = scene(target_yaw=target_yaw)
        best = -np.inf
        for flip in (False, True):
            S, T, _ = scene_keypoints(source, target, labels, box="grasp_cube",
                                      orientation="grasp", flip_target=flip)
            m = TransportMap().fit(S.points, T.points)
            best = max(best, m.check_diffeomorphism(labels.positions).min_determinant)
        assert best > 0.2, f"yaw {target_yaw}: best roll only reaches min det {best:.3f}"


class TestDefectBTheConventionsAreConverted:
    def test_the_round_trip_is_the_identity(self):
        for hand in ("panda", "xarm", "robotiq85", "yumi", "robotiq140"):
            R = yaw_matrix(0.37)
            assert np.allclose(to_wrist_convention(to_grasp_convention(R, hand), hand), R)

    def test_the_two_conventions_really_do_differ(self):
        """Assert the correction is worth applying before trusting that it is.

        If every hand's alignment were the identity there would be nothing to
        forget and no defect. Measured: 0.2 degrees on a Robotiq 2F-85, 90 on an
        XArm, 180 on a Panda.
        """
        angles = {h: ang(np.eye(3), alignment_rotation(h))
                  for h in ("robotiq85", "xarm", "panda")}
        assert angles["robotiq85"] < 5.0
        assert 85.0 < angles["xarm"] < 95.0
        assert angles["panda"] > 175.0

    def test_it_is_applied_per_hand_and_not_per_source(self):
        """The specific way the old code was wrong.

        Transporting a demonstration recorded in one hand's wrist frame, without
        saying so, attaches the *source* hand's alignment to the *target* hand's
        command. Converting into the grasp convention first and back out per hand
        is what removes that.
        """
        R = yaw_matrix(0.37)
        neutral = to_grasp_convention(R, "panda")
        assert ang(to_wrist_convention(neutral, "xarm"),
                   to_wrist_convention(neutral, "panda")) > 85.0


class TestTheInstrumentThatWouldHaveCaughtIt:
    """``carry_orientation_error``, and why the per-end metric could not."""

    def _identity_map(self):
        class Identity:
            def orthogonal_jacobian(self, X):
                return np.broadcast_to(np.eye(3), (len(np.atleast_2d(X)), 3, 3))
        return Identity()

    def test_a_matched_pair_reads_zero(self):
        R = yaw_matrix(0.3)
        assert carry_orientation_error(
            self._identity_map(), [0, 0, 0], [1, 0, 0], R, R, R, R
        ) == pytest.approx(0.0, abs=1e-4)

    def test_a_half_turn_at_both_ends_cancels(self):
        """The same task with the wrist rolled over is the same task."""
        R = yaw_matrix(0.3)
        assert carry_orientation_error(
            self._identity_map(), [0, 0, 0], [1, 0, 0],
            R, R, R @ JAW_SYMMETRY, R @ JAW_SYMMETRY,
        ) == pytest.approx(0.0, abs=1e-4)

    def test_a_half_turn_at_one_end_only_reads_180(self):
        R = yaw_matrix(0.3)
        assert carry_orientation_error(
            self._identity_map(), [0, 0, 0], [1, 0, 0], R, R, R @ JAW_SYMMETRY, R,
        ) == pytest.approx(180.0, abs=1e-3)

    def test_it_reads_near_zero_on_the_real_construction(self):
        """End to end on the scene that used to produce 180 degrees."""
        source, target, labels = scene()
        S, T, d = scene_keypoints(source, target, labels, box="grasp_cube",
                                  orientation="grasp")
        m = TransportMap().fit(S.points, T.points)
        g, r = carry_indices(labels)
        assert carry_orientation_error(
            m, labels.positions[g], labels.positions[r],
            labels.orientations[g], labels.orientations[r],
            d["target_grasp"].rotation, d["target_place_grasp"].rotation,
        ) < 8.0


class TestThePhysicalConsequence:
    def test_the_object_lands_where_the_keypoints_put_it(self):
        """In millimetres, which is the form the defect was actually costing.

        Following the plan moves the object by the hand's relative rotation about
        the grasp point, and that has to agree with the placed configuration the
        target keypoints encode. A half turn does not move the grasp point at
        all -- which is why the aim stayed exact and only this sees it -- but it
        reflects everything else through it.
        """
        source, target, labels = scene()
        S, T, d = scene_keypoints(source, target, labels, box="grasp_cube",
                                  orientation="grasp")
        m = TransportMap().fit(S.points, T.points)
        g, r = carry_indices(labels)
        warped = m.transport_positions(labels.positions)
        rot = m.transport_orientations(labels.positions, labels.orientations)
        relative = np.asarray(rot[r]) @ np.asarray(rot[g]).T

        centroid = target.points.mean(0)
        under_the_plan = relative @ (centroid - warped[g]) + warped[r]
        rotation, translation = kabsch(target.points, d["target_place_points"])
        as_intended = rotation @ centroid + translation
        gap = float(np.linalg.norm(under_the_plan - as_intended))
        assert gap < 0.02, f"plan lands the object {gap * 1000:.1f} mm off"

    def test_the_aim_stays_exact(self):
        """The property the cube exists for, unaffected by any of this."""
        source, target, labels = scene()
        S, T, d = scene_keypoints(source, target, labels, box="grasp_cube",
                                  orientation="grasp")
        m = TransportMap().fit(S.points, T.points)
        moved = m.transport_positions(np.asarray(source.grasp.tcp)[None])[0]
        assert np.linalg.norm(moved - d["target_grasp"].tcp) < 1e-4
