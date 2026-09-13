"""The fixed cube centred on the grasp, as an alternative to the cloud-fitted box.

Everything here is a **paired** claim: the same scene measured under
``box="cloud"`` and under ``box="grasp_cube"``, so the contrast is the assertion
rather than a bare threshold. The cloud-box half of each pair restates what
``TestSceneKeypoints`` already establishes, deliberately -- those findings are not
superseded, because that construction is still the default. What changes is that
a second construction now exists and behaves differently, and the difference is
what is pinned.

The scene builder is borrowed from :mod:`test_keypoint_extraction` rather than
copied. It is synthetic: one-sided box clouds, no gripper model, grasps derived
by :func:`top_down_grasp`. So these tests pin *mechanism*. Real clouds and real
grippers are a separate, simulator-marked question.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.metrics.transport import orientation_transport_error, vertical_tilt
from tpgpt.sim.keypoints import (
    CONTACT_NAMES,
    CORNER_NAMES,
    GRASP_CUBE_HALF_EXTENT,
    GraspFrame,
    ObjectPlacement,
    cube_keypoints,
    grasp_pose_frame,
    object_keypoints,
    scene_keypoints,
    task_frame,
)
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import is_rotation

from tests.unit.test_keypoint_extraction import (
    HALF,
    TABLE,
    TestSceneKeypoints,
    box_cloud,
)

CUBE = dict(box="grasp_cube")
CLOUD = dict(box="cloud")


@pytest.fixture
def scene():
    """``(source, target, labels)`` from the builder the existing tests use."""

    def build(**kw):
        source, target, labels, _ = TestSceneKeypoints()._scene(**kw)
        return source, target, labels

    return build


def fit(source, target, labels, **kw):
    S, T, _ = scene_keypoints(source, target, labels, **kw)
    return TransportMap().fit(S.points, T.points)


def aim(m, source, target) -> float:
    """``||phi(source grasp TCP) - target grasp TCP||``, in metres."""
    return float(
        np.linalg.norm(m.transport_positions(source.grasp.tcp[None])[0] - target.grasp.tcp)
    )


def retilt(placement, degrees: float) -> ObjectPlacement:
    """The same placement with its grasp's approach tilted out of plane."""
    if not degrees:
        return placement
    t = np.radians(degrees)
    return ObjectPlacement(
        placement.points,
        GraspFrame(
            tcp=placement.grasp.tcp,
            approach=[np.sin(t), 0.0, -np.cos(t)],
            closing=placement.grasp.closing,
        ),
        placement.support_height,
        placement.destination,
        placement.destination_height,
    )


class TestGraspRotation:
    def test_the_closing_axis_is_orthogonalised_against_the_approach(self):
        """``GraspFrame`` normalises its two axes independently and does not
        square them against each other. Measured on this case,
        ``approach . closing = -0.0497``. A frame built from the raw pair would
        not be a rotation, and nothing downstream would raise -- ``J_perp`` would
        carry the skew into every transported orientation, stiffness and
        damping."""
        g = GraspFrame(tcp=np.zeros(3), approach=[0, 0.1, -1.0], closing=[1, 0, 0.05])
        assert abs(float(g.approach @ g.closing)) > 1e-3  # the input really is skewed
        assert is_rotation(g.rotation)

    def test_the_approach_is_the_third_column(self):
        """Matches the ``Grasp6D`` convention: ``+X`` closing, ``+Z`` approach."""
        g = GraspFrame(tcp=np.zeros(3), approach=[0, 0, -1.0], closing=[1, 0, 0])
        assert np.allclose(g.rotation[:, 2], g.approach)
        assert np.allclose(g.rotation[:, 0], g.closing)

    def test_a_closing_axis_along_the_approach_is_refused(self):
        g = GraspFrame(tcp=np.zeros(3), approach=[0, 0, -1.0], closing=[0, 0, 1.0])
        with pytest.raises(ValueError, match="parallel to the approach"):
            _ = g.rotation


class TestGraspPoseFrame:
    @pytest.mark.parametrize(
        "closing", [[1, 0, 0.05], [-1, 0, -0.05], [0, 1, 0], [0, -1, 0], [0.7, 0.7, 0]]
    )
    def test_the_closing_sign_agrees_with_the_task_frame(self, closing):
        """The single highest-risk detail in the whole construction.

        A parallel jaw is physically identical under ``closing -> -closing``, but
        flipping it turns the cube 180 degrees and permutes every corner label.
        That is the mechanism ``task_frame``'s docstring records as costing the
        reshelving campaign 17/20 -> 5/20. If the two frames ever disagree about
        the sign, the cube's corners pair up wrongly against the task frame's and
        a half turn is planted in the middle of the map."""
        g = GraspFrame(tcp=np.zeros(3), approach=[0, 0, -1.0], closing=closing)
        assert np.allclose(grasp_pose_frame(g)[:, 0], task_frame([0, 0, 1], g.closing)[:, 0])

    def test_it_is_a_proper_rotation(self):
        g = GraspFrame(tcp=np.zeros(3), approach=[0, 0.3, -1.0], closing=[1, 0.2, 0])
        assert is_rotation(grasp_pose_frame(g))

    def test_it_needs_no_support_plane_and_so_refuses_less(self):
        """It used to refuse a vertically-closing grasp. It should not.

        **This assertion is the reverse of the one it replaces.** The old
        ``grasp_pose_frame`` called ``task_frame`` in order to re-derive the
        closing axis's sign, and inherited that function's refusal: a closing
        axis parallel to the support normal cannot be projected into the support
        plane, so no task frame exists. The refusal was correct *for a task
        frame* and was only reaching this function because of the borrowed sign
        resolution.

        With the sign taken from the grasp instead, this frame is the grasp's own
        rotation and touches no support plane at all. A hand driving in along
        ``+x`` with its jaws closing vertically along ``+z`` is a perfectly
        ordinary grasp -- a side approach onto a flat object -- and there is
        nothing left to object to. So the grasp-pose cube now admits grasps the
        cloud box cannot express, which is a small gain in reach rather than a
        regression.

        The refusal itself is not lost. ``scene_keypoints`` still builds a task
        frame for the cloud-fitted box and for the support contact, so a
        vertically-closing grasp is still rejected there, by the function whose
        geometry actually requires it.
        """
        g = GraspFrame(tcp=np.zeros(3), approach=[1, 0, 0.0], closing=[0, 0, 1.0])
        frame = grasp_pose_frame(g)
        assert is_rotation(frame)
        assert np.allclose(frame[:, 2], [1, 0, 0]), "third column is the approach"
        assert np.allclose(frame[:, 0], [0, 0, 1]), "first column is the closing axis"
        # and the task frame, which genuinely needs the support plane, still refuses it
        with pytest.raises(ValueError, match="parallel to the support normal"):
            task_frame([0, 0, 1], g.closing)


class TestLayoutIsUnchanged:
    """The cube mode must be a drop-in: same count, same order, same roles.

    Positional slicing (``viz.keypoint_figures`` reads ``points[:1]``, ``[1:5]``,
    ``[5:9]``), label-role routing (``pipeline._select_parts`` silently drops an
    unrecognised role) and the metadata contract all depend on this.
    """

    def _cloud(self):
        return box_cloud(np.array([0.0, 0.0, TABLE + HALF[2]]), faces="visible")

    def _grasp(self, points):
        from tpgpt.sim.keypoints import top_down_grasp

        return top_down_grasp(points)

    @pytest.mark.parametrize("kw, n", [(CLOUD, 9), (CUBE, 9)])
    def test_nine_points_without_contacts(self, kw, n):
        points = self._cloud()
        ks = object_keypoints(points, self._grasp(points), support_height=TABLE, **kw)
        assert len(ks) == n
        assert ks.labels[0] == "obj_center"
        assert ks.labels[1:] == [f"obj_{c}" for c in CORNER_NAMES]

    @pytest.mark.parametrize("kw", [CLOUD, CUBE])
    def test_fourteen_points_with_contacts(self, kw):
        points = self._cloud()
        ks = object_keypoints(
            points, self._grasp(points), support_height=TABLE, include_contacts=True, **kw
        )
        assert len(ks) == 14
        assert ks.labels[9:] == [f"obj_{c}" for c in CONTACT_NAMES]

    def test_the_cube_block_is_centred_on_the_grasp_not_the_cloud(self):
        points = self._cloud()
        grasp = self._grasp(points)
        cube = object_keypoints(points, grasp, support_height=TABLE, **CUBE)
        cloud = object_keypoints(points, grasp, support_height=TABLE, **CLOUD)
        assert np.allclose(cube.points[0], grasp.tcp)
        assert not np.allclose(cloud.points[0], grasp.tcp)

    def test_the_cube_block_is_literally_a_cube_keypoints_block(self):
        """Mirrors the generalisation property the cloud mode is held to."""
        points = self._cloud()
        grasp = self._grasp(points)
        ks = object_keypoints(points, grasp, support_height=TABLE, **CUBE)
        expected = cube_keypoints(
            grasp.tcp,
            task_frame([0, 0, 1], grasp.closing),
            GRASP_CUBE_HALF_EXTENT,
        )
        assert np.allclose(np.sort(ks.points[:9], axis=0), np.sort(expected.points, axis=0))

    def test_the_metadata_still_records_the_objects_own_extent(self):
        """``diagnose.closing_budget`` derives ``(aperture - width)/2`` from the
        keypoint spread along the closing axis. Under a fixed cube the keypoints
        no longer carry the object's width, so without ``cloud_half_extents`` the
        budget would silently start describing the cube and hand back a
        confident, wrong tolerance."""
        points = self._cloud()
        ks = object_keypoints(points, self._grasp(points), support_height=TABLE, **CUBE)
        assert ks.metadata["block_kind"] == "grasp_cube"
        assert ks.metadata["cube_half_extent"] == pytest.approx(GRASP_CUBE_HALF_EXTENT)
        assert np.allclose(ks.metadata["box_half_extents"], GRASP_CUBE_HALF_EXTENT)
        assert np.allclose(ks.metadata["cloud_half_extents"], HALF, atol=3e-3)

    def test_a_cloud_box_cannot_be_laid_out_in_the_grasp_pose(self):
        points = self._cloud()
        with pytest.raises(ValueError, match="snapped to the support plane"):
            object_keypoints(
                points, self._grasp(points), support_height=TABLE,
                box="cloud", orientation="grasp",
            )

    @pytest.mark.parametrize("bad", [dict(box="nope"), dict(orientation="nope")])
    def test_an_unknown_mode_is_refused_rather_than_ignored(self, bad):
        points = self._cloud()
        with pytest.raises(ValueError, match="unknown"):
            object_keypoints(points, self._grasp(points), support_height=TABLE, **bad)

    def test_coincident_keypoints_are_refused_with_a_useful_message(self):
        """Without this the failure arrives from inside the GP as
        ``LinAlgError('Gram matrix is not positive definite ...')``, naming the
        symptom and not the cause."""
        points = self._cloud()
        grasp = self._grasp(points)
        flat = GraspFrame(tcp=[0, 0, TABLE], approach=grasp.approach, closing=grasp.closing)
        with pytest.raises(ValueError, match="would make the map"):
            object_keypoints(
                points, flat, support_height=TABLE, include_contacts=True, **CUBE
            )


class TestVolumeDoesNotScaleWithObjectSize:
    """The claim the construction exists for.

    The cloud box's extent *is* the object's, so a taller target asks ``phi`` to
    inflate space in proportion. The trajectory has no reason to care how tall
    the object is, and a fixed cube does not make it care.
    """

    HEIGHTS = (0.02, 0.075, 0.15)

    def _dets(self, scene, **kw):
        out = []
        for h in self.HEIGHTS:
            src, tgt, L = scene(target_angle=0.8, target_half=np.array([0.03, 0.05, h]))
            out.append(fit(src, tgt, L, **kw).check_diffeomorphism(L.positions).min_determinant)
        return np.array(out)

    def test_the_cloud_box_inflates_volume_with_the_size_ratio(self, scene):
        dets = self._dets(scene, **CLOUD)
        assert dets[0] < dets[1] < dets[2]  # monotone in the height ratio
        assert dets[2] / dets[0] > 2.0

    def test_the_grasp_cube_is_flat_across_the_same_range(self, scene):
        dets = self._dets(scene, **CUBE)
        assert np.ptp(dets) < 1e-6

    def test_neither_construction_folds_over_this_range(self, scene):
        """Explicitly recorded: the cloud box's problem here is *scaling*, not
        folding. An earlier synthetic claimed a fold; it was an artifact of a
        hand-built scene and is withdrawn."""
        for kw in (CLOUD, CUBE):
            for h in self.HEIGHTS:
                src, tgt, L = scene(target_angle=0.8, target_half=np.array([0.03, 0.05, h]))
                assert fit(src, tgt, L, **kw).check_diffeomorphism(L.positions).consistent_sign


class TestAimAtTheGrasp:
    def test_the_cloud_box_misses_when_the_grasp_heights_differ(self, scene):
        """The delta/L root cause, end to end: the cloud box is centred on the
        object's centroid, so when the target grasp sits elsewhere on its object
        nothing carries the grasp point across."""
        src, tgt, L = scene(target_angle=0.8, target_grasp_fraction=1.0)
        matched = aim(fit(src, tgt, L, **CLOUD), src, tgt)
        src, tgt, L = scene(target_angle=0.8, target_grasp_fraction=0.5)
        mismatched = aim(fit(src, tgt, L, **CLOUD), src, tgt)
        assert matched < 0.01
        assert mismatched > 0.05
        assert mismatched > 5 * matched

    @pytest.mark.parametrize("fraction", [1.0, 0.75, 0.5, 0.25])
    def test_the_grasp_cube_carries_the_grasp_point_exactly(self, scene, fraction):
        """Exact by construction -- the cube's centre *is* the grasp and centres
        are keypoints, which ``phi`` interpolates exactly. Asserted anyway,
        because it is the property that makes the contacts affordable below."""
        src, tgt, L = scene(target_angle=0.8, target_grasp_fraction=fraction)
        assert aim(fit(src, tgt, L, **CUBE), src, tgt) < 1e-6


class TestContactsBecomeAffordable:
    @pytest.mark.parametrize("fraction", [0.75, 0.5, 0.25])
    def test_contacts_fold_the_cloud_box_but_not_the_cube(self, scene, fraction):
        """The paired form of ``test_contact_keypoints_fold_the_map_when_grasp_
        heights_differ``, which pins the same finding for the default."""
        src, tgt, L = scene(target_angle=0.8, target_grasp_fraction=fraction)
        cloud = fit(src, tgt, L, include_contacts=True, **CLOUD)
        cube = fit(src, tgt, L, include_contacts=True, **CUBE)
        assert not cloud.check_diffeomorphism(L.positions).consistent_sign
        report = cube.check_diffeomorphism(L.positions)
        assert report.consistent_sign and report.min_determinant > 0.2

    @pytest.mark.parametrize("degrees", [0, 15, 30, 45, 60])
    def test_the_cube_holds_contacts_through_a_tilted_approach(self, scene, degrees):
        """The contact plane's normal is the grasp's *approach*, which a
        task-frame cube does not share, so the two can disagree. A small cube
        does not reach far enough for that to matter; measured, a 60 mm cube
        collapses to 0.081 at 60 degrees while a 20 mm one stays above 0.9."""
        src, tgt, L = scene(target_angle=0.8, target_grasp_fraction=0.5)
        m = fit(src, retilt(tgt, degrees), L, include_contacts=True, **CUBE)
        report = m.check_diffeomorphism(L.positions)
        assert report.consistent_sign and report.min_determinant > 0.5


class TestCubeSizeSetsTheRotationStencil:
    """Why the size matters, and which metric it moves.

    ``phi`` interpolates keypoint *positions* exactly at any size. But Eq. 11
    reads ``J_perp``, a **derivative**, and the eight corners are the only thing
    pinning the local rotation -- so their distance from the centre is the width
    of the stencil over which it is estimated. A wide stencil averages the
    grasp's own rotation together with the far field, which is the affine stage.
    """

    def test_aim_is_exact_at_every_size(self, scene):
        """Position is carried exactly whatever the size -- so aim cannot be the
        metric that chooses one.

        The bound is 0.1 mm rather than machine precision because interpolation
        gets numerically *worse* as the cube shrinks: measured residual at the
        grasp is about 3.9 um for a 5 mm cube against well under 1 um for 20 mm
        and up, because crowding the keypoints together conditions the Gram
        matrix worse. Physically irrelevant at these magnitudes, but it is a
        second reason -- alongside cloud noise -- not to push the cube
        arbitrarily small."""
        src, tgt, L = scene(target_angle=0.8)
        for h in (0.005, 0.02, 0.06, 0.10):
            assert aim(fit(src, tgt, L, box="grasp_cube", cube_half_extent=h), src, tgt) < 1e-4

    def test_interpolation_conditions_worse_as_the_cube_shrinks(self, scene):
        """The ordering, recorded so the effect is not rediscovered as a bug."""
        src, tgt, L = scene(target_angle=0.8)
        tiny = aim(fit(src, tgt, L, box="grasp_cube", cube_half_extent=0.005), src, tgt)
        default = aim(fit(src, tgt, L, box="grasp_cube", cube_half_extent=0.02), src, tgt)
        assert tiny > default

    def test_the_orientation_error_grows_with_the_stencil(self, scene):
        src, tgt, L = scene(target_angle=0.8)
        errs = [
            orientation_transport_error(
                fit(src, tgt, L, box="grasp_cube", cube_half_extent=h),
                src.grasp.tcp[None], src.grasp.rotation, tgt.grasp.rotation,
            )[0]
            for h in (0.005, 0.02, 0.06)
        ]
        assert errs[0] < errs[1] < errs[2]
        assert errs[0] < 1.0 and errs[2] > 5.0

    def test_a_cube_larger_than_the_scene_folds(self, scene):
        """The upper bound, and the reason the default is 20 mm: a 100 mm half
        extent is a 200 mm cube against objects 20-75 mm and a 355 mm pick-to-
        place distance."""
        src, tgt, L = scene(target_angle=0.8)
        m = fit(src, tgt, L, box="grasp_cube", cube_half_extent=0.10)
        assert not m.check_diffeomorphism(L.positions).consistent_sign


class TestTheMidPathTiltIsAffineNotOrientation:
    """Where the transit tilt comes from, since it is easy to misattribute.

    Every keypoint block is vertical-aligned -- ``task_frame``'s third column is
    the support normal, and ``target_place_frame`` is ``source_place_frame`` -- so
    no block tilts the vertical. The tilt appears *between* blocks, and
    mid-transit is far enough from all of them that ``phi -> gamma``: the tilt
    there is the **affine** stage's. It is therefore a global property of the
    keypoint set, not a consequence of the corner orientation, and it may well be
    desirable (a consistently held object). It is measured, not scored.
    """

    def test_every_block_is_vertical_aligned(self, scene):
        src, tgt, L = scene(target_angle=0.8)
        S, T, _ = scene_keypoints(src, tgt, L, **CUBE)
        for ks, blocks in ((S, ("src_pick", "src_place")), (T, ("tgt_pick", "tgt_place"))):
            for b in blocks:
                frame = np.array(ks.metadata[b]["corner_frame"])
                assert np.allclose(frame[:, 2], [0, 0, 1])

    def test_the_mid_path_tilt_matches_the_affine_stages_own_tilt(self, scene):
        src, tgt, L = scene(target_angle=0.8)
        m = fit(src, tgt, L, **CUBE)
        A = m.affine.jacobian()
        affine_tilt = np.degrees(np.arccos(np.clip((A @ [0, 0, 1.0]) @ [0, 0, 1.0], -1, 1)))
        tilt = vertical_tilt(m, L.positions)
        assert tilt[len(tilt) // 2] == pytest.approx(affine_tilt, abs=3.0)

    def test_the_two_corner_orientations_agree_for_a_top_down_grasp(self, scene):
        """A free internal control. For a top-down grasp the task frame and the
        grasp pose differ by a half turn about the closing axis, which maps a
        cube's corners onto themselves -- so the two modes must produce the same
        map. If they ever diverge here, the sign resolution has broken."""
        src, tgt, L = scene(target_angle=0.8)
        task = fit(src, tgt, L, box="grasp_cube", orientation="task")
        pose = fit(src, tgt, L, box="grasp_cube", orientation="grasp")
        assert np.allclose(
            task.transport_positions(L.positions),
            pose.transport_positions(L.positions),
            atol=1e-9,
        )

    def test_a_tilted_approach_separates_them(self, scene):
        """And with an out-of-plane tilt they must diverge, or the grasp-pose
        mode is doing nothing."""
        src, tgt, L = scene(target_angle=0.8)
        tilted = retilt(tgt, 30.0)
        task = fit(src, tilted, L, box="grasp_cube", orientation="task")
        pose = fit(src, tilted, L, box="grasp_cube", orientation="grasp")
        args = (src.grasp.tcp[None], src.grasp.rotation, tilted.grasp.rotation)
        assert orientation_transport_error(pose, *args)[0] < orientation_transport_error(
            task, *args
        )[0]
