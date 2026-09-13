"""Grasp-aligned keypoint extraction (paper Sec. III-A).

The properties tested here are the ones that fail *silently* when broken: a
keypoint set that is geometrically plausible but wrongly paired, or wrongly
oriented, still fits a map with a near-zero residual and a positive Jacobian
determinant. Every failure found while building this module looked healthy on
those diagnostics and only showed up as a rotated gripper.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.sim.keypoints import (
    CONTACT_NAMES,
    CORNER_NAMES,
    GraspFrame,
    ObjectPlacement,
    apply_transform,
    carry_indices,
    carry_transform,
    cube_keypoints,
    fit_aligned_box,
    grasp_contacts,
    object_keypoints,
    pair_keypoints,
    placement_rotation,
    sample_box_surface,
    scene_keypoints,
    support_contact,
    task_frame,
    to_frame,
    top_down_grasp,
)
from tpgpt.transport.labels import PolicyLabels
from tpgpt.transport.maps import TransportMap

HALF = np.array([0.025, 0.04, 0.045])
TABLE = 0.80


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_cloud(centre, angle=0.0, half=HALF, n=1200, seed=0, faces="all"):
    return sample_box_surface(
        centre, yaw_matrix(angle), half, n=n, rng=np.random.default_rng(seed), faces=faces
    )


def demo_labels(grasp_xyz, place_xyz, grasp_yaw=0.0, place_yaw=0.0, n=40):
    """A minimal demonstration: close, carry, hold still, release.

    The hold matters. A demonstration whose gripper opens while the hand is
    still moving releases the object short of the slot, and the extractor
    faithfully inherits that miss -- correct behaviour, but not what a test of
    the placement logic should be measuring.
    """
    hold, carry = 5, n - 10
    positions = np.vstack(
        [
            np.repeat(np.asarray(grasp_xyz, float)[None], hold, axis=0),
            np.linspace(grasp_xyz, place_xyz, carry),
            np.repeat(np.asarray(place_xyz, float)[None], n - hold - carry, axis=0),
        ]
    )
    angles = np.concatenate(
        [
            np.full(hold, grasp_yaw),
            np.linspace(grasp_yaw, place_yaw, carry),
            np.full(n - hold - carry, place_yaw),
        ]
    )
    orientations = np.stack([yaw_matrix(a) for a in angles])
    gripper = np.where(
        (np.arange(n) >= hold) & (np.arange(n) < hold + carry), 1.0, -1.0
    )
    return PolicyLabels(
        positions=positions,
        velocities=np.gradient(positions, 0.05, axis=0),
        orientations=orientations,
        gripper=gripper,
        time_belief=np.linspace(0.0, 1.0, n),
    )


# --------------------------------------------------------------- task frame
class TestTaskFrame:
    def test_is_orthonormal_and_right_handed(self):
        frame = task_frame([0, 0, 1], [0.6, 0.8, 0.3])
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)

    def test_third_column_is_the_support_normal(self):
        frame = task_frame([0, 0, 1], [1, 0, 0])
        assert np.allclose(frame[:, 2], [0, 0, 1])

    def test_closing_axis_is_projected_into_the_support_plane(self):
        frame = task_frame([0, 0, 1], [1.0, 0.0, 5.0])
        assert frame[2, 0] == pytest.approx(0.0, abs=1e-12)

    def test_the_closing_sign_is_taken_from_the_grasp_and_not_re_derived(self):
        """Reversing the closing axis must give the *reversed* frame.

        **This assertion is the reverse of the one it replaces**, which required
        the two to come out equal. The reasoning behind that was: a parallel jaw
        closing along ``+c`` and along ``-c`` grips the same object the same
        way, so the sign is free and the frame should pick one by convention.

        The flaw is that a grasp is not an axis, it is a **pose**. ``Grasp6D``
        carries a full rotation whose first column *is* the closing direction,
        with a definite sign fixed by the planner and by which finger of that
        hand is which. There is nothing here to resolve, and resolving it anyway
        discarded that sign and invented a replacement from a world-axis test
        (``c . y >= 0``, tie-broken on ``c . x``) that depends on the object's
        yaw in the scene and knows nothing about the gripper.

        What it cost: the demonstration places its object square with the shelf,
        so the placed closing axis lands on world ``x`` at ``[1, -1e-17, 0]``.
        ``c[1]`` fell inside the epsilon, the ``c[0] < 0`` tie-break decided
        instead, and it decided the opposite way to the pick, whose ``c[1]`` is
        an unambiguous ``-0.581``. The source's own pick and place frames came
        out 180 degrees apart, which reflects the carried object through the
        grasp point and lands it twice its lateral grasp offset away -- measured
        at 4 to 100 mm over 20 cells. `ROBOTICS_NOTES.md` 7.33.
        """
        a = task_frame([0, 0, 1], [0.6, 0.8, 0.0])
        b = task_frame([0, 0, 1], [-0.6, -0.8, 0.0])
        assert np.allclose(a[:, 0], -b[:, 0]), "the closing column must reverse"
        assert np.allclose(a[:, 1], -b[:, 1]), "and so must the one it induces"
        assert np.allclose(a[:, 2], b[:, 2]), "the support normal is untouched"

    def test_refuses_a_closing_axis_along_the_support_normal(self):
        with pytest.raises(ValueError, match="parallel to the support normal"):
            task_frame([0, 0, 1], [0, 0, 1])


# ---------------------------------------------------------------- box fit
class TestAlignedBox:
    def test_recovers_the_true_extent_of_an_aligned_box(self):
        centre = np.array([0.1, -0.2, TABLE + HALF[2]])
        points = box_cloud(centre, angle=0.0)
        frame = task_frame([0, 0, 1], [1, 0, 0])
        fitted, half = fit_aligned_box(points, frame)
        assert np.allclose(fitted, centre, atol=2e-3)
        assert np.allclose(half, HALF[[0, 1, 2]], atol=2e-3)

    def test_recovers_extent_of_a_rotated_box_from_its_own_grasp(self):
        angle = 0.7
        centre = np.array([0.1, -0.2, TABLE + HALF[2]])
        points = box_cloud(centre, angle=angle)
        # Jaws close across the narrow side, which is the box's local x.
        frame = task_frame([0, 0, 1], yaw_matrix(angle)[:, 0])
        _, half = fit_aligned_box(points, frame)
        assert np.allclose(sorted(half), sorted(HALF), atol=2e-3)

    def test_dropping_the_bottom_face_alone_does_not_shorten_the_cloud(self):
        """Guards a claim it would be easy to assume and wrong to rely on.

        The side faces still reach the contact line, so a cloud missing only
        its underside already measures the right height. What the snap defends
        against is truncation of the lowest points -- mask erosion, occlusion --
        not the invisibility of one face.
        """
        centre = np.array([0.0, 0.0, TABLE + HALF[2]])
        frame = task_frame([0, 0, 1], [1, 0, 0])
        _, half = fit_aligned_box(box_cloud(centre, faces="visible"), frame)
        assert half[2] == pytest.approx(HALF[2], abs=2e-3)

    def test_support_snap_restores_a_truncated_cloud(self):
        centre = np.array([0.0, 0.0, TABLE + HALF[2]])
        points = box_cloud(centre)
        truncated = points[points[:, 2] > TABLE + 0.008]
        frame = task_frame([0, 0, 1], [1, 0, 0])

        _, unsnapped = fit_aligned_box(truncated, frame)
        _, snapped = fit_aligned_box(truncated, frame, support_height=TABLE)
        assert unsnapped[2] < HALF[2] - 3e-3
        assert snapped[2] == pytest.approx(HALF[2], abs=2e-3)


# ------------------------------------------------------------- the contacts
class TestContacts:
    def _setup(self, seed=0, n=1500):
        centre = np.array([0.0, 0.0, TABLE + HALF[2]])
        points = box_cloud(centre, n=n, seed=seed)
        frame = task_frame([0, 0, 1], [1, 0, 0])
        grasp = top_down_grasp(points, closing_axis=[1, 0, 0])
        fitted, half = fit_aligned_box(points, frame, support_height=TABLE)
        return points, grasp, frame, fitted, half

    def test_contacts_straddle_the_object_along_the_closing_axis(self):
        points, grasp, frame, centre, half = self._setup()
        contacts, from_cloud = grasp_contacts(points, grasp, frame, centre, half)
        assert from_cloud and len(contacts) == 4
        neg, pos = contacts[0], contacts[1]
        local = to_frame(np.stack([neg, pos]), frame)
        assert local[0, 0] < 0 < local[1, 0]
        assert np.linalg.norm(pos - neg) == pytest.approx(2 * HALF[0], abs=5e-3)

    def test_the_four_contacts_span_the_grasp_plane(self):
        """Two points give a line, and a line leaves the approach free.

        The plane's normal *is* the approach direction, so pinning the plane is
        what makes the transported hand arrive from the right side rather than
        merely at the right contact points.
        """
        points, grasp, frame, centre, half = self._setup()
        contacts, _ = grasp_contacts(points, grasp, frame, centre, half)
        # Every contact lies in the plane through the TCP normal to the approach.
        offsets = (contacts - grasp.tcp) @ grasp.approach
        assert np.abs(offsets).max() < 1e-9
        # And they span it: two independent directions, not one.
        centred = contacts - contacts.mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        assert singular[1] > 0.2 * singular[0]

    def test_the_contact_plane_normal_recovers_the_approach(self):
        points, grasp, frame, centre, half = self._setup()
        contacts, _ = grasp_contacts(points, grasp, frame, centre, half)
        centred = contacts - contacts.mean(axis=0)
        normal = np.linalg.svd(centred)[2][-1]
        assert abs(abs(float(normal @ grasp.approach)) - 1.0) < 1e-6

    def test_contact_axis_tracks_the_closing_axis_across_samplings(self):
        """The failure that cost 14 degrees of gripper yaw.

        Estimating the contacts' lateral coordinate from the same thin band of
        cloud made the contact axis wander with the sampling, and Eq. 11 turns
        that wander directly into gripper rotation.
        """
        for seed in range(6):
            points, grasp, frame, centre, half = self._setup(seed=seed)
            contacts, _ = grasp_contacts(points, grasp, frame, centre, half)
            neg, pos = contacts[0], contacts[1]
            axis = (pos - neg) / np.linalg.norm(pos - neg)
            assert abs(np.degrees(np.arccos(np.clip(axis @ frame[:, 0], -1, 1)))) < 1.0

    def test_falls_back_to_the_box_faces_on_a_cloud_too_sparse_to_slice(self):
        points, grasp, frame, centre, half = self._setup()
        sparse = points[:6]
        contacts, from_cloud = grasp_contacts(sparse, grasp, frame, centre, half)
        assert not from_cloud and len(contacts) == 4
        assert np.linalg.norm(contacts[1] - contacts[0]) == pytest.approx(
            2 * half[0], abs=1e-9
        )


class TestSupportContact:
    def test_lies_exactly_on_the_support_plane(self):
        centre = np.array([0.0, 0.0, TABLE + HALF[2]])
        points = box_cloud(centre, faces="visible")
        frame = task_frame([0, 0, 1], [1, 0, 0])
        contact = support_contact(points, frame, TABLE, centre)
        assert contact[2] == pytest.approx(TABLE)

    def test_is_stable_across_samplings_of_the_same_object(self):
        """An unstable contact plants a local deformation under the object."""
        centre = np.array([0.0, 0.0, TABLE + HALF[2]])
        frame = task_frame([0, 0, 1], [1, 0, 0])
        contacts = [
            support_contact(box_cloud(centre, seed=s), frame, TABLE, centre)
            for s in range(5)
        ]
        assert np.ptp(np.stack(contacts), axis=0).max() < 1e-9


# ----------------------------------------------------------- the twelve
class TestObjectKeypoints:
    def test_cardinality_and_labels_are_fixed(self):
        points = box_cloud([0.0, 0.0, TABLE + HALF[2]])
        keypoints = object_keypoints(
            points, top_down_grasp(points), support_height=TABLE, name="o"
        )
        assert len(keypoints) == 9
        assert keypoints.labels[0] == "o_center"
        assert keypoints.labels[1:] == [f"o_{n}" for n in CORNER_NAMES]

    def test_contacts_are_opt_in_and_appended(self):
        points = box_cloud([0.0, 0.0, TABLE + HALF[2]])
        keypoints = object_keypoints(
            points,
            top_down_grasp(points),
            support_height=TABLE,
            name="o",
            include_contacts=True,
        )
        assert len(keypoints) == 14
        assert keypoints.labels[9:] == [f"o_{n}" for n in CONTACT_NAMES]

    @pytest.mark.parametrize(
        "half", [(0.025, 0.04, 0.045), (0.05, 0.05, 0.02), (0.03, 0.03, 0.12)]
    )
    def test_cardinality_is_the_same_whatever_the_shape(self, half):
        points = box_cloud([0.0, 0.0, TABLE + half[2]], half=np.array(half))
        keypoints = object_keypoints(
            points, top_down_grasp(points), support_height=TABLE, name="o"
        )
        assert len(keypoints) == 9

    def test_lower_corners_lie_on_the_support_plane(self):
        """The user-facing guarantee: contact points must match exactly.

        Property (i) of Sec. III-D interpolates keypoints exactly, so pinning
        these to the plane is what makes phi take table to shelf rather than to
        somewhere a centimetre above or below it.
        """
        points = box_cloud([0.0, 0.0, TABLE + HALF[2]], faces="visible")
        keypoints = object_keypoints(
            points, top_down_grasp(points), support_height=TABLE, name="o"
        )
        for name in CORNER_NAMES[:4]:
            assert keypoints.points[keypoints.labels.index(f"o_{name}")][2] == pytest.approx(
                TABLE, abs=1e-9
            )

    def test_reduces_to_the_pose_attached_cube_it_generalises(self):
        """The scheme validated at 17/20 must be a special case of this one."""
        centre = np.array([0.1, -0.2, TABLE + HALF[2]])
        angle = 0.35
        points = box_cloud(centre, angle=angle, n=4000)
        rotation = yaw_matrix(angle)
        grasp = GraspFrame(
            tcp=centre + [0, 0, HALF[2]], approach=[0, 0, -1], closing=rotation[:, 0]
        )
        general = object_keypoints(
            points, grasp, support_height=float(centre[2] - HALF[2]), name="o"
        )
        cube = cube_keypoints(centre, rotation, HALF, name="o")

        # Same nine points, up to which horizontal axis the frame calls "x".
        got = np.sort(general.points[:9], axis=0)
        want = np.sort(cube.points, axis=0)
        assert np.allclose(got, want, atol=3e-3)

    def test_refuses_a_cloud_too_sparse_to_have_an_extent(self):
        with pytest.raises(ValueError, match="too few"):
            object_keypoints(
                np.zeros((3, 3)),
                GraspFrame([0, 0, 0], [0, 0, -1], [1, 0, 0]),
                support_height=0.0,
            )


# ---------------------------------------------------------------- the carry
class TestCarry:
    def test_indices_bracket_the_closed_gripper(self):
        labels = demo_labels([0, 0, 0.9], [0.3, 0.1, 1.0])
        first, last = carry_indices(labels)
        assert labels.gripper[first] > 0 and labels.gripper[first - 1] < 0
        assert labels.gripper[last] > 0 and labels.gripper[last + 1] < 0

    def test_rejects_a_demonstration_that_never_grasps(self):
        labels = demo_labels([0, 0, 0.9], [0.3, 0.1, 1.0])
        labels.gripper = -np.ones_like(labels.gripper)
        with pytest.raises(ValueError, match="never closes"):
            carry_indices(labels)

    def test_recovers_the_motion_the_object_underwent(self):
        """Rigid attachment: the object's motion is the hand's motion."""
        labels = demo_labels([0, 0, 0.9], [0.3, 0.1, 1.0], place_yaw=0.6)
        rotation, translation = carry_transform(labels)
        g, r = carry_indices(labels)
        moved = apply_transform(labels.positions[g][None], rotation, translation)[0]
        assert np.allclose(moved, labels.positions[r])

    def test_placement_rotation_squares_the_target_with_the_receptacle(self):
        """Inheriting the carried *amount* would leave the target skewed."""
        source_place = task_frame([0, 0, 1], [0, 1, 0])
        target_pick = task_frame([0, 0, 1], yaw_matrix(0.9) @ [0, 1, 0])
        rotation = placement_rotation(source_place, target_pick)
        assert np.allclose(rotation @ target_pick, source_place, atol=1e-12)


# ------------------------------------------------------------ the whole set
class TestSceneKeypoints:
    def _scene(
        self,
        target_half=np.array([0.03, 0.05, 0.075]),
        target_angle=-1.1,
        target_grasp_fraction=1.0,
    ):
        shelf_source, shelf_target = 0.95, 1.02
        source_centre = np.array([-0.10, 0.05, TABLE + HALF[2]])
        source_points = box_cloud(source_centre, angle=0.4, faces="visible")
        source_dest = np.array([0.20, -0.13, shelf_source])
        source_grasp = GraspFrame(
            tcp=source_centre + [0, 0, HALF[2]],
            approach=[0, 0, -1],
            closing=yaw_matrix(0.4)[:, 0],
        )
        labels = demo_labels(
            source_grasp.tcp,
            [source_dest[0], source_dest[1], shelf_source + 2 * HALF[2]],
            grasp_yaw=0.4,
            place_yaw=0.0,
        )
        target_centre = np.array([-0.16, -0.06, TABLE + target_half[2]])
        target_points = box_cloud(
            target_centre, angle=target_angle, half=target_half, seed=3, faces="visible"
        )
        source = ObjectPlacement(source_points, source_grasp, TABLE, source_dest, shelf_source)
        target = ObjectPlacement(
            target_points,
            top_down_grasp(target_points, height_fraction=target_grasp_fraction),
            TABLE,
            np.array([0.20, 0.13, shelf_target]),
            shelf_target,
        )
        return source, target, labels, shelf_target

    def test_sets_are_paired_elementwise(self):
        source, target, labels, _ = self._scene()
        S, T, _ = scene_keypoints(source, target, labels)
        assert len(S) == len(T) == 18
        assert S.labels == T.labels
        left, right = pair_keypoints(S, T)
        assert left.shape == right.shape == (18, 3)

    def test_spans_three_dimensions(self):
        source, target, labels, _ = self._scene()
        S, T, _ = scene_keypoints(source, target, labels)
        assert not S.is_degenerate() and not T.is_degenerate()

    def test_target_rests_on_its_destination_surface(self):
        """Objects differ in height, so the placed height cannot be inherited."""
        source, target, labels, shelf_target = self._scene()
        _, T, _ = scene_keypoints(source, target, labels)
        for name in CORNER_NAMES[:4]:
            assert T.points[T.labels.index(f"place_{name}")][2] == pytest.approx(
                shelf_target, abs=1e-9
            )

    def test_target_is_placed_at_its_destination_slot(self):
        source, target, labels, _ = self._scene()
        _, T, _ = scene_keypoints(source, target, labels)
        centre = T.points[T.labels.index("place_center")]
        assert np.allclose(centre[:2], target.destination[:2], atol=2e-3)

    def test_the_map_it_produces_is_a_diffeomorphism_that_interpolates(self):
        # A moderate scene change. The frame the grasp defines differs by 23
        # degrees between the two objects, which is the regime the extractor is
        # built for; see the test below for what happens when it approaches 90.
        source, target, labels, _ = self._scene(target_angle=0.8)
        S, T, _ = scene_keypoints(source, target, labels)
        transport = TransportMap().fit(S.points, T.points)
        assert transport.keypoint_residual().max() < 1e-5
        report = transport.check_diffeomorphism(labels.positions)
        assert report.consistent_sign and report.fraction_positive == 1.0

    def test_a_near_perpendicular_grasp_axis_folds_the_map(self):
        """The measured limit of the construction.

        The map has to rotate the source object's frame onto the target's. When
        the two grasps sit nearly at right angles to each other, that rotation
        approaches 90 degrees while the *placed* configurations stay aligned
        with the shelf, so the pick and place ends demand very different
        rotations and ``phi`` folds between them.

        Kept as a test because it is a real boundary and it is invisible in the
        keypoint residual, which stays at micrometres either way.
        """
        source, target, labels, _ = self._scene(target_angle=-1.1)
        S, T, _ = scene_keypoints(source, target, labels)
        transport = TransportMap().fit(S.points, T.points)
        assert transport.keypoint_residual().max() < 1e-5
        report = transport.check_diffeomorphism(labels.positions)
        assert not report.consistent_sign

    def test_the_box_alone_stays_a_diffeomorphism_when_grasp_heights_differ(self):
        """Property (ii) of Sec. III-D survives a mismatched grasp height.

        The box keypoints describe the object's shape and the grasp's
        *orientation*; nothing in them depends on where along the object the
        jaws sit. So they cannot be made to contradict each other.
        """
        source, target, labels, _ = self._scene(
            target_angle=0.8, target_grasp_fraction=0.5
        )
        S, T, _ = scene_keypoints(source, target, labels, include_contacts=False)
        report = TransportMap().fit(S.points, T.points).check_diffeomorphism(
            labels.positions
        )
        assert report.consistent_sign and report.fraction_positive == 1.0

    @pytest.mark.parametrize(
        "fraction, folds", [(1.0, False), (0.5, True)]
    )
    def test_contact_keypoints_fold_the_map_when_grasp_heights_differ(
        self, fraction, folds
    ):
        """The measured limit of putting the grasp *position* into the keypoints.

        Contact keypoints ask ``phi`` to send the source's jaw contacts onto the
        target's. The box corners simultaneously ask it to send the source
        object onto the target object rigidly. When the two grasps sit at
        different heights on their objects those demands contradict each other,
        and because Sec. III-D interpolates every keypoint *exactly*, the map
        satisfies both by folding space in between -- ``det(J)`` changes sign,
        violating property (ii).

        Measured on an otherwise identical scene: a matched grasp height gives
        ``det(J)`` a minimum of 1.04, three-quarters height 0.44, and
        mid-height -0.07.

        This is why the grasp's position along the object belongs in the grasp
        itself, which already converts to an end-effector target through the
        verified frame contract, and not in the warp.
        """
        source, target, labels, _ = self._scene(
            target_angle=0.8, target_grasp_fraction=fraction
        )
        S, T, _ = scene_keypoints(source, target, labels, include_contacts=True)
        report = TransportMap().fit(S.points, T.points).check_diffeomorphism(
            labels.positions
        )
        assert (not report.consistent_sign) == folds

    def test_a_taller_target_lands_higher_not_deeper(self):
        """Regression on the vertical snap, across a 3x range of heights."""
        for height in (0.02, 0.075, 0.15):
            half = np.array([0.03, 0.05, height])
            source, target, labels, shelf_target = self._scene(target_half=half)
            _, T, _ = scene_keypoints(source, target, labels)
            centre = T.points[T.labels.index("place_center")]
            assert centre[2] == pytest.approx(shelf_target + height, abs=3e-3)
