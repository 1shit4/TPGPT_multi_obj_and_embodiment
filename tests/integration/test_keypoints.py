"""Keypoint task parameterisation -- paper Sec. III-A, Sec. V-A."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tpgpt.sim.keypoints import CUBE_CORNERS, KeypointSet, cube_keypoints, pair_keypoints


def test_cube_keypoints_have_centre_and_eight_corners():
    """Sec. V-A: at least 3 non-collinear points are needed for a 3-D pose."""
    kp = cube_keypoints([0.1, 0.2, 0.8], half_extent=0.02, name="box")
    assert len(kp) == 9
    assert kp.labels[0] == "box_center"
    assert not kp.is_degenerate()


def test_orientation_changes_the_keypoints():
    """The prototype derived keypoints from position alone, so a rotated object
    produced an identical set and its rotation was invisible to the map."""
    R = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    upright = cube_keypoints([0.1, 0.2, 0.8], None, 0.02, name="box")
    rotated = cube_keypoints([0.1, 0.2, 0.8], R, 0.02, name="box")
    assert not np.allclose(upright.points, rotated.points)
    # The centre is unmoved; only the corners rotate.
    assert np.allclose(upright.points[0], rotated.points[0])


def test_keypoints_follow_the_pose_exactly():
    R = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
    position = np.array([0.3, -0.1, 0.9])
    kp = cube_keypoints(position, R, 0.025, include_center=False)
    assert np.allclose(kp.points, position + (CUBE_CORNERS * 0.025) @ R.T)


def test_flat_keypoints_are_detected_as_degenerate():
    """A degenerate set leaves the affine rotation undetermined (Sec. III-E-a)."""
    assert cube_keypoints([0, 0, 0], None, [0.02, 0.02, 0.0]).is_degenerate()


def test_pairing_is_restored_from_labels():
    a = cube_keypoints([0.1, 0.2, 0.8], None, 0.02, name="box")
    b = cube_keypoints([0.4, 0.0, 1.0], None, 0.02, name="box")
    shuffled = KeypointSet(points=b.points[::-1], labels=b.labels[::-1])
    _, T = pair_keypoints(a, shuffled)
    assert np.allclose(T, b.points)


def test_pairing_rejects_mismatched_label_sets():
    a = cube_keypoints([0, 0, 0], name="box")
    b = cube_keypoints([0, 0, 0], name="other")
    with pytest.raises(ValueError, match="label sets differ"):
        pair_keypoints(a, b)


def test_roundtrip_through_disk(tmp_path):
    kp = cube_keypoints([0.1, 0.2, 0.8], None, 0.02, name="box")
    loaded = KeypointSet.load(kp.save(tmp_path / "kp.json"))
    assert np.allclose(loaded.points, kp.points)
    assert loaded.labels == kp.labels


def test_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="labels for"):
        KeypointSet(points=np.zeros((3, 3)), labels=["a", "b"])


# ---------------------------------------------------------------------------
# Grasp-aligned extraction on real segmented clouds (paper Sec. III-A)
# ---------------------------------------------------------------------------


@pytest.mark.sim
class TestGraspAlignedOnRealClouds:
    """The extractor against clouds the perception stack actually produces.

    The unit tests use synthetic box clouds so the keypoint maths is isolated.
    These check the two things only a real cloud exhibits: truncated lowest
    points, and objects whose segmentation is too thin to describe them.
    """

    @pytest.fixture(scope="class")
    def scene(self):
        from tpgpt.experiments.run_keypoint_transport import build_scene

        env = build_scene(seed=0)
        try:
            yield env, env._get_observations()
        finally:
            env.close()

    def test_real_clouds_are_truncated_above_the_table(self, scene):
        """Why the support snap exists, measured rather than assumed."""
        from tpgpt.experiments.run_keypoint_transport import table_surface
        from tpgpt.perception.cameras import object_point_cloud

        env, obs = scene
        surface = table_surface(env)
        gaps = []
        for name in ("cereal", "milk", "can", "bread"):
            cloud = object_point_cloud(env, name, obs=obs)
            gaps.append(float(cloud.points[:, 2].min()) - surface)
        # Every object floats above the surface it demonstrably rests on.
        assert all(gap > 0 for gap in gaps)
        assert max(gaps) < 0.03

    def test_snapped_box_reaches_the_surface_the_object_rests_on(self, scene):
        from tpgpt.experiments.run_keypoint_transport import table_surface
        from tpgpt.perception.cameras import object_point_cloud
        from tpgpt.sim.keypoints import CORNER_NAMES, object_keypoints, top_down_grasp

        env, obs = scene
        surface = table_surface(env)
        cloud = object_point_cloud(env, "cereal", obs=obs)
        keypoints = object_keypoints(
            cloud.points, top_down_grasp(cloud.points), support_height=surface, name="o"
        )
        for corner in CORNER_NAMES[:4]:
            assert keypoints.points[keypoints.labels.index(f"o_{corner}")][2] == pytest.approx(
                surface, abs=1e-9
            )

    def test_transport_onto_every_object_stays_a_diffeomorphism(self, scene):
        from tpgpt.experiments.reshelving_pipeline import record_source_placement
        from tpgpt.experiments.run_keypoint_transport import target_placement, transport

        env, obs = scene
        labels, source, ok = record_source_placement(0)
        assert ok
        for name in ("cereal", "milk", "can", "bread"):
            target, _ = target_placement(env, name, "top_middle", obs=obs)
            result = transport(labels, source, target)
            assert result["fraction_positive"] == 1.0, name
            assert result["keypoint_residual"] < 1e-5, name
            # The hand arrives at the corresponding point on a new object.
            assert result["grasp_error"] < 0.01, name
            # And at the destination surface, not above or through it.
            assert abs(
                result["release_clearance"] - result["expected_clearance"]
            ) < 0.005, name

    def test_a_cloud_too_thin_to_describe_its_object_is_visible_as_such(self, scene):
        """The lemon: 12 points, a box a quarter of its true width, all green.

        Guards the finding rather than the behaviour. Every map diagnostic
        passes on this object, so nothing downstream can detect it -- only the
        cloud size can.
        """
        from tpgpt.experiments.run_keypoint_transport import (
            SPARSE_CLOUD_POINTS,
            target_placement,
        )

        env, obs = scene
        target, points = target_placement(env, "lemon", "top_middle", obs=obs)
        assert len(points) < SPARSE_CLOUD_POINTS
