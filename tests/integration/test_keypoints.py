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
