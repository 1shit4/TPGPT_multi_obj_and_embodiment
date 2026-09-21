"""Offline checks of the camera geometry. Run: python3 -m pytest test/ -q"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ycb_pointcloud_pipeline import camera_calibration as cc  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_calibration.yaml')
INTR = cc.Intrinsics(fx=604.7, fy=602.6, cx=425.9, cy=236.6, width=848, height=480)


def random_rotation(rng):
    q = rng.normal(size=4)
    return cc.quat_xyzw_to_matrix(q / np.linalg.norm(q))


def test_deproject_project_round_trip():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 848, 500)
    v = rng.uniform(0, 480, 500)
    z = rng.uniform(0.2, 2.0, 500)
    p = cc.deproject(u, v, z, INTR)
    uv, inside = cc.project(p, INTR)
    assert inside.all()
    assert np.allclose(uv[:, 0], u) and np.allclose(uv[:, 1], v)


def test_principal_point_is_on_the_optical_axis():
    p = cc.deproject(INTR.cx, INTR.cy, 1.3, INTR)
    assert np.allclose(p, [0, 0, 1.3])


def test_pixel_right_and_down_are_optical_plus_x_and_plus_y():
    p = cc.deproject(INTR.cx + 100, INTR.cy + 50, 1.0, INTR)
    assert p[0] > 0 and p[1] > 0


def test_link_optical_rotation_matches_realsense_rpy():
    R = cc.rpy_to_matrix(-math.pi / 2, 0.0, -math.pi / 2)
    assert np.allclose(R, cc.R_LINK_OPTICAL)
    # Optical z (viewing direction) is camera_link x (forward).
    assert np.allclose(cc.R_LINK_OPTICAL @ [0, 0, 1], [1, 0, 0])
    # Optical y (image down) is camera_link -z.
    assert np.allclose(cc.R_LINK_OPTICAL @ [0, 1, 0], [0, 0, -1])


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = random_rotation(rng)
        assert np.allclose(cc.quat_xyzw_to_matrix(cc.matrix_to_quat_xyzw(R)), R, atol=1e-9)


def test_transform_inverse():
    rng = np.random.default_rng(2)
    T = cc.make_transform(random_rotation(rng), rng.normal(size=3))
    p = rng.normal(size=(10, 3))
    assert np.allclose(cc.transform_points(cc.invert_transform(T), cc.transform_points(T, p)), p)


def test_kabsch_recovers_a_transform_and_reports_residuals():
    rng = np.random.default_rng(3)
    T = cc.make_transform(random_rotation(rng), [0.3, -0.2, 0.8])
    src = rng.uniform(-0.3, 0.3, (8, 3))
    dst = cc.transform_points(T, src) + rng.normal(scale=0.001, size=(8, 3))
    fit, res = cc.fit_rigid_transform(src, dst)
    assert np.allclose(fit[:3, 3], T[:3, 3], atol=0.005)
    assert cc.rotation_angle_deg(fit[:3, :3], T[:3, :3]) < 0.5
    assert res.max() < 0.005


def test_kabsch_refuses_collinear_points():
    src = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match='collinear'):
        cc.fit_rigid_transform(src, src)


def test_plane_fit_on_a_tilted_noisy_table():
    rng = np.random.default_rng(4)
    xy = rng.uniform(-0.5, 0.5, (3000, 2))
    tilt = math.radians(2.0)
    z = 0.02 + xy[:, 0] * math.tan(tilt) + rng.normal(scale=0.001, size=3000)
    clutter = rng.uniform(-0.5, 0.5, (400, 3))
    n, d, inliers = cc.fit_plane_ransac(np.vstack([np.c_[xy, z], clutter]))
    angle = math.degrees(math.acos(n @ [0, 0, 1]))
    assert angle == pytest.approx(2.0, abs=0.2)
    assert d == pytest.approx(0.02, abs=0.002)
    assert inliers[:3000].mean() > 0.9


def test_depth_image_to_points_matches_deproject():
    depth = np.zeros((480, 848), dtype=np.uint16)
    depth[100, 200] = 1500
    pts, pix = cc.depth_image_to_points(depth, INTR, depth_scale=1000.0)
    assert len(pts) == 1 and tuple(pix[0]) == (100, 200)
    assert np.allclose(pts[0], cc.deproject(200, 100, 1.5, INTR))


def test_voxel_downsample_and_crop():
    pts = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.5, 0.5, 0.5]])
    assert len(cc.voxel_downsample(pts, 0.01)) == 2
    kept, mask = cc.crop_box(pts, [-0.1] * 3, [0.1] * 3)
    assert len(kept) == 2 and mask.tolist() == [True, True, False]


PLACEHOLDER = {
    'base_to_camera': {'child_frame': 'camera_link', 'translation': [1.0, 0.0, 0.6],
                       'rotation_rpy_deg': [0.0, 45.0, 180.0], 'is_placeholder': True},
    'camera_link_to_optical': {'source': 'tf', 'translation': [0.0, 0.015, 0.0],
                               'rotation_rpy_deg': [-90.0, 0.0, -90.0]},
    'intrinsics': {'source': 'camera_info', 'fx': 604.7, 'fy': 602.6, 'cx': 425.9,
                   'cy': 236.6, 'width': 848, 'height': 480},
}


def test_camera_link_calibration_chains_frames():
    cal = cc.CameraCalibration.from_dict(PLACEHOLDER)
    assert cal.base_frame == 'base' and cal.camera_child_frame == 'camera_link'
    assert cal.camera_is_placeholder
    # With the yaml optical transform: optical z (view direction) = camera_link x.
    T = cal.T_base_optical(cal.T_link_optical_yaml)
    assert np.allclose(T[:3, :3] @ [0, 0, 1], cal.T_base_camera[:3, :3] @ [1, 0, 0])
    view = T[:3, :3] @ [0, 0, 1]
    assert view[0] < 0 and view[2] < 0
    with pytest.raises(ValueError, match='TF'):
        cal.T_base_optical()   # source: tf without a TF transform


def test_shipped_calibration_is_base_to_optical_and_sees_the_table():
    """The calibration in config/ is to the optical frame: its viewing axis must meet
    the table in front of the robot, and the robot base must be in the image."""
    cal = cc.CameraCalibration.load(CONFIG)
    assert cal.child_is_optical and not cal.camera_is_placeholder
    # Touched with the closed fingertip face on 2026-09-21 (three points, 0.2 mm
    # spread). The earlier -0.006 predated the camera/shelf move.
    assert cal.expected_table_height == pytest.approx(-0.0136)
    T = cal.T_base_optical()
    ray, origin = T[:3, :3] @ [0, 0, 1], T[:3, 3]
    s = (cal.expected_table_height - origin[2]) / ray[2]
    hit = origin + s * ray
    assert s > 0 and 0.2 < hit[0] < 0.8 and abs(hit[1]) < 0.3
    _, inside = cc.project(cc.transform_points(cc.invert_transform(T), [0, 0, 0]),
                           cal.intrinsics_yaml)
    assert inside[0]


def test_a_point_seen_by_the_camera_lands_where_it_is_in_base():
    """End to end: place a point in base, render it to a pixel + depth, deproject,
    transform back. This is the chain segmentation_node + planner apply."""
    cal = cc.CameraCalibration.from_dict(PLACEHOLDER)
    T_base_opt = cal.T_base_optical(cal.T_link_optical_yaml)
    p_base = np.array([0.5, 0.1, 0.05])
    p_opt = cc.transform_points(cc.invert_transform(T_base_opt), p_base)[0]
    uv, inside = cc.project(p_opt, INTR)
    assert inside[0]
    back = cc.transform_points(T_base_opt, cc.deproject(uv[0, 0], uv[0, 1], p_opt[2], INTR))[0]
    assert np.allclose(back, p_base, atol=1e-9)


def test_child_optical_frame_skips_the_link_transform():
    data = {
        'base_to_camera': {'child_frame': 'camera_color_optical_frame',
                           'translation': [1, 2, 3], 'rotation_rpy_deg': [0, 0, 90]},
        'intrinsics': {'source': 'yaml', 'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0,
                       'width': 2, 'height': 2},
    }
    cal = cc.CameraCalibration.from_dict(data)
    assert np.allclose(cal.T_base_optical(), cal.T_base_camera)


def test_intrinsics_comparison_flags_resolution_change():
    other = cc.Intrinsics(fx=604.7, fy=602.6, cx=319.9, cy=236.6, width=640, height=480)
    worst, text = INTR.compare(other)
    assert worst == float('inf') and 'resolution' in text


def test_yaml_snippet_round_trips():
    rng = np.random.default_rng(5)
    T_link_opt = cc.make_transform(cc.R_LINK_OPTICAL, [0, 0.015, 0])
    T_base_opt = cc.make_transform(random_rotation(rng), [0.9, 0.1, 0.5])
    import yaml
    block = yaml.safe_load(cc.camera_link_yaml_snippet(T_base_opt, T_link_opt))
    cal = cc.CameraCalibration.from_dict({**block, 'camera_link_to_optical': {
        'source': 'yaml', 'translation': [0, 0.015, 0], 'rotation_rpy_deg': [-90, 0, -90]}})
    assert np.allclose(cal.T_base_optical(), T_base_opt, atol=1e-5)
