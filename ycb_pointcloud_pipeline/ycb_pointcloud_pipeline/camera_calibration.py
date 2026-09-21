"""
Camera calibration: intrinsics, robot-base-to-camera extrinsics, and the
geometry used to verify them. Pure NumPy + PyYAML -- no ROS imports -- so every
function here is unit-tested offline (test/test_camera_calibration.py).

Frames (RealSense ROS convention)
---------------------------------
    base                          robot base (CRISP base_frame)
      |  T_base_camera_link       <-- PLACEHOLDER, from your hand-eye calibration
    camera_link                   x forward, y left, z up (body of the camera)
      |  T_link_optical           <-- from the realsense driver's /tf_static
    camera_color_optical_frame    z forward, x right, y down (image axes)

The segmentation node deprojects masked depth pixels with the colour stream's
intrinsics, so object clouds arrive in camera_color_optical_frame. A point in
base is therefore

    p_base = T_base_camera_link @ T_link_optical @ p_optical

If your hand-eye calibration already gives base -> camera_color_optical_frame,
set `child_frame: camera_color_optical_frame` in the YAML and the second
transform is skipped.

Deprojection (pinhole, no distortion -- RealSense colour is pre-rectified):
    x = (u - cx) * z / fx,   y = (v - cy) * z / fy
where u is the pixel column and v the pixel row, and z the metric depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import yaml

OPTICAL_FRAME_SUFFIX = '_optical_frame'


# --------------------------------------------------------------------------- rotations
def quat_xyzw_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError('quaternion has zero length')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.asarray(q)
    return q if q[3] >= 0 else -q


def rpy_to_matrix(roll, pitch, yaw):
    """ROS / URDF convention: fixed-axis X then Y then Z, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_points(T, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R_a, R_b):
    rel = np.asarray(R_a).T @ np.asarray(R_b)
    return math.degrees(math.acos(float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))


#: camera_link -> camera_*_optical_frame rotation. The optical frame's z is the
#: link's x (forward), its x is the link's -y (right), its y is the link's -z
#: (down). Equals rpy (-pi/2, 0, -pi/2), what realsense2_camera publishes.
R_LINK_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


# --------------------------------------------------------------------------- intrinsics
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = 'plumb_bob'
    distortion: list = field(default_factory=lambda: [0.0] * 5)

    @classmethod
    def from_camera_info_fields(cls, k, width, height, distortion_model='', d=()):
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                   width=int(width), height=int(height),
                   distortion_model=str(distortion_model), distortion=[float(v) for v in d])

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def has_distortion(self, tol=1e-6):
        return any(abs(v) > tol for v in self.distortion)

    def compare(self, other):
        """Largest difference in pixels between two sets of intrinsics, and a text report."""
        diffs = {'fx': self.fx - other.fx, 'fy': self.fy - other.fy,
                 'cx': self.cx - other.cx, 'cy': self.cy - other.cy}
        size = (self.width, self.height) == (other.width, other.height)
        worst = max(abs(v) for v in diffs.values())
        text = ', '.join(f'{k} {v:+.2f}px' for k, v in diffs.items())
        if not size:
            text += (f'; resolution {self.width}x{self.height} vs '
                     f'{other.width}x{other.height} (intrinsics are resolution specific!)')
            worst = float('inf')
        return worst, text


def deproject(u, v, z, intr):
    """Pixel (u = column, v = row) with metric depth z -> optical-frame point(s)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=-1)


def project(points_optical, intr):
    """Optical-frame points -> (u, v) pixels and a mask of points in front and in the image."""
    p = np.asarray(points_optical, dtype=float).reshape(-1, 3)
    z = p[:, 2]
    front = z > 1e-6
    safe = np.where(front, z, 1.0)
    u = intr.fx * p[:, 0] / safe + intr.cx
    v = intr.fy * p[:, 1] / safe + intr.cy
    inside = front & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return np.stack([u, v], axis=1), inside


def depth_image_to_points(depth, intr, depth_scale=1000.0, depth_trunc=3.0, stride=1, mask=None):
    """Aligned depth image -> (N, 3) optical-frame points and their (row, col) pixels."""
    depth = np.asarray(depth)
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[rows, cols].astype(np.float32) / float(depth_scale)
    valid = (z > 0) & (z <= depth_trunc)
    if mask is not None:
        valid &= np.asarray(mask)[rows, cols]
    rows, cols, z = rows[valid], cols[valid], z[valid]
    return deproject(cols, rows, z, intr), np.stack([rows, cols], axis=1)


# --------------------------------------------------------------------------- calibration file
def _rotation_from_yaml(block, name):
    if 'rotation_quaternion_xyzw' in block and block['rotation_quaternion_xyzw'] is not None:
        return quat_xyzw_to_matrix(block['rotation_quaternion_xyzw'])
    if 'rotation_rpy_deg' in block and block['rotation_rpy_deg'] is not None:
        r, p, y = (math.radians(float(v)) for v in block['rotation_rpy_deg'])
        return rpy_to_matrix(r, p, y)
    if 'rotation_rpy_rad' in block and block['rotation_rpy_rad'] is not None:
        return rpy_to_matrix(*(float(v) for v in block['rotation_rpy_rad']))
    raise ValueError(f'{name}: give rotation_quaternion_xyzw, rotation_rpy_deg or rotation_rpy_rad')


def _transform_from_yaml(block, name):
    t = block.get('translation')
    if t is None or len(t) != 3:
        raise ValueError(f'{name}: translation must be [x, y, z] in metres')
    R = _rotation_from_yaml(block, name)
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError(f'{name}: rotation is not a proper rotation')
    return make_transform(R, t)


@dataclass
class CameraCalibration:
    """Everything loaded from config/camera_calibration.yaml."""

    base_frame: str
    camera_child_frame: str
    T_base_camera: np.ndarray                 # base -> camera_child_frame
    camera_is_placeholder: bool
    link_to_optical_source: str               # 'tf' | 'yaml'
    optical_frame: str
    T_link_optical_yaml: np.ndarray
    intrinsics_source: str                    # 'camera_info' | 'yaml'
    intrinsics_yaml: Intrinsics
    intrinsics_is_placeholder: bool
    depth_scale: float = 1000.0
    expected_table_height: float | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get('camera_calibration', data))

    @classmethod
    def from_dict(cls, c):
        ext = c['base_to_camera']
        opt = c.get('camera_link_to_optical', {}) or {}
        intr = c.get('intrinsics', {}) or {}
        ver = c.get('verification', {}) or {}
        source = str(opt.get('source', 'tf')).lower()
        if source not in ('tf', 'yaml'):
            raise ValueError("camera_link_to_optical.source must be 'tf' or 'yaml'")
        isource = str(intr.get('source', 'camera_info')).lower()
        if isource not in ('camera_info', 'yaml'):
            raise ValueError("intrinsics.source must be 'camera_info' or 'yaml'")
        table = ver.get('expected_table_height')
        return cls(
            base_frame=str(c.get('base_frame', ext.get('parent_frame', 'base'))),
            camera_child_frame=str(ext.get('child_frame', 'camera_link')),
            T_base_camera=_transform_from_yaml(ext, 'base_to_camera'),
            camera_is_placeholder=bool(ext.get('is_placeholder', True)),
            link_to_optical_source=source,
            optical_frame=str(opt.get('child_frame', 'camera_color_optical_frame')),
            T_link_optical_yaml=(_transform_from_yaml(opt, 'camera_link_to_optical')
                                 if opt else make_transform(R_LINK_OPTICAL)),
            intrinsics_source=isource,
            intrinsics_yaml=Intrinsics(
                fx=float(intr.get('fx', 0.0)), fy=float(intr.get('fy', 0.0)),
                cx=float(intr.get('cx', 0.0)), cy=float(intr.get('cy', 0.0)),
                width=int(intr.get('width', 0)), height=int(intr.get('height', 0)),
                distortion_model=str(intr.get('distortion_model', 'plumb_bob')),
                distortion=[float(v) for v in intr.get('distortion', [0.0] * 5)],
            ),
            intrinsics_is_placeholder=bool(intr.get('is_placeholder', True)),
            depth_scale=float(c.get('depth_scale', 1000.0)),
            expected_table_height=None if table is None else float(table),
            raw=dict(c),
        )

    @property
    def child_is_optical(self):
        return self.camera_child_frame.endswith(OPTICAL_FRAME_SUFFIX)

    def T_base_optical(self, T_link_optical=None):
        """base -> optical frame. `T_link_optical` comes from TF when source is 'tf'."""
        if self.child_is_optical:
            return self.T_base_camera.copy()
        if T_link_optical is None:
            if self.link_to_optical_source == 'tf':
                raise ValueError('camera_link_to_optical.source is tf but no TF transform was '
                                 'supplied; is realsense2_camera running?')
            T_link_optical = self.T_link_optical_yaml
        return self.T_base_camera @ T_link_optical

    def intrinsics(self, camera_info_intrinsics=None):
        if self.intrinsics_source == 'yaml':
            return self.intrinsics_yaml
        if camera_info_intrinsics is None:
            raise ValueError('intrinsics.source is camera_info but none was received yet')
        return camera_info_intrinsics


def camera_link_yaml_snippet(T_base_optical, T_link_optical, child_frame='camera_link'):
    """A base_to_camera YAML block for a fitted base->optical transform."""
    T = T_base_optical @ invert_transform(T_link_optical)
    q = matrix_to_quat_xyzw(T[:3, :3])
    t = T[:3, 3]
    return (
        'base_to_camera:\n'
        '  parent_frame: base\n'
        f'  child_frame: {child_frame}\n'
        f'  translation: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]\n'
        f'  rotation_quaternion_xyzw: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n'
        '  is_placeholder: false\n'
    )


# --------------------------------------------------------------------------- cloud geometry
def crop_box(points, lo, hi):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = np.all((points >= np.asarray(lo)) & (points <= np.asarray(hi)), axis=1)
    return points[keep], keep


def voxel_downsample(points, voxel):
    """One point (the centroid) per occupied voxel."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def radius_outlier_filter(points, radius=0.01, min_neighbors=5):
    """Drop points with fewer than `min_neighbors` others within `radius` (grid hashed)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) <= min_neighbors:
        return points
    keys = np.floor(points / radius).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    keep = np.zeros(len(points), dtype=bool)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    r2 = radius * radius
    for i, k in enumerate(map(tuple, keys)):
        idx = [j for o in offsets for j in cells.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ())]
        d2 = np.sum((points[idx] - points[i]) ** 2, axis=1)
        keep[i] = int(np.count_nonzero(d2 <= r2)) - 1 >= min_neighbors
    return points[keep]


def fit_plane_ransac(points, threshold=0.005, iterations=300, seed=0, up=(0.0, 0.0, 1.0)):
    """Dominant plane. Returns (unit normal oriented towards `up`, offset d, inlier mask),
    with the plane n . p = d."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError('need at least 3 points to fit a plane')
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inliers = np.abs((points - a) @ n) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best is None:
        raise ValueError('all samples were degenerate')
    # Least-squares refinement on the inliers.
    inl = points[best]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[-1]
    if n @ np.asarray(up) < 0:
        n = -n
    d = float(n @ centroid)
    inliers = np.abs(points @ n - d) < threshold
    return n, d, inliers


def fit_rigid_transform(source, target):
    """Kabsch: T minimising sum |T @ source_i - target_i|^2. Returns (T, residuals)."""
    S = np.asarray(source, dtype=float).reshape(-1, 3)
    D = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(S) < 3 or len(S) != len(D):
        raise ValueError('need at least 3 paired points')
    cs, cd = S.mean(axis=0), D.mean(axis=0)
    H = (S - cs).T @ (D - cd)
    U, sing, Vt = np.linalg.svd(H)
    if sing[1] < 1e-9 * max(sing[0], 1e-12):
        raise ValueError('points are collinear; the rotation is undetermined')
    sign = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    T = make_transform(R, cd - R @ cs)
    residuals = np.linalg.norm(transform_points(T, S) - D, axis=1)
    return T, residuals
