"""ROS 2 helpers shared by the FR3 and TPGPT hardware nodes: PointCloud2 <-> NumPy,
pose messages, TF, RViz markers, the calibration provider that turns an object
cloud in the camera optical frame into the robot base frame, object cloud fusion,
the controller_manager client (load / configure / strict switch), and the client
for the Inspire hand's open/close services."""

import os
import time

import numpy as np
from builtin_interfaces.msg import Duration
from rclpy.time import Time
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

from ycb_pointcloud_pipeline import camera_calibration as cc

_DTYPES = {
    PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
    PointField.INT32: np.int32, PointField.UINT32: np.uint32,
    PointField.INT16: np.int16, PointField.UINT16: np.uint16,
    PointField.INT8: np.int8, PointField.UINT8: np.uint8,
}


# --------------------------------------------------------------------------- clouds
def pointcloud2_to_xyz(msg):
    """(N, 3) float64 xyz from a PointCloud2, dropping non-finite points."""
    fields = {f.name: f for f in msg.fields}
    if not {'x', 'y', 'z'} <= set(fields):
        raise ValueError('PointCloud2 has no x/y/z fields')
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3))
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [np.dtype(_DTYPES[fields[k].datatype]).newbyteorder(
            '>' if msg.is_bigendian else '<') for k in 'xyz'],
        'offsets': [fields[k].offset for k in 'xyz'],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def pointcloud2_to_xyzrgb(msg):
    """(xyz (N, 3) float64, rgb (N, 3) uint8 or None) from a PointCloud2 with a packed
    `rgb` field, dropping non-finite points."""
    fields = {f.name: f for f in msg.fields}
    n = msg.width * msg.height
    if 'rgb' not in fields or n == 0:
        return pointcloud2_to_xyz(msg), None
    names = ['x', 'y', 'z', 'rgb']
    dtype = np.dtype({
        'names': names,
        'formats': [np.dtype(_DTYPES[fields[k].datatype]).newbyteorder(
            '>' if msg.is_bigendian else '<') for k in names],
        'offsets': [fields[k].offset for k in names],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    packed = np.ascontiguousarray(arr['rgb'], dtype=np.float32).view(np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(np.uint8)
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok], rgb[ok]


def xyz_to_pointcloud2(points, frame_id, stamp, rgb=None):
    """PointCloud2 with x, y, z (+ packed rgb when `rgb` is a (3,) or (N, 3) uint8)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(points)
    fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, k in enumerate('xyz')]
    step = 12
    if rgb is not None:
        colors = np.broadcast_to(np.asarray(rgb, dtype=np.uint8), (n, 3))
        packed = ((colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8)
                  | colors[:, 2].astype(np.uint32)).view(np.float32)
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        step = 16
        data = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        data['rgb'] = packed
    else:
        data = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width = 1, n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step, msg.row_step = step, step * n
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


# --------------------------------------------------------------------------- poses
def pose_to_matrix(pose):
    p, o = pose.position, pose.orientation
    return cc.make_transform(cc.quat_xyzw_to_matrix([o.x, o.y, o.z, o.w]), [p.x, p.y, p.z])


def transform_msg_to_matrix(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return cc.make_transform(cc.quat_xyzw_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])


def matrix_to_pose_stamped(position, rotation, frame_id, stamp):
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = frame_id, stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in position)
    q = cc.matrix_to_quat_xyzw(rotation)
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = (float(v) for v in q)
    return msg


def matrix_to_transform_stamped(T, parent, child, stamp):
    msg = TransformStamped()
    msg.header.frame_id, msg.header.stamp, msg.child_frame_id = parent, stamp, child
    t = T[:3, 3]
    q = cc.matrix_to_quat_xyzw(T[:3, :3])
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
        float(v) for v in t)
    (msg.transform.rotation.x, msg.transform.rotation.y,
     msg.transform.rotation.z, msg.transform.rotation.w) = (float(v) for v in q)
    return msg


# --------------------------------------------------------------------------- markers
def _color(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _pt(p):
    return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


def _marker(ns, mid, mtype, frame_id, stamp):
    m = Marker()
    m.header.frame_id, m.header.stamp = frame_id, stamp
    m.ns, m.id, m.type, m.action = ns, int(mid), mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    m.lifetime = Duration(sec=0)
    return m


def axes_markers(ns, first_id, position, rotation, frame_id, stamp, length=0.08, width=0.006):
    """Three arrows: red = x, green = y, blue = z of `rotation` at `position`."""
    markers = []
    for i, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        m = _marker(ns, first_id + i, Marker.ARROW, frame_id, stamp)
        m.points = [_pt(position), _pt(np.asarray(position) + length * np.asarray(rotation)[:, i])]
        m.scale.x, m.scale.y, m.scale.z = width, 2 * width, 2 * width
        m.color = _color(*color)
        markers.append(m)
    return markers


def sphere_list_marker(ns, mid, points, frame_id, stamp, color, diameter=0.012, colors=None):
    m = _marker(ns, mid, Marker.SPHERE_LIST, frame_id, stamp)
    m.points = [_pt(p) for p in np.asarray(points).reshape(-1, 3)]
    m.scale.x = m.scale.y = m.scale.z = diameter
    m.color = _color(*color)
    if colors is not None:
        m.colors = [_color(*c) for c in colors]
    return m


def line_strip_marker(ns, mid, points, frame_id, stamp, colors=None, color=(1, 1, 1), width=0.004):
    m = _marker(ns, mid, Marker.LINE_STRIP, frame_id, stamp)
    m.points = [_pt(p) for p in np.asarray(points).reshape(-1, 3)]
    m.scale.x = width
    m.color = _color(*color)
    if colors is not None:
        m.colors = [_color(*c) for c in colors]
    return m


def line_list_marker(ns, mid, segments, frame_id, stamp, color=(1, 1, 1), width=0.002,
                     colors=None):
    """Independent segments; `segments` is (K, 2, 3), `colors` one per segment."""
    m = _marker(ns, mid, Marker.LINE_LIST, frame_id, stamp)
    segments = np.asarray(segments).reshape(-1, 2, 3)
    m.points = [_pt(p) for seg in segments for p in seg]
    m.scale.x = width
    m.color = _color(*color)
    if colors is not None:
        m.colors = [_color(*c) for c in colors for _ in range(2)]
    return m


def gripper_segments(tcp, approach, closing, opening=0.08, finger=0.045, stem=0.06):
    """Wireframe of a parallel gripper at its fingertip point: two fingers, the
    palm between them, a stem back along the approach and a tick at the TCP.
    (5, 2, 3)."""
    tcp, a, c = (np.asarray(v, dtype=float) for v in (tcp, approach, closing))
    left, right = tcp + c * opening / 2.0, tcp - c * opening / 2.0
    lb, rb = left - a * finger, right - a * finger
    mid = tcp - a * finger
    return np.array([[left, lb], [right, rb], [lb, rb], [mid, mid - a * stem],
                     [tcp - c * 0.004, tcp + c * 0.004]])


def text_marker(ns, mid, position, text, frame_id, stamp, height=0.025, color=(1, 1, 1)):
    m = _marker(ns, mid, Marker.TEXT_VIEW_FACING, frame_id, stamp)
    m.pose.position = _pt(position)
    m.scale.z = height
    m.text = text
    m.color = _color(*color)
    return m


def cylinder_marker(ns, mid, position, diameter, height, frame_id, stamp, color):
    m = _marker(ns, mid, Marker.CYLINDER, frame_id, stamp)
    m.pose.position = _pt(position)
    m.scale.x = m.scale.y = diameter
    m.scale.z = height
    m.color = _color(*color)
    return m


def box_marker(ns, mid, centre, rotation, size, frame_id, stamp, color):
    """An oriented box: ``size`` is the full extent along the box's own axes."""
    m = _marker(ns, mid, Marker.CUBE, frame_id, stamp)
    m.pose.position = _pt(centre)
    q = cc.matrix_to_quat_xyzw(rotation)
    m.pose.orientation.x, m.pose.orientation.y = float(q[0]), float(q[1])
    m.pose.orientation.z, m.pose.orientation.w = float(q[2]), float(q[3])
    m.scale.x, m.scale.y, m.scale.z = (float(v) for v in size)
    m.color = _color(*color)
    return m


# --------------------------------------------------------------------------- calibration
class CalibrationProvider:
    """Resolves base <- optical-frame transforms for a node, from the calibration YAML
    plus (when configured) the RealSense driver's camera_link -> optical TF."""

    def __init__(self, node, yaml_path, tf_buffer=None):
        self.node = node
        self.cal = cc.CameraCalibration.load(yaml_path)
        self.tf_buffer = tf_buffer
        self._cache = {}
        self._reported = set()
        log = node.get_logger()
        if self.cal.camera_is_placeholder:
            log.warn(f'{yaml_path}: base_to_camera is a PLACEHOLDER (is_placeholder: true). '
                     'Clouds in the base frame will be wrong until it is calibrated.')
        if self.cal.intrinsics_source == 'yaml' and self.cal.intrinsics_is_placeholder:
            log.warn('intrinsics.source is yaml and the values are marked as a PLACEHOLDER.')

    def T_base_optical(self, optical_frame):
        """base <- optical_frame, or None if the TF it needs is not available yet."""
        cal = self.cal
        if cal.child_is_optical:
            if optical_frame != cal.camera_child_frame:
                self._once(f'mismatch-{optical_frame}',
                           f'cloud frame {optical_frame!r} differs from the calibrated '
                           f'optical frame {cal.camera_child_frame!r}; using the calibration '
                           'anyway -- check the frame names')
            return cal.T_base_camera
        if cal.link_to_optical_source == 'yaml':
            return cal.T_base_optical(cal.T_link_optical_yaml)
        if optical_frame in self._cache:
            return cal.T_base_optical(self._cache[optical_frame])
        if self.tf_buffer is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(cal.camera_child_frame, optical_frame,
                                                 Time())
        except Exception as exc:  # tf2 raises several exception types
            self._once(f'tf-{optical_frame}',
                       f'waiting for TF {cal.camera_child_frame} -> {optical_frame} from '
                       f'realsense2_camera ({type(exc).__name__}); set '
                       'camera_link_to_optical.source: yaml if the driver does not publish it')
            return None
        T = transform_msg_to_matrix(tf)
        self._cache[optical_frame] = T
        nominal = cal.T_link_optical_yaml
        self.node.get_logger().info(
            f'TF {cal.camera_child_frame} -> {optical_frame}: translation '
            f'{np.round(T[:3, 3], 4).tolist()} m; differs from the YAML nominal by '
            f'{np.linalg.norm(T[:3, 3] - nominal[:3, 3]) * 1000:.1f} mm and '
            f'{cc.rotation_angle_deg(T[:3, :3], nominal[:3, :3]):.2f} deg')
        return cal.T_base_optical(T)

    def _once(self, key, text):
        if key not in self._reported:
            self._reported.add(key)
            self.node.get_logger().warn(text)


# --------------------------------------------------------------------------- clouds, fused
def colors_for(points, raw_points, raw_colors):
    """Colour each fused point with its nearest raw point's camera colour."""
    from scipy.spatial import cKDTree
    if raw_colors is None or len(raw_points) == 0 or len(points) == 0:
        return None
    _, idx = cKDTree(raw_points).query(points)
    return raw_colors[idx]


def fuse_clouds(clouds, workspace_min, workspace_max, voxel_size, outlier_radius,
                outlier_min_neighbors, max_points):
    """One object cloud from several base-frame clouds: stack, crop to the workspace
    box, voxel-downsample, drop isolated points, cap the size (GraspGen-X and the
    keypoint fit both assume at most 8192). Returns (points, stats)."""
    pts = np.vstack([np.asarray(c, dtype=float).reshape(-1, 3) for c in clouds])
    stats = {'n_clouds': len(clouds), 'n_raw': len(pts)}
    pts, _ = cc.crop_box(pts, workspace_min, workspace_max)
    stats['n_in_workspace'] = len(pts)
    pts = cc.voxel_downsample(pts, voxel_size)
    pts = cc.radius_outlier_filter(pts, outlier_radius, int(outlier_min_neighbors))
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), int(max_points), replace=False)
        pts = pts[idx]
    stats['n_fused'] = len(pts)
    return pts, stats


# --------------------------------------------------------------------------- controllers
JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
CONTAINER_WS = '/workspaces/thor_ws'


def container_config_path(filename, package='ycb_pointcloud_pipeline'):
    """Path of this package's config/<filename> as the robot container sees it
    (THOR's run.sh mounts ~/thor_ws at /workspaces/thor_ws). Raises ValueError if
    the installed file does not resolve into ~/thor_ws (build with --symlink-install)."""
    from ament_index_python.packages import get_package_share_directory
    host_file = os.path.realpath(os.path.join(get_package_share_directory(package), 'config',
                                              filename))
    host_ws = os.path.realpath(os.path.expanduser('~/thor_ws'))
    if not host_file.startswith(host_ws + os.sep):
        raise ValueError(f'{host_file} is outside ~/thor_ws, so the robot container cannot see '
                         'it; pass the params file path explicitly')
    return CONTAINER_WS + host_file[len(host_ws):]


def other_publishers(node, topic):
    """Publishers on `topic` other than `node`. CRISP ignores targets while more
    than one publisher exists."""
    return sum(1 for info in node.get_publishers_info_by_topic(topic)
               if info.node_name != node.get_name())


#: The controller parameter whose description says which stiffness messages CRISP takes
#: (tpgpt.hardware.executor.crisp_accepts_full_stiffness reads it).
STIFFNESS_TOPIC_PARAMETER = 'variable_stiffness.topic'


class ControllerManagerClient:
    """Service client for the robot container's controller_manager.

    Blocking calls: the node must be spun by an executor on another thread (or by
    a MultiThreadedExecutor with this client's callback group free), because each
    call waits on its future. Failures raise `error` (the node's own abort type)
    with a message that says what to check.
    """

    def __init__(self, node, controller_manager='/controller_manager', error=RuntimeError,
                 callback_group=None):
        from controller_manager_msgs.srv import (ConfigureController, ListControllers,
                                                 LoadController, SwitchController)
        from rcl_interfaces.srv import SetParameters
        self._srv = {'Configure': ConfigureController, 'List': ListControllers,
                     'Load': LoadController, 'Switch': SwitchController}
        self.node = node
        self.error = error
        cm = controller_manager.rstrip('/')
        make = lambda srv, name: node.create_client(  # noqa: E731
            srv, f'{cm}/{name}', callback_group=callback_group)
        self.cli_list = make(ListControllers, 'list_controllers')
        self.cli_load = make(LoadController, 'load_controller')
        self.cli_configure = make(ConfigureController, 'configure_controller')
        self.cli_switch = make(SwitchController, 'switch_controller')
        self.cli_params = make(SetParameters, 'set_parameters')

    def call(self, client, request, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise self.error(f'service {client.srv_name} not available -- is the robot container '
                             'running with the same ROS_DOMAIN_ID?')
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise self.error(f'service {client.srv_name} timed out')
            time.sleep(0.01)
        return future.result()

    def list(self):
        """{name: ControllerState}."""
        return {c.name: c for c in self.call(self.cli_list, self._srv['List'].Request()).controller}

    def parameter_description(self, controller, parameter, timeout=5.0):
        """The description of a loaded controller's parameter (its node's describe_parameters)."""
        from rcl_interfaces.srv import DescribeParameters
        client = self.node.create_client(DescribeParameters,
                                         f'/{controller.strip("/")}/describe_parameters')
        try:
            response = self.call(client, DescribeParameters.Request(names=[parameter]), timeout)
        finally:
            self.node.destroy_client(client)
        return response.descriptors[0].description if response.descriptors else ''

    def require_full_stiffness(self, controller):
        """Refuse when `controller` would ignore a 12-value stiffness message.

        The pre-patch CRISP does not fail on one: it logs a warning in the robot
        container and keeps its yaml gains, so a run meant to transport stiffness
        would silently run a constant one.
        """
        from tpgpt.hardware.executor import crisp_accepts_full_stiffness

        description = self.parameter_description(controller, STIFFNESS_TOPIC_PARAMETER)
        if not crisp_accepts_full_stiffness(description):
            raise self.error(
                f'{controller} does not take a full stiffness matrix: it would ignore the 12-value '
                'messages and run its yaml gains. The robot image is probably the pre-patch one '
                '(docker-thor:pre-full-stiffness). Pass stiffness_message:=diagonal, or put '
                'docker-thor:local back on the full-stiffness image '
                '(~/THOR_DOCKER_SETUP/setup_crisp_controller.md).')

    def ensure_loaded(self, name, controller_type=None, params_file=None, missing_hint=''):
        """Load (from `params_file`, a path inside the container) and configure `name`
        if needed. With no params_file a missing controller is an error. Returns list()."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        log = self.node.get_logger()
        controllers = self.list()
        if name not in controllers:
            if not params_file:
                raise self.error(f'{name} is not loaded{" (" + missing_hint + ")" if missing_hint else ""}')
            log.info(f'Loading {name} from {params_file} (path inside the robot container)')
            req = SetParameters.Request()
            req.parameters = [
                Parameter(name=f'{name}.type', value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING, string_value=controller_type)),
                Parameter(name=f'{name}.params_file', value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=[params_file])),
            ]
            for result in self.call(self.cli_params, req).results:
                if not result.successful:
                    raise self.error(f'setting controller parameters failed: {result.reason}')
            if not self.call(self.cli_load, self._srv['Load'].Request(name=name)).ok:
                raise self.error(f'loading {name} failed; check the robot container log (is the '
                                 'params file path correct?)')
            controllers = self.list()
        if controllers[name].state == 'unconfigured':
            if not self.call(self.cli_configure, self._srv['Configure'].Request(name=name)).ok:
                raise self.error(f'configuring {name} failed')
            controllers = self.list()
        return controllers

    def effort_owners(self, controllers=None, exclude=()):
        """Active controllers claiming any FR3 joint effort interface."""
        controllers = self.list() if controllers is None else controllers
        effort = {f'{j}/effort' for j in JOINTS}
        return [c.name for c in controllers.values() if c.name not in exclude
                and c.state == 'active' and effort & set(c.claimed_interfaces)]

    def activate(self, name, controllers=None):
        """Strictly switch to `name`, deactivating whatever owns the FR3 effort
        interfaces. CRISP seeds its target with the measured state on activation,
        so the switch itself does not move the arm."""
        log = self.node.get_logger()
        controllers = self.list() if controllers is None else controllers
        if name not in controllers:
            raise self.error(f'{name} is not loaded')
        if controllers[name].state == 'active':
            log.info(f'{name} is already active')
            return
        others = self.effort_owners(controllers, exclude=(name,))
        log.info(f'Switching: activate [{name}], deactivate {others} (strict)')
        Switch = self._srv['Switch']
        req = Switch.Request()
        req.activate_controllers = [name]
        req.deactivate_controllers = others
        req.strictness = Switch.Request.STRICT
        req.timeout.sec = 5
        if not self.call(self.cli_switch, req).ok:
            raise self.error('switch_controller failed')
        if self.list()[name].state != 'active':
            raise self.error(f'{name} did not become active')


class InspireHandClient:
    """Client for inspire_hand_node's ``<namespace>/open`` and ``<namespace>/close``
    (std_srvs/Trigger). Each call blocks the hand node until the motion is done; the
    answer says whether it worked and, for close, which fingers touched something.

    Non-blocking :meth:`send` + :meth:`wait` so a node spun on another thread can
    poll, exactly like the Franka gripper action clients it stands in for."""

    def __init__(self, node, namespace='/inspire_hand', callback_group=None):
        from std_srvs.srv import Trigger
        ns = namespace.rstrip('/') or '/inspire_hand'
        self.Trigger = Trigger
        self.names = {'open': f'{ns}/open', 'close': f'{ns}/close'}
        self.clients = {k: node.create_client(Trigger, name, callback_group=callback_group)
                        for k, name in self.names.items()}

    def available(self, timeout=5.0):
        """Names of the services that did not appear within `timeout` (empty = all up)."""
        return [self.names[k] for k, c in self.clients.items()
                if not c.wait_for_service(timeout_sec=timeout)]

    def send(self, close):
        return self.clients['close' if close else 'open'].call_async(self.Trigger.Request())

    @staticmethod
    def wait(future, timeout=20.0):
        """(success, message) once the call returns, or (False, 'timed out')."""
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                return False, f'no answer within {timeout:.0f} s'
            time.sleep(0.02)
        result = future.result()
        if result is None:
            return False, 'service call failed'
        return bool(result.success), str(result.message)
