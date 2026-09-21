"""X6: turn one planning call and one execution into the trial files of the brief.

No ROS. Imports tpgpt, so it runs in the TPGPT venv. Every function takes what
the planner and executor already produced; nothing here chooses a grasp, fits a
different map or changes a setting. Where a quantity is not stored by the
pipeline (the map's Jacobian at the grasp, the per-stage timings) it is
**recomputed on identical inputs** with the same library calls, and the record
says so.
"""

import os
import time

import numpy as np

from ycb_pointcloud_pipeline.x6_common import write_json

#: Rejection tags written by tpgpt.hardware.grasp_selection, by the stage that
#: produces them. Used to name the stage of each rejection in the record.
SCREENING_TAGS = ('from_below', 'visibility', 'on_target', 'too_wide', 'collides', 'empty',
                  'shallow', 'hits_support', 'grasp_collision', 'place_zone', 'centred',
                  'duplicate')
PLANNING_TAGS = ('plan_refused', 'map_folds', 'misses_grasp', 'wrong_orientation',
                 'path_hits_surface', 'collision', 'kinematics', 'not_tried')
#: Rejections made by the whole-path check (tpgpt.hardware.path_check).
PATH_TAGS = ('path_hits_surface', 'collision', 'kinematics')


def split_rejection(text):
    """'map_folds: min det(J) 0.05' -> ('map_folds', 'min det(J) 0.05')."""
    if not text:
        return None, None
    tag, _, detail = text.partition(':')
    return tag.strip(), detail.strip()


def rejection_stage(tag):
    if tag in SCREENING_TAGS:
        return 'screening'
    if tag in PLANNING_TAGS:
        return 'path_feasibility' if tag in PATH_TAGS else 'planning'
    return 'unknown'


# ---------------------------------------------------------------------------- poses
def grasp_matrix(grasp):
    """4x4 of a GraspFrame in the grasp convention (+X closing, +Z approach)."""
    T = np.eye(4)
    T[:3, :3] = grasp.rotation
    T[:3, 3] = grasp.tcp
    return T


def wrist_matrix(grasp, end_effector=None):
    """4x4 of the commanded end-effector frame orientation (fr3_hand_tcp) at the grasp
    point, for the hand the plan was made for (Franka Hand when not given)."""
    from tpgpt.hardware.end_effector import FRANKA_HAND
    ee = end_effector or FRANKA_HAND
    T = np.eye(4)
    T[:3, :3] = ee.ee_from_grasp(grasp.rotation)
    T[:3, 3] = grasp.tcp
    return T


def plan_end_effector(plan):
    """The hand a plan was made for (plans without the record are the Franka Hand)."""
    from tpgpt.hardware.end_effector import end_effector_for_plan
    return end_effector_for_plan(getattr(plan, 'metadata', None))


def rotations_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R)).as_quat()


# ---------------------------------------------------------------------------- candidates
def candidates_document(selection, raw_poses, raw_scores):
    """The full ranked list with verdicts, and the grasp block of record.json.

    Rank is 1-based in the generator's score order (1 = top candidate), the same
    order select_and_plan walks. The top-down fallback has no rank.
    """
    entries = []
    chosen = selection.chosen if selection is not None else None
    chosen_rank = None
    for rank, c in enumerate(selection.candidates if selection is not None else [], start=1):
        tag, detail = split_rejection(c.rejected)
        if c is chosen:
            status = 'executed'
            chosen_rank = rank
        elif tag is not None:
            status = 'rejected'
        elif chosen_rank is not None:
            status = 'admissible_after_selection_not_evaluated'
        else:
            status = 'admissible'
        d = c.as_dict()
        d.update({
            'rank': rank,
            'generator_index': c.index,
            'status': status,
            'rejection_tag': tag,
            'rejection_stage': rejection_stage(tag) if tag else None,
            'rejection_detail': detail,
            'raw_generator_pose': (np.asarray(raw_poses[c.index]).tolist()
                                   if raw_poses is not None and 0 <= c.index < len(raw_poses)
                                   else None),
            'refined_pose_grasp_convention': grasp_matrix(c.grasp).tolist(),
            'plan_metrics': c.plan_metrics,
        })
        entries.append(d)
    kinematic = bool(getattr(selection, 'fell_back', False))
    fallback_kind = ('top_down' if selection is not None and selection.source == 'top_down'
                     else 'kinematic' if kinematic else None)
    fell_back = fallback_kind is not None
    scripted = selection is not None and selection.source in ('top_down', 'side')
    ee = plan_end_effector(getattr(selection, 'plan', None)) if selection is not None else None
    if scripted:
        chosen_rank = None
    rejections = []
    for d in entries:
        if d['status'] != 'rejected':
            continue
        if chosen_rank is not None and d['rank'] >= chosen_rank:
            continue
        rejections.append({'rank': d['rank'], 'stage': d['rejection_stage'],
                           'tag': d['rejection_tag'], 'detail': d['rejection_detail']})
    doc = {
        'rank_convention': '1-based, generator score order (1 = highest score)',
        'selection_source': selection.source if selection is not None else None,
        'notes': selection.notes if selection is not None else [],
        'n_candidates_generated': 0 if raw_poses is None else int(len(raw_poses)),
        'raw_scores': None if raw_scores is None else np.asarray(raw_scores).tolist(),
        'candidates': entries,
        'fallback': (chosen.as_dict() if fell_back else None),
        'fallback_kind': fallback_kind,
        'screening_stages': getattr(selection, 'stages', None) if selection is not None else None,
        'rank_examined': getattr(selection, 'rank_examined', None) if selection is not None else None,
        'shelf_world': getattr(selection, 'world', None) if selection is not None else None,
    }
    grasp = {
        'n_candidates_generated': doc['n_candidates_generated'],
        'rank_executed': chosen_rank,
        'rejections': rejections,
        'executed_pose_grasp_convention': grasp_matrix(chosen.grasp).tolist() if chosen else None,
        'executed_pose_wrist_convention': wrist_matrix(chosen.grasp, ee).tolist() if chosen else None,
        'wrist_convention_frame': (f"{ee.controller_frame if ee else 'fr3_hand_tcp'} (commanded "
                                   'end-effector frame)'),
        'end_effector': ee.name if ee else None,
        'side_grasp': (getattr(selection, 'plan').metadata.get('side_grasp')
                       if selection is not None and selection.source == 'side' else None),
        'generator_score': (None if chosen is None or scripted else float(chosen.score)),
        'fell_back': bool(fell_back),
        'fallback_kind': fallback_kind,
        'path_check': getattr(chosen, 'path_check', None) if chosen else None,
        'screening_stages': doc['screening_stages'],
        'refined_depth_m': chosen.refined_depth if chosen else None,
        'refined_centre_m': chosen.refined_centre if chosen else None,
    }
    return doc, grasp


# ---------------------------------------------------------------------------- map
def map_block(plan, source, cube_half_extent, construction):
    """Map scalars of record.json, with det J at grasp/release recomputed.

    The planner does not keep its TransportMap on disk, so the map is refitted on
    the plan's own keypoints with the same call (TransportMap().fit) and checked
    against the plan's min det(J) -- a mismatch is recorded, not hidden. The fit
    and the label transport are timed here, on identical inputs.
    """
    from tpgpt.hardware.frames import rotation_angle
    from tpgpt.sim.keypoints import carry_indices
    from tpgpt.transport.labels import transport_labels
    from tpgpt.transport.maps import TransportMap

    m = plan.metrics
    labels = source.tool_labels
    t0 = time.perf_counter()
    tmap = TransportMap().fit(plan.source_keypoints.points, plan.target_keypoints.points)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    transport_labels(tmap, labels)
    t_transport = time.perf_counter() - t0
    gi, ri = carry_indices(labels)
    dets = np.linalg.det(tmap.jacobian(labels.positions))
    refit_min = float(dets.min())
    block = {
        'construction': construction,
        'cube_half_extent_m': float(m.get('cube_half_extent', cube_half_extent)),
        'min_det_J': float(m['min_det']),
        'det_J_at_grasp': float(dets[gi]),
        'det_J_at_release': float(dets[ri]),
        'det_J_recomputed_min': refit_min,
        'det_J_recomputation_matches': bool(abs(refit_min - float(m['min_det'])) < 1e-6),
        'keypoint_residual_mm': float(m['keypoint_residual_m']) * 1000.0,
        'aim_error_mm': float(m['aim_grasp_m']) * 1000.0,
        'aim_error_release_mm': float(m['aim_release_m']) * 1000.0,
        'orientation_error_at_grasp_deg': float(m['orientation_error_grasp_deg']),
        'orientation_error_at_release_deg': float(m['orientation_error_release_deg']),
        'source_to_target_rotation_deg': float(np.degrees(rotation_angle(
            source.placement.grasp.rotation, plan.grasp.rotation))),
        'fraction_positive_det': float(m.get('fraction_positive', np.nan)),
        'hand_clearance_pick_m': m.get('hand_clearance_pick_m'),
        'hand_clearance_place_m': m.get('hand_clearance_place_m'),
    }
    return block, {'map_fit': t_fit, 'label_transport': t_transport}


def time_screening(points, poses, scores, support_height, settings, candidates, context=None):
    """Re-run the screening stage on the same inputs: its duration, and whether it
    reproduced the verdicts the selection recorded (screening is deterministic).
    ``context`` must be the planning call's (``info['screen_context']``)."""
    from tpgpt.hardware import hand
    from tpgpt.hardware.grasp_selection import screen_candidates
    if poses is None or not len(poses):
        return 0.0, None
    t0 = time.perf_counter()
    again = screen_candidates(points, poses, scores, support_height, settings, hand.TCP_OFFSET,
                              context=context)
    elapsed = time.perf_counter() - t0
    ok = True
    for a, b in zip(again, candidates):
        tag_b, _ = split_rejection(b.rejected)
        tag_a, _ = split_rejection(a.rejected)
        if tag_b in PLANNING_TAGS:
            tag_b = None       # screening passed it; planning rejected it later
        if a.index != b.index or tag_a != tag_b:
            ok = False
            break
    return elapsed, ok


# ---------------------------------------------------------------------------- files
def save_keypoints(path, plan):
    s, t = plan.source_keypoints, plan.target_keypoints
    import json
    np.savez_compressed(
        path,
        source_points=s.points, source_labels=np.array(s.labels),
        target_points=t.points, target_labels=np.array(t.labels),
        source_metadata=np.array(json.dumps(s.metadata, default=str)),
        target_metadata=np.array(json.dumps(t.metadata, default=str)),
        frame=np.array(t.frame),
        note=np.array('source file holds src_pick + src_place blocks, target file tgt_pick + '
                      'tgt_place: all four frames'),
    )


def save_plan_array(path, plan):
    labels = plan.labels
    quat = rotations_to_quat_xyzw(labels.orientations)
    arr = np.hstack([labels.positions, quat])
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------- perception
def pick_pose_from_cloud(points, marks=None):
    """Object centroid and yaw (principal horizontal axis, degrees in [0, 180))."""
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    xy = pts[:, :2] - centroid[:2]
    yaw = float('nan')
    if len(xy) >= 3:
        w, v = np.linalg.eigh(np.cov(xy.T))
        axis = v[:, int(np.argmax(w))]
        yaw = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    out = {'centroid_m': centroid.tolist(), 'yaw_deg': yaw, 'nearest_mark_index': None,
           'nearest_mark_distance_m': None, 'source': 'fused planning cloud'}
    if marks:
        best = None
        for mark in marks:
            try:
                d = float(np.hypot(float(mark['x']) - centroid[0], float(mark['y']) - centroid[1]))
            except (TypeError, ValueError, KeyError):
                continue
            if best is None or d < best[1]:
                best = (int(mark['index']), d)
        if best is not None:
            out['nearest_mark_index'], out['nearest_mark_distance_m'] = best
    return out


def placement_error(settled_points, predicted_points, destination_xyz):
    """Settled cloud centroid against the plan's predicted placed-cloud centroid."""
    s = np.asarray(settled_points, dtype=float)
    if s.size == 0:
        return None
    c = s.mean(axis=0)
    out = {'settled_centroid_m': c.tolist(), 'n_settled_points': int(len(s))}
    if predicted_points is not None and len(predicted_points):
        p = np.asarray(predicted_points, dtype=float).mean(axis=0)
        e = (c - p) * 1000.0
        out.update({'predicted_centroid_m': p.tolist(),
                    'placement_error_mm': float(np.hypot(e[0], e[1])),
                    'placement_error_3d_mm': float(np.linalg.norm(e)),
                    'placement_error_components_mm': {'x': float(e[0]), 'y': float(e[1]),
                                                      'z': float(e[2])}})
    if destination_xyz is not None:
        d = (c[:2] - np.asarray(destination_xyz, dtype=float)[:2]) * 1000.0
        out['error_to_commanded_destination_xy_mm'] = {'x': float(d[0]), 'y': float(d[1]),
                                                       'lateral': float(np.hypot(*d))}
    return out


# ---------------------------------------------------------------------------- overlay
def render_plan_overlay(bgr, K, T_base_optical, plan, selection, destination_xyz, title=''):
    """The camera image with the plan drawn through the calibration.

    Cloud (green dots), transported TCP path (blue open / red closed, thicker
    when closer to the camera is not attempted), the hand's boxes (Franka Hand
    collision boxes, or the Inspire hand's envelope) at the chosen grasp (magenta), TCP axes at grasp and release (x red, y green,
    z blue), the commanded destination (yellow ring), and a text box.
    """
    import cv2
    from ycb_pointcloud_pipeline import camera_calibration as cc

    img = bgr.copy()
    h, w = img.shape[:2]
    T_opt_base = cc.invert_transform(T_base_optical)
    K = np.asarray(K, dtype=float).reshape(3, 3)

    def project(points):
        q = cc.transform_points(T_opt_base, np.atleast_2d(np.asarray(points, dtype=float)))
        z = np.maximum(q[:, 2], 1e-6)
        uv = np.column_stack([K[0, 0] * q[:, 0] / z + K[0, 2], K[1, 1] * q[:, 1] / z + K[1, 2]])
        return uv, q[:, 2] > 0

    def inside(p):
        return 0 <= p[0] < w and 0 <= p[1] < h

    uv, ok = project(plan.target_points)
    for (u, v), f in zip(uv.astype(int), ok):
        if f and inside((u, v)):
            cv2.circle(img, (u, v), 1, (0, 255, 0), -1)

    placed = plan.diagnostics.get('target_place_points') if plan.diagnostics else None
    if placed is not None and len(placed):
        uv, ok = project(placed)
        for (u, v), f in zip(uv.astype(int), ok):
            if f and inside((u, v)):
                img[v, u] = (255, 255, 0)

    labels = plan.labels
    uv, ok = project(labels.positions)
    pts = uv.astype(int)
    for k in range(len(pts) - 1):
        if ok[k] and ok[k + 1]:
            color = (0, 0, 255) if labels.gripper[k] > 0 else (255, 120, 0)
            cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]), color, 2, cv2.LINE_AA)
    if ok[0]:
        cv2.circle(img, tuple(pts[0]), 5, (255, 255, 255), 1, cv2.LINE_AA)

    g = plan.grasp
    R = g.rotation
    for box in plan_end_effector(plan).boxes():
        corners = box.corners() @ R.T + g.tcp
        c2, cok = project(corners)
        if not cok.all():
            continue
        c2 = c2.astype(int)
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count('1') == 1:
                    cv2.line(img, tuple(c2[i]), tuple(c2[j]), (255, 0, 255), 1, cv2.LINE_AA)

    gi, ri = int(plan.metrics['grasp_index']), int(plan.metrics['release_index'])
    for idx in (gi, ri):
        o = labels.positions[idx]
        Rt = labels.orientations[idx]
        axes = [o, o + Rt[:, 0] * 0.05, o + Rt[:, 1] * 0.05, o + Rt[:, 2] * 0.05]
        a2, aok = project(axes)
        if aok.all():
            a2 = a2.astype(int)
            for k, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                cv2.line(img, tuple(a2[0]), tuple(a2[k]), col, 2, cv2.LINE_AA)

    if destination_xyz is not None:
        dx, dy, dz = (float(v) for v in destination_xyz)
        ring = np.array([[dx + 0.03 * np.cos(a), dy + 0.03 * np.sin(a), dz]
                         for a in np.linspace(0, 2 * np.pi, 33)])
        r2, rok = project(ring)
        if rok.all():
            cv2.polylines(img, [r2.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    m = plan.metrics
    lines = [title] if title else []
    if selection is not None:
        rank = next((k for k, c in enumerate(selection.candidates, 1) if c is selection.chosen),
                    None)
        lines.append(f'grasp: {selection.source}, rank {rank if selection.source not in ("top_down", "side") else "-"}'
                     f' of {len(selection.candidates)}, score {selection.chosen.score:.3f}')
    lines.append(f"min det J {m['min_det']:.3f}  aim {m['aim_grasp_m'] * 1000:.1f} mm  "
                 f"orient {m['orientation_error_grasp_deg']:.1f} deg  tilt {m['grasp_tilt_deg']:.0f} deg")
    lines.append('path: orange = open, red = closed; magenta = hand at grasp; yellow = destination')
    y = 18
    for text in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y += 20
    return img


# ---------------------------------------------------------------------------- cartesian
def write_cartesian_positions(path, trace):
    """cartesian_positions.csv: one row per control step, base frame, metres.

    measured = fr3_hand_tcp from /current_pose; commanded = pose sent to the
    controller. Time is step_index / control_rate_hz. Empty cells = not available.
    """
    import csv
    rows = trace.rows

    def xyz(v):
        return [''] * 3 if v is None else [f'{float(c):.6f}' for c in list(v)[:3]]

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step_index', 't_s', 'stage', 'measured_x', 'measured_y', 'measured_z',
                    'commanded_x', 'commanded_y', 'commanded_z', 'gripper_command',
                    'gripper_width'])
        for i, stage in enumerate(rows['stage']):
            ee, cmd = rows['ee_pose'][i], rows['commanded_pose'][i]
            g, width = rows['gripper_command'][i], rows['gripper_width'][i]
            w.writerow([i, f'{i / trace.rate_hz:.3f}', stage,
                        *xyz(ee), *xyz(cmd),
                        '' if g is None else g, '' if width is None else width])


def _enable_3d_projection(matplotlib):
    """Register matplotlib's own mplot3d. The TPGPT venv includes system site-packages,
    and /usr/lib/python3/dist-packages/mpl_toolkits (Ubuntu's matplotlib 3.6) shadows
    the venv's 3.10 copy, so projection='3d' is unknown ("Unable to import Axes3D")."""
    import os
    from matplotlib import projections
    if '3d' in projections.get_projection_names():
        return
    import mpl_toolkits
    own = os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)), 'mpl_toolkits')
    if os.path.isdir(own):
        mpl_toolkits.__path__ = [own] + [d for d in mpl_toolkits.__path__ if d != own]
    from mpl_toolkits.mplot3d import Axes3D
    projections.register_projection(Axes3D)


def render_trajectory_3d(path, trace, plan=None, title=''):
    """trajectory_3d.png: the measured TCP path in the base frame, with its start and
    end, the planned path, the object cloud the plan was made from, the predicted
    placed cloud, and the Franka Hand's boxes (fingers open) at the planned grasp
    and release poses. Equal axis scaling, so distances read true."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _enable_3d_projection(matplotlib)

    measured = np.array([list(e)[:3] for e in trace.rows['ee_pose'] if e is not None], dtype=float)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    everything = [measured] if len(measured) else []

    if plan is not None:
        cloud = np.asarray(plan.target_points, dtype=float)
        if len(cloud):
            ax.scatter(*cloud.T, s=2, c='tab:green', alpha=0.6, label='object cloud (plan)')
            everything.append(cloud)
        placed = plan.diagnostics.get('target_place_points') if plan.diagnostics else None
        if placed is not None and len(placed):
            placed = np.asarray(placed, dtype=float)
            ax.scatter(*placed.T, s=2, c='tab:cyan', alpha=0.4, label='object at release (predicted)')
            everything.append(placed)
        labels = plan.labels
        ax.plot(*np.asarray(labels.positions, dtype=float).T, '--', c='tab:gray', lw=1,
                label='planned TCP path')
        everything.append(np.asarray(labels.positions, dtype=float))
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]
        gi, ri = int(plan.metrics['grasp_index']), int(plan.metrics['release_index'])
        poses = [('grasp', plan.grasp.tcp, plan.grasp.rotation, 'tab:purple'),
                 ('release', labels.positions[ri], labels.orientations[ri], 'tab:orange')]
        for name, tcp, R, color in poses:
            R, tcp = np.asarray(R, dtype=float), np.asarray(tcp, dtype=float)
            for k, box in enumerate(plan_end_effector(plan).boxes()):
                c = box.corners() @ R.T + tcp
                for e, (i, j) in enumerate(edges):
                    ax.plot(*np.stack([c[i], c[j]]).T, c=color, lw=0.8,
                            label=f'gripper at {name} (plan)' if k == 0 and e == 0 else None)
                everything.append(c)

    if len(measured):
        ax.plot(*measured.T, c='tab:blue', lw=1.5, label='measured TCP path')
        ax.scatter(*measured[0], s=80, c='lime', edgecolors='k', marker='o', label='start', depthshade=False)
        ax.scatter(*measured[-1], s=160, c='red', edgecolors='k', marker='X', label='end', depthshade=False)

    if everything:
        pts = np.vstack(everything)
        centre = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        half = max(float(np.ptp(pts, axis=0).max()) / 2.0, 0.05)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title or 'TCP trajectory, base frame')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- trace
class TraceRecorder:
    """Per-control-step trace. Time is the step index; no clock is stored."""

    FIELDS = ('step_index', 'stage', 'ee_pose', 'commanded_pose', 'attractor_position',
              'joint_positions', 'gripper_command', 'gripper_width', 'wrench', 'phase', 'gate')

    def __init__(self, rate_hz):
        self.rate_hz = float(rate_hz)
        self.rows = {k: [] for k in self.FIELDS}

    def __len__(self):
        return len(self.rows['step_index'])

    def add(self, stage, ee_pose, commanded_pose, attractor, joints, gripper_command,
            gripper_width, wrench, phase, gate):
        self.rows['step_index'].append(len(self))
        self.rows['stage'].append(stage)
        self.rows['ee_pose'].append(ee_pose)
        self.rows['commanded_pose'].append(commanded_pose)
        self.rows['attractor_position'].append(attractor)
        self.rows['joint_positions'].append(joints)
        self.rows['gripper_command'].append(gripper_command)
        self.rows['gripper_width'].append(gripper_width)
        self.rows['wrench'].append(wrench)
        self.rows['phase'].append(phase)
        self.rows['gate'].append(gate)

    def save(self, path):
        n = len(self)

        def arr(key, width):
            a = np.full((n, width), np.nan)
            for i, v in enumerate(self.rows[key]):
                if v is not None:
                    a[i] = np.asarray(v, dtype=float).reshape(width)
            return a

        stages = np.array(self.rows['stage'], dtype='U32') if n else np.zeros(0, dtype='U32')
        np.savez_compressed(
            path,
            step_index=np.arange(n, dtype=np.int64),
            control_rate_hz=np.array(self.rate_hz),
            stage=stages,
            ee_pose=arr('ee_pose', 7),
            attractor_pose=arr('commanded_pose', 7),
            attractor_position_grasp_point=arr('attractor_position', 3),
            joint_positions=arr('joint_positions', 7),
            gripper_command=arr('gripper_command', 1).reshape(-1),
            gripper_width=arr('gripper_width', 1).reshape(-1),
            wrench=arr('wrench', 6),
            phase=arr('phase', 1).reshape(-1),
            gate=arr('gate', 1).reshape(-1),
            note=np.array('time = step_index / control_rate_hz; poses xyz + quat xyzw, base frame; '
                          'attractor_pose = pose sent to the controller; NaN = not available'),
        )

    def policy_steps(self):
        return sum(1 for s in self.rows['stage'] if s == 'policy')

    def gripper_width_at(self, stage_prefix, which='last'):
        vals = [w for s, w in zip(self.rows['stage'], self.rows['gripper_width'])
                if s.startswith(stage_prefix) and w is not None and np.isfinite(w)]
        if not vals:
            return None
        return float(vals[-1] if which == 'last' else vals[0])


def write_record(trial_dir, record):
    write_json(os.path.join(trial_dir, 'record.json'), record)
