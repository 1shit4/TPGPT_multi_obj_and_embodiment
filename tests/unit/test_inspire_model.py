"""The Inspire RH56DFX-2R hand model against GraspGen-X's own description of it.

``tpgpt.hardware.inspire_model`` builds the hand from GraspGen-X's ``inspire_hand``
URDF. These tests check it against what GraspGen-X itself sampled from that hand
(``points.json``, open and closed), against the convention every planned grasp
arrives in (thumb on +X, fingers on -X, +Z approach), and the closing model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tpgpt.hardware import inspire_model as im


@pytest.fixture(scope="module")
def desc():
    try:
        return im.load_description()
    except FileNotFoundError as exc:                       # no GraspGen-X checkout
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def model_w(desc):
    """The model in GraspGen-X's frame W itself (tool depth 0)."""
    return im.InspireHandModel(desc, tool_depth=0.0)


def _outside(pieces, points):
    """Each point's distance outside the nearest piece (0 inside), m."""
    return np.array([max(0.0, min(float(np.max(p.planes[:, :3] @ x + p.planes[:, 3]))
                                  for p in pieces)) for x in points])


@pytest.mark.parametrize("state", ["open", "close"])
def test_the_model_covers_graspgenx_s_own_samples_of_the_hand(desc, model_w, state):
    samples = np.asarray(json.loads((im.description_dir("inspire_hand") / "points.json")
                                    .read_text())[state])
    q = desc.open_q if state == "open" else desc.close_q
    pieces = model_w.pieces(q)
    d = _outside(pieces, samples[::5])
    assert np.median(d) < 0.001 and np.percentile(d, 95) < 0.008
    V = np.vstack([p.vertices for p in pieces])
    # Conservative pieces: the model's box contains the samples' to 2 mm, and
    # exceeds it by no more than the 26-DOP excess plus the mesh/sample gap.
    assert np.all(V.min(0) <= samples.min(0) + 0.002)
    assert np.all(V.max(0) >= samples.max(0) - 0.002)
    assert np.all(samples.min(0) - V.min(0) < 0.008)
    assert np.all(V.max(0) - samples.max(0) < 0.008)


def test_at_open_the_thumb_is_on_plus_x_and_the_fingers_on_minus_x(model_w):
    tips = model_w.fingertip_points("open")
    assert tips["thumb"][0] > 0.04
    for finger in ("index", "middle", "ring", "pinky"):
        assert tips[finger][0] < -0.06
    # Right hand: index on +Y, pinky on -Y (GraspGen-X's +Y = approach x closing).
    assert tips["index"][1] > tips["middle"][1] > tips["ring"][1] > tips["pinky"][1]


def test_closing_brings_thumb_and_fingers_together_along_x(desc, model_w):
    open_tips, closed_tips = model_w.fingertip_points("open"), model_w.fingertip_points(desc.close_q)
    assert closed_tips["thumb"][0] < open_tips["thumb"][0] - 0.02        # thumb moves -X
    assert closed_tips["index"][0] > open_tips["index"][0] + 0.03        # fingers move +X


def test_mimic_joints_follow_their_drivers(model_w):
    v = model_w.joint_values([0.5, 0.6, 0.7, 0.8, 0.3, 1.0])
    assert v["index_intermediate_joint"] == pytest.approx(1.06399 * 0.8 - 0.04545)
    assert v["pinky_intermediate_joint"] == pytest.approx(1.06399 * 0.5 - 0.04545)
    assert v["thumb_intermediate_joint"] == pytest.approx(1.334 * 0.3)
    assert v["thumb_distal_joint"] == pytest.approx(0.667 * 0.3)
    assert v["thumb_proximal_yaw_joint"] == 1.0


def test_the_wrist_frame_is_graspgenx_s_world_joint(desc, model_w):
    T = model_w.link_poses(desc.open_q)["hand_base_link"]
    assert T[:3, 3] == pytest.approx([0.065, -0.01, 0.0])
    from scipy.spatial.transform import Rotation
    assert T[:3, :3] == pytest.approx(Rotation.from_euler("xyz", [1.570796, 2.356194, 0])
                                      .as_matrix(), abs=1e-9)
    # In H the fingers run along -Y and the palm faces -X: the approach (W +Z) is their
    # bisector, 45 deg from each.
    fingers_W, palm_W = -T[:3, 1], -T[:3, 0]
    assert np.degrees(np.arccos(fingers_W[2])) == pytest.approx(45.0, abs=0.01)
    assert np.degrees(np.arccos(palm_W[2])) == pytest.approx(45.0, abs=0.01)


def test_the_tool_depth_moves_the_hand_along_minus_z(desc):
    a = im.InspireHandModel(desc, tool_depth=0.0).pieces("open")
    b = im.InspireHandModel(desc, tool_depth=0.15).pieces("open")
    for pa, pb in zip(a, b):
        assert pb.vertices == pytest.approx(pa.vertices - [0.0, 0.0, 0.15])
        probe = pa.centre
        assert pa.contains(probe) and pb.contains(probe - [0.0, 0.0, 0.15])


def test_the_grasp_point_lies_in_front_of_the_palm(desc):
    """GraspGen-X's grasp point (the fingertip depth) is in the hand's pocket: between
    thumb and fingers, clear of the palm, inside the open sweep box."""
    m = im.InspireHandModel(desc)                       # G origin = the grasp point
    body = [p for p in m.pieces("open") if p.role == "body"]
    assert not any(p.contains(np.zeros(3), 0.02) for p in body)
    lo, hi = m.sweep_boxes()[0]
    assert np.all(lo < 0) and np.all(hi > 0)


def test_pieces_are_named_and_roled_for_the_path_check(model_w):
    pieces = model_w.pieces("open")
    names = [p.name for p in pieces]
    assert len(names) == len(set(names)) == 8 + 1 + 3 + 4 * 2
    for p in pieces:
        assert (p.role == "body") == p.name.startswith("body_")
    assert {p.chain for p in pieces if p.role == "finger"} == {"thumb", "index", "middle",
                                                               "ring", "pinky"}


def test_every_piece_is_convex_and_its_vertices_are_on_its_boundary(model_w):
    for p in model_w.pieces("open"):
        d = p.vertices @ p.planes[:, :3].T + p.planes[:, 3]
        assert np.all(d <= 1e-9)
        assert np.all(np.isclose(d.max(axis=1), 0.0, atol=1e-9))
        assert len(p.edges()) >= 12


def test_the_26dop_encloses_its_mesh_and_overstates_it_by_at_most_5_mm(desc):
    from scipy.spatial import ConvexHull

    for link, items in desc.link_geometry.items():
        for kind, path, origin in items:
            if kind != "mesh":
                continue
            v = im._transform(im._read_obj(path), origin)
            dv, planes = im.kdop(v)
            assert np.all(v @ planes[:, :3].T + planes[:, 3] <= 1e-9), link
            h = ConvexHull(v)
            excess = max(float(np.max(h.equations[:, :3] @ x + h.equations[:, 3])) for x in dv)
            assert excess < 0.005, link


def test_the_register_map_round_trips_and_defaults_are_unverified(desc):
    m = im.InspireHandModel(desc)
    assert all(not ok for _, _, ok in m.joint_map.values())
    assert m.registers_to_q([1000] * 4 + [1000, 0]) == pytest.approx(
        [0.128, 0.122, 0.123, 0.120, 0.082, 1.308])         # the hand's open
    assert m.close_q == pytest.approx([1.373, 1.373, 1.373, 1.373, 0.650, 1.308])
    u = np.array([0, 250, 500, 750, 1000, 400])
    assert np.array_equal(m.q_to_registers(m.registers_to_q(u)), u)


def test_the_joint_map_agrees_with_the_hand_node(desc):
    """One map, two readers: the hand node and this model read the same yaml block."""
    ih = pytest.importorskip("ycb_pointcloud_pipeline.inspire_hand")
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config/inspire_hand.yaml"
    cfg = yaml.safe_load(path.read_text())["inspire_hand"]
    node_map, model_map = ih.joint_map_from_config(cfg), im.joint_map_from_config(cfg)
    assert node_map == model_map
    m = im.InspireHandModel(desc, joint_map=model_map)
    u = [0, 123, 456, 789, 1000, 0]
    assert m.registers_to_q(u) == pytest.approx(ih.register_to_joint(u, node_map))


# ------------------------------------------------------------------- closing
def _cylinder(centre, radius, z_range, n=60, spacing=0.003):
    zs = np.arange(z_range[0], z_range[1], spacing)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.vstack([np.column_stack([centre[0] + radius * np.cos(th),
                                       centre[1] + radius * np.sin(th), np.full(n, z)])
                      for z in zs])


def test_a_cylinder_in_the_pocket_stops_the_thumb_and_the_fingers(desc):
    m = im.InspireHandModel(desc)                     # G origin = the grasp point
    # A 50 mm cylinder along the across-palm axis (G y), through the grasp point.
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ys = np.arange(-0.06, 0.04, 0.003)
    obj = np.vstack([np.column_stack([0.025 * np.cos(th), np.full(72, y), 0.025 * np.sin(th)])
                     for y in ys])
    state = m.closed_onto(obj)
    assert state.contact["thumb"] and state.opposed
    assert {"index", "middle"} <= set(state.fingers_in_contact)
    for chain, frac in state.fraction.items():
        if state.contact[chain]:
            assert 0.0 < frac < 1.0
    # Where they stopped, no finger is inside the object.
    for piece in m.pieces(state.q):
        if piece.role == "finger":
            assert int(piece.contains(obj, -im.CONTACT_DEPTH).sum()) < im.CONTACT_POINTS


def test_empty_space_closes_all_the_way(desc):
    m = im.InspireHandModel(desc)
    state = m.closed_onto(np.empty((0, 3)))
    assert not any(state.contact.values()) and not state.opposed
    assert state.q[:5] == pytest.approx(m.close_q[:5])


def test_a_finger_does_not_pass_through_a_thin_wall(desc):
    m = im.InspireHandModel(desc)
    tips_open = m.fingertip_points("open")["index"]
    tips_closed = m.fingertip_points(m.close_q)["index"]
    mid = (tips_open + tips_closed) / 2.0
    # A 3 mm wall across the index finger's closing path, halfway along it.
    g = np.arange(-0.03, 0.03, 0.002)
    yy, zz = np.meshgrid(g, g)
    wall = np.vstack([np.column_stack([np.full(yy.size, mid[0] + dx), mid[1] + yy.ravel(),
                                       mid[2] + zz.ravel()]) for dx in (-0.0015, 0.0, 0.0015)])
    state = m.closed_onto(wall)
    assert state.contact["index"]
    assert state.fraction["index"] < 0.9


#: The ruler readings of 2026-10-05 (docs/hardware/INSPIRE_HAND.md, M4), mm.
FINGER_READINGS = {1000: 216, 750: 195, 500: 168, 250: 125, 0: 89}   # middle tip below the face
THUMB_READINGS = {1000: 120, 750: 92, 500: 73, 250: 64}                # thumb to index, tip centres


def _tip_centre(m, link, chain, q, radius=0.012):
    tip = m.fingertip_points(q)[chain]
    v = m.mesh_vertices(link, q)
    return v[np.linalg.norm(v - tip, axis=1) < radius].mean(0)


def test_the_default_map_reproduces_the_ruler_readings(desc):
    """The default register map reproduces the 2026-10-05 ruler readings, with the fingers'
    open end from the mount touches: the middle fingertip within the ruler's ~7 mm at every
    setting (the manual's map missed 500 and 250 by 19 and 22), the thumb-to-index distance
    between the tips' centres within 4 mm (the manual's missed by up to 17)."""
    m = im.InspireHandModel(desc)
    for u, mm in FINGER_READINGS.items():
        lm = im.landmarks(m, m.registers_to_q([u] * 4 + [1000, 1000]))
        assert lm["middle_length"] == pytest.approx(mm, abs=7.0), u
    for u, mm in THUMB_READINGS.items():
        q = m.registers_to_q([1000] * 4 + [u, 0])
        d = np.linalg.norm(_tip_centre(m, "thumb_distal", "thumb", q)
                           - _tip_centre(m, "index_intermediate", "index", q)) * 1000.0
        assert d == pytest.approx(mm, abs=4.0), u
    # Thumb rotation (not measured yet): the URDF's 1.308 rad span is the manual's 75 deg.
    span = (m.manual_angles(m.registers_to_q([1000] * 6))["thumb_rot"]
            - m.manual_angles(m.registers_to_q([1000] * 5 + [0]))["thumb_rot"])
    assert span == pytest.approx(75.0, abs=0.5)


def test_the_real_hand_opens_less_flat_and_curls_less_than_graspgenx_s(desc):
    """Measured 2026-10-05: at 1000 the fingers are 0.12-0.13 rad from straight (the touches)
    and the thumb 0.08 (GraspGen-X's open: 0); the fingers' full curl is 1.373 rad, short of
    GraspGen-X's close, 1.47 (alpha ~42 deg by the manual's construction, not its 19); the
    thumb's is 0.65 against 0.6."""
    m = im.InspireHandModel(desc)
    assert 38.0 < m.manual_angles(desc.close_q)["index"] < 46.0
    assert np.all(m.close_q[:4] < desc.close_q[:4] - 0.05)
    assert np.all(m.open_q[:4] > desc.open_q[:4] + 0.1)
    assert m.open_q[4] > desc.open_q[4] + 0.05


def test_the_open_hand_is_the_manual_s_217_8_mm_long(desc):
    """The RH56 manual's dimension drawing (figure 1): 217.8 mm from the mounting face to
    the (middle) fingertip, straight. The model, from GraspGen-X's URDF with the wrist face
    as hand_base_link's origin, gives 218.0 -- the URDF is this hand, and its origin is
    the face that bolts to the adapter. At register 1000 the real fingers are 5 deg short
    of straight (the mount touches): 216.4, and the ruler read 216 (2026-10-05)."""
    m = im.InspireHandModel(desc)
    straight = im.landmarks(m, desc.open_q)
    assert straight["middle_length"] == pytest.approx(217.8, abs=1.0)
    assert 55.0 < straight["knuckles_index_to_pinky"] < 60.0
    lm = im.landmarks(m, "open")
    assert lm["middle_length"] == pytest.approx(216.0, abs=1.0)
    half = im.landmarks(m, m.registers_to_q([500] * 5 + [0]))
    assert half["thumb_to_index"] < lm["thumb_to_index"] - 100.0


def _slab(tilt_deg, half=0.02, span=0.05, step=0.003):
    """Two parallel faces 40 mm apart, their normal tilted ``tilt_deg`` from the closing
    axis (+x) about y, in the grasp frame."""
    a = np.radians(tilt_deg)
    n, u, v = np.array([np.cos(a), 0, np.sin(a)]), np.array([-np.sin(a), 0, np.cos(a)]), \
        np.array([0.0, 1.0, 0.0])
    g, gy = np.arange(-span, span, step), np.arange(-0.07, 0.04, step)
    return np.array([s * half * n + a1 * u + a2 * v for s in (-1, 1) for a1 in g for a2 in gy])


@pytest.mark.parametrize("tilt", [0.0, 20.0, 40.0])
def test_the_grip_angle_is_the_tilt_of_the_faces_it_squeezes(desc, tilt):
    """Faces square to the squeeze read 0; faces tilted by a read a, for every chain."""
    m = im.InspireHandModel(desc)
    pts = _slab(tilt)
    state = m.closed_onto(pts)
    angles = m.grip_angles(pts, state)
    assert state.opposed and set(angles) == {c for c, v in state.contact.items() if v}
    assert all(v == pytest.approx(tilt, abs=1.0) for v in angles.values())
    assert im.grip_angle(angles) == pytest.approx(tilt, abs=1.0)


def test_the_grip_angle_takes_the_thumb_and_the_best_finger():
    assert im.grip_angle({"thumb": 10.0, "index": 50.0, "middle": 5.0}) == 10.0
    assert im.grip_angle({"thumb": 40.0, "index": 50.0, "middle": 5.0}) == 40.0
    assert im.grip_angle({"thumb": 10.0, "index": 50.0}) == 50.0
    assert np.isnan(im.grip_angle({"index": 5.0}))
    assert np.isnan(im.grip_angle({"thumb": 5.0}))


def test_the_finger_lead_is_the_time_the_first_finger_waits_for_the_thumb():
    """The mug of 2026-10-07 21:29: index on at 0.25 of its travel, thumb at 0.58 -- the index
    on the mug 0.39 s before the thumb at one speed; the thumb first reads negative."""
    state = im.ClosedState(q=np.zeros(6), contact={"pinky": False, "ring": True, "middle": True,
                                                   "index": True, "thumb": True},
                           fraction={"pinky": 1.0, "ring": 0.446, "middle": 0.298,
                                     "index": 0.25, "thumb": 0.576})
    r = im.CLOSE_RATE_AT_500
    assert im.finger_lead(state) == pytest.approx(0.576 / r["thumb"] - 0.25 / r["index"])
    assert im.finger_lead(state) == pytest.approx(0.39, abs=0.01)
    state.fraction["thumb"] = 0.1
    assert im.finger_lead(state) < 0
    state.contact["thumb"] = False
    assert np.isnan(im.finger_lead(state))
