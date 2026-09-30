"""tpgpt.hardware.replay: the passes of a replayed demonstration, with a simulated arm."""

import json

import numpy as np
import pytest

from tpgpt.hardware import replay as rp
from tpgpt.hardware.teach import Recording, bundle_from_recording

TAU = 0.27                                   # the arm's time constant, s (measured, audit B2)
SAG = np.array([0.0, 0.0, -0.003])


def _taught_bundle(tmp_path):
    """A taught demonstration: still, 0.15 m along +x, dwell (grasp), 0.15 m +y and up, dwell (release)."""
    rate, pieces, pos = 1000.0, [], np.array([0.4, 0.0, 0.05])
    for step, seconds in ((None, 1.0), ((0.05, 0, 0), 3.0), (None, 1.0), ((0, 0.05, 0.02), 3.0),
                          (None, 1.0)):
        n = int(seconds * rate)
        d = np.zeros(3) if step is None else np.asarray(step, float) / rate
        pieces.append(pos + np.outer(np.arange(1, n + 1), d))
        pos = pieces[-1][-1]
    positions = np.concatenate(pieces)
    down = np.diag([1.0, -1.0, -1.0])
    rec = Recording(stamps=np.arange(len(positions)) / rate, positions=positions,
                    orientations=np.repeat(down[None], len(positions), axis=0),
                    grasp_stamp=4.5, release_stamp=8.5,
                    object_points=positions[4400] + np.random.default_rng(0).normal(
                        scale=0.01, size=(80, 3)),
                    support_height=0.0)
    bundle, _ = bundle_from_recording(rec)
    bundle.save(tmp_path / "bundle")
    return tmp_path / "bundle"


TILT = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_rotvec(
    np.radians(1.5) * np.array([0.6, 0.8, 0.0])).as_matrix()      # the resting tilt
TAU_ROT = 0.35


def _run_pass(root, k, info, run, noise=0.0005):
    """What the executor's replay mode records: the arm at each 50 ms step, index advancing
    0.35 -- first-order toward the attractor plus SAG, the hand toward TILT @ attractor."""
    from scipy.spatial.transform import Rotation

    attractor = rp.load_attractor(root, k)
    att_R = rp.load_attractor_rotations(root, k)
    n = len(attractor)
    dt = info["dt"]
    rng = np.random.default_rng(100 * k + run)
    wobble = rng.normal(scale=noise, size=3)                 # this run's own small offset
    x, R = attractor[0] + SAG, TILT @ att_R[0]
    s, rows = 0.0, {"label_index": [], "positions": [], "rotations": [], "step": []}
    step = 0
    while True:
        rows["label_index"].append(s)
        rows["positions"].append(x.copy())
        rows["rotations"].append(R.copy())
        rows["step"].append(step)
        a = rp.attractor_at(attractor, s)
        target_R = TILT @ att_R[min(int(round(s)), n - 1)]
        for _ in range(10):                                # the arm, integrated finely
            x = x + (a + SAG + wobble - x) * (dt / 10 / TAU)
            turn = Rotation.from_matrix(target_R @ R.T).as_rotvec() * (dt / 10 / TAU_ROT)
            R = Rotation.from_rotvec(turn).as_matrix() @ R
        if s >= n - 1:
            break
        s = min(s + info["speed_scale"], n - 1)
        step += 1
    rp.save_recording(root, k, run, **rows)


def test_the_passes_learn_the_attractor(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    info, bundle = rp.load_replay(root)
    assert info["grasp_index"] < info["release_index"]
    assert np.array_equal(rp.load_attractor(root, 0), bundle.tool_labels.positions)
    assert np.array_equal(rp.load_attractor_rotations(root, 0), bundle.tool_labels.orientations)
    # pass 0 twice, with the same attractor
    assert rp.pending_pass(root) == (0, 1)
    _run_pass(root, 0, info, 1)
    assert rp.pending_pass(root) == (0, 2)
    with pytest.raises(FileNotFoundError):
        rp.update(root)                                    # one run of two: not yet
    _run_pass(root, 0, info, 2)
    with pytest.raises(FileExistsError):
        rp.pending_pass(root)                              # recorded: update first
    errors, tilts = [], []
    for k in range(4):
        report = rp.update(root)
        errors.append(report["error_rms_mm"])
        tilts.append(report["tilt_rms_deg"])
        assert rp.pending_pass(root) == (k + 1, 1)
        _run_pass(root, k + 1, info, 1)
    assert report["spread_from_pass"] == 0 and report["spread_max_mm"] == 0.0
    first = json.loads((root / "pass_000" / "report.json").read_text())
    assert first["runs"] == 2 and 0.3 < first["spread_max_mm"] < 3.0
    assert first["gain"] == 1.0
    assert errors[-1] < errors[0] / 4
    assert tilts[0] > 1.0 and tilts[-1] < 0.2              # the resting tilt, learned away
    # every later pass is checked over the bracket from where the last one went
    margin = np.load(root / "pass_001" / "margin.npz")
    assert str(margin["source"]) == "the bracket from pass 0"
    assert float(margin["overshoot"]) == rp.BRACKET_OVERSHOOT
    assert margin["start_positions"].shape == (info["n_labels"], 3)
    assert margin["clearance"].min() >= rp.MARGIN_FLOOR
    rows = rp.status(root)
    assert [r["runs"] for r in rows] == [2, 1, 1, 1, 1]


def test_update_needs_a_recording_and_init_refuses_to_overwrite(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    with pytest.raises(FileExistsError):
        rp.init_replay(tmp_path / "bundle", root)
    with pytest.raises(FileNotFoundError):
        rp.update(root)


def test_repeat_adds_a_run_to_the_last_pass(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay", pass0_runs=1)
    info, _ = rp.load_replay(root)
    _run_pass(root, 0, info, 1)
    assert rp.add_run(root) == (0, 2)
    assert rp.pending_pass(root) == (0, 2)


def test_a_sideways_stray_is_measured_a_lag_along_the_path_is_not():
    from scipy.spatial.transform import Rotation

    P = np.stack([np.linspace(0.3, 0.6, 60), np.zeros(60), np.full(60, 0.2)], axis=1)
    R = np.repeat(np.eye(3)[None], 60, axis=0)
    late = np.roll(P, 5, axis=0)                              # 5 labels behind, on the path
    late[:5] = P[0]
    side, tilt = rp.strays(P, R, late, R)
    assert side.max() < 1e-9 and tilt.max() < 1e-9
    off = P + [0.0, 0.004, 0.0]
    turned = Rotation.from_euler("x", 2.0, degrees=True).as_matrix() @ R
    side, tilt = rp.strays(P, R, off, turned)
    assert np.allclose(side, 0.004) and np.allclose(np.degrees(tilt), 2.0)


def test_pass_0_margin_is_what_the_stiffness_checks_measured():
    R = np.repeat(np.eye(3)[None], 40, axis=0)                # the hand not turning
    clearance, tilt = rp.first_pass_margin(R, np.full(40, 30.0), 0.05 / 0.35)
    assert np.allclose(clearance, rp.FIRST_PASS_STRAY)
    assert np.allclose(np.degrees(tilt), 1.32)                 # measured at 30, 2026-09-30
    _, stiffer = rp.first_pass_margin(R, np.full(40, 60.0), 0.05 / 0.35)
    assert np.allclose(np.degrees(stiffer), 0.90)
    _, softer = rp.first_pass_margin(R, np.full(40, 7.5), 0.05 / 0.35)
    assert np.allclose(np.degrees(softer), 2.47 * 15.0 / 7.5)  # below the softest: scaled up
    from scipy.spatial.transform import Rotation
    turning = Rotation.from_euler("z", np.linspace(0, 20, 40), degrees=True).as_matrix()
    _, moving = rp.first_pass_margin(turning, np.full(40, 30.0), 0.05 / 0.35)
    assert moving.max() > tilt.max()                           # the turning lag adds


SHELF = (__import__("pathlib").Path(__file__).resolve().parents[2]
         / "ycb_pointcloud_pipeline" / "config" / "shelf.yaml")


def test_the_check_is_on_the_taught_path_with_the_margin_not_on_the_attractor(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    info, bundle = rp.load_replay(root)
    n = info["n_labels"]
    verdict, _ = rp.check_pass(root, 0, SHELF, rotational=np.full(n, 60.0))
    assert verdict is None, verdict                          # the taught path is clear
    assert (root / "pass_000" / "margin.npz").exists()       # what it was checked with
    (root / "pass_001").mkdir()
    dipped = rp.load_attractor(root, 0).copy()
    dipped[20:40, 2] -= 0.12                                 # an attractor 12 cm down: the spring
    np.save(root / "pass_001" / "attractor.npy", dipped)     # pulls there, the arm does not go
    np.savez(root / "pass_001" / "margin.npz", clearance=np.full(n, 0.003),
             tilt=np.zeros(n), source="pass 0")
    verdict, _ = rp.check_pass(root, 1, SHELF)
    assert verdict is None, verdict
    np.savez(root / "pass_001" / "margin.npz", clearance=np.full(n, 0.2),   # the arm straying
             tilt=np.zeros(n), source="pass 0")                           # 20 cm: into the table
    verdict, _ = rp.check_pass(root, 1, SHELF)
    assert verdict is not None and "table" in verdict


def test_attractor_at_interpolates_between_labels():
    a = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    assert np.allclose(rp.attractor_at(a, 0.5), [0.5, 0, 0])
    assert np.allclose(rp.attractor_at(a, 1.5), [2.0, 0, 0])
    assert np.allclose(rp.attractor_at(a, 9.0), [3.0, 0, 0])


def test_pass_0_margin_from_measured_stiffness_checks(tmp_path):
    for k, tilt, tau, across in ((15.0, 4.0, 0.9, 6.0), (60.0, 1.0, 0.4, 5.0)):
        d = tmp_path / f"check_{int(k)}"
        d.mkdir()
        (d / "report.json").write_text(json.dumps({"rotational_stiffness": k, "segments": {
            "hold_start": {"tilt_median_deg": tilt * 0.5}, "hold_end": {"tilt_median_deg": tilt},
            "move_out": {"across_max_mm": across}, "turn_out": {"tau_s": tau},
            "turn_back": {"tau_s": tau * 0.9}}}))
    model = rp.model_from_checks([tmp_path / "check_15", tmp_path / "check_60"])
    assert model["stray_m"] == pytest.approx(0.006)
    assert [r["k"] for r in model["rotation"]] == [15.0, 60.0]
    R = np.repeat(np.eye(3)[None], 10, axis=0)
    period = 0.05 / 0.35
    c, t = rp.first_pass_margin(R, np.full(10, 60.0), period, model)
    assert np.allclose(c, 0.006) and np.allclose(np.degrees(t), 1.0)
    _, t = rp.first_pass_margin(R, np.full(10, 15.0), period, model)
    assert np.allclose(np.degrees(t), 4.0)
    _, t = rp.first_pass_margin(R, np.full(10, 7.5), period, model)        # softer than measured
    assert np.allclose(np.degrees(t), 8.0)
    _, t = rp.first_pass_margin(R, np.full(10, 90.0), period, model)       # stiffer: no less
    assert np.allclose(np.degrees(t), 1.0)
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    rp.use_checks(root, [tmp_path / "check_15", tmp_path / "check_60"])
    n = rp.load_replay(root)[0]["n_labels"]
    _, _, source = rp.pass_margin(root, 0, np.full(n, 60.0))
    assert source == "stiffness checks"


def test_a_check_asked_above_the_cap_counts_as_the_cap(tmp_path):
    """Runs asked for 80 and 100 N m/rad ran at 60 (the controller clamps): they are 60 runs."""
    for name, k, tilt in (("a", 60.0, 0.8), ("b", 100.0, 0.9)):
        d = tmp_path / name
        d.mkdir()
        (d / "report.json").write_text(json.dumps({"rotational_stiffness": k, "segments": {
            "hold_end": {"tilt_median_deg": tilt}, "turn_out": {"tau_s": 0.3},
            "move_out": {"across_max_mm": 5.0}}}))
    model = rp.model_from_checks([tmp_path / "a", tmp_path / "b"])
    assert [r["k"] for r in model["rotation"]] == [60.0]
    assert model["rotation"][0]["tilt_deg"] == 0.9                 # the worse of the two


def test_placement_offset_says_how_to_put_the_object_back():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation

    cloud = np.random.default_rng(0).normal(scale=0.03, size=(200, 3)) * [2, 1, 0.3]
    cloud -= cloud.mean(axis=0)                                  # turning it keeps its centre
    taught = SimpleNamespace(rotation=np.eye(3), rest_index=0, full_cloud=cloud + [0.5, 0.2, 0.0])
    turned = Rotation.from_euler("z", 12, degrees=True).as_matrix()
    now = SimpleNamespace(rotation=turned, rest_index=0,
                          full_cloud=(cloud @ turned.T) + [0.515, 0.192, 0.0])
    off = rp.placement_offset(taught, now)
    assert off["move_x"] == pytest.approx(-0.015, abs=1e-3)       # 15 mm toward the robot
    assert off["move_y"] == pytest.approx(0.008, abs=1e-3)        # 8 mm to its left
    assert off["turn_deg"] == pytest.approx(-12.0, abs=1e-6)      # turn back clockwise
    assert not off["in_place"]
    text = rp.placement_advice(off)
    assert "15 mm toward the robot" in text and "8 mm to the robot's left" in text
    assert "12 deg clockwise" in text
    same = SimpleNamespace(rotation=np.eye(3), rest_index=0, full_cloud=taught.full_cloud + 0.003)
    assert rp.placement_offset(taught, same)["in_place"]
    flipped = SimpleNamespace(rotation=np.eye(3), rest_index=1, full_cloud=taught.full_cloud)
    assert "turn it over" in rp.placement_advice(rp.placement_offset(taught, flipped))
    assert rp.object_name({"bundle": "/x/foam_brick_20260925_183904/source_bundle"}) == "foam_brick"


def test_joint_limit_zones_find_where_a_joint_sits_at_its_limit():
    """The hand turned about its approach axis until joint 7 runs out (as at the 18:39 pick)."""
    from scipy.spatial.transform import Rotation

    from tpgpt.hardware.fr3 import HOME_Q, default_model

    fr3 = default_model()
    T = fr3.tcp(HOME_Q)
    turns = np.linspace(0.0, 2.6, 40)                       # joint 7 starts at 0.785 of 3.016
    zones = {}
    for sign in (1.0, -1.0):
        R = np.stack([T[:3, :3] @ Rotation.from_rotvec([0, 0, sign * a]).as_matrix()
                      for a in turns])
        zones[sign] = rp.joint_limit_zones(np.repeat(T[:3, 3][None], len(turns), axis=0), R,
                                           HOME_Q, fr3)
    found = [z for zs in zones.values() for z in zs if z["joint"] == 7]
    assert found and found[0]["last"] == len(turns) - 1        # at the end of the turn
    assert rp.joint_limit_zones(np.repeat(T[:3, 3][None], 5, axis=0),
                                np.repeat(T[:3, :3][None], 5, axis=0), HOME_Q, fr3) == []


def test_diagnostic_records_are_not_runs(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    info, _ = rp.load_replay(root)
    path = rp.save_diagnostic(root, 0, "partial", label_index=[0.0, 0.35])
    assert path.name == "partial_1.npz"
    assert rp.save_diagnostic(root, 0, "partial", label_index=[0.0]).name == "partial_2.npz"
    assert rp.recordings(root, 0) == [] and rp.pending_pass(root) == (0, 1)



def test_the_bracket_reaches_past_the_taught_path_but_not_where_the_arm_never_went():
    P = np.stack([np.linspace(0.3, 0.6, 20), np.zeros(20), np.full(20, 0.1)], axis=1)
    R = np.repeat(np.eye(3)[None], 20, axis=0)
    above = P + [0.0, 0.0, 0.012]                               # the pass went 12 mm high
    pos, _ = rp.bracket_path(P, R, above, R, 0.0)
    assert np.allclose(pos, above)
    pos, _ = rp.bracket_path(P, R, above, R, 1.0)
    assert np.allclose(pos, P)
    pos, _ = rp.bracket_path(P, R, above, R, 1.5)
    assert np.allclose(pos, P - [0.0, 0.0, 0.006])             # half past it: 6 mm low, not 12
    # after the pass: a run inside the bracket is fine, one beside it is outside
    inside = P + [0.0, 0.0, 0.003]
    beside = P + [0.0, 0.008, 0.0]
    d_in, _ = rp._off_bracket(P, R, above, R, inside, R, 0.5)
    d_out, _ = rp._off_bracket(P, R, above, R, beside, R, 0.5)
    assert d_in.max() < 1e-9 and np.allclose(d_out, 0.008)


def test_segment_windows_stop_at_the_segment():
    info = {"n_labels": 30, "frozen_windows": [[8, 12], [20, 24]]}
    ids = rp.segment_ids(info)
    assert list(ids[[0, 8, 13, 20, 29]]) == [0, 1, 2, 3, 4]
    x = np.zeros(30)
    x[13] = 5.0                                                 # a big stray at the carry's start
    y = rp._segment_max(x, ids, 5)
    assert y[12] == 0.0 and y[13:19].min() == 5.0               # not borrowed by the pick window


def test_the_bracket_check_refuses_an_overshoot_into_the_table(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    info, bundle = rp.load_replay(root)
    n = info["n_labels"]
    P = np.asarray(bundle.tool_labels.positions)
    R = np.asarray(bundle.tool_labels.orientations)
    (root / "pass_001").mkdir()
    np.save(root / "pass_001" / "attractor.npy", P)
    for high, expect in ((0.02, None), (0.20, "table")):        # the last pass went this high
        np.savez(root / "pass_001" / "margin.npz", clearance=np.full(n, 0.002),
                 tilt=np.full(n, 0.01), source="the bracket from pass 0",
                 start_positions=P + [0.0, 0.0, high], start_rotations=R, overshoot=0.5,
                 step=0.25)
        verdict, _ = rp.check_pass(root, 1, SHELF, with_object=False)
        if expect is None:
            assert verdict is None, verdict
        else:
            assert verdict is not None and expect in verdict and "past the taught path" in verdict
