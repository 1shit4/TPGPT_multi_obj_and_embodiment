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
    # every later pass's margin is measured: at least the run-to-run spread
    margin = np.load(root / "pass_001" / "margin.npz")
    assert str(margin["source"]) == "pass 0"
    assert margin["clearance"].min() > 0.0
    assert margin["clearance"].max() * 1000 >= first["stray_max_mm"] - 1e-6
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


def test_pass_0_margin_scales_with_the_rotational_stiffness():
    R = np.repeat(np.eye(3)[None], 40, axis=0)                # the hand not turning
    clearance, tilt = rp.first_pass_margin(R, np.full(40, 30.0), 0.05 / 0.35)
    assert np.allclose(clearance, rp.FIRST_PASS_STRAY)
    assert np.allclose(np.degrees(tilt), 1.5)                  # the measured resting tilt
    _, stiffer = rp.first_pass_margin(R, np.full(40, 60.0), 0.05 / 0.35)
    assert np.allclose(np.degrees(stiffer), 0.75)
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
