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


def _run_pass(root, k, info):
    """What the executor's replay mode records: the arm at each 50 ms step, index advancing 0.35."""
    attractor = rp.load_attractor(root, k)
    n = len(attractor)
    dt = info["dt"]
    step_index = info["speed_scale"]                       # labels per control step
    x = attractor[0] + SAG
    s, rows = 0.0, {"label_index": [], "positions": [], "step": []}
    step = 0
    while True:
        rows["label_index"].append(s)
        rows["positions"].append(x.copy())
        rows["step"].append(step)
        a = rp.attractor_at(attractor, s)
        for _ in range(10):                                # the arm, integrated finely
            x = x + (a + SAG - x) * (dt / 10 / TAU)
        if s >= n - 1:
            break
        s = min(s + step_index, n - 1)
        step += 1
    rp.save_recording(root, k, **rows)


def test_the_passes_learn_the_attractor(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    info, bundle = rp.load_replay(root)
    assert info["grasp_index"] < info["release_index"]
    assert rp.pending_pass(root) == 0
    assert np.array_equal(rp.load_attractor(root, 0), bundle.tool_labels.positions)
    errors = []
    for k in range(4):
        _run_pass(root, k, info)
        with pytest.raises(FileExistsError):
            rp.pending_pass(root)                          # recorded: update first
        report = rp.update(root)
        errors.append(report["error_rms_mm"])
        assert rp.pending_pass(root) == k + 1
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < errors[0] / 4
    rows = rp.status(root)
    assert [r["recorded"] for r in rows] == [True] * 4 + [False]
    assert json.loads((root / "pass_000" / "report.json").read_text())["gain"] == 0.5


def test_init_refuses_to_overwrite_and_update_needs_a_recording(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    with pytest.raises(FileExistsError):
        rp.init_replay(tmp_path / "bundle", root)
    with pytest.raises(FileNotFoundError):
        rp.update(root)


SHELF = (__import__("pathlib").Path(__file__).resolve().parents[2]
         / "ycb_pointcloud_pipeline" / "config" / "shelf.yaml")


def test_a_pass_into_the_table_is_refused(tmp_path):
    root = rp.init_replay(_taught_bundle(tmp_path), tmp_path / "replay")
    verdict, _ = rp.check_pass(root, 0, SHELF)
    assert verdict is None, verdict                          # the taught path is clear
    dipped = rp.load_attractor(root, 0).copy()
    dipped[20:40, 2] -= 0.12                                 # a learned attractor 12 cm too low
    (root / "pass_001").mkdir()
    np.save(root / "pass_001" / "attractor.npy", dipped)
    verdict, _ = rp.check_pass(root, 1, SHELF)
    assert verdict is not None and "table" in verdict


def test_attractor_at_interpolates_between_labels():
    a = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    assert np.allclose(rp.attractor_at(a, 0.5), [0.5, 0, 0])
    assert np.allclose(rp.attractor_at(a, 1.5), [2.0, 0, 0])
    assert np.allclose(rp.attractor_at(a, 9.0), [3.0, 0, 0])
