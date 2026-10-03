"""The Franka Hand's grasp screening, pinned before the multi-finger hand changed shared code.

Adding the Inspire hand (2026-10-04) touches the code every Franka Hand plan
goes through: ``screen_candidates``, the funnel's gripper pair, the jaw stage,
the distinct stage. The Franka runs are finished, and their numbers stand only
if the method they ran is unchanged. This pins it: 60 seeded GraspGen-X-like
poses on the test cup, screened twice (no context; a camera and a shelf slot
over the destination), and every candidate's outcome -- rejection reason, the
grasp point after refinement, the refinement itself -- compared with
``tests/data/franka_screen_baseline.json``, written from the code at 0aceaec.

Regenerate (only when a change to the Franka Hand's screening is intended, and
say so in the commit)::

    ~/venvs/tpgpt/bin/python -m tests.unit.test_franka_regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tpgpt.hardware import hand
from tpgpt.hardware.grasp_selection import ScreenContext, screen_candidates

from tests.unit.test_grasp_selection import CUP_CENTRE, TABLE, _slot_over_destination, cup_cloud

BASELINE = Path(__file__).resolve().parents[1] / "data" / "franka_screen_baseline.json"
N_POSES = 60
SEED = 20261004


def _poses() -> tuple[np.ndarray, np.ndarray]:
    """GraspGen-X-convention poses (gripper base) around the cup, tilts 0-110 deg."""
    rng = np.random.default_rng(SEED)
    poses, scores = [], []
    for _ in range(N_POSES):
        tilt = np.radians(rng.uniform(0.0, 110.0))
        yaw = rng.uniform(-np.pi, np.pi)
        approach = np.array([np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), -np.cos(tilt)])
        side = rng.normal(size=3)
        closing = side - approach * float(side @ approach)
        closing /= np.linalg.norm(closing)
        R = np.column_stack([closing, np.cross(approach, closing), approach])
        r = rng.uniform(0.0, 0.06)
        th = rng.uniform(-np.pi, np.pi)
        tcp = np.array([CUP_CENTRE[0] + r * np.cos(th), CUP_CENTRE[1] + r * np.sin(th),
                        TABLE + rng.uniform(0.0, 0.13)])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp - approach * hand.TCP_OFFSET
        poses.append(T)
        scores.append(rng.uniform(0.0, 1.0))
    return np.stack(poses), np.array(scores)


def _contexts() -> dict:
    from tpgpt.hardware.shelf import ShelfWorld

    return {
        "bare": None,
        "camera_and_shelf": ScreenContext(
            camera_position=np.array([1.2, 0.5, 0.87]),
            world=ShelfWorld.build(_slot_over_destination(roof=0.45)),
            destination=np.array([0.55, 0.25]), destination_height=0.15),
    }


def run() -> dict:
    """Screen the pinned poses in each context; a JSON-ready record."""
    cloud = cup_cloud()
    poses, scores = _poses()
    out = {}
    for name, ctx in _contexts().items():
        screened = screen_candidates(cloud, poses, scores, TABLE, context=ctx)
        out[name] = {
            "stages": [[str(s[0]), int(s[1]), int(s[2]), bool(s[3])] for s in screened.stages],
            "candidates": [{
                "index": c.index,
                "rejected": c.rejected,
                "tcp": np.round(c.grasp.tcp, 6).tolist(),
                "closing": np.round(c.grasp.closing, 6).tolist(),
                "refined_depth": round(float(c.refined_depth), 6),
                "refined_centre": round(float(c.refined_centre), 6),
            } for c in screened],
        }
    return out


def test_the_franka_hand_screening_is_unchanged():
    expected = json.loads(BASELINE.read_text())
    got = run()
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name]["stages"] == expected[name]["stages"], name
        assert len(got[name]["candidates"]) == len(expected[name]["candidates"])
        for g, e in zip(got[name]["candidates"], expected[name]["candidates"]):
            assert g["index"] == e["index"], name
            assert g["rejected"] == e["rejected"], (name, g["index"])
            assert np.allclose(g["tcp"], e["tcp"], atol=1e-6), (name, g["index"])
            assert np.allclose(g["closing"], e["closing"], atol=1e-6), (name, g["index"])
            assert abs(g["refined_depth"] - e["refined_depth"]) < 1e-6
            assert abs(g["refined_centre"] - e["refined_centre"]) < 1e-6


def test_the_baseline_exercises_every_stage():
    """A fixture that never reaches a stage pins nothing about it."""
    expected = json.loads(BASELINE.read_text())
    reasons = {(c["rejected"] or "admissible").split(":")[0]
               for ctx in expected.values() for c in ctx["candidates"]}
    stages = {s[0] for ctx in expected.values() for s in ctx["stages"]}
    assert {"from_below", "on_target", "jaws", "distinct"} <= stages
    assert {"visibility", "grasp_collision", "place_zone", "centred"} <= stages
    assert "admissible" in reasons and "from_below" in reasons
    assert len(reasons) >= 4


if __name__ == "__main__":
    BASELINE.write_text(json.dumps(run(), indent=1) + "\n")
    print(f"wrote {BASELINE}")
