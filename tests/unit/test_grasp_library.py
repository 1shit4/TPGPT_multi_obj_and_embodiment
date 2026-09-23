"""The fixed per-object grasp library, and the coverage it lets us measure.

Why these two things are tested together: the library exists so that a plan does
not depend on a random draw from GraspGen-X, and coverage is the question that
only becomes answerable once the draw is gone -- "of this object's grasps, how
many reach that destination?". The properties that matter are that the library
is *the same* every time (so two runs differ only in the task), that carrying it
into the scene is exact, and that the funnel coverage prints adds up.

No server and no YCB scan are needed here: the generator and the scan are stubbed,
which is the point -- everything below is about the bookkeeping around them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.hardware import coverage as coverage_mod
from tpgpt.hardware import grasp_library
from tpgpt.hardware.end_effector import FRANKA_HAND
from tpgpt.hardware.grasp_selection import SelectionSettings
from tpgpt.hardware.shelf import ShelfWorld

from tests.unit.test_hardware import synthetic_bundle
from tests.unit.test_grasp_selection import CUP_CENTRE, cup_cloud, graspgen_pose, top_down

TABLE = -0.10


def fake_poses(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return poses, np.linspace(1.0, 0.5, n)


# ------------------------------------------------------------------- library
def test_the_library_is_asked_for_once_and_then_comes_from_disk(tmp_path, monkeypatch):
    """The second call must not reach the sampler, and must return the same arrays."""
    rng = np.random.default_rng(0)
    poses, scores = fake_poses(8, rng)
    calls = []

    monkeypatch.setattr(grasp_library, "canonical_cloud",
                        lambda *a, **k: rng.normal(size=(64, 3)) * 0.0 + 1.0)

    def sampler(cloud, **kw):
        calls.append(kw)
        return poses, scores

    import tpgpt.hardware.grasp_selection as gs
    monkeypatch.setattr(gs, "request_candidates", sampler)

    first = grasp_library.library("banana", size=8, cache_dir=tmp_path)
    second = grasp_library.library("banana", size=8, cache_dir=tmp_path)
    assert len(calls) == 1, "the second call went to the sampler instead of the cache"
    assert calls[0]["topk"] == 8 and calls[0]["num_grasps"] == 16
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert grasp_library.library("banana", size=8, cache_dir=tmp_path, rebuild=True)
    assert len(calls) == 2, "rebuild=True must ask again"


def test_a_different_size_is_a_different_library(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    monkeypatch.setattr(grasp_library, "canonical_cloud", lambda *a, **k: np.ones((64, 3)))
    import tpgpt.hardware.grasp_selection as gs
    monkeypatch.setattr(gs, "request_candidates",
                        lambda cloud, **kw: fake_poses(int(kw["topk"]), rng))
    assert len(grasp_library.library("banana", size=4, cache_dir=tmp_path)[0]) == 4
    assert len(grasp_library.library("banana", size=6, cache_dir=tmp_path)[0]) == 6


def test_placing_the_library_is_the_rigid_transform_itself():
    """A library pose carried into the scene is the fit applied to it, exactly."""
    rng = np.random.default_rng(2)
    poses, _ = fake_poses(5, rng)
    poses[:, :3, :3] = np.linalg.qr(rng.normal(size=(5, 3, 3)))[0]
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    position = np.array([0.4, -0.2, 0.05])

    out = grasp_library.placed(poses, rotation, position)
    for before, after in zip(poses, out):
        assert np.allclose(after[:3, 3], rotation @ before[:3, 3] + position)
        assert np.allclose(after[:3, :3], rotation @ before[:3, :3])
    # ... and the grasp keeps its geometry: a point held in the hand stays held.
    held = np.array([0.0, 0.0, 0.01, 1.0])
    for before, after in zip(poses, out):
        moved = rotation @ (before @ held)[:3] + position
        assert np.allclose((after @ held)[:3], moved)


# ------------------------------------------------------------------ coverage
@pytest.fixture(scope="module")
def scene():
    """A cup on a table, grasps on its rim all round, and a demonstration to transport."""
    source = synthetic_bundle(table=TABLE, shelf=0.15)
    cloud = cup_cloud()
    poses = []
    for angle in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tcp = np.array([*CUP_CENTRE, TABLE + 0.10 - 0.012]) + 0.0435 * radial
        poses.append(graspgen_pose(tcp, top_down(radial, tcp)))
    return source, cloud, np.stack(poses)


def run(scene, destination, poses=None, **kw):
    source, cloud, library_poses = scene
    return coverage_mod.coverage(
        "cup", None, ShelfWorld(model=None, table_height=TABLE), source,
        destination, destination[2], TABLE, camera_position=np.array([1.2, 0.5, 0.9]),
        start_q=None, end_effector=FRANKA_HAND, cloud=cloud,
        poses=library_poses if poses is None else poses,
        settings=SelectionSettings(check_visibility=False, check_centred=False,
                                   kinematic_criterion="sim"),
        **kw)


def test_the_funnel_adds_up_and_the_rim_grasps_reach_the_shelf(scene):
    report = run(scene, np.array([0.55, 0.25, 0.15]), plan_budget=4)
    assert report["library"] == 8
    assert report["stages"][0][1] == 8, "the funnel starts at the whole library"
    for _, before, after, _ in report["stages"]:
        assert after <= before
    assert report["screened_survivors"] > 0, "rim grasps must survive screening"
    assert report["planned"] == min(4, report["screened_survivors"]), "the budget is the bound"
    assert report["admissible"] > 0, "a reachable shelf must admit at least one rim grasp"
    assert report["first_rank"] == 0 or report["first_rank"] < report["planned"]
    assert "cup:" in coverage_mod.describe(report)


def test_grasps_through_the_table_do_not_survive_screening(scene):
    """The funnel must remove what cannot be picked, not carry it to the planner."""
    _, _, poses = scene
    under = poses.copy()
    under[:, 2, 3] -= 0.09                              # hand driven down through the table
    report = run(scene, np.array([0.55, 0.25, 0.15]), poses=under, plan_budget=4)
    assert report["admissible"] == 0
    assert report["screen_rejections"], "a refusal must say which stage removed them"


def test_an_unreachable_destination_admits_nothing(scene):
    """Four metres away: what survives screening must die in the path check."""
    report = run(scene, np.array([4.0, 0.0, 0.15]), plan_budget=4)
    assert report["admissible"] == 0
    assert report["why_the_planned_ones_failed"], "and it must say why"
