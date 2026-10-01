"""Regression coverage for geometry provenance, refusals and durable evidence."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tpgpt.experiments.audit_artifacts import atomic_json, outcome_summary
from tpgpt.grasp.filters import waypoint_indices


def test_sampling_keeps_critical_endpoints_and_detects_unsampled_fault():
    old = waypoint_indices(10, 4, 3, 7, critical=False)
    new = waypoint_indices(10, 1, 3, 7)
    critical = waypoint_indices(10, 4, 3, 7)
    collision = {2}
    assert not collision.intersection(old)
    assert collision.intersection(new)
    assert {0, 3, 7, 9}.issubset(critical)


@pytest.fixture
def selector(monkeypatch):
    import tpgpt.experiments.pipeline as p
    import tpgpt.grasp.filters as f
    import tpgpt.perception.cameras as c
    import tpgpt.perception.obstacles as o
    monkeypatch.setattr(o, "robot_bodies", lambda env: [])
    monkeypatch.setattr(o, "self_filtered", lambda env, points, bodies: points)
    monkeypatch.setattr(c, "scene_point_cloud", lambda *a, **kw: np.empty((0, 3)))
    monkeypatch.setattr(p, "carry_indices", lambda labels: (1, 2))
    keypoints = NS(points=np.eye(3), is_degenerate=lambda: False)
    monkeypatch.setattr(p, "scene_keypoints", lambda *a, **kw: (keypoints, keypoints, {}))
    monkeypatch.setattr(p, "_select_parts", lambda *a: (keypoints, keypoints))
    class Map:
        def fit(self, *a): return self
        def check_diffeomorphism(self, *a): return NS(consistent_sign=True)
        def transport_positions(self, p): return p
        def transport_orientations(self, p, r): return r
    monkeypatch.setattr(p, "TransportMap", Map)
    seen = []
    def check(env, positions, rotations, pair, scene, points, *a, **kw):
        seen.append(points.copy())
        # Different geometry produces a different decision.
        bad = bool(points[0, 0] > 5)
        return dict(violations=int(bad), first_violation=1 if bad else None,
                    reachable_fraction=1., faults={}, tested=3, joined_at=0)
    monkeypatch.setattr(f, "path_feasibility_observed", check)
    labels = NS(positions=np.zeros((3, 3)), orientations=np.tile(np.eye(3), (3, 1, 1)))
    grasp = NS(pose=np.eye(4))
    source, target = NS(points=np.zeros((1, 3))), NS(points=np.full((1, 3), 10.))
    def choose(**kw):
        return p._choose_grasp(None, source, lambda g: target, labels, (), [grasp], **kw)
    return choose, seen


def test_target_geometry_changes_decision_and_strict_preserves_rejection(selector):
    choose, seen = selector
    audit = {}
    assert choose(selection_mode="strict", audit=audit) is None
    assert seen[-1][0, 0] == 10
    assert audit["path_check"]["examined"][0]["rejected"] == "collision"
    assert choose(selection_mode="strict", target_cloud_correction=False) is not None
    assert seen[-1][0, 0] == 0
    legacy = choose(selection_mode="legacy")
    assert legacy[2]["path_check"]["fell_back"]
    forced = choose(selection_mode="strict", force_diagnostic=True)
    assert forced[2]["path_check"]["fell_back"]


def test_edge_exclusion_is_forwarded_to_the_path_cloud(selector, monkeypatch):
    import tpgpt.perception.cameras as c
    seen = []
    def cloud(*args, **kwargs):
        seen.append(kwargs["target_edge_margin_pixels"])
        return np.empty((0, 3))
    monkeypatch.setattr(c, "scene_point_cloud", cloud)
    choose, _ = selector
    choose(target_edge_margin_pixels=1)
    assert seen == [1]


@pytest.mark.parametrize("unreachable,joined", [([1], 0), ([2], 0), ([], 2)])
def test_task_pose_rejection_cannot_hide_in_high_reachable_fraction(selector, monkeypatch,
                                                                  unreachable, joined):
    import tpgpt.grasp.filters as f
    choose, _ = selector
    def check(*a, **kw):
        assert kw["critical_waypoints"]
        return dict(violations=0, first_violation=None, reachable_fraction=.99,
                    faults={}, tested=100, joined_at=joined,
                    unreachable_waypoints=unreachable)
    monkeypatch.setattr(f, "path_feasibility_observed", check)
    audit = {}
    assert choose(selection_mode="strict", audit=audit) is None
    assert audit["path_check"]["examined"][0]["rejected"] == "task_pose_unreachable"
    legacy = choose(selection_mode="legacy")
    assert not legacy[2]["path_check"]["fell_back"]
    changed = choose(selection_mode="legacy", require_task_waypoints=True)
    assert changed[2]["path_check"]["fell_back"]


def test_strict_funnel_does_not_restore_candidates(monkeypatch):
    import tpgpt.grasp.filters as f
    monkeypatch.setattr(f, "by_support_approach", lambda *a: np.array([], dtype=int))
    monkeypatch.setattr(f, "by_target", lambda grasps, indices, *a: indices)
    monkeypatch.setattr(f, "jaw_width_verdicts", lambda grasps, indices, *a: {int(i): "fits" for i in indices})
    monkeypatch.setattr(f, "by_jaw_width", lambda grasps, indices, *a: indices)
    monkeypatch.setattr(f, "suppress_duplicates", lambda grasps, indices, **kw: indices)
    grasps = [NS(score=1.)]
    strict = f.filter_grasps(grasps, "panda", np.zeros((1, 3)), selection_mode="strict")
    legacy = f.filter_grasps(grasps, "panda", np.zeros((1, 3)), selection_mode="legacy")
    assert len(strict.survivors) == 0
    assert legacy.survivors.tolist() == [0]
    assert legacy.flags["stage_decisions"][0]["accepted"] == []
    assert legacy.flags["stage_decisions"][0]["restored"] is True


def test_summary_includes_setup_error_and_refusal_in_denominator():
    rows = [dict(outcome="success", executed=True), dict(outcome="setup_error"),
            dict(outcome="no_grasp_survived", refused=True),
            dict(outcome="policy_stalled", executed=True)]
    summary = outcome_summary(rows, 5)
    assert summary["overall_success"] == .2
    assert summary["conditional_execution_success"] == .5
    assert summary["setup_errors"] == summary["refusals"] == summary["execution_failures"] == 1


def test_source_roundtrip_and_hash_tampering(tmp_path):
    from tpgpt.experiments.audit_artifacts import save_source, load_source
    from tpgpt.sim.keypoints import GraspFrame, ObjectPlacement
    from tpgpt.transport.labels import PolicyLabels
    labels = PolicyLabels(positions=np.zeros((4, 3)))
    placement = ObjectPlacement(np.ones((8, 3)), GraspFrame([0, 0, 1], [0, 0, -1], [1, 0, 0]),
                                .8, np.ones(3), 1.)
    save_source(tmp_path, labels, placement, True)
    actual = load_source(tmp_path)
    np.testing.assert_array_equal(actual[1].points, placement.points)
    (tmp_path / "placement.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_source(tmp_path)


def test_checkpoint_resume_does_not_build_scene(tmp_path, monkeypatch):
    import tpgpt.experiments.forensics as f
    import tpgpt.experiments.pipeline as p
    import tpgpt.experiments.reshelving_pipeline as source
    monkeypatch.setattr(source, "record_source_placement", lambda *a, **kw: pytest.fail("must reuse source"))
    monkeypatch.setattr(f, "load_source", lambda path: (None, None))
    (tmp_path / "source.json").write_text('{}')
    calls = []
    def broken(*a, **kw):
        calls.append(1)
        raise ValueError("missing scene")
    monkeypatch.setattr(p, "build_scene", broken)
    cells = [dict(gripper="panda", obj="meat", slot="top_middle")]
    out = tmp_path / "out"
    first = f.run_cells(cells, out, tmp_path)
    second = f.run_cells(cells, out, tmp_path)
    assert first == second
    assert len(calls) == 1
    assert first[0]["outcome"] == "setup_error"


@pytest.fixture
def observed_env(monkeypatch):
    import mujoco
    import tpgpt.grasp.grasps as g
    import tpgpt.sim.kinematics as k
    import tpgpt.perception.obstacles as o
    qpos = np.zeros(2)
    data = NS(qpos=qpos, _data=None)
    env = NS(sim=NS(data=data, model=NS(_model=None)), robots=[NS(
        composite_controller=NS(part_controllers={"right": NS(qpos_index=[0])}))])
    monkeypatch.setattr(mujoco, "mj_forward", lambda *a: None)
    monkeypatch.setattr(g, "alignment_rotation", lambda pair: np.eye(3))
    import tpgpt.grasp.filters as f
    monkeypatch.setattr(f, "contact_offset", lambda pair: np.zeros(3))
    monkeypatch.setattr(k, "solve_ik", lambda env, p, *a, **kw: NS(reachable=True, qpos=p[:1]))
    monkeypatch.setattr(o, "points_inside_robot", lambda *a, **kw: {"by_body": {}})
    return env


def test_entry_connection_rejects_collision_between_clear_endpoints(observed_env, monkeypatch):
    import tpgpt.grasp.filters as f
    import tpgpt.perception.obstacles as o
    env = observed_env
    def collision(*a, **kw):
        q = env.sim.data.qpos[0]
        return {"by_body": {"robot0_link": 1} if .4 < q < .6 else {}}
    monkeypatch.setattr(o, "points_inside_robot", collision)
    positions = np.array([[1., 0, 0], [1.1, 0, 0], [1.2, 0, 0]])
    kw = dict(bodies=[], validate_entry=True, stop_early=True, critical_waypoints=True)
    result = f.path_feasibility_observed(env, positions, np.tile(np.eye(3), (3, 1, 1)),
        None, np.ones((1, 3))*4, np.empty((0, 3)), 1, 2, **kw)
    assert result["violations"] == 1
    assert result["entry_connection"]["clear"] is False
    assert result["entry_connection"]["samples"] > 1
    np.testing.assert_array_equal(env.sim.data.qpos, [0, 0])


def test_observed_restores_state_on_exception(observed_env, monkeypatch):
    import tpgpt.grasp.filters as f
    import tpgpt.perception.obstacles as o
    def broken(*a, **kw): raise RuntimeError("checker failure")
    monkeypatch.setattr(o, "points_inside_robot", broken)
    with pytest.raises(RuntimeError, match="checker failure"):
        f.path_feasibility_observed(observed_env, np.ones((3, 3)),
            np.tile(np.eye(3), (3, 1, 1)), None, np.ones((1, 3))*3,
            np.empty((0, 3)), 1, 2, bodies=[])
    np.testing.assert_array_equal(observed_env.sim.data.qpos, [0, 0])


@pytest.mark.parametrize("body,violations", [("gripper0_finger", 0), ("gripper0_palm", 3), ("robot0_link", 3)])
def test_strict_target_contacts_allow_fingers_but_not_palm(observed_env, monkeypatch, body, violations):
    import tpgpt.grasp.filters as f
    import tpgpt.perception.obstacles as o
    monkeypatch.setattr(o, "points_inside_robot", lambda *a, **kw: {"by_body": {body: 1}})
    result = f.path_feasibility_observed(observed_env, np.ones((3, 3)),
        np.tile(np.eye(3), (3, 1, 1)), None, np.empty((0, 3)),
        np.ones((1, 3)), 1, 2, bodies=[], strict_contacts=True)
    assert result["violations"] == violations


def test_support_contact_is_allowed_but_side_contact_is_not(observed_env):
    import tpgpt.grasp.filters as f
    # Initially distant obstacle avoids halo removal; translated load meets it.
    positions = np.array([[0, 0, 0], [0, 0, 1.], [1., 0, 1.]])
    held = np.array([[0., 0, 0], [0., 0, .02]])
    below = np.array([[1., 0, -.003]])
    side = np.array([[1.003, 0, .02]])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    good = f.path_feasibility_observed(observed_env, positions, rotations, None,
        below, held, 1, 2, bodies=[])
    bad = f.path_feasibility_observed(observed_env, positions, rotations, None,
        side, held, 1, 2, bodies=[])
    assert good["violations"] == 0
    assert bad["violations"] > 0


def test_forced_pin_rejects_hash_or_effective_pose_changes():
    from tpgpt.experiments.audit_artifacts import validate_pin
    pose = np.eye(4)
    validate_pin("frozen", pose, "frozen", pose.copy())
    with pytest.raises(ValueError, match="cache"):
        validate_pin("changed", pose, "frozen", pose)
    shifted = pose.copy()
    shifted[2, 3] += .001  # same index, different post-clamp pose
    with pytest.raises(ValueError, match="effective"):
        validate_pin("frozen", shifted, "frozen", pose)


def test_campaign_scene_error_is_checkpointed_and_summarized(tmp_path, monkeypatch):
    import tpgpt.experiments.run_experiments as c
    import tpgpt.experiments.audit_artifacts as a
    monkeypatch.setattr(c, "record_source_placement", lambda *a, **kw: (None, None, True))
    monkeypatch.setattr(a, "save_source", lambda *a: None)
    monkeypatch.setattr(c, "_design_record", lambda *a: {})
    monkeypatch.setattr(c, "write_manifest", lambda *a, **kw: None)
    def broken(*a, **kw): raise ValueError("scene failed")
    monkeypatch.setattr(c, "build_scene", broken)
    result = c.campaign("test", [dict(obj="can", slot="top middle")], tmp_path, verbose=False)
    assert result["successes"] == 0
    assert result["rows"][0]["outcome"] == "setup_error"
    assert (tmp_path / "checkpoint_000.json").exists()


def test_transported_label_pin_includes_orientation_and_gripper():
    from tpgpt.experiments.audit_artifacts import labels_digest
    from tpgpt.transport.labels import PolicyLabels
    labels = PolicyLabels(positions=np.zeros((3, 3)), gripper=np.zeros(3),
                          orientations=np.tile(np.eye(3), (3, 1, 1)))
    before = labels_digest(labels)
    assert before == labels_digest(labels.replace())
    assert before != labels_digest(labels.replace(gripper=np.ones(3)))
    r = labels.orientations.copy()
    r[1] = np.diag([-1., -1., 1.])
    assert before != labels_digest(labels.replace(orientations=r))


def test_collision_between_stride_four_locations_is_checked(observed_env, monkeypatch):
    import tpgpt.grasp.filters as f
    import tpgpt.perception.obstacles as o
    env = observed_env
    def wall(*a, **kw):
        return {"by_body": {"robot0_link": 1} if env.sim.data.qpos[0] == 2 else {}}
    monkeypatch.setattr(o, "points_inside_robot", wall)
    p = np.column_stack((np.arange(10.), np.zeros((10, 2))))
    r = np.tile(np.eye(3), (10, 1, 1))
    kw = dict(bodies=[], stop_early=True)
    old = f.path_feasibility_observed(env, p, r, None, np.ones((1, 3))*20,
                                     np.empty((0, 3)), 3, 7, stride=4, **kw)
    new = f.path_feasibility_observed(env, p, r, None, np.ones((1, 3))*20,
                                     np.empty((0, 3)), 3, 7, stride=1, **kw)
    assert old["violations"] == 0
    assert new["first_violation"] == 2


def test_funnel_checks_effective_pose_without_mutating_cached_candidate(monkeypatch):
    import tpgpt.grasp.filters as f
    raw = NS(pose=np.eye(4), score=1.)
    def clamp(g, height, pair):
        pose = g.pose.copy()
        pose[2, 3] = height + .1
        return NS(pose=pose, score=g.score), .1
    monkeypatch.setattr(f, "clamp_to_support", clamp)
    monkeypatch.setattr(f, "by_support_approach", lambda gs, ii, *a: ii)
    monkeypatch.setattr(f, "by_target", lambda gs, ii, *a: ii)
    monkeypatch.setattr(f, "jaw_width_verdicts", lambda gs, ii, *a: {int(i): "fits" for i in ii})
    monkeypatch.setattr(f, "by_jaw_width", lambda gs, ii, *a: ii)
    monkeypatch.setattr(f, "suppress_duplicates", lambda gs, ii, **kw: ii)
    def collision(gs, ii, *a, **kw):
        return np.asarray([i for i in ii if gs[i].pose[2, 3] > .8], dtype=int)
    monkeypatch.setattr(f, "by_collision", collision)
    args = dict(scene_points=np.ones((1, 3)), selection_mode="strict", support_height=.8)
    before = f.filter_grasps([raw], "panda", np.zeros((1, 3)), **args)
    after = f.filter_grasps([raw], "panda", np.zeros((1, 3)), effective_poses=True, **args)
    assert before.survivors.tolist() == []
    assert after.survivors.tolist() == [0]
    assert after.flags["pose_basis"] == "support_clamped"
    assert raw.pose[2, 3] == 0  # generation inputs and cache identity are unchanged
