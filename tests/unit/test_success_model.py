"""The outcome-space test of the cross-gripper claim, on rows whose answer is known."""

import numpy as np
import pytest

from tpgpt.metrics.success_model import factor_effects, mixed_effects, per_level_table, wilson


def _rows(rng, grippers, objects, seeds, p_by_gripper):
    rows = []
    for s in seeds:
        for g in grippers:
            for o in objects:
                for c in ("P0|top_middle", "P1|top_left"):
                    rows.append({
                        "gripper": g, "object": o, "destination": c.split("|")[1],
                        "seed": s, "condition_id": f"{c}|{s}",
                        "outcome": "success" if rng.random() < p_by_gripper[g] else "failed",
                    })
    return rows


class TestWilson:
    def test_known_interval(self):
        lo, hi = wilson(84, 210)
        assert lo == pytest.approx(0.336, abs=1e-3)
        assert hi == pytest.approx(0.467, abs=1e-3)

    def test_zero_trials_is_nan_not_a_number(self):
        assert all(np.isnan(v) for v in wilson(0, 0))


class TestFactorEffects:
    def test_a_hand_that_never_places_makes_the_gripper_factor_significant(self):
        rng = np.random.default_rng(0)
        rows = _rows(rng, ("a", "b", "c"), ("x", "y", "z"), (0, 1, 2),
                     {"a": 0.8, "b": 0.8, "c": 0.02})
        fe = factor_effects(rows)
        assert fe["factors"]["gripper"]["p"] < 0.01
        assert fe["factors"]["gripper"]["df"] == 2
        assert fe["factors"]["object"]["p"] > 0.05

    def test_indifferent_hands_are_not_significant(self):
        rng = np.random.default_rng(1)
        rows = _rows(rng, ("a", "b", "c"), ("x", "y", "z"), (0, 1, 2),
                     {"a": 0.5, "b": 0.5, "c": 0.5})
        fe = factor_effects(rows)
        assert fe["factors"]["gripper"]["p"] > 0.05
        assert "HC1" in fe["covariance"], "18 clusters is below MIN_CLUSTERS"

    def test_enough_clusters_switches_to_cluster_robust(self):
        rng = np.random.default_rng(5)
        rows = _rows(rng, ("a", "b"), ("x", "y"), tuple(range(12)), {"a": 0.5, "b": 0.5})
        assert "cluster-robust" in factor_effects(rows)["covariance"]  # 24 clusters

    def test_a_factor_with_one_level_is_dropped(self):
        rng = np.random.default_rng(2)
        rows = _rows(rng, ("a", "b"), ("x",), (0, 1, 2), {"a": 0.5, "b": 0.5})
        fe = factor_effects(rows)
        assert "object" not in fe["factors"]
        assert "gripper" in fe["factors"]


class TestMixedEffects:
    def test_one_seed_has_no_random_effect_to_fit(self):
        rng = np.random.default_rng(3)
        rows = _rows(rng, ("a", "b"), ("x", "y"), (0,), {"a": 0.5, "b": 0.5})
        assert mixed_effects(rows) is None

    def test_three_seeds_fit(self):
        rng = np.random.default_rng(4)
        rows = _rows(rng, ("a", "b", "c"), ("x", "y"), (0, 1, 2),
                     {"a": 0.8, "b": 0.5, "c": 0.1})
        me = mixed_effects(rows)
        assert me["groups"] == 3
        assert me["fixed_effects"]["C(gripper)[T.c]"]["mean"] < 0


def test_per_level_table_counts():
    rows = [{"gripper": "a", "outcome": "success"}, {"gripper": "a", "outcome": "failed"},
            {"gripper": "b", "outcome": "success"}]
    t = per_level_table(rows)
    assert t["a"]["placed"] == 1 and t["a"]["n"] == 2
    assert t["b"]["rate"] == 1.0
