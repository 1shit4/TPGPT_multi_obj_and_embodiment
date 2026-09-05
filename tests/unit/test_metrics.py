"""Curve metrics and the Mann-Whitney ranking -- paper Sec. IV-B."""

import numpy as np
import pytest

from tpgpt.metrics import (
    area_between_curves,
    compare_trajectories,
    dynamic_time_warping,
    final_approach_angle,
    final_position_error,
    frechet_distance,
    rank_methods,
)


@pytest.fixture
def line():
    t = np.linspace(0, 1, 50)
    return np.c_[t, np.zeros(50)]


def test_identical_curves_score_zero(line):
    assert frechet_distance(line, line) == pytest.approx(0.0, abs=1e-12)
    assert dynamic_time_warping(line, line) == pytest.approx(0.0, abs=1e-9)
    assert area_between_curves(line, line) == pytest.approx(0.0, abs=1e-9)
    assert final_position_error(line, line) == pytest.approx(0.0, abs=1e-12)


def test_constant_offset_gives_that_offset(line):
    """A known answer: a curve shifted by d has Frechet distance d."""
    shifted = line + np.array([0.0, 0.25])
    assert frechet_distance(line, shifted) == pytest.approx(0.25, abs=1e-9)
    assert area_between_curves(line, shifted) == pytest.approx(0.25, rel=1e-6)


def test_final_position_error_uses_the_endpoints(line):
    other = line.copy()
    other[-1] += np.array([0.3, 0.4])
    assert final_position_error(line, other) == pytest.approx(0.5)


def test_approach_angle_of_perpendicular_curves():
    a = np.c_[np.linspace(0, 1, 20), np.zeros(20)]
    b = np.c_[np.zeros(20), np.linspace(0, 1, 20)]
    assert final_approach_angle(a, b) == pytest.approx(np.pi / 2, abs=1e-9)


def test_approach_angle_is_zero_for_parallel_approaches(line):
    assert final_approach_angle(line, line + np.array([0.0, 0.5])) == pytest.approx(0.0, abs=1e-9)


def test_area_requires_planar_curves():
    curve = np.zeros((10, 3))
    with pytest.raises(ValueError, match="2-D"):
        area_between_curves(curve, curve)


def test_compare_trajectories_skips_area_in_three_dimensions():
    a = np.random.default_rng(0).normal(size=(30, 3))
    metrics = compare_trajectories(a, a + 0.01)
    assert "area" not in metrics
    assert set(metrics) == {"frechet", "dtw", "final_position_error", "final_approach_angle"}


class TestRanking:
    def test_a_clearly_better_method_wins(self):
        rng = np.random.default_rng(0)
        scores = {
            "good": rng.normal(1.0, 0.2, 40),
            "mediocre": rng.normal(3.0, 0.2, 40),
            "poor": rng.normal(6.0, 0.2, 40),
        }
        result = rank_methods(scores)
        assert result.ordered() == ["good", "mediocre", "poor"]
        assert result.points["good"] == 2
        assert result.ranks["good"] == 1

    def test_indistinguishable_methods_share_a_rank(self):
        rng = np.random.default_rng(1)
        scores = {"a": rng.normal(1.0, 0.5, 40), "b": rng.normal(1.0, 0.5, 40)}
        result = rank_methods(scores)
        assert result.points == {"a": 0, "b": 0}
        assert result.ranks["a"] == result.ranks["b"]

    def test_direction_can_be_flipped(self):
        rng = np.random.default_rng(2)
        scores = {"high": rng.normal(5.0, 0.2, 30), "low": rng.normal(1.0, 0.2, 30)}
        assert rank_methods(scores, lower_is_better=False).ordered()[0] == "high"

    def test_wins_record_the_p_values(self):
        rng = np.random.default_rng(3)
        result = rank_methods(
            {"a": rng.normal(1.0, 0.1, 30), "b": rng.normal(4.0, 0.1, 30)}
        )
        assert len(result.wins) == 1
        winner, loser, p = result.wins[0]
        assert (winner, loser) == ("a", "b") and p < 0.05
