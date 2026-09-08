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
    path_deviation,
    point_to_curve_distance,
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


class TestPathDeviation:
    """Measuring how far a trajectory strays from a reference path.

    This is the instrument for integration drift, replacing a comparison of two
    separately minimised distances that could come out negative
    (``ROBOTICS_NOTES`` section 7.26). The properties below are the ones that
    made the old measure unusable, tested directly.
    """

    def _line(self, n=50):
        return np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)

    def test_a_point_on_the_path_is_at_zero_distance(self):
        curve = self._line()
        assert point_to_curve_distance(curve[10][None], curve)[0] == pytest.approx(0.0)

    def test_distance_is_to_the_segment_not_to_the_nearest_vertex(self):
        """The failure mode this replaces: over-reporting by the vertex spacing.

        A point exactly halfway along a coarse segment is *on* the path. Scoring
        it against the nearest vertex would call it half the spacing away --
        millimetres on a 20 Hz demonstration, which is the size of the quantity
        being measured.
        """
        coarse = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        midpoint = np.array([[0.5, 0.0, 0.0]])
        assert point_to_curve_distance(midpoint, coarse)[0] == pytest.approx(0.0)

    def test_deviation_is_never_negative(self):
        """The whole point. A difference of two minima could be, and was."""
        rng = np.random.default_rng(0)
        curve = self._line()
        points = curve[::5] + rng.normal(scale=0.01, size=(10, 3))
        assert (point_to_curve_distance(points, curve) >= 0).all()

    def test_a_perpendicular_offset_is_reported_at_its_true_size(self):
        curve = self._line()
        off = np.array([[0.5, 0.02, 0.0]])
        assert point_to_curve_distance(off, curve)[0] == pytest.approx(0.02, abs=1e-9)

    def test_arriving_late_on_the_same_path_is_not_drift(self):
        """Timing is deliberately excluded: 'off the path' and 'late' differ.

        A trajectory that retraces the reference exactly but only gets halfway
        has strayed nowhere. Mixing the two is what made the earlier numbers
        unreadable, so the separation is asserted rather than assumed.
        """
        curve = self._line()
        slow = curve[: len(curve) // 2]
        assert path_deviation(slow, curve)["max"] == pytest.approx(0.0, abs=1e-12)

    def test_it_reports_where_the_worst_excursion_was(self):
        curve = self._line()
        points = curve.copy()
        points[30, 1] = 0.05
        stats = path_deviation(points, curve)
        assert stats["argmax"] == 30
        assert stats["max"] == pytest.approx(0.05)

    def test_a_single_point_reference_still_works(self):
        """A degenerate path must not raise; it has no segments to project onto."""
        d = point_to_curve_distance(np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 0]]))
        assert d[0] == pytest.approx(1.0)
