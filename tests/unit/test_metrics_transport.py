"""What a transportation map does to orientation.

The property under test throughout: ``J_perp`` is one rotation per point, and
Eq. 11 uses that same rotation for the gripper. So the angle the map adds to the
hand and the angle it adds to the vertical are the *same number*, and any test
that confirms one must confirm the other on the same map. A metric that reported
a corrected gripper without reporting the tilt it cost would be exactly the kind
of half-measurement that ROBOTICS_NOTES 7.27 was written about.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.metrics.transport import (
    JAW_SYMMETRY,
    lift_deviation,
    orientation_transport_error,
    tilt_profile,
    vertical_tilt,
)
from tpgpt.transport.maps import TransportMap
from tpgpt.utils.rotations import is_rotation


def rot(axis: str, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def rigid_map(rotation: np.ndarray, translation=(0.0, 0.0, 0.0)) -> TransportMap:
    """A map whose warp is exactly one rigid motion, fitted on cube corners.

    Eight corners of a cube determine an affine map uniquely, so the residual has
    nothing left to explain and ``phi`` reduces to ``gamma``. That makes the
    expected ``J_perp`` known exactly rather than approximately.
    """
    from tpgpt.sim.keypoints import CUBE_CORNERS

    S = np.vstack([np.zeros(3), CUBE_CORNERS * 0.05])
    T = S @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)
    return TransportMap().fit(S, T)


class TestVerticalTilt:
    def test_a_pure_translation_leaves_vertical_alone(self):
        tilt = vertical_tilt(rigid_map(np.eye(3), (0.3, -0.2, 0.7)), np.zeros((1, 3)))
        assert tilt[0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("degrees", [10.0, 30.0, 45.0, 90.0])
    def test_a_rotation_about_a_horizontal_axis_tilts_vertical_by_that_angle(
        self, degrees
    ):
        tilt = vertical_tilt(rigid_map(rot("x", degrees)), np.zeros((1, 3)))
        assert tilt[0] == pytest.approx(degrees, abs=1e-4)

    def test_a_yaw_about_the_vertical_does_not_tilt_it(self):
        """The distinction the task frame is built on.

        ``task_frame`` keeps the support normal as one axis, so it can only ever
        produce a yaw -- and a yaw costs no tilt. That is the whole reason the
        task frame is safe and the full grasp pose is not.
        """
        tilt = vertical_tilt(rigid_map(rot("z", 75.0)), np.zeros((1, 3)))
        assert tilt[0] == pytest.approx(0.0, abs=1e-6)


class TestOrientationTransportError:
    def test_a_map_carrying_the_hand_exactly_scores_zero(self):
        R_source = np.eye(3)
        for degrees in (0.0, 15.0, 30.0):
            R = rot("x", degrees)
            err = orientation_transport_error(
                rigid_map(R), np.zeros((1, 3)), R_source, R @ R_source
            )
            assert err[0] == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("degrees", [15.0, 30.0, 45.0])
    def test_a_map_that_carries_no_rotation_scores_the_full_disagreement(self, degrees):
        """The task-frame cube's failure mode, stated as a number.

        The map is the identity, so it delivers the source orientation unchanged
        while the target grasp is tilted. The error is the whole tilt -- which is
        exactly what was measured for a grasp-centred cube built in the task
        frame.
        """
        err = orientation_transport_error(
            rigid_map(np.eye(3)), np.zeros((1, 3)), np.eye(3), rot("x", degrees)
        )
        assert err[0] == pytest.approx(degrees, abs=1e-4)

    def test_the_tilt_it_costs_equals_the_orientation_it_buys(self):
        """The conservation law, asserted directly on one map.

        A map cannot rotate the gripper without rotating the space around it,
        because Eq. 11 uses the same ``J_perp`` for both. Any design that claims
        to buy orientation for free has a bug, and this is the test that would
        catch it.
        """
        for degrees in (10.0, 30.0, 60.0):
            m = rigid_map(rot("y", degrees))
            bought = degrees - orientation_transport_error(
                m, np.zeros((1, 3)), np.eye(3), rot("y", degrees)
            )
            cost = vertical_tilt(m, np.zeros((1, 3)))
            assert bought[0] == pytest.approx(cost[0], abs=1e-4)


class TestJawSymmetry:
    def test_the_symmetry_is_a_half_turn_about_the_approach_axis(self):
        assert is_rotation(JAW_SYMMETRY)
        # columns are (closing, jaw, approach): the first two flip, the third holds
        assert np.allclose(JAW_SYMMETRY @ np.array([0, 0, 1.0]), [0, 0, 1.0])
        assert np.allclose(JAW_SYMMETRY @ np.array([1.0, 0, 0]), [-1.0, 0, 0])

    def test_a_grasp_rolled_over_is_the_same_grasp(self):
        """Without this, a perfectly correct grasp scores 180 degrees wrong.

        A parallel jaw closing along ``+c`` and along ``-c`` is one physical
        grasp. ``task_frame`` carries the same lesson for the keypoint frame.
        """
        target = rot("z", 20.0)
        rolled = target @ JAW_SYMMETRY
        m = rigid_map(np.eye(3))
        assert orientation_transport_error(
            m, np.zeros((1, 3)), target, rolled, symmetric=True
        )[0] == pytest.approx(0.0, abs=1e-6)
        assert orientation_transport_error(
            m, np.zeros((1, 3)), target, rolled, symmetric=False
        )[0] == pytest.approx(180.0, abs=1e-6)


class TestTiltProfile:
    def _path(self, n=40):
        return np.column_stack(
            [np.linspace(0, 0.2, n), np.zeros(n), np.linspace(0, 0.1, n)]
        )

    def test_an_untilted_map_reports_zero_everywhere(self):
        out = tilt_profile(rigid_map(np.eye(3), (0.1, 0.1, 0)), self._path())
        assert out["tilt_max"] == pytest.approx(0.0, abs=1e-6)
        assert out["tilt_mid_path"] == pytest.approx(0.0, abs=1e-6)

    def test_a_uniformly_rotated_map_reports_the_same_tilt_mid_path_as_anywhere(self):
        """A single map rotation tilts the *whole* path, which is the cost a
        local correction is meant to avoid. Distinguishing that from a tilt
        confined to the grasp is the entire purpose of reporting a profile."""
        out = tilt_profile(rigid_map(rot("x", 25.0)), self._path())
        assert out["tilt_max"] == pytest.approx(25.0, abs=1e-3)
        assert out["tilt_mid_path"] == pytest.approx(25.0, abs=1e-3)

    def test_the_indexed_readings_are_reported_only_when_asked_for(self):
        path = self._path()
        assert "tilt_at_grasp" not in tilt_profile(rigid_map(np.eye(3)), path)
        out = tilt_profile(rigid_map(np.eye(3)), path, grasp_index=5, release_index=30)
        assert "tilt_at_grasp" in out and "tilt_at_release" in out

    def test_an_out_of_range_index_is_clamped_rather_than_crashing(self):
        out = tilt_profile(rigid_map(np.eye(3)), self._path(), grasp_index=10_000)
        assert np.isfinite(out["tilt_at_grasp"])


class TestLiftDeviation:
    def test_an_untilted_map_leaves_a_vertical_lift_vertical(self):
        path = np.array([[0, 0, 0.0], [0, 0, 0.15]])
        assert lift_deviation(
            rigid_map(np.eye(3), (0.2, 0.1, 0.0)), path, 0, 1
        ) == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("degrees", [15.0, 30.0])
    def test_a_tilted_map_makes_the_lift_diagonal_by_that_angle(self, degrees):
        path = np.array([[0, 0, 0.0], [0, 0, 0.15]])
        assert lift_deviation(rigid_map(rot("x", degrees)), path, 0, 1) == pytest.approx(
            degrees, abs=1e-3
        )

    def test_a_zero_length_segment_is_nan_not_zero(self):
        """'Not measurable' must never come back as a plausible number -- the
        recurring lesson of 7.13, where an unmeasured offset returned zero."""
        path = np.array([[0, 0, 0.0], [0, 0, 0.0]])
        assert np.isnan(lift_deviation(rigid_map(np.eye(3)), path, 0, 1))
