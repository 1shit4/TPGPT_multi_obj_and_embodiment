"""The campaign's record of what cannot be reconstructed afterwards."""

import numpy as np
import pytest

from tpgpt.reporting.record import (
    FINGERPRINT_POINTS,
    Timings,
    condition_block,
    condition_id,
    fingerprint,
    fingerprint_grid,
    fingerprint_spec,
    pose_matrix,
)


def test_the_fingerprint_grid_is_the_same_every_time_it_is_asked_for():
    """Two cells' fingerprints are comparable only if the grid is identical.

    Not a tautology about ``linspace``: the grid is built from module constants
    and flattened in a stated order, and a comparison of two maps silently
    becomes meaningless if either moves.
    """
    assert np.array_equal(fingerprint_grid(), fingerprint_grid())
    assert fingerprint_grid().shape == (FINGERPRINT_POINTS ** 3, 3)
    spec = fingerprint_spec()
    assert spec["points_per_axis"] == FINGERPRINT_POINTS


def test_the_fingerprint_of_an_identity_map_is_zero_everywhere():
    """``phi(x) - x`` and not ``phi(x)``, which is the whole storage argument."""

    class Identity:
        def transport_positions(self, X):
            return np.asarray(X, dtype=float)

    displacement = fingerprint(Identity())
    assert displacement.dtype == np.float32
    assert np.abs(displacement).max() == 0.0


def test_the_fingerprint_records_the_displacement_a_map_applies():
    shift = np.array([0.01, -0.02, 0.03])

    class Shift:
        def transport_positions(self, X):
            return np.asarray(X, dtype=float) + shift

    displacement = fingerprint(Shift())
    assert np.allclose(displacement, shift, atol=1e-7)


def test_a_pose_is_stored_as_a_matrix_so_it_carries_no_sign_ambiguity():
    """A quaternion and its negation are the same rotation; a matrix is not.

    This project lost twenty cells to an undetected half turn between two
    frames, so the stored representation is the one that cannot express the
    ambiguity.
    """
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = np.asarray(pose_matrix([1.0, 2.0, 3.0], rotation))
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[:3, :3], rotation)
    assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_the_condition_block_lists_every_condition_by_value():
    """A reader must not have to find the commit to learn what ``P1`` was."""
    picks = {"P0": (-0.14, -0.15, 0.0), "P1": (-0.21, 0.02, 55.0)}
    block = condition_block(("P0", "P1"), ("top_middle", "top_left"), picks)
    assert len(block["conditions"]) == 4
    assert condition_id("P1", "top_left") in block["conditions"]
    assert block["pick_configs"]["P1"]["yaw_deg"] == 55.0


def test_a_condition_id_distinguishes_pick_from_destination():
    assert condition_id("P0", "top_middle") != condition_id("P0", "top_left")
    assert condition_id("P0", "top_middle") != condition_id("P1", "top_middle")


def test_timings_record_one_entry_per_stage():
    timings = Timings()
    with timings.stage("map_fit"):
        pass
    with timings.stage("policy_refit"):
        pass
    assert set(timings.stages) == {"map_fit", "policy_refit"}
    assert all(v >= 0.0 for v in timings.stages.values())
    assert set(timings.as_dict()) == {"seconds_map_fit", "seconds_policy_refit"}


def test_a_stage_that_raises_still_records_its_time():
    """Otherwise a failed cell loses the cost of the stage that failed."""
    timings = Timings()
    with pytest.raises(ValueError):
        with timings.stage("map_fit"):
            raise ValueError("boom")
    assert "map_fit" in timings.stages
