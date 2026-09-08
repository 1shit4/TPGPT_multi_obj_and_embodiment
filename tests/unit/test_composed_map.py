"""Two transportation maps applied in sequence.

The property that motivates the class: **determinants multiply**, so each stage
can be verified on its own and a composite of two diffeomorphisms is one. A
convex blend has no such guarantee -- its Jacobian picks up a rank-one cross term
that can flip the sign by itself, discoverable only by sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.transport.affine import AffineMap, IdentityAffine
from tpgpt.transport.gp import GaussianProcessRegressor
from tpgpt.transport.maps import ComposedMap, TransportMap, fit_local_correction


@pytest.fixture
def stage_one(rigid_keypoints):
    S, T, _, _ = rigid_keypoints
    return TransportMap().fit(S, T)


@pytest.fixture
def contacts():
    """Four jaw contacts around a grasp, spaced as a can's cross-section is."""
    g = np.array([0.30, 0.10, 0.19])
    w = 0.029
    return np.array([g + [w, 0, 0], g - [w, 0, 0], g + [0, w, 0], g - [0, w, 0]])


def wanted(first, contacts, push=(0.0, 0.012, 0.0)):
    """Where the contacts must end up, given that stage 1 already moved them.

    The realistic case, and the one the class is for: stage 1 has carried the
    contacts to roughly the right place and stage 2 adds a small correction. A
    target expressed in the *source* frame instead would be asking stage 2 to
    undo stage 1, which is not a local correction at all.
    """
    return first.transport_positions(contacts) + np.asarray(push, dtype=float)


class TestItReducesToOneStage:
    def test_composing_with_an_identity_second_stage_changes_nothing(self, stage_one):
        """The reduction test: it is also the check on the combined variance
        rule, which must collapse to the single-stage one when the second stage
        is deterministic and exact."""
        X = np.random.default_rng(0).uniform(-0.2, 0.2, (12, 3))
        identity = TransportMap(residual=None, affine=IdentityAffine()).fit(X, X)
        composed = ComposedMap(stage_one, identity)
        assert np.allclose(
            composed.transport_positions(X), stage_one.transport_positions(X), atol=1e-9
        )
        assert np.allclose(composed.jacobian(X), stage_one.jacobian(X), atol=1e-9)
        _, s_one = stage_one.transport_positions(X, return_std=True)
        _, s_two = composed.transport_positions(X, return_std=True)
        assert np.allclose(s_one, s_two, atol=1e-9)


class TestDeterminantsMultiply:
    def test_the_composite_determinant_is_the_product(self, stage_one, contacts):
        """The whole argument for composing rather than blending."""
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=0.06
        )
        X = np.random.default_rng(1).uniform(-0.2, 0.2, (10, 3))
        d1 = np.linalg.det(composed.first.jacobian(X))
        d2 = np.linalg.det(
            composed.second.jacobian(composed.first.transport_positions(X))
        )
        assert np.allclose(np.linalg.det(composed.jacobian(X)), d1 * d2, rtol=1e-9)

    def test_each_stage_can_be_checked_on_its_own(self, stage_one, contacts):
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=0.06
        )
        X = np.random.default_rng(2).uniform(-0.2, 0.2, (10, 3))
        a, b = composed.stage_reports(X)
        assert a.consistent_sign and b.consistent_sign
        assert composed.check_diffeomorphism(X).consistent_sign

    def test_the_jacobian_matches_finite_differences(self, stage_one, contacts):
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=0.06
        )
        x = np.array([[0.05, -0.05, 0.02]])
        h, numeric = 1e-6, np.zeros((3, 3))
        for j in range(3):
            step = np.zeros((1, 3))
            step[0, j] = h
            numeric[:, j] = (
                composed.transport_positions(x + step)[0]
                - composed.transport_positions(x - step)[0]
            ) / (2 * h)
        assert np.allclose(composed.jacobian(x)[0], numeric, atol=1e-6)


class TestTheCorrectionIsLocal:
    """The measured property that makes a second stage worth having."""

    PUSH = np.array([0.0, 0.012, 0.0])

    def _displacement(self, composed, point):
        moved = composed.transport_positions(np.atleast_2d(point))
        baseline = composed.first.transport_positions(np.atleast_2d(point))
        return float(np.linalg.norm(moved[0] - baseline[0]))

    def test_it_hits_the_points_it_was_given(self, stage_one, contacts):
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=0.06
        )
        assert np.allclose(
            composed.transport_positions(contacts), wanted(stage_one, contacts), atol=1e-6
        )

    @pytest.mark.parametrize(
        "locality, leak_at_20cm", [(0.03, 1e-3), (0.06, 5e-3)]
    )
    def test_it_dies_out_within_a_few_length_scales(
        self, stage_one, contacts, locality, leak_at_20cm
    ):
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=locality
        )
        centre = contacts.mean(axis=0)
        assert self._displacement(composed, centre + [0.2, 0, 0]) < leak_at_20cm
        assert self._displacement(composed, centre + [0.8, 0, 0]) < 1e-5

    def test_a_shorter_locality_leaks_less(self, stage_one, contacts):
        centre = contacts.mean(axis=0) + [0.1, 0, 0]
        short = self._displacement(
            fit_local_correction(stage_one, contacts, wanted(stage_one, contacts), locality=0.03),
            centre,
        )
        long_ = self._displacement(
            fit_local_correction(stage_one, contacts, wanted(stage_one, contacts), locality=0.12),
            centre,
        )
        assert short < long_

    def test_a_global_affine_second_stage_would_leak_everywhere(self, stage_one, contacts):
        """Kept as a live assertion, not a comment: this is the failure
        ``fit_local_correction`` exists to make unreachable, and the codebase's
        habit is to pin a measured boundary rather than describe it."""
        moved = stage_one.transport_positions(contacts)
        leaky = TransportMap(
            residual=GaussianProcessRegressor(length_scale=0.03, optimize=False),
            affine=AffineMap(),
        ).fit(moved, moved + self.PUSH)
        far = moved.mean(axis=0) + [0.8, 0, 0]
        assert float(
            np.linalg.norm(leaky.transport_positions(far[None])[0] - far)
        ) > 0.010


class TestItRefusesToFailQuietly:
    def test_the_keypoint_residual_covers_both_stages(self, stage_one, contacts):
        """Stage 1's keypoints no longer land exactly on their targets once a
        second stage runs. That disturbance is a real cost of composing and must
        be visible, not absorbed."""
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts), locality=0.06
        )
        assert len(composed.keypoint_residual()) == len(stage_one.source) + len(contacts)

    def test_mismatched_shapes_are_refused(self, stage_one, contacts):
        with pytest.raises(ValueError, match="must match"):
            fit_local_correction(stage_one, contacts, contacts[:2])

    def test_a_correction_larger_than_its_reach_is_refused(self, stage_one, contacts):
        """The trap that caught me while writing these tests.

        Targets expressed in the *source* frame ask stage 2 to undo stage 1 --
        here a 411 mm correction through a 30 mm-locality stage. It still
        interpolated its four points to machine precision while excursing 336 mm
        five centimetres away, so the property-(i) check gives false confidence
        and only this guard catches it."""
        with pytest.raises(ValueError, match="not a local correction"):
            fit_local_correction(stage_one, contacts, contacts, locality=0.03)

    def test_a_tiny_correction_is_not_smoothed_away(self, stage_one, contacts):
        """With the signal variance left to its default, a sub-millimetre
        correction initialises below the noise floor and the stage silently does
        nothing at all."""
        tiny = np.array([0.0, 0.0005, 0.0])
        composed = fit_local_correction(
            stage_one, contacts, wanted(stage_one, contacts, tiny), locality=0.06
        )
        assert np.allclose(
            composed.transport_positions(contacts),
            wanted(stage_one, contacts, tiny), atol=1e-6,
        )
