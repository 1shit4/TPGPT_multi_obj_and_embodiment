"""An affine stage that deliberately does nothing.

The property under test: ``IdentityAffine`` must be *exactly* the identity, not
approximately -- because its whole purpose is to force a second transportation
stage's warp into the nonlinear residual, where a length scale can localise it.
Any leakage into the affine part acts globally and silently, which is the failure
this class exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.transport.affine import AffineMap, IdentityAffine
from tpgpt.transport.gp import GaussianProcessRegressor
from tpgpt.transport.maps import TransportMap


@pytest.fixture
def contacts():
    """Four jaw contacts around a grasp, spaced as a can's cross-section is."""
    g = np.array([0.3, 0.1, 0.19])
    w = 0.029
    return np.array([g + [w, 0, 0], g - [w, 0, 0], g + [0, w, 0], g - [0, w, 0]])


class TestItIsExactlyTheIdentity:
    def test_predict_returns_its_input_bit_for_bit(self, contacts):
        affine = IdentityAffine().fit(contacts, contacts + [0, 0.012, 0])
        assert np.array_equal(affine.predict(contacts), contacts)

    def test_the_jacobian_is_the_identity_and_takes_no_arguments(self, contacts):
        """``TransportMap.jacobian`` calls ``self.affine.jacobian()`` with no
        arguments, so this is the shape of the contract that matters."""
        affine = IdentityAffine().fit(contacts, contacts)
        assert np.array_equal(affine.jacobian(), np.eye(3))

    def test_it_preserves_the_one_dimensional_shape_convention(self, contacts):
        affine = IdentityAffine().fit(contacts, contacts)
        assert affine.predict(contacts[0]).shape == (3,)
        assert affine.predict(contacts).shape == (4, 3)

    def test_the_inverse_is_also_exact(self, contacts):
        affine = IdentityAffine().fit(contacts, contacts)
        assert np.array_equal(affine.inverse(contacts), contacts)

    def test_it_is_an_affine_map_so_existing_call_sites_keep_working(self, contacts):
        assert isinstance(IdentityAffine().fit(contacts, contacts), AffineMap)

    def test_unfitted_use_still_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            IdentityAffine().predict(np.zeros((1, 3)))

    def test_mismatched_shapes_are_refused(self, contacts):
        with pytest.raises(ValueError, match="must match"):
            IdentityAffine().fit(contacts, contacts[:2])


class TestWhyItExists:
    """The measured reason: a plain affine stage makes a *local* correction global.

    A second stage fitted on four contacts that all move the same way has its
    displacement absorbed into the affine ``T_bar``, which applies everywhere.
    The residual then has nothing left to fit and its length scale -- the knob
    that is supposed to control locality -- stops doing anything at all.
    """

    PUSH = np.array([0.0, 0.012, 0.0])  # a 12 mm correction at the contacts

    def _stage(self, contacts, affine, length_scale):
        return TransportMap(
            residual=GaussianProcessRegressor(
                length_scale=length_scale, optimize=False
            ),
            affine=affine,
        ).fit(contacts, contacts + self.PUSH)

    def _displacement(self, stage, point):
        moved = stage.transport_positions(np.atleast_2d(point))
        return float(np.linalg.norm(moved[0] - point))

    @pytest.mark.parametrize("length_scale", [0.03, 0.12])
    def test_a_plain_affine_stage_leaks_the_correction_across_the_whole_workspace(
        self, contacts, length_scale
    ):
        far = contacts.mean(axis=0) + [0.8, 0.0, 0.0]
        stage = self._stage(contacts, AffineMap(), length_scale)
        # 12 mm applied 80 cm away, identically to at the contacts
        assert self._displacement(stage, far) == pytest.approx(0.012, abs=5e-4)

    @pytest.mark.parametrize(
        "length_scale, leak_at_20cm", [(0.03, 1e-4), (0.06, 5e-4)]
    )
    def test_the_identity_affine_confines_it_and_the_length_scale_sets_the_radius(
        self, contacts, length_scale, leak_at_20cm
    ):
        centre = contacts.mean(axis=0)
        stage = self._stage(contacts, IdentityAffine(), length_scale)
        assert self._displacement(stage, centre) > 0.008  # still corrects nearby
        assert self._displacement(stage, centre + [0.2, 0, 0]) < leak_at_20cm
        assert self._displacement(stage, centre + [0.8, 0, 0]) < 1e-6

    def test_a_shorter_length_scale_leaks_less_than_a_longer_one(self, contacts):
        """The ordering is the claim; the absolute numbers depend on spacing."""
        centre = contacts.mean(axis=0)
        probe = centre + [0.1, 0, 0]
        short = self._displacement(self._stage(contacts, IdentityAffine(), 0.03), probe)
        long_ = self._displacement(self._stage(contacts, IdentityAffine(), 0.12), probe)
        assert short < long_
