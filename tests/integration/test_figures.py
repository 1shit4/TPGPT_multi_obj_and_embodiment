"""The theory figures render and encode the properties they claim."""

import numpy as np
import pytest

from tpgpt.experiments.figures import cyclic_demonstration, flat_to_curved_surface
from tpgpt.viz.figures import (
    figure_space_deformation,
    figure_transported_field,
    figure_uncertainty_fields,
)


@pytest.fixture
def scene():
    source, target = flat_to_curved_surface()
    return source, target, cyclic_demonstration()


def test_all_three_figures_render(scene, tmp_path):
    source, target, demo = scene
    paths = [
        figure_space_deformation(source, target, tmp_path / "fig2.png"),
        figure_transported_field(demo, source, target, None, tmp_path / "fig4.png"),
        figure_uncertainty_fields(demo, source, target, tmp_path / "fig5.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 10_000


def test_the_example_actually_needs_the_nonlinear_stage(scene):
    """Fig. 2 is only informative if a rigid motion cannot solve it."""
    from tpgpt.transport.maps import TransportMap

    source, target, _ = scene
    affine_only = TransportMap(residual=None).fit(source, target)
    full = TransportMap().fit(source, target)
    assert np.linalg.norm(target - affine_only.transport_positions(source), axis=1).max() > 1.0
    assert full.keypoint_residual().max() < 1e-2


def test_transport_uncertainty_is_lowest_at_the_keypoints(scene):
    """The property Fig. 5 (left) illustrates."""
    from tpgpt.transport.maps import TransportMap

    source, target, _ = scene
    tm = TransportMap().fit(
        np.c_[source, np.zeros(len(source))], np.c_[target, np.zeros(len(target))]
    )
    at_keypoints = tm.transport_positions(
        np.c_[source, np.zeros(len(source))], return_std=True
    )[1]
    far = tm.transport_positions(np.array([[0.0, 300.0, 0.0]]), return_std=True)[1]
    assert far[0] > 10 * at_keypoints.max()
