import numpy as np
import pytest

from tpgpt.perception.cameras import target_edge_mask


def test_edge_exclusion_keeps_depth_discontinuity_and_distant_obstacle():
    ids = np.zeros((7, 7), dtype=int)
    ids[3, 3] = 1
    depth = np.ones((7, 7)) * 2.
    depth[3, 3] = 1.
    depth[3, 2] = 1.004  # target-like residual at an adjacent background pixel
    depth[3, 4] = 1.2  # real background beyond the silhouette
    depth[3, 1] = 1.004  # outside the explicitly bounded edge margin
    mask = target_edge_mask(ids, depth, {1})
    assert mask[3, 2]
    assert not mask[3, 4]
    assert not mask[3, 1]
    assert not target_edge_mask(ids, depth, {1}, 0).any()
    assert not target_edge_mask(ids, depth, {9}).any()


@pytest.mark.parametrize("margin,tolerance", [(-1, .01), (1.5, .01), (1, -.01)])
def test_invalid_edge_parameters(margin, tolerance):
    with pytest.raises(ValueError):
        target_edge_mask(np.zeros((2, 2)), np.ones((2, 2)), {1}, margin, tolerance)
