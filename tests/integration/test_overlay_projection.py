"""World-to-pixel projection for the report overlays.

The property is a round trip. Every point of an object's cloud came from that
camera's own depth buffer, so projecting those points back must land them
inside that object's segmentation mask. Nothing about object shape enters, so
the test cannot be fooled the way comparing a body origin to a mask centroid
can -- the visible-surface centroid of a tall carton is nowhere near its body
centre, and that confound hid this bug once already.

The failure it guards against is silent: a vertical flip leaves the columns
correct, so keypoints land on the right part of the scene left-to-right and
merely at the wrong height. That reads as a keypoint that has drifted, not as a
projection that is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("robosuite")

OBJECTS = ("cereal", "can", "milk")


@pytest.fixture(scope="module")
def env():
    from tpgpt.experiments.pipeline import build_scene

    environment = build_scene(OBJECTS, gripper="panda", shelf_variant="cubby", seed=0)
    yield environment
    environment.close()


@pytest.mark.parametrize("name", OBJECTS)
def test_a_cloud_projects_back_onto_its_own_object(env, name):
    from tpgpt.perception.cameras import object_point_cloud
    from tpgpt.reporting.overlay import project_to_pixels

    obs = env._get_observations()
    cloud = object_point_cloud(env, name, obs=obs)
    assert len(cloud) > 40, f"too little cloud on the {name} to test with"

    segmentation = np.asarray(obs["workspace_segmentation_instance"]).squeeze()
    height, width = segmentation.shape[:2]
    # The mask, flipped into the same orientation the overlay draws in.
    mask = segmentation[::-1] == env.object_names.index(name) + 1

    pixels, visible = project_to_pixels(env, "workspace", cloud.points, height, width)
    rows = np.rint(pixels[visible, 1]).astype(int)
    columns = np.rint(pixels[visible, 0]).astype(int)
    inside = np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1)
    landed = mask[inside].mean()
    assert landed > 0.8, (
        f"only {landed:.0%} of the {name} cloud projected back onto the "
        f"{name}; the overlay's row convention disagrees with perception's"
    )


def test_the_two_modules_agree_on_which_way_up_a_row_is(env):
    """Stated directly, because it is the thing that was wrong.

    ``perception.cameras`` flips the buffers before unprojecting, so the camera
    matrix's rows are the flipped, human-readable ones. The overlay must not
    flip again.
    """
    from tpgpt.perception.cameras import object_point_cloud
    from tpgpt.reporting.overlay import project_to_pixels

    obs = env._get_observations()
    cloud = object_point_cloud(env, "can", obs=obs)
    pixels, visible = project_to_pixels(env, "workspace", cloud.points, 256, 256)
    segmentation = np.asarray(obs["workspace_segmentation_instance"]).squeeze()
    mask = segmentation[::-1] == env.object_names.index("can") + 1

    rows = np.rint(pixels[visible, 1]).astype(int)
    columns = np.rint(pixels[visible, 0]).astype(int)
    upright = mask[np.clip(rows, 0, 255), np.clip(columns, 0, 255)].mean()
    upside_down = mask[np.clip(255 - rows, 0, 255), np.clip(columns, 0, 255)].mean()
    assert upright > upside_down, "the overlay is drawing rows upside down"
