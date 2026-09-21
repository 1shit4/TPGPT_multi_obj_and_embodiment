"""GraspGen-X in the hardware planner, against a live server.

Skips when no server answers (start one with
``ycb_pointcloud_pipeline/scripts/run_graspgenx_server.sh --daemon``); never
starts one itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.grasp.server import server_available
from tpgpt.hardware import hand
from tpgpt.hardware.grasp_selection import request_candidates, select_and_plan
from tpgpt.hardware.planner import PlanningError, TargetScene, plan

from tests.unit.test_grasp_selection import TABLE, cup_cloud
from tests.unit.test_hardware import synthetic_bundle

pytestmark = pytest.mark.skipif(not server_available(), reason="GraspGen-X server not running")


def test_a_cup_wider_than_the_hand_is_gripped_by_its_wall():
    bundle = synthetic_bundle()
    cloud = cup_cloud()
    scene = TargetScene(points=cloud, destination=np.array([0.55, 0.25, 0.15]),
                        destination_height=0.15, support_height=TABLE)
    with pytest.raises(PlanningError, match="object_too_wide"):
        plan(bundle, scene)                                   # the scripted top-down grasp

    poses, scores = request_candidates(cloud)
    assert len(poses) > 0
    selection = select_and_plan(bundle, scene, poses, scores)
    m = selection.plan.metrics
    assert selection.source == "graspgenx"
    assert m["jaw_verdict"] == "fits"
    assert m["object_extent_closing_m"] > hand.MAX_OPENING
    assert m["object_width_closing_m"] < 0.03                  # a wall, not the cup
    assert m["jaw_engagement_m"] >= hand.MIN_ENGAGEMENT
    assert m["aim_grasp_m"] < 0.01
