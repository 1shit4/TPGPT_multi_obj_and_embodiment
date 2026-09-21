"""Grasp choice on a real cloud: the Franka Hand model, the jaw check, selection.

The case that motivated this: a cup is wider than the hand opens, so measuring
the whole cloud along the closing axis refused every grasp on it -- including
the one that closes on a 3 mm rim. These tests pin that the jaw check measures
what lies *between the pads*, and that a generated candidate is only chosen once
the transported plan it produces passes its own checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.hardware import hand
from tpgpt.hardware.grasp_selection import (
    SelectionSettings,
    pose_to_grasp_frame,
    screen_candidates,
    select_and_plan,
)
from tpgpt.hardware.planner import PlanningError, TargetScene, plan

from tests.unit.test_hardware import synthetic_bundle

TABLE = -0.10


def cup_cloud(centre_xy=(0.42, 0.02), radius=0.045, wall=0.003, height=0.10,
              spacing=0.003) -> np.ndarray:
    """An open cup standing on the table: outer wall, inner wall and floor.

    90 mm across, wider than the hand's 80 mm opening.
    """
    cx, cy = centre_xy
    zs = np.arange(TABLE + spacing, TABLE + height, spacing)
    pts = []
    for r in (radius, radius - wall):
        n = int(2 * np.pi * r / spacing)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for z in zs:
            pts.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, z)]))
    g = np.arange(-radius, radius, spacing)
    xx, yy = np.meshgrid(g, g)
    floor = np.hypot(xx, yy) < radius - wall
    pts.append(np.column_stack([cx + xx[floor], cy + yy[floor], np.full(floor.sum(), TABLE + 0.005)]))
    return np.vstack(pts)


def top_down(closing, tcp) -> np.ndarray:
    """Grasp-convention rotation for a vertical descent closing along ``closing``."""
    c = np.asarray(closing, dtype=float)
    c = c / np.linalg.norm(c)
    a = np.array([0.0, 0.0, -1.0])
    return np.column_stack([c, np.cross(a, c), a])


def graspgen_pose(tcp, rotation) -> np.ndarray:
    """A GraspGen-X pose (at the gripper base) whose fingertips are at ``tcp``."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(tcp) - rotation[:, 2] * hand.TCP_OFFSET
    return T


CUP_CENTRE = np.array([0.42, 0.02])
RIM_TCP = np.array([CUP_CENTRE[0] + 0.0435, CUP_CENTRE[1], TABLE + 0.10 - 0.012])


class TestHandModel:
    def test_matches_graspgen_franka_panda_bbox(self):
        """GraspGen-X publishes the same hand's bbox in its base frame
        (``x_grippers/franka_panda/config.json``): x [-0.1006, 0.1039],
        y [-0.0315, 0.0315], z [-0.0259, 0.1123]. Up to the half turn about the
        approach a symmetric hand does not see, the franka_description boxes
        must agree."""
        corners = np.vstack([b.corners() for b in hand.hand_boxes()])
        base = corners + [0.0, 0.0, hand.TCP_OFFSET]
        assert np.ptp(base[:, 0]) == pytest.approx(0.1039 + 0.1006, abs=4e-3)
        assert base[:, 1].max() == pytest.approx(0.0315, abs=1e-3)
        assert base[:, 2].min() == pytest.approx(-0.0259, abs=1e-3)
        assert base[:, 2].max() == pytest.approx(0.1123, abs=2e-3)

    def test_pads_are_eighty_mm_apart_at_the_tcp(self):
        lo, hi = hand.pad_box()
        assert hi[0] - lo[0] == pytest.approx(0.08)
        assert lo[2] < 0.0 < hi[2]

    def test_pose_conversion_puts_the_tcp_at_the_fingertips(self):
        R = top_down([1, 0, 0], None)
        g = pose_to_grasp_frame(graspgen_pose(RIM_TCP, R))
        assert np.allclose(g.tcp, RIM_TCP)
        assert np.allclose(g.approach, [0, 0, -1])


class TestJaws:
    def test_the_cup_is_wider_than_the_hand(self):
        assert np.ptp(cup_cloud()[:, 0]) > hand.MAX_OPENING

    def test_a_rim_grasp_fits(self):
        jaw = hand.check_jaws(cup_cloud(), RIM_TCP, top_down([1, 0, 0], RIM_TCP))
        assert jaw.verdict == "fits", jaw
        assert jaw.width < 0.01

    def test_a_pinch_on_the_top_of_the_rim_is_shallow(self):
        # TCP 5 mm above the rim: only ~4 mm of the 18.5 mm pads have wall behind them.
        tcp = RIM_TCP + [0.0, 0.0, 0.017]
        jaw = hand.check_jaws(cup_cloud(), tcp, top_down([1, 0, 0], tcp))
        assert jaw.verdict == "shallow", jaw
        assert jaw.engagement < hand.MIN_ENGAGEMENT
        full = hand.check_jaws(cup_cloud(), RIM_TCP, top_down([1, 0, 0], RIM_TCP))
        assert full.engagement > 0.015

    def test_closing_across_the_whole_cup_does_not(self):
        tcp = np.array([*CUP_CENTRE, TABLE + 0.07])
        jaw = hand.check_jaws(cup_cloud(), tcp, top_down([1, 0, 0], tcp))
        assert jaw.verdict == "too_wide"

    def test_a_grasp_in_air_is_empty(self):
        tcp = np.array([*CUP_CENTRE, TABLE + 0.30])
        assert hand.check_jaws(cup_cloud(), tcp, top_down([1, 0, 0], tcp)).verdict == "empty"

    def test_the_palm_inside_the_object_collides(self):
        # A thin post, gripped 9 cm below its top: fingers fit, palm does not.
        z = np.arange(TABLE, TABLE + 0.2, 0.002)
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        post = np.vstack([np.column_stack([0.4 + 0.006 * np.cos(t) + 0 * z,
                                           0.006 * np.sin(t) + 0 * z, z]) for t in th])
        tcp = np.array([0.4, 0.0, TABLE + 0.11])
        jaw = hand.check_jaws(post, tcp, top_down([1, 0, 0], tcp))
        assert jaw.verdict == "collides"
        assert "body" in jaw.blocked_by


class TestRefinement:
    def test_a_shallow_rim_grasp_is_slid_in_until_the_pads_are_full(self):
        from tpgpt.hardware.grasp_selection import refine_grasp

        shallow = RIM_TCP + [0.0, 0.0, 0.017]          # only ~4 mm of pad on the wall
        g0 = pose_to_grasp_frame(graspgen_pose(shallow, top_down([1, 0, 0], None)))
        assert hand.check_jaws(cup_cloud(), g0.tcp, g0.rotation).verdict == "shallow"
        g, depth, centre = refine_grasp(cup_cloud(), g0, TABLE)
        jaw = hand.check_jaws(cup_cloud(), g.tcp, g.rotation)
        assert jaw.verdict == "fits"
        assert jaw.engagement > 0.015
        assert 0.015 < depth <= 0.025
        assert np.allclose(g.approach, g0.approach) and np.allclose(g.closing, g0.closing)

    def test_it_centres_the_wall_between_the_pads(self):
        from tpgpt.hardware.grasp_selection import refine_grasp

        off = RIM_TCP + [0.008, 0.0, 0.0]               # wall 8 mm off centre
        g0 = pose_to_grasp_frame(graspgen_pose(off, top_down([1, 0, 0], None)))
        g, _, centre = refine_grasp(cup_cloud(), g0, TABLE)
        lo, hi = hand.pad_box()
        loc = hand.to_grasp_frame(cup_cloud(), g.tcp, g.rotation)
        between = np.all((loc >= lo) & (loc <= hi), axis=1)
        assert abs((loc[between, 0].max() + loc[between, 0].min()) / 2) < 0.002
        assert centre == pytest.approx(-0.008, abs=0.002)

    def test_it_never_pushes_the_fingers_into_the_table(self):
        from tpgpt.hardware.grasp_selection import SelectionSettings, refine_grasp

        # A 20 mm tall block: going to full pad engagement would put the pads
        # under the table, so refinement must stop at the clearance limit.
        g1 = np.arange(-0.02, 0.0201, 0.003)
        xx, yy = np.meshgrid(g1, g1)
        top = np.column_stack([0.4 + xx.ravel(), yy.ravel(), np.full(xx.size, TABLE + 0.02)])
        sides = np.vstack([np.column_stack([np.full(8, 0.4 + sx), np.full(8, sy),
                                            np.linspace(TABLE, TABLE + 0.02, 8)])
                           for sx in (-0.02, 0.02) for sy in g1])
        block = np.vstack([top, sides])
        tcp = np.array([0.4, 0.0, TABLE + 0.018])
        g0 = pose_to_grasp_frame(graspgen_pose(tcp, top_down([1, 0, 0], None)))
        s = SelectionSettings()
        g, depth, _ = refine_grasp(block, g0, TABLE, s)
        assert hand.lowest_point(g.tcp, g.rotation) - TABLE >= s.min_support_clearance - 1e-9


class TestScreening:
    def test_each_rejection_is_named(self):
        cloud = cup_cloud()
        R_down = top_down([1, 0, 0], None)
        R_up = R_down @ np.diag([1.0, -1.0, -1.0])            # approaching from below
        low = RIM_TCP - [0, 0, 0.095]                          # fingertips under the table
        poses = np.stack([
            graspgen_pose(RIM_TCP, R_down),                    # good
            graspgen_pose(RIM_TCP + [0.001, 0, 0], R_down),    # duplicate of the good one
            graspgen_pose(RIM_TCP, R_up),
            graspgen_pose([*CUP_CENTRE, TABLE + 0.07], R_down),
            graspgen_pose(low, R_down),
        ])
        scores = np.array([0.9, 0.8, 0.95, 0.7, 0.6])
        out = {c.index: c.rejected for c in screen_candidates(cloud, poses, scores, TABLE)}
        assert out[0] is None
        assert out[1] == "duplicate"
        assert out[2].startswith("from_below")
        # Closing across the middle of the empty cup: the simulation's on-target
        # stage (held point > 4 cm from the cloud) runs before the jaw width, as
        # in filter_grasps, so it names the rejection; the jaws would refuse it too.
        assert out[3].startswith(("on_target", "too_wide"))
        assert out[4].startswith(("hits_support", "too_wide", "collides"))


class TestSelection:
    @pytest.fixture(scope="class")
    def bundle(self):
        return synthetic_bundle()

    def _scene(self):
        return TargetScene(points=cup_cloud(), destination=np.array([0.55, 0.25, 0.15]),
                           destination_height=0.15, support_height=TABLE)

    def test_the_old_whole_cloud_width_would_refuse_the_cup_and_the_rim_plans(self, bundle):
        scene = self._scene()
        scene.grasp = pose_to_grasp_frame(graspgen_pose(RIM_TCP, top_down([1, 0, 0], None)))
        result = plan(bundle, scene)
        assert result.metrics["jaw_verdict"] == "fits"
        assert result.metrics["object_width_closing_m"] < 0.01
        assert result.metrics["object_extent_closing_m"] > hand.MAX_OPENING

    def test_the_best_scoring_executable_candidate_is_chosen(self, bundle):
        R = top_down([1, 0, 0], None)
        poses = np.stack([
            graspgen_pose([*CUP_CENTRE, TABLE + 0.07], R),     # best score, too wide
            graspgen_pose(RIM_TCP, R),                         # executable
        ])
        sel = select_and_plan(bundle, self._scene(), poses, np.array([0.95, 0.7]))
        assert sel.source == "graspgenx"
        assert sel.chosen.index == 1
        assert sel.plan.metrics["min_det"] > 0.1
        assert sel.plan.metrics["aim_grasp_m"] < 0.01

    def test_no_server_falls_back_to_top_down_and_says_so(self, bundle):
        from tests.unit.test_keypoint_extraction import box_cloud

        points = box_cloud(np.array([0.40, 0.05, TABLE + 0.05]), angle=-0.6,
                           half=np.array([0.025, 0.04, 0.05]), seed=3, faces="visible")
        scene = TargetScene(points=points, destination=np.array([0.55, 0.25, 0.15]),
                            destination_height=0.15, support_height=TABLE)
        sel = select_and_plan(bundle, scene, None, None, SelectionSettings(fallback_top_down=True),
                              unavailable_reason="server down")
        assert sel.source == "top_down"
        assert any("server down" in n for n in sel.notes)

    def test_without_fallback_nothing_executable_raises(self, bundle):
        R = top_down([1, 0, 0], None)
        poses = np.stack([graspgen_pose([*CUP_CENTRE, TABLE + 0.07], R)])
        with pytest.raises(PlanningError, match="no_executable_grasp"):
            select_and_plan(bundle, self._scene(), poses, np.array([0.9]),
                            SelectionSettings(fallback_top_down=False))


# --------------------------------------------------------------- the whole path
def _slot_over_destination(roof: float):
    """A shelf slot around the test destination (0.55, 0.25), board at 0.15.

    The front edge runs along +x at y = 0.12, so the slot opens toward the pick
    (the cup at y = 0.02) and extends 0.3 m in +y; walls at x = 0.40 and 0.70.
    """
    from tpgpt.hardware.shelf import ShelfModel

    return ShelfModel.from_config({"shelf": {
        "measured": True, "front_left": [0.40, 0.12], "front_right": [0.70, 0.12],
        "depth": 0.30, "bottom": TABLE, "top": roof, "wall_thickness": 0.018,
        "back_thickness": 0.0,
        "boards": [{"name": "board", "top": 0.15, "thickness": 0.018},
                   {"name": "roof", "top": roof, "thickness": 0.018, "support": False}],
    }})


class TestWholePath:
    @pytest.fixture(scope="class")
    def bundle(self):
        return synthetic_bundle()

    def _scene(self, destination=(0.55, 0.25, 0.15)):
        return TargetScene(points=cup_cloud(), destination=np.array(destination),
                           destination_height=float(destination[2]), support_height=TABLE)

    def _poses(self):
        return np.stack([graspgen_pose(RIM_TCP, top_down([1, 0, 0], None))]), np.array([0.9])

    def test_the_chosen_plan_carries_its_path_check(self, bundle):
        poses, scores = self._poses()
        sel = select_and_plan(bundle, self._scene(), poses, scores)
        c = sel.chosen.path_check
        assert c["violations"] == 0 and c["reachable_fraction"] >= 0.6
        assert sel.rank_examined == 1 and not sel.fell_back
        assert any("kinematics only" in n for n in sel.notes)
        assert sel.plan.metrics["path_check"] is c

    def test_a_shelf_the_hand_cannot_enter_rejects_every_candidate_by_name(self, bundle):
        from tpgpt.hardware.grasp_selection import PathContext
        from tpgpt.hardware.shelf import ShelfWorld

        world = ShelfWorld.build(_slot_over_destination(roof=0.30))   # the hand ends in the roof
        poses, scores = self._poses()
        with pytest.raises(PlanningError, match="no_executable_grasp") as err:
            select_and_plan(bundle, self._scene(), poses, scores,
                            SelectionSettings(check_place_zone=False),
                            path=PathContext(world=world))
        rejected = [c.rejected for c in err.value.candidates if c.rejected]
        assert any(r.startswith("collision") and "shelf" in r for r in rejected)

    def test_a_shelf_the_path_stays_clear_of_admits_the_candidate(self, bundle):
        from tpgpt.hardware.grasp_selection import PathContext
        from tpgpt.hardware.shelf import ShelfModel, ShelfWorld

        # The synthetic demonstration carries the cup diagonally up from the
        # table, so any shelf built round its destination is struck at the
        # board's front edge (the check above). This one stands beside the path.
        beside = ShelfModel.from_config({"shelf": {
            "measured": True, "front_left": [0.40, 0.55], "front_right": [0.70, 0.55],
            "depth": 0.30, "bottom": TABLE, "top": 0.40, "wall_thickness": 0.018,
            "boards": [{"name": "board", "top": 0.15, "thickness": 0.018}],
        }})
        world = ShelfWorld.build(beside)
        poses, scores = self._poses()
        sel = select_and_plan(bundle, self._scene(), poses, scores, path=PathContext(world=world))
        assert sel.chosen.path_check["violations"] == 0
        assert sel.world["model"]["measured"] is True

    def test_out_of_reach_falls_back_only_when_allowed(self, bundle):
        far = (1.10, 0.25, 0.15)
        poses, scores = self._poses()
        sel = select_and_plan(bundle, self._scene(far), poses, scores,
                              SelectionSettings(max_aim_grasp=1.0, min_path_clearance=-1.0))
        assert sel.fell_back and sel.chosen.path_check["verdict"].startswith("kinematics")
        with pytest.raises(PlanningError):
            select_and_plan(bundle, self._scene(far), poses, scores,
                            SelectionSettings(max_aim_grasp=1.0, min_path_clearance=-1.0,
                                              allow_kinematic_fallback=False))


class TestFunnelStages:
    def _screen(self, context, **settings):
        from tpgpt.hardware.grasp_selection import screen_candidates

        poses = np.stack([graspgen_pose(RIM_TCP, top_down([1, 0, 0], None))])
        out = screen_candidates(cup_cloud(), poses, np.array([0.9]), TABLE,
                                SelectionSettings(**settings), context=context)
        return out, {row[0]: row for row in out.stages}

    def test_a_camera_above_sees_a_top_down_approach(self):
        from tpgpt.hardware.grasp_selection import ScreenContext

        out, stages = self._screen(ScreenContext(camera_position=np.array([0.4, 0.0, 1.0])))
        assert stages["visibility"][1:] == [1, 1, False]
        assert out[0].rejected is None

    def test_a_stage_that_would_empty_the_set_passes_it_through_and_says_so(self):
        from tpgpt.hardware.grasp_selection import ScreenContext

        # A camera under the table: a top-down approach points straight away from it.
        out, stages = self._screen(ScreenContext(camera_position=np.array([0.42, 0.02, -2.0])))
        assert stages["visibility"] == ["visibility", 1, 1, True]
        assert out[0].rejected is None

    def test_the_place_zone_is_judged_against_the_shelf_model(self):
        from tpgpt.hardware.grasp_selection import ScreenContext
        from tpgpt.hardware.shelf import ShelfWorld

        # Released with its fingertips at 0.238 m, the hand's body spans the roof.
        ctx = ScreenContext(world=ShelfWorld.build(_slot_over_destination(roof=0.30)),
                            destination=np.array([0.55, 0.25]), destination_height=0.15)
        out, stages = self._screen(ctx)
        # The only candidate would arrive through the roof: flagged, passed through.
        assert stages["place_zone"] == ["place_zone", 1, 1, True]
        ctx.world = ShelfWorld.build(_slot_over_destination(roof=0.60))
        out, stages = self._screen(ctx)
        assert stages["place_zone"] == ["place_zone", 1, 1, False]

    def test_release_positions_keep_the_grasp_offset_from_the_object_base(self):
        from tpgpt.hardware.grasp_selection import (Candidate, ScreenContext, _release_positions,
                                                    horizontal_centre)

        cloud = cup_cloud()
        g = pose_to_grasp_frame(graspgen_pose(RIM_TCP, top_down([1, 0, 0], None)))
        c = Candidate(index=0, score=0.9, grasp=g, tilt_deg=0.0)
        ctx = ScreenContext(destination=np.array([0.55, 0.25]), destination_height=0.15)
        release = _release_positions([c], cloud, TABLE, ctx)[0]
        base = horizontal_centre(cloud)
        assert np.allclose(release[:2] - [0.55, 0.25], g.tcp[:2] - base[:2], atol=1e-9)
        assert release[2] == pytest.approx(0.15 + g.tcp[2] - TABLE)
