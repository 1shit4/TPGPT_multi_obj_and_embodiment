"""Grasp filtering, on geometry with known answers.

Each filter is tested against a case constructed so the right answer is
obvious -- a hand buried in a wall, an object wider than the jaws, a grasp
approaching from a direction nothing ever saw. The funnel's *fallback*
behaviour gets as much attention as the filters themselves: a stage that
empties the candidate set and returns nothing turns "the hand would have
collided" into "there is no grasp", which is a worse answer, so it must pass
its input through and say that it did.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.grasp.filters import (
    FilterFunnel,
    by_collision,
    by_jaw_width,
    jaw_width_verdicts,
    by_target,
    by_visibility,
    filter_grasps,
    suppress_duplicates,
)
from tpgpt.grasp.grasps import Grasp6D
from tpgpt.grasp.grippers import gripper_config_path, gripper_geometry, resolve_pair

pytestmark = pytest.mark.skipif(
    not gripper_config_path("franka_panda").is_file(),
    reason="GraspGen-X gripper descriptions not installed",
)

PAIR = "panda"


def grasp_at(position, approach=(0, 0, -1), yaw=0.0, score=0.8, gripper="franka_panda"):
    """A grasp at a position, approaching along ``approach``."""
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(approach @ reference) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    closing = np.cross(reference, approach)
    closing /= np.linalg.norm(closing)
    c, s = np.cos(yaw), np.sin(yaw)
    closing = c * closing + s * np.cross(approach, closing)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([closing, np.cross(approach, closing), approach])
    pose[:3, 3] = np.asarray(position, dtype=float)
    return Grasp6D(
        pose=pose, score=score, gripper=gripper,
        width=gripper_geometry(gripper).aperture,
    )


def box_cloud(centre, half, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    points[np.arange(n), axis] = np.sign(points[np.arange(n), axis])
    return np.asarray(centre) + points * np.asarray(half)


class TestVisibility:
    def test_a_grasp_reaching_around_the_far_side_is_dropped(self):
        """The generator invents the surface the camera never saw and proposes
        grasps on it."""
        camera = {"agentview": np.array([1.0, 0.0, 1.0])}
        towards = grasp_at([0, 0, 0.9], approach=[-1, 0, 0])   # camera -> object
        away = grasp_at([0, 0, 0.9], approach=[1, 0, 0])       # object -> camera
        kept = by_visibility([towards, away], np.array([0, 1]), camera)
        assert kept.tolist() == [0]

    def test_side_approaches_survive(self):
        """Around 90 degrees is legitimate and is much of the point of 6-DoF."""
        camera = {"agentview": np.array([1.0, 0.0, 1.0])}
        side = grasp_at([0, 0, 0.9], approach=[0, 1, 0])
        assert by_visibility([side], np.array([0]), camera).tolist() == [0]

    def test_seen_by_any_camera_is_enough(self):
        cameras = {"a": np.array([1.0, 0, 1]), "b": np.array([-1.0, 0, 1])}
        grasp = grasp_at([0, 0, 0.9], approach=[1, 0, 0])
        assert len(by_visibility([grasp], np.array([0]), cameras)) == 1


class TestTarget:
    def test_a_grasp_that_would_hold_nothing_is_dropped(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        pair = resolve_pair(PAIR)
        on = grasp_at([0, 0, 0.85 + gripper_geometry("franka_panda").tcp_depth])
        off = grasp_at([0.5, 0, 0.85 + gripper_geometry("franka_panda").tcp_depth])
        kept = by_target([on, off], np.array([0, 1]), cloud, pair)
        assert kept.tolist() == [0]


class TestJawWidth:
    def test_an_object_wider_than_the_jaws_is_dropped(self):
        pair = resolve_pair(PAIR)
        aperture = gripper_geometry("franka_panda").aperture
        depth = gripper_geometry("franka_panda").tcp_depth
        grasp = grasp_at([0, 0, 0.85 + depth])
        # The width that matters is along this grasp's own closing axis, so the
        # box has to be widened along that axis and not just made big.
        closing = np.argmax(np.abs(grasp.closing))
        narrow_half = np.array([0.03, 0.03, 0.05])
        wide_half = narrow_half.copy()
        narrow_half[closing] = aperture * 0.25
        wide_half[closing] = aperture * 1.5
        narrow = box_cloud([0, 0, 0.85], narrow_half)
        wide = box_cloud([0, 0, 0.85], wide_half)
        assert len(by_jaw_width([grasp], np.array([0]), narrow, pair)) == 1
        assert len(by_jaw_width([grasp], np.array([0]), wide, pair)) == 0

    def test_width_is_measured_along_the_grasps_own_closing_axis(self):
        """A bottle is narrow at the neck and wide at the base; which matters
        depends on where the grasp sits, so a global bounding box will not do.
        """
        pair = resolve_pair(PAIR)
        aperture = gripper_geometry("franka_panda").aperture
        depth = gripper_geometry("franka_panda").tcp_depth
        # A slab: wide in x, narrow in y. A grasp closing across y fits, one
        # closing across x does not, on the very same cloud.
        slab = box_cloud([0, 0, 0.85], [aperture * 1.5, 0.02, 0.05])
        fits, does_not = None, None
        for yaw in np.linspace(0, np.pi, 9):
            grasp = grasp_at([0, 0, 0.85 + depth], yaw=yaw)
            kept = len(by_jaw_width([grasp], np.array([0]), slab, pair)) == 1
            if abs(grasp.closing[1]) > 0.95:
                fits = kept
            if abs(grasp.closing[0]) > 0.95:
                does_not = kept
        assert fits is True and does_not is False

    def test_too_little_cloud_is_not_treated_as_too_wide(self):
        """Absence of evidence is not evidence of a wide object.

        It is not evidence of a *narrow* one either, which is what the previous
        version of this test asserted by requiring the grasp to be kept. The
        distinction it was protecting is preserved and now explicit: the verdict
        is ``"unverified"``, never ``"too_wide"``, so the report can name a
        perception failure instead of a grasping one -- the "different report
        line" this docstring always asked for, which did not exist until it was
        built.
        """
        pair = resolve_pair(PAIR)
        grasp = grasp_at([0, 0, 0.9])
        verdicts = jaw_width_verdicts([grasp], np.array([0]), np.zeros((2, 3)), pair)
        assert verdicts[0] == "unverified"

    def test_an_unverifiable_grasp_is_not_certified_as_fitting(self):
        """The measured case: ``yumi/can``.

        An 11-point slab reported 5.9 mm across a 50.0 mm can, whose hand opens
        to exactly 50.0 mm. Waving it through on "absence of evidence" is what
        sent the arm to a grasp that cannot exist, and the jaws then travelled
        39.6 mm through the can while the grip force bled from 20.3 N to 0.6 N.
        """
        pair = resolve_pair(PAIR)
        grasp = grasp_at([0, 0, 0.9])
        sparse = np.zeros((2, 3))
        assert len(by_jaw_width([grasp], np.array([0]), sparse, pair)) == 0

    def test_a_lower_bound_that_already_exceeds_the_aperture_needs_no_quorum(self):
        """A sparse cloud can still prove an object too wide.

        The observed extent is a *lower bound* on the object, so if the bound
        alone does not fit the jaws, neither does the object -- and that holds
        however few points drew it. Only the positive verdict needs a quorum.
        """
        pair = resolve_pair(PAIR)
        aperture = gripper_geometry("franka_panda").aperture
        depth = gripper_geometry("franka_panda").tcp_depth
        grasp = grasp_at([0, 0, 0.85 + depth])
        # Four points, fewer than MIN_JAW_WIDTH_POINTS, spanning more than the
        # hand can open along this grasp's own closing axis.
        span = aperture * 1.5
        pts = np.array([grasp.closing * s + [0, 0, 0.85] for s in (-span / 2, span / 2)])
        verdicts = jaw_width_verdicts([grasp], np.array([0]), pts, pair)
        assert verdicts[0] == "too_wide"
        assert len(by_jaw_width([grasp], np.array([0]), pts, pair)) == 0

    def test_the_funnel_marks_a_cell_whose_width_could_not_be_verified(self):
        """The mark is the point of the change.

        Dropping unverifiable grasps would be useless on its own, because the
        funnel restores them when the stage empties the set -- so the run still
        goes ahead. What makes it honest is that the flag says the grasp was
        chosen without a width check, and names the cloud as the reason.
        """
        pair = resolve_pair(PAIR)
        grasp = grasp_at([0, 0, 0.9])
        funnel = filter_grasps([grasp], pair, np.zeros((2, 3)))
        assert funnel.flags["jaw_width"]["unverified"] == 1
        assert funnel.flags["jaw_width"]["fits"] == 0
        assert funnel.flags["cloud_too_sparse_for_jaw_width"] is True
        # and it did not come back empty
        assert len(funnel.survivors) == 1


class TestCollision:
    def test_a_hand_inside_a_wall_is_dropped_and_one_beside_it_is_not(self):
        pair = resolve_pair(PAIR)
        wall = box_cloud([0.3, 0.0, 0.9], [0.005, 0.3, 0.3], n=4000)
        inside = grasp_at([0.3, 0.0, 0.9])
        clear = grasp_at([-0.3, 0.0, 0.9])
        kept = by_collision([inside, clear], np.array([0, 1]), wall, pair, corridor=0.0,
                            samples=1)
        assert kept.tolist() == [1]

    def test_the_approach_corridor_is_checked_not_just_the_final_pose(self):
        """A grasp can be clear where it ends and pass through a neighbour on
        the way in."""
        pair = resolve_pair(PAIR)
        # An obstacle above the grasp, in the path of a downward approach.
        obstacle = box_cloud([0.0, 0.0, 1.15], [0.10, 0.10, 0.01], n=4000)
        grasp = grasp_at([0.0, 0.0, 0.90], approach=[0, 0, -1])
        final_only = by_collision([grasp], np.array([0]), obstacle, pair,
                                  corridor=0.0, samples=1)
        with_corridor = by_collision([grasp], np.array([0]), obstacle, pair,
                                     corridor=0.30, samples=6)
        assert len(final_only) == 1
        assert len(with_corridor) == 0

    def test_a_single_stray_point_does_not_veto_a_grasp(self):
        """Depth edges produce flyer points exactly where two objects meet,
        which is where grasps live."""
        pair = resolve_pair(PAIR)
        stray = np.array([[0.0, 0.0, 0.95]])
        grasp = grasp_at([0.0, 0.0, 0.90])
        assert len(by_collision([grasp], np.array([0]), stray, pair,
                                corridor=0.0, samples=1)) == 1


class TestDuplicates:
    def test_near_identical_grasps_are_thinned_to_the_best(self):
        grasps = [
            grasp_at([0, 0, 0.9], score=0.7),
            grasp_at([0.002, 0, 0.9], score=0.9),   # duplicate of the first
            grasp_at([0.5, 0, 0.9], score=0.6),     # far away, distinct
        ]
        kept = suppress_duplicates(grasps, np.array([0, 1, 2]))
        assert len(kept) == 2
        assert 2 in kept
        assert 1 in kept and 0 not in kept          # the better of the pair

    def test_same_position_different_approach_is_not_a_duplicate(self):
        grasps = [
            grasp_at([0, 0, 0.9], approach=[0, 0, -1]),
            grasp_at([0, 0, 0.9], approach=[1, 0, 0]),
        ]
        assert len(suppress_duplicates(grasps, np.array([0, 1]))) == 2


class TestFunnel:
    def test_a_stage_that_would_empty_the_set_falls_back_and_says_so(self):
        """The load-bearing behaviour. Returning nothing turns 'the hand would
        have collided' into 'there is no grasp', which is a worse answer."""
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        # Every grasp is nowhere near the object, so 'on target' empties.
        grasps = [grasp_at([2.0, 0, 0.9]), grasp_at([2.1, 0, 0.9])]
        funnel = filter_grasps(grasps, PAIR, cloud)
        assert len(funnel.survivors) > 0
        assert funnel.flags.get("on target_fell_back")
        assert "FELL BACK" in funnel.describe()

    def test_the_funnel_records_every_stage(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        depth = gripper_geometry("franka_panda").tcp_depth
        grasps = [grasp_at([0, 0, 0.85 + depth], yaw=y) for y in (0.0, 0.5, 1.0)]
        funnel = filter_grasps(grasps, PAIR, cloud)
        names = [s.name for s in funnel.stages]
        assert names[0] == "generated"
        assert "on target" in names and "jaw width" in names and "distinct" in names
        assert all(s.entered >= s.survived for s in funnel.stages)

    def test_survivors_come_back_best_first(self):
        cloud = box_cloud([0, 0, 0.85], [0.03, 0.03, 0.05])
        depth = gripper_geometry("franka_panda").tcp_depth
        grasps = [
            grasp_at([0, 0, 0.85 + depth], yaw=0.0, score=0.5),
            grasp_at([0.10, 0, 0.85 + depth], yaw=0.0, score=0.9),
        ]
        funnel = filter_grasps(grasps, PAIR, cloud)
        scores = [grasps[i].score for i in funnel.survivors]
        assert scores == sorted(scores, reverse=True)

    def test_the_worst_stage_is_reported_for_explaining_a_failure(self):
        funnel = FilterFunnel()
        funnel.add("visibility", "x", np.arange(10), np.arange(9))
        funnel.add("collision", "y", np.arange(9), np.arange(2))
        assert funnel.rejected_by == "collision"

    def test_nothing_rejected_means_no_blame(self):
        funnel = FilterFunnel()
        funnel.add("visibility", "x", np.arange(5), np.arange(5))
        assert funnel.rejected_by is None


# ---------------------------------------------- the scene-derived zone filters
def test_support_approach_keeps_a_descent_and_rejects_a_hand_coming_up():
    """The table is solid, so nothing reaches an object from underneath it.

    The constraint is the support's own normal and nothing else: no
    demonstration is consulted, which is the whole point of replacing
    ``by_demonstration_consistency`` with this.
    """
    from tpgpt.grasp.filters import by_support_approach

    grasps = [
        grasp_at((0, 0, 1.0), approach=(0, 0, -1)),        # straight down
        grasp_at((0, 0, 1.0), approach=(1, 0, -1)),        # 45 deg, downward
        grasp_at((0, 0, 1.0), approach=(1, 0, -0.02)),     # 88.9 deg, downward
        grasp_at((0, 0, 1.0), approach=(1, 0, 0.3)),       # pointing upward
        grasp_at((0, 0, 1.0), approach=(0, 0, 1)),         # straight up
    ]
    kept = by_support_approach(grasps, np.arange(5))
    # The 88.9 degree one is inside the 90 degree geometric bound and outside
    # the 85 degree limit: the fingers hang below the tool point, so a level
    # hand still grazes the surface.
    assert kept.tolist() == [0, 1]


def test_support_approach_follows_the_normal_it_is_given():
    """A sloped or vertical support changes the vector and nothing else."""
    from tpgpt.grasp.filters import by_support_approach

    sideways = grasp_at((0, 0, 1.0), approach=(1, 0, 0))
    downward = grasp_at((0, 0, 1.0), approach=(0, 0, -1))
    grasps = [sideways, downward]
    # A wall whose outward normal points along -x: now the sideways grasp is
    # the admissible one and the descent is the one arriving through the wall.
    kept = by_support_approach(grasps, np.arange(2), support_normal=(-1, 0, 0))
    assert kept.tolist() == [0]


def test_hand_envelope_reads_the_hand_rather_than_assuming_one():
    from tpgpt.grasp.filters import hand_envelope

    length, radius = hand_envelope(resolve_pair(PAIR))
    # A Panda hand is about 10 cm from fingertip to wrist and a few cm across.
    assert 0.05 < length < 0.30
    assert 0.01 < radius < 0.15


def test_centre_offset_measures_horizontally_only():
    """Gravity acts vertically, so where on the object's height it is gripped
    changes no moment; only the lever arm in the horizontal plane does."""
    from tpgpt.grasp.filters import offset_from_centre

    grasp = grasp_at((0.10, 0.0, 1.0), approach=(0, 0, -1))
    tcp = grasp.tcp_position()
    directly_below = np.array([tcp[0], tcp[1], tcp[2] - 0.20])
    assert offset_from_centre(grasp, directly_below) == 0.0
    sideways = np.array([tcp[0] - 0.03, tcp[1] + 0.04, tcp[2]])
    assert np.isclose(offset_from_centre(grasp, sideways), 0.05)


def test_centre_offset_filter_keeps_the_centred_grip():
    from tpgpt.grasp.filters import MAX_CENTRE_OFFSET, by_centre_offset

    centre = np.array([0.0, 0.0, 0.9])
    near = grasp_at((0.005, 0.0, 1.0), approach=(0, 0, -1))
    far = grasp_at((MAX_CENTRE_OFFSET + 0.02, 0.0, 1.0), approach=(0, 0, -1))
    kept = by_centre_offset([near, far], np.arange(2), centre)
    assert kept.tolist() == [0]


# ------------------------------------------------- the executed trajectory
def straight_path(start, end, n=60, rotation=None):
    """A straight fingertip path with a fixed orientation, as a transport would
    hand it over: positions plus grasp-convention rotations."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    positions = start + t * (end - start)
    # Approach straight down: +Z of the grasp frame is the approach axis.
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]) \
        if rotation is None else np.asarray(rotation, float)
    return positions, np.repeat(R[None], n, axis=0)


def test_path_clearance_finds_a_wall_the_thirteen_pose_sample_would_miss():
    """The failure 7.38 records: a path whose *ends* are clear and whose middle
    is inside a panel. A sampled check passes it; this one does not."""
    from tpgpt.grasp.filters import path_clearance
    from tpgpt.perception.obstacles import Obstacle

    wall = Obstacle("wall", "box", np.array([0.0, 0.0, 1.0]), np.eye(3),
                    np.array([0.30, 0.01, 0.30]))
    positions, rotations = straight_path((0.0, -0.40, 1.0), (0.0, 0.40, 1.0))
    result = path_clearance(
        None, positions, rotations, resolve_pair(PAIR), obstacles=[wall]
    )
    assert result["inside_fraction"] > 0.0
    assert result["max_depth"] > 0.005
    assert result["culprits"] == {"wall": result["inside_waypoints"]}


def test_path_clearance_reports_a_clear_path_as_clear():
    from tpgpt.grasp.filters import path_clearance
    from tpgpt.perception.obstacles import Obstacle

    far_away = Obstacle("wall", "box", np.array([0.0, 0.0, 0.0]), np.eye(3),
                        np.array([0.05, 0.05, 0.05]))
    positions, rotations = straight_path((0.0, -0.40, 1.0), (0.0, 0.40, 1.0))
    result = path_clearance(
        None, positions, rotations, resolve_pair(PAIR), obstacles=[far_away]
    )
    assert result["inside_fraction"] == 0.0
    assert result["max_depth"] == 0.0
    assert result["culprits"] == {}


def test_the_carried_object_is_only_checked_while_it_is_being_carried():
    """Outside the carry the object stands on a surface and is not attached to
    the hand at all, so testing it there reports it colliding with the table it
    is resting on."""
    from tpgpt.grasp.filters import path_clearance
    from tpgpt.perception.obstacles import Obstacle

    # A slab the *object* passes through in the middle of the path, while the
    # hand -- 100 mm above it -- stays clear.
    slab = Obstacle("slab", "box", np.array([0.0, 0.0, 0.85]), np.eye(3),
                    np.array([0.30, 0.02, 0.02]))
    positions, rotations = straight_path((0.0, -0.40, 0.95), (0.0, 0.40, 0.95))
    load = np.array([[0.0, -0.40, 0.85], [0.005, -0.40, 0.85]])

    everywhere = path_clearance(
        None, positions, rotations, resolve_pair(PAIR), obstacles=[slab],
        carried_points=load, carry_span=(0, len(positions) - 1),
    )
    late = path_clearance(
        None, positions, rotations, resolve_pair(PAIR), obstacles=[slab],
        carried_points=load, carry_span=(50, len(positions) - 1),
    )
    assert (everywhere["depth_object"] > 0).any()
    # Restricted to the last ten waypoints, the object is past the slab.
    assert not (late["depth_object"] > 0).any()


def test_path_clearance_refuses_mismatched_positions_and_rotations():
    from tpgpt.grasp.filters import path_clearance

    positions, rotations = straight_path((0, 0, 1), (0, 0.1, 1), n=10)
    with pytest.raises(ValueError, match="against"):
        path_clearance(None, positions, rotations[:5], resolve_pair(PAIR),
                       obstacles=[])
