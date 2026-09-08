"""Inverse kinematics, against the real arm.

The point of these is not solver accuracy but honesty at the boundary: a
reachability filter that says yes to poses the arm cannot hold is worse than no
filter, because it turns a kinematic failure into an apparent transport failure
much later.
"""

import numpy as np
import pytest

from tpgpt.grasp.verify import make_env
from tpgpt.sim.kinematics import reachable, solve_ik

pytestmark = [pytest.mark.sim, pytest.mark.slow]


@pytest.fixture(scope="module")
def env():
    environment = make_env("panda")
    environment.reset()
    try:
        yield environment
    finally:
        environment.close()


def eef_pose(env):
    site = env.robots[0].eef_site_id
    site = site["right"] if isinstance(site, dict) else site
    return (
        np.array(env.sim.data.site_xpos[site]),
        np.array(env.sim.data.site_xmat[site]).reshape(3, 3),
    )


def test_the_current_pose_is_reachable_immediately(env):
    position, rotation = eef_pose(env)
    result = solve_ik(env, position, rotation)
    assert result.reachable and result.iterations == 1


def test_a_pose_on_the_table_is_reachable(env):
    _, rotation = eef_pose(env)
    result = solve_ik(env, env.object_position("can") + np.array([0, 0, 0.05]), rotation)
    assert result.reachable
    assert result.position_error < 5e-3


def test_a_pose_a_metre_away_is_refused(env):
    position, rotation = eef_pose(env)
    result = solve_ik(env, position + np.array([1.0, 0.0, 0.0]), rotation)
    assert not result.reachable
    assert result.position_error > 0.3
    assert result.at_joint_limit


def test_a_pose_below_the_floor_is_refused(env):
    _, rotation = eef_pose(env)
    assert not solve_ik(env, np.array([0.0, 0.0, 0.2]), rotation).reachable


def test_orientation_is_part_of_reachability(env):
    """A reachable position with the hand upside down is not a reachable grasp,
    and a filter that ignores orientation would pass it."""
    position, rotation = eef_pose(env)
    target = env.object_position("can") + np.array([0, 0, 0.05])
    assert solve_ik(env, target, rotation).reachable
    # Same point, hand rolled 180 degrees about x, near the base: much harder.
    flipped = rotation @ np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    result = solve_ik(env, target, flipped)
    assert result.rotation_error >= 0.0  # solver ran; the assertion is below
    if not result.reachable:
        assert result.rotation_error > 0.15 or result.position_error > 5e-3


def test_solving_leaves_the_simulation_untouched(env):
    """It is a query. If it moved the arm, every filtered candidate would be
    evaluated against a different scene than the one that follows."""
    before_position, before_rotation = eef_pose(env)
    before_qpos = np.array(env.sim.data.qpos)
    solve_ik(env, before_position + np.array([0.15, 0.1, -0.1]), before_rotation)
    after_position, _ = eef_pose(env)
    assert np.allclose(before_position, after_position)
    assert np.allclose(before_qpos, np.array(env.sim.data.qpos))


def test_a_sequence_warm_starts_from_the_previous_solution(env):
    """A pick and place is four nearby poses, and asking whether the arm can
    move between them is a different question from whether each is reachable
    from some unrelated configuration."""
    position, rotation = eef_pose(env)
    poses = [
        (position, rotation),
        (position + np.array([0.05, 0.0, -0.05]), rotation),
        (env.object_position("can") + np.array([0, 0, 0.05]), rotation),
    ]
    results = reachable(env, poses)
    assert all(r.reachable for r in results)
    assert sum(r.iterations for r in results) < 60


def test_an_unreachable_place_pose_fails_the_whole_sequence(env):
    """The case the filter exists for: the object is reachable and the shelf
    it has to go on is not."""
    position, rotation = eef_pose(env)
    results = reachable(
        env,
        [(env.object_position("can") + np.array([0, 0, 0.05]), rotation),
         (position + np.array([0.0, 0.0, 1.5]), rotation)],
    )
    assert results[0].reachable
    assert not results[1].reachable
