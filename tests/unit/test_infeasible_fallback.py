"""Handling commanded poses the arm cannot achieve.

The controller has no inverse kinematics: it turns a pose error into a force
and pulls. Against a pose that does not exist for this arm it pulls forever,
and the lag gate reports that as a stall. These tests cover the decision logic
that recognises the situation and does something else instead -- the physics is
covered by the campaigns, but the *branches* are rare enough there that a
regression could hide for a long time.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.sim.rollout import _feasibility_action, _next_feasible


class FakeIK:
    """Stands in for ``solve_ik`` with a scripted notion of what is reachable.

    ``position_ok`` and ``orientation_ok`` are callables of the queried
    position, so a test can carve the workspace up however it needs.
    """

    def __init__(self, position_ok=lambda p: True, orientation_ok=lambda p: True):
        self.position_ok = position_ok
        self.orientation_ok = orientation_ok
        self.calls = 0

    def __call__(self, env, position, rotation=None, **kwargs):
        self.calls += 1
        ok = self.position_ok(position)
        if rotation is not None:
            ok = ok and self.orientation_ok(position)
        return type("R", (), {"reachable": bool(ok), "qpos": np.zeros(7)})()


class FakePolicy:
    def __init__(self, n=40):
        from tpgpt.transport.labels import PolicyLabels

        self.labels = PolicyLabels(
            positions=np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1),
            orientations=np.tile(np.eye(3), (n, 1, 1)),
            gripper=np.full(n, -1.0),
            time_belief=np.linspace(0, 1, n),
        )


@pytest.fixture
def patched(monkeypatch):
    """Install a scripted IK in place of the real solver."""
    def install(fake):
        import tpgpt.sim.kinematics as kin

        monkeypatch.setattr(kin, "solve_ik", fake)
        return fake
    return install


class TestFeasibilityAction:
    def test_a_reachable_pose_changes_nothing(self, patched):
        patched(FakeIK())
        relax, skip = _feasibility_action(
            None, None, np.zeros(3), np.eye(3), np.zeros(3),
            FakePolicy(), 0.0, 0.015, None,
        )
        assert relax is False and skip is None

    def test_an_unreachable_hand_angle_relaxes_the_orientation(self, patched):
        """The common case: the point is fine, the wrist angle is not. Giving up
        the angle is nearly free in transit and keeps the path."""
        patched(FakeIK(orientation_ok=lambda p: False))
        relax, skip = _feasibility_action(
            None, None, np.zeros(3), np.eye(3), np.zeros(3),
            FakePolicy(), 0.0, 0.015, None,
        )
        assert relax is True
        assert skip is None

    def test_an_unreachable_position_skips_ahead(self, patched):
        """Nothing to relax: the point itself is out of reach, so the attractor
        has to go somewhere else."""
        patched(FakeIK(position_ok=lambda p: p[0] > 0.5))
        relax, skip = _feasibility_action(
            None, None, np.zeros(3), np.eye(3), np.zeros(3),
            FakePolicy(), 0.0, 0.015, None,
        )
        assert relax is False
        assert skip is not None and skip > 0

    def test_a_skip_in_progress_is_not_re_decided(self, patched):
        """Re-choosing a target every few steps makes it flicker, and the
        attractor chases a moving point instead of leaving the dead region."""
        fake = patched(FakeIK(position_ok=lambda p: False))
        relax, skip = _feasibility_action(
            None, None, np.zeros(3), np.eye(3), np.zeros(3),
            FakePolicy(), 0.0, 0.015, 17,
        )
        assert skip == 17
        assert fake.calls == 0, "a skip in progress must not cost an IK solve"


class TestNextFeasible:
    def test_it_finds_a_reachable_label_ahead(self, patched):
        patched(FakeIK(position_ok=lambda p: p[0] > 0.5))
        i = _next_feasible(None, FakePolicy(), 0.0, np.zeros(3), 0.015)
        assert i is not None
        assert FakePolicy().labels.positions[i][0] > 0.5

    def test_it_searches_forward_only(self, patched):
        """Going backwards would undo progress and can loop."""
        patched(FakeIK(position_ok=lambda p: p[0] < 0.2))
        i = _next_feasible(None, FakePolicy(), 0.5, np.zeros(3), 0.015)
        assert i is None or i >= 20

    def test_nothing_reachable_ahead_returns_none(self, patched):
        """Then the caller carries on and the watchdog ends the run, which is
        honest: the rest of the path is not executable."""
        patched(FakeIK(position_ok=lambda p: False))
        assert _next_feasible(None, FakePolicy(), 0.0, np.zeros(3), 0.015) is None


class TestFilterCorridors:
    def test_the_reachability_filter_checks_more_than_its_endpoints(self):
        """A candidate reachable at the standoff and at the grasp but not in
        between must be rejected; a hand that cannot hold its angle above the
        object never reaches the object."""
        from tpgpt.grasp.filters import APPROACH_FRACTIONS, LIFT_FRACTIONS

        assert len(APPROACH_FRACTIONS) >= 4
        assert 1.0 in APPROACH_FRACTIONS and 0.0 in APPROACH_FRACTIONS
        # Intermediate points, not just the two ends.
        assert any(0.0 < f < 1.0 for f in APPROACH_FRACTIONS)
        assert len(LIFT_FRACTIONS) >= 1
