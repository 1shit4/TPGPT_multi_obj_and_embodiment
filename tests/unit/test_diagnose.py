"""Attributing a failure to the stage that caused it.

The property under test throughout: the diagnosis must name the *first* thing
that went wrong, not the last. A run whose hand closed on empty air and then
"placed" the object 30 cm away must be blamed on the grasp, because fixing the
placement would change nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.experiments.diagnose import (
    attractor_drift,
    closing_budget,
    reach_axes,
    CARRY_FRACTION,
    LIFT_HEIGHT,
    REACH_TOLERANCE,
    STAGES,
    Diagnosis,
    Stage,
    diagnose,
    tally,
    tally_text,
)
from tpgpt.experiments.pipeline import RunResult
from tpgpt.sim.rollout import SimRollout


class FakeEnv:
    def __init__(self, slot=(0.22, 0.0, 1.081)):
        self._slot = np.array(slot, dtype=float)

    def slot_poses(self):
        return {"top_middle": self._slot}


def make_rollout(
    n=200,
    close_step=60,
    release_step=150,
    held_from=None,
    lift=0.30,
    final_xy=(0.22, 0.0),
    success=True,
    reach_offset=0.0,
):
    """A trace of a run that went a particular way.

    ``held_from`` is when the object first touches the hand; ``None`` means it
    never did, which is what a grasp closing on air looks like.
    """
    slot = np.array([0.22, 0.0, 1.081])
    command = -np.ones(n)
    command[close_step:release_step] = 1.0

    z = np.full(n, 0.85)
    if held_from is not None:
        rise = np.linspace(0.0, lift, release_step - held_from)
        z[held_from:release_step] = 0.85 + rise
        z[release_step:] = 0.85 + lift

    held = np.zeros(n)
    if held_from is not None:
        held[held_from:release_step] = 1.0

    x = np.full(n, 0.0)
    y = np.full(n, 0.0)
    x[release_step:] = final_xy[0]
    y[release_step:] = final_xy[1]
    x[close_step:release_step] = np.linspace(0.0, final_xy[0], release_step - close_step)

    positions = np.tile(slot, (n, 1))
    positions[close_step] = slot + np.array([reach_offset, 0.0, 0.0])

    return SimRollout(
        positions=positions,
        attractors=positions.copy(),
        velocities=np.zeros((n, 3)),
        time_belief=np.linspace(0, 1, n),
        velocity_std=np.zeros((n, 3)),
        gripper=command,
        contact_force=np.zeros(n),
        success=success,
        final_offset=np.array([final_xy[0] - 0.22, final_xy[1], 0.0]),
        metadata={
            "steps": n,
            "approach_gap": 0.04,
            "probe": {
                "object_x": x, "object_y": y, "object_z": z,
                "jaw": np.zeros(n), "held": held,
            },
        },
    )


def make_result(rollout, transported_start=(0.22, 0.0, 1.081)) -> RunResult:
    result = RunResult(prompt="p", gripper="panda", shelf_variant="cubby", seed=0)
    result.object_name, result.slot = "can", "top_middle"
    result.rollout = rollout
    result.transported = np.array([transported_start], dtype=float)
    return result


class TestStageAttribution:
    def test_a_clean_run_passes_every_stage(self):
        rollout = make_rollout(held_from=62)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "success"
        assert d.first_failure is None
        assert [s.name for s in d.stages] == list(STAGES)

    def test_closing_on_air_is_blamed_on_the_grasp_not_the_placement(self):
        """The whole point of the module: a hand that gripped nothing must not
        be reported as a placement error, even though the object is far from
        the slot at the end."""
        rollout = make_rollout(held_from=None, lift=0.0,
                               final_xy=(0.0, 0.0), success=False)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "grasp"
        assert "closed on nothing" in d.first_failure.detail

    def test_a_dropped_object_is_blamed_on_the_carry(self):
        rollout = make_rollout(held_from=62, success=False, final_xy=(0.0, 0.0))
        # Let go a third of the way through the carry.
        rollout.metadata["probe"]["held"][90:] = 0.0
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "carry"
        assert d.measurements["carry_fraction"] < CARRY_FRACTION

    def test_a_good_grasp_that_misses_the_slot_is_blamed_on_the_placement(self):
        rollout = make_rollout(held_from=62, success=False, final_xy=(0.0, 0.30))
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "place"

    def test_a_hand_that_never_arrived_is_blamed_on_the_reach(self):
        rollout = make_rollout(held_from=None, lift=0.0, success=False)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]) + np.array([0.2, 0, 0]))
        assert d.blame == "reach"
        assert d.measurements["reach_error"] > REACH_TOLERANCE

    def test_a_nudge_is_not_a_lift(self):
        rollout = make_rollout(held_from=62, lift=LIFT_HEIGHT / 4, success=False)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "lift"

    def test_a_gripper_that_never_closes_is_caught(self):
        """This is the bug that made every earlier campaign meaningless, in
        reverse: a rollout whose gripper channel never commands a close."""
        rollout = make_rollout(held_from=None, success=False)
        rollout.gripper[:] = -1.0
        d = diagnose(make_result(rollout), FakeEnv())
        assert d.blame == "reach"
        assert "never commanded to close" in d.first_failure.detail
        # Later stages are reported as not reached, never as passed.
        assert [s.name for s in d.stages] == list(STAGES)
        assert all(not s.ok for s in d.stages[1:])


class TestToolFrame:
    """When the rollout runs in the tool frame it already records fingertips."""

    def test_the_offset_is_not_added_twice(self):
        """The bug this guards against reported three runs that delivered their
        object into the slot as having failed to reach it -- in the same table
        that reported them successful."""
        rollout = make_rollout(held_from=62)
        rollout.metadata["tool_offset"] = [0.0, 0.0, -0.0411]
        d = diagnose(self._with_grasp(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.measurements["reach_error"] < 0.001
        assert d.blame == "success"

    def _with_grasp(self, rollout):
        """A result carrying a real grasp, so the offset lookup actually runs."""
        from tpgpt.grasp.grasps import Grasp6D

        result = make_result(rollout)
        result.grasp = Grasp6D(pose=np.eye(4), score=1.0,
                               gripper="franka_panda", width=0.08)
        return result

    @pytest.mark.parametrize("offset", [[0.0, 0.0, 0.0], None])
    def test_a_wrist_frame_rollout_still_adds_it(self, offset):
        """Zero, or no statement at all, both mean the rollout recorded the
        wrist -- so the fingertip offset has to be applied."""
        from tpgpt.grasp.grasps import contact_offset

        rollout = make_rollout(held_from=62)
        if offset is None:
            rollout.metadata.pop("tool_offset", None)
        else:
            rollout.metadata["tool_offset"] = offset
        result = self._with_grasp(rollout)
        d = diagnose(result, FakeEnv(), grasp_point=np.array([0.22, 0.0, 1.081]))
        expected = np.linalg.norm(contact_offset("panda"))
        assert d.measurements["reach_error"] == pytest.approx(expected, abs=2e-3)


class TestNoTrace:
    def test_a_run_that_never_executed_says_so(self):
        result = RunResult(prompt="p", gripper="panda", shelf_variant="cubby", seed=0)
        assert diagnose(result, FakeEnv()).blame == "approach"

    def test_a_rollout_without_a_probe_says_so(self):
        """A missing trace is a gap in the *instrument*, not a task failure, so
        it is reported whether or not the run worked."""
        for success in (False, True):
            rollout = make_rollout(held_from=62, success=success)
            rollout.metadata.pop("probe")
            d = diagnose(make_result(rollout), FakeEnv())
            assert any("no object trace" in s.detail for s in d.stages)
            assert d.succeeded is success


class TestOutcomeWinsOverProxies:
    """The stage thresholds are proxies; the task outcome is ground truth."""

    def test_a_successful_run_is_never_blamed_on_a_stage(self):
        """A hand can arrive further from the planned grasp than the tolerance
        allows, grip a different part of the object, and still do the task."""
        rollout = make_rollout(held_from=62, success=True, reach_offset=0.20)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        assert d.blame == "success"
        assert "reach" in d.proxy_misfires
        assert "read as failed" in d.summary_of_misfires()

    def test_the_funnel_never_reports_fewer_finishing_than_succeeding(self):
        rollout = make_rollout(held_from=62, success=True, reach_offset=0.20)
        d = diagnose(make_result(rollout), FakeEnv(),
                     grasp_point=np.array([0.22, 0.0, 1.081]))
        counts = tally([d])
        assert counts["settle"] == 1
        assert all(counts[name] == 1 for name in STAGES)


class TestTally:
    def test_counting_stops_at_the_first_failure(self):
        """A run that fails at 'grasp' must not be counted as having lifted,
        even if a later stage happens to read as ok."""
        early = Diagnosis(stages=[
            Stage("approach", True, ""), Stage("reach", True, ""),
            Stage("grasp", False, ""), Stage("lift", True, ""),
            Stage("carry", True, ""), Stage("place", True, ""),
            Stage("settle", True, ""),
        ])
        counts = tally([early])
        assert counts["reach"] == 1
        assert counts["grasp"] == 0
        assert counts["lift"] == 0

    def test_the_table_names_every_stage(self):
        text = tally_text(tally([]), 0)
        for name in STAGES:
            assert name in text

    @pytest.mark.parametrize("blame", ["grasp", "lift", "carry", "place"])
    def test_blame_matches_the_first_cross(self, blame):
        stages = [Stage(n, n != blame, "") for n in STAGES]
        for i, stage in enumerate(stages):
            if stage.name == blame:
                for later in stages[i + 1:]:
                    later.ok = True
                break
        assert Diagnosis(stages=stages).blame == blame


def _grasp(gripper="panda"):
    """A grasp at the origin with the identity frame.

    With an identity pose the axes are the world axes, which makes the axis
    decomposition readable in the tests: closing is ``+X`` (the convention in
    :mod:`tpgpt.grasp.grasps`), approach is ``+Z``, and the remaining finger
    axis ``approach x closing`` is ``+Y``.
    """
    from tpgpt.grasp.grasps import Grasp6D

    return Grasp6D(pose=np.eye(4), score=1.0, gripper=gripper, width=0.08)


class TestReachAxes:
    """Splitting a reach error into the three axes that mean different things.

    The scalar distance this replaces mixed a millimetre-scale budget (across
    the jaws) with a 130 mm one (along the approach), and on the campaigns since
    deleted it rated the better keypoint configuration worse. Section 7.26.
    """

    def test_an_error_across_the_jaws_lands_on_the_closing_axis(self):
        axes = reach_axes(_grasp(), np.array([0.03, 0.0, 0.0]))
        assert axes["closing"] == pytest.approx(0.03)
        assert axes["approach"] == pytest.approx(0.0)
        assert axes["jaw"] == pytest.approx(0.0)

    def test_an_error_along_the_approach_does_not_pollute_the_closing_axis(self):
        """The whole point: 60 mm deep and 60 mm off-centre are not the same run.

        A straight-line distance scores them identically. One is a grip taken
        slightly high on the object and usually works; the other closes on air.
        """
        deep = reach_axes(_grasp(), np.array([0.0, 0.0, 0.06]))
        across = reach_axes(_grasp(), np.array([0.06, 0.0, 0.0]))
        assert deep["closing"] == pytest.approx(0.0)
        assert across["closing"] == pytest.approx(0.06)
        assert np.linalg.norm([0.0, 0.0, 0.06]) == pytest.approx(
            np.linalg.norm([0.06, 0.0, 0.0])
        )  # identical to the scalar, and that is the defect

    def test_the_third_axis_runs_along_the_fingers(self):
        axes = reach_axes(_grasp(), np.array([0.0, 0.02, 0.0]))
        assert axes["jaw"] == pytest.approx(0.02)
        assert axes["closing"] == pytest.approx(0.0)

    def test_short_and_deep_are_distinguished_by_sign(self):
        """Stopping short and driving past fail differently; keep the sign."""
        short = reach_axes(_grasp(), np.array([0.0, 0.0, -0.02]))
        deep = reach_axes(_grasp(), np.array([0.0, 0.0, 0.02]))
        assert short["signed_approach"] > 0 > deep["signed_approach"]
        assert short["approach"] == deep["approach"] == pytest.approx(0.02)

    def test_the_axes_recover_the_scalar_in_quadrature(self):
        error = np.array([0.01, -0.02, 0.03])
        axes = reach_axes(_grasp(), error)
        recovered = np.linalg.norm(
            [axes["closing"], axes["jaw"], axes["approach"]]
        )
        assert recovered == pytest.approx(np.linalg.norm(error))


def _aperture_or_skip(short_name="panda") -> float:
    """The hand's jaw aperture, looked up by **registry key**.

    Deliberately passes the short name rather than the GraspGen-X one, because
    resolving that is the part worth testing: ``gripper_geometry`` alone only
    accepts ``franka_panda``, so a by-identity lookup would quietly return no
    budget at all for every caller that speaks in registry keys.
    """
    from tpgpt.grasp.grippers import gripper_geometry, resolve_pair
    try:
        return float(gripper_geometry(resolve_pair(short_name).graspgen).aperture)
    except Exception:  # pragma: no cover - absent sibling checkout
        pytest.skip("gripper geometry needs the sibling GraspGen-X checkout")


class TestClosingBudget:
    def test_it_returns_none_rather_than_a_plausible_default(self):
        """A missing measurement must refuse, not invent.

        This is the lesson of section 7.13, where ``contact_offset`` returned a
        zero for an unmeasured hand and a 41 mm frame error read for weeks as
        the arm simply missing its target.
        """
        result = RunResult(prompt="p", gripper="", shelf_variant="cubby", seed=0)
        assert closing_budget(result) is None

    def test_a_wider_object_leaves_a_smaller_budget(self):
        """``(aperture - width) / 2`` is the room left on each side.

        Measured along *this grasp's* closing axis rather than from an
        axis-aligned box: a 30 x 100 mm box yawed 45 degrees measures 92 x 92
        and would read as ungraspable (section 7.18).
        """
        aperture = _aperture_or_skip()

        class Keys:
            def __init__(self, half):
                self.points = np.array([[-half, 0, 0], [half, 0, 0]])

        narrow = RunResult(prompt="p", gripper="panda", shelf_variant="c", seed=0)
        narrow.grasp, narrow.target_keypoints = _grasp(), Keys(0.010)
        wide = RunResult(prompt="p", gripper="panda", shelf_variant="c", seed=0)
        wide.grasp, wide.target_keypoints = _grasp(), Keys(0.030)

        assert closing_budget(narrow) == pytest.approx((aperture - 0.020) / 2)
        assert closing_budget(wide) < closing_budget(narrow)

    def test_an_object_wider_than_the_hand_has_no_budget_at_all(self):
        aperture = _aperture_or_skip()

        class Keys:
            points = np.array([[-aperture, 0, 0], [aperture, 0, 0]])

        result = RunResult(prompt="p", gripper="panda", shelf_variant="c", seed=0)
        result.grasp, result.target_keypoints = _grasp(), Keys()
        assert closing_budget(result) == 0.0


class TestAttractorDrift:
    """Pointwise deviation of the integrated attractor from its planned path."""

    def _path(self, n=40):
        return np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)

    def test_an_attractor_that_stayed_on_the_path_has_no_drift(self):
        path = self._path()
        rollout = make_rollout()
        rollout.attractors = path.copy()
        assert attractor_drift(rollout, path)["drift_max"] == pytest.approx(0.0)

    def test_drift_is_reported_at_its_true_size_and_place(self):
        path = self._path()
        attractors = path.copy()
        attractors[20, 1] = 0.04
        rollout = make_rollout()
        rollout.attractors = attractors
        out = attractor_drift(rollout, path)
        assert out["drift_max"] == pytest.approx(0.04)
        assert out["drift_at_step"] == 20

    def test_drift_is_never_negative(self):
        """The measure this replaces could be, being a difference of two minima."""
        rng = np.random.default_rng(1)
        path = self._path()
        rollout = make_rollout()
        rollout.attractors = path + rng.normal(scale=0.005, size=path.shape)
        out = attractor_drift(rollout, path)
        assert out["drift_max"] >= 0 and out["drift_median"] >= 0

    def test_missing_inputs_report_nothing_rather_than_zero(self):
        """"Not measured" and "measured as zero" must be distinguishable."""
        rollout = make_rollout()
        rollout.attractors = self._path()
        assert attractor_drift(rollout, None) == {}
