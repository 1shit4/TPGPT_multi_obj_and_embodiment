"""The execution arithmetic, checked against transcriptions of the original.

These are the most-debugged expressions in the project, and until now they had
**no direct test at all** -- ``rollout_policy`` was exercised only through a
four-minute end-to-end campaign, so every bug in them had to be found in
physics. Two of the three recorded bugs were in the *interaction* between the
gate and the clamp rather than in either alone (``ROBOTICS_NOTES`` 7.14, 7.16).

Each test below inlines a **verbatim transcription** of the expression as it
stood in ``tpgpt/sim/rollout.py`` before extraction, and asserts exact equality
with the extracted helper. That is deliberate: it means editing a helper without
consciously editing its transcription fails the suite, so the helper cannot
quietly drift away from the behaviour the validated 17/20 result was measured
under. Equality is ``==``, not ``approx`` -- a single-ulp change in the gate
diverges a trajectory within tens of steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.policy.rollout import (
    clamp_attractor,
    clamp_speed,
    compliance_norm,
    lag_gate,
    phase_epsilon_from,
    steer_attractor,
    stiffness_threshold,
)

#: The demonstration's own gains: 350 N/m free, 900 N/m at the insertion, with a
#: damping ratio of 0.9 (``tpgpt/sim/demo.py::_stiffness_matrices``).
K_FREE, D_FREE = np.eye(3) * 350.0, np.eye(3) * (2 * np.sqrt(350.0) * 0.9)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestComplianceNorm:
    def test_it_matches_the_original_expression(self):
        original = float(np.linalg.norm(np.linalg.solve(K_FREE, D_FREE), ord=2))
        assert compliance_norm(K_FREE, D_FREE) == original

    def test_it_is_the_impedance_time_constant(self):
        """``||K^-1 D||`` has units of seconds and sets the tracking lag.

        With these gains it is 96 ms, so an arm following a setpoint moving at
        0.25 m/s settles 24 mm behind it -- the number ``ROBOTICS_NOTES`` 2.7
        measures as the demonstration's own mean lag (23 mm).
        """
        assert compliance_norm(K_FREE, D_FREE) == pytest.approx(0.0962, abs=1e-4)
        assert compliance_norm(K_FREE, D_FREE) * 0.25 == pytest.approx(0.024, abs=1e-3)

    def test_a_full_matrix_stiffness_is_handled(self):
        """Transport rotates the stiffness ellipsoid, so K is not diagonal."""
        th = 0.4
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        K, D = R @ K_FREE @ R.T, R @ D_FREE @ R.T
        assert compliance_norm(K, D) == pytest.approx(compliance_norm(K_FREE, D_FREE))


class TestClampSpeed:
    def test_it_matches_the_original_in_place_expression(self, rng):
        for _ in range(200):
            v = rng.normal(size=3) * 0.3
            limit = 0.25
            original = v.copy()
            speed = float(np.linalg.norm(original))
            if speed > limit:
                original *= limit / speed
            assert np.array_equal(clamp_speed(v, limit), original)

    def test_a_slow_command_is_untouched(self):
        v = np.array([0.01, 0.0, 0.0])
        assert np.array_equal(clamp_speed(v, 0.25), v)

    def test_direction_is_preserved_and_magnitude_capped(self):
        out = clamp_speed(np.array([3.0, 4.0, 0.0]), 1.0)
        assert np.linalg.norm(out) == pytest.approx(1.0)
        assert out[0] / out[1] == pytest.approx(3.0 / 4.0)

    def test_it_does_not_mutate_its_input(self):
        """The original mutated the prediction; three later reads relied on it.

        Returning a new array is what lets the caller keep an explicit local,
        but it also means a caller that forgets to switch its logging site
        records the *unclamped* velocity -- invisible in any success rate.
        """
        v = np.array([3.0, 4.0, 0.0])
        clamp_speed(v, 1.0)
        assert np.array_equal(v, [3.0, 4.0, 0.0])


class TestLagGate:
    def _original(self, position, attractor, expected, static_sag, lag_tolerance):
        gate, behind = 1.0, 0.0
        if lag_tolerance:
            behind = float(np.linalg.norm(position - attractor))
            excess = behind - expected - static_sag
            gate = float(np.clip(1.0 - excess / lag_tolerance, 0.0, 1.0))
        return gate, behind

    def test_it_matches_the_original_expression(self, rng):
        for _ in range(300):
            pos, att = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1
            expected, sag = abs(rng.normal()) * 0.02, abs(rng.normal()) * 0.005
            assert lag_gate(pos, att, expected, sag, 0.035) == self._original(
                pos, att, expected, sag, 0.035
            )

    def test_an_arm_that_is_keeping_up_gets_a_fully_open_gate(self):
        """Trailing by exactly the physics-expected amount is not a fault."""
        pos, att = np.zeros(3), np.array([0.024, 0.0, 0.0])
        gate, behind = lag_gate(pos, att, expected=0.024, static_sag=0.0, lag_tolerance=0.035)
        assert gate == 1.0 and behind == pytest.approx(0.024)

    def test_it_gates_on_excess_lag_not_raw_lag(self):
        """The whole design: a fast segment must not throttle itself.

        24 mm behind at 0.25 m/s is correct and must score 1.0; 24 mm behind
        while commanded to hold still is a genuine fault and must throttle.
        """
        pos, att = np.zeros(3), np.array([0.024, 0.0, 0.0])
        moving, _ = lag_gate(pos, att, 0.024, 0.0, 0.035)
        dwelling, _ = lag_gate(pos, att, 0.0, 0.0, 0.035)
        assert moving == 1.0
        assert dwelling < 1.0

    def test_it_closes_fully_once_excess_reaches_the_tolerance(self):
        pos, att = np.zeros(3), np.array([0.040, 0.0, 0.0])
        assert lag_gate(pos, att, 0.0, 0.0, 0.035)[0] == 0.0

    def test_static_sag_is_forgiven(self):
        """Lag the arm cannot close -- a held weight -- must not shut the gate.

        Without this the gate waits forever for an arrival that is not coming.
        """
        pos, att = np.zeros(3), np.array([0.030, 0.0, 0.0])
        assert lag_gate(pos, att, 0.0, 0.0, 0.035)[0] < 1.0
        assert lag_gate(pos, att, 0.0, 0.030, 0.035)[0] == 1.0

    def test_a_disabled_gate_returns_zero_not_nan_for_behind(self):
        """``behind`` feeds ``blocked_lag - behind`` in the watchdog.

        A NaN there makes every comparison false, so the run would take the
        sag-rebaseline path three times and report a stall where there is none.
        """
        gate, behind = lag_gate(np.zeros(3), np.ones(3), 0.0, 0.0, None)
        assert gate == 1.0
        assert behind == 0.0 and not np.isnan(behind)


class TestClampAttractor:
    def _original(self, attractor, position, compliance, speed_limit, expected,
                  static_sag, lag_tolerance):
        lag = attractor - position
        max_lag = speed_limit * compliance
        if lag_tolerance:
            max_lag = min(max_lag, expected + static_sag + lag_tolerance)
        lag_norm = float(np.linalg.norm(lag))
        if lag_norm > max_lag:
            return position + lag * (max_lag / lag_norm), max_lag, True
        return attractor, max_lag, False

    def test_it_matches_the_original_expression(self, rng):
        for _ in range(300):
            att, pos = rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.2
            comp, expected = 0.0962, abs(rng.normal()) * 0.02
            got = clamp_attractor(att, pos, comp, 0.75, expected, 0.0, 0.035)
            want = self._original(att, pos, comp, 0.75, expected, 0.0, 0.035)
            assert np.array_equal(got[0], want[0])
            assert got[1] == want[1] and got[2] == want[2]

    def test_the_clamp_never_exceeds_the_gate_reopening_threshold(self, rng):
        """The 7.14 deadlock, asserted unreachable rather than merely fixed.

        The clamp once allowed ``speed_limit * compliance`` -- from the fastest
        label -- while the gate reopened only within
        ``expected + sag + tolerance`` -- from the current speed. Whenever the
        first exceeded the second, the clamp parked the attractor beyond the
        gate's own reopening threshold and the two locked: one run sat at phase
        0.18 for 112 of 361 steps. Taking the smaller makes it impossible.
        """
        tol = 0.035
        for _ in range(300):
            att, pos = rng.normal(size=3) * 0.3, rng.normal(size=3) * 0.3
            expected, sag = abs(rng.normal()) * 0.02, abs(rng.normal()) * 0.01
            out, max_lag, _ = clamp_attractor(att, pos, 0.0962, 0.75, expected, sag, tol)
            assert max_lag <= expected + sag + tol + 1e-12
            assert float(np.linalg.norm(out - pos)) <= max_lag + 1e-9

    def test_an_attractor_already_close_is_left_exactly_alone(self):
        att, pos = np.array([0.01, 0.0, 0.0]), np.zeros(3)
        out, _, clamped = clamp_attractor(att, pos, 0.0962, 0.75, 0.02, 0.0, 0.035)
        assert np.array_equal(out, att) and clamped is False

    def test_clamping_preserves_direction(self):
        att, pos = np.array([1.0, 1.0, 0.0]), np.zeros(3)
        out, max_lag, clamped = clamp_attractor(att, pos, 0.0962, 0.75, 0.0, 0.0, 0.035)
        assert clamped is True
        assert np.linalg.norm(out - pos) == pytest.approx(max_lag)
        assert out[0] == pytest.approx(out[1])

    def test_without_a_tolerance_the_speed_limit_alone_bounds_it(self):
        """The ``lag_tolerance=None`` ablation must not silently tighten.

        Computing ``expected`` unconditionally and always taking the ``min()``
        would make the ablation that exists to show the gate is needed run with
        a *tighter* clamp than the configuration it is compared against.
        """
        att, pos = np.array([1.0, 0.0, 0.0]), np.zeros(3)
        _, max_lag, _ = clamp_attractor(att, pos, 0.0962, 0.75, 0.0, 0.0, None)
        assert max_lag == pytest.approx(0.75 * 0.0962)


class TestSteerAttractor:
    def test_it_matches_the_original_expression(self, rng):
        for _ in range(200):
            att, tgt = rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.2
            direction = tgt - att
            distance = float(np.linalg.norm(direction))
            if distance <= 0.02:
                want, arrived = att, True
            else:
                want = att + direction / distance * min(0.75 * 0.05, distance)
                arrived = False
            got, got_arrived = steer_attractor(att, tgt, 0.75, 0.05, 0.02)
            assert np.array_equal(got, want) and got_arrived == arrived

    def test_it_reports_arrival_inside_the_tolerance(self):
        att = np.zeros(3)
        _, arrived = steer_attractor(att, np.array([0.01, 0, 0]), 0.75, 0.05, 0.02)
        assert arrived is True

    def test_a_step_never_overshoots_the_target(self):
        att, tgt = np.zeros(3), np.array([0.025, 0.0, 0.0])
        out, _ = steer_attractor(att, tgt, 10.0, 0.05, 0.02)
        assert float(np.linalg.norm(out - att)) <= float(np.linalg.norm(tgt - att)) + 1e-12

    def test_the_step_is_capped_at_the_demonstrated_speed(self):
        """A detour must be a motion, not a jump."""
        out, _ = steer_attractor(np.zeros(3), np.array([5.0, 0, 0]), 0.75, 0.05, 0.02)
        assert float(np.linalg.norm(out)) == pytest.approx(0.75 * 0.05)


class _Labels:
    def __init__(self, time_rate=None, n=200, stiffness=None, damping=None):
        self.time_rate = time_rate
        self.positions = np.zeros((n, 3))
        self.stiffness = stiffness
        self.damping = damping


class TestPhaseEpsilon:
    def test_it_matches_the_original_expression(self):
        labels = _Labels(time_rate=np.full(200, 1.0 / (0.05 * 199)))
        nominal_rate = float(np.mean(labels.time_rate))
        original = max(1e-6, 0.25 * nominal_rate * 0.05 * 40)
        assert phase_epsilon_from(labels, 0.05, 40, 0.25) == original

    def test_it_falls_back_when_the_demonstration_has_no_rate(self):
        labels = _Labels(time_rate=None, n=200)
        original = max(1e-6, 0.25 * (1.0 / 200 / 0.05) * 0.05 * 40)
        assert phase_epsilon_from(labels, 0.05, 40, 0.25) == original

    def test_it_scales_with_the_length_of_the_task(self):
        """Derived from the demonstration's pace, not a magic number."""
        short = phase_epsilon_from(_Labels(np.full(100, 1 / (0.05 * 99))), 0.05, 40, 0.25)
        long = phase_epsilon_from(_Labels(np.full(400, 1 / (0.05 * 399))), 0.05, 40, 0.25)
        assert short > long

    def test_it_is_never_zero(self):
        """A zero threshold makes any crawl read as progress."""
        assert phase_epsilon_from(_Labels(np.zeros(200)), 0.05, 40, 0.0) == 1e-6


class TestStiffnessThreshold:
    def test_it_matches_the_original_expression(self):
        K = np.stack([np.eye(3) * k for k in (350.0, 350.0, 900.0)])
        D = np.stack([np.eye(3) * (2 * np.sqrt(k) * 0.9) for k in (350.0, 350.0, 900.0)])
        labels = _Labels(stiffness=K, damping=D)
        compliances = np.array([compliance_norm(k, d) for k, d in zip(K, D)])
        assert stiffness_threshold(labels) == float(
            0.5 * (compliances.min() + compliances.max())
        )

    def test_it_sits_between_the_soft_and_firm_segments(self):
        """It must separate free-space transit from the grasp and insertion."""
        K = np.stack([np.eye(3) * k for k in (350.0, 900.0)])
        D = np.stack([np.eye(3) * (2 * np.sqrt(k) * 0.9) for k in (350.0, 900.0)])
        threshold = stiffness_threshold(_Labels(stiffness=K, damping=D))
        firm = compliance_norm(K[1], D[1])       # insertion: stiff, low compliance
        soft = compliance_norm(K[0], D[0])       # free space: soft, high compliance
        assert firm < threshold < soft

    def test_labels_without_stiffness_give_zero(self):
        assert stiffness_threshold(_Labels()) == 0.0


def test_the_policy_layer_does_not_import_the_simulator():
    """The layering that keeps the physics-free bed fast.

    ``tpgpt/sim/`` may import ``tpgpt/policy/``; the reverse would drag MuJoCo
    and robosuite into every fast unit test and into the surrogate plant.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, tpgpt.policy.rollout as r; "
         "print(any(m.startswith('tpgpt.sim') for m in sys.modules))"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


# ---------------------------------------------------------------------------
# The attractor laws and the surrogate-plant bed.
# ---------------------------------------------------------------------------

from tpgpt.policy.gp_policy import GPPolicy            # noqa: E402
from tpgpt.policy.rollout import (                     # noqa: E402
    AnchorSchedule,
    ExecutionLaw,
    advance_attractor,
    advance_step,
    attractor_step,
    gate_step,
    rollout_impedance,
)
from tpgpt.transport.labels import PolicyLabels        # noqa: E402

DT = 1.0 / 20.0


def straight_line_labels(n=120, speed=0.168):
    """A constant-speed straight path: the case with a closed-form lag."""
    x = np.zeros((n, 3))
    x[:, 0] = np.linspace(0.0, speed * DT * (n - 1), n)
    return PolicyLabels(
        positions=x, velocities=np.gradient(x, DT, axis=0),
        stiffness=np.tile(K_FREE, (n, 1, 1)), damping=np.tile(D_FREE, (n, 1, 1)),
        time_belief=np.linspace(0, 1, n), time_rate=np.full(n, 1 / (DT * (n - 1))),
    )


class _Prediction:
    """Minimal stand-in for a PolicyPrediction with one row."""

    def __init__(self, velocity, reference=None, K=K_FREE, D=D_FREE):
        self.velocity = np.atleast_2d(velocity)
        self.reference = None if reference is None else np.atleast_2d(reference)
        self.stiffness, self.damping = np.array([K]), np.array([D])


class TestAdvanceAttractor:
    def test_integrate_matches_the_original_expression(self, rng):
        for _ in range(200):
            a, v, gate = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.2, rng.random()
            assert np.array_equal(
                advance_attractor(a, v, DT, gate), a + v * DT * gate
            )

    def test_a_zero_gain_anchor_is_bitwise_the_integrate_law(self, rng):
        """The property that protects the validated 17/20 path.

        Every default must reduce to exactly today's behaviour, bit for bit,
        so a law parameter cannot move the regression by accident.
        """
        for _ in range(200):
            a, v, r, gate = (rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.2,
                             rng.normal(size=3) * 0.1, rng.random())
            assert np.array_equal(
                advance_attractor(a, v, DT, gate, law="anchor", reference=r,
                                  anchor_gain=0.0),
                advance_attractor(a, v, DT, gate),
            )

    def test_reference_is_the_anchor_at_full_gain(self, rng):
        for _ in range(100):
            a, v, r = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.2, rng.normal(size=3)
            assert np.allclose(
                advance_attractor(a, v, DT, 1.0, law="reference", reference=r),
                advance_attractor(a, v, DT, 1.0, law="anchor", reference=r,
                                  anchor_gain=1.0),
            )

    def test_reference_at_full_gate_lands_on_the_reference(self):
        out = advance_attractor(np.zeros(3), np.ones(3), DT, 1.0,
                                law="reference", reference=np.array([0.5, 0.0, 0.0]))
        assert np.allclose(out, [0.5, 0.0, 0.0])

    def test_the_result_lies_between_the_integrated_point_and_the_reference(self, rng):
        """An anchor interpolates; it must never extrapolate past either end."""
        for _ in range(200):
            a, v, r = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.1
            gate, k = rng.random(), rng.random()
            stepped = a + v * DT * gate
            out = advance_attractor(a, v, DT, gate, law="anchor", reference=r,
                                    anchor_gain=k, anchor_gated=False)
            span = np.linalg.norm(r - stepped)
            assert np.linalg.norm(out - stepped) <= span + 1e-12
            assert np.linalg.norm(out - r) <= span + 1e-12

    def test_gating_makes_the_anchor_inert_when_the_gate_is_shut(self):
        """The axis that is easy to leave implicit, and changes what the law means.

        Gated, a shut gate freezes the attractor entirely -- the anchor cannot
        drag it on while the arm catches up. Ungated, the anchor keeps pulling,
        which defeats the gate's whole purpose.
        """
        a, v, r = np.zeros(3), np.ones(3), np.array([1.0, 0.0, 0.0])
        gated = advance_attractor(a, v, DT, 0.0, law="anchor", reference=r,
                                  anchor_gain=0.5, anchor_gated=True)
        ungated = advance_attractor(a, v, DT, 0.0, law="anchor", reference=r,
                                    anchor_gain=0.5, anchor_gated=False)
        assert np.array_equal(gated, a)
        assert np.linalg.norm(ungated - r) < np.linalg.norm(a - r)

    def test_an_unknown_law_raises(self):
        with pytest.raises(ValueError, match="unknown attractor law"):
            advance_attractor(np.zeros(3), np.zeros(3), DT, 1.0, law="teleport")

    def test_a_law_needing_a_reference_refuses_without_one(self):
        """It must not silently fall back to integrating.

        That is the shape of the by-identity lookup which returned a confident
        zero for an unmeasured gripper and read as the arm missing its target
        for weeks (7.13, 7.27).
        """
        with pytest.raises(ValueError, match="needs the policy's `reference`"):
            advance_attractor(np.zeros(3), np.zeros(3), DT, 1.0, law="anchor")


class TestAnchorSchedule:
    def test_it_is_strong_at_rest_and_weak_at_speed(self):
        s = AnchorSchedule(dwell=0.5, transit=0.05, speed_scale=0.2)
        assert s.gain(0.0) == pytest.approx(0.5)
        assert s.gain(1.0) < 0.1
        assert s.gain(0.0) > s.gain(0.3) > s.gain(1.0)

    def test_it_serialises_for_the_manifest(self):
        """A callable would serialise as a memory address (7.26 rule 2)."""
        import json

        json.dumps(AnchorSchedule(dwell=0.5, transit=0.05).to_dict())

    def test_a_constant_gain_still_works(self):
        law = ExecutionLaw(dt=DT, speed_limit=0.6, max_label_speed=0.2, anchor_gain=0.3)
        assert law.gain_at(0.0) == 0.3 and law.gain_at(0.2) == 0.3


class TestComposition:
    def test_the_composed_step_equals_its_two_halves(self, rng):
        """The property that stops the bed and the robot diverging.

        The robot calls the halves, because the infeasibility fallback runs
        between them; the bed calls the composition. If those disagreed, the
        sweep would measure something that does not ship, and nothing would
        fail to say so.
        """
        law = ExecutionLaw(dt=DT, speed_limit=0.6, max_label_speed=0.2,
                           attractor_law="anchor", anchor_gain=0.3)
        for _ in range(100):
            pos, att = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1
            pred = _Prediction(rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.1)
            one = attractor_step(pos, att, pred, law)
            halves = advance_step(pos, att, gate_step(pos, att, pred, law), law, pred)
            assert np.array_equal(one.attractor, halves.attractor)
            assert one.gate == halves.gate and one.max_lag == halves.max_lag

    def test_explicit_defaults_equal_omitted_defaults(self, rng):
        """Guards the default path against every future edit."""
        a = ExecutionLaw(dt=DT, speed_limit=0.6, max_label_speed=0.2)
        b = ExecutionLaw(dt=DT, speed_limit=0.6, max_label_speed=0.2,
                         attractor_law="integrate", anchor_gain=0.0,
                         anchor_gated=True, query_at="attractor")
        for _ in range(100):
            pos, att = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1
            pred = _Prediction(rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.1)
            assert np.array_equal(
                attractor_step(pos, att, pred, a).attractor,
                attractor_step(pos, att, pred, b).attractor,
            )

    def test_the_clamp_invariant_holds_for_every_law_and_gain(self, rng):
        for law_name, gain in (("integrate", 0.0), ("anchor", 0.05), ("anchor", 0.5),
                               ("anchor", 1.0), ("reference", 0.0)):
            law = ExecutionLaw(dt=DT, speed_limit=0.6, max_label_speed=0.2,
                               attractor_law=law_name, anchor_gain=gain)
            for _ in range(80):
                pos, att = rng.normal(size=3) * 0.3, rng.normal(size=3) * 0.3
                pred = _Prediction(rng.normal(size=3) * 0.4, rng.normal(size=3) * 0.3)
                out = attractor_step(pos, att, pred, law)
                assert float(np.linalg.norm(out.attractor - pos)) <= out.max_lag + 1e-9

    def test_an_unknown_setting_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unknown attractor law"):
            ExecutionLaw(dt=DT, speed_limit=1.0, max_label_speed=0.2, attractor_law="nope")
        with pytest.raises(ValueError, match="unknown query site"):
            ExecutionLaw(dt=DT, speed_limit=1.0, max_label_speed=0.2, query_at="elsewhere")


class TestSurrogatePlant:
    def test_a_parked_attractor_is_reached_exactly(self):
        """The 2.7 property: the lag comes from the setpoint *moving*.

        If the bed missed this it would be modelling something other than an
        impedance controller, and every number from it would be suspect.
        """
        A = np.linalg.solve(D_FREE, K_FREE)
        x, a = np.zeros(3), np.array([0.1, 0.0, 0.0])
        for _ in range(400):
            x = x + A @ (a - x) * DT
        assert np.linalg.norm(x - a) < 1e-9

    def test_the_settled_lag_matches_the_closed_form(self):
        """Zero-order hold: ``L* = v dt / (1 - (1 - A dt/m)^m)``.

        At ``m = 1`` this is ``K^-1 D v`` -- exactly the ``expected`` term the
        gate subtracts -- which is why the bed defaults there. Larger ``m``
        approaches the real plant, whose lag is ~1.28x bigger at this dt.
        """
        policy = GPPolicy().fit(straight_line_labels())
        A = float(np.linalg.solve(D_FREE, K_FREE)[0, 0])
        speed = 0.168
        for m in (1, 4, 8):
            out = rollout_impedance(policy, dt=DT, lag_tolerance=None, substeps=m)
            predicted = speed * DT / (1 - (1 - A * DT / m) ** m)
            assert out.lag[-5] == pytest.approx(predicted, rel=0.02), m

    def test_substeps_one_reproduces_the_gates_own_model_of_the_lag(self):
        policy = GPPolicy().fit(straight_line_labels())
        out = rollout_impedance(policy, dt=DT, lag_tolerance=None, substeps=1)
        assert out.lag[-5] == pytest.approx(compliance_norm(K_FREE, D_FREE) * 0.168, rel=0.02)

    def test_it_refuses_to_run_an_unstable_integration(self):
        """Silently oscillating would make every number downstream meaningless."""
        policy = GPPolicy().fit(straight_line_labels())
        with pytest.raises(ValueError, match="unstable"):
            rollout_impedance(policy, dt=1.0)

    def test_every_law_completes_the_phase_on_a_straight_path(self):
        policy = GPPolicy().fit(straight_line_labels())
        for kw in ({}, dict(attractor_law="anchor", anchor_gain=0.2),
                   dict(attractor_law="reference"),
                   dict(attractor_law="anchor", anchor_gain=AnchorSchedule(dwell=0.5, transit=0.05)),
                   dict(query_at="measured", attractor_law="anchor", anchor_gain=0.3)):
            out = rollout_impedance(policy, dt=DT, **kw)
            assert out.metadata["terminated_on_phase"], kw
            assert not out.metadata["budget_exhausted"], kw

    def test_it_reports_no_success_flag(self):
        """7.27: a scalar proxy invented here would repeat a documented mistake."""
        out = rollout_impedance(GPPolicy().fit(straight_line_labels()), dt=DT)
        assert "success" not in out.metadata
        assert "placement_error" not in out.metadata

    def test_it_names_the_law_that_produced_it(self):
        out = rollout_impedance(GPPolicy().fit(straight_line_labels()), dt=DT,
                                attractor_law="anchor", anchor_gain=0.25)
        assert out.metadata["attractor_law"] == "anchor"
        assert out.metadata["anchor_gain"] == 0.25
        assert out.metadata["query_at"] == "attractor"

    def test_it_is_deterministic(self):
        """The paired comparison design is void without this."""
        policy = GPPolicy().fit(straight_line_labels())
        a = rollout_impedance(policy, dt=DT)
        b = rollout_impedance(policy, dt=DT)
        assert np.array_equal(a.attractors, b.attractors)
        assert np.array_equal(a.positions, b.positions)
