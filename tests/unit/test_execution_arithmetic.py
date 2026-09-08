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
