"""A jaw reading that means the same thing on every hand.

The instrument this replaces, ``_jaw_opening``, is the sum of the gripper's
joint positions. Measured across the nine registered hands it has the wrong
sign on two, no signal on two more, and units that differ by three orders of
magnitude -- and reading it as a width misattributed a whole multi-gripper
comparison (ROBOTICS_NOTES.md 7.28). So the replacement is tested for exactly
the properties that one lacked: one sign, one scale, and a refusal rather than
a plausible default when it has not been calibrated.

The environment here is a stand-in, not robosuite: the quantity under test is
arithmetic on geom positions, and a real simulator would make the expected
answer approximate instead of exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.experiments.diagnose import jaw_closure_probe

#: Calibration for a hand whose fingers travel 40 mm along its own local +X.
CALIBRATION = {
    "closing_in_site": [1.0, 0.0, 0.0],
    "finger_geoms": ["left_pad", "right_pad"],
    "spread_open": 0.100,
    "spread_closed": 0.060,
    "plus_one_closes": True,
}


class FakeModel:
    def __init__(self, names):
        self.geom_names = tuple(names)

    def geom_name2id(self, name):
        return self.geom_names.index(name)

    def site_name2id(self, name):
        return 0


class FakeData:
    def __init__(self, positions, rotation):
        self.geom_xpos = np.asarray(positions, dtype=float)
        self.site_xmat = [np.asarray(rotation, dtype=float).reshape(9)]


class FakeSim:
    def __init__(self, names, positions, rotation):
        self.model = FakeModel(names)
        self.data = FakeData(positions, rotation)


class FakeGripper:
    important_sites = {"grip_site": "grip_site"}


class FakeRobot:
    gripper = FakeGripper()


class FakeEnv:
    """Two pads separated along world X, plus one geom that is not a finger."""

    def __init__(self, separation, rotation=np.eye(3), names=None):
        names = names or ["left_pad", "right_pad", "knuckle"]
        positions = [
            [-separation / 2, 0.0, 0.0],
            [+separation / 2, 0.0, 0.0],
            [0.0, 0.0, 0.5],          # far away: must be ignored
        ]
        self.sim = FakeSim(names, positions, rotation)
        self.robots = [FakeRobot()]

    def set_separation(self, separation):
        self.sim.data.geom_xpos[0, 0] = -separation / 2
        self.sim.data.geom_xpos[1, 0] = +separation / 2


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(
        "tpgpt.grasp.grippers.gripper_frame", lambda short: dict(CALIBRATION)
    )


class TestScale:
    def test_fully_open_reads_zero(self, calibrated):
        assert jaw_closure_probe(FakeEnv(0.100), "any")() == pytest.approx(0.0)

    def test_fully_closed_reads_one(self, calibrated):
        assert jaw_closure_probe(FakeEnv(0.060), "any")() == pytest.approx(1.0)

    def test_halfway_reads_a_half(self, calibrated):
        assert jaw_closure_probe(FakeEnv(0.080), "any")() == pytest.approx(0.5)

    def test_the_same_fraction_means_the_same_thing_on_a_hand_with_other_numbers(
        self, monkeypatch
    ):
        """The whole point: one threshold, every hand.

        A second hand whose spread runs 0.300 to 0.220 must report 0.5 at its
        own midpoint, not at the first hand's. ``sum |qpos|`` could not do this
        -- 'shut' was 0.001 on a Panda and 4.9 on an XArm.
        """
        monkeypatch.setattr(
            "tpgpt.grasp.grippers.gripper_frame",
            lambda short: {**CALIBRATION, "spread_open": 0.300,
                           "spread_closed": 0.220},
        )
        assert jaw_closure_probe(FakeEnv(0.260), "other")() == pytest.approx(0.5)


class TestSign:
    def test_closing_always_raises_the_reading(self, calibrated):
        env = FakeEnv(0.100)
        probe = jaw_closure_probe(env, "any")
        readings = []
        for separation in (0.100, 0.090, 0.080, 0.070, 0.060):
            env.set_separation(separation)
            readings.append(probe())
        assert np.all(np.diff(readings) > 0), readings


class TestSqueezeIsNotHidden:
    def test_pressed_past_the_free_air_close_reads_above_one(self, calibrated):
        """Squeezing an object is the difference between holding it and
        shutting on nothing, so it must not be clipped away."""
        assert jaw_closure_probe(FakeEnv(0.050), "any")() > 1.0

    def test_forced_wider_than_open_reads_below_zero(self, calibrated):
        assert jaw_closure_probe(FakeEnv(0.120), "any")() < 0.0


class TestRefusals:
    """'Not measurable' must never come back as a plausible number.

    A hand with no calibration reading ``0.0`` would say 'wide open' forever,
    and the run would look like a grasp that never closed. This is the lesson
    of ``contact_offset``, which used to return a zero vector for an unmeasured
    hand (7.13).
    """

    def test_an_uncalibrated_hand_is_nan(self, monkeypatch):
        monkeypatch.setattr(
            "tpgpt.grasp.grippers.gripper_frame", lambda short: None
        )
        assert np.isnan(jaw_closure_probe(FakeEnv(0.08), "unknown")())

    def test_a_calibration_with_no_travel_is_nan(self, monkeypatch):
        monkeypatch.setattr(
            "tpgpt.grasp.grippers.gripper_frame",
            lambda short: {**CALIBRATION, "spread_closed": 0.100},
        )
        assert np.isnan(jaw_closure_probe(FakeEnv(0.08), "stuck")())

    def test_missing_finger_geoms_in_the_model_is_nan(self, monkeypatch):
        """The scene can be built without the geoms the calibration names."""
        monkeypatch.setattr(
            "tpgpt.grasp.grippers.gripper_frame", lambda short: dict(CALIBRATION)
        )
        env = FakeEnv(0.08, names=["something_else", "and_another", "knuckle"])
        assert np.isnan(jaw_closure_probe(env, "any")())


class TestWristOrientation:
    def test_the_reading_holds_with_the_wrist_rotated(self, calibrated):
        """Calibration is taken with the arm stationary; runs are not.

        The closing axis is stored in ``grip_site`` coordinates and projected
        through the live site rotation, so a rotated wrist must give the same
        closure for the same finger separation. Reading along a fixed world
        axis instead would make a 90-degree wrist roll report the jaws shut.
        """
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        env = FakeEnv(0.080, rotation=yaw)
        # pads now separated along world Y, which is the site's local +X
        env.sim.data.geom_xpos[0] = [0.0, -0.040, 0.0]
        env.sim.data.geom_xpos[1] = [0.0, +0.040, 0.0]
        assert jaw_closure_probe(env, "any")() == pytest.approx(0.5)
