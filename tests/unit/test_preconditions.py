"""The checks that refuse to start a campaign on a broken setup.

Every one of these corresponds to a bug that was found only *after* a 12 to 25
minute physics campaign had produced plausible-looking numbers. The tests below
reproduce each broken setup and assert the check catches it, so the harness
itself has regression coverage -- which is the thing that was missing when the
bugs were introduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpgpt.experiments.diagnose import Precondition, replay_preconditions, require

CALIBRATION = {
    "closing_in_site": [1.0, 0.0, 0.0],
    "finger_geoms": ["left_pad", "right_pad"],
    "spread_open": 0.100,
    "spread_closed": 0.060,
    "plus_one_closes": True,
}


class PandaGripper:
    important_sites = {"grip_site": "grip_site"}


class Robotiq140Gripper:
    important_sites = {"grip_site": "grip_site"}


class FakeOpt:
    timestep = 0.002


class FakeModel:
    geom_names = ("left_pad", "right_pad")
    opt = FakeOpt()

    def geom_name2id(self, name):
        return self.geom_names.index(name)

    def site_name2id(self, name):
        return 0


class FakeData:
    def __init__(self, time):
        self.time = time
        self.geom_xpos = np.array([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
        self.site_xmat = [np.eye(3).reshape(9)]


class FakeSim:
    def __init__(self, time):
        self.model = FakeModel()
        self.data = FakeData(time)


class FakeRobot:
    def __init__(self, gripper):
        self.gripper = gripper


class FakeEnv:
    #: Matches ``TabletopShelf``: settling runs until the objects stop, so the
    #: budget is what this scene *took*, not the 60-step floor. Comparing
    #: against the floor rejected all 40 cells of a campaign once settling
    #: became adaptive (7.32).
    SETTLE_STEPS = 60
    settle_steps_taken = 300

    def __init__(self, gripper_class=PandaGripper, time=0.0, position=(0.1, 0.2, 0.9)):
        self.sim = FakeSim(time)
        self.robots = [FakeRobot(gripper_class())]
        self._position = np.asarray(position, dtype=float)

    def object_position(self, name):
        return self._position.copy()


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(
        "tpgpt.grasp.grippers.gripper_frame", lambda short: dict(CALIBRATION)
    )


def named(checks, name):
    return next(c for c in checks if c.name == name)


class TestAllPass:
    def test_a_clean_setup_passes_every_check(self, calibrated):
        checks = replay_preconditions(FakeEnv(), "panda", "can")
        assert all(c.ok for c in checks), [c for c in checks if not c.ok]
        require(checks)  # must not raise

    def test_every_check_reports_what_it_measured_even_when_it_passes(self, calibrated):
        """A green check that says nothing cannot be told from one that did not
        run -- the 7.19 failure, where an instrument returning zeros was
        recorded as having disproved a hypothesis."""
        for check in replay_preconditions(FakeEnv(), "panda", "can"):
            assert check.detail.strip(), check.name


class TestGripperMounted:
    def test_the_wrong_hand_on_the_arm_is_caught(self, calibrated):
        """``build_scene`` hardcoded ``robots="Panda"`` and passed no
        ``gripper_types``, so a three-hand comparison ran a Panda three times.
        Nothing in the resulting numbers looked wrong."""
        checks = replay_preconditions(FakeEnv(PandaGripper), "robotiq140", "can")
        assert not named(checks, "gripper_mounted").ok
        assert "Robotiq140Gripper" in named(checks, "gripper_mounted").detail
        with pytest.raises(RuntimeError, match="gripper_mounted"):
            require(checks, "robotiq140/can")

    def test_the_right_hand_passes(self, calibrated):
        checks = replay_preconditions(FakeEnv(Robotiq140Gripper), "robotiq140", "can")
        assert named(checks, "gripper_mounted").ok


class TestSceneUnstepped:
    def test_a_reused_environment_is_caught(self, calibrated):
        """Reusing one scene across replays produced placement errors of 354,
        698 and 1170 mm purely from accumulated disturbance."""
        checks = replay_preconditions(FakeEnv(time=4.75), "panda", "can")
        assert not named(checks, "scene_unstepped").ok
        assert "4.7500" in named(checks, "scene_unstepped").detail

    def test_the_scenes_own_settling_is_allowed(self, calibrated):
        """60 settle steps at 2 ms is 0.120 s and is legitimate: the meshes are
        dropped and must come to rest before a cloud is captured. Comparing
        against zero instead rejected every real scene -- which is what the
        first version of this check did."""
        checks = replay_preconditions(FakeEnv(time=0.600), "panda", "can")
        assert named(checks, "scene_unstepped").ok

    def test_one_control_step_past_the_settle_budget_is_caught(self, calibrated):
        checks = replay_preconditions(FakeEnv(time=0.602), "panda", "can")
        assert not named(checks, "scene_unstepped").ok

    def test_the_budget_is_reported_alongside_the_reading(self, calibrated):
        detail = named(
            replay_preconditions(FakeEnv(time=0.600), "panda", "can"),
            "scene_unstepped",
        ).detail
        assert "0.6000" in detail and "300 settle steps" in detail


class TestObjectPlacement:
    def test_the_first_cell_records_a_reference_and_passes(self, calibrated):
        checks = replay_preconditions(FakeEnv(), "panda", "can", object_reference=None)
        assert named(checks, "object_placement").ok
        assert "reference" in named(checks, "object_placement").detail

    def test_a_moved_object_is_caught_and_the_distance_reported(self, calibrated):
        env = FakeEnv(position=(0.1, 0.2, 0.9))
        checks = replay_preconditions(
            env, "panda", "can", object_reference=(0.1, 0.2, 0.93)
        )
        check = named(checks, "object_placement")
        assert not check.ok
        assert "30.000 mm" in check.detail

    def test_an_identical_placement_passes(self, calibrated):
        checks = replay_preconditions(
            FakeEnv(position=(0.1, 0.2, 0.9)), "panda", "can",
            object_reference=(0.1, 0.2, 0.9),
        )
        assert named(checks, "object_placement").ok


class TestClosureCalibrated:
    def test_an_uncalibrated_hand_is_caught(self, monkeypatch):
        monkeypatch.setattr("tpgpt.grasp.grippers.gripper_frame", lambda s: None)
        checks = replay_preconditions(FakeEnv(), "panda", "can")
        assert not named(checks, "closure_calibrated").ok

    def test_a_hand_not_known_to_shut_on_plus_one_is_caught(self, monkeypatch):
        monkeypatch.setattr(
            "tpgpt.grasp.grippers.gripper_frame",
            lambda s: {**CALIBRATION, "plus_one_closes": False},
        )
        checks = replay_preconditions(FakeEnv(), "panda", "can")
        check = named(checks, "closure_calibrated")
        assert not check.ok
        assert "+1 shuts = False" in check.detail


class TestRequire:
    def test_it_lists_passed_checks_too(self, calibrated):
        checks = replay_preconditions(FakeEnv(time=1.0), "panda", "can")
        with pytest.raises(RuntimeError) as info:
            require(checks, "cell")
        message = str(info.value)
        assert "[ok ] gripper_mounted" in message
        assert "[FAIL] scene_unstepped" in message
        assert "cell" in message

    def test_an_empty_list_is_vacuously_fine(self):
        require([])

    def test_it_raises_on_any_single_failure(self):
        with pytest.raises(RuntimeError):
            require([Precondition("a", True, "x"), Precondition("b", False, "y")])
