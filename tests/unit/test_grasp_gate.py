"""Holding at the grasp until the jaws have hold, rather than on a schedule.

The gripper schedule comes from the demonstration, and its carry begins **one
waypoint** after the close command. That is enough headroom for the Panda it was
recorded on and nowhere near enough for a hand with wider jaws: measured across
the registry, a Robotiq 2F-140 first touches a can **15 waypoints** after being
told to close, because its 125 mm jaws have some 30 mm of travel per finger
before they reach anything. The other four hands reach contact within one
waypoint. So the lift began six seconds before that hand had hold, and the run
was scored as never having grasped while the can was in fact carried 426 mm.

Two alternatives were measured and rejected, and the tests below pin why, since
both are the obvious things to reach for:

* **a dwell scaled per hand** is an open-loop step budget sized for one case,
  which is exactly the shape of ``SETTLE_STEPS = 60`` -- 0.12 s where the cereal
  needed 0.96 -- and that silently handed over objects still in flight (7.32).
  It is also not derivable here: on one hand the delay runs 0, 0, 10 and 15
  waypoints across four objects and is *not* monotonic in the object's width;
* **a closure-rate rule** cannot see contact at all. The jaws are
  position-commanded and ``closure`` reports where the fingers *are*, so it
  asymptotes toward the commanded value whether or not anything is between them.
  On that same can the rate had fallen from 0.343 to 0.012 per waypoint thirteen
  waypoints *before* contact, then rose to 0.148 *at* contact.

Contact is the signal that means what it says, so the gate uses it directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tpgpt.sim.kinematics as kinematics
from tpgpt.sim.replay import (
    GRASP_CONTACT_STEPS,
    GRASP_GATE_MAX_STEPS,
    replay_labels,
)
from tpgpt.transport.labels import PolicyLabels


@pytest.fixture(autouse=True)
def stub_ik(monkeypatch):
    """Stand in for inverse kinematics, which needs a real MuJoCo model.

    The gate's behaviour is *when it stops stepping*, which is decidable
    without physics. Driving a real scene here would make the test slow and
    would be testing the simulator.
    """
    monkeypatch.setattr(
        kinematics, "solve_ik",
        lambda env, position, rotation=None, **kw: kinematics.IKResult(
            True, np.zeros(7), 0.0, 0.0, 1
        ),
    )


class FakeArm:
    """The smallest thing ``replay_labels`` will drive.

    Deliberately not a MuJoCo scene: the gate's logic is *when* it stops
    stepping, and that is decidable without physics. A simulator here would
    make the test slow and would test the simulator.
    """

    def __init__(self, contact_after: int | None):
        #: control steps of closing before the jaws touch, or None for never
        self.contact_after = contact_after
        self.steps = 0
        self.closing_steps = 0
        self.action_dim = 8

    # --- the surface replay_labels uses ------------------------------------
    @property
    def robots(self):
        return [self]

    @property
    def composite_controller(self):
        return self

    @property
    def part_controllers(self):
        return {"right": self}

    @property
    def qpos_index(self):
        return np.arange(7)

    @property
    def eef_site_id(self):
        return 0

    @property
    def sim(self):
        return self

    @property
    def data(self):
        return self

    @property
    def qpos(self):
        return np.zeros(7)

    @property
    def site_xpos(self):
        return np.zeros((1, 3))

    @property
    def site_xmat(self):
        return np.eye(3).reshape(1, 9)

    def step(self, action):
        self.steps += 1
        if action[-1] > 0:
            self.closing_steps += 1

    def holds(self) -> bool:
        return (
            self.contact_after is not None
            and self.closing_steps >= self.contact_after
        )


def labels(n: int = 4, close_at: int = 1) -> PolicyLabels:
    grip = np.full(n, -1.0)
    grip[close_at:] = 1.0
    return PolicyLabels(
        positions=np.zeros((n, 3)),
        velocities=np.zeros((n, 3)),
        orientations=np.stack([np.eye(3)] * n),
        gripper=grip,
        time_belief=np.linspace(0, 1, n),
    )


def run(contact_after, settle_steps=8, **kw):
    env = FakeArm(contact_after)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = replay_labels(
            env, labels(), settle_steps=settle_steps,
            grasp_gate=lambda e: e.holds(), **kw,
        )
    return env, result, [w for w in caught if w.category is RuntimeWarning]


class TestTheGateWaits:
    def test_it_holds_until_the_jaws_touch(self):
        """The case the gate exists for: a hand slower than the schedule."""
        env, result, warned = run(contact_after=60)
        spent = result.metadata["grasp_gate_steps"]
        assert spent >= 60, f"gave up after {spent} steps with contact at 60"
        assert spent <= 60 + GRASP_CONTACT_STEPS, (
            f"kept closing for {spent} steps after contact at 60"
        )
        assert not warned

    def test_it_does_not_wait_on_a_hand_that_is_already_holding(self):
        """Four of the five registry hands reach contact within one waypoint."""
        env, result, warned = run(contact_after=1)
        assert result.metadata["grasp_gate_steps"] <= GRASP_CONTACT_STEPS + 1
        assert not warned

    def test_a_brush_does_not_count_as_a_grasp(self):
        """Contact must persist, or a knock reads as a grip.

        ``xarm/cereal`` is the measured case: something touches the box four
        waypoints before the jaws are told to close and shoves it 22.1 mm. A
        gate that fired on a single step of contact would report a grasp that
        does not exist.
        """
        class Flickers(FakeArm):
            def holds(self):
                # touches every other step, never continuously
                return self.closing_steps % 2 == 0

        env = Flickers(contact_after=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            replay_labels(env, labels(), settle_steps=8,
                          grasp_gate=lambda e: e.holds())
        assert any(w.category is RuntimeWarning for w in caught), (
            "intermittent contact should exhaust the cap, not satisfy the gate"
        )

    def test_a_hand_that_never_closes_is_named_rather_than_hanging(self):
        env, result, warned = run(contact_after=None)
        assert result.metadata["grasp_gate_steps"] == GRASP_GATE_MAX_STEPS
        assert len(warned) == 1
        assert "never took hold" in str(warned[0].message)

    def test_the_cap_is_generous_against_the_worst_measured_hand(self):
        """A Robotiq 2F-140 needs 15 waypoints, which is 120 control steps."""
        assert GRASP_GATE_MAX_STEPS >= 120 * 1.5


class TestNothingElseChanges:
    def test_without_a_gate_the_behaviour_is_exactly_as_before(self):
        """The default must be inert, so every existing caller is unaffected."""
        plain = FakeArm(contact_after=None)
        replay_labels(plain, labels(), settle_steps=8)
        assert plain.steps == 4 * 8
        assert replay_labels(
            FakeArm(None), labels(), settle_steps=8
        ).metadata["grasp_gate_steps"] is None

    def test_the_gate_replaces_the_settle_rather_than_adding_to_it(self):
        """Its steps count toward the pose's own settle, not on top of it.

        Otherwise every gated waypoint would cost the settle twice, which
        changes the trajectory's timing rather than only its grasp.
        """
        env, result, _ = run(contact_after=2, settle_steps=8)
        # one gated waypoint plus three ungated ones, and the gate's own steps
        # come out of the gated waypoint's budget
        assert env.steps <= 4 * 8 + GRASP_CONTACT_STEPS


class TestFingerGrouping:
    """Counting fingers, not contacts, and taking the count from the gripper.

    Contact is not a grip: a single finger on the side counts, and so does the
    palm resting on top. Two distinct fingers is the minimum that can pinch, on
    a hand with two fingers or five, so the gate counts fingers.

    Grouping geoms into fingers cannot be derived reliably. Three schemes were
    measured across the registry and each is wrong on at least one hand -- the
    body sub-tree splits a Robotiq's linkage into four chains and collapses a
    Yumi's into one, the driving actuator collapses the XArm's coupled pair and
    over-counts the three-finger hand's palm-spread joint, and the sign along
    the closing axis cannot see a Yumi's geoms at all. What resolves it is that
    **GraspGen-X's own config declares the count**, so the derivation becomes a
    choice rather than a guess.
    """

    def test_the_finger_count_comes_from_the_gripper_config(self):
        from tpgpt.grasp.grippers import declared_fingers, resolve_pair

        counts = {
            h: declared_fingers(resolve_pair(h).graspgen)
            for h in ("panda", "yumi", "xarm", "robotiq85", "robotiq140",
                      "robotiq3f", "rethink", "umi", "inspire")
        }
        assert counts["panda"] == 2 and counts["robotiq140"] == 2
        assert counts["robotiq3f"] == 3 and counts["inspire"] == 3

    def test_an_unknown_count_is_refused_rather_than_assumed(self):
        """Two is the commonest case and assuming it is how a registry rots.

        `contact_offset` once returned a *zero* for an unmeasured hand -- a
        plausible-looking number that read as the arm missing its target.
        """
        import json

        from tpgpt.grasp import grippers

        original = grippers.gripper_config_path

        class Fake:
            def read_text(self):
                return json.dumps({"type": "suction_cup"})

        grippers.gripper_config_path = lambda name: Fake()
        try:
            with pytest.raises(ValueError, match="does not name a finger count"):
                grippers.declared_fingers("whatever")
        finally:
            grippers.gripper_config_path = original

    def test_two_fingers_is_not_enough_on_a_three_finger_hand(self):
        """Opposition, not a count -- and a count would have been wrong.

        A parallel jaw's two fingers are necessarily opposed, so "two touching"
        is safe there. A Robotiq 3F carries **two fingers on one side and one on
        the other**, measured at -63.8 mm, -61.6 mm and +71.9 mm along its own
        closing axis, so two of its fingers touching can be the pair on one
        side -- two fingers pushing the object rather than pinching it.
        """
        from tpgpt.experiments.diagnose import is_pinched

        def scene(contacts):
            """A scene whose only contacts are the object against those geoms."""
            class Contact:
                def __init__(self, g):
                    self.geom1, self.geom2 = 0, g

            model = type("m", (), {"ngeom": 40, "geom_bodyid": [1] + [0] * 39})
            data = type("d", (), {"ncon": len(contacts),
                                  "contact": [Contact(g) for g in contacts]})
            sim = type("s", (), {"model": model, "data": data})
            return type("e", (), {"object_body_ids": {"thing": 1}, "sim": sim})()

        groups = [{10}, {11}, {12}]          # two on one side, one on the other
        sides = [-63.8, -61.6, +71.9]
        assert not is_pinched(scene([10, 11]), "thing", groups, sides), (
            "two fingers on the same side is a shove, not a pinch"
        )
        assert is_pinched(scene([10, 12]), "thing", groups, sides)
        assert is_pinched(scene([11, 12]), "thing", groups, sides)
        assert not is_pinched(scene([12]), "thing", groups, sides)

    def test_a_hand_with_no_measured_closing_axis_is_refused(self):
        """A guessed side would silently make a shove read as a pinch."""
        import tpgpt.grasp.grippers as grippers
        from tpgpt.experiments import diagnose

        keep = grippers.gripper_frame
        grippers.gripper_frame = lambda short: None
        try:
            with pytest.raises(ValueError, match="no measured closing axis"):
                diagnose.finger_sides(object(), "nonesuch", [{1}])
        finally:
            grippers.gripper_frame = keep

    def test_counting_fingers_and_not_geoms_is_the_point(self):
        """A Robotiq has five collision geoms per finger and a Yumi has one.

        So a threshold on the number of contacting geoms would mean "one finger"
        on one hand and "five fingers" on another, which is exactly the
        cross-hand incomparability that made the raw jaw reading useless (7.28).
        """
        from tpgpt.experiments.diagnose import fingers_touching

        class Scene:
            object_body_ids = {"thing": 1}

            class sim:
                class model:
                    ngeom = 12
                    geom_bodyid = [1, 1] + [0] * 10

                class data:
                    ncon = 2

                    class _c:
                        def __init__(self, a, b):
                            self.geom1, self.geom2 = a, b

                    contact = [_c(0, 5), _c(1, 6)]

        # five geoms on one finger, one on the other: two fingers, six geoms
        groups = [{5, 7, 8, 9, 10}, {6}]
        assert fingers_touching(Scene(), "thing", groups) == 2
        # both contacts on the *same* finger is one finger, not two
        assert fingers_touching(Scene(), "thing", [{5, 6}, {11}]) == 1


class FakeGripperModel:
    """robosuite's gripper interface, modelled faithfully enough to fail.

    Two things matter and both bit this project. ``format_action`` moves
    ``current_action`` by ``speed * np.sign(action)`` -- the sign only, so a
    commanded 0.25 and a commanded 1.0 are the same instruction. And the two
    finger elements carry **opposite** multipliers, which differ per hand, so
    the closing direction has to be discovered rather than assumed.
    """

    SPEED = 0.2

    def __init__(self):
        self.dof = 1
        self.current_action = np.zeros(2)

    def format_action(self, action):
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.SPEED * np.sign(action),
            -1.0, 1.0,
        )
        return self.current_action


class JawsThatIntegrate:
    """A hand whose fingers track ``current_action``, and a force that only
    rises once they are past the object's surface.

    ``touches_at`` is the closure fraction at which the fingers first reach the
    object; ``None`` means nothing is between them, so force never rises.
    """

    def __init__(self, touches_at=0.0, force_per_fraction=120.0):
        self.touches_at = touches_at
        self.force_per_fraction = force_per_fraction
        self.gripper = FakeGripperModel()
        self.history: list[float] = []
        self.action_dim = 8

    # --- the closure the fingers have actually reached ---------------------
    @property
    def fraction(self) -> float:
        a = np.asarray(self.gripper.current_action, dtype=float)
        direction = np.array([-1.0, 1.0])
        return float(np.clip((float(a @ direction) / len(direction) + 1.0) / 2.0, 0.0, 1.0))

    @property
    def robots(self):
        return [self]

    @property
    def composite_controller(self):
        return self

    @property
    def part_controllers(self):
        return {"right": self}

    @property
    def qpos_index(self):
        return np.arange(7)

    @property
    def eef_site_id(self):
        return 0

    @property
    def sim(self):
        return self

    @property
    def data(self):
        return self

    @property
    def qpos(self):
        return np.zeros(7)

    @property
    def site_xpos(self):
        return np.zeros((1, 3))

    @property
    def site_xmat(self):
        return np.eye(3).reshape(1, 9)

    def step(self, action):
        self.gripper.format_action(np.atleast_1d(action[-1]))
        self.history.append(self.fraction)

    def force(self) -> float:
        if self.touches_at is None:
            return 0.0
        return max(0.0, self.fraction - self.touches_at) * self.force_per_fraction


def run_force(touches_at=0.0, target=10.0, **kw):
    env = JawsThatIntegrate(touches_at)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = replay_labels(
            env, labels(), settle_steps=8,
            hold_when=lambda e: e.force() >= target, **kw,
        )
    return env, result, [w for w in caught if w.category is RuntimeWarning]


class TestCloseToForceThenHold:
    """Close until the grip is firm, then stay -- not until contact, and not
    to a command magnitude.

    **Two earlier versions of this fix were inert or wrong, and the tests did
    not catch either.** The first commanded a magnitude below 1, assuming a
    position target; robosuite integrates the sign and discards the size, so 19
    of 20 cells took a different command and reached an identical closure. The
    second closed until first *contact*, which fires before the close is even
    commanded because the fingers already straddle the object -- it froze the
    jaws at 0.00-0.05 closure against a baseline of 0.38-0.71.

    Both slipped through because the double shared the code's assumption. This
    one models the sign integrator *and* a force that only rises once the
    fingers are past the surface, so each wrong trigger fails a test here.
    """

    def test_it_stops_once_the_grip_is_firm(self):
        env, result, warned = run_force(touches_at=0.4, target=10.0)
        held = result.metadata["grasp_hold_fraction"]
        assert 0.4 < held < 1.0, f"held at {held}: not past contact, or fully shut"
        assert env.force() >= 10.0
        assert not warned

    def test_it_does_not_stop_at_first_contact(self):
        """The trigger that failed in the field: the fingers are already
        touching, so a contact test freezes a hand that has barely moved."""
        env, _, _ = run_force(touches_at=0.4, target=10.0)
        assert env.fraction > 0.4, (
            f"froze at first contact ({env.fraction}) instead of closing to force"
        )

    def test_a_firmer_target_closes_further(self):
        """Force is adaptive in the way a command magnitude is not."""
        soft = run_force(touches_at=0.4, target=4.0)[0].fraction
        firm = run_force(touches_at=0.4, target=20.0)[0].fraction
        assert firm > soft

    def test_it_differs_from_commanding_the_jaws_shut(self):
        """The assertion `flip_target` needed, and which the first fix passed
        while being inert (7.34)."""
        held = run_force(touches_at=0.4, target=10.0)[0]
        plain = JawsThatIntegrate(0.4)
        replay_labels(plain, labels(), settle_steps=8, hold_when=None)
        assert plain.fraction == pytest.approx(1.0)
        assert held.fraction < plain.fraction

    def test_nothing_between_the_jaws_warns_and_shuts(self):
        env, result, warned = run_force(touches_at=None)
        assert warned, "closed on nothing without saying so"
        assert result.metadata["grasp_hold_fraction"] == pytest.approx(1.0)

    def test_opening_resets_the_hold(self):
        env = JawsThatIntegrate(0.4)
        grip = np.array([-1.0, 1.0, -1.0, 1.0])
        two = PolicyLabels(
            positions=np.zeros((4, 3)), velocities=np.zeros((4, 3)),
            orientations=np.stack([np.eye(3)] * 4), gripper=grip,
            time_belief=np.linspace(0, 1, 4),
        )
        replay_labels(env, two, settle_steps=4,
                      hold_when=lambda e: e.force() >= 10.0)
        assert min(env.history) == pytest.approx(0.0)
