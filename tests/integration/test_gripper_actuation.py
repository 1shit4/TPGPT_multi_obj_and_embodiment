"""Driving one actuator is not driving a hand — measured, not doubled.

``tests/unit/test_gripper_action.py`` pins the *shape* of the gripper command.
It cannot pin the consequence, because a double built from the same assumption
as the code cannot test that assumption (ROBOTICS_NOTES.md section 7.28). This
module mounts the real hands and measures what their fingers actually do.

**What the defect was.** ``action[-1] = 1.0`` drives the last entry of the
action vector. robosuite lays that vector out as the arm's degrees of freedom
followed by the gripper's, so for a hand with ``dof > 1`` this commands one
actuator and leaves the rest at zero. Measured here, fully open to fully closed,
40 control steps each way, seeded:

==========  =====  =========================  =========================
hand        dof    last entry only            whole gripper block
==========  =====  =========================  =========================
inspire         6  ~10 mm of finger travel    **~57 mm**
umi             2  **one** finger moves       **both** fingers move
==========  =====  =========================  =========================

The Inspire hand's registry entry recorded 0.7 mm and the note read "it does not
actuate". The UMI's maximum travel is unchanged because its *left* finger
travels the same either way -- what the scalar command lost was the right
finger, which never moved at all. A one-jawed UMI still produces a plausible
closing axis and a plausible spread, which is why this failed silently.

Marked ``sim`` and ``slow``: each measurement builds a robosuite environment and
steps 80 control steps.
"""

import numpy as np
import pytest

from tpgpt.grasp.grippers import GRIPPER_PAIRS, gripper_action, resolve_pair

pytestmark = [pytest.mark.sim, pytest.mark.slow]

#: Geoms moving less than this when the hand closes are structure, not fingers.
#: Matches ``measure_frames.MIN_TRAVEL``.
MIN_TRAVEL = 2e-4

#: A geom that has moved this far is unambiguously a finger that travelled,
#: rather than a link nudged by its neighbour. Used to count *which* fingers
#: moved, which is the UMI's whole story.
CLEAR_TRAVEL = 1e-3

#: Control steps to hold each command for. Matches ``measure_frames``.
SETTLE_STEPS = 40


def _closing_travel(robosuite_name, *, whole_block):
    """Finger displacement from fully open to fully closed.

    Args:
        robosuite_name: Gripper class name, e.g. ``"InspireRightHand"``.
        whole_block: Drive every gripper degree of freedom (the repair) rather
            than only the last entry of the action vector (the defect).

    Returns:
        ``(max_travel_m, n_geoms_past_MIN_TRAVEL, n_geoms_past_CLEAR_TRAVEL)``.
    """
    import robosuite as suite

    # Seeded before ``make`` and ``reset``, never after. robosuite samples the
    # object's placement at reset, and an unseeded sample moves the measured
    # travel by up to 0.6 um between two otherwise identical runs.
    np.random.seed(0)

    env = suite.make(
        "Lift",
        robots="Panda",
        gripper_types=robosuite_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        sim = env.sim
        gripper = env.robots[0].gripper
        gripper = gripper["right"] if isinstance(gripper, dict) else gripper
        geom_ids = [
            i for i, name in enumerate(sim.model.geom_names)
            if name and "gripper0" in name
        ]
        site = sim.model.site_name2id(gripper.important_sites["grip_site"])

        def positions():
            return np.array([sim.data.geom_xpos[i].copy() for i in geom_ids])

        def command(value):
            if whole_block:
                return gripper_action(env, gripper, value)
            action = np.zeros(env.action_dim)
            action[-1] = value
            return action

        for _ in range(SETTLE_STEPS):
            env.step(command(-1.0))
        opened = positions()
        rotation = np.array(sim.data.site_xmat[site]).reshape(3, 3)

        for _ in range(SETTLE_STEPS):
            env.step(command(1.0))

        travel = np.linalg.norm((positions() - opened) @ rotation, axis=1)
        return (
            float(travel.max()),
            int((travel > MIN_TRAVEL).sum()),
            int((travel > CLEAR_TRAVEL).sum()),
        )
    finally:
        env.close()


def _multi_dof():
    """Registry hands with more than one gripper actuator.

    Derived, not listed. A hardcoded list went stale the moment five
    multi-actuator hands were registered, and the control below asserts
    ``dof == 1`` of everything not in it -- so a stale list fails the *control*
    rather than the case it was guarding, which is the least useful place for
    it to break.
    """
    from robosuite.models.grippers import GRIPPER_MAPPING

    return tuple(
        short for short, pair in GRIPPER_PAIRS.items()
        if GRIPPER_MAPPING[pair.robosuite]().dof > 1
    )


#: The cases the defect could reach at all.
MULTI_DOF = _multi_dof()


def test_the_inspire_hand_actuates():
    """It was excluded from every campaign on a measurement of 0.7 mm.

    Driven on all six of its actuators it travels tens of millimetres, so the
    exclusion was a property of the instrument and not of the hand. The margin
    asserted here is deliberately far below the ~47 mm actually observed: the
    claim is "it moves", not "it moves exactly this far".
    """
    scalar, _, _ = _closing_travel("InspireRightHand", whole_block=False)
    full, _, _ = _closing_travel("InspireRightHand", whole_block=True)

    assert full > 0.02, f"inspire travelled only {full * 1000:.2f} mm"
    assert full > 2 * scalar, (
        f"inspire: {full * 1000:.2f} mm driven fully against "
        f"{scalar * 1000:.2f} mm driving one actuator"
    )


def test_the_umi_closes_both_fingers_only_when_both_are_driven():
    """The UMI's defect is *which* fingers move, not how far the furthest goes.

    Its maximum travel is the same either way, because the finger the scalar
    command happens to drive travels the same distance regardless. What changes
    is that the other one moves at all. A gripper closing one jaw still reports
    a plausible closing axis and a plausible spread, so nothing downstream could
    have caught this.
    """
    _, _, scalar_fingers = _closing_travel("UMIGripper", whole_block=False)
    _, _, full_fingers = _closing_travel("UMIGripper", whole_block=True)

    assert full_fingers > scalar_fingers, (
        f"umi: {full_fingers} geoms travelled past {CLEAR_TRAVEL * 1000:.0f} mm "
        f"driven fully, against {scalar_fingers} driving one actuator"
    )


@pytest.mark.parametrize(
    "short", [s for s in GRIPPER_PAIRS if s not in MULTI_DOF]
)
def test_a_single_actuator_hand_is_commanded_identically(short):
    """The control, and the reason the repair is surgical.

    For a one-actuator hand the last entry *is* the whole gripper block, so the
    repaired command is the **same vector**. Asserted on the vector rather than
    on the resulting physics: two runs of the identical command still differ by
    a few tenths of a micrometre of solver noise, so a physics comparison could
    only ever assert a tolerance, whereas this asserts the actual claim.

    This is what makes an unchanged ``gripper_frames.json`` entry for these
    seven hands evidence rather than coincidence.
    """
    from robosuite.models.grippers import GRIPPER_MAPPING

    gripper = GRIPPER_MAPPING[resolve_pair(short).robosuite]()
    assert gripper.dof == 1, f"{short} has dof {gripper.dof}; it belongs in MULTI_DOF"

    class _Env:
        action_dim = 7 + gripper.dof

    old = np.zeros(_Env.action_dim)
    old[-1] = 1.0
    assert np.array_equal(gripper_action(_Env(), gripper, 1.0), old)
