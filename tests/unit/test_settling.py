import pytest

from tpgpt.experiments.settling import settled_verdict
from tpgpt.experiments.audit_artifacts import paired_outcomes


def samples():
    return [dict(position=[.18, 0., 1.], in_slot=True, supported=True,
                 robot_contact=False) for _ in range(20)]


@pytest.mark.parametrize("field,value", [("in_slot", False), ("supported", False),
                                         ("robot_contact", True)])
def test_settling_requires_release_support_and_original_score(field, value):
    rows = samples()
    assert settled_verdict(rows, 20)["passed"]
    rows[10][field] = value
    assert not settled_verdict(rows, 20)["passed"]


def test_settling_rejects_motion_even_when_every_position_is_in_slot():
    rows = samples()
    rows[-1]["position"][0] += .004
    assert not settled_verdict(rows, 20)["passed"]
    assert not settled_verdict(rows[:3], 20)["passed"]


def row(pose):
    return dict(cell=dict(gripper="panda", obj="brick", pick_pose=pose),
                outcome="success", metrics={"candidate_cache": {"candidate_sha256": "same"}})


def test_pairing_does_not_conflate_explicit_placements():
    a, b = row([0., 0., 0.]), row([.1, 0., 0.])
    assert paired_outcomes([a], [b])["pairs"] == []
    assert len(paired_outcomes([a, b], [a, b])["pairs"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        paired_outcomes([a], [a, a])
