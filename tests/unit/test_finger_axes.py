"""The shared frame geometry, tested without a simulator.

``measure_frames.finger_axes`` derives a hand's approach and closing directions
from where its geoms were and where they went. It was inline in
``measure_frame`` until ``tpgpt.grasp.describe`` needed the same axes about a
different origin; extracting it means the two cannot drift apart, and it means
the geometry can be tested on constructed inputs where the right answer is known
rather than only on a simulated hand where it is not.

The construction throughout is a hand pointing along ``+Z`` with two fingers at
``x = +-40 mm``, closing to ``+-10 mm``.
"""

import numpy as np
import pytest

from tpgpt.grasp.measure_frames import finger_axes


def _jaw(open_x=0.04, closed_x=0.01, z=0.1, rotation=None):
    """Two finger geoms and two stationary palm geoms."""
    opened = np.array([
        [+open_x, 0.0, z], [-open_x, 0.0, z],      # fingers
        [0.0, 0.0, 0.0], [0.0, 0.01, 0.0],         # palm, does not move
    ])
    closed = opened.copy()
    closed[0, 0], closed[1, 0] = +closed_x, -closed_x
    if rotation is not None:
        opened, closed = opened @ rotation.T, closed @ rotation.T
    return opened, closed


ORIGIN = np.zeros(3)
EYE = np.eye(3)


def test_the_approach_points_from_the_root_towards_the_fingers():
    opened, closed = _jaw()
    approach, _, _, _ = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert approach == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_the_approach_flips_when_the_fingers_are_behind_the_root():
    """The UMI's fingers lie along ``grip_site -Z``; the sign is measured."""
    opened, closed = _jaw(z=-0.1)
    approach, _, _, _ = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert approach == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_the_closing_axis_is_the_principal_direction_of_travel():
    opened, closed = _jaw()
    _, closing, _, _ = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert abs(closing[0]) == pytest.approx(1.0, abs=1e-9)
    assert closing[1:] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_the_closing_axis_is_orthogonal_to_the_approach():
    """Asserted because the approach component is explicitly projected out.

    A closing direction with an approach component in it is not a closing
    direction: it describes fingers moving along the object rather than onto it.
    """
    opened, closed = _jaw()
    # Fingers that also creep forward as they close, as a revolute linkage does.
    closed[0, 2] = closed[1, 2] = 0.13
    approach, closing, _, _ = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert float(approach @ closing) == pytest.approx(0.0, abs=1e-9)


def test_the_basis_takes_grasp_coordinates_into_the_local_frame():
    """``basis`` columns are closing, the third axis, and approach, in order."""
    opened, closed = _jaw()
    approach, closing, basis, _ = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert basis[:, 0] == pytest.approx(closing, abs=1e-9)
    assert basis[:, 2] == pytest.approx(approach, abs=1e-9)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-9)
    assert basis.T @ basis == pytest.approx(np.eye(3), abs=1e-9)


def test_stationary_geoms_are_excluded():
    opened, closed = _jaw()
    _, _, _, moving = finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    assert moving.tolist() == [True, True, False, False]


def test_a_hand_that_did_not_actuate_is_refused():
    """Refusing beats returning an axis derived from solver noise."""
    opened, closed = _jaw()
    with pytest.raises(RuntimeError, match="did not actuate"):
        finger_axes(opened, opened.copy(), ORIGIN, EYE, ORIGIN)
    # One finger moving is also not enough to fit an axis to.
    half = opened.copy()
    half[0, 0] = 0.01
    with pytest.raises(RuntimeError, match="did not actuate"):
        finger_axes(opened, half, ORIGIN, EYE, ORIGIN)


def test_the_answer_does_not_depend_on_how_the_hand_is_oriented_in_the_world():
    """The axes are the hand's own, so rotating the whole hand changes nothing.

    This is the property the frame contract rests on: ``alignment`` is a
    constant per hand, not a function of where the arm happens to be holding it.
    """
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    plain = finger_axes(*_jaw(), ORIGIN, EYE, ORIGIN)
    turned = finger_axes(*_jaw(rotation=world), ORIGIN, world, ORIGIN)

    assert turned[0] == pytest.approx(plain[0], abs=1e-9)
    assert abs(float(turned[1] @ plain[1])) == pytest.approx(1.0, abs=1e-9)


def test_the_origin_shifts_the_frame_but_not_the_axes():
    """``measure_frame`` measures about ``grip_site``; ``describe`` about the
    root body. Same hand, same axes, different origin."""
    opened, closed = _jaw()
    root = np.zeros(3)
    at_root = finger_axes(opened, closed, root, EYE, root)
    at_site = finger_axes(opened, closed, np.array([0.0, 0.0, 0.09]), EYE, root)
    assert at_site[0] == pytest.approx(at_root[0], abs=1e-9)
    assert at_site[1] == pytest.approx(at_root[1], abs=1e-9)


def test_the_travel_threshold_is_honoured():
    opened, closed = _jaw(open_x=0.04, closed_x=0.0399)  # 0.1 mm of travel
    with pytest.raises(RuntimeError, match="did not actuate"):
        finger_axes(opened, closed, ORIGIN, EYE, ORIGIN)
    approach, _, _, moving = finger_axes(
        opened, closed, ORIGIN, EYE, ORIGIN, min_travel=1e-5
    )
    assert int(moving.sum()) == 2
