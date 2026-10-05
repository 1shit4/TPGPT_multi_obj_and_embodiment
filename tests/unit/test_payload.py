"""Weighing the flange's payload: the manufacturer's data placed by the mount, and the fit
of a payload to the robot's external joint torques at rest (plan M5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tpgpt.hardware import payload as pl

CONFIG = Path(__file__).resolve().parents[2] / "ycb_pointcloud_pipeline/config"


@pytest.fixture(scope="module")
def hand():
    try:
        return pl._hand_from_config(CONFIG / "inspire_hand.yaml")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_the_cad_payload_is_the_hand_and_its_mount(hand):
    cad = pl.inspire_cad_payload(hand, mount_mass=0.035)
    assert cad.mass == pytest.approx(0.5332 + 0.035)
    # The hand's centre of mass is 81 mm along it from the ring's face; the mount puts the
    # face 12.45 mm out, and the light mount pulls the total in a little.
    assert 0.080 < cad.com[2] < 0.095
    assert np.hypot(cad.com[0], cad.com[1]) < 0.012
    I = np.asarray(cad.inertia)
    assert np.allclose(I, I.T) and np.all(np.linalg.eigvalsh(I) > 0)


def _synthetic(true: pl.Payload, modelled: pl.Payload, bias, noise, seed=0):
    from tpgpt.hardware.fr3 import HOME_Q, default_model

    fr3 = default_model()
    rng = np.random.default_rng(seed)
    dm = true.mass - modelled.mass
    ds = true.first_moment() - modelled.first_moment()
    out = []
    for off in pl.POSES:
        q = np.asarray(HOME_Q) + np.asarray(off, dtype=float)
        tau = pl._regressor(fr3, q) @ np.r_[dm, ds] + bias + rng.normal(0.0, noise, 7)
        out.append((q, tau))
    return out


def test_the_fit_recovers_a_payload_through_joint_offsets(hand):
    true = pl.inspire_cad_payload(hand)
    bias = np.array([0.1, -0.3, 0.2, 0.25, -0.1, 0.2, 0.05])     # N m, sensor offsets
    fit = pl.fit_payload(_synthetic(true, pl.FRANKA_HAND_DESK, bias, noise=0.02))
    # 0.02 N m of noise per joint: the mass to +-4.4 g, the centre of mass to ~2 mm (1 sigma).
    assert fit.payload.mass == pytest.approx(true.mass, abs=0.015)
    assert np.allclose(fit.payload.com, true.com, atol=0.006)
    assert np.allclose(fit.bias, bias, atol=0.06)
    assert fit.flipped_mass == pytest.approx(2 * pl.FRANKA_HAND_DESK.mass - fit.payload.mass)
    assert not fit.notes and max(fit.residual_rms) < 0.05


def test_with_the_right_payload_in_desk_only_the_offsets_are_left(hand):
    true = pl.inspire_cad_payload(hand)
    bias = np.array([0.1, -0.3, 0.2, 0.25, -0.1, 0.2, 0.05])
    fit = pl.fit_payload(_synthetic(true, true, bias, noise=0.02), modelled=true)
    assert fit.payload.mass == pytest.approx(true.mass, abs=0.015)


def test_static_friction_signed_by_the_approach_is_separated_from_the_payload(hand):
    """The first weighing (2026-10-05): every joint stopped short on the side it came from,
    and friction held up to 0.6 N m. With each sample's approach the fit separates it."""
    true = pl.inspire_cad_payload(hand)
    bias = np.array([0.1, -0.3, 0.2, 0.25, -0.1, 0.2, 0.05])
    friction = np.array([0.03, 0.6, 0.12, 0.23, 0.42, 0.27, 0.18])
    rng = np.random.default_rng(1)
    samples = []
    for q, tau in _synthetic(true, pl.FRANKA_HAND_DESK, bias, noise=0.02):
        approach = rng.choice([-1.0, 1.0], 7)
        samples.append((q, tau + friction * approach, approach))
    fit = pl.fit_payload(samples)
    assert fit.payload.mass == pytest.approx(true.mass, abs=0.02)
    assert np.allclose(fit.friction, friction, atol=0.03)
    # Without the approach the friction is unmodelled: the residuals say so.
    blind = pl.fit_payload([(q, tau) for q, tau, _ in samples])
    assert max(blind.residual_rms) > 0.15 and any("residuals" in n for n in blind.notes)


def test_the_approach_is_the_side_a_joint_stopped_on():
    a = pl.approach_of([0.0, 0.1, 0.2], [0.07, 0.03, 0.2])
    assert a.tolist() == [1.0, -1.0, 0.0]


def test_every_weighing_pose_clears_the_table_and_the_shelf(hand):
    shelf = yaml.safe_load((CONFIG / "shelf.yaml").read_text())
    rows = pl.pose_clearances(hand, shelf_config=shelf)
    assert len(rows) == len(pl.POSES)
    assert min(r["above_table"] for r in rows) > 0.15
    assert min(r["shelf_clearance"] for r in rows) > 0.08
    assert all(max(abs(v) for v in r["offsets"]) <= 0.8 for r in rows)


def test_the_vendor_urdf_gives_the_stored_inertial():
    urdf = Path("~/inspire_mount (2)/2-RH56DFQ RH56DFX RH56BFX- 202506/3-URDF/No Wrist").expanduser()
    found = list(urdf.rglob("urdf_right_2025_4_18.urdf")) if urdf.exists() else []
    if not found:
        pytest.skip("the manufacturer's URDF is not unpacked here")
    h = pl.vendor_hand_inertial(found[0])
    assert h["mass"] == pytest.approx(pl.INSPIRE_HAND_INERTIAL["mass"], abs=1e-4)
    assert np.allclose(h["com_H"], pl.INSPIRE_HAND_INERTIAL["com_H"], atol=1e-4)
    assert np.allclose(h["inertia_H"], pl.INSPIRE_HAND_INERTIAL["inertia_H"], rtol=1e-3, atol=1e-7)
