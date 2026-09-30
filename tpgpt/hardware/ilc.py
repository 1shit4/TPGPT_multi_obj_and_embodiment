"""Learning the demonstration's attractor from replays: iterative learning control. No ROS.

Step 4 of the execution-gap plan (``docs/hardware/FAITHFULNESS_AUDIT.md`` B1). TPGPT's
position labels are the **commanded attractor** -- the point the impedance spring pulls
toward -- not the measured pose (Sec. V: the policy learns "the desired attractor
position, orientation, stiffness, damping"; ``docs/sim/ROBOTICS_NOTES.md`` 2.7). The
simulation's scripted teacher commands an attractor and records it, and its arm lags that
attractor; a target execution lags its transported attractor the same way, so the two lags
cancel. A demonstration taught by hand has no attractor, only the path the hand took.

So the taught path is replayed on the robot, under the stiffness the labels carry and at
the speed the executor will run, and the attractor is learned that makes the arm retrace
it. After pass ``k``, with the error ``e_k = reference - measured_k`` on the label grid,

    a_{k+1}(i) = a_k(i) + L * Q[ e_k(i + n_lead) ]

the standard P-type update with a phase lead and a zero-phase low-pass ``Q`` (e.g. Bristow,
Tharayil and Alleyne, IEEE Control Systems Magazine 2006):

* **L** (``gain``): the share of the last pass's error added to the next attractor. At 1.0
  the next attractor is moved by the whole error, which removes a repeatable error in one
  pass if the lead is right; at 0.5 an error halves each pass (25 -> 12.5 -> 6 -> 3 mm).
  The default is 1.0 because what the passes remove is nearly all repeatable (the lag and
  the arm's resting offset) and the noise it would otherwise learn is small and filtered
  by ``Q``. On the simulated arm of ``tests/unit/test_ilc.py`` (the measured 0.27 s time
  constant, 0.3 mm of noise), 27.6 mm RMS falls to 9.1 and then 0.9 mm in two updates at
  1.0; at 0.5 it takes five updates to reach 1.1 mm. If a pass comes out worse than the
  one before, the next update says so (``replay.update``); run it at 0.5.
* **n_lead** (``lead_s``): the arm's error at label ``i`` was caused by the attractor a little
  earlier, so the correction for ``i`` is read that much later. 0.25 s, the arm's own time
  constant along the motion as measured on 2026-09-29 (229-309 ms at 350 N/m, audit B2
  step 3) -- about twice CRISP's nominal ``2/sqrt(K)``.
* **Q** (``cutoff_hz``): pass-to-pass jitter is not learned.
* **Contact windows** (``frozen``): where the hand touches something -- the jaws closing,
  the object set down on a board -- the error is contact force, not lag, and learning it
  would press the attractor into the object or the board. There the update is bridged: a
  straight line between the updates on either side, so the attractor stays continuous and
  the lag correction carries through.
* **Caps** (``max_update``, ``max_correction``): a pass moves no label more than 30 mm, and
  no label ends up more than 60 mm from the taught path.

**Orientation is learned the same way** (:func:`ilc_rotation_update`). The hand's
orientation is held by a rotational spring as its position is held by a translational one,
and it errs the same two ways: a resting tilt (1.5 deg measured at 30 N m/rad on
2026-09-29, the same holding and moving: the wrist's friction and load against a soft
spring) and a lag behind a turning target. The error is the rotation that takes the
measured orientation to the taught one, as a rotation vector in the base frame, and the
attractor's orientation is turned by ``L`` times it, low-passed, led and bridged exactly
as the position. Its lead (``rotation_lead_s``, 0.35 s) is CRISP's nominal ``2/sqrt(k)``
over ``k`` at 30 N m/rad and is **not measured**: the translational one came out twice its
nominal. The taught hand turns slowly (5.6 deg/s at most at speed_scale 0.35), so a wrong
lead costs little; the resting tilt, which is most of the error, needs none. Caps: 3 deg
per pass, 10 deg from the taught orientation.

Everything is on the label grid; :func:`label_positions` and :func:`label_rotations` put
a replay's measured poses on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IlcSettings:
    gain: float = 1.0
    lead_s: float = 0.25
    cutoff_hz: float = 1.5
    max_update: float = 0.03
    max_correction: float = 0.06
    rotation_lead_s: float = 0.35
    max_rotation_update_deg: float = 3.0
    max_rotation_correction_deg: float = 10.0


def _crossings(sample_label, n_labels: int):
    """For each label: the two samples that straddle the first time it was reached, and
    where between them it falls (``(j0, j1, w)``)."""
    s = np.asarray(sample_label, dtype=float)
    j0, j1, w = np.zeros(n_labels, int), np.zeros(n_labels, int), np.zeros(n_labels)
    for k in range(n_labels):
        j = int(np.searchsorted(s, k, side="left"))
        if j <= 0:
            continue
        if j >= len(s):
            j0[k] = j1[k] = len(s) - 1
            continue
        j0[k], j1[k] = j - 1, j
        w[k] = np.clip((k - s[j - 1]) / max(s[j] - s[j - 1], 1e-12), 0.0, 1.0)
    return j0, j1, w


def label_positions(sample_label, sample_positions, n_labels: int) -> np.ndarray:
    """Measured position at the moment the replay reached each label.

    ``sample_label`` is the (fractional, non-decreasing) label index being commanded
    when each pose was measured; while the replay pauses (the gripper acting) it stays
    constant. The position for label ``k`` is interpolated between the two samples that
    straddle the first time the index reached ``k`` -- the arm as it was when label
    ``k`` became its target, so a pause after a label does not count.
    """
    x = np.asarray(sample_positions, dtype=float).reshape(-1, 3)
    j0, j1, w = _crossings(sample_label, n_labels)
    return x[j0] + w[:, None] * (x[j1] - x[j0])


def label_rotations(sample_label, sample_rotations, n_labels: int) -> np.ndarray:
    """Measured orientation when the replay reached each label (as :func:`label_positions`)."""
    from scipy.spatial.transform import Rotation

    R = np.asarray(sample_rotations, dtype=float).reshape(-1, 3, 3)
    j0, j1, w = _crossings(sample_label, n_labels)
    step = Rotation.from_matrix(np.transpose(R[j0], (0, 2, 1)) @ R[j1]).as_rotvec()
    return R[j0] @ Rotation.from_rotvec(step * w[:, None]).as_matrix()


def rotation_errors(reference, measured) -> np.ndarray:
    """``(N, 3)``: the rotation taking each measured orientation to the taught one, as a
    rotation vector in the base frame (``R_ref = exp(e) R_meas``)."""
    from scipy.spatial.transform import Rotation

    r = np.asarray(reference, dtype=float).reshape(-1, 3, 3)
    m = np.asarray(measured, dtype=float).reshape(-1, 3, 3)
    return Rotation.from_matrix(r @ np.transpose(m, (0, 2, 1))).as_rotvec()


def frozen_mask(n_labels: int, windows) -> np.ndarray:
    """Boolean mask of the labels inside any ``(first, last)`` window (inclusive)."""
    mask = np.zeros(n_labels, dtype=bool)
    for first, last in windows or ():
        mask[max(int(first), 0):min(int(last), n_labels - 1) + 1] = True
    return mask


def _bridge(update: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Replace the update inside each frozen run by a straight line between its neighbours."""
    out = update.copy()
    n = len(update)
    i = 0
    while i < n:
        if not frozen[i]:
            i += 1
            continue
        j = i
        while j < n and frozen[j]:
            j += 1
        left = out[i - 1] if i > 0 else (update[j] if j < n else np.zeros(3))
        right = update[j] if j < n else left
        for t, k in enumerate(range(i, j), start=1):
            out[k] = left + (right - left) * t / (j - i + 1)
        i = j
    return out


def _lowpass(x: np.ndarray, sample_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase second-order Butterworth low-pass along axis 0 (a no-op above Nyquist)."""
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sample_hz
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist or len(x) < 10:
        return x.copy()
    b, a = butter(2, cutoff_hz / nyquist)
    return filtfilt(b, a, x, axis=0, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def _learned(error, label_period, frozen, gain, lead_s, cutoff_hz, max_step):
    """``L * Q[e(i + lead)]``, bridged over contact and capped per label: ``(update, lead,
    how many labels the cap shortened)``.

    The error measured in contact is replaced by the bridge *before* the lead and the
    filter too, not only the update after them: otherwise the lead reads it into the
    labels just before a window and the filter smears it into both sides.
    """
    lead = int(round(lead_s / label_period))
    error = _bridge(error, frozen)
    led = np.concatenate([error[lead:], np.repeat(error[-1:], lead, axis=0)]) if lead else error
    update = _bridge(gain * _lowpass(led, 1.0 / label_period, cutoff_hz), frozen)
    size = np.linalg.norm(update, axis=1)
    capped = int((size > max_step).sum())
    update *= np.minimum(1.0, max_step / np.maximum(size, 1e-12))[:, None]
    return update, lead, capped


def ilc_update(attractor, measured, reference, label_period: float, frozen=None,
               settings: IlcSettings = IlcSettings()) -> tuple[np.ndarray, dict]:
    """The next pass's attractor position, and a report of this pass.

    Args:
        attractor: ``(N, 3)`` what this pass commanded, one row per label.
        measured: ``(N, 3)`` where the arm was, per label (:func:`label_positions`).
        reference: ``(N, 3)`` the taught path the arm should retrace.
        label_period: execution time per label, s (``dt / speed_scale``).
        frozen: ``(N,)`` bool, labels in contact (:func:`frozen_mask`).
    """
    a = np.asarray(attractor, dtype=float).reshape(-1, 3)
    m = np.asarray(measured, dtype=float).reshape(-1, 3)
    r = np.asarray(reference, dtype=float).reshape(-1, 3)
    n = len(a)
    frozen = np.zeros(n, dtype=bool) if frozen is None else np.asarray(frozen, dtype=bool)
    error = r - m
    update, lead, capped_update = _learned(error, label_period, frozen, settings.gain,
                                           settings.lead_s, settings.cutoff_hz,
                                           settings.max_update)
    new = a + update
    offset = new - r
    reach = np.linalg.norm(offset, axis=1)
    capped_total = int((reach > settings.max_correction).sum())
    new = r + offset * np.minimum(1.0, settings.max_correction / np.maximum(reach, 1e-12))[:, None]
    free = ~frozen
    norms = np.linalg.norm(error, axis=1)
    report = {
        "n_labels": n, "lead_labels": lead, "frozen_labels": int(frozen.sum()),
        "error_rms_mm": float(np.sqrt(np.mean(norms[free] ** 2)) * 1000) if free.any() else 0.0,
        "error_max_mm": float(norms[free].max() * 1000) if free.any() else 0.0,
        "update_max_mm": float(np.linalg.norm(new - a, axis=1).max() * 1000),
        "capped_update": capped_update, "capped_total": capped_total,
    }
    return new, report


def ilc_rotation_update(attractor, measured, reference, label_period: float, frozen=None,
                        settings: IlcSettings = IlcSettings()) -> tuple[np.ndarray, dict]:
    """The next pass's attractor orientation, and a report: :func:`ilc_update` for rotations.

    All three are ``(N, 3, 3)`` in one convention (the plan's grasp convention). The
    update is ``R_att <- exp(L Q[e(i + lead)]) R_att`` with ``e`` from
    :func:`rotation_errors`, then held within ``max_rotation_correction_deg`` of the taught
    orientation.
    """
    from scipy.spatial.transform import Rotation

    A = np.asarray(attractor, dtype=float).reshape(-1, 3, 3)
    R = np.asarray(reference, dtype=float).reshape(-1, 3, 3)
    n = len(A)
    frozen = np.zeros(n, dtype=bool) if frozen is None else np.asarray(frozen, dtype=bool)
    error = rotation_errors(R, measured)
    update, lead, capped_update = _learned(error, label_period, frozen, settings.gain,
                                           settings.rotation_lead_s, settings.cutoff_hz,
                                           np.radians(settings.max_rotation_update_deg))
    new = Rotation.from_rotvec(update).as_matrix() @ A
    offset = Rotation.from_matrix(new @ np.transpose(R, (0, 2, 1))).as_rotvec()
    angle = np.linalg.norm(offset, axis=1)
    limit = np.radians(settings.max_rotation_correction_deg)
    capped_total = int((angle > limit).sum())
    offset *= np.minimum(1.0, limit / np.maximum(angle, 1e-12))[:, None]
    new = Rotation.from_rotvec(offset).as_matrix() @ R
    free = ~frozen
    deg = np.degrees(np.linalg.norm(error, axis=1))
    moved = np.degrees(np.linalg.norm(rotation_errors(new, A), axis=1))
    report = {
        "lead_labels": lead,
        "error_rms_deg": float(np.sqrt(np.mean(deg[free] ** 2))) if free.any() else 0.0,
        "error_max_deg": float(deg[free].max()) if free.any() else 0.0,
        "update_max_deg": float(moved.max()),
        "capped_update": capped_update, "capped_total": capped_total,
    }
    return new, report


def residuals_at(reference, measured, orientations, indices, measured_rotations=None) -> dict:
    """The error at chosen labels, split into the hand's axes (mm), and the tilt (deg).

    ``orientations`` in the grasp convention (+X closing, +Z approach): the closing axis
    is the one that loses an object (about ``(80 mm - width) / 2`` of room for this hand),
    the approach axis only changes how deep it is gripped.
    """
    r = np.asarray(reference, dtype=float).reshape(-1, 3)
    m = np.asarray(measured, dtype=float).reshape(-1, 3)
    R = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    tilt = (None if measured_rotations is None else
            np.degrees(np.linalg.norm(rotation_errors(R, measured_rotations), axis=1)))
    out = {}
    for name, i in indices.items():
        e = r[int(i)] - m[int(i)]
        out[name] = {"total_mm": float(np.linalg.norm(e) * 1000),
                     "closing_mm": float(abs(e @ R[int(i)][:, 0]) * 1000),
                     "approach_mm": float(abs(e @ R[int(i)][:, 2]) * 1000)}
        if tilt is not None:
            out[name]["tilt_deg"] = float(tilt[int(i)])
    return out
