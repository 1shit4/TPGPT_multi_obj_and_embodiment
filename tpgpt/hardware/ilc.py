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

* **L** (``gain``): the share of the last pass's error added to the next attractor. At 0.5
  an error halves each pass (25 -> 12.5 -> 6 -> 3 mm); 1.0 removes a pure offset in one pass
  but learns noise and can oscillate.
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
* **Caps** (``max_update``, ``max_correction``): a pass moves no label more than 20 mm, and
  no label ends up more than 60 mm from the taught path.

Everything is on the label grid; :func:`label_positions` puts a replay's measured poses on
it. Position only: the orientation is replayed as taught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IlcSettings:
    gain: float = 0.5
    lead_s: float = 0.25
    cutoff_hz: float = 1.5
    max_update: float = 0.02
    max_correction: float = 0.06


def label_positions(sample_label, sample_positions, n_labels: int) -> np.ndarray:
    """Measured position at the moment the replay reached each label.

    ``sample_label`` is the (fractional, non-decreasing) label index being commanded
    when each pose was measured; while the replay pauses (the gripper acting) it stays
    constant. The position for label ``k`` is interpolated between the two samples that
    straddle the first time the index reached ``k`` -- the arm as it was when label
    ``k`` became its target, so a pause after a label does not count.
    """
    s = np.asarray(sample_label, dtype=float)
    x = np.asarray(sample_positions, dtype=float).reshape(-1, 3)
    out = np.empty((n_labels, 3))
    for k in range(n_labels):
        j = int(np.searchsorted(s, k, side="left"))
        if j <= 0:
            out[k] = x[0]
        elif j >= len(s):
            out[k] = x[-1]
        else:
            w = (k - s[j - 1]) / max(s[j] - s[j - 1], 1e-12)
            out[k] = x[j - 1] + np.clip(w, 0.0, 1.0) * (x[j] - x[j - 1])
    return out


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


def ilc_update(attractor, measured, reference, label_period: float, frozen=None,
               settings: IlcSettings = IlcSettings()) -> tuple[np.ndarray, dict]:
    """The next pass's attractor, and a report of this pass.

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
    lead = int(round(settings.lead_s / label_period))
    led = np.concatenate([error[lead:], np.repeat(error[-1:], lead, axis=0)]) if lead else error
    update = settings.gain * _lowpass(led, 1.0 / label_period, settings.cutoff_hz)
    update = _bridge(update, frozen)
    size = np.linalg.norm(update, axis=1)
    capped_update = int((size > settings.max_update).sum())
    update *= np.minimum(1.0, settings.max_update / np.maximum(size, 1e-12))[:, None]
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


def residuals_at(reference, measured, orientations, indices) -> dict:
    """The error at chosen labels, split into the hand's axes (mm).

    ``orientations`` in the grasp convention (+X closing, +Z approach): the closing axis
    is the one that loses an object (about ``(80 mm - width) / 2`` of room for this hand),
    the approach axis only changes how deep it is gripped.
    """
    r = np.asarray(reference, dtype=float).reshape(-1, 3)
    m = np.asarray(measured, dtype=float).reshape(-1, 3)
    R = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    out = {}
    for name, i in indices.items():
        e = r[int(i)] - m[int(i)]
        out[name] = {"total_mm": float(np.linalg.norm(e) * 1000),
                     "closing_mm": float(abs(e @ R[int(i)][:, 0]) * 1000),
                     "approach_mm": float(abs(e @ R[int(i)][:, 2]) * 1000)}
    return out
