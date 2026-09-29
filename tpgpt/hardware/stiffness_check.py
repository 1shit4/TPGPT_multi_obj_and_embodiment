"""What a stiffness does on the arm: jitter while holding, lag while moving. No ROS.

The analysis half of ``cartesian_target_publisher``'s stiffness check (step 3 of the
execution-gap plan, ``docs/hardware/FAITHFULNESS_AUDIT.md`` B2). Two questions decide
the stiffness labels' numbers, and both are measured on the arm, not assumed:

* **Does it vibrate?** On 2026-09-15, 800 N/m translational with 60 N·m/rad rotational
  put 1-1.5 mm of jitter on a constant command; 400/30 stayed under 0.54 mm. The two
  were raised together, so which one caused it is unknown. :func:`hold_jitter` measures
  a hold: the measured position about its own trend (detrended, so a slow settle or
  drift is not counted), its size and its dominant frequency.
* **How far behind does it run, and in which direction?** An impedance arm trails a
  moving target by ``K^-1 D v`` (viscous) plus ``F / K`` (friction, gravity). With a
  stiffness soft along the motion the lag should be mostly *along* the motion;
  :func:`move_lag` splits it into along and across.

Times are the pose messages' own stamps, relative to the first one.
"""

from __future__ import annotations

import numpy as np


def _uniform(stamps, positions):
    """Positions resampled onto a uniform grid at the median sample period."""
    t = np.asarray(stamps, dtype=float)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if dt <= 0.0:
        return t, x, 0.0
    grid = np.arange(t[0], t[-1] + 1e-12, dt)
    return grid, np.stack([np.interp(grid, t, x[:, i]) for i in range(3)], axis=1), dt


def detrended(stamps, positions, degree: int = 1) -> np.ndarray:
    """Positions minus a per-axis polynomial fit in time (degree 1: minus a straight line)."""
    t = np.asarray(stamps, dtype=float)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(t) <= degree + 1:
        return np.zeros_like(x)
    tc = t - t.mean()
    V = np.vander(tc, degree + 1)
    coef, *_ = np.linalg.lstsq(V, x, rcond=None)
    return x - V @ coef


def hold_jitter(stamps, positions, skip: float = 1.0) -> dict:
    """Jitter of a hold: the position about its linear trend, after ``skip`` seconds of settling.

    Returns ``{n, rate_hz, p95_mm, p2p_mm, peak_hz, peak_mm}``: the 95th percentile of the
    residual's norm, its largest per-axis range, and the strongest spectral component
    above 0.5 Hz (single-sided amplitude). Empty dict when there is too little data.
    """
    t = np.asarray(stamps, dtype=float)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    keep = t >= t[0] + skip if len(t) else np.zeros(0, bool)
    t, x = t[keep], x[keep]
    if len(t) < 20:
        return {}
    grid, xu, dt = _uniform(t, x)
    r = detrended(grid, xu)
    out = {"n": int(len(t)), "rate_hz": float(1.0 / dt) if dt else 0.0,
           "p95_mm": float(np.percentile(np.linalg.norm(r, axis=1), 95) * 1000),
           "p2p_mm": float((r.max(axis=0) - r.min(axis=0)).max() * 1000)}
    if dt > 0 and len(r) >= 32:
        freqs = np.fft.rfftfreq(len(r), dt)
        amp = 2.0 * np.abs(np.fft.rfft(r, axis=0)) / len(r)
        band = freqs > 0.5
        if band.any():
            k = np.argmax(amp[band].max(axis=1))
            out["peak_hz"] = float(freqs[band][k])
            out["peak_mm"] = float(amp[band][k].max() * 1000)
    return out


def recent_oscillation(stamps, positions, seconds: float = 1.0, min_hz: float = 3.0
                       ) -> tuple[float, float]:
    """``(peak-to-peak, frequency)`` of the strongest oscillation at ``min_hz`` or faster
    in the last ``seconds``, about a quadratic trend (m, Hz).

    From the spectrum (Hann window, amplitude-corrected), so it measures one oscillation,
    not the whole path: a lag building up as friction lets go bends the path by
    millimetres (4.3 mm about a quadratic, on a mock arm with 10 N of friction at
    350 N/m) but puts almost nothing above 3 Hz, and tiny sensor noise adds nothing
    either. Two earlier versions -- the path's range, then its range together with a
    count of zero crossings -- both cut on that bend. The stiffness check's safety cut.
    A frequency between two 1 Hz bins reads up to ~15 % low (the Hann window's
    scalloping: 3.4 mm for a 4 mm oscillation at 7.5 Hz).
    """
    t = np.asarray(stamps, dtype=float)
    if len(t) < 32:
        return 0.0, 0.0
    keep = t >= t[-1] - seconds
    grid, x, dt = _uniform(t[keep], np.asarray(positions, dtype=float).reshape(-1, 3)[keep])
    if dt <= 0.0 or len(grid) < 32:
        return 0.0, 0.0
    r = detrended(grid, x, degree=2)
    window = np.hanning(len(r))
    spectrum = 2.0 * np.abs(np.fft.rfft(r * window[:, None], axis=0)) / window.sum()
    freqs = np.fft.rfftfreq(len(r), dt)
    band = freqs > min_hz - 0.5 / (len(r) * dt)     # half a bin: the min_hz bin itself counts
    if not band.any():
        return 0.0, 0.0
    amp = spectrum[band].max(axis=1)
    k = int(np.argmax(amp))
    return float(2.0 * amp[k]), float(freqs[band][k])


class OscillationWatch:
    """True once the oscillation has stayed above ``limit`` for ``persist`` seconds.

    Persistence is what separates a vibration from the arm catching up at the end of a
    move: closing 20-40 mm of lag in a fraction of a second spills into the 3 Hz bin of
    :func:`recent_oscillation` (read 4.2-4.6 mm on a mock arm, no oscillation), but only
    while it is inside the 1 s window, so never for 1.5 s on end.
    """

    def __init__(self, limit: float, persist: float = 1.5):
        self.limit, self.persist = float(limit), float(persist)
        self.since: float | None = None

    def update(self, t: float, size: float) -> bool:
        if size <= self.limit:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        return t - self.since >= self.persist


def recent_p2p(stamps, positions, seconds: float = 0.5) -> float:
    """Largest per-axis range over the last ``seconds``, about a quadratic trend (m).

    The quadratic takes out the commanded motion itself (a smooth move bends less than
    a millimetre over half a second), so what is left is oscillation. The safety cut
    of the stiffness check.
    """
    t = np.asarray(stamps, dtype=float)
    if len(t) < 10:
        return 0.0
    keep = t >= t[-1] - seconds
    r = detrended(t[keep], np.asarray(positions, dtype=float).reshape(-1, 3)[keep], degree=2)
    return float((r.max(axis=0) - r.min(axis=0)).max()) if len(r) else 0.0


def move_lag(commanded, measured, direction, commanded_speed, min_speed: float = 0.02) -> dict:
    """Lag of a move split into along and across the direction of motion (mm).

    ``commanded``/``measured`` are ``(N, 3)`` positions sampled together; the lag is
    ``commanded - measured``. Only samples commanded faster than ``min_speed`` count.
    """
    u = np.asarray(direction, dtype=float)
    u = u / max(np.linalg.norm(u), 1e-12)
    e = np.asarray(commanded, dtype=float) - np.asarray(measured, dtype=float)
    fast = np.asarray(commanded_speed, dtype=float) > min_speed
    if not fast.any():
        return {}
    along = e[fast] @ u
    across = np.linalg.norm(e[fast] - np.outer(along, u), axis=1)
    return {"n": int(fast.sum()),
            "along_median_mm": float(np.median(along) * 1000),
            "along_max_mm": float(np.max(along) * 1000),
            "across_median_mm": float(np.median(across) * 1000),
            "across_max_mm": float(np.max(across) * 1000)}
