"""A source demonstration taught by hand on the real FR3, packaged as a SourceBundle.

The simulated source (:mod:`tpgpt.hardware.export_source`) is a scripted
pick-and-place recorded in MuJoCo. This module builds the **same bundle** from a
kinesthetic demonstration: the arm is floated in a zero-stiffness controller
(or Franka's own guiding mode), a person moves it through the task, and the
ROS node ``tpgpt_teach_node`` records ``/current_pose`` together with two key
presses marking the grasp and the release. Nothing here imports ROS, so the
conversion can be re-run offline from the saved ``raw.npz``::

    python -m tpgpt.hardware.teach --raw ~/tpgpt_demos/<demo>/raw.npz \
        --out ~/tpgpt_demos/<demo>/source_bundle

What each field of the bundle is made of, and why:

* **positions** -- the measured ``fr3_hand_tcp`` (the fingertip centre, which is
  the reference ``scene_keypoints`` and ``transport_labels`` expect), resampled
  onto the control grid. ``docs/sim/SIM_CLAUDE.md`` warns that labels are the *commanded
  attractor* (the point the impedance spring pulls toward), not the measured
  pose, because an impedance arm lags its attractor by ``K^-1 D v``. A hand-held
  demonstration has no attractor at all. This version still uses the measured path
  as the attractor, so the executing arm trails it by its whole lag (audit B1): the
  viscous part ``D/K * v`` (2/sqrt(350) s along the motion with the directional
  stiffness: 5 mm at 5 cm/s) plus a static part from joint friction and gravity
  (~10 N / K). Replacing it with the attractor that reproduces the taught path on
  the robot is the next stage of the execution-gap plan.
* **orientations** -- the measured ``fr3_hand_tcp`` rotation converted to the
  grasp convention (``+X`` closing, ``+Z`` approach) with
  :func:`~tpgpt.hardware.frames.fr3_tcp_to_grasp`, as ``export_source`` does
  with ``to_grasp_convention``.

With another hand on the flange (:mod:`tpgpt.hardware.end_effector`, e.g. the
Inspire hand), ``/current_pose`` is still the controller frame, so the recorded
pose is converted with that hand's description: the position is moved to the
hand's grasp point (``tool_offset``, the palm point for the Inspire hand) and the
rotation to the grasp convention (``grasp_from_ee``). The hand used is taken
from ``recording.metadata['end_effector']`` when the teach node wrote it.
* **velocities** -- Savitzky-Golay derivative of the resampled positions at the
  control period, so they are smooth enough for the policy fit and agree with
  the positions they belong to.
* **stiffness / damping** -- by default **directional** (:func:`directional_stiffness`,
  since 2026-09-29): 350 N/m along the direction of motion and 600 across it, stiff in
  every direction while the hand is still, damped with ``2 K^(1/2)``. A hand-held
  demonstration has no stiffness to record, so -- as in the simulation, whose scripted
  teacher sets 350/900 per segment -- the profile is the teacher's specification.
  ``stiffness_profile="static"`` gives the constant 400/40 every earlier bundle used.
* **gripper** -- ``+1`` between the grasp and release marks, ``-1`` elsewhere. No
  gripper is actuated during teaching; the marks only say *when* the object was
  held, which ``carry_indices`` and ``carry_transform`` need.
* **time_belief / time_rate** -- ``linspace(0, 1, M)`` and its constant rate,
  from the step index and the control frequency. No wall-clock time survives
  resampling (``docs/sim/SIM_CLAUDE.md`` rule 5).
* **placement** -- the object cloud captured before the demonstration, the
  grasp *actually taught* (TCP pose at the grasp mark), the measured table
  height, and a destination derived from the carry: the object's placed pose is
  implied by the grasp and the hand's motion (``carry_transform``), so the
  destination is set to the centre of the carried cloud and the lateral offset
  ``scene_keypoints`` inherits is zero.

Everything is expressed in the real robot's ``base`` frame, so no change of
frame is needed (the simulated bundle needs one, from robosuite's world).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tpgpt.hardware.end_effector import FRANKA_HAND, EndEffector

from tpgpt.hardware.frames import rotation_angle
from tpgpt.hardware.source_bundle import SourceBundle
from tpgpt.sim.keypoints import (
    GraspFrame,
    ObjectPlacement,
    apply_transform,
    carry_indices,
    carry_transform,
    fit_aligned_box,
    task_frame,
)
from tpgpt.transport.labels import PolicyLabels

#: Label frequency, as the simulated source (``control_freq`` 20).
TEACH_CONTROL_FREQ = 20

#: Static translational stiffness written into every label, N/m. The stock
#: ``cartesian_impedance_controller`` in ``controllers.real.yaml`` runs 400 N/m,
#: and the executor's default (``use_transported_stiffness:=false``) uses it.
TEACH_STIFFNESS = 400.0

#: Static translational damping, N s/m, likewise from ``controllers.real.yaml``.
TEACH_DAMPING = 40.0

#: The stiffness labels a taught demonstration carries by default: ``directional``
#: (:func:`directional_stiffness`) or ``static`` (TEACH_STIFFNESS / TEACH_DAMPING
#: throughout, how every demonstration before 2026-09-29 was converted).
STIFFNESS_PROFILE = "directional"

#: Directional stiffness, N/m: compliant along the direction the hand moves, stiff
#: across it (and in every direction while it is still). Chosen from this arm
#: (2026-09-29): 350 along keeps the ~10 N joint friction's lag at 10/350 = 29 mm,
#: inside the lag gate's 45 mm; 600 across is the highest value below the 800 that
#: vibrated with rotational 60 (2026-09-15; 600/30 still to be checked on the arm).
DIRECTIONAL_ALONG = 350.0
DIRECTIONAL_ACROSS = 600.0

#: Speeds over which the directionality is blended in, m/s: isotropic (stiff) at or
#: below the first, fully directional at or above the second. The 18:39 demonstration
#: creeps into its grasp at 1-2.5 mm/s and averages 21 mm/s over the last 30 mm.
DIRECTIONAL_RAMP = (0.002, 0.020)

#: Path length over which the direction of motion is taken, m: the chord of the last
#: stretch of path, not the instantaneous velocity, which is noise at creeping speed.
DIRECTIONAL_TANGENT = 0.010

#: Half-width of the moving log-Euclidean mean applied to the stiffness labels, in labels
#: (+-5 = 0.55 s of demonstration). Measured 2026-09-29 on the 18:39 demonstration planned
#: onto banana_20260926_171121 and queried where its dry run queried (attractor, phase):
#: unsmoothed, the policy's fitted K reached 236..980 N/m against labels of 350..600, 17.9 %
#: of queries more than 20 N/m outside them (the GP overshoots matrices whose axes turn);
#: +-3 gave 1.3 %, +-5 and +-6 0.0 % (349..609), and the grasp and release labels stay stiff
#: in every direction (599..600); +-8 starts softening the grasp (592).
DIRECTIONAL_SMOOTHING = 5

#: Below this speed the hand counts as still, m/s.
STILL_SPEED = 0.01

#: Still stretches longer than this are shortened to it, seconds. A long pause
#: in a demonstration becomes many labels with zero velocity at one place,
#: which the policy reproduces as a pause the stall watchdog then has to forgive.
MAX_DWELL = 1.0

#: Samples kept on either side of the motion when trimming, seconds.
TRIM_MARGIN = 0.25

#: Savitzky-Golay window at the control frequency (odd), and polynomial order.
SMOOTHING_WINDOW = 7
SMOOTHING_ORDER = 2

#: Fewest labels a demonstration may have.
MIN_LABELS = 40

#: Fewest points in the captured object cloud (``planner.MIN_CLOUD_POINTS``).
MIN_OBJECT_POINTS = 40

#: The taught grasp point must lie within this distance of the object cloud, m.
#: The fingertip centre sits inside the object between the pads, so the nearest
#: *visible* point is up to about half the object's width away.
GRASP_TO_CLOUD_TOLERANCE = 0.05

#: Above this tilt of the carry rotation the derived destination height, which
#: assumes the object was turned only about the vertical, is reported as
#: approximate.
MAX_CARRY_TILT_DEG = 10.0


class TeachError(RuntimeError):
    """A demonstration was refused. The message starts with the stage name."""


@dataclass
class Recording:
    """What the teaching node records, before any processing. Base frame.

    Attributes:
        stamps: ``(N,)`` message header stamps of ``/current_pose``, seconds.
            Used only to resample onto the control grid, then discarded.
        positions: ``(N, 3)`` measured ``fr3_hand_tcp`` positions.
        orientations: ``(N, 3, 3)`` measured ``fr3_hand_tcp`` rotations.
        grasp_stamp: Stamp at which the grasp was marked.
        release_stamp: Stamp at which the release was marked.
        object_points: ``(M, 3)`` object cloud captured before the demonstration.
        support_height: Height of the surface the object stood on.
        placed_points: Optional ``(K, 3)`` cloud captured after the release, used
            only as a cross-check of the derived placement.
        metadata: Free-form (object name, calibration, node parameters).
    """

    stamps: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    grasp_stamp: float
    release_stamp: float
    object_points: np.ndarray
    support_height: float
    placed_points: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            stamps=np.asarray(self.stamps, dtype=float),
            positions=np.asarray(self.positions, dtype=float),
            orientations=np.asarray(self.orientations, dtype=float),
            grasp_stamp=np.array(float(self.grasp_stamp)),
            release_stamp=np.array(float(self.release_stamp)),
            object_points=np.asarray(self.object_points, dtype=float).reshape(-1, 3),
            support_height=np.array(float(self.support_height)),
            placed_points=(np.zeros((0, 3)) if self.placed_points is None
                           else np.asarray(self.placed_points, dtype=float).reshape(-1, 3)),
            metadata=np.array(json.dumps(self.metadata, default=str)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        with np.load(Path(path), allow_pickle=False) as d:
            placed = d["placed_points"]
            return cls(
                stamps=d["stamps"],
                positions=d["positions"],
                orientations=d["orientations"],
                grasp_stamp=float(d["grasp_stamp"]),
                release_stamp=float(d["release_stamp"]),
                object_points=d["object_points"],
                support_height=float(d["support_height"]),
                placed_points=placed if len(placed) else None,
                metadata=json.loads(str(d["metadata"])),
            )


# --------------------------------------------------------------------------- labels
def resample(stamps, positions, orientations, dt: float):
    """Positions and rotations on a uniform grid of period ``dt``.

    Linear interpolation for positions and geodesic (slerp) interpolation for
    rotations, both between the two nearest measured samples. At 1 kHz input and
    20 Hz output every grid point has a sample within 0.5 ms.
    """
    from scipy.spatial.transform import Rotation, Slerp

    stamps = np.asarray(stamps, dtype=float).reshape(-1)
    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    keep = np.concatenate([[True], np.diff(stamps) > 0.0])
    stamps = stamps[keep]
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)[order][keep]
    orientations = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)[order][keep]
    if len(stamps) < 2:
        raise TeachError("too_few_labels: fewer than two distinct pose samples were recorded")
    grid = np.arange(stamps[0], stamps[-1] + 1e-9, dt)
    grid_positions = np.stack([np.interp(grid, stamps, positions[:, i]) for i in range(3)], axis=1)
    grid_rotations = Slerp(stamps, Rotation.from_matrix(orientations))(grid).as_matrix()
    return grid, grid_positions, grid_rotations


def smooth(positions: np.ndarray, dt: float, window: int = SMOOTHING_WINDOW,
           order: int = SMOOTHING_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay smoothed positions and their time derivative."""
    from scipy.signal import savgol_filter

    n = len(positions)
    window = int(window) | 1
    if n <= order + 1 or window <= order:
        return positions.copy(), np.gradient(positions, dt, axis=0)
    window = min(window, n if n % 2 else n - 1)
    smoothed = savgol_filter(positions, window, order, axis=0, mode="interp")
    velocity = savgol_filter(positions, window, order, deriv=1, delta=dt, axis=0, mode="interp")
    return smoothed, velocity


def _still_runs(still: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` index ranges of consecutive True values."""
    runs, start = [], None
    for i, value in enumerate(still):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(still)))
    return runs


def select_samples(speed: np.ndarray, dt: float, protected: tuple[int, ...],
                   still_speed: float = STILL_SPEED, max_dwell: float = MAX_DWELL,
                   trim_margin: float = TRIM_MARGIN) -> np.ndarray:
    """Indices kept after trimming the still ends and shortening long pauses.

    The ``protected`` indices (the grasp and release samples) are always kept,
    so a pause at the grasp is shortened but never removes the moment the
    gripper channel changes.
    """
    n = len(speed)
    moving = np.flatnonzero(speed > still_speed)
    if len(moving) == 0:
        raise TeachError(
            f"no_motion: the TCP never moved faster than {still_speed * 100:.0f} cm/s"
        )
    margin = int(round(trim_margin / dt))
    first = max(0, min(int(moving[0]) - margin, *protected))
    last = min(n - 1, max(int(moving[-1]) + margin, *protected))
    keep = np.zeros(n, dtype=bool)
    keep[first:last + 1] = True

    cap = max(2, int(round(max_dwell / dt)))
    still = speed <= still_speed
    for start, stop in _still_runs(still[first:last + 1]):
        start, stop = start + first, stop + first
        if stop - start <= cap:
            continue
        head = cap // 2
        keep[start + head: stop - (cap - head)] = False
    for index in protected:
        keep[index] = True
    return np.flatnonzero(keep)


def motion_directions(positions: np.ndarray, tangent_length: float = DIRECTIONAL_TANGENT
                      ) -> np.ndarray:
    """Unit direction of travel at each sample: the chord over the last ``tangent_length`` of path.

    Before the path has covered that length, the chord of its first ``tangent_length``
    is used. Where the path has not moved at all the direction is zero.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(positions)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    directions = np.zeros((n, 3))
    first = min(int(np.searchsorted(arc, tangent_length)), n - 1)
    for i in range(n):
        if arc[i] >= tangent_length:
            j = int(np.searchsorted(arc, arc[i] - tangent_length, side="right")) - 1
            chord = positions[i] - positions[max(j, 0)]
        else:
            chord = positions[first] - positions[0]
        length = float(np.linalg.norm(chord))
        if length > 1e-9:
            directions[i] = chord / length
    return directions


def log_euclidean_smooth(stiffness: np.ndarray, half_width: int) -> np.ndarray:
    """Moving log-Euclidean mean of SPD matrices: ``exp(mean(log K))`` over ``+-half_width``.

    The standard average of stiffness matrices: it stays positive definite and does not
    shrink them the way an entry-wise average of matrices with turning axes does.
    """
    K = np.asarray(stiffness, dtype=float)
    if half_width <= 0:
        return K.copy()
    values, vectors = np.linalg.eigh(K)
    logs = np.einsum("nij,nj,nkj->nik", vectors, np.log(values), vectors)
    csum = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(logs, axis=0)])
    n = len(K)
    lo = np.clip(np.arange(n) - half_width, 0, n)
    hi = np.clip(np.arange(n) + half_width + 1, 0, n)
    mean = (csum[hi] - csum[lo]) / (hi - lo)[:, None, None]
    mean = 0.5 * (mean + np.transpose(mean, (0, 2, 1)))
    values, vectors = np.linalg.eigh(mean)
    return np.einsum("nij,nj,nkj->nik", vectors, np.exp(values), vectors)


def directional_stiffness(
    positions: np.ndarray, velocities: np.ndarray,
    along: float = DIRECTIONAL_ALONG, across: float = DIRECTIONAL_ACROSS,
    ramp: tuple[float, float] = DIRECTIONAL_RAMP,
    tangent_length: float = DIRECTIONAL_TANGENT,
    smoothing: int = DIRECTIONAL_SMOOTHING,
) -> tuple[np.ndarray, np.ndarray]:
    """Stiffness and damping labels: compliant along the motion, stiff across it.

    ``K_i = across I - (across - along) w_i u_i u_i^T`` with ``u_i`` the direction of
    travel (:func:`motion_directions`) and ``w_i`` the speed blended from 0 at
    ``ramp[0]`` to 1 at ``ramp[1]``. So a still hand is stiff in every direction --
    the simulated teacher's firm grasp and release (``tpgpt.sim.demo``: 900 N/m at the
    grasp, insertion and release, 350 in transit) -- and a moving one gives way
    along its motion only: what meets it head-on is not forced through, and what
    decides a grasp (the closing axis, never the direction of motion in a taught
    pick) stays stiff.

    The profile is the demonstration's stiffness label (Sec. III-A), transported
    with the map's rotation part like every stiffness (Sec. III-G). Since the
    orientation labels rotate by the same ``J_perp`` (Eq. 11), the stiffness seen
    from the hand is the same at matching labels of the source and the target.

    The profile is then smoothed over ``+-smoothing`` labels with a log-Euclidean
    mean (:func:`log_euclidean_smooth`): unsmoothed, the policy's regression of it
    overshot to 236..980 N/m between labels (:data:`DIRECTIONAL_SMOOTHING`).

    ``D_i = 2 K_i^(1/2)``: the damping CRISP applies (``d_pos: -1``) along the
    stiffness's own axes, so the labels, the lag gate's ``||K^-1 D||`` and the arm
    agree. The simulated teacher used ``1.8 sqrt(k)``.

    Deviation from the simulation, and why: the sim's teacher is isotropic (one ``k``
    per segment, stiff at the insertion as a whole); this one is directional, soft
    along the insertion. ``docs/hardware/FAITHFULNESS_AUDIT.md``, B2.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    speeds = np.linalg.norm(np.asarray(velocities, dtype=float).reshape(-1, 3), axis=1)
    lo, hi = float(ramp[0]), float(ramp[1])
    weight = np.clip((speeds - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    u = motion_directions(positions, tangent_length)
    eye = np.eye(3)
    K = across * eye[None] - (across - along) * weight[:, None, None] * np.einsum("ni,nj->nij", u, u)
    K = log_euclidean_smooth(K, int(smoothing))
    values, vectors = np.linalg.eigh(K)
    D = np.einsum("nij,nj,nkj->nik", vectors, 2.0 * np.sqrt(np.maximum(values, 0.0)), vectors)
    return K, D


def labels_from_recording(
    recording: Recording,
    control_freq: int = TEACH_CONTROL_FREQ,
    stiffness: float = TEACH_STIFFNESS,
    damping: float = TEACH_DAMPING,
    still_speed: float = STILL_SPEED,
    max_dwell: float = MAX_DWELL,
    smoothing_window: int = SMOOTHING_WINDOW,
    end_effector: EndEffector | None = None,
    stiffness_profile: str = STIFFNESS_PROFILE,
    along: float = DIRECTIONAL_ALONG,
    across: float = DIRECTIONAL_ACROSS,
) -> PolicyLabels:
    """Policy labels (grasp point, grasp convention, base frame) from a recording."""
    ee = end_effector or recording_end_effector(recording)
    dt = 1.0 / float(control_freq)
    if not recording.release_stamp > recording.grasp_stamp:
        raise TeachError(
            "release_before_grasp: the release was marked "
            f"{recording.grasp_stamp - recording.release_stamp:.2f} s before the grasp"
        )
    grid, positions, tcp_rotations = resample(
        recording.stamps, recording.positions, recording.orientations, dt
    )
    # Controller frame -> the hand's grasp point (identity for the Franka Hand).
    positions = ee.grasp_point(positions, tcp_rotations)
    if not (grid[0] <= recording.grasp_stamp and recording.release_stamp <= grid[-1]):
        raise TeachError(
            "marks_outside_recording: the grasp and release marks must both fall "
            "between the start and the stop of the recording"
        )
    positions, velocities = smooth(positions, dt, smoothing_window)
    closed = (grid >= recording.grasp_stamp) & (grid < recording.release_stamp)
    if not closed.any():
        raise TeachError("carry_too_short: no control step lies between the grasp and release marks")
    grasp_index = int(np.flatnonzero(closed)[0])
    release_index = int(np.flatnonzero(closed)[-1])

    # Keep the jaws-open labels just after the release. Without them the gripper
    # channel ends closed and the executor never opens the hand: measured on a real
    # demonstration held still for 8 s after the release mark, trimming the still
    # end left the release on the last of 445 labels and no open label after it.
    after_release = int(round(TRIM_MARGIN / dt))
    if release_index + after_release >= len(grid):
        raise TeachError(
            "release_at_end: the recording stopped within "
            f"{TRIM_MARGIN:.2f} s of the release mark; keep recording a moment after it"
        )
    protected = (grasp_index, *range(release_index, release_index + after_release + 1))
    speed = np.linalg.norm(velocities, axis=1)
    keep = select_samples(speed, dt, protected, still_speed, max_dwell)
    positions = positions[keep]
    tcp_rotations = tcp_rotations[keep]
    gripper = np.where(closed[keep], 1.0, -1.0)
    # Velocities from the kept positions: removed samples are still ones, so the
    # path is continuous across a cut and a finite difference stays honest.
    _, velocities = smooth(positions, dt, smoothing_window)

    n = len(positions)
    if n < MIN_LABELS:
        raise TeachError(f"too_few_labels: {n} labels after trimming (need {MIN_LABELS})")
    speeds = np.linalg.norm(velocities, axis=1)
    if stiffness_profile == "static":
        K = np.repeat((np.eye(3) * float(stiffness))[None], n, axis=0)
        D = np.repeat((np.eye(3) * float(damping))[None], n, axis=0)
        stiffness_info = {"static_stiffness": float(stiffness), "static_damping": float(damping),
                          "expected_execution_lag_at_max_speed_m":
                              float(damping / stiffness * speeds.max())}
    elif stiffness_profile == "directional":
        K, D = directional_stiffness(positions, velocities, along, across)
        # Moving, the lag is along the motion, where the time constant is D/K = 2/sqrt(along).
        stiffness_info = {"directional_along": float(along), "directional_across": float(across),
                          "directional_ramp_mps": list(DIRECTIONAL_RAMP),
                          "directional_tangent_m": DIRECTIONAL_TANGENT,
                          "directional_smoothing_labels": DIRECTIONAL_SMOOTHING,
                          "expected_execution_lag_at_max_speed_m":
                              float(2.0 / np.sqrt(along) * speeds.max())}
    else:
        raise ValueError(f"stiffness_profile must be 'directional' or 'static', not "
                         f"{stiffness_profile!r}")
    return PolicyLabels(
        positions=positions,
        velocities=velocities,
        orientations=ee.grasp_from_ee(tcp_rotations),
        stiffness=K,
        damping=D,
        gripper=gripper,
        time_belief=np.linspace(0.0, 1.0, n),
        time_rate=np.full(n, 1.0 / (dt * max(n - 1, 1))),
        metadata={
            "control_freq": int(control_freq),
            "dt": dt,
            "n_labels": n,
            "duration_s": dt * (n - 1),
            "scene": "taught_on_fr3",
            "end_effector": ee.name,
            "n_raw_samples": int(len(recording.stamps)),
            "n_grid_samples": int(len(grid)),
            "recorded_duration_s": float(grid[-1] - grid[0]),
            "max_speed_mps": float(speeds.max()),
            "mean_speed_mps": float(speeds.mean()),
            "stiffness_profile": stiffness_profile,
            **stiffness_info,
        },
    )


# --------------------------------------------------------------------------- placement
def placement_from_recording(recording: Recording, labels: PolicyLabels) -> tuple[ObjectPlacement, dict]:
    """The source scene: object cloud, taught grasp, support and destination."""
    points = np.atleast_2d(np.asarray(recording.object_points, dtype=float))
    if len(points) < MIN_OBJECT_POINTS:
        raise TeachError(
            f"object_not_seen: the captured object cloud has {len(points)} points "
            f"(need {MIN_OBJECT_POINTS}); call capture_scene with the arm out of view"
        )
    grasp_index, release_index = carry_indices(labels)
    R_grasp = labels.orientations[grasp_index]
    grasp = GraspFrame(tcp=labels.positions[grasp_index], approach=R_grasp[:, 2],
                       closing=R_grasp[:, 0])
    gap = float(np.linalg.norm(points - grasp.tcp, axis=1).min())
    if gap > GRASP_TO_CLOUD_TOLERANCE:
        raise TeachError(
            f"grasp_not_on_object: the TCP at the grasp mark is {gap * 1000:.0f} mm from the "
            f"nearest object point (max {GRASP_TO_CLOUD_TOLERANCE * 1000:.0f} mm); was the grasp "
            "marked at the object, and did the object move after capture_scene?"
        )

    support = float(recording.support_height)
    rotation, translation = carry_transform(labels)
    place_points = apply_transform(points, rotation, translation)
    place_grasp = grasp.transformed(rotation, translation)
    tilt = float(np.degrees(rotation_angle(np.eye(3), _vertical_part(rotation))))
    lift = float(labels.positions[release_index, 2] - labels.positions[grasp_index, 2])
    destination_height = support + lift
    frame = task_frame(np.array([0.0, 0.0, 1.0]), place_grasp.closing)
    centre, _ = fit_aligned_box(place_points, frame)
    destination = np.array([centre[0], centre[1], destination_height])

    info = {
        "grasp_index": grasp_index,
        "release_index": release_index,
        "grasp_to_cloud_m": gap,
        "carry_translation_m": translation.tolist(),
        "carry_rotation_deg": float(np.degrees(rotation_angle(np.eye(3), rotation))),
        "carry_tilt_deg": tilt,
        "carry_tilt_ok": tilt <= MAX_CARRY_TILT_DEG,
        "lift_at_release_m": lift,
        "place_points": place_points,
        "place_grasp": place_grasp,
    }
    placed = recording.placed_points
    if placed is not None and len(placed) >= MIN_OBJECT_POINTS:
        info["placed_cloud_centroid_xy_error_m"] = float(np.linalg.norm(
            np.asarray(placed)[:, :2].mean(axis=0) - place_points[:, :2].mean(axis=0)))
        info["placed_cloud_lowest_z_m"] = float(np.percentile(np.asarray(placed)[:, 2], 1.0))
    placement = ObjectPlacement(
        points=points,
        grasp=grasp,
        support_height=support,
        destination=destination,
        destination_height=destination_height,
    )
    return placement, info


def _vertical_part(rotation: np.ndarray) -> np.ndarray:
    """The rotation with its turn about world ``z`` removed (its tilt alone)."""
    z = np.asarray(rotation)[:, 2]
    axis = np.cross([0.0, 0.0, 1.0], z)
    s, c = float(np.linalg.norm(axis)), float(z[2])
    if s < 1e-12:
        return np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * K @ K


# --------------------------------------------------------------------------- bundle
def recording_end_effector(recording: Recording) -> EndEffector:
    """The hand a recording was taught with (Franka Hand when not recorded)."""
    data = (recording.metadata or {}).get("end_effector")
    return FRANKA_HAND if not data else EndEffector.from_dict(data)


def bundle_from_recording(recording: Recording, **label_options) -> tuple[SourceBundle, dict]:
    """A :class:`SourceBundle` plus a report of the checks, from a recording."""
    ee = label_options.get("end_effector") or recording_end_effector(recording)
    label_options["end_effector"] = ee
    labels = labels_from_recording(recording, **label_options)
    placement, info = placement_from_recording(recording, labels)
    report = {k: v for k, v in info.items() if k not in ("place_points", "place_grasp")}
    report.update({k: labels.metadata[k] for k in (
        "n_labels", "duration_s", "recorded_duration_s", "max_speed_mps", "mean_speed_mps",
        "expected_execution_lag_at_max_speed_m")})
    report["n_object_points"] = int(len(placement.points))
    report["support_height_m"] = placement.support_height
    report["destination_m"] = placement.destination.tolist()
    report["destination_height_m"] = placement.destination_height
    metadata = {
        "description": "TPGPT source demonstration taught by hand on the FR3, fingertip labels "
                       "in the grasp convention, expressed in the robot base frame",
        "source": "taught_on_fr3",
        "source_gripper": "fr3_hand" if ee.is_parallel_jaw else ee.name,
        "end_effector": ee.to_dict(),
        "control_freq": int(labels.metadata["control_freq"]),
        "n_labels": len(labels),
        "frame": "base",
        "orientation_convention": "grasp (+X closing, +Z approach)",
        "position_reference": (f"{ee.controller_frame} measured, moved to the {ee.name} grasp "
                               "point, used as the attractor"),
        # "static" makes the executor's transit_rotational_scale auto resolve to 1.0; any
        # other value to the simulation's 0.25 (tpgpt_executor_node).
        "stiffness": labels.metadata["stiffness_profile"],
        **{k: v for k, v in labels.metadata.items()
           if k.startswith(("static_", "directional_"))},
        "teach_report": report,
        "recording": recording.metadata,
    }
    try:
        from tpgpt.reporting.provenance import provenance
        metadata["provenance"] = provenance()
    except Exception as exc:  # provenance is informative, never a reason to lose a demo
        metadata["provenance"] = {"error": str(exc)}
    return SourceBundle(tool_labels=labels, placement=placement, metadata=metadata), info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a taught recording into a source bundle")
    parser.add_argument("--raw", type=Path, required=True, help="raw.npz written by tpgpt_teach_node")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-freq", type=int, default=TEACH_CONTROL_FREQ)
    parser.add_argument("--stiffness-profile", choices=("directional", "static"),
                        default=STIFFNESS_PROFILE)
    parser.add_argument("--along", type=float, default=DIRECTIONAL_ALONG,
                        help="directional: N/m along the motion")
    parser.add_argument("--across", type=float, default=DIRECTIONAL_ACROSS,
                        help="directional: N/m across the motion, and in every direction when still")
    parser.add_argument("--stiffness", type=float, default=TEACH_STIFFNESS, help="static: N/m")
    parser.add_argument("--damping", type=float, default=TEACH_DAMPING, help="static: N s/m")
    parser.add_argument("--still-speed", type=float, default=STILL_SPEED)
    parser.add_argument("--max-dwell", type=float, default=MAX_DWELL)
    parser.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW)
    args = parser.parse_args(argv)
    bundle, _ = bundle_from_recording(
        Recording.load(args.raw), control_freq=args.control_freq, stiffness=args.stiffness,
        damping=args.damping, still_speed=args.still_speed, max_dwell=args.max_dwell,
        smoothing_window=args.smoothing_window, stiffness_profile=args.stiffness_profile,
        along=args.along, across=args.across,
    )
    bundle.save(args.out)
    print(f"wrote {args.out}: {bundle.tool_labels.describe()}")
    print(json.dumps(bundle.metadata["teach_report"], indent=2, default=str))


if __name__ == "__main__":
    main()
