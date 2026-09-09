"""Which stage of a run actually failed, and by how much.

An end-to-end outcome such as ``placed_in_the_wrong_place`` names where the run
*stopped*, not where it *went wrong*, and the two are rarely the same stage. A
hand that closed on empty air and a hand that gripped the object correctly and
set it down 20 cm off both finish with the object away from its slot, and the
placement error is identical evidence for two problems with nothing in common.

So a run is replayed against its own recorded trace and split into the physical
stages a pick-and-place has, each with a pass/fail and a measured margin:

``approach``   the arm reached the start of the transported path.
``reach``      the hand arrived at the chosen grasp pose before the jaws moved.
``grasp``      the jaws closed *on the object* rather than on air or on a wall.
``lift``       the object came off its support.
``carry``      it stayed in the hand for the transit.
``place``      it was released over the destination.
``settle``     it ended up resting in the slot.

The first stage that fails is the one to fix; everything after it is a
consequence. Reporting the last stage instead is what makes a campaign look like
a transport problem when it is a grasping problem.

Stage thresholds are measured quantities, not preferences -- see the constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpgpt.sim.keypoints import carry_indices

#: The hand must be within this of the commanded grasp point when the jaws move.
#:
#: The impedance controller's own steady-state lag is 20-45 mm (``ROBOTICS_NOTES``
#: section 2), so anything under that is the controller behaving normally. Above
#: it the jaws are closing somewhere the arm was never asked to be.
#:
#: **This scalar is kept for continuity, and it is the weaker of the two tests.**
#: It is a straight-line distance in world coordinates, so it adds up errors
#: along three axes that have wildly different consequences -- see
#: :func:`reach_axes`. Prefer ``reach_closing`` where a grasp and a gripper are
#: known. Section 7.26.
REACH_TOLERANCE = 0.045

#: Fraction of the jaw aperture allowed as closing-axis error when the object's
#: own width cannot be measured.
#:
#: With the width known the budget is exact -- ``(aperture - width) / 2``, the
#: room actually left on each side. Without it, no budget is exact and this is a
#: deliberately loose stand-in: half the half-aperture. It is documented rather
#: than tuned, because a threshold nobody can derive is how a metric stops
#: meaning anything.
CLOSING_BUDGET_FRACTION = 0.25

#: A rise of this much means the object left its support rather than being
#: nudged. Objects here are 45-153 mm tall; the table is flat, so a genuine lift
#: clears it by more than the couple of millimetres of settling jitter.
LIFT_HEIGHT = 0.02

#: Held for at least this fraction of the steps between closing and opening.
CARRY_FRACTION = 0.6

#: Released within this of the slot centre, measured laterally.
PLACE_TOLERANCE = 0.06

STAGES = ("approach", "reach", "grasp", "lift", "carry", "place", "settle")

#: One plain sentence per stage, for the report.
STAGE_TEXT = {
    "approach": "the arm reached the start of the transported motion",
    "reach": "the hand arrived where the grasp was planned",
    "grasp": "the jaws closed on the object",
    "lift": "the object came off the table",
    "carry": "the object stayed in the hand while it was carried",
    "place": "the object was let go above the right shelf",
    "settle": "the object ended up resting in the slot",
}


@dataclass
class Stage:
    name: str
    ok: bool
    detail: str
    value: float | None = None

    def __str__(self) -> str:
        return f"{'ok  ' if self.ok else 'FAIL'} {self.name:<9} {self.detail}"


@dataclass
class Diagnosis:
    """Stage-by-stage account of one execution.

    The stages are diagnostic *proxies* with measured thresholds; the task
    outcome is ground truth. When the two disagree -- a run that put the object
    in its slot while the hand arrived further from the planned grasp than
    :data:`REACH_TOLERANCE` allows, because it gripped a different part of the
    object and that worked -- the outcome wins. Otherwise a campaign's funnel
    reports fewer runs reaching the last stage than it reports succeeding, which
    is not a subtlety a reader should have to resolve.
    """

    stages: list[Stage] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    succeeded: bool = False

    @property
    def first_failure(self) -> Stage | None:
        if self.succeeded:
            return None
        return next((s for s in self.stages if not s.ok), None)

    @property
    def proxy_misfires(self) -> list[str]:
        """Stages whose threshold said "failed" on a run that nonetheless worked.

        Kept rather than hidden: a proxy that fires on a success is a proxy
        whose threshold is worth revisiting.
        """
        if not self.succeeded:
            return []
        return [s.name for s in self.stages if not s.ok]

    @property
    def blame(self) -> str:
        failure = self.first_failure
        return failure.name if failure is not None else "success"

    @property
    def reached_stage(self) -> int:
        """How many stages completed before the first failure."""
        return sum(1 for s in self.stages if s.ok) if self.first_failure else len(self.stages)

    def summary_of_misfires(self) -> str:
        names = self.proxy_misfires
        if not names:
            return ""
        return (
            "the task succeeded although "
            + ", ".join(names) + " read as failed; the threshold is a proxy"
        )

    def summary(self) -> str:
        failure = self.first_failure
        if failure is None:
            return "every stage completed"
        return f"failed at '{failure.name}': {failure.detail}"


@dataclass(frozen=True)
class Precondition:
    """One checkable statement about a run's setup.

    Attributes:
        name: Short identifier, used in the abort message.
        ok: Whether it holds.
        detail: What was measured, always -- including when it passed, so a
            green check still says what it saw. A check that reports only
            "ok" cannot be distinguished from a check that did not run, and
            7.19 is about exactly that: a diagnostic whose instrument was
            returning zeros was recorded as having *disproved* a hypothesis.
    """

    name: str
    ok: bool
    detail: str


def replay_preconditions(
    env, gripper: str, object_name: str, object_reference=None
) -> list[Precondition]:
    """Assert a replay cell's setup before spending physics on it.

    Every check here corresponds to a bug that was found *after* a 12 to 25
    minute campaign had already run, and every one of them is answerable in
    milliseconds from a freshly built environment. Running them on the first
    cell and aborting turns "discover the harness was wrong once the results
    look strange" into "refuse to start".

    The four:

    ``gripper_mounted``
        The hand actually on the arm is the one asked for. ``build_scene``
        hardcoded ``robots="Panda"`` and passed no ``gripper_types``, so a
        three-hand comparison ran a Panda three times and reported it as a
        cross-embodiment result. Nothing in the numbers looked wrong.
    ``scene_unstepped``
        The only stepping that has happened is the scene's own object settling,
        so this environment has not been *driven* yet. A replay leaves the
        object displaced and the arm parked at the end of the path; reusing one
        environment made the *ordering* decide the result, at placement errors
        of 354, 698 and 1170 mm and 0% reachability on the last object.

        Compared against ``SETTLE_STEPS * timestep`` rather than against zero,
        because ``TabletopShelf._reset_internal`` legitimately runs 60 sim steps
        so the dropped meshes come to rest -- 0.120 s at a 2 ms timestep. A
        cloud captured before they settle describes a pose the object is no
        longer in. Anything past that budget is a control step, which is what
        this check exists to refuse.
    ``object_placement``
        The object sits where the same seed put it last time, **for this hand**.
        Catches an unseeded sampler, which would silently turn a comparison of
        constructions into a comparison of scenes.

        Keyed per hand, not shared across them, because it is not shared: the
        same seed with a different gripper mounted settles the can 14.6 mm away
        (``[-0.1255, -0.0683, 0.8399]`` on a Panda against
        ``[-0.1139, -0.0777, 0.8426]`` on a Robotiq 2F-140). The placement
        sampler is seeded identically; the 60 settle steps then run with a
        different hand in the scene, and the meshes come to rest somewhere
        slightly different. So a cross-hand comparison carries a ~15 mm scene
        difference by construction, and this check must not report that as a
        reproducibility failure.
    ``closure_calibrated``
        This hand has a measured closure calibration and ``+1`` is known to
        shut it. Without this the jaw trace is uninterpretable, which is how a
        Robotiq 2F-140 closing harder was read as its jaws flying open
        (7.28).

    Args:
        env: A freshly built, unstepped environment.
        gripper: Registry short name of the hand that was requested.
        object_name: Object this cell manipulates.
        object_reference: The object position recorded on the first cell, or
            ``None`` on the first cell itself.

    Returns:
        One :class:`Precondition` per check, in order. Use :func:`require` to
        turn a failure into an abort.
    """
    from tpgpt.grasp.grippers import gripper_frame, resolve_pair

    checks: list[Precondition] = []

    expected = resolve_pair(gripper).robosuite
    mounted = env.robots[0].gripper
    mounted = mounted["right"] if isinstance(mounted, dict) else mounted
    actual = type(mounted).__name__
    checks.append(Precondition(
        "gripper_mounted",
        actual == expected,
        f"requested {gripper!r} -> {expected}; mounted {actual}",
    ))

    elapsed = float(env.sim.data.time)
    # **The budget is what the scene actually spent settling, not its minimum.**
    # ``SETTLE_STEPS`` is now a floor, not the count: settling runs until the
    # objects stop moving (7.32), which takes 150 to 600 steps depending on the
    # scene. Comparing against the 60-step floor rejected every cell of a
    # campaign -- correctly, by its own logic, and uselessly.
    steps = int(getattr(env, "settle_steps_taken", 0) or
                getattr(env, "SETTLE_STEPS", 0))
    settle = float(steps * float(env.sim.model.opt.timestep))
    checks.append(Precondition(
        "scene_unstepped",
        elapsed <= settle + 1e-9,
        f"sim.data.time = {elapsed:.4f} s against the {steps} settle steps "
        f"this scene took ({settle:.4f} s)",
    ))

    position = np.asarray(env.object_position(object_name), dtype=float)
    if object_reference is None:
        checks.append(Precondition(
            "object_placement", True,
            f"{object_name} at {np.round(position, 4).tolist()} (reference for later cells)",
        ))
    else:
        moved = float(np.linalg.norm(position - np.asarray(object_reference, float)))
        checks.append(Precondition(
            "object_placement", moved < 1e-6,
            f"{object_name} moved {moved * 1000:.3f} mm from the first cell's placement",
        ))

    frame = gripper_frame(gripper) or {}
    span = float(frame.get("spread_open", 0.0)) - float(frame.get("spread_closed", 0.0))
    shuts = bool(frame.get("plus_one_closes", False))
    reading = jaw_closure_probe(env, gripper)()
    checks.append(Precondition(
        "closure_calibrated",
        span > 1e-6 and shuts and np.isfinite(reading),
        f"travel {span * 1000:.1f} mm, +1 shuts = {shuts}, "
        f"reads {reading:.3f} at reset",
    ))
    return checks


def require(checks: list[Precondition], context: str = "") -> None:
    """Raise unless every precondition holds.

    Args:
        checks: Output of :func:`replay_preconditions`.
        context: Named in the message, e.g. the cell being started.

    Raises:
        RuntimeError: listing every check and what it measured, passed ones
            included, so the abort message is a complete picture of the setup
            rather than a single line about the first thing that broke.
    """
    if all(check.ok for check in checks):
        return
    lines = [
        f"  [{'ok ' if check.ok else 'FAIL'}] {check.name}: {check.detail}"
        for check in checks
    ]
    raise RuntimeError(
        f"preconditions failed{' for ' + context if context else ''}; "
        "refusing to run:\n" + "\n".join(lines)
    )


def object_probe(env, object_name: str, gripper: str | None = None):
    """A ``probe`` for :func:`~tpgpt.sim.rollout.rollout_policy`.

    Records what the *object* is doing, which is the half of the story the
    end-effector trace cannot tell. Kept to scalars and one 3-vector so a
    thousand-step rollout costs a few tens of kilobytes.

    Args:
        env: The environment being run.
        object_name: Object to watch.
        gripper: Registry short name of the mounted hand. When given, the trace
            also carries ``closure``, a cross-hand reading of how far the jaws
            have shut (:func:`jaw_closure_probe`). Without it only ``jaw`` is
            recorded, which is **hand-local** and must not be compared across
            embodiments -- see :func:`_jaw_opening`. Optional so single-hand
            callers keep working, but any multi-hand comparison must pass it.
    """
    closure = jaw_closure_probe(env, gripper) if gripper else None

    def probe(env_) -> dict:
        position = env_.object_position(object_name)
        record = {
            "object_x": float(position[0]),
            "object_y": float(position[1]),
            "object_z": float(position[2]),
            "jaw": float(_jaw_opening(env_)),
            "held": float(_gripper_touches(env_, object_name)),
        }
        if closure is not None:
            record["closure"] = float(closure())
        # The hand's *measured* orientation. Where the fingertips are is the
        # wrist plus a hand-specific offset rotated into the world, so using the
        # planned rotation instead of the real one mis-states the fingertip
        # position by up to twice the offset times the angular error -- 14 mm
        # for a Panda 20 degrees off. That is the same size as the effect being
        # measured.
        rotation = _eef_rotation(env_)
        for i in range(3):
            for j in range(3):
                record[f"eef_R{i}{j}"] = float(rotation[i, j])
        return record

    return probe


def _eef_rotation(env) -> np.ndarray:
    """World rotation of the arm's ``grip_site``."""
    try:
        site = env.robots[0].gripper
        site = site["right"] if isinstance(site, dict) else site
        name = f"{site.naming_prefix}grip_site"
        index = env.sim.model.site_name2id(name)
        return np.array(env.sim.data.site_xmat[index]).reshape(3, 3)
    except Exception:  # pragma: no cover - embodiment without that site
        return np.eye(3)


def _jaw_opening(env) -> float:
    """Sum of the gripper's finger joint positions. **Hand-local only.**

    Kept because it needs no calibration and is a cheap way to see that
    *something* moved on a hand already known to work. It is **not** comparable
    across hands and must never be read as a width or thresholded:

    * the sign flips -- closing *lowers* it on the Panda and UMI (prismatic
      fingers travelling toward each other) and *raises* it on the five
      revolute hands (linkages folding inward on a rising angle);
    * the units differ -- metres on the prismatic hands, radians on the
      revolute ones, so "shut" is 0.001 on a Panda and 4.9 on an XArm;
    * for the Rethink and Yumi hands the joints named in ``gripper.joints``
      move 0.0012 and 0.0000 while the fingers travel 45.9 mm and 38.3 mm, so
      there is no signal at all.

    Use :func:`jaw_closure_probe` for anything cross-hand. Reading this number
    as a width is what misattributed the first multi-gripper comparison: a
    Robotiq 2F-140 closing harder, 0.63 -> 1.38 rad, was read as its jaws
    flying open. ROBOTICS_NOTES.md section 7.28.
    """
    try:
        gripper = env.robots[0].gripper
        gripper = gripper["right"] if isinstance(gripper, dict) else gripper
        ids = [env.sim.model.joint_name2id(j) for j in gripper.joints]
        return float(np.sum(np.abs(env.sim.data.qpos[[
            env.sim.model.jnt_qposadr[i] for i in ids
        ]])))
    except Exception:  # pragma: no cover - embodiment without named joints
        return float("nan")


def jaw_closure_probe(env, gripper: str):
    """A cross-hand reading of how far the jaws have shut, in ``[0, 1]``.

    ``0`` is fully open and ``1`` is fully closed on air, for every hand, so one
    threshold means the same thing on a Panda and on a Robotiq 2F-140. Built
    from **geom displacement** rather than joint positions, for the reasons in
    :func:`_jaw_opening`: displacement needs no joint names, has one sign by
    construction, and is in metres on every hand.

    The calibration -- which geoms are fingers, the closing axis, and the spread
    at both extremes -- is measured once per hand by
    :mod:`tpgpt.grasp.measure_frames` and cached in ``gripper_frames.json``.
    Here the current spread is projected onto the *live* ``grip_site`` closing
    axis, so the reading is valid with the wrist at any orientation, unlike the
    stationary-arm measurement it is calibrated against.

    Values outside ``[0, 1]`` are **not** clipped, but neither bound identifies
    a grasp. Above 1 means the fingers were pressed past their free-air closed
    pose, which happens when they are pressed onto an object *and* when they are
    simply pressed harder or longer than the 40 settle steps the calibration
    used -- measured at ``closure_max`` 1.04 with ``held_steps`` 0 on an XArm.
    Below 0 means forced wider than the open pose. Both are real states of the
    hand and worth keeping; contact is what ``held_steps`` measures, and closure
    says *how* the jaws got there rather than whether anything was in them.

    Args:
        env: A constructed robosuite environment.
        gripper: Registry short name, e.g. ``"robotiq140"``. Needed because the
            calibration is per hand and there is no way to recover it from the
            environment alone.

    Returns:
        A zero-argument callable returning the closure fraction, or ``nan``
        when this hand has no calibration -- never a plausible default. An
        uncalibrated hand reading ``0.0`` would say "wide open" forever, which
        is exactly the failure mode that
        :func:`tpgpt.grasp.grippers.contact_offset` was changed to refuse.
    """
    from tpgpt.grasp.grippers import gripper_frame

    frame = gripper_frame(gripper) or {}
    names = frame.get("finger_geoms")
    closing = frame.get("closing_in_site")
    span = float(frame.get("spread_open", 0.0)) - float(frame.get("spread_closed", 0.0))
    if not names or closing is None or span <= 1e-6:
        return lambda: float("nan")

    model = env.sim.model
    ids = [model.geom_name2id(n) for n in names if n in model.geom_names]
    grip = env.robots[0].gripper
    grip = grip["right"] if isinstance(grip, dict) else grip
    site = model.site_name2id(grip.important_sites["grip_site"])
    closing = np.asarray(closing, dtype=float)
    spread_open = float(frame["spread_open"])
    if len(ids) < 2:
        return lambda: float("nan")

    def closure() -> float:
        data = env.sim.data
        axis = np.array(data.site_xmat[site]).reshape(3, 3) @ closing
        projected = np.asarray(data.geom_xpos)[ids] @ axis
        spread = float(projected.max() - projected.min())
        return (spread_open - spread) / span

    return closure


def _gripper_touches(env, object_name: str) -> bool:
    """Whether any gripper geom is in contact with the object.

    Contact is the only honest test of a grasp. Proximity is not: a hand can sit
    a centimetre from an object all the way through a carry, which is exactly
    what a failed grasp looks like from the end-effector trace alone.
    """
    try:
        gripper = env.robots[0].gripper
        gripper = gripper["right"] if isinstance(gripper, dict) else gripper
        body = env.object_body_ids[object_name]
        object_geoms = {
            i for i in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[i] == body
        }
        gripper_geoms = {
            env.sim.model.geom_name2id(g)
            for g in gripper.contact_geoms
            if g in env.sim.model.geom_names
        }
        data = env.sim.data
        for i in range(data.ncon):
            contact = data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if pair & object_geoms and pair & gripper_geoms:
                return True
        return False
    except Exception:  # pragma: no cover
        return False


def diagnose(result, env, grasp_point: np.ndarray | None = None) -> Diagnosis:
    """Attribute one run to the stage that failed.

    Args:
        result: A finished :class:`~tpgpt.experiments.pipeline.RunResult`.
        env: The environment it ran in, for the slot pose.
        grasp_point: Where the jaws were meant to close, in world coordinates.
            Defaults to the chosen grasp's own contact point.
    """
    diagnosis = Diagnosis()
    rollout = result.rollout
    diagnosis.succeeded = bool(rollout is not None and rollout.success)
    if rollout is None:
        diagnosis.stages.append(
            Stage("approach", False, "the run never reached the robot")
        )
        return diagnosis

    trace = rollout.metadata.get("probe") or {}
    if not trace:
        diagnosis.stages.append(
            Stage("approach", False, "no object trace was recorded")
        )
        return diagnosis

    # How the *clock* fared, which the stage list cannot show. A run can do
    # every physical step correctly and still be scored a failure because its
    # phase never reached 1.0 and the gripper therefore never opened -- measured
    # on three can runs that delivered the object to within 5-30 mm of the slot
    # and were reported as stalls.
    phases = np.asarray(rollout.time_belief)
    diagnosis.measurements["final_phase"] = float(phases[-1])
    diagnosis.measurements["frozen_steps"] = int(np.sum(np.diff(phases) <= 1e-9))
    diagnosis.measurements["steps"] = int(len(phases))

    z = np.asarray(trace["object_z"])
    held = np.asarray(trace["held"]).astype(bool)
    xy = np.column_stack([trace["object_x"], trace["object_y"]])
    command = np.asarray(rollout.gripper)
    eef = np.asarray(rollout.positions)

    # --- approach: did the arm arrive where the policy starts? ---------------
    gap = float(rollout.metadata.get("approach_gap", 0.0))
    start_error = float(np.linalg.norm(eef[0] - result.transported[0])) \
        if result.transported is not None else float("nan")
    diagnosis.stages.append(Stage(
        "approach", start_error <= 0.08,
        f"started {start_error * 1000:.0f} mm from the transported start "
        f"(it had {gap * 1000:.0f} mm to travel)",
        start_error,
    ))

    # --- reach: where was the hand when the jaws were told to close? ---------
    closing = np.flatnonzero(command > 0)
    if not len(closing):
        diagnosis.stages.append(Stage(
            "reach", False, "the gripper was never commanded to close"
        ))
        diagnosis.measurements["close_step"] = None
        return _pad(diagnosis)
    close_step = int(closing[0])
    diagnosis.measurements["close_step"] = close_step
    diagnosis.measurements["close_phase"] = float(rollout.time_belief[close_step])

    if grasp_point is None and result.grasp is not None:
        from tpgpt.grasp.grasps import contact_offset, grasp_to_eef_pose

        position, rotation = grasp_to_eef_pose(result.grasp, result.gripper)
        grasp_point = position + rotation @ contact_offset(result.gripper)
    measured = _measured_rotations(trace)
    if grasp_point is not None:
        # The hand's own contact point, not its site, is what has to arrive.
        offset = _hand_contact(
            result, eef[close_step],
            measured[close_step] if measured is not None else None,
        )
        reach_error = float(np.linalg.norm(offset - np.asarray(grasp_point)))
    else:
        reach_error = float(np.linalg.norm(eef[close_step, :2] - xy[close_step]))
    diagnosis.measurements["reach_error"] = reach_error

    # Separate the two things a reach error can be. The *attractor* is what the
    # policy commanded; the *position* is where the arm got to. If the commanded
    # point is on the object and the arm is not, that is the controller lagging
    # and the fix is in the rollout. If the commanded point is itself off the
    # object, the transported motion is aimed wrong and the fix is in the map --
    # and no amount of tuning the arm will help.
    if grasp_point is not None:
        attractors = np.asarray(rollout.attractors)
        commanded = float(np.linalg.norm(
            _hand_contact(
                result, attractors[close_step],
                measured[close_step] if measured is not None else None,
            ) - np.asarray(grasp_point)
        ))
        diagnosis.measurements["commanded_error"] = commanded
        diagnosis.measurements["lag_at_close"] = float(
            np.linalg.norm(eef[close_step] - attractors[close_step])
        )
        # And the timing question: does the commanded path ever pass through the
        # grasp point, just not at the step the jaws were told to shut? A path
        # that comes within a few millimetres at some other step is a schedule
        # problem, not an aiming one.
        along = np.linalg.norm(
            np.array([
                _hand_contact(
                    result, a, measured[i] if measured is not None else None
                )
                for i, a in enumerate(attractors)
            ])
            - np.asarray(grasp_point), axis=1,
        )
        diagnosis.measurements["closest_commanded"] = float(along.min())
        diagnosis.measurements["closest_commanded_step"] = int(along.argmin())
        if result.transported is not None:
            planned = np.linalg.norm(
                np.array([_hand_contact(result, p) for p in result.transported])
                - np.asarray(grasp_point), axis=1,
            )
            diagnosis.measurements["closest_transported"] = float(planned.min())
        cause = (
            "the arm did not get there" if commanded <= REACH_TOLERANCE
            else "the motion was aimed there wrongly"
        )
    else:
        cause = ""

    # --- reach, split into the axes that mean different things ---------------
    # The scalar above is kept, but the verdict comes from the closing axis
    # wherever the grasp frame is known: that is the only direction in which
    # being wrong loses the object outright. See reach_axes.
    axes, budget = None, None
    if grasp_point is not None and result.grasp is not None:
        axes = reach_axes(result.grasp, offset - np.asarray(grasp_point))
        budget = closing_budget(result)
        diagnosis.measurements.update(
            reach_closing=axes["closing"],
            reach_approach=axes["approach"],
            reach_signed_approach=axes["signed_approach"],
            reach_jaw=axes["jaw"],
            closing_budget=budget,
        )

    if axes is not None and budget is not None:
        reach_ok = axes["closing"] <= budget
        detail = (
            f"the jaws closed {axes['closing'] * 1000:.0f} mm off the closing "
            f"axis against a {budget * 1000:.0f} mm budget "
            f"({axes['signed_approach'] * 1000:+.0f} mm along the approach, "
            f"{axes['jaw'] * 1000:.0f} mm along the fingers), at step "
            f"{close_step}"
        )
        value = axes["closing"]
    else:
        reach_ok = reach_error <= REACH_TOLERANCE
        detail = (
            f"the jaws closed {reach_error * 1000:.0f} mm from the planned "
            f"grasp point, at step {close_step} (straight-line distance; the "
            f"grasp frame was not available to split it by axis)"
        )
        value = reach_error
    diagnosis.stages.append(Stage(
        "reach", reach_ok, detail + (f" ({cause})" if cause else ""), value,
    ))

    # How far the integrated attractor strayed from the path it was meant to
    # follow -- pointwise, so it cannot come out negative. See attractor_drift.
    diagnosis.measurements.update(attractor_drift(rollout, result.transported))

    # --- grasp: did anything touch the object? -------------------------------
    contact_steps = int(held.sum())
    diagnosis.measurements["contact_steps"] = contact_steps
    first_contact = int(np.flatnonzero(held)[0]) if contact_steps else None
    diagnosis.stages.append(Stage(
        "grasp", contact_steps > 0,
        f"the hand was in contact with the object on {contact_steps} of "
        f"{len(held)} steps"
        + (f", from step {first_contact}" if first_contact is not None else
           " -- it closed on nothing"),
        float(contact_steps),
    ))

    # --- lift: did the object leave the table? -------------------------------
    rise = float(z.max() - z[0])
    diagnosis.measurements["lift_height"] = rise
    diagnosis.stages.append(Stage(
        "lift", rise >= LIFT_HEIGHT,
        f"the object rose {rise * 1000:.0f} mm above where it started",
        rise,
    ))

    # --- carry: was it held for the transit? ---------------------------------
    opening = np.flatnonzero(command[close_step:] < 0)
    release_step = int(close_step + opening[0]) if len(opening) else len(command) - 1
    diagnosis.measurements["release_step"] = release_step
    window = slice(close_step, max(release_step, close_step + 1))
    span = max(1, window.stop - window.start)
    carried = float(held[window].sum()) / span
    diagnosis.measurements["carry_fraction"] = carried
    diagnosis.stages.append(Stage(
        "carry", carried >= CARRY_FRACTION,
        f"it was in the hand for {carried:.0%} of the carry "
        f"(steps {close_step} to {release_step})",
        carried,
    ))

    # --- place: was it let go over the destination? --------------------------
    target = env.slot_poses()[result.slot]
    at_release = float(np.linalg.norm(xy[min(release_step, len(xy) - 1)] - target[:2]))
    diagnosis.measurements["release_error_xy"] = at_release
    diagnosis.stages.append(Stage(
        "place", at_release <= PLACE_TOLERANCE,
        f"it was let go {at_release * 1000:.0f} mm from the middle of the slot",
        at_release,
    ))

    # --- settle: is it actually in the slot? ---------------------------------
    final = float(np.linalg.norm(xy[-1] - target[:2]))
    height_error = float(z[-1] - target[2])
    diagnosis.measurements["final_error_xy"] = final
    diagnosis.stages.append(Stage(
        "settle", bool(rollout.success),
        f"it came to rest {final * 1000:.0f} mm from the slot centre and "
        f"{height_error * 1000:+.0f} mm from its height",
        final,
    ))
    return diagnosis



def reach_axes(grasp, error: np.ndarray) -> dict:
    """Split a reach error into the gripper's own three axes.

    **Why the scalar distance was the wrong instrument.** A grasp is not
    isotropic, and the three directions a hand can be wrong in have almost
    nothing to do with each other:

    * **closing** -- along the line the jaws travel. This is the one that
      decides the task. The object has to end up *between* the fingers, and once
      the error exceeds the room left inside the open jaws there is no partial
      credit: the hand closes on air. Budget is millimetres to a few
      centimetres.
    * **approach** -- along the direction the hand advances. Being short or deep
      along this axis mostly changes *where up the object* it is gripped, which
      it often tolerates. Section 7.2 measured 120-135 mm of tolerance here for
      the parallel jaws (and only 15 mm for the UMI, which is unusual in this as
      in much else).
    * **jaw** -- the remaining lateral direction, ``approach x closing``, along
      the length of the fingers. On a body with any symmetry about the closing
      axis -- a can, a bottle -- sliding along this is nearly free.

    Adding these three in quadrature, which is what a straight-line distance
    does, mixes a millimetre-scale budget with a 130 mm one. A run 60 mm deep
    and perfectly centred scores identically to one 60 mm off-centre, and the
    first succeeds while the second cannot. Measured on the campaigns since
    deleted, this scalar rated the *better* keypoint configuration worse
    (section 7.26), which is the clearest possible sign of a broken metric.

    Args:
        grasp: The chosen :class:`~tpgpt.grasp.grasps.Grasp6D`, whose ``closing``
            and ``approach`` properties define the frame.
        error: ``(3,)`` world-frame vector from the intended grasp point to
            where the hand actually was.

    Returns:
        Unsigned magnitude along each axis, plus ``signed_approach``, which is
        positive when the hand stopped **short** of the object and negative when
        it drove past. Short and deep fail in different ways and it is worth not
        throwing the sign away.
    """
    error = np.asarray(error, dtype=float).reshape(3)
    closing = np.asarray(grasp.closing, dtype=float)
    approach = np.asarray(grasp.approach, dtype=float)
    jaw = np.cross(approach, closing)
    norm = np.linalg.norm(jaw)
    jaw = jaw / norm if norm > 1e-12 else np.zeros(3)
    along = float(error @ approach)
    return {
        "closing": abs(float(error @ closing)),
        "approach": abs(along),
        "signed_approach": -along,
        "jaw": abs(float(error @ jaw)),
    }


def closing_budget(result, fraction: float = CLOSING_BUDGET_FRACTION) -> float | None:
    """How far off the closing axis the hand may be and still capture the object.

    An open jaw of aperture ``a`` closing on an object of width ``w`` leaves
    ``(a - w) / 2`` of room on each side. That is the budget, and it is exact:
    beyond it the object is outside one finger before the jaws start moving.

    The object's width is measured **along this grasp's own closing axis**, from
    the target keypoints -- which are a box fitted to the object's cloud, so
    their extent along that axis is the width the jaws actually meet. Taking an
    axis-aligned bounding box instead is a known trap: a 30 x 100 mm box yawed
    45 degrees measures 92 x 92 and reads as ungraspable (section 7.18).

    The width comes from ``metrics["object_width_closing"]``, recorded from the
    **cloud** where the object's size is actually known, and falls back to the
    target keypoints' **pick block alone**.

    Two corrections are behind that, both measured:

    * A first version took the extent over all of ``target_keypoints``, which
      holds the placed block as well, so the "width" was the pick-to-place
      distance (239-277 mm) and the budget clamped to 0.0 for every object and
      every hand. A zero budget is a plausible number, which makes it worse
      than no number.
    * With ``box="grasp_cube"`` the box points are a fixed cube on the grasp,
      so their extent is 40.0 mm for every object -- the cube's own size. That
      is the construction working as intended: removing the object's size from
      the keypoints is the whole point of it. But it means the width cannot be
      read back out of them, so this **returns ``None``** for a cube set with
      no recorded cloud width rather than reporting the cube.

    Returns:
        The budget in metres, or ``None`` when neither the gripper geometry nor
        the keypoints are available -- never a plausible-looking default. A
        missing measurement that returns a number is how a 41 mm frame error
        read as the arm missing its target for weeks (section 7.13).
    """
    if result.grasp is None or not result.gripper:
        return None
    try:
        from tpgpt.grasp.grippers import gripper_geometry, resolve_pair

        # Resolve through the registry rather than looking the hand up by
        # identity. ``gripper_geometry`` wants the GraspGen-X name, so a
        # registry key such as ``"panda"`` raises and the budget silently
        # disappears. That is the shape of the bug in section 7.13, where a
        # by-identity lookup turned an unmeasured hand into a confident zero.
        name = resolve_pair(result.grasp.gripper).graspgen
        aperture = float(gripper_geometry(name).aperture)
    except Exception:  # pragma: no cover - absent sibling checkout
        return None

    # **From the cloud when it is recorded, because the keypoints may not know
    # the object's size any more.** With ``box="grasp_cube"`` the nine box
    # points are a *fixed* cube on the grasp -- deliberately, since that is what
    # removes the object-size volume scaling -- so their extent along the closing
    # axis is the cube's own 2 x 20 mm for every object alike. Measured across
    # nine hands and five objects:
    #
    # ==========================  ==========================================
    # construction                width read from the keypoints (mm)
    # ==========================  ==========================================
    # cloud box                   bread 43.6, can 45.2, cereal 46.2,
    #                             lemon 26.4, milk 60.4  -- the real widths
    # grasp cube                  **40.0 for every object**  -- the cube
    # ==========================  ==========================================
    #
    # So the cube construction erases from the keypoints the very quantity this
    # function needs, and that is a consequence of the construction working as
    # intended rather than a defect in it. The width has to come from the cloud.
    width = None
    metrics = getattr(result, "metrics", None) or {}
    recorded = metrics.get("object_width_closing")
    if recorded is not None and np.isfinite(recorded):
        width = float(recorded)

    if width is None:
        # **The pick block only.** ``scene_keypoints`` emits two blocks -- the
        # object where it is picked and the same object where it is placed --
        # and ``_select_parts`` keeps both, so ``target_keypoints.points`` spans
        # the whole pick-to-place distance: 239 to 277 mm on the real objects,
        # against an aperture of at most 125 mm, so the budget clamped to 0.0
        # for every object and every hand. Zero is a *plausible* number and
        # therefore worse than ``None``: indistinguishable from a genuinely
        # impossible grasp.
        points = getattr(result.target_keypoints, "points", None)
        labels = getattr(result.target_keypoints, "labels", None)
        if points is None or len(points) < 2:
            return fraction * aperture
        points = np.asarray(points, dtype=float)
        if labels is not None and len(labels) == len(points):
            pick = [i for i, label in enumerate(labels) if label.startswith("pick_")]
            if len(pick) >= 2:
                points = points[pick]

        # Refuse if these keypoints are a fixed cube. Their extent is the
        # cube's, not the object's, and returning it would give the same budget
        # for a lemon and a milk carton -- a plausible number that is simply
        # not a measurement of this object.
        if _is_grasp_cube(result.target_keypoints):
            return None
        width = float(np.ptp(points @ result.grasp.closing))

    return max(0.5 * (aperture - width), 0.0)


def _is_grasp_cube(keypoints) -> bool:
    """Whether a keypoint set's box block is a fixed cube rather than a fitted box.

    Read from the metadata ``block_kind`` that ``object_keypoints`` records, so
    it is what the construction *said* it did rather than inferred from the
    numbers. Any block being a cube is enough: the pick block is the one whose
    extent would be misread, and both blocks always share a mode.
    """
    metadata = getattr(keypoints, "metadata", None) or {}
    for block in metadata.values():
        if isinstance(block, dict) and block.get("block_kind") == "grasp_cube":
            return True
    return False


def attractor_drift(rollout, transported) -> dict:
    """How far the integrated attractor strayed from the path it should follow.

    The rollout advances ``attractor += velocity * dt * gate`` and never
    consults the transported labels again, so nothing prevents the integral from
    wandering. This measures whether it did.

    **Measured pointwise against the path, not as the difference of two
    minima.** The earlier diagnostic compared how close the attractor came to
    the grasp against how close the *plan* came to the grasp, and called the
    difference drift. That is not a drift measurement: the two minima can occur
    at different moments, it goes negative whenever the attractor happens to cut
    a corner closer, and it is dominated by whichever curve was sampled more
    finely. Numbers derived that way are withdrawn (section 7.26).

    Returns two complementary quantities, because "off the path" and "late" are
    different faults with different fixes:

    * ``deviation`` -- distance to the nearest point on the transported path,
      ignoring timing. This is integration drift proper, and a fix belongs in
      policy execution.
    * ``frechet`` -- order-preserving curve distance, which does not let the two
      curves be matched out of sequence.

    Returns ``{}`` when either curve is missing, so a caller can tell "not
    measured" from "measured as zero".
    """
    from tpgpt.metrics.curves import frechet_distance, path_deviation

    attractors = np.asarray(getattr(rollout, "attractors", []), dtype=float)
    path = np.asarray(transported, dtype=float) if transported is not None else None
    if path is None or attractors.ndim != 2 or len(attractors) == 0 or len(path) == 0:
        return {}
    stats = path_deviation(attractors, path)
    return {
        "drift_max": stats["max"],
        "drift_median": stats["median"],
        "drift_at_step": stats["argmax"],
        "drift_frechet": frechet_distance(attractors, path),
    }


def unreachable_segments(replay, labels, window: int = 12) -> dict:
    """Which *part* of a path the arm could not hold, not just how much of it.

    A reachable fraction is an aggregate: it says a quarter of a trajectory is
    unreachable and cannot say whether that quarter is the descent onto the
    object, the lift, the transit or the insertion. Those are different problems
    with different fixes -- an unreachable grasp is a grasp-selection failure, an
    unreachable transit is a path-shape failure -- and 7.25 measured the aggregate
    version of this while leaving the location open.

    Segments are cut from the demonstration's own gripper channel rather than by
    fraction, so they mean the same thing on trajectories of different lengths:

    * ``approach``  -- start until ``window`` labels before the jaws close
    * ``grasp``     -- the ``window`` labels either side of closing
    * ``carry``     -- between the grasp and the release windows
    * ``place``     -- the ``window`` labels either side of opening
    * ``retreat``   -- after the release window

    Args:
        replay: A :class:`~tpgpt.sim.replay.ReplayResult` carrying
            ``metadata["reachable_per_waypoint"]``.
        labels: The label set that was replayed, for its gripper channel.
        window: Half-width of the grasp and place windows, in labels.

    Returns:
        ``{"<segment>_unreachable": fraction, "<segment>_track_max": metres}``
        per segment, plus ``"worst_segment"``. Empty when the per-waypoint trace
        is missing -- never a plausible-looking zero.
    """
    meta = getattr(replay, "metadata", None) or {}
    reachable = meta.get("reachable_per_waypoint")
    tracking = meta.get("tracking_error_per_waypoint")
    if reachable is None or tracking is None or not len(reachable):
        return {}
    reachable = np.asarray(reachable, dtype=bool)
    tracking = np.asarray(tracking, dtype=float)
    n = len(reachable)
    try:
        close, release = carry_indices(labels)
    except Exception:  # no gripper channel: fall back to thirds
        close, release = n // 3, 2 * n // 3

    bounds = {
        "approach": (0, max(0, close - window)),
        "grasp": (max(0, close - window), min(n, close + window)),
        "carry": (min(n, close + window), max(0, release - window)),
        "place": (max(0, release - window), min(n, release + window)),
        "retreat": (min(n, release + window), n),
    }
    out: dict = {}
    worst, worst_share = None, -1.0
    for name, (lo, hi) in bounds.items():
        lo, hi = int(np.clip(lo, 0, n)), int(np.clip(hi, 0, n))
        if hi <= lo:
            continue
        share = float(1.0 - reachable[lo:hi].mean())
        out[f"{name}_unreachable"] = share
        out[f"{name}_track_max"] = float(tracking[lo:hi].max())
        if share > worst_share:
            worst, worst_share = name, share
    out["worst_segment"] = worst
    return out


def grasp_slip(rollout, positions=None) -> dict:
    """How far the object moved *relative to the hand* while it was held.

    A grasp that holds is one where the object-to-fingertip vector stays put.
    Once the jaws close the attachment is meant to be rigid -- ``carry_transform``
    assumes exactly that when it derives the placed pose from the trajectory --
    so any drift in that vector is the object sliding, rolling or being squeezed
    out.

    This matters for a tilted grasp in particular: a hand rotated away from the
    demonstration's holds the object against a different component of gravity,
    and whether that still holds is a physics question no amount of map geometry
    can answer.

    Measured against the vector at **first contact** rather than against the
    commanded pose, so it is drift in the grasp and not tracking error.

    Args:
        rollout: Anything carrying ``metadata["probe"]`` with ``object_x/y/z``
            and ``held`` channels -- a :class:`~tpgpt.sim.rollout.SimRollout` or
            a :class:`~tpgpt.sim.replay.ReplayResult`.
        positions: ``(N, 3)`` measured hand positions. Defaults to
            ``rollout.positions``.

    Returns:
        ``{"slip_max", "slip_final", "slip_at_step", "held_steps"}`` in metres,
        or ``{}`` when the trace or the contact channel is missing -- never a
        plausible-looking zero, which is the recurring lesson of 7.13.
    """
    trace = (getattr(rollout, "metadata", None) or {}).get("probe") or {}
    if not {"object_x", "object_y", "object_z", "held"} <= set(trace):
        return {}
    hand = np.asarray(positions if positions is not None else rollout.positions, dtype=float)
    obj = np.column_stack([trace["object_x"], trace["object_y"], trace["object_z"]])
    held = np.asarray(trace["held"]).astype(bool)
    n = min(len(hand), len(obj), len(held))
    hand, obj, held = hand[:n], obj[:n], held[:n]
    if not held.any():
        return {"slip_max": float("nan"), "slip_final": float("nan"),
                "slip_at_step": None, "held_steps": 0}
    grip = obj[held] - hand[held]
    drift = np.linalg.norm(grip - grip[0], axis=1)
    return {
        "slip_max": float(drift.max()),
        "slip_final": float(drift[-1]),
        "slip_at_step": int(np.flatnonzero(held)[int(drift.argmax())]),
        "held_steps": int(held.sum()),
    }


def _hand_contact(result, position: np.ndarray, rotation=None) -> np.ndarray:
    """The point between the fingertips, given a recorded end-effector position.

    A grasp pose is at the gripper base; the jaws close a hand-specific depth
    further along. Comparing the wrist site to the grasp point instead reports
    the fingertip depth -- 103 to 195 mm across the registry -- as a reach error.

    **Whether the offset has to be added at all depends on the frame the rollout
    ran in.** With a ``tool_offset`` the rollout already records fingertip
    positions, and adding the offset a second time puts the measurement 41 mm
    out -- enough that three runs which delivered their object into the slot
    were reported as having failed to reach it, in the same table that reported
    them successful. The rollout says which frame it used, so this asks.

    Args:
        rotation: The hand's orientation at that instant. Falls back to the
            planned grasp rotation, which is only right where the hand is
            already holding the planned pose.
    """
    from tpgpt.grasp.grasps import contact_offset, grasp_to_eef_pose

    if result.grasp is None or _already_at_the_tool(result):
        return np.asarray(position)
    if rotation is None:
        _, rotation = grasp_to_eef_pose(result.grasp, result.gripper)
    return np.asarray(position) + np.asarray(rotation) @ contact_offset(
        result.gripper
    )


def _already_at_the_tool(result) -> bool:
    """Whether the rollout recorded fingertip positions rather than wrist ones."""
    rollout = getattr(result, "rollout", None)
    offset = (rollout.metadata.get("tool_offset") if rollout is not None else None)
    return offset is not None and float(np.linalg.norm(offset)) > 1e-9


def _measured_rotations(trace: dict):
    """``(N, 3, 3)`` hand rotations from a probe trace, or ``None`` if absent."""
    if "eef_R00" not in trace:
        return None
    return np.stack(
        [np.asarray(trace[f"eef_R{i}{j}"]) for i in range(3) for j in range(3)],
        axis=-1,
    ).reshape(-1, 3, 3)


def _pad(diagnosis: Diagnosis) -> Diagnosis:
    """Mark the stages after a fatal one as not reached, rather than omitting.

    An absent stage in a table reads as "passed"; "not reached" is the truth and
    the difference matters when the rows are counted up across a campaign.
    """
    done = {s.name for s in diagnosis.stages}
    for name in STAGES:
        if name not in done:
            diagnosis.stages.append(Stage(name, False, "not reached"))
    return diagnosis


def tally(diagnoses: list[Diagnosis]) -> dict:
    """How many runs got past each stage. The headline of a campaign.

    A run that succeeded counts as having passed every stage, whatever the
    proxies said, so the funnel can never report fewer runs finishing than
    succeeded.
    """
    counts = {name: 0 for name in STAGES}
    for diagnosis in diagnoses:
        if diagnosis.succeeded:
            for name in STAGES:
                counts[name] += 1
            continue
        for stage in diagnosis.stages:
            if stage.ok:
                counts[stage.name] += 1
            else:
                break
    return counts


def tally_text(counts: dict, total: int) -> str:
    lines = [f"{'stage':<10}{'passed':>8}  what it means"]
    for name in STAGES:
        lines.append(f"{name:<10}{counts[name]:>4}/{total:<3}  {STAGE_TEXT[name]}")
    return "\n".join(lines)
