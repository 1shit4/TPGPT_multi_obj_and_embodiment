"""Refitting the policy on transported labels (paper Sec. III-B, step 2).

The paper's pipeline is two-stage. Stage 1 transports the labels
``(X, Xdot, R, K, D)`` into the target space. Stage 2 -- this module -- fits a
*state-dependent policy* ``g`` to those transported labels. That second stage is
what makes the method a policy generalisation rather than trajectory reshaping,
and the paper is emphatic (Secs. I and II) that replaying a warped trajectory is
not an acceptable substitute.

The regressor is deliberately interchangeable ("agnostic to the method used for
policy learning, e.g. DMP, KMP, GP, NN, GMM, LPV"). A Gaussian Process is the
default because Sec. III-H needs its predictive variance: that variance is
exactly ``Sigma_f_hat``, the epistemic term of Eq. (13).

Input is position **and** a belief of time, per Sec. V and ref. [12]. A policy
of position alone cannot represent a path that revisits a position, which every
pick-and-place does.

The phase is a *belief*, not a clock: it is propagated by the policy's own
predicted rate and then reconciled with where the robot actually is, so that a
lagging robot does not let the phase run away from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpgpt.policy.orientation import rotation_from_6d, rotation_to_6d
from tpgpt.policy.spd import log_cholesky_to_spd, spd_to_log_cholesky
from tpgpt.transport.gp import GaussianProcessRegressor
from tpgpt.transport.labels import PolicyLabels


@dataclass
class PolicyPrediction:
    """What the policy returns at a queried state."""

    velocity: np.ndarray                    # (n, 3)
    velocity_std: np.ndarray                # (n, 3) -- Sigma_f_hat of Eq. (13)
    reference: np.ndarray | None = None     # (n, 3) regressed attractor position
    orientation: np.ndarray | None = None   # (n, 3, 3)
    stiffness: np.ndarray | None = None     # (n, 3, 3)
    damping: np.ndarray | None = None       # (n, 3, 3)
    gripper: np.ndarray | None = None       # (n,)
    time_rate: np.ndarray | None = None     # (n,)

    def __len__(self) -> int:
        return self.velocity.shape[0]


class GPPolicy:
    """Gaussian-Process policy ``g`` fitted on a set of policy labels.

    All output channels share one kernel and therefore one Cholesky factor, as
    Appendix A prescribes. Because the channels have wildly different units
    (m/s against log-Cholesky stiffness against a unit-interval phase), each is
    standardised to unit variance before the shared fit and restored afterwards.

    Channel offsets encode the right prior behaviour away from the data:

    * **Velocity gets a zero prior mean**, which Appendix A calls for explicitly
      -- "it is safer to have a zero mean prior, such that the robot does not
      attempt to do any movement if there is no significant evidence from the
      human demonstration."
    * Every other channel is regressed as a deviation from its label mean, so
      far from the demonstration the policy falls back to the average commanded
      orientation, stiffness, damping and gripper state rather than to zero
      (a zero stiffness matrix would make the robot limp).

    Args:
        use_time_belief: Include the phase in the policy input.
        regressor_kwargs: Forwarded to
            :class:`~tpgpt.transport.gp.GaussianProcessRegressor`.
    """

    #: Channels that keep a zero prior mean (Appendix A).
    #:
    #: **The gripper belongs here and its absence was a bug.** Appendix A's
    #: argument for the velocity channel is that away from the data the safe
    #: answer is "do not move"; the same argument applies verbatim to the jaws,
    #: where the safe answer is "do not close". Centred on its label mean
    #: instead, the gripper channel reverts off-manifold to the fraction of the
    #: demonstration spent holding the object -- which on the reshelving
    #: demonstration is **+0.1**, and ``rollout_policy`` reads anything above
    #: zero as *shut*.
    #:
    #: Measured on the source policy: query it at phase 0.000 with a position
    #: taken from anywhere past phase 0.10 and it returns +0.099 to +0.100 --
    #: the prior, meaning "close" -- at every one. In the real-world campaign
    #: that closed the jaws of ``robotiq3f/mug/P1`` at **step 5**, 261 mm from
    #: the grasp, and six cells closed before phase 0.10 against a
    #: demonstration that closes at 0.250. All six failed.
    #:
    #: This is not the lag gate, which works: the gate holds the *phase* back
    #: when the arm falls behind, and the executed close phase is 0.247 to 0.251
    #: on 204 of 210 cells. The gripper command is not a function of phase
    #: alone -- it is a GP over position *and* phase, and off-manifold in
    #: **position** the phase gate cannot help it.
    ZERO_MEAN_CHANNELS = ("velocity", "gripper")

    #: Length-scale bounds in standardised input units.
    #:
    #: A policy needs *different* kernel priors from a transportation map, and
    #: getting this wrong is silent and fatal. The map interpolates sparse
    #: keypoints that must each be matched exactly, so its length scale should
    #: track the keypoint spacing. A policy is a **field**: it has to command a
    #: sensible motion at states the demonstration never visited, because the
    #: robot starts wherever it starts and the transported labels start
    #: somewhere else entirely.
    #:
    #: Deriving the bounds from the minimum spacing between labels -- which is
    #: what the geometric heuristic does -- measures how finely the trajectory
    #: was sampled, not the structure of the field. On a 200-label, 20 Hz
    #: demonstration the labels are 0.3 mm apart, the fitted length scale
    #: collapses to 0.025, and the commanded velocity is *exactly zero* 2 cm
    #: off the path. The policy then has no basin of attraction at all and the
    #: robot never moves.
    #:
    #: Inputs are standardised, so these are in units of input standard
    #: deviation: the lower bound keeps a basin of roughly a fifth of the
    #: demonstration's extent.
    LENGTH_SCALE_BOUNDS = (0.4, 3.0)

    #: Likelihood noise, relative to unit-variance standardised targets.
    #:
    #: This is the second place a policy needs different priors from a
    #: transportation map. The map must interpolate its keypoints exactly
    #: (property (i) of Sec. III-C), so its noise is numerical jitter. A policy
    #: must **not** interpolate: a 20 Hz demonstration puts its labels a few
    #: millimetres apart along a curve, an order of magnitude closer than the
    #: kernel length scale, so the Gram matrix becomes near-singular (measured
    #: condition number 5e9), the weights blow up with alternating signs, and
    #: the field is fine *on* the demonstration but explodes just off it.
    #:
    #: Measured on the reshelving demonstration, whose peak speed is 0.25 m/s,
    #: the largest speed commanded within 4 cm of the path was:
    #:
    #: ===============  ==========  ==============
    #: noise            cond(K)     max speed
    #: ===============  ==========  ==============
    #: 1e-6             4.9e9       13.4 m/s
    #: 1e-4             4.9e7       3.1 m/s
    #: 1e-2 (default)   8.8e3       0.69 m/s
    #: ===============  ==========  ==============
    #:
    #: At 1e-6 the arm was flung off the path within ten control steps.
    NOISE_VARIANCE = 1e-2

    #: Extra weight on the phase input, relative to the standardised position.
    #:
    #: Position and phase are not the same kind of input and one isotropic
    #: length scale cannot serve both. A pick-and-place path passes through the
    #: hover pose **twice** -- once descending to insert, once ascending to
    #: retreat -- at nearly the same position but very different phases. If the
    #: kernel cannot separate those two branches it blends them, the descent and
    #: the ascent cancel, and the rollout stalls at the hover pose with a
    #: commanded speed of 0.01 m/s.
    #:
    #: Weighting the phase axis is equivalent to giving the kernel a shorter
    #: length scale along it while keeping the long positional length scale that
    #: provides the basin of attraction.
    PHASE_WEIGHT = 8.0

    def __init__(
        self,
        use_time_belief: bool = True,
        phase_weight: float = PHASE_WEIGHT,
        **regressor_kwargs,
    ):
        self.use_time_belief = use_time_belief
        self.phase_weight = float(phase_weight)
        regressor_kwargs.setdefault("noise_variance", self.NOISE_VARIANCE)
        regressor_kwargs.setdefault("length_scale_bounds", self.LENGTH_SCALE_BOUNDS)
        self.regressor_kwargs = regressor_kwargs
        self.gp: GaussianProcessRegressor | None = None
        self._layout: list[tuple[str, int, int]] = []
        self._offset: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._input_mean: np.ndarray | None = None
        self._input_scale: np.ndarray | None = None
        self.labels: PolicyLabels | None = None
        self._train_inputs: np.ndarray | None = None

    # ------------------------------------------------------------- inputs
    def _make_inputs(self, positions: np.ndarray, time_belief=None) -> np.ndarray:
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if not self.use_time_belief:
            return positions
        if time_belief is None:
            raise ValueError(
                "this policy was fitted with a time belief; pass time_belief to predict()"
            )
        t = np.asarray(time_belief, dtype=float).reshape(-1, 1)
        if t.shape[0] != positions.shape[0]:
            raise ValueError(
                f"time_belief has {t.shape[0]} entries but positions has {positions.shape[0]}"
            )
        return np.hstack([positions, t])

    def _normalize_inputs(self, Z: np.ndarray) -> np.ndarray:
        """Standardise inputs so one isotropic kernel spans position and phase.

        Position is in metres and the phase is a unit interval; a single length
        scale over the raw concatenation would be dominated by whichever has the
        larger spread. The phase axis then carries the extra
        :data:`PHASE_WEIGHT` so the kernel can separate branches of a
        self-intersecting path.
        """
        standardised = (Z - self._input_mean) / self._input_scale
        if self.use_time_belief:
            standardised = standardised.copy()
            standardised[:, -1] *= self.phase_weight
        return standardised

    # ---------------------------------------------------------------- fit
    def fit(self, labels: PolicyLabels) -> "GPPolicy":
        """Fit ``g`` on a label set (typically the transported one)."""
        if labels.velocities is None:
            raise ValueError("velocity labels are required to fit a policy")
        self.labels = labels

        Z = self._make_inputs(labels.positions, labels.time_belief)
        self._input_mean = Z.mean(axis=0)
        self._input_scale = np.where(Z.std(axis=0) > 1e-9, Z.std(axis=0), 1.0)

        # The attractor channel. Sec. V states that the policy learns "the
        # desired attractor position ... as a function of the current
        # position-time input", and that absolute reference is what gives the
        # policy a restoring action. A velocity field alone has none: fitted
        # from demonstration velocities with a zero-mean prior, at an unvisited
        # state it blends the velocities of nearby labels, which points *along*
        # the demonstration and never back towards it. Measured on reshelving,
        # a velocity-only policy drifted 8 cm off the transported path within 25
        # control steps and then crawled, never completing the task.
        blocks: list[tuple[str, np.ndarray]] = [
            ("velocity", labels.velocities),
            ("attractor", labels.positions),
        ]
        if labels.orientations is not None:
            blocks.append(("orientation", rotation_to_6d(labels.orientations)))
        if labels.stiffness is not None:
            blocks.append(("stiffness", spd_to_log_cholesky(labels.stiffness)))
        if labels.damping is not None:
            blocks.append(("damping", spd_to_log_cholesky(labels.damping)))
        if labels.gripper is not None:
            blocks.append(("gripper", labels.gripper[:, None]))
        if labels.time_rate is not None:
            blocks.append(("time_rate", labels.time_rate[:, None]))

        Y, self._layout, offsets, scales = [], [], [], []
        cursor = 0
        for name, block in blocks:
            block = np.asarray(block, dtype=float)
            offset = (
                np.zeros(block.shape[1])
                if name in self.ZERO_MEAN_CHANNELS
                else block.mean(axis=0)
            )
            centred = block - offset
            scale = np.maximum(centred.std(axis=0), 1e-9)
            Y.append(centred / scale)
            self._layout.append((name, cursor, cursor + block.shape[1]))
            offsets.append(offset)
            scales.append(scale)
            cursor += block.shape[1]

        self._offset = np.concatenate(offsets)
        self._scale = np.concatenate(scales)
        self._train_inputs = self._normalize_inputs(Z)
        self.gp = GaussianProcessRegressor(**self.regressor_kwargs).fit(
            self._train_inputs, np.hstack(Y)
        )
        return self

    # ------------------------------------------------------- time belief
    def update_time_belief(
        self,
        position: np.ndarray,
        time_belief: float,
        dt: float,
        correction: float = 0.2,
    ) -> float:
        """Advance the phase and reconcile it with the observed position.

        Sec. V lists the "time belief update" among the quantities the policy
        learns, and calling it a *belief* rather than a clock is the point: it
        has to be reconciled with where the robot actually got to. Any lag --
        Euler integration error, GP smoothing at a corner, or the robot's own
        dynamics -- moves the query ``(x, t)`` off the ridge ``{(x(s), s)}``
        that the policy was trained on, and the further off that ridge the query
        drifts, the less the prediction means.

        The correction is a nearest-label update in the *joint* position-phase
        space, using the standardised metric the kernel already uses. The joint
        metric matters: matching on position alone is ambiguous for exactly the
        paths that need a phase at all, since a pick-and-place revisits the same
        position twice. The current belief breaks that tie.

        Note:
            This is a consistency mechanism, not a cure for a badly conditioned
            policy. Measured on the reshelving demonstration, an
            under-regularised policy (``noise_variance`` 1e-6) fails to complete
            the task at *any* correction gain -- and the correction makes it
            worse, not better, because it keeps re-anchoring the phase to a
            robot that is not moving. Regularisation is what makes the rollout
            work; see :data:`NOISE_VARIANCE`.

        Args:
            position: ``(3,)`` current measured position.
            time_belief: Current phase.
            dt: Time step.
            correction: Blend towards the observed phase, in ``[0, 1]``.
                ``0`` reproduces the open-loop clock; ``1`` discards the
                feed-forward term entirely.

        Returns:
            The updated phase.
        """
        self._check_fitted()
        if not self.use_time_belief:
            return time_belief

        pred = self.predict(np.atleast_2d(position), np.array([time_belief]))
        rate = float(pred.time_rate[0]) if pred.time_rate is not None else 1.0
        predicted = time_belief + max(rate, 0.0) * dt
        if correction <= 0.0:
            return predicted

        query = self._normalize_inputs(
            self._make_inputs(np.atleast_2d(position), np.array([predicted]))
        )
        nearest = int(np.argmin(((self._train_inputs - query) ** 2).sum(axis=1)))
        observed = float(self.labels.time_belief[nearest])
        return (1.0 - correction) * predicted + correction * observed

    # ------------------------------------------------------------ predict
    def predict(self, positions: np.ndarray, time_belief=None) -> PolicyPrediction:
        """Query the policy.

        Returns:
            A :class:`PolicyPrediction`. ``velocity_std`` is the epistemic
            uncertainty ``Sigma_f_hat`` that Eq. (13) adds to the transport
            uncertainty.
        """
        self._check_fitted()
        Z = self._normalize_inputs(self._make_inputs(positions, time_belief))
        mean_scaled, std_scaled = self.gp.predict(Z, return_std=True)
        mean = mean_scaled * self._scale + self._offset

        out: dict[str, np.ndarray] = {}
        for name, lo, hi in self._layout:
            out[name] = mean[:, lo:hi]

        velocity = out["velocity"]
        v_lo, v_hi = next((lo, hi) for n, lo, hi in self._layout if n == "velocity")
        velocity_std = std_scaled[:, None] * self._scale[v_lo:v_hi]

        return PolicyPrediction(
            velocity=velocity,
            reference=out.get("attractor"),
            velocity_std=velocity_std,
            orientation=(
                rotation_from_6d(out["orientation"]) if "orientation" in out else None
            ),
            stiffness=(
                log_cholesky_to_spd(out["stiffness"]) if "stiffness" in out else None
            ),
            damping=log_cholesky_to_spd(out["damping"]) if "damping" in out else None,
            gripper=out["gripper"][:, 0] if "gripper" in out else None,
            time_rate=out["time_rate"][:, 0] if "time_rate" in out else None,
        )

    def attractor(self, positions: np.ndarray, time_belief=None) -> np.ndarray:
        """Impedance attractor: a regressed reference plus a velocity feed-forward.

        The controller of Sec. V is a Cartesian impedance controller, commanded
        with an attractor rather than a velocity, so the command has two parts.

        The **reference** is the regressed attractor position of Sec. V. It is
        an absolute position on the demonstrated path, so it pulls the robot
        back when it drifts. Without it the policy has no restoring action at
        all -- see the note in :meth:`fit`.

        The **feed-forward** sets the speed along the path. At steady state a
        commanded offset ``dx`` produces ``K dx`` of force against ``D xdot`` of
        damping, so realising ``xdot`` needs ``dx = K^-1 D xdot``. With the
        transported ``K_hat`` and ``D_hat``, this reproduces the demonstrated
        speed profile under the demonstrated impedance.

        On the demonstration the reference equals the current position and the
        command reduces to ``x + K^-1 D xdot``.
        """
        pred = self.predict(positions, time_belief)
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        reference = pred.reference if pred.reference is not None else positions

        if pred.stiffness is None or pred.damping is None:
            return reference + pred.velocity
        feedforward = np.einsum(
            "nij,njk,nk->ni", np.linalg.inv(pred.stiffness), pred.damping, pred.velocity
        )
        return reference + feedforward

    def _check_fitted(self) -> None:
        if self.gp is None:
            raise RuntimeError("GPPolicy.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.gp is None:
            return "GPPolicy(unfitted)"
        channels = [name for name, _, _ in self._layout]
        return f"GPPolicy(M={len(self.labels)}, channels={channels})"
