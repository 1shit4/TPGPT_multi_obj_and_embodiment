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
predicted rate and then corrected against where the robot actually is. Running
it open loop does not work, and fails in a specific and instructive way -- see
:meth:`GPPolicy.update_time_belief`.
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
    ZERO_MEAN_CHANNELS = ("velocity",)

    def __init__(self, use_time_belief: bool = True, **regressor_kwargs):
        self.use_time_belief = use_time_belief
        regressor_kwargs.setdefault("noise_variance", 1e-6)
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
        larger spread.
        """
        return (Z - self._input_mean) / self._input_scale

    # ---------------------------------------------------------------- fit
    def fit(self, labels: PolicyLabels) -> "GPPolicy":
        """Fit ``g`` on a label set (typically the transported one)."""
        if labels.velocities is None:
            raise ValueError("velocity labels are required to fit a policy")
        self.labels = labels

        Z = self._make_inputs(labels.positions, labels.time_belief)
        self._input_mean = Z.mean(axis=0)
        self._input_scale = np.where(Z.std(axis=0) > 1e-9, Z.std(axis=0), 1.0)

        blocks: list[tuple[str, np.ndarray]] = [("velocity", labels.velocities)]
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
        """Advance the phase and correct it against the observed position.

        Sec. V lists the "time belief update" among the quantities the policy
        learns, and calling it a *belief* rather than a clock is the point: it
        has to be reconciled with where the robot actually got to.

        Integrating the predicted rate open loop fails badly. Any lag -- from
        Euler integration, from GP smoothing at a corner, or from the robot's
        own dynamics -- moves the query ``(x, t)`` off the ridge
        ``{(x(s), s)}`` that the policy was trained on. There the zero-mean
        prior of Appendix A takes over, the predicted velocity decays towards
        zero, the lag grows, and the rollout stalls outright. Measured on a
        synthetic pick-and-place: a 9 mm lag at 60 steps collapsed the speed
        from 0.084 m/s to 0.0002 m/s by step 80 and the rollout stopped 23 cm
        short of the goal.

        The correction is a nearest-label update in the *joint* position-phase
        space, using the standardised metric the kernel already uses. The joint
        metric matters: matching on position alone is ambiguous for exactly the
        paths that need a phase at all, since a pick-and-place revisits the same
        position twice. The current belief breaks that tie.

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
        """Impedance attractor that realises the predicted velocity.

        The controller of Sec. V is a Cartesian impedance controller, which is
        commanded with an attractor rather than a velocity. At steady state a
        commanded offset ``dx`` produces ``K dx`` of force against ``D xdot`` of
        damping, so tracking ``xdot`` requires

            ``x_desired = x + K^-1 D xdot``.

        With the transported ``K_hat`` and ``D_hat`` this reproduces the
        demonstrated speed profile under the demonstrated impedance. Falls back
        to a unit time constant when no stiffness or damping was learned.
        """
        pred = self.predict(positions, time_belief)
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if pred.stiffness is None or pred.damping is None:
            return positions + pred.velocity
        offset = np.einsum(
            "nij,njk,nk->ni", np.linalg.inv(pred.stiffness), pred.damping, pred.velocity
        )
        return positions + offset

    def _check_fitted(self) -> None:
        if self.gp is None:
            raise RuntimeError("GPPolicy.fit() must be called before use")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.gp is None:
            return "GPPolicy(unfitted)"
        channels = [name for name, _, _ in self._layout]
        return f"GPPolicy(M={len(self.labels)}, channels={channels})"
