# Robotics notes — findings, results, and decisions

Running log of the project from a robotics point of view: what the theory
demands, what the implementation actually does, what the numbers came out as,
and where we deviated from the paper and why.

Companion to `CLAUDE.md` (process/environment rules). Paper references are to
arXiv:2404.13458v2.

---

## 1. Audit of the original prototype (2026-09-05)

The starting point implemented roughly a quarter of the paper. Recording it
here so none of it gets reintroduced.

### Was correct
- `gamma` via SVD/Procrustes with the reflection fix (Eqs. 4-7).
- Residual GP fitted on `T - gamma(S)` (Eq. 8), i.e. the right target.
- `phi = gamma + psi . gamma` composition (Eq. 2).
- Velocity transport was algebraically Eq. 9: `v_hat = (I + J_psi) A v`.
- The analytic RBF Jacobian matched finite differences to 2e-9. (Its manual
  weight computation is only valid because the affine residual is exactly
  zero-mean by construction — a correctness argument the code never made.)

### Was missing
- **The policy refit (Sec. III-B step 2).** Nothing fitted `g`; the code
  replayed the warped trajectory open loop. That is precisely the trajectory
  reshaping the paper argues against in Secs. I and II.
- **The affine rotation in the orientation transport.** `polar()` was applied
  to `(I + J_psi)` alone, dropping `A`. Silently correct only for
  pure-translation targets, which is all that was ever tested.
- **Stiffness and damping transport** (`K_hat = J_perp K J_perp^T`) entirely.
- **All uncertainty** (Eqs. 12, 13, 16). `return_std` was never called.
- The property (ii) `det(J) > 0` diagnostic (Sec. III-C).
- The `det(Sigma) = 0 -> identity` fallback (Sec. III-E-a).
- SVGP / inducing points (App. A).

### Was broken
| Defect | Consequence |
|---|---|
| Timestamps from `time.time()` around a post-hoc replay loop, `dt ~ 1e-5 s`, then a `> 1e-4` guard | **Velocity labels were mostly zeros.** |
| `sim_policy_transport.py:576` subtracted **2 metres** from z at the grasp transition (`#why?` in the source) | Gripper driven under the table on every run. |
| GP length scale collapsed to its bound, **1e-5** | `psi` inert at every trajectory point. |
| Target scene was a pure translation | `gamma` solved it exactly; the nonlinear machinery was never exercised. |
| `np.random.seed()` called *after* `env.reset()` | No reproducibility. |
| Recording env built with `offscreen=False` | Demo frames were all `None`; the collage was blank. |
| No success metric, no tests, no git | — |

---

## 2. Deviations from the paper, and why

### 2.1 Kernel hyperparameters are bounded by keypoint geometry

**The paper does not specify hyperparameter bounds.** Left unbounded, marginal
likelihood maximisation on the handful of keypoints in a task parameterisation
fails in *both* directions, and both failures destroy a property the paper
depends on:

- **Length scale -> 0.** Observed in the prototype (1e-5). `psi` becomes a
  spike train: it matches the keypoints (property (i)) and is identically zero
  everywhere else, so the nonlinear stage does nothing at all.
- **Length scale and amplitude -> infinity together.** The degenerate regime in
  which a squared-exponential kernel imitates a low-order polynomial. Measured
  during development: length scale pinned to its ceiling at 2.83, amplitude
  2.84, and `psi` predicting a **1.85 m** deformation at a point 5 m from any
  keypoint. This breaks Sec. III-E-b outright — far from the source
  distribution `psi` must decay to zero so `phi` reduces to `gamma` (Fig. 2d).

**Decision.** Tie both hyperparameters to the data:
- length scale in `[0.5 * d_min, 2.0 * d_max]` over pairwise keypoint distances
  (`geometric_length_scale_bounds`);
- `sigma_p^2` in `[1e-8, 10] x` the mean squared residual
  (`residual_signal_variance_bounds`) — the deformation away from the keypoints
  should not vastly exceed the deformation at them.

**Result after the fix** (measured across seven scenarios):

| Scenario | length scale | `sigma_p^2` | keypoint residual | `psi` at 5 m |
|---|---|---|---|---|
| flat sheet -> parabola (25 kp, coplanar) | 0.419 | 1.8e-3 | 1.4e-5 m | 6e-87 |
| cube corners, pure translation | 0.020 | 1.4e-20 | 9e-17 m | 0 |
| cube corners, rotation + nonlinear | 0.034 | 9.0e-5 | 4.0e-8 m | 0 |
| 20x20 cloud -> wavy surface | 0.453 | 7.3e-4 | 1.9e-6 m | 6e-76 |
| rigid 3D (12 kp) | 0.037 | 1.4e-20 | 3.6e-16 m | 0 |
| near-duplicate keypoints | — | 1.4e-20 | 1.0e-16 m | 0 |
| 2D flat -> parabola | 0.327 | 2.3e-3 | 5.5e-6 m | 9e-95 |

Two behaviours worth noting. For **rigid** targets the amplitude collapses to
its floor, so the map degrades gracefully to pure `gamma` — the nonlinear stage
switches itself off when there is nothing nonlinear to explain. And `psi`
decays to numerical zero far from the keypoints in every nonlinear case, which
is exactly the out-of-distribution property Fig. 2d illustrates.

### 2.2 Property (i) holds to ~10 um, not to machine precision

`T = phi(S)` is exact (1e-16 m) whenever the residual field is zero, i.e. for
rigid targets. For smooth *dense* keypoint sets the residual is 1e-6 to 1e-5 m,
set by the conditioning of the Gram matrix: a long length scale over a fine
grid makes `K` nearly singular (condition number ~1e8-1e9), and the diagonal
jitter needed for a stable Cholesky smooths away the lowest-energy
interpolation modes.

**This is not worth chasing.** 10 um is two to three orders of magnitude below
the accuracy of any real keypoint source — AprilTag pose estimation and stereo
depth are millimetre-scale at best. The paper's own Fig. 7 reports matching
accuracies far coarser than this. Jitter is applied *relative* to the signal
variance so the behaviour is scale invariant.

### 2.3 The policy takes position **and** a belief of time

Sec. V states the real-robot policy of ref. [12] is a function of position and
a belief of time, not of position alone. This is not optional for reshelving: a
pick-and-place path revisits the same `(x, y)` at different phases (descend to
grasp, ascend with the object), so a pure `xdot = f(x)` policy cannot represent
it — the two passes would demand contradictory velocities at the same state.
`PolicyLabels` therefore carries a `time_belief` channel.

### 2.4 Derivative components are treated as independent (Eq. 12)

Eq. 12 propagates `Var[xdot_hat_i] = sum_d Var[J_id] xdot_d^2`, citing the
weighted-sum-of-Gaussians rule. That assumes the Jacobian entries in a row are
independent. For a squared-exponential GP they are exactly independent under
the prior (`k11 = (sigma_p^2 / ell^2) delta_de`) and only weakly correlated
under the posterior. We follow the paper.

---

## 3. Simulator choice

**MuJoCo 3.7.0 + robosuite 1.5.2, already installed.** No change. Measured on
this machine:

- Headless physics: **160 control-steps/s** at 20 Hz control, ~800 MB RSS.
- With EGL offscreen rendering at 256x256: ~1.0 GB RSS, negligible extra time.
- **46 robots, 28 grippers** registered — directly serves the cross-embodiment
  goal without writing any new robot models.
- `JOINT_TORQUE` part controller with `use_torque_compensation` — this is what
  makes a *true* Cartesian impedance controller possible. It matters: robosuite's
  own variable-impedance OSC takes a **diagonal** 6-vector of gains and
  therefore **cannot represent** `K_hat = J_perp K J_perp^T`, which is a full
  symmetric matrix in general. Transporting stiffness faithfully requires
  bypassing it.
- MuJoCo native `flexcomp` cloth (`<elasticity>`, no plugin needed in 3.7) runs
  at **8.3x realtime** for a 9x9 sheet, so the deferred dressing task of
  Sec. V-B is viable on CPU when we get to it.

Rejected: PyBullet (worse contact fidelity), Isaac/Genesis (GPU), raw MuJoCo
(would mean rewriting robosuite's robot/gripper/arena scaffolding).

---

## 4. Results log

*(Populated as experiments run.)*

### 4.1 Transportation core — unit validation (Phase 1)

| Quantity | Check | Result |
|---|---|---|
| Log marginal likelihood gradient | vs central differences | 2e-10 rel. |
| GP posterior derivative mean (Eq. 16) | vs central differences | < 1e-9 |
| GP posterior derivative variance (Eq. 16) | vs exact numerical limit from the posterior covariance | < 1e-6 rel. |
| SVGP vs exact GP at `M = N` | mean, std, derivative mean, derivative std | < 1e-8 |
| SVGP convergence | error decreases monotonically in `M` | confirmed |
| `phi` Jacobian (Eq. 9) | vs central differences | < 1e-6 |
| `R_hat` (Eq. 11) | orthonormal, `det = +1`, equals `A R` for rigid targets | exact |
| `K_hat` (Sec. III-G) | symmetric, positive definite, eigenvalues preserved | exact |
| Property (ii) | `det(J) > 0` fraction on curved targets | 100% |

### 2.5 A policy needs different kernel priors from a transportation map

This is the single most consequential finding of the implementation, and it cost
the most to isolate. The transport map and the refitted policy use the same GP
code, and giving them the same priors breaks the policy silently.

**Length scale.** `geometric_length_scale_bounds` derives its lower bound from
the *minimum* pairwise distance between training inputs. For a transport map
that is right: the keypoints are sparse and each must be matched exactly. For a
policy it is meaningless — on a 200-label, 20 Hz demonstration the labels are
0.3 mm apart, which measures how finely the trajectory was sampled, not the
structure of the field. The fitted length scale collapsed to 0.025 (standardised
units) and the commanded velocity was **exactly zero 2 cm off the path**. The
policy had no basin of attraction whatsoever, and since the robot's home pose and
the transported start `phi(x_0)` differ by of order 0.1 m, the arm never moved at
all. `GPPolicy.LENGTH_SCALE_BOUNDS` is therefore set in standardised input units
(0.4 to 3.0), independent of label density.

**Likelihood noise.** A transport map must interpolate (property (i)); a policy
must not. Labels a few millimetres apart along a curve, an order of magnitude
closer than the kernel length scale, make the Gram matrix near-singular; the
weights blow up with alternating signs, and the field is well behaved *on* the
demonstration and explosive just off it.

| `noise_variance` | cond(K) | max speed within 4 cm of the path |
|---|---|---|
| 1e-6 | 4.9e9 | **13.4 m/s** |
| 1e-4 | 4.9e7 | 3.1 m/s |
| **1e-2 (default)** | 8.8e3 | 0.69 m/s |

The demonstration's peak speed is 0.25 m/s. At 1e-6 the arm was flung off the
path within ten control steps.

Regularisation is also what determines whether the rollout completes at all.
Free rollout of the reshelving policy, endpoint error against a 0.60 m path:

| phase weight | noise | belief correction | completes | endpoint error |
|---|---|---|---|---|
| 1 | 1e-6 | 0.0 | yes | 0.607 m |
| 1 | 1e-6 | 0.2 | **no** | 0.618 m |
| 8 | 1e-6 | 0.0 | yes | 0.464 m |
| 8 | 1e-6 | 0.2 | **no** | 0.224 m |
| 1 | 1e-2 | 0.0 | yes | 0.0005 m |
| 8 | 1e-2 | 0.2 | yes | 0.0098 m |

**Correction to an earlier conclusion.** An earlier round of measurements, taken
before the priors were fixed, attributed a rollout stall to the time belief
running open loop and reported that closing it fixed the stall. That was a real
measurement but the wrong diagnosis: the table above shows the noise term is what
determines success, and that at 1e-6 the belief correction makes matters *worse*,
because it keeps re-anchoring the phase to a robot that is not moving. The belief
correction is retained — it keeps the position and phase of the query consistent,
and it pairs with the lag gate during execution — but it is a consistency
mechanism, not a cure for a badly conditioned policy.

**Phase weight.** Position and phase are different kinds of input and one
isotropic length scale cannot serve both. A pick-and-place path passes through
the hover pose twice, descending to insert and ascending to retreat, at nearly
the same position but very different phases. With an isotropic kernel the two
branches blend, the descent and the ascent cancel, and the rollout stalls at the
hover pose commanding 0.01 m/s. Weighting the phase axis (`PHASE_WEIGHT = 8`)
separates them while keeping the long positional length scale that provides the
basin of attraction. Measured on the transported labels, free-rollout endpoint
error fell from *failure* at weight 1 to 6-8 mm at weight 8.

### 2.6 Output channels are charted, not regressed raw

Stiffness and damping live on `S_3^+` and orientation on `SO(3)`. Regressing
their raw entries does not stay on those manifolds — a fitted stiffness can come
out indefinite, which an impedance controller turns into a **negative-stiffness
direction**, i.e. an unstable axis that actively pushes the robot away.

- SPD channels are regressed in the **log-Cholesky** chart (`M = L L^T`, log of
  the diagonal). Every point of the chart maps back to a positive-definite
  matrix, so interpolation is unconditionally safe.
- Orientation is regressed in the **continuous 6-D representation** (first two
  columns of `R`, third recovered by Gram-Schmidt). Quaternions double-cover
  `SO(3)`, so a regressor trained on them can be pulled to the wrong hemisphere.

All channels share one kernel and one Cholesky factor, per Appendix A; they are
standardised to unit variance first because m/s, log-stiffness and a unit-interval
phase cannot share a single amplitude. Velocity keeps a **zero** prior mean
(Appendix A); every other channel is regressed as a deviation from its label
mean, so far from the demonstration the robot falls back to the average
commanded impedance rather than going limp.

### 4.2 Policy refit — validation (Phase 2)

Synthetic pick-and-place with a self-intersecting path (120 labels, 20 Hz):

| Quantity | Result |
|---|---|
| Velocity reconstruction on labels | 7.5e-6 m/s |
| Orientation reconstruction (geodesic) | 1.2e-12 rad |
| Stiffness reconstruction | 1.3e-3 N/m, min eigenvalue 400.0 |
| Gripper reconstruction | 7.1e-4 |
| Velocity far from data | exactly 0 (zero-mean prior) |
| Stiffness far from data | 543 N/m, still positive definite |
| Epistemic std, on data -> off data | 7.6e-5 -> 3.9e-2 m/s |
| Closed-loop rollout endpoint error | 6.2 mm |
| Closed-loop rollout path deviation | 5.1 mm |
| Rollout arc length vs demo | 0.602 m vs 0.596 m (+1%) |
| Attractor identity `K dx = D xdot` | holds to 1e-6 |
