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
