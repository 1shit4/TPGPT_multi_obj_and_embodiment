# Robotics notes — findings, results, and decisions

Running log of the TPGPT project from a robotics point of view: what the theory
demands, what the implementation actually does, what the numbers came out as,
and where we deviated from the paper and why.

Companion to `CLAUDE.md` (process and environment rules). All references are to
Franzese et al., arXiv:2404.13458v2.

**Headline result.** A single demonstration, transported into 20 randomised
reshelving scenes: **17/20 success, 7.6 mm mean placement error** (median 7.9 mm,
worst 11.7 mm), with keypoints displaced up to **375 mm** between source and
target. The scripted teacher's own placement error is ~10 mm, so the transported
policy places the object about as accurately as the demonstration it came from.

**The nonlinear stage is what makes that work.** Ablating `psi` and keeping only
the affine `gamma` drops success to **4/20** (p = 3.3e-5, Mann-Whitney), because
the two objects move independently between scenes and no single rigid motion can
match both — `gamma` alone leaves up to 84 mm of keypoint residual. See §4.6.

---

## 1. Audit of the original prototype

The starting point implemented roughly a quarter of the paper. Recorded here so
none of it gets reintroduced.

### Was correct
- `gamma` via SVD/Procrustes with the reflection fix (Eqs. 4-7).
- Residual GP fitted on `T - gamma(S)` (Eq. 8), i.e. the right target.
- The `phi = gamma + psi . gamma` composition (Eq. 2).
- Velocity transport was algebraically Eq. 9: `v_hat = (I + J_psi) A v`.
- The analytic RBF Jacobian matched finite differences to 2e-9. It is only
  valid because the affine residual is exactly zero-mean by construction — a
  correctness argument the code never made.

### Was missing
- **The policy refit (Sec. III-B step 2).** Nothing fitted `g`; the code
  replayed the warped trajectory open loop. That is precisely the trajectory
  reshaping the paper argues against in Secs. I and II.
- **The affine rotation in the orientation transport.** `polar()` was applied to
  `(I + J_psi)` alone, dropping `A`. Silently correct only for pure-translation
  targets, which is all that was ever tested.
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
| Keypoints derived from object **position only** | A rotated object gave an identical keypoint set; its rotation was invisible to the map. |
| No success metric, no tests, no git | — |

---

## 2. Deviations from the paper, and why

### 2.1 Kernel hyperparameters are bounded by keypoint geometry

**The paper does not specify hyperparameter bounds.** Left unbounded, marginal
likelihood maximisation on the handful of keypoints in a task parameterisation
fails in *both* directions, and each failure destroys a property the paper
depends on:

- **Length scale -> 0.** Observed in the prototype (1e-5). `psi` becomes a spike
  train: it matches the keypoints (property (i)) and is identically zero
  everywhere else, so the nonlinear stage does nothing at all.
- **Length scale and amplitude -> infinity together.** The degenerate regime in
  which a squared-exponential kernel imitates a low-order polynomial. Measured
  during development: length scale pinned to its ceiling at 2.83, amplitude
  2.84, and `psi` predicting a **1.85 m** deformation at a point 5 m from any
  keypoint. This breaks Sec. III-E-b outright — far from the source distribution
  `psi` must decay to zero so that `phi` reduces to `gamma` (Fig. 2d).

**Decision.** Tie both hyperparameters to the data: length scale in
`[0.5 d_min, 2.0 d_max]` over pairwise keypoint distances
(`geometric_length_scale_bounds`), and `sigma_p^2` in `[1e-8, 10] x` the mean
squared residual (`residual_signal_variance_bounds`) — the deformation away from
the keypoints should not vastly exceed the deformation at them.

**Result after the fix**, across seven scenarios:

| Scenario | length scale | `sigma_p^2` | keypoint residual | `psi` at 5 m |
|---|---|---|---|---|
| flat sheet -> parabola (25 kp, coplanar) | 0.419 | 1.8e-3 | 1.4e-5 m | 6e-87 |
| cube corners, pure translation | 0.020 | 1.4e-20 | 9e-17 m | 0 |
| cube corners, rotation + nonlinear | 0.034 | 9.0e-5 | 4.0e-8 m | 0 |
| 20x20 cloud -> wavy surface | 0.453 | 7.3e-4 | 1.9e-6 m | 6e-76 |
| rigid 3D (12 kp) | 0.037 | 1.4e-20 | 3.6e-16 m | 0 |
| near-duplicate keypoints | — | 1.4e-20 | 1.0e-16 m | 0 |
| 2D flat -> parabola | 0.327 | 2.3e-3 | 5.5e-6 m | 9e-95 |

Two behaviours worth noting. For **rigid** targets the amplitude collapses to its
floor, so the map degrades gracefully to pure `gamma` — the nonlinear stage
switches itself off when there is nothing nonlinear to explain. And `psi` decays
to numerical zero far from the keypoints in every nonlinear case, which is
exactly the out-of-distribution property Fig. 2d illustrates.

### 2.2 Property (i) holds to ~10 um, not to machine precision

`T = phi(S)` is exact (1e-16 m) whenever the residual field is zero, i.e. for
rigid targets. For smooth *dense* keypoint sets the residual is 1e-6 to 1e-5 m,
set by the conditioning of the Gram matrix: a long length scale over a fine grid
makes `K` nearly singular (condition number ~1e8-1e9), and the diagonal jitter
needed for a stable Cholesky smooths away the lowest-energy interpolation modes.

**Not worth chasing.** 10 um is two to three orders of magnitude below the
accuracy of any real keypoint source — AprilTag pose estimation and stereo depth
are millimetre-scale at best. The paper's own Fig. 7 reports matching accuracies
far coarser than this. Jitter is applied *relative* to the signal variance so
the behaviour is scale invariant.

In the actual reshelving pipeline the measured keypoint residual is **2.6e-7 m**.

### 2.3 The policy takes position **and** a belief of time

Sec. V states the real-robot policy of ref. [12] is a function of position and a
belief of time, not of position alone. This is not optional for reshelving: a
pick-and-place path revisits the same `(x, y)` at different phases (descending to
grasp, ascending with the object), so a pure `xdot = f(x)` policy cannot
represent it — the two passes would demand contradictory velocities at the same
state.

### 2.4 Derivative components are treated as independent (Eq. 12)

Eq. 12 propagates `Var[xdot_hat_i] = sum_d Var[J_id] xdot_d^2`, citing the
weighted-sum-of-Gaussians rule. That assumes the Jacobian entries in a row are
independent. For a squared-exponential GP they are exactly independent under the
prior (`k11 = (sigma_p^2 / ell^2) delta_de`) and only weakly correlated under the
posterior. We follow the paper.

### 2.5 A policy needs different kernel priors from a transportation map

The single most consequential finding of the implementation, and the one that
cost the most to isolate. Both use the same GP code, and giving them the same
priors breaks the policy **silently**.

**Length scale.** `geometric_length_scale_bounds` derives its lower bound from
the *minimum* pairwise distance between training inputs. For a transport map
that is right — the keypoints are sparse and each must be matched exactly. For a
policy it is meaningless: on a 200-label, 20 Hz demonstration the labels are
0.3 mm apart, which measures how finely the trajectory was sampled, not the
structure of the field. The fitted length scale collapsed to 0.025 (standardised
units) and the commanded velocity was **exactly zero 2 cm off the path**. The
policy had no basin of attraction at all, and since the robot's home pose and the
transported start `phi(x_0)` differ by of order 0.1 m, the arm never moved.
`GPPolicy.LENGTH_SCALE_BOUNDS` is therefore set in standardised input units
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

Regularisation also determines whether the rollout completes at all. Free
rollout of the reshelving policy, endpoint error against a 0.60 m path:

| phase weight | noise | belief correction | completes | endpoint error |
|---|---|---|---|---|
| 1 | 1e-6 | 0.0 | yes | 0.607 m |
| 1 | 1e-6 | 0.2 | **no** | 0.618 m |
| 8 | 1e-6 | 0.0 | yes | 0.464 m |
| 8 | 1e-6 | 0.2 | **no** | 0.224 m |
| 1 | 1e-2 | 0.0 | yes | 0.0005 m |
| 8 | 1e-2 | 0.2 | yes | 0.0098 m |

> **Correction to an earlier conclusion.** An earlier round of measurements,
> taken before the priors were fixed, attributed a rollout stall to the time
> belief running open loop, and reported that closing the loop fixed it. That was
> a real measurement but the wrong diagnosis. The table above shows the noise
> term is what determines success, and that at 1e-6 the belief correction makes
> matters *worse*, because it keeps re-anchoring the phase to a robot that is not
> moving. The belief correction is retained — it keeps the position and phase of
> the query consistent, and it pairs with the lag gate during execution — but it
> is a consistency mechanism, not a cure for a badly conditioned policy.

**Phase weight.** Position and phase are different kinds of input and one
isotropic length scale cannot serve both. A pick-and-place path passes through
the hover pose twice — descending to insert, ascending to retreat — at nearly the
same position but very different phases. With an isotropic kernel the two
branches blend, the descent and the ascent cancel, and the rollout stalls at the
hover pose commanding 0.01 m/s. Weighting the phase axis (`PHASE_WEIGHT = 8`)
separates them while keeping the long positional length scale that provides the
basin of attraction: free-rollout endpoint error went from *failure* at weight 1
to 6-8 mm at weight 8.

### 2.6 Output channels are charted, not regressed raw

Stiffness and damping live on `S_3^+` and orientation on `SO(3)`. Regressing
their raw entries does not stay on those manifolds — a fitted stiffness can come
out indefinite, which an impedance controller turns into a **negative-stiffness
direction**, an unstable axis that actively pushes the robot away.

- SPD channels use the **log-Cholesky** chart (`M = L L^T`, log of the diagonal).
  Every point of the chart maps back to a positive-definite matrix.
- Orientation uses the continuous **6-D representation** (first two columns of
  `R`, third by Gram-Schmidt). Quaternions double-cover `SO(3)`, so a regressor
  trained on them can be pulled to the wrong hemisphere.

All channels share one kernel and one Cholesky factor, per Appendix A, and are
standardised to unit variance first because m/s, log-stiffness and a
unit-interval phase cannot share one amplitude. Velocity keeps a **zero** prior
mean (Appendix A); every other channel is regressed as a deviation from its label
mean, so far from the demonstration the robot falls back to the average commanded
impedance rather than going limp.

### 2.7 Labels are the commanded attractor, not the measured pose

**The most damaging bug found during integration, and a conceptual one.**

Sec. V says the policy learns "the desired attractor position, orientation,
stiffness, damping" as a function of the current pose. Recording the *measured*
end-effector pose as the position label and then *executing* those labels as
attractor commands applies the impedance controller's tracking lag twice.

The lag is not small. Measured on the demonstration itself:

| segment | mean lag | speed |
|---|---|---|
| approach | 12.9 mm | 0.129 m/s |
| descend | 33.1 mm | 0.132 m/s |
| **grasp (dwell)** | 12.0 mm, **4.8 mm at the end** | 0.082 m/s |
| transfer | 45.6 mm | 0.214 m/s |
| insert | 10.5 mm | 0.102 m/s |

The demonstration succeeds because its attractor **stops** at each waypoint and
the arm settles to ~5 mm before the fingers close. With measured-pose labels the
transported rollout closed its gripper 20-56 mm from the object every time and
never picked anything up. Switching the position and orientation labels to the
commanded attractor took success from 0/8 to 3/8 immediately, and the rest of the
execution work took it to 17/20. The measured trajectory is retained in
`metadata["measured_positions"]` along with the lag statistics.

### 2.8 Execution: attractor integration, lag gating, and the approach phase

Three execution decisions, each forced by a measured failure.

**The dynamical system is integrated as a moving attractor**, and the arm follows
it through the impedance — which is how the demonstration itself was generated.
Recomputing the attractor from the measured position each step does not work: in
an impedance controller the attractor *is* the reference, so the restoring speed
is the position error times `K/D` — about 12 per second here. A 24 mm deviation
then commands 0.29 m/s regardless of what the policy asked for. Measured that
way, the arm ran at 0.37 m/s while the policy commanded 0.002 m/s, overshot the
grasp, and lost the path.

**Progress is gated on the arm's excess lag.** With no scripted dwells the
gripper closes on schedule while the arm is still behind; measured across eight
scenes, the gripper closed at step 50 every time while the arm did not reach the
object until step 54-63. The gate compares the lag against the `||K^-1 D|| v`
that the impedance law predicts, so free motion runs at full speed while a dwell
— where the commanded velocity goes to zero, so the expected lag does too —
demands that the arm actually arrive.

| `lag_tolerance` | success | placed < 25 mm | mean steps | median error |
|---|---|---|---|---|
| off | 3/8 | 2/8 | 208 | 195.8 mm |
| 0.012 m | 3/8 | 5/8 | 745 | 9.4 mm |
| 0.020 m | 5/8 | 5/8 | 511 | 9.0 mm |
| **0.035 m (default)** | **7/8** | **7/8** | 380 | **8.3 mm** |

**An approach phase brings the arm to the transported start.** The velocity prior
is zero (Appendix A), so a policy fitted on demonstration velocities has no
restoring component off the demonstration — at an unvisited state it blends the
velocities of nearby labels, which points *along* the path rather than *towards*
it. The arm's home pose is identical in every scene but the transported start
`phi(x_0)` is not; the measured gap averages 128 mm and reaches 269 mm. On a real
robot the operator hands the arm to roughly the right pose; here a reaching
primitive does the same job, and the gap is reported as a diagnostic.

### 2.9 Scene geometry is constrained by the arm, not by Table II

Table II gives the paper's randomisation ranges. The object ranges are
reproduced exactly (0.225 m in x, 0.366 m in y, 94.6 deg of yaw). The goal ranges
are not: with a Panda at the edge of a 0.8 m table, **end-effector x saturates
near 0.24 m** at every usable height, so the paper's 0.675 m of vertical goal
travel would put the upper slots outside the reachable envelope. The shelf is a
single board at a randomised height instead, giving 0.24 m of vertical and 0.26 m
of lateral goal variation.

Two scene details that were found the hard way:

- **Product width is set by the gripper.** Yaw is randomised over 94.6 deg, and a
  fixed-yaw top-down grasp collides with the box corners. The demonstration
  therefore aligns the grasp yaw with the object, which is what makes the
  demonstration genuinely `SE(3)`-dependent and the orientation transport
  meaningful rather than decorative. The box half-width (0.025 m) keeps its
  diagonal (0.071 m) clear of the gripper's 0.08 m opening.
- **The Panda's hand is deeper than its fingers.** Descending to a grasp within
  about 0.12 m of the shelf board's front edge catches the hand on the underside
  of the board — a failure that looked like a controller problem and was a
  geometry problem. The product spawn range is shifted away from the shelf rather
  than shortened, so the full Table II x-range survives.
- **The grasp must be taken at the top face of the box.** Offsets below the box
  half-height bottom out on its shoulder or the table and the grasp fails
  outright (measured: 0/4 at offsets of 0, 15 and 30 mm; 4/4 at 45 mm).

---

## 3. Simulator choice

**MuJoCo 3.7.0 + robosuite 1.5.2**, already installed. No change. Measured on
this machine:

- Headless physics: **160 control-steps/s** at 20 Hz control, ~800 MB RSS.
- With EGL offscreen rendering at 256x256: ~1.0 GB RSS, negligible extra time.
- **46 robots, 28 grippers** registered — directly serves the cross-embodiment
  goal without writing any new robot models.
- `JOINT_TORQUE` with `use_torque_compensation` — this is what makes a *true*
  Cartesian impedance controller possible, and it matters: robosuite's own
  variable-impedance OSC takes a **diagonal** six-vector of gains and therefore
  **cannot represent** `K_hat = J_perp K J_perp^T`, which is a full symmetric
  matrix in general. Transporting stiffness faithfully requires bypassing it.
- MuJoCo native `flexcomp` cloth (`<elasticity>`, no plugin needed in 3.7) runs
  at **8.3x realtime** for a 9x9 sheet, so the deferred dressing task of Sec. V-B
  is viable on CPU when we get to it.

Rejected: PyBullet (worse contact fidelity), Isaac/Genesis (GPU), raw MuJoCo
(would mean rewriting robosuite's robot/gripper/arena scaffolding).

One robosuite trap worth recording: the part controller rescales every action
from `[input_min, input_max]` onto `[output_min, output_max]`. Leaving
`output_max` at its default of 1 silently divides every commanded torque by the
ratio — the arm barely moves and the impedance law looks wrong when it is only
being attenuated. `CartesianImpedanceController` reads that value back from the
controller so the two cannot disagree.

---

## 4. Results log

### 4.1 Transportation core — unit validation

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

### 4.2 Policy refit — validation

Synthetic pick-and-place with a self-intersecting path (120 labels, 20 Hz):

| Quantity | Result |
|---|---|
| Velocity reconstruction (median) | 6.3% of mean speed |
| Orientation reconstruction (geodesic) | < 0.01 rad |
| Stiffness reconstruction | 5 N/m on 700 N/m (0.7%) |
| Velocity far from data | exactly 0 (zero-mean prior) |
| Stiffness far from data | 543 N/m, still positive definite |
| Epistemic std, on data -> off data | grows by ~9x |
| Free-rollout endpoint error | 6.2 mm |
| Free-rollout path deviation | 5.1 mm |

### 4.3 Cartesian impedance controller — physical validation

Deflection under a known 10 N external load, against the `dx = K^-1 F` prediction:

| Stiffness | measured | predicted |
|---|---|---|
| 200 N/m isotropic | 49.2 mm | 50.0 mm |
| 400 N/m isotropic | 24.7 mm | 25.0 mm |
| 800 N/m isotropic | 12.6 mm | 12.5 mm |
| diag(200, 800, 800), force in x | 48.2 mm in x | 50.0 mm |
| diag(200, 800, 800), force in y | 12.9 mm in y | 12.5 mm |
| **Rz(45) diag(200,800,800) Rz(45)^T**, force in x | **[30.3, 17.3, 0.0] mm** | [31.3, 18.8, 0.0] mm |

The last row is the point of the whole controller: with a transported (rotated)
stiffness the deflection is **not parallel to the applied force**. A diagonal
gain vector cannot produce that, and the eigenvalues — the physical stiffness
magnitudes — are preserved exactly under the congruence.

### 4.4 Scripted demonstration — reliability

20/20 successful demonstrations across randomised scenes, ~0.9 s each, mean
placement error ~10 mm. 200 labels at 20 Hz (10.0 s). All label families
populated; 85% of velocity labels non-zero, the remaining 15% being the grasp
and release dwells where the attractor genuinely holds still.

### 4.5 End-to-end transport — 20 randomised scenes, one demonstration

```
success 85% (17/20)
placement error on successes: mean 7.6 mm, median 7.9 mm, max 11.7 mm
keypoint residual (property i):   2.6e-7 m
det(J) > 0 (property ii):         100% of transported labels, every scene
keypoint displacement:            up to 375 mm
affine residual:                  up to 84 mm  (the nonlinear stage's workload)
approach gap:                     128 mm mean, 269 mm max
rollout length:                   318 steps mean
transport / epistemic / total velocity std:  0.0127 / 0.0035 / 0.0138 m/s
```

The three failures all ran out of step budget rather than mis-transporting: the
keypoint residual and Jacobian diagnostics are identical for them.

That the **transport uncertainty exceeds the epistemic uncertainty** by roughly
4x is the expected ordering for this task and matches Sec. III-H: there are only
18 keypoints but 200 policy labels, so the map is the less certain of the two.

### 4.6 Ablation: does the nonlinear stage earn its place?

The question the prototype could never answer, because its target scene was a
pure translation that `gamma` solves exactly. Here the two objects move
independently between source and target, so no single rigid motion can match
both. Same demonstration, same 20 scenes, same policy and execution — only the
residual regressor `psi` is removed.

| Transportation map | success | median placement error | on-success median | max keypoint residual |
|---|---|---|---|---|
| **full, `gamma + psi`** | **17/20** | **8.0 mm** | 7.9 mm | 2.6e-7 mm |
| affine only, `gamma` | 4/20 | 335.7 mm | 20.0 mm | **84.0 mm** |

Mann-Whitney U on placement error: the full map beats affine-only at
**p = 3.3e-5**.

Removing the nonlinear stage costs 65 percentage points of success. The 84 mm
of keypoint residual it leaves behind is the deformation the affine stage
physically cannot represent, and it is far larger than the ~25 mm of positional
tolerance the grasp allows. The four affine-only successes are the scenes where
the two objects happened to move nearly rigidly together.

**SV-GPT sits between the two.** With M = 12 inducing points against 18
keypoints, the sparse bound leaves a 7.3 mm keypoint residual and still placed
the object in 6/6 scenes, but at 13.3 mm median error against the exact GP's
8.3 mm — significantly worse (p = 0.021 over the same six scenes). The
approximation error of the variational bound shows up directly as task error,
exactly as Appendix A's caveat implies. Use the exact GP when the keypoint set is
small; the sparse variant exists for the 400-point cloud regime of Sec. V-C,
where exact inference is the thing that does not scale.

Reproduce with `python -m tpgpt.experiments.ablate_residual`.

### 4.7 Theory figures

`python -m tpgpt.experiments.figures` reproduces Figs. 2, 4 and 5. Fig. 5
confirms the paper's central claim about uncertainty structure: transport
uncertainty is minimal at the **keypoints** and grows away from them, epistemic
uncertainty is minimal along the **transported demonstration**, and the total is
the variance sum, low only where both are.

One subtlety worth recording: Eq. 12's transport uncertainty is a function of the
*source* position while the policy's epistemic uncertainty is a function of the
*target* position. Plotting the source-space field against target-space
coordinates puts the low-uncertainty band in visibly the wrong place. The figure
samples a regular grid in source space and draws it at `phi(grid)`, so both
fields share one warped mesh.

---

## 5. Grasp generation and scene understanding

Groundwork for the cross-object, cross-embodiment stage. The reshelving pipeline
assumes one known object and one known gripper: its demonstration hard-codes a
top-down pinch with the yaw aligned to a box. Neither assumption survives a
different object shape or a different hand, so the grasp has to come from the
object's geometry and the hand's geometry. That is what GraspGen-X provides.

**Scope.** Only the first two stages of the new-scene flow are built: text ->
object and destination, and object -> ranked grasps per gripper, plus the
conversion into an end-effector command. Filtering, choosing one grasp, and
keypoint extraction for the new scene are deliberately deferred pending
discussion, and section 5.6 is the measured case for why that discussion is
needed.

### 5.1 Two processes, because one is impossible

GraspGen-X needs Python 3.11 with `diffusers==0.11.1` and
`huggingface-hub==0.25.2`; TPGPT runs Python 3.10 with MuJoCo and robosuite. The
pins are mutually exclusive, so they cannot share an interpreter. The model is
also 1.6 GB and takes 3.3 s to load, which rules out spawning it per call.

It therefore runs as a long-lived ZMQ server in its own conda environment and
TPGPT talks to it over msgpack. `tpgpt/grasp/client.py` re-implements the wire
protocol rather than importing `graspgenx.serving.zmq_client`: that module is
itself torch-free, but importing it executes a package `__init__` that pulls in
torch. The client needs only `pyzmq`, `msgpack`, `msgpack-numpy` and `numpy`.

Measured on this machine: 4-12 s per inference on CPU, 2.6 GB resident for the
server, 3.3 s one-off model load.

### 5.2 Nine gripper pairs, spanning every kinematic family

GraspGen-X conditions on a gripper's **swept volume** rather than on trained-in
weights, so one checkpoint serves any hand whose `config.json` it can read. Nine
of the hands it ships are also simulated by robosuite:

| short | robosuite | GraspGen-X | family | aperture | TCP depth | closing axis |
|---|---|---|---|---|---|---|
| panda | PandaGripper | franka_panda | parallel_2f | 80.0 mm | 103.4 mm | x |
| robotiq85 | Robotiq85Gripper | robotiq_2f_85 | revolute_2f | 85.0 mm | 136.0 mm | x |
| robotiq140 | Robotiq140Gripper | robotiq_2f_140 | revolute_2f | 125.0 mm | 195.0 mm | x |
| rethink | RethinkGripper | sawyer_hand | revolute_2f | 66.0 mm | 110.0 mm | x |
| xarm | XArm7Gripper | xarm_hand | revolute_2f | 85.0 mm | 136.0 mm | **y** |
| umi | UMIGripper | franka_umi | parallel_2f | 80.0 mm | 177.0 mm | **y** |
| robotiq3f | RobotiqThreeFingerGripper | robotiq_3f | revolute_3f | 110.0 mm | 190.0 mm | unmeasured |
| yumi | YumiRightGripper | abb_yumi | parallel_2f | 50.0 mm | 125.0 mm | unmeasured |
| inspire | InspireRightHand | inspire_hand | revolute_3f | 80.0 mm | 150.0 mm | unmeasured |

Apertures span 50 to 125 mm and TCP depths 103 to 195 mm. **That near-2x depth
range is the cross-embodiment problem in one number**: the same contact on the
same object puts the end effector in a very different place for each hand, so an
end-effector pose is not portable even when the grasp is.

### 5.3 Three conventions that fail silently

Each of these was measured against known geometry, and each produces
plausible-looking wrong output when got wrong.

**Segmentation ids are offset by one.** robosuite maps each geom to its
instance's index and then adds 1, so 0 means background. Using the raw index
selects the background: the mask covered 55 607 of 65 536 pixels and the
resulting "object cloud" was 2.5 m across. That does not look like an
off-by-one, it looks like a broken camera calibration, which is exactly how it
was first misdiagnosed here.

**Observation images are vertically flipped** relative to the camera matrix.
Both the mask and the depth must be flipped back together. Getting this wrong
placed the cloud 28 cm from the object; getting the pixel ordering wrong instead
(row before column, rather than the `[col*z, row*z, z, 1]` the camera matrix
expects) moved it 26 cm the other way. With both correct the error is 1.5 to
3.7 cm per camera, which is the partial view, not the transform.

**Boundary pixels poison the bounding box.** A segmentation mask includes the
object's edge pixels, whose depth samples land partly on whatever is behind it.
On a 5 x 5 x 9 cm box the fused extents came out [17.3, 8.8, 9.7] cm; with one
pixel of mask erosion, [5.8, 6.2, 8.9] cm. Two pixels tightens it further but
empties the mask for small or distant objects, so one is the default.

Acceptance, measured against each object's true mesh AABB read from MuJoCo:

| object | mesh AABB | fused cloud | error |
|---|---|---|---|
| milk | 5.2 x 5.2 x 14.4 cm | 3.6 x 4.4 x 13.3 cm | 1.6 cm |
| can | 5.0 x 5.0 x 8.0 cm | 4.2 x 4.3 x 7.8 cm | 0.8 cm |
| cereal | 10.2 x 3.9 x 15.0 cm | 9.8 x 3.5 x 14.4 cm | 0.6 cm |
| bread | 6.1 x 6.2 x 4.8 cm | 5.0 x 4.9 x 4.2 cm | 1.3 cm |

All four bounding boxes contain the body origin. The residual is the unobserved
back side, as expected of a partial view.

> A note on method: the first version of this test compared against sizes
> derived by hand from each object's `horizontal_radius` and offsets, and bread
> "failed" by 3.1 cm. The mesh is 4.8 cm tall, not the 7.5 cm those numbers
> imply. The test now reads the AABB from MuJoCo, which is both correct and
> robust to the objects spawning at random yaw.

### 5.4 The frame contract, measured and then verified in physics

GraspGen-X emits a pose at the **gripper base** with `+Z` approach and `+X`
closing. robosuite is commanded at the `grip_site`, at the fingertips.
Converting is two steps:

1. Translate along the approach axis by the hand's own `fingertip` depth.
2. Rotate about the approach axis so the closing directions agree.

Measured by locating each hand's two opposing finger bodies and expressing their
separation in the `grip_site` frame: **`grip_site +Z` is the approach axis for
every gripper tested**, so the entire frame difference is whether the jaws close
along X or Y. For the symmetric two-finger hands the *sign* is irrelevant; the
*axis* is not.

Grippers whose convention was not measured -- the two three-finger hands, which
have no single closing axis, and the Yumi, whose fingers measured 65 degrees off
any axis -- **refuse conversion rather than defaulting to identity**. A wrong
rotation about the approach axis produces a pose that still looks entirely
plausible while closing across the object's long dimension.

**Verified end to end in physics.** A synthesised top-down grasp -- constructed,
not generated, so the test isolates the conversion from candidate quality --
converted through the contract and executed with the Cartesian impedance
controller lifted the object on the first attempt for both the Panda (103 mm TCP
depth) and the Robotiq 2F-85 (136 mm, opposite closing sign): +16.5 cm and
+14.9 cm respectively, tracking error under 1 cm.

### 5.5 Deterministic language, on purpose

No model, no network, no API key. The same prompt and the same scene always give
the same entity ids, and a prompt that fails to parse fails because a word is
missing from `tpgpt/language/vocabulary.py` -- a fix that is a one-line edit
rather than a re-prompt.

Receptacles are matched **structurally**, not by string comparison. A slot is
identified by two independent fields, its level and its lateral position, so
those are resolved separately. Enumerating every phrasing as an alias produced
400 strings per slot and, worse, made "top shelf" match all three top slots
equally -- turning a perfectly ordinary instruction into an ambiguity error.

The one place a default is applied is a level named without a slot ("the top
shelf"), which resolves to the middle slot **and says so in the rationale**.
Everything else that is ambiguous is refused with the candidates it considered:
an unknown noun, a pronoun, a slot named without a level, two levels in one
phrase, or a missing destination.

### 5.6 What the grasps actually look like, and why filtering is the next problem

One cloud per object, 40 candidates per gripper, seed 1:

| object | cloud points | panda best | robotiq85 best | robotiq140 best |
|---|---|---|---|---|
| milk | 770 | 0.853 | 0.924 | 0.978 |
| can | 44 | 0.766 | 0.703 | 0.863 |
| cereal | 2056 | 0.903 | 0.977 | 0.997 |
| bread | 180 | 0.628 | 0.798 | 0.831 |

Mean best score rises with aperture: panda 0.788, robotiq85 0.851,
robotiq140 0.917 -- a bigger jaw finds more feasible grasps, as it should.

**The embodiment effect is visible in the poses.** On the cereal box the best
grasps differ between the Panda and the two Robotiqs by 17.7 and 22.0 cm, while
the two Robotiqs -- same kinematic family -- differ by only 4.5 cm. Different
hands genuinely choose different grasps on identical evidence.

**And here is the case for the deferred filtering stage.** Only 21 to 29 percent
of candidates approach from above; the majority come from the side or from
below, where the fingers would strike the table. Executing the top-scoring
candidates directly, six per object, lifted nothing: the successful ones were
never in the top six, and two candidates knocked the object over. This is not a
conversion error -- section 5.4 shows a correct grasp lifts first try. It is
candidate selection, and it has three identifiable causes worth weighing in that
discussion:

- **The cloud is one-sided.** `agentview` contributes 1795 of 2056 points on the
  cereal box, so the generator proposes approaches from the one direction it can
  see. The sibling project measured a raw candidate median approach elevation of
  180 degrees and added an explicit visibility filter.
- **Reachability is not modelled at all.** Nothing in the candidate set knows
  where the arm is or whether the pose is achievable.
- **Sparse clouds still produce confident candidates.** The can at seed 1 gave
  44 points -- a 6.7 cm tall sliver of an 8 cm object -- and still returned 40
  candidates scored up to 0.863, placed for the object it could see rather than
  the object that is there.

### 5.7 Scene

`Reshelving` is untouched, so the 17/20 transport result still reproduces. The
new `TabletopShelf` scene carries several robosuite mesh objects and a
**staircase** shelf: the upper level sits further away as well as higher, so
both levels are reachable from directly above, which a conventional stacked
shelf would not be -- the upper board would roof the lower one. Slot x stays
under 0.24 m, the reach limit measured for the Panda in this arena.

Objects settle for 60 simulation steps at reset. Without it they drift up to
2.7 cm after the observation is taken, so every cloud would describe a pose the
object is no longer in.


---

## 6. Grasp-aligned keypoint extraction

The last undesigned stage of the cross-object pipeline. The reshelving keypoints
are two boxes of known size read out of the simulator (`product_main` and
`goal_marker`, the latter a box of exactly `PRODUCT_HALF_SIZE` at the goal, so
the existing scheme is already "the object where it starts" plus "the object
where it ends"). Neither survives a change of object: an arbitrary mesh has no
known extent and no task-meaningful body frame, and the new scene has no goal
marker, so the placed configuration has to be computed rather than read.

### 6.1 The construction

Keypoints are defined by **role**, not by geometry, which is what supplies the
elementwise pairing Sec. III-A assumes without having to match anything:

* **Task frame** (`task_frame`), columns `(c, a, n)`: `n` is the support
  normal, `c` the grasp's closing axis projected into the support plane, and
  `a = n x c`. It is defined identically for a carton, a can and a lemon, and
  for every gripper, because every grasp has a closing direction and every
  resting object has a support normal. Principal axes were rejected: they are
  shape dependent and their signs flip arbitrarily on near-symmetric objects.
* **A box fitted to the object's own cloud in that frame**, its lower face
  snapped to the support plane. Centre plus eight corners: nine keypoints per
  configuration, matching Sec. V-A's cube but measured rather than assumed.
* **Four configurations**: source object at pick and at place, target object at
  pick and at place. 18 paired keypoints in all.

The source's placed configuration is *derived, not observed*. Once the jaws
close the object is rigidly attached, so its motion is the hand's motion:
`T_obj(release) = T_eef(release) . T_eef(grasp)^-1` read straight off the
labels (`carry_transform`). This is exact, needs no second image, and is
consistent with the labels -- unlike an observation at release, which sees an
object half occluded by the gripper holding it.

The target's placed configuration is inherited **relative to the receptacle**:
lateral offset within the slot from the demonstration, orientation from
`placement_rotation`, height from the destination surface. No transform between
the source and target *objects* is fitted anywhere. A carton and a lemon are not
related by one, so fitting it would inject an arbitrary rotation into `phi`.
The recipe is shared; the geometry is not.

### 6.2 Acceptance: the validated result is a special case

Running the Sec. V-A campaign through the general extractor instead of the two
known bodies:

| Keypoints | Success | Median placement error | Failing seeds |
|---|---|---|---|
| Two known boxes (the validated scheme) | 17/20 | 8.0 mm | 8, 11, 16 |
| Grasp-aligned box from the cloud | **17/20** | **8.0 mm** | **8, 11, 16** |

Not merely the same rate -- the *same three seeds*, which are the step-budget
timeouts of section 4.5. Feeding the extractor a synthetic cloud sampled from
the product's known geometry isolates the keypoint construction from
perception, which has its own acceptance tests.

### 6.3 Transporting onto objects the demonstration never saw

One reshelving demonstration (a 5 x 5 x 9 cm box) transported onto five
tabletop objects, described only by segmented point clouds, onto a different
shelf. Scored geometrically: `demo.py` and `rollout.py` are still specific to
reshelving, so executing here would mean changing the two modules the 17/20
depends on.

| Object | Cloud pts | Fitted box (cm) | Grasp err | Release err | Jaw err | det(J)>0 |
|---|---|---|---|---|---|---|
| cereal | 1495 | 2.7 x 9.7 x 15.4 | 1.5 mm | 0.6 mm | 0.7 deg | 100% |
| milk | 396 | 4.2 x 4.2 x 14.2 | 1.6 mm | 0.2 mm | 1.1 deg | 100% |
| can | 505 | 4.5 x 4.2 x 8.5 | 1.3 mm | 0.5 mm | 1.8 deg | 100% |
| bread | 215 | 3.7 x 4.6 x 4.8 | 2.0 mm | 0.6 mm | 0.8 deg | 100% |
| lemon | 12 | 0.5 x 1.3 x 3.7 | 2.0 mm | 0.5 mm | 5.0 deg | 100% |

"Grasp err" is the distance from `phi(demonstration grasp point)` to the
materially corresponding point on the target object; "jaw err" the angle
between the transported gripper's closing axis and the target grasp's. Release
clearance matched its intended value to within 0.7 mm on every object.

**Rotating the grasp's closing axis barely matters.** Sweeping the target
grasp's closing direction over a full 90 degrees moved the grasp error only
from 1.5 to 2.3 mm and the jaw error stayed under 1.6 degrees. The recipe is
self-consistent under that rotation because the same frame convention builds
both boxes.

### 6.4 Contact keypoints, and why they are off by default

> **Reinstated, and this section is now the primary record.** It was once
> marked superseded by 7.11; 7.11 is withdrawn and 7.22 carries the
> confirmation. The conclusion below -- that the grasp contacts belong off by
> default -- is supported by two independent measurements that agree: the
> grasp-height table in this section, and the determinant collapse in 7.22
> (adding contacts to the box drops `min det(J)` from ~0.68 to 0.0074 and folds
> five maps in six).
>
> Both of those are **geometry**, computed from the fitted map with no physics
> involved, which is why they survived the deletion of the end-to-end campaigns
> in 7.26 when almost nothing else did. What 7.11 had against this section was a
> pose-alignment figure that scored the map at a single point, plus success
> rates measured while the gripper was shut the whole time (7.12). Neither could
> see a folded map.
>
> **What this section does not settle** is whether contacts should be off
> *forever*. They are off because interpolating them *exactly, alongside the box*
> folds the map. 7.22 measures the aim cost of leaving them out at about 53 mm,
> which is large. Section 8 carries the design question that follows.


The natural extension -- adding the two jaw contacts and the support contact,
found by intersecting the grasp plane with the cloud -- was built, measured, and
**turned off by default**. It is kept behind `include_contacts` because it is
the ablation for the claim below.

Contact keypoints pin *where along the object the jaws sit*. The box corners
simultaneously pin the object's shape. Whenever the target grasp sits at a
different height on its object than the source grasp did, those two demands
contradict each other, and because Sec. III-D interpolates every keypoint
**exactly**, the map satisfies both by folding space in between:

| Target grasp height | Grasp err, cereal | Release clearance (want 154 mm) | min det(J), real cloud | min det(J), synthetic |
|---|---|---|---|---|
| top (matches the source) | 2.1 mm | 157 mm | +0.51 | +1.04 |
| three-quarter height | 38.8 mm | 122 mm | +0.38 | +0.44 |
| mid-height | 43.4 mm | 90 mm | +0.07 | **-0.07 (folded)** |

The box keypoints are unaffected at every grasp height -- 1.5 mm and
`min det(J)` +0.67 throughout -- because nothing in them depends on where the
grasp sits.

The fold itself only appeared on the synthetic scene, where the objects differ
more sharply; on the real clouds the same mismatch drove `det(J)` to within 0.07
of folding while costing 64 mm of release clearance. Degrading towards a fold is
enough: the map is already unusable well before the determinant changes sign.

**The conclusion is architectural.** The grasp's position along the object does
not belong in the warp. It is already handled where it should be: a grasp
converts to an end-effector target through the frame contract in
`tpgpt/grasp/grasps.py`, verified in physics in section 5.5. What the warp needs
from the grasp is only its *orientation*, and it has that from the task frame.

### 6.5 Failures that every diagnostic called healthy

Four bugs during this work produced keypoint sets that were geometrically
plausible, fitted maps with micrometre residuals and 100% positive Jacobian
determinants, and were wrong. None would have been caught by the map's own
diagnostics; all showed up only as a rotated or misplaced gripper.

- **Sampling a single cloud point as a jaw contact.** Taking the point nearest a
  percentile along the closing axis leaves its other two coordinates at whatever
  that sample happened to be. On the reshelving box the two contacts landed at
  unrelated lateral offsets, tilting the contact axis 12.8 degrees in the source
  scene against 1.3 degrees in the target. Eq. 11 turns that straight into
  gripper yaw: 14 degrees of it, enough to catch the jaws on the box corners.
  Fixed by building the contact from robust statistics -- the closing coordinate
  measured, the other two taken from the box and the grasp.
- **The support contact as a footprint centroid.** Same class of error: a thin
  band of a randomly sampled cloud is an unstable statistic, and it moved 3.3 mm
  between two extractions of the same box. `phi` interpolates exactly, so that
  inconsistency becomes a local deformation planted directly under the object.
  Cost 2 of 20 episodes. Fixed by using the box's own lateral centre.
- **Inheriting the carried rotation instead of the placed orientation.** The
  demonstration turns its object by `-theta_source` to square it with the shelf.
  Copying that *amount* onto a target starting at a different yaw leaves it
  skewed by the difference -- measured at 8 degrees of residual gripper yaw.
  Both rotations are about the support normal and therefore commute, so
  conjugating through the task frames (the first attempt) is a no-op. What must
  be inherited is the placed orientation *relative to the receptacle*, exactly
  as the placed position already is.
- **Two different heights for the same object.** The pick box is snapped to the
  support plane; the placed box was positioned from the raw cloud, whose lowest
  points are truncated. The same cereal box measured 15.4 cm at the pick and
  14.7 cm at the place.

### 6.6 Measured limits

- **`env.table_top` is not where objects rest.** robosuite's `TableArena` puts
  the table's *top* surface at `table_offset` and hangs the thickness below it;
  the property adds half the thickness on top, so it reads 23 mm above the
  plane objects actually sit on. Snapping a box to it shortens the box, moves
  the grasp keypoints down the object, and makes the transported gripper close
  above the product. This is scene furniture rather than a transport bug, so it
  is documented rather than changed -- the shelf and goal heights in both scenes
  are derived from the same property and are consistent with each other.
- **The support snap is doing real work.** The fused cloud's lowest point stood
  **0.5 to 24.5 mm** above the table across the five objects. Mask erosion trims
  a pixel from every silhouette boundary, including the line where the object
  meets the surface. Note that dropping only an object's *bottom face* does not
  reproduce this -- the side faces still reach the contact line. Truncation, not
  the invisibility of one face, is what the snap defends against.
- **A 12-point cloud produces a confident, wrong box -- and `min det(J)` is the
  one diagnostic that notices.** The lemon's fused cloud had 12 points and
  yielded a box 0.5 x 1.3 cm across a fruit roughly 5 cm wide. Keypoint residual
  (0.08 um), `fraction_positive` (100%) and grasp error (2.0 mm) all stayed
  green, because they measure the map's self-consistency rather than whether the
  box describes the object. The *minimum* determinant does not:

  | Object | Cloud pts | min det(J) |
  |---|---|---|
  | milk / can / cereal | 396-1495 | 0.67-0.69 |
  | bread | 215 | 0.35 |
  | lemon | 12 | **0.008** |

  A box collapsed in one direction compresses space by the same factor, so
  `min det(J)` degrades smoothly with how badly the object is observed. It is
  worth watching alongside `fraction_positive`, which stays at 100% throughout
  and says nothing. This is the same failure mode as section 5.6, where 44
  points still produced 40 confidently scored grasps, and it is an argument for
  the visibility criterion that the filtering discussion has to settle.
- **The trajectory between the two keypoint clusters is unconstrained.** `phi`
  is a static warp, not an object tracker: pick and place are pinned exactly and
  the lift between them is smooth interpolation. Nothing in the keypoint set
  knows about a shelf edge that needs clearing.

---

## 7. Grasp selection, real shelves, and the end-to-end pipeline

### 7.1 The frame contract was much worse than a single letter

The contract had been "GraspGen-X emits +Z approach and +X closing; robosuite's
``grip_site`` is the same basis up to whether the jaws close along X or Y". That
held for six parallel jaws and refused everything else. Measuring nine hands by
**actuating them** -- driving each fully open then fully closed and taking the
principal direction of the finger displacements -- showed all three parts of it
are false somewhere:

| Hand | closing angle | anisotropy | `grip_site` vs config depth | note |
|---|---|---|---|---|
| panda | 0.0 deg | 7e11 | -6 mm | |
| robotiq85 | 0.0 deg | 553 | +9 mm | |
| robotiq140 | 0.0 deg | 1292 | **+75 mm** | long fingers |
| rethink | 0.0 deg | 1e12 | -1 mm | |
| xarm | -90.0 deg | 37609 | +11 mm | |
| umi | -90.0 deg | 732277 | **-177 mm** | site at the base; fingers along **-Z** |
| robotiq3f | 0.2 deg | **4.3** | -40 mm | three fingers, 2+1 |
| yumi | 0.0 deg | 5e8 | -28 mm | |
| inspire | 7.4 deg | **6.3** | **-150 mm** | site at the base; approach along **-Y** |

Two naming-based attempts failed before this one. robosuite's
``important_geoms`` lists geom names that do not exist in the compiled model for
some grippers -- the XArm's pads are ``gripper0_right_left_finger_pad_1``
against a listed ``gripper0_finger1_pad_collision`` -- and the Yumi's lists are
empty entirely. Taking the separation between named pad groups put the UMI's
fingers **3.17 m** apart. Motion needs no names and no special case for three
fingers or five.

The earlier note that "the Yumi's fingers measured 65 degrees off any axis" was
wrong; it came from the naming-based attempt. The Yumi is a plain 0-degree jaw.

**Anisotropy is worth keeping.** Every parallel jaw scores 553 or above; both
multi-finger hands score under 7. Nothing lands between, so "does one closing
axis describe this hand" is a measurement rather than a judgement.

The contract is now a full 3x3 alignment plus a 3-vector contact offset, because
neither an angle nor a depth can express a frame that is flipped end for end
(UMI) or whose contact sits 100 mm off the approach axis (Inspire).

### 7.2 The contact depth had to be calibrated, not derived

Three geometric definitions of "where the hand holds an object" were tried --
the config's fingertip depth, robosuite's base-to-``grip_site`` offset, and the
centroid of the closed fingers. **Each worked for six hands and failed three**,
and not the same three. Long finger links drag a centroid backwards, a flipped
frame inverts the sign, and an anthropomorphic thumb opposes from the side.

So the depth is calibrated: sweep it, execute a real grasp at each value, and
keep the middle of the widest band that lifts. That also measures how much depth
error each hand tolerates, which varies far more than expected:

| Hand | calibrated offset | working band |
|---|---|---|
| panda | +37.5 mm | 120 mm |
| robotiq85 / 140 / rethink / xarm | +30.0 mm | 135 mm |
| yumi | -7.5 mm | 60 mm |
| robotiq3f | +22.5 mm | 30 mm |
| umi | +30.0 mm | **15 mm** |
| inspire | none | **0 mm** |

A hand with a 135 mm band absorbs a 2 cm perception error without noticing; the
UMI does not. That number belongs in any claim about robustness.

### 7.3 A hand that converts perfectly and grasps nothing

The Inspire five-finger hand measures cleanly, converts cleanly, and lifted
**0.0 cm across 24 combinations** of object, grasp yaw and contact offset, at
every depth in the sweep. Five fingers driven by one open/close command do not
pinch a can from above.

``MEASURED_PAIRS`` (nine) and ``VERIFIED_PAIRS`` (eight) are therefore separate.
Measuring a hand's frame says where to send it; it does not say the hand can do
the job. Collapsing the two would have let a hand that cannot execute a grasp
join a campaign and surface later as a transport failure.

Related fix: the impedance controller emitted **one** gripper command
regardless of the hand. The UMI takes two and the Inspire hand six, and
robosuite pads a short action with zeros, so the fingers half-closed and the
object slid out -- silently.

### 7.4 The filter funnel

Seven filters, numpy and scipy only. Ported from the sibling project: visibility
against the viewing ray, target containment tested at the **fingertips** (its
measured lesson: testing the base rejected 52 of 58 valid grasps on a cup), and
scene collision with ``min_hits`` counting **distinct scene points** so one
depth flyer cannot veto a grasp. New here: jaw width measured along each
candidate's own closing axis at its own grasp height, duplicate suppression,
kinematic reachability, and demonstration consistency.

Every stage falls back to passing its input through and raising a flag rather
than returning an empty set, so a hard scene answers "here is the least bad
grasp and here is what is wrong with it" instead of "no grasp exists".

On a live cereal box, 1.6 s for the whole funnel:

```
generated       100 ->  100
visibility      100 ->   86   approaches within 100 deg of a camera's view
demonstrated     86 ->   17   approaches within 45 deg of the demonstrated one
on target        17 ->   17
jaw width        17 ->   17
collision        17 ->   17   hand and approach clear of the rest of the scene
reachable        17 ->   11   the arm can reach the grasp *and* the placement
distinct         11 ->    8
```

### 7.5 Reachability is the filter that matters, and the tolerance is not free

Before the demonstration filter existed, reachability cut **50 candidates to 8**
-- by far the largest single reduction, and the filter the sibling project
explicitly lacks. It is real damped-least-squares IK on MuJoCo's own site
Jacobian, checked at four poses: pre-grasp, grasp, place and retreat. A grasp
that can be reached but whose *placement* cannot is worthless, and nothing
geometric can see that.

Its tolerance had to be **loosened to 15 mm**. At 5 mm it rejected every
candidate for the top shelf, where IK converges to 4-8 mm because the arm is
near its workspace boundary -- while the impedance controller itself only tracks
to about 20 mm. A reachability test stricter than the controller rejects poses
the robot would in fact have reached.

### 7.6 Two things that had always been wrong and were invisible

**The shelf was in the collision group.** robosuite does not draw group 0, so
the shelf was physically present, collided correctly, and appeared in no render
ever made of this scene. It also let **depth pass straight through it**.

Making it visible immediately broke perception: ``agentview`` and ``frontview``
sit on the far side of the shelf from the table, so object clouds went from
**1433 points to zero**. They had been seeing through a wall. A ``workspace``
camera was added with a clear line past the shelf, and the stock ``sideview``
picked up the rest.

**The source keypoint frame used the wrong axis.** It took the product's *y*
axis as the closing direction, while the Panda's jaws close along the commanded
frame's *x*. The reshelving product is square in cross-section -- 25 by 25 mm --
so both choices fit the same box and the validated result never noticed. Off
that scene it decides which of the *target* object's axes the jaws map onto, and
90 degrees out put an 80 mm hand across a cereal box's 96 mm face while a
correctly chosen candidate closed across its 39 mm one.

### 7.7 A latent 180-degree ambiguity that cost 17/20 -> 5/20

The keypoint frame's sign was pinned by a world-axis test (``c . y >= 0``).
That is deterministic but arbitrary, and it depends on the **object's own yaw**,
so a source and a target standing at different yaws can flip independently.
Flipping ``c`` also flips ``a``, turning the frame 180 degrees about ``n`` and
permuting every corner label -- which plants a half turn in the middle of the
map that nothing downstream can see. Keypoint residual stays at micrometres and
``det(J)`` stays positive.

It stayed hidden for as long as the closing axis was taken from the product's
``y`` axis: over reshelving's +-47 degree yaw range that test never fired.
Correcting the axis to the one the jaws actually close along made it fire about
half the time, and the campaign fell to **5/20, every failure a stall**.

The fix is to sign the target frame against the *source* frame rather than
against the world, which is the choice that minimises the rotation between them.
The campaign recovered to **16/20 at 11.2 mm** (against 17/20 at 8.0 mm before
the axis correction). The remaining seed is not yet accounted for.

### 7.8 The lag gate could deadlock, and did

Every end-to-end run stalled at 600 steps having moved nothing. The cause was
not the transport:

The rollout advances its attractor only once the arm has caught up, so the
gripper cannot act on a pose the robot has not reached. The allowance is
``compliance * speed`` -- the lag expected *while moving*, which vanishes as the
arm slows. But an impedance-controlled arm carrying an object settles at a
**constant** offset: a load needs force, and a finite stiffness only produces
force when stretched. Measured with a Panda holding a can: **46 mm of sag
against a 35 mm tolerance**, at zero contact force, with the target pose
verified reachable.

That is a deadlock, not a delay. The gate shuts, the attractor freezes, the arm
has nowhere new to go, so the offset never shrinks and the gate never reopens.
The scripted demonstration never hit it because its attractor advances on a
timer regardless of how far behind the arm is.

Three changes:

- **The gate now recognises sag.** An offset that has stopped shrinking over 40
  steps is treated as load the arm cannot pull out of a finite stiffness, and
  subtracted, up to three times per run.
- **A watchdog.** If that does not help, the run stops immediately and reports
  ``arm_could_not_hold_the_pose`` instead of burning 600 steps and blaming the
  policy. Failing runs now end in 40-380 steps.
- **Orientation is compliant in transit.** The teacher is soft through free
  space and firm at the grasp and the insertion; that profile was transported
  for translation and ignored for rotation, which was pinned at a fixed gain
  the whole way. It now follows the demonstration's own stiffness profile.

Plus grasp selection by executability: rather than checking four poses, each
surviving candidate's *whole warped path* is scored by how much of it the arm
can follow with its commanded orientation, and the best is taken.

**The reshelving campaign went from 16/20 to 17/20 at 8.4 mm**, restoring the
validated baseline -- the sag had been costing a seed there too.

### 7.9 Two workspaces, and why the difference matters

The **reachable** workspace is every point the hand can be put; the
**dexterous** workspace is every point it can be put while holding a given
orientation. For a task built on top-down grasps only the second matters, and
they are far apart. Measured on a vertical slice through the Panda's workspace:
**99% reachable, 78% dexterous**.

All six shelf slots are usable, with at least 16 cm of headroom above each. So
the shelf is not the constraint. A *tall object* is: placing a 15.3 cm cereal
box on the top shelf puts the hand about 31 cm above the slot, past the measured
headroom.

This is why "reachable" is not a useful filter on its own, and why a run that
fails here must not be reported as a stalled policy: it is a property of the arm
and the object, not of the transport.

### 7.10 Where the end-to-end pipeline stands  *(withdrawn)*

> **Withdrawn -- see 7.26.** This was a status snapshot of the end-to-end
> pipeline, and it already carried one withdrawn result of its own. Status
> snapshots go stale the moment the code moves, and this one described a
> pipeline three rounds of changes ago. The current state lives in
> `CLAUDE.md`, which is rewritten rather than appended to.


### 7.11 The grasp contacts are load-bearing, and section 6.4 was wrong  *(withdrawn)*

> **Withdrawn -- see 7.26.** Every number in this section was withdrawn while
> it was still being written -- the success rates by 7.12 (every rollout ran
> with the gripper shut, so the outcomes measured nothing), and the conclusion
> by 7.22. It argued that section 6.4 was wrong about contact keypoints; 6.4
> is now reinstated. Nothing here is load-bearing and keeping the argument
> invites someone to re-derive it.


### 7.12 Every rollout ran with the gripper shut

The end-to-end runs were failing for a reason that had nothing to do with
keypoints, transport, or the arm: **the gripper closed at step 0 and never
opened**, with the hand still 122 to 152 mm from the object. It carried a shut
hand along the whole trajectory, touched nothing (0 N throughout), and reported
having placed the object in the wrong place.

The cause was mine. Dropping the demonstration's leading approach segment with
``labels[25:]`` sliced every array including the time belief, so the trimmed
labels began at a phase of **0.126**. The rollout starts its clock at **0.0**.
That query sits outside the range the policy was fitted on, and a GP
extrapolates -- the gripper channel extrapolated past ``-1`` to closed.

``PolicyLabels.__getitem__`` now renormalises the time belief to run 0 to 1 over
the slice. A slice of a demonstration is a demonstration in its own right.

A second bug in the same area: the stall watchdog aborted when the arm *was*
still closing the gap, which is precisely when it should keep waiting. It now
resets its counter while the lag is shrinking and only intervenes when it is
not.

**What this invalidates.** Every end-to-end success rate recorded before this
point, including the seven-way keypoint ablation of 7.22. Any run reported as
"placed in the wrong place" with the object unmoved was a run whose hand was
never open. The pose-alignment measurements are unaffected -- they come from the
transportation map, not from executing anything.

With both fixed, and the same nine tasks: objects are now **carried**, by 92 to
482 mm, where before they sat still. One run reaches 17 mm from its slot. One of
nine completes. Failures are now genuine manipulation failures rather than an
artefact.

Reshelving is unaffected throughout: **17/20 at 8.4 mm**, and 337 tests pass.

### 7.13 Naming the stage that failed, instead of the step it stopped at

The outcomes above -- `policy_stalled`, `placed_in_the_wrong_place` -- name where
a run *stopped*, and that is almost never where it went wrong. A hand that
closed on empty air and a hand that gripped correctly and set the object down
20 cm off both finish with the object away from its slot, and the placement
error is identical evidence for two problems with nothing in common. Every
success rate reported before this point was therefore a number without a cause
attached.

`tpgpt/experiments/diagnose.py` replays a run against a per-step trace of the
**object**, not just the arm -- its position, the jaw opening, and whether any
gripper geom is actually in contact with it -- and splits the run into the seven
things a pick-and-place has to do in order: approach, reach, grasp, lift, carry,
place, settle. The first cross is the fault; everything after it is a
consequence.

It changed the reading of the same runs immediately. Of five tasks, the outcome
field said three different things; the stage attribution said the hand was
arriving in the right place (11-42 mm) and then **failing to hold what it closed
on** -- 6 or 7 contact steps out of 300-400, and 2-4 mm of lift.

Two diagnostics were themselves lying, both by returning a plausible number
rather than an error:

- **`contact_offset("panda")` returned zero.** The lookup behind it matches by
  object *identity*, so a hand named by its registry key -- the way hands travel
  through most of this codebase -- fell through to "never measured" and got a
  zero offset back. Zero is a perfectly plausible offset, so the resulting
  41 mm frame error read as the arm missing its target. Both `contact_offset`
  and `alignment_rotation` now accept either form.
- **`contact_force` read 0 N through a rollout that was visibly carrying a
  cereal box.** `cfrc_ext` is filled by `mj_rnePostConstraint`, which `mj_step`
  runs only when something in the model asks for it; with no such sensor
  declared the array simply stays at zero. It is now computed explicitly.

### 7.14 The attractor clamp and the lag gate disagreed about "too far"

Section 7.8 diagnosed a deadlock and treated it with sag re-baselining, a
watchdog and compliant transit rotation. Those helped, but they were treating a
symptom: the deadlock is **structural**, and it is guaranteed whenever the
demonstration is fast enough.

Two mechanisms bound how far the attractor may sit ahead of the arm, and they
were computed from different speeds:

- the **clamp** allowed `speed_limit * compliance`, and `speed_limit` is
  `3 x` the *fastest label in the demonstration*;
- the **gate** shut at `compliance * current_speed + tolerance`.

So whenever `compliance * (speed_limit - current_speed) > tolerance`, the clamp
parked the attractor beyond the distance at which the gate would reopen. From
there the two lock together: the gate shuts, and the frozen attractor is
*dragged along behind the arm by the clamp* rather than being caught up to, so
the lag never shrinks. With `K = 350`, `D = 33.7` and a 0.2 m/s demonstration
the clamp's limit is 58 mm against a 35 mm tolerance -- and the measured lag
during a deadlock was 50-61 mm, the clamp's own number, held exactly.

Measured on one cereal-box run: the phase sat at **0.18 for 112 of 361 steps**.

> **The four-task sweep over `lag_tolerance` that was here is withdrawn
> (7.26)** -- it was an end-to-end measurement on campaigns that cannot be
> reproduced. What it does not affect is the deadlock itself, which is
> arithmetic rather than a measurement: with `K = 350` and `D = 33.7` the two
> limits are 58 mm and 35 mm, they cannot both be satisfied, and no experiment
> is needed to see that. The fix -- taking the smaller of the two -- makes the
> deadlocked state unreachable by construction, and is covered by unit tests.
>
> The sweep did carry one qualitative observation worth re-testing later: with
> the tolerance opened wide enough that the deadlock disappeared entirely,
> success got *worse* rather than better, and a different failure took its place
> -- which suggests the gate is doing real work rather than merely being in the
> way. Treat that as a hypothesis, not a result.

The failure that appeared when the gate was opened too far was `approach`, the
gripper acting before the arm has arrived, which is precisely what the gate
exists to prevent.

So the fix is not a bigger tolerance, which only trades one failure for the
other. The clamp now takes the **smaller** of the two limits, so the attractor
can never be further away than the distance at which the arm is allowed to catch
up: the gate stays tight at 35 mm and the deadlock becomes unreachable by
construction.

Worth noting what the change does *not* touch. At full speed the two limits
nearly coincide -- 54 mm against the old 58 mm -- so fast free-space segments run
as before. It binds only where the commanded speed goes to zero, which is the
dwell at the grasp and at the insertion, and which is exactly where the deadlock
was.

### 7.15 GraspGen-X is unseeded, so an unrepeated run measures nothing

The default planner is a **diffusion** model sampled from noise, and neither the
client nor the server exposes a seed. Two runs of the same scene get different
candidate sets, and the difference decides the task: the same cereal box on the
same seed was picked up on one run and missed on the next.

This invalidates any comparison drawn from single runs -- a two-run-per-cell
keypoint ablation compares two random draws at least as much as it compares the
two keypoint sets. `tpgpt/grasp/cache.py` keys candidates by the cloud they were
asked for, so the same scene asks the same question and gets the same answer;
a difference between two runs is then a difference in what actually changed. It
also removes 4-12 s of inference from every run.

The quantum in the key is a normalisation, not a tolerance -- rounding has bin
boundaries -- and a miss costs an inference, never a wrong answer.

### 7.16 The funnel, and the run that does everything right and is scored a failure

With the clamp fixed and the stages instrumented, fifteen runs across five
objects -- one hand, one slot, three seeds each -- give the first honest account
of where the pipeline loses tasks:

| stage | got past it | |
|---|---|---|
| approach | 13/14 | reached the start of the transported motion |
| reach | 10/14 | the hand arrived where the grasp was planned |
| grasp | 9/14 | the jaws closed on the object |
| lift | 6/14 | the object came off the table |
| carry | 5/14 | it stayed in the hand |
| place | 4/14 | it was let go over the right shelf |
| settle | 1/14 | it ended up resting in the slot |

(One of the fifteen produced no valid map and never reached the robot.)

Three losses at `reach`, three at `lift`, three at `settle`. The last three are
the interesting ones, because all three are **the same object doing everything
right**: the can was gripped, lifted 400-417 mm, carried for 100% of the transit
and delivered to within **5, 11 and 30 mm** of its slot -- and then scored a
failure, because the run hit its step budget while still holding the object in
the air. The clock never reached the segment that opens the fingers.

The cause is a gap in the watchdog of 7.8, not in the transport. That watchdog
tested the gate for being **exactly zero**. A gate held at 0.05 by a constant
sag is not zero, so the test never fired -- while the clock advanced at a
twentieth of nominal. The run neither deadlocks nor arrives; it crawls, and
burns the budget doing it. All three were reported as `policy_stalled`, which
blames the transport for a gate doing its job too well.

Progress is now measured on the **phase** rather than on the gate, against a
threshold derived from the demonstration's own pace (a quarter of nominal over
the patience window), so a crawl is detected exactly as a deadlock is.

> **The end-to-end numbers that were here are withdrawn (7.26).** They reported
> a fifteen-run before/after and a threshold at which reach error starts costing
> grips -- the latter being the same claim as the withdrawn 7.19, restated. Both
> came from campaigns that cannot be reproduced.
>
> The *bug* is not in doubt and does not rest on them. It is a logic error you
> can read directly: the watchdog tested the gate for being exactly zero, and a
> gate held slightly open by a constant sag is never exactly zero. The
> arithmetic is enough on its own -- a gate sitting at 0.05 advances the clock at
> one twentieth of nominal, so a run neither deadlocks nor arrives, and burns
> its entire step budget crawling. `tests/unit/` covers the corrected logic.

The fix is to measure progress on the **phase** against a threshold derived from
the demonstration's own pace, which is what the code now does. That makes a crawl
detectable by exactly the same mechanism as a deadlock, instead of being
invisible to it.

### 7.17 The report overlays were drawing every keypoint at the wrong height

A third silent one, and it had been making the reports actively misleading
rather than merely incomplete: the pick keypoints rendered on a red can sitting
next to the cereal box they were meant to describe.

robosuite stores observation images bottom-up while the camera matrix assumes a
top-down raster. `perception.cameras` handles this by flipping the depth and
mask buffers **before** unprojecting -- which means the matrix's row convention
is the flipped, human-readable one already. The overlay flipped the picture for
display, correctly, and then flipped the projected rows as well.

It stayed hidden because a vertical flip leaves the **columns** untouched. Every
keypoint landed on the right part of the scene left-to-right and merely at the
wrong height, which reads as a keypoint that has drifted -- a plausible thing
for a keypoint to do -- rather than as a projection that is wrong.

The test that settles it is a round trip, and it deliberately involves no object
shape: each object's cloud came from that camera's own depth buffer, so
projecting those points back must land them inside that object's mask.

| | unflipped | flipped |
|---|---|---|
| cereal | 94% | 41% |
| can | 88% | 63% |
| milk | 100% | 0% |

The first attempt at this test compared a projected body origin against a mask
centroid, and it was worthless: the centroid of a tall carton's *visible*
surface is nowhere near its body centre, so it reported an error of 62 px where
the projection was right. That confound is why the round trip is the test kept
in `tests/integration/test_overlay_projection.py`.

### 7.18 How much of each object the cameras actually get

Measured at reset, with the shelf variant `cubby` and all five objects on the
table:

| object | true extent (mm) | cloud points | cloud extent (mm) |
|---|---|---|---|
| cereal | 30 x 100 x 150 | 975 | 90 x 83 x 144 |
| milk | 40 x 40 x 158 | 345 | 47 x 49 x 125 |
| can | 50 x 50 x 81 | 255 | 40 x 46 x 73 |
| bread | 40 x 48 x 49 | 81 | 37 x 34 x 40 |
| lemon | 40 x 40 x 69 | **17** | 10 x 19 x 22 |

Two things to take from this, and one trap.

**The trap first.** The cereal's cloud spans 90 x 83 mm horizontally against a
true 30 x 100, and it is tempting to read that as an object too wide for an
80 mm Panda jaw to close on -- which would make the task impossible rather than
hard, and would be a serious thing to conclude. It is wrong: the box is yawed
about 45 degrees, so its *axis-aligned* extent is the diagonal. A 30 x 100 box
at 45 degrees has a 92 x 92 mm footprint, which is what was measured. The
keypoint extractor fits an oriented box in the grasp frame, so it is not fooled
by this; a reader comparing cloud extents to apertures would be.

Checked properly against the geometry, every hand can close on every object
except **yumi on the can** -- a 50 mm jaw and a 50 mm can, with no margin.

**Observation falls off a cliff with size**, and the count drops 975 -> 345 ->
255 -> 81 -> 17 as the object shrinks. The 17 is the worst case, from this
five-object scene; the three-object campaign scene gives the lemon 61 to 91.

Cloud size matters at the extreme: `MIN_CLOUD_POINTS` is 40, and the 17-point
lemon is below it, so that object is rejected before anything else happens.

> **What this section must not be used for.** Two attempts were made here to
> read a *task success rate* off the cloud sizes above -- first claiming that
> success followed observation quality, then that it followed the object's
> similarity in height to the demonstrated one. Both rested on end-to-end
> campaigns that are no longer reproducible (7.26), and the two readings
> contradicted each other on the same objects, which is on its own enough to
> distrust both.
>
> The measurement above is worth keeping because it is *not* one of those: it
> is a direct observation of what the cameras return at reset, repeatable in
> seconds, and independent of anything downstream. Treat it as a fact about
> perception and nothing more. Whether cloud size predicts success is an open
> question, and answering it needs the point count recorded alongside a
> stage-attributed outcome in a campaign whose code is committed.

### 7.19 How close the hand gets is the whole story, and tracking is not why  *(withdrawn)*

> **Withdrawn -- see 7.26.** This concluded that the distance from the
> fingertips to the planned grasp at the moment the jaws close is the whole
> remaining problem, and that it separates success from failure cleanly at
> about 25 mm. It was measured on the `contacts` keypoint family that 7.22
> later retired, on campaigns that are no longer reproducible, and the clean
> separation does not survive either change. The framing also outlived its
> evidence: 7.20, 7.23 and 7.25 all took 'reach error is the bottleneck' as
> settled and reasoned from it.


### 7.20 The map is fitted at the fingertips and applied at the wrist

The keypoints are anchored at the point where the hand **holds** the object --
the source block at `position + R @ contact_offset`, the target block at
`GraspFrame.from_grasp`'s `tcp_position`. The demonstration, however, is
recorded at `grip_site`, the wrist. Those are a hand-specific lever arm apart:
41 mm on a Panda, 61 mm on a 2F-140, 117 mm on a UMI.

A transportation map interpolates its keypoints *exactly* and says nothing about
anywhere else. Warping the wrist path therefore gives up the one guarantee the
keypoints provide, over precisely the distance that matters, at precisely the
moment it matters.

**Why this is a correction and not a tuning.** The "end-effector" whose
trajectory the paper transports is the tool point -- the place the hand actually
holds the object. This project had been transporting the flange. Those are
different points on a rigid body, and a map that is exact only at its keypoints
cannot be applied at one of them after being fitted at the other.

The argument stands on the geometry alone and does not need a success rate to
support it: the keypoints are built at the contact point, so the contact point
is the only place the map makes a promise about.

> **The size of the improvement is not recorded here on purpose.** It was
> measured on eight end-to-end runs, and those campaigns are no longer
> reproducible (7.26). Re-measuring it is cheap and does not need the
> simulator -- warp the wrist path and the fingertip path through the same map
> and compare each one's closest approach to the chosen grasp. That is pure
> geometry, deterministic, and takes milliseconds per case.

**A retraction.** This hypothesis was raised early, tested, and recorded as
disproved -- the two columns came out identical. That test ran while
`contact_offset("panda")` was silently returning **zeros** (7.13), so it
compared a quantity with itself. The lesson is not about frames: a diagnostic
that reports "no difference" deserves the same suspicion as one that reports a
surprising difference, and the cheapest guard is to assert the instrument reads
non-zero before trusting a null result.

Implemented as an optional `tool_offset` on `rollout_policy`, defaulting to
zero. At zero it is byte-identical to the previous behaviour, which is what the
reshelving campaign runs and which still gives **17/20 at 9.1 mm on seeds
8, 11, 16** after the change.

### 7.21 What the destination costs: the mirrored side, not the lower shelf  *(withdrawn)*

> **Withdrawn -- see 7.26.** An isolation study over six destinations. It had
> already been rewritten once after its first version turned out to be an
> artefact of the keypoint set held fixed around it, and the rewrite rests on
> campaigns that are no longer reproducible. Twice-burned on the same
> question.


### 7.22 The keypoint ablation, run properly: the box alone, and why

> **How much of this survives, and why.** This is the only end-to-end campaign
> in the project that was internally controlled: seven keypoint combinations,
> two objects, three seeds each, 42 runs, all in **one process against one code
> state**, with the grasp cache pinning the candidate pool so the comparison is
> of keypoints rather than of draws from an unseeded diffusion model. Every
> other campaign has been deleted (7.26); this one is kept because everything
> varied inside it is genuinely comparable.
>
> It still has one defect it cannot be cured of: the code that produced it was
> never committed, so the run cannot be reproduced exactly. Split the table by
> what would need re-running to check it:
>
> * **`min det(J)`, `map folded` and `aim` are geometry.** They are computed
>   from the fitted map and the transported labels, with no physics anywhere.
>   Re-deriving them needs a cached point cloud and a cached grasp, takes
>   milliseconds, and is deterministic. Trust these.
> * **`success` is physics.** It needs the full rollout, and it is the column
>   that cannot be reproduced. Treat it as a hypothesis.

| keypoints | pts/block | **aim** (geometry) | min det(J) | map folded | success (physics) |
|---|---|---|---|---|---|
| **box** | 9 | 53 mm | 0.53 - 0.79 | 0/6 | 4/6 |
| box + support | 10 | 60 mm | 0.55 - 0.88 | 0/6 | 3/6 |
| contacts + support | 5 | **7.7 mm** | 0.27 - 0.52 | 0/6 | 2/6 |
| contacts | 4 | **5.2 mm** | 0.18 - 0.70 | 0/6 | 0/6 |
| support | 1 | - | degenerate | - | 0/6 |
| box + contacts | 13 | **4.5 mm** | **0.0074** | **5/6** | 0/6 |
| box + contacts + support | 14 | **6.0 mm** | **0.0066** | **5/6** | 0/6 |

**"Aim" is how close the transported plan ever passes to the chosen grasp
point** -- measured on the warped trajectory before the robot moves, so it says
whether the plan is even pointed at the right place. This column was not in the
original version of this table, and leaving it out hid the most important thing
in the experiment.

**Adding the contacts improves the aim tenfold, from 53 mm to about 5 mm.** The
contact keypoint pins the map at the grasp, which is exactly what it is for and
exactly what the box cannot do -- no box corner is anywhere near where the
fingers close. That result is geometry and it is not in doubt.

**And aim runs the opposite way to success.** The two sets that aim best succeed
never; the two that aim worst succeed most. Whatever decides these runs, it is
not how accurately the plan is pointed at the grasp, and any future work that
optimises aim alone should expect to convert nothing. There are two distinct
mechanisms behind that, and they are worth separating.

**Adding the jaw contacts to the box collapses the map.** `min det(J)` falls a
hundredfold and five of six maps are rejected as non-diffeomorphic. The value is
identical to four decimal places across runs, so this is a deterministic
geometric consequence and not a sampling accident.

The cause is redundancy, and it is worth stating precisely because the contacts
are perfectly good on their own -- `contacts + support` yields a valid map and
succeeds twice. The contacts sit **inside the box's own convex hull**,
centimetres from corners that already pin the warp. A map required to
interpolate both exactly must bend sharply over that short distance, and that is
exactly where the determinant goes to zero.

**There is a number attached to that, and it is the useful part.** Suppose the
map has to move one point by a distance `d` relative to another point a distance
`L` away that must not move. Getting from one to the other over that gap needs a
deformation gradient of roughly `d / L`. The Jacobian determinant passes through
zero -- the map folds, space turns inside out -- when that ratio approaches the
smallest singular value of the map's undistorted part, which is about **0.8**
here (a healthy `det(J)` of 0.53 in three dimensions means singular values
averaging 0.53^(1/3) = 0.81).

Put the measured numbers in. The box map is already about **53 mm** wrong at
the grasp point, which is `d` -- that is what the aim column says. The contacts
sit "centimetres" inside the box hull, so `L` is roughly **20 to 40 mm**. That
gives `d / L` between **1.3 and 2.7**, against a folding threshold near 0.8.

**The two constraints disagree by more than the distance available to reconcile
them.** That is why `min det(J)` lands on 0.0074 identically to four decimal
places -- it is not a sampling accident or a tuning failure, it is geometry with
no room left in it. The same arithmetic run backwards gives the design rule for
anything built to fix this: a correction that pins the contact needs at least
`L > 1.25 x d`, about **66 mm** of clear space to blend out over.

**The second mechanism, which is separate and easier to miss.** Contacts
*alone* do **not** fold -- `min det(J)` is 0.18, no map is rejected -- and the
aim is 5.2 mm, near-perfect. And they still succeed 0 of 6. So the contacts fail
even when the map is valid and the plan is pointed correctly, which means bad
aim was never the whole story.

A `det(J)` of 0.18 against the box's 0.53 means the map is compressing volume
about **fivefold** somewhere along the path. Four or five keypoints simply do
not constrain a three-dimensional warp: the map is pinned where the contacts are
and free to deform violently everywhere else. The box's nine points are worse at
the grasp and far better everywhere else. So the real trade is **local precision
against global conditioning**, not "contacts good" against "contacts bad", and
any design that wants both has to supply both -- see section 8.

The same redundancy explains why the dedicated support point earns nothing:
`fit_aligned_box` **snaps the box's lower face to the support plane**, so the
four bottom corners already *are* contact points on the surface the object rests
on. The guarantee that the object meets the shelf rather than being dropped onto
it or driven into it is supplied by the box, four times over. Adding a fifth
coincident constraint adds no information.

4/6 against 3/6 is one run and not separable at this sample size, and it is in
the column that cannot be reproduced anyway. **The box is the default because of
its `det(J)`, not because of its success rate** -- it is the only set that both
yields a valid map every time and keeps the map well conditioned away from the
keypoints.

**This restores section 6.4, and section 7.11, which had overturned it, is
withdrawn.** 6.4 concluded the contact keypoints should be off by default. It
was overturned on two pieces of evidence that were both bad: a *geometric*
pose-alignment measurement which never executed anything and so could not see a
folded map, and end-to-end success rates measured while the gripper was shut the
whole time (7.12). Both favoured the contacts; neither was capable of detecting
the failure that actually matters.

The general lesson is about which measurement is allowed to settle a design
question. Pose alignment is a property of the map at one point; the determinant
is a property of the map everywhere. Only the second can see `det(J)` collapse
between two keypoints, and only the second was ever going to answer this. Note
that this cuts both ways now: the aim column added above is *also* a property of
the map at one place, and on its own it would have picked the contacts.

---

### 7.23 The walls are free; the roof is not  *(withdrawn)*

> **Withdrawn -- see 7.26.** An isolation study over shelf geometry, resting
> on campaigns that are no longer reproducible. Its one structural claim --
> that a shelf with a roof cannot be solved from a top-down demonstration,
> because the approach direction is wrong by construction -- is geometry
> rather than measurement, and is kept as an open item in section 8.


### 7.24 The UMI failure is the hand's, not the tool-frame change  *(withdrawn)*

> **Withdrawn -- see 7.26.** An A/B on one gripper, resting on campaigns that
> are no longer reproducible. The observation worth keeping is structural and
> is now an open item in section 8: the UMI's contact offset is the only one
> in the registry with a large lateral component, `[0.0, -0.035, -0.112]`,
> where every parallel jaw is almost purely along the approach axis.


### 7.25 Kinematic infeasibility, and what the robot did about it before

**What it was doing: nothing.** The Cartesian impedance controller has no
inverse kinematics. It turns a pose error into a force,
`f = K(x_d - x) + D(v_d - v)`, and applies `J^T f`. Against a pose that does not
exist for this arm -- the right point with a hand angle the elbow cannot produce
-- it pulls forever, settling about **38 mm** away with a standing error no
stiffness removes. The arm does not stop, does not skip, does not re-plan.

The consequence is that infeasibility never looked like infeasibility. The lag
gate saw the residual error, shut, and the run was reported as a stall -- so a
kinematic limit arrived wearing the costume of a control failure, which is
exactly the confusion 7.13 was built to end.

Measured by replaying a warped path pose by pose under position control:
**about a quarter of a trajectory is unreachable with its commanded
orientation**, and on those poses the tracking error is 38 mm against 13 mm on
the feasible ones. Position control is not the fix; the poses do not exist.

Infeasibility *was* handled in one place -- grasp selection scores each
candidate by `executable_fraction` and keeps the best of eight, trying both
frame signs. But it selects among candidates and never modifies a trajectory:
no re-timing, no relaxing an orientation that does not matter, no use of the
arm's redundancy.

**Three changes.**

*The filter now checks corridors rather than endpoints.* `by_reachability` went
from four poses to twelve: five down the approach corridor onto the object, two
on the lift after closing, five down the corridor onto the shelf and back out.
A candidate can be reachable at the standoff and at the grasp while the descent
between them is not, and a hand that cannot hold its angle 4 cm above an object
never reaches the object.

*The selector now scores the trajectory it executes.* `executable_fraction` was
being called on the **wrist-frame** labels while execution transports the
**tool-frame** ones -- a mismatch introduced with 7.20, displacing the scored
path from the run path by the hand's contact depth: 41 mm on a Panda, **117 mm
on the UMI**, which fails at `approach` in every campaign. It also now samples
densely around the grasp and the release. Density itself turned out not to
matter -- 14 uniform samples give the same answer as 200, to within a few points
-- but *location* does: a uniform sweep spends its budget on free-space transit,
where the hand can give up a few degrees and lose nothing, while the two moments
that decide the task are a handful of labels wide.

*A runtime fallback, in two tiers, consulted only while the gate is already
shut* -- so a healthy run pays nothing, confirmed at 0 relaxations and 0 skips
on a clean can-to-shelf run:

1. **Relax the orientation** when the position is reachable and only the hand
   angle is not. This is the common case and nearly free: the demonstration is
   already compliant in transit, so a few degrees of wrist angle in free space
   costs nothing.
2. **Skip to the next feasible pose** when the position itself is out of reach.
   The attractor is steered to the next label the arm can hold, capped at the
   demonstrated speed so the detour is a motion rather than a jump, and the
   **phase** hands over on arrival so the gripper schedule stays in step with
   where the hand actually is.

When nothing ahead is reachable either, the run carries on and the watchdog ends
it -- the honest outcome, since the rest of the path is not executable.

Reshelving after the change: **17/20 at 9.9 mm, seeds 8, 11, 16** -- the same
three, so the validated path is intact.

**The gating costs nothing when it is not needed**, which is the one claim here
that does not depend on a success rate: the fallback is consulted only while the
lag gate is already partly shut, so a run that is tracking normally never enters
either tier. Instrumented on a clean can-to-shelf run, that is 0 relaxations and
0 skips.

> **The end-to-end A/B that accompanied this is withdrawn (7.26).** It compared
> stage-by-stage counts with the fallback off and on over fifteen tasks, and
> concluded the change converted nothing. That may well be right, but the
> campaigns behind it are not reproducible and the sample was fifteen runs, on
> which a one-run difference is noise. What survives is the part below, which
> was measured by replaying poses under position control rather than by scoring
> outcomes.

**What the replay measurement showed, and it does not need a campaign.** Two
tiers were built. Tier 1 relaxes the orientation when the position is reachable
and only the wrist angle is not. Tier 2 skips ahead to the next reachable pose
when the position itself is out of reach. Instrumenting the runs showed tier 1
firing often and **tier 2 never firing at all**.

That is worth keeping because it is a clean confirmation of the diagnosis rather
than a measure of the outcome: infeasibility in this task is *always*
orientation-only and never position. The arm can always get its hand to the
right point; what it sometimes cannot do is get the hand there at the commanded
angle. The skip-ahead path is correct code that has never yet been needed, and
should be treated as untested in practice.

> A bug worth recording because the unit tests could not have caught it: the
> fallback read the commanded rotation before it was assigned, and the loop had
> been ordered so that nothing else did. It crashed the first regression run.
> The rotation and gripper readout now sit immediately after the prediction they
> come from. Running the regression, not the suite, is what found it.

### 7.26 Why the end-to-end campaigns were deleted, and what replaces them

On 2026-09-07 every end-to-end campaign in `outputs/` was deleted -- 254 MB
across `outputs/campaigns/` (five campaigns, 97 runs) and
`outputs/campaigns_contacts/` (three campaigns, 43 runs), plus the
`outputs/index.html` that indexed them. Sections 7.10, 7.11, 7.19, 7.21, 7.23
and 7.24 were withdrawn with them, and the numbers were stripped out of 7.18,
7.20, 7.22 and 7.25. This section records why, because the reasoning is more
valuable than any of the numbers were.

**The problem is not that the numbers were noisy. It is that the code that
produced them no longer exists.**

#### The timeline

Reconstructed from file modification times and `git log`:

| time | what happened |
|---|---|
| 00:15 - 00:27 | the three `campaigns_contacts` runs |
| **00:29** | **`diagnose.py` modified** -- the measuring instrument itself |
| 00:40 | `keypoints` campaign, 42 runs |
| 00:58 | `grippers` campaign |
| 01:03 | `slots` campaign |
| **01:09** | **`run_experiments.py` modified** -- the harness that writes the results |
| 01:14 | `shelves` campaign |
| **11:39 - 11:41** | **`pipeline.py`, `filters.py`, `rollout.py` modified** -- grasp filtering and execution |
| 12:00 | `objects` campaign |

Three separate rounds of code changes landed *between* campaign runs. One of
them changed the instrument that measures the error. One changed the execution
path itself. So the eight campaigns are not one experiment with a variable
changed -- they are eight experiments run against at least four different
systems, and every comparison across them is meaningless.

#### The part that cannot be repaired

Four of the five files involved -- `pipeline.py`, `filters.py`,
`run_experiments.py`, `diagnose.py` -- were **untracked**. They had never been
committed, not once, in any version. `rollout.py` was tracked but modified since
its last commit. The manifests recorded a title, a description, run counts and a
timestamp: no git SHA, no settings, no keypoint set, no tolerances.

There is therefore no commit anyone could check out to reproduce any campaign in
this project, and no record of which code produced which table. That is not
"noisy data" -- noisy data can be re-measured and averaged. This is data that
cannot be attributed to a system, which makes it not evidence at all.

#### What it cost

Three findings in these notes were published and then had to be retracted, and
all three failed the same way: **the conclusion outlived the settings it was
measured under, and nothing in the note said what those settings were.**

* **7.11** overturned 6.4 using success rates from rollouts whose gripper never
  opened (7.12). Every number in it was withdrawn while it was still being
  written.
* **7.21** was rewritten in full after its first version turned out to be an
  artefact of the keypoint set that happened to be selected at the time.
* **7.19** measured a clean success/failure split at about 25 mm on the
  `contacts` keypoints. 7.22 then changed the default keypoints, the split
  stopped existing, and nobody re-measured -- so 7.20, 7.23 and 7.25 all went on
  reasoning from "reach error is the bottleneck" as though it were settled.

The through-line is that the system and the measurement were changed in the same
sitting, with no record of which was which. Tests meant to find bugs generated
three wrong conclusions instead, and each wrong conclusion then motivated
further work.

#### What survived, and why

Everything kept in `outputs/` after the deletion is one of three things:

* **Geometry.** Keypoint residuals, `min det(J)`, how close a transported plan
  passes to a grasp -- all computed from a fitted map with no physics anywhere.
  Deterministic, re-derivable in milliseconds from a cached cloud and a cached
  grasp. `outputs/keypoints/report.json` is this.
* **Calibration.** The measured gripper frame contract
  (`outputs/keypoints/calibration.log`, `inspire_probe.log`), which backs 5.4
  and 7.2. These are measurements of the hardware description, not of a run.
* **The validated Sec. V-A result.** `outputs/reshelving/`, the 20-scene
  regression at 17/20, plus `outputs/ablation/` and `outputs/figures/`. This one
  is genuinely re-runnable: fixed seeds, no diffusion model in the loop, and it
  has been re-measured repeatedly across code changes at 17/20 with the same
  three failing seeds each time.

`outputs/grasp_cache/` is also kept, and it is kept for a different reason: it
is an *input*, not a result. GraspGen-X is an unseeded diffusion model (7.15),
so the cache is the only thing making any grasp comparison repeatable at all.
Deleting it would make future experiments incomparable with each other, not just
with the past.

Section 7.22 is the one end-to-end campaign kept in the notes, because it was
internally controlled: all seven variants in one process against one code state,
with the cache pinning the candidate pool. Its geometry columns are trustworthy;
its success column is a hypothesis.

#### The rules now in force

1. **Commit before running a campaign.** A campaign launched from a dirty or
   untracked working tree is not evidence. The runner should refuse, or stamp
   the output `DIRTY` so it can never be mistaken for a result.
2. **Every manifest records its own provenance**: git SHA, `git diff --stat` if
   dirty, the complete settings dictionary, the keypoint set, every tolerance.
   A campaign that cannot name the code that produced it is deleted on sight.
3. **The instrument is versioned separately from the system.** If
   `diagnose.py` changes, every campaign measured with the old version is stale
   and must be marked so, not silently compared against.
4. **Prefer geometry to physics.** Anything answerable from the fitted map --
   aim, determinant, residual, curvature -- should be answered that way:
   milliseconds instead of 25 seconds, deterministic instead of stochastic, and
   with no rollout in between to confound it. Reserve end-to-end runs for
   confirming a result, never for finding one.
5. **Say what a table means.** Every campaign gets a paragraph stating what was
   varied, what was held fixed, what threshold defines each column, and what
   result would confirm or refute the claim. The deleted `index.html` had
   columns named `approach / reach / grasp / lift` and nowhere said what any of
   them measured or what number would count as good.
6. **Change one thing at a time, and never the ruler and the system together.**

#### How those rules are enforced, rather than merely written down

Rules in a notes file get forgotten, which is how this happened in the first
place. All six are now mechanical.

**`tpgpt/reporting/provenance.py` (new).** `provenance()` returns the commit,
the branch, the tracked files modified since it, `git diff --stat`, the
installed versions of the packages that change what a run does, and -- the field
that matters -- **`untracked_code`**, every file under `tpgpt/` or `tests/` that
git has never seen.

That last one is the whole lesson. A plain "is the tree dirty" check would have
declared the deleted campaigns **clean**, because the four files that made them
unidentifiable had never been committed: an untracked file appears in no diff,
so a diff-based check cannot see it. `reproducible` is therefore true only when
there is a commit, nothing is modified, *and* nothing is untracked.

Nothing in the module raises. Git missing, not a repository, a repository in a
state git refuses to answer about -- all of them return "unknown", never
"clean". An absent measurement that reports a passing value is the same failure
as the zeroed `contact_offset` of 7.13, and it is guarded against here by
construction.

**Every manifest carries it, by default.** `write_manifest` records provenance
unless a caller explicitly opts out, because the failure being guarded against
is *forgetting*, and an opt-in guard against forgetting does not work.

**The index says so, loudly.** Each campaign card now opens with a green
"Reproducible at `<sha>`" line or a red **"Not reproducible"** one that names
the specific reason and the specific files, ending "treat these numbers as a
scratch experiment, not as evidence". Previously a reader saw a title and a
timestamp, so eight campaigns produced by four different states of the code
looked identical and directly comparable. They were not.

**The runner checks before it spends the time.** `run_experiments.main` calls
`warn_if_unreproducible()` at the *start*, so a dirty tree costs one line of
output immediately rather than a deleted directory hours later.
`--require-clean` turns the warning into a refusal.

**A campaign now records what it *was*, not only how it went.** The `settings`
field used to hold `{runs, succeeded, stages}` -- outcomes, with nothing saying
what produced them. It now holds `varied` (the campaign's own axis, derived from
the settings that actually ran) and `fixed` (everything held constant), plus the
scene objects, the default keypoint set and the step budget. Outcomes moved to a
separate `results` field, and every stage threshold is written into
`thresholds`, so a reader never has to find `REACH_TOLERANCE` in the source to
know what a column meant.

The `varied`/`fixed` split is aimed squarely at how 7.21 went wrong. Run against
the current campaign definitions, `shelves` reports
`varied: {seed, shelf_variant, slot}` -- which makes it visible on the index
card, without reading any code, that the shelf study also varies the
destination and therefore cannot separate the two.

### 7.27 The reach error was measured with the wrong ruler

The number every end-to-end conclusion rested on was
``||fingertips - planned grasp||`` at the step the jaws were told to close: one
straight-line distance in world coordinates, scored against a single tolerance
of 45 mm.

**It cannot work, and the reason is that a hand is not isotropic.** A gripper
has three axes and being wrong along them means three unrelated things:

| axis | what it means | how much is survivable |
|---|---|---|
| **closing** | across the jaws, along the line the fingers travel | the room left inside the open jaws: ``(aperture - object width) / 2``, so **15 mm** for a 50 mm can in an 80 mm Panda jaw |
| **approach** | along the direction the hand advances | **120-135 mm** on the parallel jaws, and only 15 mm on the UMI (7.2) |
| **jaw** | along the fingers, ``approach x closing`` | on anything symmetric about the closing axis -- a can, a bottle -- nearly free |

A straight-line distance adds those in quadrature, so it mixes a budget of
millimetres with one of 130 mm. A run 60 mm deep and perfectly centred scores
*identically* to one 60 mm off-centre. The first grips the object slightly high
and usually works; the second closes on air. No single threshold can separate
them, because the quantity being thresholded has already thrown away the
distinction.

**The evidence that it was broken, before the campaigns were deleted.** Two
things, either of which should have been enough:

* On the runs that survive in 7.22, the scalar rated the **better** keypoint
  configuration **worse**. The box keypoints score about 53 mm and the contact
  keypoints about 5 mm, and it is the box that is the defensible default.
* Roughly two thirds of the *successful* runs were scored as failing the
  ``reach`` stage. A proxy that fires on a majority of successes is not a strict
  proxy, it is a broken one.

#### What replaces it

``diagnose.reach_axes`` splits the error into the grasp's own frame and returns
each component separately, plus the **sign** along the approach axis -- stopping
short and driving past are different faults and it is worth not discarding
which.

``diagnose.closing_budget`` computes the tolerance rather than assuming one:
``(aperture - width) / 2``, with the width measured **along that grasp's own
closing axis** from the target keypoints. Measuring it from an axis-aligned
bounding box instead is the trap in 7.18 -- a 30 x 100 mm box yawed 45 degrees
measures 92 x 92 and reads as ungraspable. Where the geometry cannot be
determined it returns ``None``, never a plausible-looking default.

The ``reach`` stage now passes or fails on the closing component against that
budget, and falls back to the old scalar only when the grasp frame is unknown,
saying so in its text. The scalar is still recorded, because a metric that
silently changes meaning between two campaigns is its own problem.

> A real bug the tests caught while being written: ``gripper_geometry`` accepts
> only the GraspGen-X name (``franka_panda``), so passing the registry key
> (``panda``) raised, was swallowed, and the budget silently disappeared. That
> is the by-identity lookup of 7.13 for the third time in this project. It now
> resolves through ``resolve_pair`` first.

#### And the drift measure was not a drift measure

``attractor_drift`` replaces a quantity that could not have been right. The old
one compared *how close the attractor came to the grasp* against *how close the
transported plan came to the grasp*, and called the difference drift. It is a
difference of two separately minimised distances: the two minima need not occur
at the same moment, it goes negative whenever the attractor happens to cut a
corner closer than the plan, and it is dominated by whichever curve was sampled
more finely. It routinely came out negative, which should have been the tell.

The replacement measures the two curves against each other, pointwise: for every
attractor sample, the distance to the nearest point on the transported path,
using ``metrics.curves.point_to_curve_distance``. Distance to the **polyline**,
not to the nearest vertex -- vertex distance over-reports by up to half the
vertex spacing, which on a 200-label 20 Hz demonstration is millimetres, the
same size as the thing being measured.

Timing is deliberately excluded: a trajectory that retraces the path exactly but
arrives late has strayed nowhere. "Off the path" and "late" are different faults
with different fixes, and mixing them is what made the earlier numbers
unreadable. ``drift_frechet`` is reported alongside for the order-preserving
view, and the whole thing returns ``{}`` rather than zeros when either curve is
missing, so "not measured" stays distinguishable from "measured as zero".

41 unit tests cover the three instruments and the provenance recorder, asserting
the specific properties the old measures failed: that deviation cannot be
negative, that a point mid-segment is on the path, that a late trajectory shows
no drift, that a deep error does not pollute the closing axis, and that an
untracked source file is fatal to reproducibility.

### 7.28 The jaw reading was metres on one hand and radians on the next

A multi-gripper replay comparison gave the grasp-pose cube 3 of 4 on a Panda and
0 of 4 on both a Robotiq 2F-85 and a Robotiq 2F-140, while the *geometry* for all
three was healthy: `min det(J)` between 0.79 and 0.98, aim 0.0 mm, transported
gripper orientation within 1.1 degrees, and 86 to 98 percent of the path
reachable. Something was going wrong in physics that the map could not see.

The number that looked like the explanation was the jaw trace. Recorded per
waypoint, it read:

| hand | settle steps | jaw min | jaw at lift | held steps | place error |
|---|---|---|---|---|---|
| panda | 8 | 0.0430 | 0.0497 | 148 | 20.6 mm |
| panda | 32 | 0.0010 | 0.0010 | 29 | 382.9 mm |
| robotiq140 | 8 | 0.1999 | **0.6286** | 6 | 336.1 mm |
| robotiq140 | 32 | 0.1905 | **1.3789** | 3 | 399.5 mm |

Read as a width, that says the Robotiq's jaws were **wide open at the lift**, and
opened *further* the longer they were commanded shut. The obvious cause is an
inverted close command, and robosuite's per-gripper `format_action` appears to
confirm it: the Panda's is `current_action + [-1, +1] * speed * sign(action)` and
the Robotiq 2F-140's is `current_action + [+1, -1] * speed * sign(action)` --
opposite signs on the same `+1`. A grep confirmed nothing in `tpgpt/` normalises
that, and three sites issue a raw binary `±1` (`sim/replay.py:166`,
`sim/rollout.py:346`, `controllers/cartesian_impedance.py:261`). The conclusion
drawn was a pipeline-wide sign bug undermining the project's multi-embodiment
claim.

**That conclusion was wrong, and it is withdrawn.** Two independent measurements
say `+1` shuts every hand in the registry.

The first is direct. Mount each hand, hold the arm still, command `-1` then `+1`
then `-1`, and measure the **spread of the moving gripper geoms along the
measured closing axis** -- naming-free, so it needs no per-family special case
and works for three- and five-fingered hands:

| hand | spread at `-1` | at `+1` | reopened | travel | `+1` shuts? |
|---|---|---|---|---|---|
| panda | 103.8 mm | 25.6 mm | 103.8 mm | 78.2 mm | yes |
| robotiq85 | 131.4 | 102.3 | 131.4 | 29.1 | yes |
| robotiq140 | 165.5 | 110.6 | 165.5 | 54.8 | yes |
| rethink | 76.8 | 30.8 | 76.8 | 45.9 | yes |
| xarm | 106.9 | 71.7 | 106.9 | 35.2 | yes |
| umi | 100.0 | 55.7 | 99.9 | 44.2 | yes |
| robotiq3f | 165.9 | 76.4 | 165.8 | 89.5 | yes |
| yumi | 56.9 | 18.5 | 57.1 | 38.3 | yes |
| inspire | 66.2 | 65.5 | 65.4 | **0.7** | nominally |

Every hand's fingers converge on `+1` and return on `-1`, reversibly to within
0.1 mm. The differing `format_action` multipliers are not opposite *commands*;
they are opposite **joint conventions** in the two models, and each hand's
multiplier compensates for its own. Robosuite's uniform docstring, "-1 => open,
1 => closed", is correct for all of them.

The second is a cross-check that was already in the repository.
`grasp/verify.calibrate_depth` sweeps 13 approach depths per hand, drives each
one with `gripper=1.0`, and only stores an offset when the reference can rises
more than `LIFT_THRESHOLD = 50 mm`. `gripper_frames.json` carries a
`calibrated_depth` for eight of the nine hands, the Robotiq 2F-140's among them.
It had already picked the can up on `+1`, months before.

**What was actually broken was the instrument.** `diagnose._jaw_opening` returns
`sum |qpos|` over `gripper.joints`. Its own docstring said "Not a width in metres
-- hands differ", and it was read as one anyway. Measured fully open to fully
closed:

| hand | joints | type | `sum abs(qpos)` open | closed | direction on close |
|---|---|---|---|---|---|
| panda | 2 | prismatic | 0.0794 | 0.0010 | **decreases** |
| umi | 2 | prismatic | 0.0678 | 0.0230 | **decreases** |
| robotiq85 | 6 | revolute | 0.9896 | 1.8056 | increases |
| robotiq140 | 6 | revolute | 0.2352 | 1.9881 | increases |
| xarm | 6 | revolute | 0.2143 | 4.8980 | increases |
| robotiq3f | 11 | revolute | 1.6764 | 7.1434 | increases |
| inspire | 12 | revolute | 5.3075 | 5.8104 | increases |
| rethink | 2 | prismatic | 0.0225 | 0.0237 | **+0.0012: none** |
| yumi | 2 | prismatic | 0.0250 | 0.0250 | **0.0000: none** |

Three separate defects, each enough on its own:

1. **The sign is hand-dependent.** A prismatic finger pair travels toward each
   other, so their positions shrink toward zero and `sum |qpos|` *falls* on
   closing. A revolute linkage folds inward on a *rising* angle, so it *climbs*.
   Two of nine hands go one way and five the other. The Robotiq's 0.63 to 1.38
   was the hand closing **harder**, not opening -- and "harder with more settle
   steps" is exactly what a linkage under a sustained command does.
2. **The units are hand-dependent.** Metres on the prismatic hands, radians on
   the revolute ones. "Shut" is 0.0010 on a Panda and 4.8980 on an XArm, a
   factor of 4900, so no threshold and no cross-hand comparison is possible.
   The Panda's 0.043 and the Robotiq's 0.199 were never comparable quantities.
3. **For two hands there is no signal at all.** The Rethink's joints move
   0.0012 and the Yumi's move 0.0000 while their fingers travel 45.9 mm and
   38.3 mm. Whatever `gripper.joints` names for those models, it is not the
   actuated pair. A hand whose jaw channel is a constant would have been read as
   never closing, on any threshold.

So the physics failure on the two Robotiq hands is **real and still
unexplained**, and the sign hypothesis was a wrong answer built on a broken
ruler. What the table above actually licenses is one narrow statement: the
Robotiq 2F-140 closed. Why it did not then complete the task is open.

#### The fix

`diagnose.jaw_closure_probe(env, gripper)` returns a closure fraction where
**0 is fully open and 1 is fully closed on air, for every hand**, so one
threshold means one thing across the registry. It is built from geom
displacement rather than joint positions, for the reasons above: displacement
needs no joint names, has one sign by construction, and is in metres everywhere.

The calibration -- which geoms are fingers, the closing axis in `grip_site`
coordinates, and the spread at both extremes -- is measured once per hand by
`grasp/measure_frames.py` and cached in `gripper_frames.json` alongside
`alignment` and `contact_offset`. At run time the live `grip_site` rotation is
applied before projecting, so the reading holds with the wrist at any
orientation; the calibration is taken with the arm stationary and reading along a
fixed world axis instead would make a 90-degree wrist roll report the jaws shut.

Values outside `[0, 1]` are **not** clipped. Above 1 means the fingers were
pressed past their free-air closed pose, which is what squeezing an object looks
like, so `closure_max > 1` is positive evidence of a grasp; below 0 means forced
wider than open. Clipping would erase the one signal that separates "holding" from
"shut on nothing". An uncalibrated hand returns `nan`, never 0.0 -- a zero would
read as "wide open throughout", indistinguishable from a hand that never closed,
which is the recurring lesson of 7.13.

`_jaw_opening` is kept, because it needs no calibration and cheaply shows that
*something* moved on a hand already known to work, but its docstring now states
all three defects and says not to threshold it.

**The blast radius was checked and is small.** Nothing in `tpgpt/` ever
thresholded the `jaw` channel or compared it across hands: a grep for reads of it
finds only `reach_axes["jaw"]`, which is the lateral *reach* axis
(`approach x closing`) and an unrelated quantity that happens to share the word.
So the defect never reached a stage attribution, a filter or a success criterion
-- it reached exactly one place, a human reading the trace, and produced one
wrong diagnosis there. That is worth stating because "the instrument was wrong"
and "every number downstream of it is wrong" are very different claims, and only
the first one is true here.

Two incidental fixes came out of the same reading:

- **`measure_frames.main` overwrote `gripper_frames.json` wholesale.**
  `measure_frame` does not produce `calibrated_depth`, which comes from the
  expensive 13-grasp physics sweep, so re-running the frame measurement silently
  emptied it. `grippers._physics_verified` reads that field to decide which hands
  campaigns may use, and **falls back to all measured pairs when the list comes
  back empty** -- so the Inspire hand, which lifts nothing, would have quietly
  re-entered every campaign. `main` now merges.
- **The Inspire hand's fingers travel 0.7 mm.** The registry docstring explains
  its 24 failed grasp attempts as "a five-fingered hand driven by one open/close
  command does not pinch a can from above". The simpler explanation, measured, is
  that it does not actuate: 0.7 mm against 29 to 90 mm for every other hand. It
  is not a grasp-strategy failure, it is a model that barely moves.

#### Two back-end facts checked at the same time, both good

While the sign hypothesis was being tested, two other assumptions behind a
multi-hand campaign were checked. Both hold, and one corrects the documentation.

**Any registered gripper works without restarting the server.** `CLAUDE.md` said
only `franka_panda`, `robotiq_2f_85` and `robotiq_2f_140` were available and that
"other grippers need the server restarted with them". That is wrong:
`GraspGenX/graspgenx/serving/zmq_server.py:110` loads a sampler **lazily** on the
first `infer` request naming a gripper. Asked for five it had never served, on the
same 1500-point synthetic cylinder cloud, it returned a full candidate set for
each:

| gripper | grasps returned | wall time |
|---|---|---|
| `sawyer_hand` | 100 | 25.4 s |
| `xarm_hand` | 100 | 27.7 s |
| `franka_umi` | 100 | 30.0 s |
| `abb_yumi` | 100 | 30.5 s |
| `robotiq_3f` | 100 | 71.5 s |

Against the usual 4-12 s per inference, so the first call for a hand costs
something, but no restart and no re-plan of the campaign. `python -m
tpgpt.grasp.server` reports `loaded grippers`, and that is a record of what has
been *asked for*, not a whitelist.

**And the server honours `gripper_name` rather than falling back to its
default.** Worth checking, because immediately after the five calls above
`loaded_grippers` still reported only the original three, which is exactly what a
silent fallback to `franka_panda` would look like -- and a silent fallback would
have made a nine-hand campaign into one hand run nine times, the same failure as
the `build_scene` hardcode. (After the nine-hand sweep it reports all nine, so
that reading was transient rather than a real symptom; the check below was run
before that was known and stands on its own regardless.)

The test is geometric. A grasp pose is anchored at the gripper *base*, which sits
`tcp_depth` back from the fingertips along the approach, so a deeper hand's base
poses must sit systematically further from the object. Same cloud, 100 candidates
each, mean distance from the base pose to the cloud centroid:

| hand | published `tcp_depth` | base to centroid | sd |
|---|---|---|---|
| panda | 103.4 mm | 135.1 mm | 17.9 |
| yumi | 125.0 mm | 153.2 mm | 18.6 |
| robotiq85 | 136.0 mm | 157.2 mm | 17.5 |
| robotiq3f | 190.0 mm | 189.8 mm | 15.8 |
| robotiq140 | 195.0 mm | 205.9 mm | 17.5 |

The standoff tracks the published depth monotonically across a 92 mm spread, at a
per-hand spread of under 19 mm. A fallback would have given five statistically
identical rows. The gripper is honoured.

#### Why this took a campaign to find, and what now prevents it

This is the fourth harness bug in this thread found *after* a 12 to 25 minute
physics campaign had produced plausible numbers. The others: labels replayed in
the wrist frame instead of the tool frame (the cube held the object for 0 to 1 of
119 steps); no `env.reset()` between replays, so ordering decided the result
(placement errors of 354, 698 and 1170 mm, 0 percent reachable on the last
object); and `build_scene` hardcoding `robots="Panda"` with no `gripper_types`,
so a three-hand comparison ran a Panda three times and nothing in the numbers
looked wrong.

Every one of those four is answerable in **milliseconds** from a freshly built
environment. The cost was paid because the campaign was the first test of the
harness, which entangles "is the harness right" with "what is the answer" and
pays physics time to discover a one-line bug.

`diagnose.replay_preconditions` now checks four statements on **every** cell
before any physics runs, and `diagnose.require` aborts the campaign on a failure,
printing every check and what it measured -- passed ones included, because a
green check that reports nothing cannot be distinguished from a check that did
not run, which is precisely the 7.19 failure:

| check | catches |
|---|---|
| `gripper_mounted` | the hand on the arm is not the one requested |
| `scene_unstepped` | `sim.data.time != 0`, so this environment has already been driven |
| `object_placement` | the object is not where the same seed put it on the first cell |
| `closure_calibrated` | the hand has no measured closure calibration, or `+1` is not known to shut it |

24 unit tests cover the two new instruments, asserting the properties the old
measure failed: one sign across a closing sweep, the same fraction at two hands'
different midpoints, a squeeze reported above 1 rather than clipped, `nan` rather
than 0.0 for an uncalibrated hand, an unchanged reading under a 90-degree wrist
roll, and each of the four preconditions catching its own historical bug.

### 7.29 The grasp centre is a good enough representation of a grasp

The grasp cube is centred on the grasp point with a **fixed** 20 mm half extent
and encodes nothing about the hand -- not the jaw aperture, not the fingertip
depth, not the finger count. Every keypoint result before this was measured on a
Panda, so the cross-embodiment claim rested on an untested assumption.

A hand can reach the map through exactly two channels, and both are *inputs* to
it rather than parameters of it: which grasp GraspGen-X returns for that hand,
since the planner conditions on its swept volume; and the tool offset the labels
are expressed in, `contact_offset(hand)`, which spans **24.3 mm (yumi) to
134.4 mm (inspire)**, a factor of 5.5. So a construction that quietly depended on
the hand would show `min det(J)`, the aim or the transported orientation moving
with one of those. That is the test.

All nine registered hands, five objects, four constructions, real cached
GraspGen-X candidates per hand at mid grasp height, geometry only. 150 cells,
**134 scored**. `outputs/keypoints_grippers/`.

**Pooled, over every hand and object:**

| construction | n | `min det` med | `min det` min | aim med | orient med | orient max | resid med |
|---|---|---|---|---|---|---|---|
| cloud box | 35 | 0.629 | 0.198 | 62.2 mm | 6.10 deg | 38.2 deg | 0.14 um |
| task-frame cube | 35 | 0.961 | 0.549 | 0.0 mm | 4.14 deg | 39.4 deg | 0.65 um |
| **grasp-pose cube** | 35 | 0.916 | 0.555 | 0.0 mm | **0.82 deg** | **1.8 deg** | 0.53 um |
| composed | 29 | 0.456 | 0.102 | 6.8 mm | 1.11 deg | 10.5 deg | **10 864 um** |

`det(J) > 0` holds on every cell of all four. The residual gate of 1e-5 is met by
35/35 of each of the first three and by **0/29** of the composed variant.

#### The answer: yes, on three independent measurements

**1. The transported orientation is hand-independent and an order of magnitude
better.** The grasp-pose cube's per-hand median runs **0.5 to 1.6 degrees** across
all nine hands, worst single cell **1.8 degrees**. The cloud box, on the same
grasps and objects, runs 5.4 to 14.3 with a worst cell of 38.2. Seven times better
in the median, twenty times at the tail, and the spread across hands is eight
times tighter (sd of the nine hand medians: 0.33 deg against 2.75).

**2. Conditioning does not track the tool offset.** Correlating each hand's
median `min det(J)` against its offset, over the nine hands:

| construction | Pearson r(tool offset, median `min det`) |
|---|---|
| cloud box | **-0.616** |
| grasp-pose cube | **+0.135** |

The cloud box degrades as the hand gets deeper, which is mechanically sensible:
a deeper hand puts the label path further from the object's centroid, and that
centroid is where its box is centred. The cube shows essentially nothing across a
5.5x range. The UMI at 117.2 mm scores 0.889 and the Inspire at 134.4 mm scores
0.956 against the Panda's 0.693 at 41.1 mm -- the *source* hand is not the best.

**3. The object moves the map more than the hand does.** Taking each hand's
median across its objects and each object's median across its hands:

| construction | metric | sd of 9 hand medians | sd of 5 object medians | ratio |
|---|---|---|---|---|
| cloud box | `min det` | 0.058 | 0.208 | 3.6x |
| grasp-pose cube | `min det` | 0.077 | 0.152 | 2.0x |
| cloud box | orient | 2.749 deg | 4.616 deg | 1.7x |
| grasp-pose cube | orient | **0.332 deg** | **0.465 deg** | 1.4x |

A supporting measurement rather than the main one -- 2.0x is not enormous. The
decisive numbers are in point 1, where the *absolute* level differs sevenfold and
not merely the spread.

**Caveat, stated rather than buried.** `r(aperture, median orientation error) =
+0.499` for the grasp-pose cube: a wider jaw correlates with slightly worse
orientation. The whole correlated range is 0.5 to 1.6 degrees, against a quantity
the cloud box gets wrong by up to 38, so the effect is real and negligible.

**This is geometry, not execution.** It says the map is well conditioned, exact
at its keypoints and correctly oriented for every hand. It says nothing about
whether the arm can follow the resulting path or the hand can hold the object.
That is Tier 2 and it is not answered here.

#### Composition is dead, and this is what killed it

It **violates property (i)**: median keypoint residual **10 864 um -- 10.9 mm** --
against 0.14 to 0.65 um for the other three, with **0 of 29 cells** meeting the
gate all 105 cells of the other three meet. A map that misses its own keypoints by
a centimetre has no exactness guarantee left to spend, which is the entire reason
Sec. III-D interpolates them exactly.

It also refuses on 6 of 35 cells, `fit_local_correction` raising because the
correction exceeds its own 30 mm locality: 35.8 mm (robotiq140/cereal), 38.4
(rethink), 43.1 (umi), 49.1 (inspire), 72.7 (yumi/cereal), 30.8 (yumi/lemon).
The guard added in the earlier round is doing exactly its job -- without it those
would have been silent 30-70 mm deformations. And it loses the aim it existed to
keep: 6.8 mm median against the plain cube's exact 0.0.

#### A gap this exposed: a cube cannot tell you the object's width

`diagnose.closing_budget` derives the aiming tolerance as
`(aperture - object width) / 2` from the target keypoints. With a fixed grasp
cube that cannot work, and the sweep shows it plainly -- width read from the
keypoints, averaged per object:

| construction | width read from the keypoints |
|---|---|
| cloud box | bread 43.6, can 45.2, cereal 46.2, lemon 26.4, milk 60.4 mm |
| **grasp cube** | **40.0 mm for every object** |

40.0 mm is the cube's own 2 x 20 mm extent. **This is not a defect in the
construction** -- removing the object's size from the keypoints is exactly what
kills the volume scaling of 7.22 -- but it means the size cannot be read back out
of them, and a budget derived from a cube gives a lemon and a milk carton the
same tolerance.

Fixed by recording `metrics["object_width_closing"]` from the **cloud**, at the
one point in `pipeline.run` where both the cloud and the chosen grasp are in
scope. `closing_budget` prefers that and returns `None` for a cube set without
it, rather than reporting the cube.

A second and more serious defect in the same function was found at the same time:
it had been taking the extent over **all** of `target_keypoints`, which holds the
placed block as well as the picked one, so the "width" was the pick-to-place
distance (239-277 mm) against an aperture of at most 125 mm and the budget
clamped to **0.0 for every object and every hand**. Zero is a plausible number,
which makes it worse than `None`: indistinguishable from a genuinely impossible
grasp, and precisely the failure this function's `None` return was written to
prevent (7.13). It was invisible because every test fake in `test_diagnose.py`
carried `.points` and no `.labels`, so there was no placed block to exclude --
the fake was simpler than the object the function actually receives.

#### Two systematic skips, one of which corrects an earlier reading

| skip | cells | cause |
|---|---|---|
| lemon | 8 of 9 hands | GraspGen-X raises `selected index k out of range` on its 17-point cloud, below `MIN_CLOUD_POINTS` of 40 and too few for the planner's top-k. **Hand-independent** |
| milk | panda and inspire only | every one of 100 candidates lies beyond the 45 deg approach filter, closest 49.8 deg. **Hand-dependent and scene-dependent** -- see below |

The milk had been recorded as failing the filter outright. That was true of the
one hand it had been measured on. But the correction needs a correction of its
own, found while running Tier 2.

**The rejection is not a property of the milk.** Within this sweep seven of nine
hands found a milk grasp inside the filter and two did not, which reads as a
hand effect -- the planner conditions on each hand's swept volume. Then the Tier
2 replay, on a **Panda**, picked and placed the milk at 14.8 mm: the same hand
this sweep recorded as having no admissible candidate.

The difference is the scene. Tier 1 builds a **five-object** scene
(`ABLATION_OBJECTS`, with the lemon); Tier 2 builds a **four-object** one
(`REPLAY_OBJECTS`, without it). The placement sampler is seeded identically, but
a different object set lays the scene out differently, so the milk presents a
different cloud and the planner draws from a different candidate set.

So: **whether an object survives the approach filter depends on the hand and on
what else is in the scene.** Neither table alone licenses "the milk cannot be
grasped", and any claim about an object failing a filter must name the scene it
was measured in. This is the 7.21 lesson again -- an isolation study is only as
good as the settings held fixed around it -- arriving through a door nobody was
watching, since the object *set* had not been thought of as a setting at all.

#### One more thing the rebuilt-per-hand scene revealed

**The same seed does not settle the objects identically across hands.** The
placement sampler is seeded the same, but the scene's 60 settle steps then run
with a different gripper attached, and the can comes to rest at
`[-0.1255, -0.0683, 0.8399]` with a Panda against `[-0.1139, -0.0777, 0.8426]`
with a Robotiq 2F-140 -- **14.6 mm apart**. Every cross-hand comparison carries
that difference by construction. It is recorded per row as `object_position`
rather than assumed away, and the reproducibility precondition is keyed per hand
so it does not report this as a failure.

### 7.30 The map transports across hands; the execution does not

Tier 2, the physics half of 7.29. The transported path is followed pose by pose
under position control -- IK per waypoint, a stiff `JOINT_POSITION` controller,
**no GP policy, no attractor integration, no lag gate** -- so every result is
attributable to the plan rather than to the executor. Replay is an upper bound:
what fails here is the keypoints' or the frame's; what fails only under the policy
is the dynamics thread's.

Six hands spanning 24.3 to 117.2 mm of tool offset and 50 to 125 mm of aperture,
four objects, two constructions. 48 cells, **46 ran**. Real GraspGen-X candidates
per hand, full 6-DoF pose used as-is. `outputs/keypoint_replay_grippers6/`,
commit `e07c738`, `reproducible: True`.

#### The result

| hand | closing angle | aperture | tool offset | success | median `min det` |
|---|---|---|---|---|---|
| **panda** *(the source hand)* | 0 deg | 80 mm | 41.1 mm | **5/8** | 0.932 |
| yumi | 0 deg | 50 mm | 24.3 mm | 2/6 | 0.706 |
| robotiq85 | 0 deg | 85 mm | 47.8 mm | 1/8 | 0.667 |
| robotiq140 | 0 deg | 125 mm | 60.8 mm | 1/8 | 0.788 |
| xarm | -90 deg | 85 mm | 26.7 mm | 0/8 | 0.700 |
| umi | -90 deg | 80 mm | 117.2 mm | 0/8 | 0.684 |

**Only the hand the demonstration was recorded on works: 4 successes in the 38
cells on every other hand.**

The two constructions **tie**: cloud box 4/23, grasp-pose cube 5/23. The cube's
3-of-4 against 2-of-4 from the Panda-only run does **not** generalise, and that
earlier claim is superseded.

#### They tie on the score and differ completely in how they fail

| stage the cell died at | cloud box | cube |
|---|---|---|
| path <50% reachable | 4 | 5 |
| reachable but **never touched the object** | **11** | **3** |
| brief contact (<100 steps) | 5 | 5 |
| firm grip (>=100 steps) | **3** | **10** |
| of those firm grips, succeeded | **3/3** | **5/10** |

**The cube acquires the object 3.3x more often and then drops half of them.** It
converts an aiming problem into a holding problem. Contact rate overall: cube
15/23 cells, cloud box 8/23.

#### Four mechanisms

**1. The path is unreachable (9 cells, 0 succeeded).** Entirely the UMI, 8 of its
8 cells, at 0-30% reachable and 44-164 mm of tracking error. The transported
labels are a *fingertip* path and IK must solve for the **wrist**, which sits
`contact_offset` behind them: `wrist = target - R.offset`. The UMI's offset is
117.2 mm, nearly 3x the Panda's, so every waypoint asks the wrist 117 mm further
back and out of the arm's envelope. Not a keypoint failure and not a grasp
failure -- the frame conversion doing exactly what it should, on a hand this arm
cannot accommodate.

Note the direction: the cube is *worse* than the cloud box on UMI reachability in
all four objects (30->4, 20->0, 24->6, 30->26). The cube aims exactly at the
planned grasp so it inherits the full setback; the cloud box's 52-108 mm aim
error pulls the path somewhere more nearly reachable. **Being right about the
target is a disadvantage when the target is unreachable.**

**2. The hand arrives and the object is not between the fingers (14 cells, 0
succeeded).** Eleven are the cloud box, and there the cause is measured: it aims
wrong. Median `aim` 73.2 mm in the cells where it never touched, against 42.9 mm
where it did. A plan passing 73 mm from the intended grasp puts the fingers 73 mm
from the object and a Panda jaw is 80 mm wide. **This is the non-tautological half
of 7.29 arriving in physics.**

The three cube cells in this category are **all the cereal**, on three different
hands, and they are the important ones:

| hand | reach | track | `aim` | `orient` | `min det` | held |
|---|---|---|---|---|---|---|
| xarm | 84% | 4.7 mm | 0.0 mm | 0.6 deg | 0.832 | **0** |
| robotiq85 | **100%** | **5.7 mm** | **0.0 mm** | **0.4 deg** | 0.864 | **0** |
| robotiq140 | 97% | 12.0 mm | 0.0 mm | 0.5 deg | 0.790 | **0** |

Every quantity the map controls is perfect on the robotiq85 row -- fully
reachable, tracked to 5.7 mm, aimed exactly, oriented to 0.4 degrees -- and the
hand never touches the cereal. **This is the clearest evidence in the project that
matching TCP and orientation is not sufficient for a grasp.**

The mechanism is **not established**. The candidate is that the hand's *body*,
not its fingertips, contacts the cereal during the approach and pushes it away:
the cereal is the tallest object and these three hands are 145-270 mm deep.
Testing it needs the object's position trace during the approach, and this run
**did not persist the probe traces** -- `replay_variant` extracts summary scalars
only. That is an instrumentation gap and the first thing to fix.

**3. Firm grip, then the object slips out (5 cells, all the cube).** Of 13 firm
grips, 8 succeeded; the 5 failures are all the cube and slip separates them:

| | n | median slip | median track | median place |
|---|---|---|---|---|
| firm grip, succeeded | 8 | **18.4 mm** | 20.2 mm | 31.5 mm |
| firm grip, failed | 5 | **50.1 mm** | **9.0 mm** | 136.3 mm |

The tracking error is *lower* in the failures, so this is not the arm missing the
path: it follows well, holds for over a hundred steps, and the object rotates out
anyway. The obvious explanations do not survive: `r(tilt_mid_path, slip) = +0.352`
and the cube's median mid-path tilt among firm grips is **6.7 deg against the
cloud box's 10.8** -- it tilts the transit *less*; `r(min_det, slip) = +0.071`;
`r(tracking, slip) = +0.026`. `r(orientation_error, slip) = -0.521` is a confound,
since the cube owns both the low orientation errors and all five slips.

The testable hypothesis: **the cloud box's aim error may be accidentally helping.**
Its three firm grips succeeded 3/3 with aim errors of 26.5, 43.8 and 87.1 mm --
it grips somewhere other than the planned point, plausibly nearer the centre of
mass where the gravity torque about the grip is smaller, while the cube grips
exactly where the planner said, which may be a pinch on a narrow face. Comparing
grip point to centroid settles it. **n = 3, so this is a hypothesis.**

**4. The jaw is too narrow (1 clear cell).** `yumi/can/cube`: held 6 steps, placed
360 mm away. Aperture 50 mm against a can measuring 45.2 mm along the closing
axis gives `(50 - 45.2)/2 = 2.4 mm` per side; every other hand has 10-40 mm. Real,
invisible to the keypoints, and **not the driver of the run**: success against
aperture is not monotone, the two 80 mm hands being 5/8 and 0/8.

#### What this does to 7.29

**The geometry does not predict the physics.** Over the 23 cube cells:
`r(min_det, held) = -0.071`, `r(aim, held) = +0.015`,
`r(orientation_error, held) = -0.027`, `r(tilt_mid_path, held) = +0.106`. All
approximately zero.

And a criticism of 7.29 that should have been stated there: **for the cube
variants `aim` is ~0 and `orient` is ~0 by construction.** The cube's centre *is*
the target grasp point and `phi` interpolates keypoints exactly, so the aim is
guaranteed. The corners are laid out in the source and target grasp frames, so
`J_perp` is asked to recover a rotation built into the keypoints, and it nearly
does -- a self-consistency check on cube size, which is why size moved it from 0.1
to 20.6 degrees, not a fact about the world.

What in 7.29 stays informative: `min det(J)` (nothing pins it, and the cloud box
really folds where the cube does not), mid-path tilt and lift deviation (far from
any keypoint), **every cloud-box number** (nothing pins those either, which is what
makes the aim comparison above meaningful), and the composed variant's 10.9 mm
residual.

**The two tiers read together:** the transportation map is sound and
hand-independent, and that is necessary and nowhere near sufficient. Execution is
limited by what the map does not model -- the hand's setback from the labels, its
bulk during approach, its jaw width, and the stability of the grip it achieves.

#### What would make this false

- **One seed, one slot, one demonstration.** The hand-versus-object confound
  cannot be separated without more scenes.
- **The Panda advantage may be a grasp-selection artefact.** The demonstration was
  recorded on a Panda and candidates are filtered to within 45 degrees of *its*
  approach; the Panda's median `min det` is 0.932 against 0.667-0.788 for the
  others, so it may simply be getting better target grasps rather than executing
  better. Testing needs a demonstration recorded on another hand, which does not
  exist.
- **Mechanisms 2 and 3 are described, not diagnosed.** Both need the per-waypoint
  object trace this run did not save.
- **`n` is small everywhere** -- 3 cloud-box firm grips, 5 cube slips, 1 aperture
  case. None of the per-mechanism claims is offered as statistically significant.
- **Replay is not the policy.** These are upper bounds under position control; the
  GP policy with an impedance controller and a lag gate will do worse.

## 8. Open items

> **Read 7.26 first.** Every end-to-end campaign has been deleted, so the items
> below are stated as *design questions with a proposed measurement*, not as
> conclusions with a number. Nothing here should acquire a number until the
> provenance rules at the end of 7.26 are in force.

**The two open threads, and they are independent of each other.** The interface
between them is a single object -- the transported label set. The keypoints and
the map *produce* it; the policy and the rollout *consume* it. Neither needs the
other to be correct, which means they can be built and tested separately and
should be.

**Thread A -- keypoints and the map. Pure geometry, no simulator needed.**

The trade measured in 7.22 is local precision against global conditioning. The
box keypoints keep the map well conditioned everywhere (`min det(J)` ~ 0.53) and
miss the grasp by about 53 mm. The contact keypoints hit the grasp to about
5 mm and leave the map either folded (with the box) or compressing volume
fivefold (without it). Both requirements are legitimate; no keypoint set on the
current menu satisfies both.

The arithmetic in 7.22 says why, and bounds any fix: reconciling a
`d` = 53 mm disagreement needs `L` > ~66 mm of space to blend over, and the
contacts currently sit 20-40 mm from the box corners they contradict. Options,
cheapest first, all of them answerable offline from a cached cloud and a cached
grasp in milliseconds per variant:

- **Correct the trajectory, not the map.** Post-adjust the transported labels so
  the path passes through the known target grasp pose, decaying over ~7 cm.
  Not a map, so there is no determinant to fold. Directly analogous to
  `carry_transform`, which already derives the placed pose outside the map.
  This is the right *first* experiment because it isolates the question 7.22
  leaves open -- does better aim convert to success at all? -- before anything
  is paid for it.
- **Delete the box corners that conflict.** 7.22's own diagnosis is that the
  contacts are redundant with corners centimetres away. If they are redundant,
  drop the corner and keep the contact rather than interpolating both. Raises
  `L`, preserves exact interpolation.
- **Partial interpolation.** Regularise the map's own GP so it fits keypoints in
  a least-squares sense instead of exactly. Regularisation is the standard cure
  for this ill-conditioning. Costs property (i) of Sec. III-C, but as a
  continuous knob: a valid map aiming at 10 mm may beat a folded one aiming at
  5 mm.
- **A localised correction term.** The map is already `phi = gamma + psi . gamma`
  ([maps.py](tpgpt/transport/maps.py)). Add a third, compactly-supported term
  centred on the grasp contact. Because the support is compact, its influence
  mid-trajectory is exactly zero rather than merely small, and the no-folding
  condition is closed-form -- check it and shrink the correction until it passes,
  instead of fitting a map and discovering afterwards that it folded.
- **Blending two whole maps** (contacts-fitted and box-fitted) is the same idea
  with a weaker guarantee: the Jacobian picks up a rank-1 term that can only be
  checked by sampling. If it is done, the blend weight must be a function of
  **position**, not phase -- a phase-dependent weight is not a spatial map, and
  then `J(x)` is undefined, which breaks the velocity, orientation, stiffness
  and uncertainty transports of Eqs. 9-13 that all depend on it.

Also unresolved and symmetric: **nothing pins the map at the placement end
either.** If precision at the grasp matters, precision at the insertion matters
equally.

**Thread B -- policy execution. Needs the simulator, but not the keypoints.**

Testable against a *fixed* reference path -- the untransported demonstration
replayed in the source scene, where the ground truth is known exactly -- so it
is completely decoupled from thread A. Open questions, with the analysis behind
them in 2.7 and 2.8:

- The rollout integrates `prediction.velocity` and **never reads
  `prediction.reference`**, the regressed attractor-position channel that
  [gp_policy.py](tpgpt/policy/gp_policy.py) already fits on the transported
  label positions. That channel is the paper's own Sec. V formulation and the
  only restoring term the policy has -- the velocity field, having a zero-mean
  prior, points *along* the demonstration and never back toward it.
- `GPPolicy.attractor()` returns `reference + K^-1 D velocity` and is called
  only from a unit test. The `K^-1 D velocity` term is lag compensation and is
  correct **only against measured-pose labels**; against this project's
  attractor labels (2.7) it would land the arm `K^-1 D v` ~ 24 mm *ahead* of
  where the demonstrating arm was. Use `reference` alone, or change the labels.
- The controller accepts `velocity_desired` and nothing ever passes it, which is
  the entire reason a steady-state lag exists (2.8). Whether to cancel the lag
  or keep it is a real design choice, not an oversight -- see 2.7 for why the
  lag transports correctly on its own and may be the desired behaviour.

**Instrumentation, which blocks both threads.** The reach error is currently a
scalar 3-D distance to one planned grasp point. Only the component along the
gripper's **closing axis** decides whether the object ends up between the jaws;
the approach axis tolerates 120-135 mm on a parallel jaw (7.2). Decompose it by
axis before running anything. Attractor-versus-plan drift needs a pointwise or
Fréchet measure -- `frechet_distance` in
[metrics/curves.py](tpgpt/metrics/curves.py) already exists and is unused --
rather than the difference of two separately-minimised distances.

**Also open:**

- **A front-approach demonstration** for a shelf with a roof. A top-down teach
  cannot solve one by construction: the approach direction is wrong, and no
  amount of warping fixes a direction the demonstration never contained. This is
  geometry, not a measured failure, which is why it survives the withdrawal of
  7.23.
- **The UMI hand.** Its contact offset `[0.0, -0.035, -0.112]` is the only one
  in the registry with a large *lateral* component; every parallel jaw is almost
  purely along the approach axis. 7.2 also measured it tolerating only 15 mm of
  depth error against 120-135 mm for the parallel jaws. Both are properties of
  the hand, independent of any campaign, and both are worth checking before
  anything else about that gripper.
- **Grasp filtering and selection.** Section 5.6 is the measured case. At
  minimum this needs a visibility criterion (the cloud is one-sided, and
  approaches from the unobserved side dominate), reachability, and collision
  against the rest of the scene. Whether to filter, re-rank, or fuse more views
  at source is the design question.
- ~~**Keypoint extraction for the new scene.**~~ Done; see section 6.

**Deferred, and why:**

- **Physics execution in `TabletopShelf`.** Section 6.3 is scored
  geometrically. Executing there needs `reshelving_waypoints` generalised into
  a grasp-and-place-pose builder and `rollout_policy`'s success readout made
  scene-agnostic -- it currently reads `env.product_position`,
  `env.goal_position` and `env._check_success()`. `TabletopShelf` already
  provides `is_object_in_slot` and `slot_poses` for it. Kept separate so a
  keypoint failure and a physics failure cannot be confused for one another.

**Deferred from the transportation work:**

- Sec. IV baselines (Reshaped-KMP, Laplacian Editing, LWT, Ensemble-NN,
  Ensemble Neural Flows, TP-GMM, TP-HMM), Table I, Figs. 6-10 and the
  Mann-Whitney ranking. `tpgpt/metrics/` already provides everything they need.
- Sec. V-B dressing (MuJoCo `flexcomp` verified viable on CPU) and Sec. V-C
  surface cleaning, which would most exercise SV-GPT and stiffness transport
  together.
- Sec. V-D DINO keypoint correspondences.
- The three step-budget failures in section 4.5; they are execution timeouts,
  not transport failures.

**Smaller items:**

- Three gripper pairs have unmeasured `grip_site` frames and currently refuse
  conversion: the two three-finger hands and the Yumi (section 5.4).
- Only `franka_panda`, `robotiq_2f_85` and `robotiq_2f_140` are loaded on the
  running GraspGen-X server; the other six pairs need it restarted to load them.
- `birdview` contributes almost nothing to the fused clouds (8 of 2056 points on
  the cereal box). The default camera set is worth revisiting alongside the
  visibility discussion.
