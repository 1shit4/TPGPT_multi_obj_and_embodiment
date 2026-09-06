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

## 6. Open items

**Next, and needs discussion before implementation** (the flow for a new scene
is: prompt -> object and destination -> grasps -> **filter, choose one** ->
**keypoints** -> transport -> execute; the two bold stages are open):

- **Grasp filtering and selection.** Section 5.6 is the measured case. At
  minimum this needs a visibility criterion (the cloud is one-sided, and
  approaches from the unobserved side dominate), reachability, and collision
  against the rest of the scene. Whether to filter, re-rank, or fuse more views
  at source is exactly the design question.
- **Keypoint extraction for the new scene.** The reshelving keypoints are the
  eight corners of a box. An arbitrary mesh has no such natural set, and the
  keypoints must be *paired* between source and target scenes for the
  transportation map to be defined at all. What plays the role of a corner for
  a milk carton against a cereal box is the open question.

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
