# The dynamics back-end: which attractor law to execute a transported policy with

**Status:** complete for everything measurable without the simulator. Two
simulator tiers are specified and queued (see *What is still missing*).
**Code:** branch `dynamics-execution`, worktree `/home/ishita/TPGPT-dynamics`.
**Raw data:** `outputs/dynamics/rows.json` (352 runs),
`outputs/dynamics_loaded/rows.json`. **Reproduce:**
`python -m tpgpt.experiments.sweep_execution --out outputs/dynamics`.

---

## 1. What question this answers

When the robot executes a transported policy it does not follow a path
directly. It chases a single moving point called the **attractor** — think of an
invisible spring, one end bolted to the gripper and the other held at the
attractor, so the further apart they are the harder the arm is pulled. Executing
a policy means deciding, every 50 ms, where to put that point.

The shipped rule is dead reckoning: ask the learned policy which way to go, and
step the attractor that far.

```
attractor += velocity × dt × gate
```

There are two obvious things that might be wrong with it, and this study
settles both.

**First, the policy is asked "which way from *here*?" at the attractor, not at
the arm.** The arm is somewhere else — behind, as we will see, necessarily so.
Evaluating a policy at a point the robot is not at looks like a mistake.

**Second, the rule has no memory of the path.** Each step is added to the last,
and nothing ever compares the running total against the path it was meant to
retrace. Errors can only accumulate. Meanwhile the policy already computes a
quantity that *is* an absolute statement of where the hand should be — a channel
called `reference`, fitted from the transported label positions, which is the
paper's own Sec. V formulation. It has been computed and thrown away since the
day it was written.

So: **five candidate laws, two query sites, and the question of which to ship.**

---

## 2. The answer, up front

**Keep the shipped law. Keep querying at the attractor.** Both alternatives are
measurably worse, and one of them is worse by an order of magnitude.

| | verdict | evidence |
|---|---|---|
| **Query at the attractor vs. the measured pose** | **attractor**, decisively | querying at the measured pose is 8–48 mm worse in every conditioning regime, `p ≤ 0.03` throughout |
| **V (integrate) vs. VR (anchor) vs. R (reference)** | **V**, the shipped law | every alternative is significantly *worse* where the map is well-behaved; none is significantly better anywhere |
| **Should the anchor be gated?** | immaterial | with the gate shut on >50% of steps, gating changes the final lag by <0.2 mm |
| **Is the dwell creep a defect?** | **no** | 0.00 mm on the axis that decides the grasp; it was a measurement artefact |

That is a negative result, and it is worth stating plainly: **the execution
layer was already right.** The work's value is that this is now *known* rather
than assumed, that the machinery to re-decide it exists when the keypoint work
changes the conditions, and that `rollout_policy` has direct tests for the first
time in its life.

It also **overturns a claim I made earlier in this project's planning**, from a
prototype with one sample per condition. That prototype suggested a
speed-scheduled anchor cut drift by 40% on aggressive warps. With 43 warps and a
paired test, the effect is 0.49 mm and `p = 0.44` — noise.

---

## 3. The finding that dissolved: dwell creep

This one deserves to lead, because it is a worked example of the project's own
recurring failure mode catching its own author.

**The setup.** The demonstration deliberately *dwells*: at the grasp it holds
the attractor at one point for 15 control steps (0.75 s) while the jaws close,
and again at the release. That pause is how the arm is given time to arrive
before the gripper acts. During it the labels command **exactly zero velocity**.

**The alarming measurement.** The fitted policy does not command zero there — it
commands 0.006 to 0.041 m/s, because a Gaussian Process smooths the corners of
the sharp stop-and-go. Integrate that and the attractor travels **11.4 mm during
a window where it should stand still**, at exactly the moment the jaws close. I
reported that twice as the one clear defect in the execution layer.

**The correction.** 11.4 mm is *path length* — total distance travelled. The
**net displacement**, how far it actually ended up from where it started, is
**4.64 mm**. The attractor wanders and comes back. And when that 4.64 mm is
split into the gripper's own axes:

| axis | what being wrong along it means | how much room there is | measured |
|---|---|---|---|
| **closing** | across the jaws — the object ends up outside the fingers | **~15 mm** for this box in a Panda jaw, `(80 − 50)/2` | **0.00 mm** |
| **approach** | along the direction the hand descends — grips slightly higher or lower | 120–135 mm (§7.2) | **4.64 mm** |
| jaw | along the fingers — free on anything symmetric | large | 0.00 mm |

**Every millimetre of it is on the axis with 120 mm of tolerance, and none on
the one with 15 mm.** The dwell creep is not a defect. It never was.

The instrument that showed this is `reach_axes`, which I built last session for
precisely this class of error (§7.27 — a scalar reach distance that mixed a
millimetre-scale budget with a 130 mm one, and consequently ranked the *better*
keypoint configuration worse). It then caught me making the same mistake.

**The release dwell**, measured here for the first time, behaves identically:
3.47 mm net, 0.00 mm on the closing axis.

![Dwell creep decomposed](figures/fig_dwell_axes.png)

*Figure 1. **Left:** for each law, how far the attractor travelled during the
grasp dwell (grey) against how far it actually moved (blue). The gap between
them is the wandering-and-returning. **Middle and right:** that net movement
split into the hand's own axes for the grasp and the release dwells. Red is the
closing axis, which decides whether the object ends up between the fingers and
has about 15 mm of room; green is the approach axis, which tolerates 120–135 mm.
The dashed line is 4.8 mm, the demonstration's own residual lag at the end of
its dwell — the tightest precision anchor in the project. Read it as: the red
bars are the ones that matter, and they are invisible.*

Note in passing that the anchor laws *reduce* the net displacement (4.64 → 1.3–3.3 mm)
while *increasing* the closing-axis component (0.00 → 0.09–0.68 mm). They
improve the number that does not matter and worsen the one that does. Both are
far inside budget, so this decides nothing on its own — but it is the first hint
of the pattern the statistics confirm below.

---

## 4. How the laws were compared

### 4.1 The constraint that shaped everything

A parallel effort is redesigning the keypoints and the transportation map, and
that question is **unresolved**. So no measurement here may depend on the map
being correct, or it inherits whatever that work later changes — which is
exactly how three findings in `ROBOTICS_NOTES` came to outlive the settings they
were measured under (§7.26).

Three things follow:

1. **The demonstration is the real recorded one**, loaded from
   `outputs/reshelving/source_labels.npz` — 200 labels, both dwells, the real
   speed profile, and its own recorded arm trace. Not a synthetic stand-in.
2. **The warps are synthetic and ours.** A lattice of our own choosing is
   displaced by a known analytic field of Gaussian bumps, with the amplitude
   forced to zero near the grasp and the release. **Pinning those two poses**
   keeps the task meaningful — the object and the slot stay where the labels aim
   — while the interior deforms as much as we ask. No keypoint design is
   imported.
3. **The independent variable is `min det(J)`**, the same conditioning number
   the keypoint study reports. So the answer is a *lookup*: whatever the
   keypoint work settles on, its `min det(J)` says which law to use.

### 4.2 The surrogate arm, and why it can be trusted

A real rollout costs 25–40 s. This study is 352 runs, and the simulator is
currently saturated by the parallel session. So the arm is replaced by its own
equation.

An impedance-controlled arm that is tracking and not in contact **is** a
first-order system: `ẋ = D⁻¹K(a − x)`. That runs in about 2 ms per control step.
Crucially, the gate, the clamp, the attractor law, the phase update and the
stall watchdog are **the same functions the robot runs** — they were extracted
into the simulator-free policy layer for exactly this reason, so the instrument
cannot drift away from the system it measures.

Two properties make the bed trustworthy, and both are asserted in unit tests:

- **A parked attractor is reached exactly.** This is the property §2.7
  establishes for the real controller — the lag comes from the setpoint
  *moving*, not from failing to settle. A bed that missed this would be
  modelling something else entirely.
- **The settled lag has a closed form that the bed reproduces to 0.06%.**

### 4.3 A side finding: the lag gate's model is 1.28× optimistic

Deriving that closed form turned up something about the **shipped** system, not
the bed.

The lag gate decides the arm is behind by comparing its lag against
`expected = ‖K⁻¹D‖·v`. But the attractor is held fixed for a whole control step
— a *zero-order hold*, because the robot holds one action while the simulator
integrates internally at a much finer rate. Working the recursion through:

```
L* = v·dt / (1 − (1 − A·dt/m)^m),      A = D⁻¹K
```

At `m = 1` this reduces to `‖K⁻¹D‖·v`, exactly the gate's model. As the
sub-stepping refines it rises to `v·dt / (1 − exp(−dt/τ))`. **The real plant is
the fine limit.** With `dt = 50 ms` and `τ = 96 ms` the ratio is **1.28**.

So the gate carries a standing excess of about **4.6 mm at full demonstrated
speed** — roughly an eighth of its 35 mm tolerance — before anything has
actually gone wrong. Not a bug, and not urgent, but it means the gate runs
slightly tighter than its arithmetic suggests, and it is worth knowing before
anyone tunes `lag_tolerance` again.

![The lag model](figures/fig_lag_model.png)

*Figure 2. **Left:** the settled lag of the bed's arm as the Euler sub-stepping
is refined. At one sub-step (grey dashed) it matches the gate's own model; as it
refines it approaches the real plant (red dashed). **Right:** the ratio of true
lag to the gate's model, against control period. The gap closes only when the
control period is small compared with the impedance time constant `τ = 96 ms`;
at this project's 50 ms it is 1.28×.*

### 4.4 What is measured, and what is deliberately not

| metric | what it means | why this one |
|---|---|---|
| **attractor drift** | pointwise distance from the commanded attractor to the path it was meant to retrace, worst over the run | pointwise, so it **cannot come out negative**. The measure it replaces compared two separately-minimised distances and routinely did (§7.26) |
| **arm error** | distance from the arm to `planned − τ·v`, i.e. where the demonstrating arm actually was, warped | the quantity that matters for the task |
| **excess lag** | measured lag minus the physics-expected `τ·v` | **zero is correct**, not small-is-better |
| **dwell net displacement** | net movement across a zero-velocity window, split by hand axis | net, not path length — see §3 |
| clamp fraction, gate-shut fraction | how often the two safety mechanisms fired | tells you whether a result measured the law or the clamp |

**Deliberately absent: any success flag or placement error.** The bed has no
contact and no object, so any success number would be invented. §7.27 is a whole
section about a scalar proxy that rated the better configuration worse;
manufacturing one here would be that mistake made on purpose.

**Also forbidden, and worth naming so it cannot creep back in:** comparing the
arm against the *label* path. That charges the impedance lag (12–45 mm) as an
error, when §2.7 shows that lag is correct behaviour which lands the arm where
the teacher's arm was. It would penalise a correct executor by 23 mm.

---

## 5. Results

**43 warps × 8 laws = 344 runs, plus 8 at identity transport. Zero failures.**
Every law completed the phase on every warp; none stalled, none exhausted its
step budget.

### 5.1 How to read the tables

Rows are the eight laws:

| law | what it does |
|---|---|
| **V** | the shipped rule: `a += v·dt·gate`. Pure dead reckoning |
| **VR-a k=0.20** | V, then pull 20% of the way toward `reference` each step |
| **VR-a k=0.50** | the same at 50% |
| **VR-sched** | the same with a gain that is strong when the policy commands a hold (k≈0.44) and weak in transit (k≈0.08) |
| **VR-sched ungated** | as above, but the anchor keeps pulling even while the gate is shut |
| **VR-m k=0.50** | anchor at 50%, but the policy is queried at the **measured arm pose** |
| **R-a** | `a = reference` outright: no integrator at all |
| **R-m** | the same, queried at the **measured** pose |

Columns are conditioning bins — how hard the transportation map deforms space,
by `min det(J)`. **1.0 is no deformation; 0 is a fold.** "Well conditioned"
(>0.6) is roughly where today's box keypoints sit; "aggressive" (<0.35) is
roughly where the contact keypoints sat (§7.22 measured 0.18–0.27 for those).

Numbers are **millimetres, median [inter-quartile range], lower is better**.

### 5.2 Attractor drift — how far the commanded point strayed from its plan

| law | well conditioned (n=23) | moderate (n=14) | aggressive (n=6) |
|---|---|---|---|
| **V** | **3.93** [3.7, 4.5] | **3.77** [3.6, 5.1] | 4.42 [3.3, 5.8] |
| VR-a k=0.20 | 4.78 [4.5, 5.0] | 4.38 [4.0, 5.3] | 4.49 [4.0, 4.8] |
| VR-a k=0.50 | 6.24 [5.9, 6.5] | 6.20 [5.1, 6.4] | 5.60 [4.9, 6.8] |
| VR-sched | 5.03 [4.4, 5.4] | 4.16 [3.8, 5.3] | **3.96** [3.7, 4.4] |
| VR-sched ungated | 5.03 [4.4, 5.4] | 4.16 [3.8, 5.3] | 3.96 [3.7, 4.4] |
| VR-m k=0.50 | 14.90 [13.8, 17.7] | 14.73 [12.4, 17.0] | 15.06 [13.3, 56.3] |
| R-a | 11.23 [10.1, 12.6] | 9.84 [9.1, 11.7] | 9.53 [7.7, 53.6] |
| R-m | 25.68 [23.0, 49.1] | 23.70 [20.6, 40.3] | 53.18 [25.9, 72.1] |

### 5.3 Arm error — how far the arm ended up from the demonstrating arm

| law | well conditioned (n=23) | moderate (n=14) | aggressive (n=6) |
|---|---|---|---|
| **V** | **5.34** [4.8, 5.6] | **4.67** [4.5, 5.5] | 6.05 [5.2, 7.4] |
| VR-a k=0.20 | 6.55 [5.9, 6.8] | 6.25 [6.1, 6.9] | 6.77 [6.1, 7.1] |
| VR-a k=0.50 | 7.53 [6.9, 7.9] | 7.31 [6.8, 7.9] | 7.07 [6.5, 8.5] |
| VR-sched | 6.41 [6.0, 6.8] | 6.07 [5.7, 6.9] | **5.86** [5.2, 6.5] |
| VR-m k=0.50 | 14.42 [13.6, 17.4] | 14.27 [12.1, 17.1] | 14.85 [13.0, 57.5] |
| R-a | 10.68 [9.6, 12.2] | 9.98 [8.6, 11.2] | 9.10 [8.8, 53.7] |
| R-m | 25.49 [22.6, 48.9] | 23.48 [19.8, 40.1] | 52.64 [25.6, 72.1] |

![Laws against conditioning](figures/fig_law_vs_conditioning.png)

*Figure 3. Each faint dot is one warp; heavy markers are the bin medians the
tables report. The x-axis runs from easy (right, undeformed) to hard (left,
nearly folded); the y-axis is logarithmic because the families are an order of
magnitude apart. **Blue is the shipped law** and sits at the bottom across the
whole range. Green are the anchor family, consistently just above it. Red are
the reference-only laws; the dashed lines are the two that query at the measured
pose, and they are the worst throughout.*

### 5.4 The paired tests

Every law ran on **the same warps**, so the comparison is paired — worth roughly
a factor of four in samples over an unpaired design. Wilcoxon signed-rank
against V on attractor drift:

| law | well (n=23) | moderate (n=14) | aggressive (n=6) |
|---|---|---|---|
| VR-a k=0.20 | **+0.95 mm, p=0.0002 worse** | +0.33, p=0.46 | −0.38, p=0.69 |
| VR-a k=0.50 | **+2.11 mm, p<0.0001 worse** | **+1.62, p=0.002 worse** | +0.73, p=0.56 |
| VR-sched | **+0.96 mm, p=0.0001 worse** | +0.38, p=0.50 | −0.49, p=0.44 |
| VR-m k=0.50 | **+10.93 mm, p<0.0001 worse** | **+10.62, p=0.0001 worse** | **+8.33, p=0.03 worse** |
| R-a | **+7.41 mm, p<0.0001 worse** | **+5.84, p=0.0001 worse** | **+3.84, p=0.03 worse** |
| R-m | **+21.15 mm, p<0.0001 worse** | **+19.44, p=0.0001 worse** | **+48.50, p=0.03 worse** |

Positive means worse than V. **Not one cell in this table is significantly
better than the shipped law.**

### 5.5 What the numbers say

**Querying at the measured pose is decisively wrong here.** Compare `R-a`
against `R-m`, and `VR-a k=0.50` against `VR-m k=0.50` — the only difference in
each pair is the query site, and it costs 8–27 mm. The module docstring's
long-standing warning is vindicated: the velocity channel has a zero-mean prior,
so at a state away from the labels it returns *no motion at all*, and an arm
that is legitimately trailing by 24 mm is exactly such a state. Note the
`reference` channel is *not* enough to rescue it: `VR-m` has a restoring term
and is still 10 mm worse than its attractor-queried twin.

**The anchors are worse where it is easy and no better where it is hard.** At
good conditioning every anchor is significantly worse — the reference is a
*smoothed* version of a path the integrator is already following well, so
pulling toward it fights the feed-forward. In the aggressive bin `VR-sched` is
nominally 0.49 mm better, but `p = 0.44`, `n = 6` is at the floor below which
the test cannot reach significance anyway, and 0.49 mm is a *tenth* of the
demonstration's own 4.8 mm precision. That is not a result.

**The reference-only laws have a dangerous tail.** `R-m` reaches 72 mm and `R-a`
53 mm in the aggressive bin. The mechanism is understood: `reference` sets an
*absolute* position, so the attractor clamp — which projects it back onto a
sphere around the arm — becomes the dominant term, and "reference position, then
projected near the measurement" is arithmetically close to the very mode §2.8
records as failing.

**Gating the anchor is immaterial.** `VR-sched` and `VR-sched ungated` are
identical to two decimal places. Even with a disturbance shutting the gate on
over half the steps, the two differ by under 0.2 mm. The axis is real,
measurable, and does not matter.

**And the floor is low.** At identity transport — the policy fitted on the
demonstration's own labels, no map at all — the shipped law's arm error is
**5.32 mm**, against a demonstration whose own placement error is about 10 mm.
The executor reproduces the demonstration to better than the demonstration's own
precision. There is not much room left to win.

---

## 6. What this study cannot decide

The surrogate arm is an honest model of an impedance controller that is
**tracking and not in contact**. It is not a robot. Specifically it has:

- **no contact**, so nothing about whether the jaws actually close on the object,
  whether it slips, or whether the hand collides with a shelf;
- **no inverse kinematics**, so nothing about the roughly quarter of a
  trajectory that §7.25 measured as unreachable at its commanded orientation —
  which is the single largest known execution-side failure mode;
- **no orientation task** competing for the arm's effort, and no arm inertia
  beyond the first-order approximation;
- **no task outcome at all**, deliberately.

So the correct reading of everything above is: **it ranks hypotheses; it does
not confirm them.** What it can do is rule laws out, bound the gains, and say
which questions are worth spending simulator time on. On that basis it has ruled
out four of the five alternatives outright, which is most of the value.

Two beds are specified and **queued** until the parallel session's MuJoCo runs
free the machine (it is currently at 84% CPU with 357 MiB free, and CLAUDE.md
requires checking `free -h` before a long simulation run):

**Tier 1 — identity transport, exact ground truth, no map.** Fit the policy on
the demonstration's *own* labels and roll it out in the demonstration's *own*
scene. `record_source_placement(0)` returns `metadata["measured_positions"]`,
which is where the demonstrating arm actually was — an exact target, with no
transportation map anywhere in the loop. The question is "can the policy
reproduce the demonstration in the scene it was recorded in?", and a law that
fails it is disqualified whatever this study said. Roughly 2 minutes per
configuration.

**Tier 3 — the 20-scene reshelving regression.** The validated gate, 17/20 at
9.1 mm with seeds 8, 11 and 16 failing. Note this can only detect a
*regression*, not an improvement: 17/20 against 12/20 is `p = 0.155` by Fisher's
exact test, so the count has almost no power. **The sharp test is the identity
of the failing seeds** — a new seed failing while another recovers is real
signal that the count hides.

**Also queued: the byte-identity check.** The refactor that made this study
possible moved arithmetic out of `rollout_policy`, and although the defaults are
asserted bitwise identical at the unit level, the end-to-end proof is to run
three reshelving seeds against the baseline commit and against this branch and
compare the recorded trajectory arrays with `np.array_equal` — not `allclose`.
That needs the simulator too.

A final honesty note on what "flawless" can mean. By the rule of three, zero
failures in 20 runs bounds the true per-run failure rate only at 14%. No
campaign in this budget can certify the absence of failures. What can be
defended is: *no violation of the mechanism invariants — which are arithmetic,
not statistics — across 352 runs on this bed*, plus whatever the simulator tiers
add.

---

## 7. What changed in the code

| change | why |
|---|---|
| Gate, clamp, speed cap, steering and two label-derived scalars moved verbatim into `tpgpt/policy/rollout.py` | that module imports only numpy, so a test bed can call **the same code the robot runs**. Writing the gate and clamp a second time is how an instrument stops matching its system — and two of the three bugs ever recorded against these expressions were in the *interaction* between them (§7.14, §7.16) |
| `ExecutionLaw` + `attractor_step` compose the whole per-step decision | a bed calling the same helpers in a different *order* would still measure something that does not ship, and nothing would fail to warn. The robot calls the two halves because the infeasibility fallback runs between them; a test asserts the composition and the halves agree bit-for-bit |
| Three laws, two query sites, `anchor_gated`, `AnchorSchedule` | the study above. `AnchorSchedule` is a dataclass rather than a callable because a lambda serialises into a manifest as a memory address (§7.26 rule 2) |
| `rollout_impedance` | ~2 ms per control step against 25–40 s per real run |
| `load` and `blocked_steps` fault injection | an undisturbed surrogate arm fires the gate on 2% of steps and the clamp on none, so a bed without these cannot study a law that interacts with them |
| **First direct tests for `rollout_policy`** | it was previously exercised only through a four-minute end-to-end campaign. Each test inlines a verbatim transcription of the pre-extraction arithmetic and asserts *exact* equality, so a helper cannot drift from the behaviour the validated 17/20 was measured under |

**Defaults are unchanged everywhere**, and that is asserted rather than assumed:
`anchor` at `k = 0` is bitwise identical to `integrate`, and an `ExecutionLaw`
built with explicit defaults is bitwise identical to one built with none.

**Two things deliberately not done.** Passing `velocity_desired` to the
controller would remove the impedance lag, but §2.7 shows that lag is
*transport-invariant* and lands the arm where the demonstrating arm was, so
removing it also requires relabelling the demonstration with measured poses —
which edits `tpgpt/sim/demo.py`, the shared upstream the parallel session's
baseline depends on. And `GPPolicy.attractor()` returns `reference + K⁻¹D·v`,
which is correct only against measured-pose labels; against this project's
attractor labels it would land the arm ~24 mm *ahead* of the demonstration.

**One latent bug found and not fixed here.** `rollout_policy` binds `offset` to
the tool offset, then rebinds the same name to the scoring offset before writing
`metadata["tool_offset"]` — so that field holds the object-to-goal vector.
`diagnose._already_at_the_tool` thresholds it at 1e-9, and a placement offset is
never that small, so it returns `True` for every run. Today that is
*accidentally correct*, because `diagnose` is only called from `pipeline.py`
which always passes a real tool offset; it becomes wrong the moment anyone runs
the tool-frame ablation. The fix belongs in its own commit, coordinated, because
the consumer is the parallel session's file and changing the system and the
ruler together is what §7.26 rule 6 forbids.

---

## 8. Verification status

```bash
export MUJOCO_GL=egl
P=/home/ishita/mujoco_env/bin/python
$P -m pytest tests/unit -q     # 480 passed, 40 s
$P -m pytest tests -q          # 580 passed, 6 min
$P -m tpgpt.experiments.sweep_execution --out outputs/dynamics          # quiet
$P -m tpgpt.experiments.sweep_execution --out outputs/dynamics_loaded --load 0.08
```

**Unit: 480 passed.** 59 of those are new and cover the gate, the clamp, the
speed cap, the steering step, the three laws, the composition and the surrogate
plant. `rollout_policy`'s arithmetic has direct cover for the first time.

**Full suite: 580 passed.** One integration test (`test_graspgen.py::TestServer::
test_reports_its_loaded_grippers`) failed on the first run and passed on
re-run — it asserts against the state of the shared GraspGen-X server, which the
parallel session was using at the time, and it touches none of the files this
branch changes. Not a defect here, but worth noting as a flaky test whose
outcome depends on another process.

**Defaults are asserted unchanged**, at the unit level, bitwise: `anchor` at
`k = 0` equals `integrate`, and an `ExecutionLaw` with explicit defaults equals
one with none, over hundreds of random draws. The end-to-end proof — three
reshelving seeds against the baseline commit, compared with `np.array_equal` —
is queued with the simulator tiers.

**Provenance.** This branch is a `git worktree` at `a89f0e3`, so
`provenance()` reports it reproducible; the parallel session's uncommitted work
stays in its own tree. The nine files this branch touches do not overlap the
three that session is editing, so the merge is clean apart from a possible
renumbering of `ROBOTICS_NOTES` §7.28 if both sessions added one.
