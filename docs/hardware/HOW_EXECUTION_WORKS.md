# How execution works, in plain words

How the executor runs a plan on the FR3: the steps, the rules it follows, the equations behind
each rule, and the known limits of the design. **This file describes the current code. Change it
in the same commit as any change to how execution runs.** It is not a log: run results belong in
`FAITHFULNESS_AUDIT.md` and the plan folders.

Code: `ycb_pointcloud_pipeline/ycb_pointcloud_pipeline/tpgpt_executor_node.py` (the ROS side) and
`tpgpt/hardware/executor.py` (the arithmetic, no ROS). Numbers are the defaults.

## The words used

| word | what it means |
|---|---|
| **label** | One row of the plan, one per 50 ms of the demonstration: a hand pose, a stiffness, a gripper state. The plan has 488. |
| **command** | The target pose sent to the arm every 50 ms. The arm is pulled toward it by a spring (CRISP's Cartesian impedance controller). It never sits exactly on it. |
| **clock** | How far through the plan we are, 0 to 1. It decides which label we are at, and so when the gripper acts. |
| **taught path** | Where your hand moved the arm when you taught the demonstration. |
| **learned command** | The command the replay learned so that the arm, lagging and held back by friction, ends up on the taught path. It runs ahead of the arm. |
| **arm path** | Where the arm went under the learned command in the replay. The executor compares the arm against it. |
| **lead** | How far the learned command is ahead of the arm path at a label: \|command label − arm path label\|. 10-16 mm when still, up to 35 mm while moving. |
| **margin** | How much the arm path varied between two replay runs, per label (1.0-3.6 mm). Inside it the arm counts as on its path. |
| **mark** | The taught path at the grasp and at the release: where the hand must be when the gripper acts. |
| **behind / beside** | The arm's distance from where it should be, split into the part along the direction of motion and the part across it. |
| **posture** | The FR3 has 7 joints but a hand pose fixes only 6 numbers, so the elbow can swing with the hand still. Which way it sits is the posture. |

## Which plan is executed

The planner is given a **bundle** (one demonstration, prepared). There are two kinds:

* a **taught bundle** (`source_bundle/`, `source_bundle_directional/`): its labels are the taught
  path itself;
* the **learned bundle** (`source_bundle_learned/`): its labels are the learned command, plus the
  arm path, its margin, the taught labels and the marks.

Either is transported into the new scene by the same map and planned the same way; the planner
checks the **arm path** of a learned plan (not its command), and records the joint posture it
checked the arm in at every 4th label. **Every plan we run now is a learned plan.** Taught plans
are run by the simulation's rules, described at the end for completeness.

Why the learned command: sent the taught path as its command, this arm strayed 17 mm RMS from it
(10-15 mm sideways), because joint friction leaves it 6-9 N short of where the spring points and
it lags about twice what the spring model predicts. Three replay passes, each correcting the
command by how far the arm missed, brought that to 2.7 mm.

**What the planner's check assumes:** that the transported arm path is where the arm will go in
the new scene. That is exact only for the part of the lag a spring model explains and the map
carries; the rest of the arm's offset depends on its posture, and how much of it carries over is
not known. The follow mode below exists to keep the arm on that path anyway.

## A run, in order

1. **Load and fit** (2-3 s): the plan's labels are fitted with the policy, a Gaussian-process
   model that gives, at any point near the path, a velocity, an orientation, a stiffness and a
   gripper command (the paper's method: transport the labels, refit a policy, execute it).
2. **Checks before moving**: the arm within 0.6 m of the start, fresh joint readings, no other
   publisher on the controller's topics; a summary; you type `yes`.
3. **Approach**: a straight line from where the arm is to the plan's start (the first label of the
   learned command), at most 5 cm/s and 0.3 rad/s, at a fixed 400 N/m. The elbow's target walks
   from the posture at activation to the first checked posture.
4. **Settle**: wait until the hand is within 200 mm of the start (`approach_settle_tolerance`,
   at most 5 s), hold 1.5 s, measure how far off it rests. **The arm is not brought onto the
   start**: the policy begins from wherever it rested.
5. **Follow the plan** every 50 ms (next section) until the clock reaches the end.
6. **At the gripper events**: the holds (below). The clock pauses while the jaws move.
7. **End**: the last command is held 1 s; CRISP keeps holding it. The 20 Hz log
   (`<plan>/executions/robot_<time>.csv`) and every pose and joint reading (`..._highrate.npz`)
   are saved, after an abort too.

## Following a learned plan, every 50 ms

Symbols: `p` = where the hand is (measured), `a` = the policy's command, `c` = the correction,
`i` = where the command is along its labels (a fractional label number), `x(i)` = the arm path
there, `u` = the arm path's direction of motion there (none when it is still), `m` = the margin,
`lead` = the lead there, `dt` = 0.05 s.

**1. How far off the arm is.**

    e      = x(i) − p                 where it should be, minus where it is
    behind = e · u                    along the motion; > 0 means behind
    e_side = e − behind · u           across it (all of e when the path is still)

**2. Pacing: how fast the clock may run** (the follow mode's lag gate).

    pace = clip( 1 − (behind − m) / 20 mm , 0, 1 )

behind by up to the margin: full speed; 10 mm more: half speed; 20 mm more: stopped. The command
and the clock advance together, by `pace`:

    a ← a + v(a) · 0.35 · dt · pace       v = the policy's velocity at a (0.35 = speed_scale)
    clock ← clock + (the policy's rate) · dt · pace

**3. The correction**: an offset added to the command, for what waiting cannot close. It grows
every step (except inside a contact window, see 6):

    drive = (1 − pace) · behind · u                       the gap behind, as much as the clock is held
          + (|e_side| − m) · e_side / |e_side|            beside it, beyond the margin
    c ← c + 1.5/s · dt · drive                            7.5 % of the drive per step
    if |c| > 30 mm:  c ← c · 30 mm / |c|                  capped at 30 mm

The part **beside** the path acts all the time, clock running or not. The part **behind** acts
whenever the clock is slowed, in proportion to how much: nothing at full speed (`pace` = 1, which
means the arm is within the margin), all of the gap once the clock has stopped. There is no
waiting before it starts. How fast it grows (1.5/s means 1.5 per second: 7.5 % of the gap per
50 ms step):

* **if the arm does not move**, the gap stays the same and `c` grows by 1.5 × the gap per second,
  in a straight line, until its cap: a 30 mm gap reaches the 30 mm cap in 0.67 s;
* **if the arm follows the command**, the gap shrinks as `c` grows, so it closes like a decaying
  exponential with a time constant of 1/1.5 = 0.67 s: 63 % in 0.67 s, 95 % in 2 s.

1.5/s is slower than the arm's own response (its lag time constant is 0.15-0.3 s) on purpose: a
correction faster than the arm can follow overshoots. At 3.0/s (tried on 2026-10-02, same plan)
the release came closer (5.7 against 11.2 mm) but the carry swung more. If the gap was lag that
would have closed by itself, the arm then runs ahead and the unwinding takes the excess back.

**When the clock restarts**: there is no separate rule. `pace` is recomputed every step from the
current gap, so as the arm closes in the clock speeds up by itself, and runs at full speed once
the arm is within the margin.

**4. Unwinding**: what the correction added along the motion is given back once the arm runs ahead.

    ahead = −behind − m
    if ahead > 0 and c · u > 0:   c ← c − min( c · u , 1.5/s · dt · ahead ) · u

It shrinks by 7.5 % of how far ahead the arm is, per step, and stops at zero; the part across the
path is not touched.

**5. The command sent, and the bound.**

    q = a + c
    R = lead + 30 mm + 10 mm
    if |q − p| > R:   q ← p + (q − p) · R / |q − p|

The bound keeps the push on a held-back arm finite (at 600 N/m, R = 50-75 mm is 30-45 N). 30 mm is
the correction's cap, 10 mm a slack. **It is measured from the arm**, so an arm already more than
R from its command gets a command parked on the bound: the push cannot grow, however large the
correction becomes.

Why 30 mm for the cap: a safety limit, not physics -- how far the command may be moved off the
plan, and with the bound, how hard a held-back arm is pushed. It was set on 2026-10-01 from the
offsets measured then (10-16 mm standing, up to 25-36 mm needed while moving). It trades pushing
hard enough against friction for not pushing hard into something real; nothing derives the value.

**6. No correction in contact.** From 10 labels before to 10 after the grasp, and from 10 before to
5 after the release, `c` is frozen (neither grown nor unwound): the transported stiffness alone
sets the contact force.

**7. Blocked: stop and hold.** Counted while the clock is stopped (`pace` < 0.05) or the correction
is at its cap. After 40 counted steps (2 s), if the arm got no closer to its path by 1 mm **and**
moved less than 2 mm, the run stops and the arm holds where it is. Otherwise the count restarts.

**8. Stiffness and posture sent with every command.**

* Stiffness: the plan's, transported by the map (`K_new = J⊥ K J⊥ᵀ`, J⊥ the map's local rotation)
  and sent as a full 3×3: 350 N/m along the motion and 600 across it, 600 in every direction when
  still (grasp and release included). Turning stiffness 30 N·m/rad throughout.
* Posture: the posture the planner checked at label `i` (interpolated between its every-4th-label
  postures), kept 0.1 rad inside every joint limit and within 0.5 rad of the measured joints per
  joint. CRISP pulls the elbow toward it with 2 N·m/rad per joint (at most 5 N·m), only along the
  elbow's free swing: the part that would move the hand is removed (projected out) by CRISP. So
  the posture and the command need not match exactly. The posture is indexed by `i`, where the
  command is, and describes the arm on its arm path there, where it should be.

## The holds at the gripper events

**Grasp.** The clock is paused. The command is walked (at 5 cm/s) by the hand's own error from the
grasp mark, `q ← q + (mark − p)`, then a small push builds:

    I ← I + 1.5/s · dt · (mark − p),   |I| ≤ 30 mm,   sent: q + I

until the hand is within 4 mm of the mark for 3 steps, or 4 s have passed. Then the jaws close,
and the correction takes over the hold's offset so the command does not jump back.

**Release.** The clock is paused. The command is held still; when the hand has moved less than
0.5 mm over 5 steps, the jaws open. **It does not check the mark**: the hand opens wherever it
settled. Why: the object is already set down, and pushing the hand onto the mark could slide or
press it along the board. (Proposed by Claude on 2026-10-01; you have since said pressing down on
the board is harmless and the hand must reach the mark. To be changed.)

## Safety guards

* **Tracking error**: the hand more than 80 mm from the policy's command `a` → abort and hold.
* **Not following**: over 40 steps the hand moved less than 30 % of what `a` moved (once `a` moved
  a few mm) → abort and hold. It catches an arm that does not respond at all (a controller that
  dropped out, a protective stop), which the follow mode would otherwise wait on.
* **Time**: the policy runs longer than 300 s → abort.
* **CRISP's own limits**: it never acts on more than 0.1 m / 0.5 rad of pose error, and limits
  the joint torques and how fast they change.

On any abort the elbow's target is set to where the elbow is, and the arm holds its current pose.

## Known limits of the current design

* **The start is not checked.** The policy begins wherever the approach left the arm, up to 200 mm
  off. A gap there is carried into the run, and if it exceeds the bound (5) it cannot be pushed
  out.
* **The bound is measured from the arm**, not from where the arm should be (5).
* **The correction is capped at 30 mm** (3).
* **The release opens wherever the hand settles** (holds).
* **A stretched arm cannot be pulled outward.** Near full stretch, a pull at the hand along the
  direction of reach turns into almost no joint torque (the arm's weakest direction): the arm stops
  short and the fingertips get the rest of the way only by tilting the hand, against the turning
  spring. Across the runs so far, the more stretched the wrist at the start, the larger the gap
  there. The planner checks that every pose is reachable, not how stretched the arm is when it
  gets there.
* **The turning stiffness is soft.** A force F at the fingertips, 228 mm below the wrist, tilts the
  hand by about F × 0.228 m / 30 N·m/rad (31 N → 13°). 30 was chosen on 2026-09-30: at 60 the wrist
  vibrated (26 Hz) as joint 7 entered its limit zone during a replay; 45 has not been tried there.
  The learned bundle was learned at 30, and the executor refuses other values for it.
* **The arm path's transfer to a new scene is assumed** (see "Which plan is executed").

## Taught plans: the simulation's rules (not used now)

A taught plan's command is the taught path itself. It is followed with the code the simulation
runs (`tpgpt/policy/rollout.py`), with `a` = the command, `K, D` = the label's stiffness and damping:

    expected = ‖K⁻¹D‖ · |v|                                  the lag a spring model predicts (2-6 mm here)
    gate     = clip( 1 − (|p − a| − expected − sag) / 45 mm , 0, 1 )
    a ← a + v · dt · gate,   clock ← clock + rate · dt · gate
    clamp:   |a − p| ≤ min( 3 · v_max · ‖K⁻¹D‖ , expected + sag + 45 mm )     (about 16 mm here)
    watchdog: clock still for 40 steps and the gap not closing by 2 mm → sag = |p − a| − expected
              (forgiven), up to 3 times; then stalled

What each part is for: the **gate** stops the clock while the arm is too far behind, so the gripper
never acts on a pose not reached; the **clamp** bounds the push on a blocked arm; the **watchdog**
stops waiting for a gap that will not close. On the robot, also: the resting gap measured at the
start is subtracted from every command (`offset_compensation: z`, its vertical part), betting the
arm sags the same everywhere; after a stall the gate is switched off (`on_stall: continue`); an
unreachable orientation is relaxed or a stretch skipped (the infeasible-pose fallback).

Why learned plans do not use them: they judge "normal lag" by the spring model (2-6 mm), while the
learned command must run 17-35 mm ahead of the arm by design. On the 18:39 plan the gate would
have run the clock at 0.67 on the median step and the clamp would have cut the command on 262 of
488 labels. And with an arm that does not move, the clamp parks the command at its radius, which
reopens the gate to `1 − radius/45 mm` (0.64 at 16 mm): the clock creeps on and the gripper acts on
schedule wherever the arm is; only the "not following" guard stops it.
