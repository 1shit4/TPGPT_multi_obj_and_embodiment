# How execution works, in plain words

What happens between "the plan is made" and "the object is on the shelf", what every correction
on top of the plan does, why each one exists, and where the system still has gaps. Written
2026-10-02, after the mug runs of that evening. Code: `ycb_pointcloud_pipeline/tpgpt_executor_node.py`
(the ROS side) and `tpgpt/hardware/executor.py` (the arithmetic, no ROS).

## The short version

The plan is a list of 488 **labels**, one every 50 ms of the demonstration: where the hand should
be, how it should be turned, how stiff the arm should be, and whether the gripper is open or
closed. The arm is driven by a **spring**: every 50 ms we tell the controller (CRISP) a target
pose and a stiffness, and the arm is pulled toward that target like a mass on a spring. It never
sits exactly on the target: friction and the spring's own softness leave it behind or beside it.

Everything below is about that gap: how the plan is written so the arm ends up where it should,
and what the executor does when the arm still doesn't.

## The words used

| word | what it means |
|---|---|
| **command** (or attractor) | The target pose we send to the arm every 50 ms. The spring pulls the arm toward it. |
| **label** | One row of the plan: a pose, a stiffness, a gripper state. Label 0 is the start, 488 the end. |
| **clock** (phase) | How far through the plan we are, 0 to 1. Each step it moves forward by one step's worth, unless something slows it. Where the clock is decides which label we are at, and so when the gripper acts. |
| **lag** | How far the arm is from where it should be. **Behind** = short of it along the direction of motion; **beside** = off to the side. |
| **taught path** | Where your hand moved the arm when you taught the demonstration. |
| **arm path** | Where the arm actually went when the replay ran the learned command (below). The executor compares the arm against this. |
| **margin** | How much the arm's path varied between two replay runs of the same command: 1.0-3.6 mm per label. Inside it, the arm counts as on its path. |
| **mark** | The taught path at the grasp and at the release: where the hand must be when the gripper acts. |
| **stiffness** | How hard the spring pulls per mm of gap. 600 N/m means 0.6 N per mm. |
| **null space / posture** | The FR3 has 7 joints but a hand pose fixes only 6 numbers, so the elbow can swing while the hand stays still. Which way the elbow sits is the posture. |

## Why the plan's command is not the taught path (the learned bundle)

When the taught path itself was sent as the command (replay pass 0), the arm strayed from it by
17 mm RMS, 10-15 mm sideways. The reason is the arm, not the plan: joint friction leaves it
6-9 N short of where the spring points (10-16 mm at 600 N/m, even when standing still), and it
lags about twice what the spring model predicts while moving.

So the demonstration was **replayed** on the robot and the command corrected after each pass, by
how far the arm missed (learning by repetition, "ILC"). After 3 passes the arm followed the taught
path to 2.7 mm RMS. What came out is the **learned bundle**
(`~/tpgpt_demos/banana_20260925_183904/source_bundle_learned`):

* the **learned command**: it runs 17-35 mm **ahead** of where the arm should be, so that friction
  and lag leave the arm on the taught path;
* the **arm path**: where the arm went under that command, with its **margin**;
* the taught path's **marks** at the grasp and the release.

The planner carries all three into the new scene with the same transport map, checks the **arm
path** (not the command) against the shelf, the table and the camera's points, and records the
**posture** it checked the arm in at every 4th label.

One demonstration, one replay: the learned bundle is used for every object and slot.

## What happens in a run, in order

1. **Load and fit** (2-3 s). The plan's labels are fitted with the policy (a smooth function that
   gives, for any point near the path, which way to move, how to turn, how stiff, gripper).
2. **Checks before anything moves.** The arm is within 0.6 m of the plan's start
   (`max_start_distance`); fresh joint readings arrive; nobody else is publishing targets to the
   controller. A summary is printed and you type `yes`.
3. **Approach.** A straight line from where the arm is to the plan's start, at most 5 cm/s and
   0.3 rad/s, at a fixed 400 N/m. The posture target walks the elbow from where it was at the
   start onto the first checked posture.
4. **Settle at the start.** The executor waits until the arm is within **200 mm** of the start
   (`approach_settle_tolerance`; at most 5 s), then holds 1.5 s and measures how far off it rests.
   **It does not bring the arm onto the start**: the policy begins from wherever the arm rested.
   (For a taught plan that resting gap is then subtracted from every command; a learned plan does
   not, since its command already includes the arm's offset.)
5. **The policy, every 50 ms:**
   1. read where the hand is (`/current_pose`, 250 Hz) and the joints (`/joint_states`, 30 Hz);
   2. ask the policy what to do at the command's position: move direction and speed, turn,
      stiffness, gripper;
   3. move the command along by that velocity × 50 ms × **how fast the clock may go** (1 = full
      speed, 0 = stopped; this is the gate or the pacing, below);
   4. apply the corrections below (they differ for taught and learned plans);
   5. send the command pose, the 3×3 stiffness and the elbow posture to CRISP;
   6. advance the clock by the same factor;
   7. log a row (`<plan>/executions/robot_<time>.csv`).
6. **When the gripper is due to act**, the executor first holds: see "the holds" below. The clock is
   paused while the jaws move.
7. **End.** The last command is held 1 s; CRISP keeps holding it after that. The 20 Hz log and a
   full-rate record of every pose and joint reading (`..._highrate.npz`) are saved, after an abort too.

## Two ways of following a plan

### A taught plan: the simulation's rules (lag gate, clamp, watchdog)

This is the code the simulation runs, shared line for line (`tpgpt/policy/rollout.py`).

* **Lag gate.** If the arm falls behind the command by more than the spring is expected to leave it
  (the spring model's lag, 2-6 mm here) plus a tolerance (45 mm on the robot, 35 in the sim),
  the gate closes gradually: at 20 mm of excess the clock runs at 0.56, at 45 mm it stops. The
  command stops advancing with it. Why: so the gripper never acts on a pose the arm has not
  reached. In the sim, without it, the gripper closed with the arm 20-56 mm short on 8 of 8 scenes.
* **Clamp.** The command is never allowed further than a radius from the arm (here 16 mm), so a
  blocked arm is never pushed with an ever-growing force.
* **Watchdog.** If the clock has not moved for 40 steps (2 s) and the arm is not closing the gap,
  the gap is declared permanent ("static sag") and forgiven: the gate stops counting it, up to 3
  times. Then the run is declared stalled.
* Hardware additions for taught plans: the resting gap at the start is subtracted
  (`offset_compensation`); after a stall the gate is switched off and the plan is followed at its
  own speed (`on_stall: continue`); an unreachable orientation is relaxed or a stretch skipped
  (the sim's infeasible-pose fallback).

### A learned plan: following the arm path ("follow mode")

The lag gate and clamp judge "normal lag" by the spring model, 2-6 mm. The learned command must
run 17-35 mm ahead of the arm, by design. Measured on the 18:39 plan: the gate would have slowed
the clock to 0.67 on the median step and the clamp would have cut the command on 262 of 488
labels, undoing the learning. So a learned plan is compared against its **arm path** instead:
"is the arm where the arm went in the replay?", not "is the arm on the command?". On top of the
plan, these act (numbers are the defaults):

| what | what it does | why it exists (what was measured) |
|---|---|---|
| **Pacing** (our lag gate) | When the arm is **behind** its arm path by more than the margin, the clock and the command slow; 20 mm beyond the margin they stop. | The same job as the lag gate: the plan must not run ahead of the arm. On 2026-10-01 (mug) the clock was slowed on 36 % of steps. |
| **Correction, beside the path** | When the arm sits **beside** its arm path beyond the margin, an offset is slowly added to the command (1.5 × the excess per second, at most **30 mm**) until the arm is back. | Waiting cannot fix a sideways offset: friction holds the arm 10-16 mm off its command even when still. Without corrections (mug, 10-01): 7-18 mm beside the path, 15 mm off at the grasp, 16 at the release; with them: 2-6 mm, 3.7 and 2.0. |
| **Correction, behind the path, while the clock is held** | As the clock slows, the gap *behind* is handed to the same correction (all of it when the clock has stopped). | 2026-10-01 22:20: the arm rested 18-21 mm behind with the clock stopped and nothing in its way. Waiting closes lag, not a standing gap. |
| **Unwinding** | What the correction added *along* the path is given back once the arm runs ahead of its path. | 2026-10-02 00:47: after a stick-slip the leftover correction held the mug 24-31 mm ahead and set it down ~10 labels early. With unwinding (01:45): on time, never more than 4 mm ahead. |
| **No correction in contact** | Around the grasp (10 labels either side) and the set-down (10 before, 5 after) the correction is frozen. | The transported stiffness alone must set contact forces. Grown, the correction pushed 21 mm (12 N) into the board at a set-down (10-01). |
| **Bound** | The command never goes further from the arm than the plan's own lead + 30 mm + 10 mm. | The clamp's job (a bounded push), without cutting the lead the plan needs. |
| **Blocked: stop and hold** | If the clock is stopped or the correction is at its cap for 40 steps (2 s), the arm got no closer to its path (by 1 mm) **and moved less than 2 mm**, the run stops and the arm holds where it is. | A safety stop for an arm that really is stuck. It used to fire on a moving arm (22:41); since then it needs the arm not to move. |
| **Grasp hold** | Before the jaws close: the command is walked so the **hand** lands on the grasp mark (by the arm's own error, then a small bounded push), until within 4 mm or 4 s. | The jaws must close where the plan says. Tonight's first run: 3.0 mm. |
| **Release: wait only** | Before the jaws open: the executor only waits for the arm to settle; no push. | Your decision (10-01): pushing a set-down object onto the mark can only slide it along the board. |
| **Posture target** | The elbow is pulled toward the posture the path check verified, label by label (within 0.5 rad of where it is, never into a joint limit). | Without it the elbow is pulled toward home; on 00:47/00:50 that swung joints 1 and 3 by 50-65° as joint 2 crossed 0 and the arm stuck. Tonight's first run: within median 7° of the checked posture. Applies to taught plans too. |

### Safety guards, in both modes

* **Tracking error**: the hand more than 80 mm from the policy's command → abort and hold.
* **Not following**: over 40 steps the hand moved less than 30 % of what the command moved → abort.
* **Time**: the policy takes longer than 300 s → abort.
* **CRISP's own limits**: it never acts on more than 0.1 m / 0.5 rad of error, and limits joint
  torques and how fast they change.

### Stiffness

The labels carry a **direction-dependent** stiffness: 350 N/m along the direction of motion, 600
across it and in every direction when the arm is still (grasp and release included), transported
by the map with the rest of the plan and sent as a full 3×3 matrix (CRISP was patched to accept
it). The hand's turning stiffness is 30 N·m/rad throughout (at 60 the wrist vibrated near joint 7's
limit). The approach uses a fixed 400 N/m.

## The lag gate and the follow mode, side by side, on tonight's stuck run

Run 23:28 (mug, top_right): the arm ended the approach **54 mm** short of the start (the wrist
~100 mm short, the hand tilted 13.5° toward the target). The run began anyway (the 200 mm settle
tolerance). From there:

| | follow mode (what ran) | the simulation's lag gate (computed on this plan, not run) |
|---|---|---|
| the clock | stopped (pacing 0.00 at label 4) | stops at first (gate 0.00 at 54 mm) |
| the push on the arm | grew to 31 N (correction at its 30 mm cap) | the clamp pulls the command back to **16 mm** from the arm: about 10 N |
| then | the arm did not move (the hand spring and the turning spring cancelled at every joint to within 1.8 N·m, less than joint friction); after 2 s **blocked: stop and hold** | at 16 mm the gate reopens to **0.64**: the clock creeps on with the arm still stuck. The simulation's loop finishes a whole task this way with an arm that never moves, gripper schedule included (measured, `SafetyMonitor` docstring); on the robot the "not following" guard would stop it. |

So both stop the clock when the lag is large, which is what you want. Neither gets a stuck arm
moving: the gate pushes *less* (its clamp), and then lets the clock run on. And on every normal
step of a learned plan the gate would slow the clock and cut the lead the learning put there.

## Tonight's two runs, in these terms

* **23:25, mug → top_left: complete.** Rested 29 mm off at the start. Hand within 3.0 mm of the
  grasp mark and 3.2 mm of the release. Beside the arm path median 3.3 mm, worst 20.3; the clock
  slowed on 384 of 1479 steps (slowest 0.56); the correction reached its 30 mm cap; elbow within
  median 7° (worst 29°) of the checked posture.
* **23:28, mug → top_right: blocked at label 4**, as above. The posture target's share of the
  joint torques was at most 0.07 N·m: not the cause.

## What the system does not handle yet

* **The start is never checked.** The policy begins wherever the approach left the arm (up to
  200 mm away). Over 15 runs the resting gap at the start was 15-54 mm. It does not depend on how
  far out the start is (correlation 0.03); on the three runs with full joint records it grew with
  the load on the shoulder joint (15 / 29 / 54 mm at 27 / 41 / 53 N·m), which suggests friction
  rising with load. Of the mug runs, all 7 that started within 29 mm completed; none of the 4 that
  started 35 mm or more off did (two of those were stopped by "blocked" bugs since fixed).
* **The correction is capped at 30 mm**, so a start gap much larger than that cannot be made up
  by the follow mode.
* **The soft turning stiffness (30 N·m/rad)** lets the hand tilt instead of the arm reaching: a
  31 N pull at the fingertips, 228 mm below the wrist, tilts the hand by about 13°.
* **That the replay's corrections carry over to a new scene is assumed**, not measured: the arm's
  offset depends on its posture, and the learning was done at the demonstration's postures.

Options discussed so far (none built): hold at the start until the arm is on its path, as the
grasp hold does (pushes harder: up to ~50 N here; it may break free with a slip, or tilt the hand
further); a stiffer approach (600 instead of 400 N/m); a stiffer turning spring (ruled out on
09-30 for the wrist vibration).
