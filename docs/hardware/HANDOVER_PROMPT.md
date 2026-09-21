# Prompt to hand to the agent with the robot

Copy everything below the line into that session, together with
`paper/HARDWARE_EXPERIMENT.md`.

---

You have a Franka arm, two end-effectors, and this project's transportation-map
pipeline. Design, implement and run the real-robot experiment specified in
`HARDWARE_EXPERIMENT.md`, then return an analysis.

**Context you need and should not have to infer.** The paper claims that one
demonstration — recorded once, on one arm, with a parallel-jaw hand, on one
object — can be executed on an object it never saw, by a hand of different
geometry, at a destination it never visited, with no further demonstration, no
policy training and no simulator. The transfer is a geometric transportation map
whose keypoints are a fixed-extent cube placed at the grasp pose and oriented by
it, so the map encodes nothing about the object's size or the hand's morphology.
This experiment is the real-robot evidence for that claim. Simulation already
covers the breadth; this covers the credibility.

**Work in this order.**

1. **Read the pipeline first and report back before touching the robot.** Tell me
   which entry point runs a single trial end to end, what it currently records,
   and precisely what is missing against the `record.json` schema in the brief.
   Do not start building until we have agreed that list.
2. **Build the trial harness**, not a campaign script. One function that takes
   (hand, object, destination) and runs one trial to completion, writing the full
   trial directory. The campaign is a loop over that function, and a trial that
   raises must leave its directory behind with whatever it managed to record.
3. **Add a pre-flight check that runs before every trial and aborts on failure**:
   the mounted hand matches the requested hand; the demonstration file's hash
   matches the recorded one; the camera extrinsics file is the one calibrated
   this session; the git tree is clean; the object is visible and its cloud
   exceeds the minimum point count. Abort before moving the arm, not after.
4. **Characterise the measurement instrument before trial 1** — ten repeated
   measurements of one stationary object, re-running segmentation each time.
   Report the standard deviation. Repeat at the end of the campaign.
5. **Run the grid**, randomised within each hand, completing one hand before
   re-mounting.
6. **Analyse and report.**

**Things that will invalidate the experiment, so guard against them in code.**

- Re-recording the demonstration for the second hand. There is exactly one
  demonstration and its hash is checked every session.
- Overriding the pipeline's grasp choice. Selection walks the generator's ranked
  list and executes the first candidate admissible at every waypoint of the
  transported path; that rule is part of the method. Record which rank ran and
  why earlier ranks were rejected. Never force a rank in a main-grid trial.
- Tuning anything per hand, per object or per trial — cube extent, controller
  gains, filter thresholds. If a hand cannot work without a change, that is a
  finding to report, not a change to make.
- Wall-clock timestamps in recorded data. Time is `step_index / control_rate`.
- Running with a dirty git tree. Stamp the SHA, the modified files and the
  untracked code files into every trial record, and refuse to start if dirty.

**What to return.**

- `analysis.md`: the statistics listed in §5 of the brief, each with the test
  named and its assumptions stated. Wilson intervals for proportions, Fisher's
  exact between hands, bootstrap CIs for medians, point-biserial correlation
  between `min_det_J` and success.
- `table5.csv` and a LaTeX `tabular` in IEEEtran style: hand, object, trials,
  placed, median error (mm), dominant failure stage, with a totals row carrying
  Wilson intervals.
- `summary.csv`: one flat row per trial.
- The photograph set listed in §6, named so each is identifiable.
- A short section titled **"What went wrong"** listing every aborted or excluded
  trial with its reason.

**How to report.** State what was measured, with what instrument, and what would
make the conclusion false. If a number is surprising, say so and say what you
checked. Do not summarise a failure as a rate — attribute it to the stage at
which it occurred. If the experiment cannot answer something it was meant to
answer, say that plainly rather than answering a nearby question instead.
