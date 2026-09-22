# Fresh start on the new setup (from 2026-09-22)

The camera and the shelf moved before 2026-09-21. **Nothing recorded before that counts as
evidence**:

- the X6 trials in `outputs/hardware/hw_2026_09_x6/`, and their clouds and plans;
- the 2026-09-15 banana demonstration in `~/tpgpt_demos/` (its object cloud went through the old
  calibration);
- any conclusion drawn from those.

They stay on disk, untouched, for history. The only exception is the FK fixture
(`tests/data/fr3_still_samples.npz`), which is robot-only.

The pipeline changed as well (`SHELF_AND_PATH_CHECK.md`) and has not yet run on a single capture
from the new setup. So the order below builds up one piece at a time, and every stage must pass
before the next one moves the arm closer to the shelf.

## 0. Before anything moves

- The shelf is measured (tape + touches, 2026-09-22, `shelf.yaml`) and needs no `cloud_fit`: the
  camera, not the tape, was found off at the shelf (about 10 mm sideways, `CALIBRATION.md`).
- For the first runs, pass **`-p on_stall:=abort`**. The default, `continue`, switches the lag gate
  off after a stall (audit C1).
- Leave X6 alone (`experiment:=false`, the default). Plans go to `~/tpgpt_plans/`.

## 1. Sensing only (no motion)

Robot container, camera and segmentation up (`commands.md`). Check:

- `/ycb_scene_cloud` is published (`ros2 topic hz`);
- the object clouds and masks look right (`/ycb_detections/image`);
- the calibration overlay still sits on the fingertips (`calibration_check_node`), since the camera
  must not have been knocked;
- optionally `shelf_fit` with the shelf empty, as a check of the camera against the shelf (it is
  expected to refuse while the camera is ~10 mm off there).

## 2. A fresh demonstration

Record one demonstration in the new setup, with the Franka Hand: `tpgpt_teach.launch.py` and
`tpgpt_teach_keyboard` (`commands.md`). The object, pick spot and shelf slot are your choice. The
demonstration carries its own scene, so later targets can differ. This is the single demonstration
every later run transports. Keep its directory and its hashes.

## 3. Planning dry runs (no motion)

`tpgpt_pick_place` with `dry_run:=true` (the default). Use several objects and pick spots, aimed at
the table first and then at each shelf slot. For each run, record:

- `Shelf world:` in the log: `explained_by_model` close to 1 for an empty, correctly modelled
  shelf, and `removed_as_robot` > 0 whenever the arm is in view;
- in RViz, the shelf model over the shelf cloud, the chosen grasp and the transported path;
- the rejection tally, the rank chosen, the path check (`reachable_fraction`, faults) and its time;
- **`grasp_offset` (audit A5):** plan each scene once with the default and once with
  `-p grasp_offset:="[0.0, 0.0, 0.0]"`. Compare where the chosen grasp sits on the object in the
  overlay image (`<plan>/overlay.png`). The offset was set before the recalibration. Decide it here,
  before any grasp is executed.

## 4. Grasping on the table

Set the destination on the table, beside the pick, so the shelf plays no part. Execute
(`dry_run:=false`) at the default `speed_scale`, with a hand on the e-stop. Try several objects.
This tests calibration and grasp accuracy, the jaws, and the executor, including its
infeasible-pose fallback: read the completion log line.

## 5. Onto the shelf

With the measured shelf: first the top board, then the bottom one. Watch the move onto the path
(`approach_in`) and the placement in RViz before typing `yes`.

## 6. After that

End-to-end on new objects. Then the X6 rework: a new `run_id`, the new slots from `CALIBRATION.md`,
the new demonstration's hashes, and the `method:` block updated to match the new selection (it still
says `max_plans: 12`, `fallback_top_down: true`).
