# Camera calibration and measured geometry

Calibration of the RealSense D435 (`base` -> `camera_color_optical_frame`), and the scene geometry
measured by touching. Referenced from `CLAUDE.md`.

## Current calibration (2026-09-21)

In `ycb_pointcloud_pipeline/config/camera_calibration.yaml` (rebuild after editing it):

```yaml
translation: [1.196360, 0.504620, 0.860023]
rotation_quaternion_xyzw: [-0.371615, -0.835432, 0.368495, 0.167833]
expected_table_height: -0.0136
```

Camera 1.20 m out, 0.50 m to the side, 0.86 m up, pitched 42 deg down. Intrinsics come live from
`camera_info` at **848x480** — the stream must run at that resolution.

**Accuracy is about +/-5 mm across the workspace.** Validated on a fresh depth frame against three
surfaces whose heights were measured by touching:

| surface | cloud minus touch |
|---|---|
| table | -4.9 mm (tilt 0.61 deg) |
| top shelf | +0.1 mm |
| bottom shelf | +5.4 mm |

Backups of earlier passes: `~/calib_tools/camera_calibration.yaml.bak` (the 2026-09-11 original),
`.bak2` (first pass of 2026-09-21).

## Measured geometry (touched, 2026-09-21)

Base frame. **z = 0 is the robot's base plane.** Touched with the closed fingertip face, corrected
by the 9.5 mm probe offset; repeatability about 1 mm.

| what | x | y | z / surface height |
|---|---|---|---|
| table | | | **-0.0136** (slopes ~0.4-0.6 deg) |
| top shelf, left slot | 0.6842 | -0.2774 | **0.3213** |
| top shelf, right slot | 0.4636 | -0.3529 | **0.3213** |
| bottom shelf, left slot | 0.6842 | -0.2774 | **0.0768** |
| bottom shelf, right slot | 0.4636 | -0.3529 | **0.0768** |

The bottom slots take the top slots' x/y — the user confirmed they sit directly below. The two slots
are 233 mm apart along an axis 19 deg off base -x: **the shelf is not square to the robot.**

These are **not yet in `x6_experiment.yaml`**, which still holds the pre-move values
(`top_shelf`/`bottom_shelf` at a single `(0.54, -0.4)`, heights 0.35 / 0.08, table -0.006 — the top
shelf height there is 29 mm too high).

The bottom shelf is poorly seen by this camera (grazing angle, recessed under the top shelf): plane
fits there carry 5-17 deg of spurious tilt. Trust the touched height, not the cloud.

## How to recalibrate

Needs the robot container, the camera (Terminal 1), `calibration_check_node` (Terminal 2), RViz
(`rviz2 -d ~/calibration.rviz`), and `free_drive` to hand-guide the arm.

1. `bash ~/calib_setup.sh`, then **`bash ~/calib_preflight.sh` — every line must read OK.** It exists
   because a zeroed `probe_offset` or a node running stale code is invisible while you work.
2. Put **small high-contrast stickers in the middle of flat surfaces, at least 5 cm from any edge**,
   spread in x, y **and z** (table, both shelf boards). Not corners: edges return "flying pixels"
   whose depth interpolates to the background — corners gave 33.7 mm RMS against ~8 mm for stickers.
3. For each: arm out of view, wait ~2 s for the cloud to settle, **Publish Point** on the sticker,
   then close the gripper, place the **centre of the fingertip face** on the sticker and call
   `/calibration_check_node/record_correspondence`. Keep hand tilt **under 30 deg** — beyond that an
   edge of the face touches instead of its centre.
4. `/calibration_check_node/solve_extrinsics` writes `~/tpgpt_calibration/extrinsics_fit_*.yaml`,
   which records each pair's raw TCP position and rotation, so a pair can be re-derived offline.
5. **Judge the result by surfaces, not by fit RMS.** Fit the table and both shelf boards in the cloud
   and compare with touched heights (`~/calib_tools/validate.py`). A single good table reading
   proves nothing: the first 2026-09-21 fit read -0.5 mm on the table while 9 mm out elsewhere — a
   residual rotation. Pruning outlier pairs lowered fit RMS while making the surfaces *worse*.
6. If the surfaces disagree in a position-dependent way, refit jointly against the point pairs and
   horizontal-plane constraints at the touched heights. That is how the current values were made.

## Tooling

All in the home directory, so they survive reboots.

| file | purpose |
|---|---|
| `~/calib_setup.sh` | Reapplies runtime params: temporal filter alpha 0.05 (stops the cloud jittering), spatial filter, `depth_stride 2`, **`probe_offset [0,0,0.0095]`** |
| `~/calib_preflight.sh` | Verifies all of the above, and that the node is newer than the last build |
| `~/calibration.rviz` | Fixed Frame `base`, scene cloud as RGB8 flat squares (RGB8 is needed to see stickers), RobotModel off (its meshes live in the container) |
| `~/calib_tools/touch_point.py` | Live readout of the fingertip-face position and hand tilt, for measuring by touch |
| `~/calib_tools/click_measure.py` | Click a point in RViz; fits the local plane and reports its height |
| `~/calib_tools/shelf_points.py` | Touches (Enter) and RViz clicks, printed in the shelf model's axes (along the front / into / up); pairs click and touch by label to give the camera's error at the shelf. tpgpt venv. |
| `~/calib_tools/validate.py`, `final_fit.py` | Score candidate extrinsics against the real table plane |
| `~/calib_tools/ripple.py` | Depth noise check: temporal noise should be under 0.5 mm with the filters on |

The heavy temporal smoothing is for calibration only. It is wrong for a live scene where objects and
the arm move — revisit it before running trials.

## Why the probe offset is 9.5 mm

From `franka_description` (see `TPGPT_.../tpgpt/hardware/hand.py`): the finger pad is an 18.5 mm box
centred 103.65 mm along the hand's +Z, so its face is at 112.9 mm; `fr3_hand_tcp` is at 103.4 mm.
With the offset set, three table touches agreed to 0.2 mm; without it, to 3.2 mm.
