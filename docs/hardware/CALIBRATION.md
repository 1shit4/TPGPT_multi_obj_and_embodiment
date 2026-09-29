# Camera calibration and measured geometry

Calibration of the RealSense D435 (`base` -> `camera_color_optical_frame`), and the scene geometry
measured by touching. Referenced from `CLAUDE.md`.

## Current calibration (2026-09-26)

In `ycb_pointcloud_pipeline/config/camera_calibration.yaml` (rebuild after editing it):

```yaml
translation: [1.199143, 0.452138, 0.859710]
rotation_quaternion_xyzw: [-0.382130, -0.837698, 0.349104, 0.174255]
expected_table_height: -0.0132
```

The setup was moved by someone else and the camera knocked ~46 mm / 2.9 deg; the old calibration
read the table +24.6 mm and the top board +44.6 mm out. 9 correspondences, all touched face-flat
(fit RMS 7.1 mm), refit jointly with two plane constraints. **Verified: table +2.4 mm, top board
+2.2 mm.** Backups `.bak` (09-11) `.bak2`/`.bak3` (09-21) `.bak4` (09-23) in `~/calib_tools/`.

**Calibrating from the arm's own geometry does NOT work** -- tried 2026-09-26 as a way to skip the
stickers. Three sweeps of one unchanged setup returned camera positions **176 mm apart**, and after
fitting the arm was still 12-17 mm from the model. `fr3.LinkHull` is the *convex hull* of a
*non-convex* link, so "distance to the hull" is not "distance to the arm"; the camera also sees one
side only. Scripts kept, marked broken at the top: `~/calib_tools/arm_sweep.py`,
`check_against_arm.py`. **Use correspondences.**

What the arm *can* do is tell you the camera moved at all: the TCP overlay
(`/tpgpt/verify/overlay`) is pure projection from the robot's own pose -- no depth, no hulls, so it
is immune to warm-up and to the furniture moving. If the cross is off the fingertips, the camera
moved. 46 mm at 1 m is about 29 px, which reads as "a little off".

## Measured geometry (touched, 2026-09-26)

Base frame. **z = 0 is the robot's base plane.** Repeatability about 1 mm.

| what | x | y | z / surface height |
|---|---|---|---|
| table | | | **-0.0132** |
| top shelf, left slot | 0.6884 | -0.2758 | **0.3211** |
| top shelf, right slot | 0.4634 | -0.3505 | **0.3211** |
| bottom shelf (both slots) | as above | | **0.0731** |

In `config/shelf.yaml` under `shelf: anchors:`.

**The shelf itself keeps coming back to the same place.** Across the 09-23 and 09-26 disturbances
the slot marks returned within 1.6-3.9 mm in x/y and the table within 1.2 mm. What moves is the
**top board**: +9.7 mm on 09-23, back down to 0.3211 on 09-26 (0.3213 originally). Check the board,
not the shelf.

**The bottom board cannot be touched from above** -- the top board is over it. Reach in
horizontally and rest the finger **pads** on it (`~/calib_tools/rest_point.py`, read `PADS z`):
three rests gave 0.0730/0.0731/0.0732, a **0.2 mm** spread, the best measurement of that board yet
(the 09-23 value spread 2.7 mm and sat 1.6 mm higher). The tool's caution line reading "body
+23 mm" is expected and correct -- it is the 31.6 - 8.75 mm the hand geometry predicts for a
horizontal hand on a pad's side face, with the body hanging past the front edge in free air.
The camera cannot settle this board: it sees it at a grazing angle, fitting 8-12 deg of spurious
tilt and +12 mm.

**Diagnosing what moved**, in order:
1. `python3 ~/calib_tools/check_calibration.py` -- surfaces against the touched anchors.
2. **Table error = the camera**, since the table cannot move relative to the robot... *unless the
   table was moved too* (it was, slightly, on 09-26). When everything may have shifted, no cloud
   surface is a reference: touch first, then compare the cloud against those fresh touches.
3. Shelf height **relative to the table** cancels a global camera shift; a real shelf move does not.

## Let the camera warm up first (measured 2026-09-23)

**The D435's depth drifts by about 10 mm during the first ~15 minutes after the driver starts.**
That is larger than the calibration's own accuracy, so a cold camera will make a good calibration
look broken. Measured on the table, against its touched height, from a cold start:

| elapsed | table error | top shelf error |
|---|---|---|
| 0.1 min | -8.1 mm | -2.7 mm |
| 5.8 min | -4.0 mm | -1.2 mm |
| 11.6 min | -1.1 mm | +0.9 mm |
| 17.4 min | +1.0 mm | +1.6 mm |
| 26.1 min | +2.2 mm | +2.2 mm |

It settles after roughly 15-17 minutes, and warm the calibration reads **+1.5 to +2 mm** on both
surfaces -- better than the +/-5 mm quoted above, which was measured without controlling for this.

**Wait ~15 minutes after starting the camera before calibrating, measuring, or running trials.**
`~/calib_tools/warmup_drift.py` samples the table and top shelf every ~3 minutes if you need to
confirm it has settled.

A corollary for diagnosing: a *uniform* shift of every surface is thermal or the camera; the shelf
having moved would change the shelf's height **relative to the table**, which a global drift leaves
untouched. Check the relative number before concluding anything has been bumped.

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
   `/calibration_check_node/record_correspondence`. Always
   **press the fingertip FACE flat on the surface** so its centre is the contact point — that is
   what `probe_offset` models. On a horizontal face the hand is near vertical (tilt < ~30 deg);
   on a **vertical** face (an object's front, e.g. on the bottom shelf where the hand cannot
   come down from above) the hand is near horizontal, ~80 deg tilt, and that is **correct**.
   The error to avoid is the face meeting the surface at an angle, where an edge contacts.
4. `/calibration_check_node/solve_extrinsics` writes `~/tpgpt_calibration/extrinsics_fit_*.yaml`,
   which records each pair's raw TCP position and rotation, so a pair can be re-derived offline.
5. **Judge the result by surfaces, not by fit RMS.** Fit the table and both shelf boards in the cloud
   and compare with touched heights (`~/calib_tools/check_calibration.py`). A single good table reading
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
| `~/calib_tools/rest_point.py` | Height of the lowest point of the finger **pads**, live. For a surface you cannot reach from above (the bottom board): rest the pads on it and read `PADS z`. Straight down it reads 9.5 mm below the TCP, matching `touch_point.py` |
| `~/calib_tools/click_measure.py` | Click a point in RViz; fits the local plane and reports its height |
| `~/calib_tools/shelf_points.py` | Touches (Enter) and RViz clicks, printed in the shelf model's axes (along the front / into / up); pairs click and touch by label to give the camera's error at the shelf. tpgpt venv. |
| `~/calib_tools/check_calibration.py` | **The health check.** Table and both boards vs their touched anchors, plus shelf-relative-to-table; warns if the camera is cold |
| `~/calib_tools/warmup_drift.py` | Tracks the table while the camera warms, to confirm it has settled |
| `~/calib_tools/validate.py`, `final_fit.py` | One-off: score candidate extrinsics from a specific `extrinsics_fit_*.yaml` against the table plane |
| `~/calib_tools/ripple.py` | Depth noise check: temporal noise should be under 0.5 mm with the filters on |

The heavy temporal smoothing is for calibration only. It is wrong for a live scene where objects and
the arm move — revisit it before running trials.

## Why the probe offset is 9.5 mm

From `franka_description` (see `TPGPT_.../tpgpt/hardware/hand.py`): the finger pad is an 18.5 mm box
centred 103.65 mm along the hand's +Z, so its face is at 112.9 mm; `fr3_hand_tcp` is at 103.4 mm.
With the offset set, three table touches agreed to 0.2 mm; without it, to 3.2 mm.
