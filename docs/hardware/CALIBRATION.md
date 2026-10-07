# Camera calibration and measured geometry

Calibration of the RealSense D435 (`base` -> `camera_color_optical_frame`), and the scene geometry
measured by touching. Referenced from `CLAUDE.md`.

## Current calibration (2026-10-07)

```yaml
translation: [1.177384, 0.463198, 0.865572]
rotation_quaternion_xyzw: [-0.406543, -0.828382, 0.343094, 0.175476]
expected_table_height: -0.0132
```

The camera was knocked, then re-aimed by the user. Verified: **table -3.5 mm, top board -0.2 mm**,
and laterally the shelf fit wants only +3.7 / +4.0 mm and 0.70 deg. Backups `.bak`..`.bak5` in
`~/calib_tools/`.

**A lateral camera move is invisible to the height check.** Before this recalibration the old
extrinsics still read the table +2.1 mm and the top board +0.2 mm -- yet they were **34 mm out
sideways**. A sideways shift or a yaw does not change the measured height of a horizontal surface.
Always run the **shelf face fit** as well (`tpgpt.hardware.shelf.fit_to_cloud`): the giveaway was
**0 points on the shelf's front faces** and a -34.4 mm correction, against 3633 points over a
606 mm span and +3.7 mm after.

## Touching with the Inspire hand mounted

The Franka Hand's `probe_offset` of `[0, 0, 0.0095]` is meaningless with the Inspire mounted, and
`rest_point.py` models Franka boxes. Use the Inspire's **index fingertip** instead: at the pointing
pose it is a fixed point in the TCP frame, which is all `probe_offset` needs.

```bash
ros2 service call /inspire_hand/point std_srvs/srv/Trigger          # inspire_hand_node running
ros2 param set /calibration_check_node probe_offset "[-0.01507, -0.02846, 0.12135]"
```
Set it **after** `calib_setup.sh` (which writes the Franka value), and expect `calib_preflight.sh`
to flag that one line as BAD -- it checks for `0.0095`. Every other line must still read OK.

Derivation, should the mount change: `grasp_from_ee(R) = R @ C` exactly, so the tip offset is
`C @ tip_local + tool_offset` with `C = ee.grasp_from_ee(I)` and `tip_local` the
`index_intermediate` mesh vertex furthest from its joint, at `registers_to_q([0,0,0,1000,1000,1000])`.
**Build the end effector with the measured mount** -- `inspire_end_effector(**base)` alone silently
uses the default 135 deg / 12.0 mm and the offset comes out 3.7 mm wrong; pass `_mount(params)` from
`inspire_mount._base_from_config` too. Verified against `inspire_mount.touch_points` to 0.000 mm.
The little finger's is `[-0.01426, +0.04029, +0.11505]`.

**It is a worse probe than the Franka's fingertip face.** The mount's taped ring rocks ~5 mm at the
fingertips and its yaw is +-1.4 deg at a 121 mm lever, so correspondences come out ~11 mm RMS
against 7-8 mm. **Vary the hand's orientation between touches**: held at one orientation the mount
error becomes a single constant vector and is absorbed straight into the fitted camera position --
a clean-looking fit that is systematically wrong.

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
