# X6: where every piece of data is stored

`<root>` means `~/thor_ws/src/ycb_point_cloud/TPGPT_multi_obj_and_embodiment/outputs/hardware/hw_2026_09_x6/`
(`experiment.data_root` + `experiment.run_id` in `config/x6_experiment.yaml`).

That `outputs/` directory is git-ignored by TPGPT, except `*.md` and
`manifest.json` files. So bulk arrays and video never go into git, but the manifest
that names the code behind the run does.

**Status:** built (the robustness re-runs are not). The recorder runs inside
`tpgpt_pick_place` when started with `experiment:=true`, and the layout below is what it
writes. Scratch trials (id starting with `s`) go to `<root>/scratch/` instead of `trials/`. With `experiment:=false` nothing here is written.

## Inputs (read, never written by a trial)

| what | where |
|---|---|
| Campaign config (method frozen, factors, seeds) | `x6_experiment/config/x6_experiment.yaml` |
| Hand descriptions | `x6_experiment/config/hands/panda.yaml`, `hands/inspire.yaml` (placeholder) |
| Segmentation prompts | the package's own `ycb_pointcloud_pipeline/config/target_objects.yaml` (chosen on labelled frames, `docs/hardware/SEGMENTATION.md`); each trial records the file it ran with. X6 had its own until 2026-09-29 |
| Camera extrinsics as loaded | `~/thor_ws/install/ycb_pointcloud_pipeline/share/ycb_pointcloud_pipeline/config/camera_calibration.yaml` (installed copy of `ycb_pointcloud_pipeline/config/camera_calibration.yaml`) |
| Demonstration as recorded by `tpgpt_teach_node` | `~/tpgpt_demos/banana_<YYYYMMDD_HHMMSS>/` (`raw.npz`, `teach.json`, `source_bundle/`) |
| The brief | `HARDWARE_EXPERIMENT.md` |

## Campaign directory

```
<root>/
  manifest.json                 # written by the FIRST experiment:=true trial: config sha256,
                                #   provenance of all 3 repos + venvs, hand files sha256, demo sha256
  orders/order_panda.csv        # randomised order sheet (trial id -> object, mark, slot), seed 6001
  orders/order_inspire.csv      #   seed 6002
  trial_index.csv               # appended by every trial: trial_id, hand, object, destination, status
  demonstration/
    source_bundle/              # byte copy of ~/tpgpt_demos/banana_<time>/source_bundle
    teach.json  raw.npz         # copies, for the record
    sha256.txt                  # checked before every trial
  hands/
    panda.yaml  inspire.yaml    # copies as used, with caliper measurements
  sessions/<session_id>/        # session_id = s01, s02, ... (not a timestamp)
    calibration.json            # optional session: extrinsics file sha256 at session start
    session_log.jsonl           # which trials ran in this session, aborts, remounts
  instrument/
    before_first_trial/rep_00 .. rep_09/   # rgb.png depth.npy mask.png cloud.npy capture.json
    before_first_trial/instrument.json     # centroid per rep, SD per axis and lateral
    after_remount/...
    campaign_end/...
  trials/
    trial_0001/                 # name = trial_id given on the command line; never overwritten
      record.json               # all scalars (brief §3 schema), rewritten after every stage
                                #   conditions.pick_pose = cloud centroid xyz + yaw (from the cloud)
      preflight.json            # each check, value, pass/fail
      rgb.png depth.npy         # perception frame(s) the grasp was computed from
      mask.png cloud.npy        # segmentation and fused base-frame cloud (N x 3)
      camera_info.json          # intrinsics, T_base_camera
      candidates.json           # FULL ranked list: raw + refined pose, score, verdict per stage
      keypoints.npz             # source/target keypoints, pick and place blocks (all four)
      plan.npy                  # transported waypoints N x 7 (xyz + quat xyzw), base frame
      plan_dir/                 # the planner's own output (plan.json, transported_labels.npz, ...)
      plan_overlay.png          # camera image + planned path + grasp hand + release + destination
      trace.npz                 # per control step: step_index, ee_pose, attractor_pose,
                                #   joint_positions, gripper_command, gripper_width, wrench;
                                #   a multi-finger hand: hand_names, hand_position, hand_effort
      settled_rgb.png settled_depth.npy settled_mask.png settled_cloud.npy
      operator.json             # placed, first_failure_stage, notes, abort/exclusion reason
    trial_0002/ ...
  scratch/trial_s001/ ...       # test trials (trial_id starting with s), never analysed
  robustness/rb_0001/ ...       # 2nd/3rd admissible candidate re-runs, after both grids
  summary.csv                   # one flat row per trial (regenerated from the records)
  analysis/
    analysis.md                 # §5 statistics, tests named, assumptions stated, "What went wrong"
    table5.csv  table5.tex      # IEEEtran tabular with Wilson totals row
    figures/                    # plots, if any
```

You give the trial id on the command line (`trial_id:=0001`). By convention panda
uses `0001`–`0030` and inspire `0031`–`0060`. Hand, object and destination come from the
command line. The pick pose comes from the point cloud. A trial that raises or is
stopped with Ctrl-C still leaves its directory, with whatever it recorded and
`outcome.operator_aborted` / `abort_reason` set. Reusing an existing trial id is refused.

## Things you record by hand (outside the node)

| what | where to put it | file name |
|---|---|---|
| Fixed-camera trial video | `<root>/video/` (also copy or symlink into each trial dir as `video.mp4`) | `trial_0001.mp4` |
| Photo 1, demonstration mid-carry | `<root>/photos/` | `p1_demo_midcarry.jpg` |
| Photo 2, 4-frame strip per successful (hand, object) | `<root>/photos/` | `p2_<hand>_<object>_trial<NNNN>_{1_approach,2_closed,3_midcarry,4_released}.jpg` |
| Photo 3, both hands at the same grasp, same object, same viewpoint | `<root>/photos/` | `p3_samegrasp_<object>_{panda,inspire}.jpg` |
| Photo 4, workspace overview | `<root>/photos/` | `p4_workspace.jpg` |
| Photo 5, each hand alone, plain background, same scale | `<root>/photos/` | `p5_hand_{panda,inspire}.jpg` |
| Photo 6, every failure at the moment it happens (frame grab from video) | `<root>/photos/failures/` | `p6_trial<NNNN>_<stage>.jpg` |
| Caliper measurements of both hands | `x6_experiment/config/hands/<hand>.yaml` → `measured:` | |

## Existing (non-X6) output locations, unchanged

| what | where |
|---|---|
| Planner runs done by hand | `~/tpgpt_plans/<object>_<time>/` |
| Executor CSV logs done by hand | `~/tpgpt_plans/<object>_<time>/executions/` |
| Demos | `~/tpgpt_demos/` |
| GraspGen-X server log | `~/.ros/graspgenx_server.log` |
