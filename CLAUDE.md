# CLAUDE.md — working rules and context for the TPGPT project

## What this project is

A research-grade implementation of **Franzese, Prakash, Della Santina, Kober,
"Generalizable Motion Policies through Keypoint Parameterization and
Transportation Maps"** (arXiv:2404.13458v2, `TPGPT.pdf` in this directory).

Long-term goal: transport policies across different **objects** and different
**embodiments** from a **single** source demonstration, with a publishable
implementation and analysis. Everything here is built to serve that.

Robotics-side findings, results and decisions live in **`ROBOTICS_NOTES.md`**.
This file is process, environment and code rules.

## Environment — read this before running anything

| Item | Value |
|---|---|
| Interpreter | `/home/ishita/mujoco_env/bin/python` (Python 3.10) |
| **Not** the default | `python3` resolves to conda 3.13, which has none of the packages |
| Simulator | MuJoCo 3.7.0 + robosuite 1.5.2 (source install at `/home/ishita/robosuite`) |
| Grasp generation | GraspGen-X, **out of process**. See "Two-process setup" below. |
| GL backend | `MUJOCO_GL=egl`. **`osmesa` is broken on this box** (PyOpenGL cannot bind). |
| Compute | **CPU only. No GPU.** Never add a CUDA dependency. |
| Memory | Tight: ~2.5-3.5 GiB free, several GiB of swap already in use. |

```bash
export MUJOCO_GL=egl
/home/ishita/mujoco_env/bin/python -m pytest tests/unit -q          # fast, no sim
/home/ishita/mujoco_env/bin/python -m pytest tests -q               # everything
```

### Two-process setup for grasp generation

GraspGen-X cannot run in `mujoco_env`. It needs Python 3.11 with
`diffusers==0.11.1` and `huggingface-hub==0.25.2`, which conflict with this
project's stack, and its model is 1.6 GB. It runs as a **long-lived ZMQ server**
in its own conda environment; TPGPT talks to it with a torch-free client.

```bash
python -m tpgpt.grasp.server                    # status, and how to start it
/home/ishita/task_embod_aware_grasp/6dof_GraspMAS/scripts/run_server.sh --daemon
```

- Server: `127.0.0.1:5556`, CPU, **2.6 GB resident**, 3.3 s model load,
  4-12 s per inference. Budget that against MuJoCo's ~1 GB.
- Assets come from the sibling checkout via `GRASPGENX_GRIPPER_CFG_DIR` and
  `GRASPGENX_PROJECT_ROOT`. Nothing is duplicated into this repo.
- Only `franka_panda`, `robotiq_2f_85` and `robotiq_2f_140` are loaded by
  default; other grippers need the server restarted with them.
- `tests/integration/test_graspgen.py` **skips** when the server is unreachable,
  so the suite still runs offline. Never start the server implicitly from a test.

### Memory and CPU rules
- Check `free -h` before any long simulation run.
- Render at 256x256 by default; stream frames to disk, never accumulate a long
  episode of frames in RAM.
- Always close environments explicitly (`env.close()`).
- Run pytest **serially**. Do not use `pytest-xdist`/`-n auto`.
- Headless physics costs ~160 control-steps/s and ~800 MB RSS; add ~250 MB for
  offscreen rendering.

## Rules for changing code

1. **Run the tests after every change.** The unit suite takes 3 seconds; the
   full suite ~75 s. Run the full suite before declaring anything done.
2. **Cite the paper in docstrings.** Every function implementing theory names
   the section or equation it comes from. This is the point of the project.
3. **Deviations from the paper must be documented** in `ROBOTICS_NOTES.md` with
   the measurement that motivated them, not silently absorbed into code.
4. **Measure before concluding, and correct the record when a measurement
   supersedes an earlier one.** `ROBOTICS_NOTES.md` §2.5 contains a worked
   example: an early diagnosis blamed the time belief for a rollout stall, and
   a later measurement showed the cause was under-regularisation. The note now
   carries both, marked as a correction.
5. **No wall-clock time in recorded data.** Timestamps come from the step index
   and the control frequency.
6. **Seed before `env.reset()`**, never after.
7. Prefer extending an existing module over adding a parallel one; check
   `tpgpt/transport/` and `tpgpt/utils/` first.

## Things that will bite you again

Each of these cost real debugging time. Full detail in `ROBOTICS_NOTES.md`.

- **A policy is not a transportation map.** They share the GP code but need
  opposite priors: the map interpolates sparse keypoints exactly; the policy
  must regularise (`noise_variance` 1e-2) and needs a length scale set in
  standardised units, not from label spacing. Getting this wrong fails
  *silently* — the policy fits its labels perfectly and commands zero velocity
  2 cm away.
- **Labels are the commanded attractor, not the measured pose.** An
  impedance-controlled arm lags its attractor by 20-45 mm. Recording the
  measured pose and executing it as an attractor applies that lag twice.
- **robosuite rescales actions onto `output_max`.** Leaving it at the default of
  1 divides every commanded torque by the ratio; the impedance law then looks
  broken when it is only attenuated.
- **The arm's reachable envelope, not the paper, sets the scene geometry.**
  End-effector x saturates near 0.24 m in this arena.
- **Check `free -h` before long runs**, and never accumulate rendered frames in
  memory — stream them with `FrameWriter`.
- **Segmentation instance ids are offset by one.** robosuite adds 1 so that 0
  means background; using the raw index masks the background and yields a 2.5 m
  "object cloud" that looks like a calibration failure, not an off-by-one.
- **Observation images are vertically flipped** relative to the camera matrix,
  and the pixel vector is `[col*z, row*z, z, 1]`. Either mistake moves the cloud
  about 27 cm, in opposite directions.
- **A grasp pose is at the gripper base, not the fingertips.** Converting it
  needs the hand's own TCP depth (103-195 mm across the registry) plus a
  rotation about the approach axis. Never assume identity: unmeasured grippers
  refuse conversion deliberately.
- **Point clouds sent to GraspGen-X must be capped** (8192 points). Its outlier
  removal is `torch.cdist(X, X)` and a large cloud OOM-kills the server.

## Scope decisions (agreed 2026-09-05)

**In scope now:** the full theory of Sec. III + Appendix A, uncertainty
propagation, the policy refit of Sec. III-B step 2, the **reshelving** scene of
Sec. V-A, the theory figures 2/4/5, a package restructure with a pytest suite,
and git.

**Explicitly deferred** (the layout leaves room for them, no rework needed):
- Sec. IV baselines: Reshaped-KMP, Laplacian Editing, LWT, Ensemble-NN,
  Ensemble Neural Flows, TP-GMM, TP-HMM; Table I; Figs. 6-10 and the
  Mann-Whitney ranking. `tpgpt/metrics/` already provides the metrics they need.
- Sec. V-B dressing (MuJoCo `flexcomp` cloth is viable on CPU — verified at
  8.3x realtime for a 9x9 sheet — but not built).
- Sec. V-C surface cleaning and Sec. V-D DINO keypoints.

**Dependencies:** minimal, CPU-only. `similaritymeasures`, `pyyaml`,
`pytest-cov`, `imageio` were added. Torch was explicitly *not* added.

## Package map (paper section -> module)

| Paper | Module |
|---|---|
| Sec. III-E-a, Eqs. 4-7 (`gamma`) | `tpgpt/transport/affine.py` |
| App. A, Eqs. 14-16 (GP, derivative posterior) | `tpgpt/transport/gp.py` |
| App. A (SV-GPT, inducing points) | `tpgpt/transport/svgp.py` |
| Sec. III-D/F/G, Eqs. 2, 9, 10, 11 (`phi`, labels) | `tpgpt/transport/maps.py` |
| Sec. III-A (label set) | `tpgpt/transport/labels.py` |
| Sec. III-H, Eqs. 12, 13 | `tpgpt/transport/uncertainty.py` |
| Sec. III-B step 2 (refit `g`) | `tpgpt/policy/` |
| Sec. V (simulation, keypoints, impedance control) | `tpgpt/sim/` |
| Sec. IV metrics | `tpgpt/metrics/` |
| Grasp generation, gripper registry, frame contract | `tpgpt/grasp/` |
| Object clouds and the scene graph | `tpgpt/perception/` |
| Deterministic prompt parsing | `tpgpt/language/` |
| Figs. 2, 4, 5 | `tpgpt/viz/`, `tpgpt/experiments/` |

## Current state

**Transportation (done).** All of Sec. III, Appendix A, uncertainty propagation,
the policy refit and the Sec. V-A reshelving validation. One demonstration
transported into 20 randomised scenes: 17/20 success at 7.6 mm mean placement
error; 4/20 without the nonlinear stage (p = 3.3e-5). See `ROBOTICS_NOTES.md` §4.

**Grasping and scene understanding (in progress).** Text prompt -> object and
destination -> object point cloud -> ranked 6-DoF grasps per gripper ->
end-effector target. Nine gripper pairs registered, six with measured frames.
See `ROBOTICS_NOTES.md` §5.

226 tests: 173 offline, 53 requiring the simulator, 10 gated on the grasp server.

**Next, and blocked on discussion** — do not implement these without agreeing the
design first:

1. **Grasp filtering and selection.** Measured motivation in §5.6: only ~25% of
   candidates approach from above, and executing the top-scoring ones lifts
   nothing. Needs visibility, reachability and collision criteria.
2. **Keypoint extraction for the new scene.** An arbitrary mesh has no natural
   corner set, and keypoints must be *paired* between source and target for the
   transportation map to exist at all.

## History

`git log` starts from the original three-script prototype
(`affine_transform.py`, `warping_transform.py`, `sim_policy_transport.py`),
committed verbatim before the rewrite and removed afterwards. Its known defects
are catalogued in `ROBOTICS_NOTES.md` §1 so they are not reintroduced.
