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
| GL backend | `MUJOCO_GL=egl`. **`osmesa` is broken on this box** (PyOpenGL cannot bind). |
| Compute | **CPU only. No GPU.** Never add a CUDA dependency. |
| Memory | Tight: ~2.5-3.5 GiB free, several GiB of swap already in use. |

```bash
export MUJOCO_GL=egl
/home/ishita/mujoco_env/bin/python -m pytest tests/unit -q          # fast, no sim
/home/ishita/mujoco_env/bin/python -m pytest tests -q               # everything
```

### Memory and CPU rules
- Check `free -h` before any long simulation run.
- Render at 256x256 by default; stream frames to disk, never accumulate a long
  episode of frames in RAM.
- Always close environments explicitly (`env.close()`).
- Run pytest **serially**. Do not use `pytest-xdist`/`-n auto`.
- Headless physics costs ~160 control-steps/s and ~800 MB RSS; add ~250 MB for
  offscreen rendering.

## Rules for changing code

1. **Run the tests after every change.** Unit suite is under a second; there is
   no excuse for skipping it. Run the full suite before declaring anything done.
2. **Cite the paper in docstrings.** Every function implementing theory names
   the section or equation it comes from. This is the point of the project.
3. **Deviations from the paper must be documented** in `ROBOTICS_NOTES.md` with
   the reason, not silently absorbed into code.
4. **No wall-clock time in recorded data.** Timestamps come from the step index
   and the control frequency. (The prototype used `time.time()` around a replay
   loop and produced velocity labels that were mostly zero.)
5. **Seed before `env.reset()`**, never after.
6. Prefer extending an existing module over adding a parallel one; check
   `tpgpt/transport/` and `tpgpt/utils/` for something reusable first.

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
| Figs. 2, 4, 5 | `tpgpt/viz/`, `tpgpt/experiments/` |

## History

`git log` starts from the original three-script prototype
(`affine_transform.py`, `warping_transform.py`, `sim_policy_transport.py`),
committed verbatim before the rewrite. Its known defects are catalogued in
`ROBOTICS_NOTES.md` so they are not reintroduced.
