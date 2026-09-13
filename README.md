# TPGPT — Policy Transportation via Keypoints and Transportation Maps

A research implementation of Franzese, Prakash, Della Santina and Kober,
*"Generalizable Motion Policies through Keypoint Parameterization and
Transportation Maps"* ([arXiv:2404.13458v2](https://arxiv.org/abs/2404.13458)),
validated in MuJoCo on a robot reshelving task.

A single demonstration is transported into new scenes by fitting a nonlinear map
`phi` that carries a set of source keypoints onto their target counterparts. The
map transports every policy label — position, velocity, end-effector
orientation, Cartesian stiffness and damping — and a new policy is refitted on
the transported labels and executed under Cartesian impedance control.

**Result: one demonstration, 20 randomised scenes, 17/20 success at 7.6 mm mean
placement error**, with keypoints displaced up to 375 mm. Ablating the nonlinear
stage of the map drops that to 4/20 (p = 3.3e-5), so the paper's central
machinery is doing the work rather than a rigid alignment.

## Two stages

**Policy transportation** (validated): one demonstration transported into a new
scene by fitting a nonlinear map that carries source keypoints onto their target
counterparts, transporting every policy label through it, and refitting a policy
on the result.

**Grasping and scene understanding** (working): a text prompt resolves to an
object and a destination, the object is segmented into a point cloud from
simulated depth cameras, and GraspGen-X generates 6-DoF grasps for whichever
gripper is fitted. Transporting the demonstration onto an unseen object held by
an unseen hand places **16 of 20** under position-control replay — see
`outputs/keypoints_sweep/FINDINGS.md`.

**Policy execution** (studied): how the transported labels should actually drive
the robot — nine candidate attractor laws, two query sites, on a physics-free
bed and then in simulation. See `docs/dynamics_execution.md`.

## Before you run anything

**This repository is wired to one machine and will not run elsewhere unchanged.**
Nothing here is portable configuration — the paths below are hardcoded in
`CLAUDE.md`, in the commands in this file, and in a few module docstrings:

| what | hardcoded as | why it is not the system Python |
|---|---|---|
| interpreter | `/home/ishita/mujoco_env/bin/python` | `python3` resolves to a conda 3.13 with none of the packages |
| GraspGen-X assets | a sibling checkout at `/home/ishita/task_embod_aware_grasp/` | reached via `GRASPGENX_GRIPPER_CFG_DIR` and `GRASPGENX_PROJECT_ROOT`; nothing is duplicated into this repo |
| robosuite | source install at `/home/ishita/robosuite` | patched locally |
| GL backend | `MUJOCO_GL=egl` | `osmesa` is broken on that box — PyOpenGL cannot bind |

To run this somewhere else: create a Python 3.10 environment from
`requirements.txt`, point `PY` at it, set the two `GRASPGENX_*` variables if you
need grasp generation, and expect `MUJOCO_GL` to need a different value. The
unit suite needs none of that and is the fastest way to check an environment
works. **CPU only — never add a CUDA dependency.**

## Quick start

```bash
export MUJOCO_GL=egl
PY=/home/ishita/mujoco_env/bin/python        # see the warning above

$PY -m pytest tests/unit -q                                  # 592 tests, ~25 s, no simulator
$PY -m pytest tests -q                                       # everything, ~6 min
$PY -m tpgpt.experiments.run_reshelving --config configs/reshelving.yaml
$PY -m tpgpt.experiments.figures --out outputs/figures       # paper Figs. 2, 4, 5
$PY -m tpgpt.experiments.ablate_residual                     # nonlinear-stage ablation
```

Grasp generation needs an external GraspGen-X server (its dependencies conflict
with this project's, so it runs in its own conda environment):

```bash
$PY -m tpgpt.grasp.server                                    # status / how to start
$PY -m tpgpt.experiments.run_grasp_scene \
     --prompt "put the milk carton on the top shelf" \
     --grippers panda,robotiq85,robotiq140
```

## What maps to what

| Paper | Module |
|---|---|
| Sec. III-E-a, Eqs. 4-7 — affine `gamma` | `tpgpt/transport/affine.py` |
| App. A, Eqs. 14-16 — GP, derivative posterior | `tpgpt/transport/gp.py` |
| App. A — SV-GPT, inducing points | `tpgpt/transport/svgp.py` |
| Sec. III-D/F/G, Eqs. 2, 9, 10, 11 — `phi` and label transport | `tpgpt/transport/maps.py` |
| Sec. III-A — the label set | `tpgpt/transport/labels.py` |
| Sec. III-H, Eqs. 12, 13 — uncertainty | `tpgpt/transport/uncertainty.py` |
| Sec. III-B step 2 — refitting `g` | `tpgpt/policy/` |
| Sec. V — scene, keypoints, impedance control, execution | `tpgpt/sim/` |
| Sec. IV-B — metrics and the Mann-Whitney ranking | `tpgpt/metrics/` |
| GraspGen-X bridge, gripper registry, frame contract | `tpgpt/grasp/` |
| Object point clouds and the scene graph | `tpgpt/perception/` |
| Deterministic prompt parsing (no model, no network) | `tpgpt/language/` |
| Figs. 2, 4, 5 | `tpgpt/viz/`, `tpgpt/experiments/figures.py` |
| Keypoint constructions and their sweep | `tpgpt/experiments/run_keypoint_transport.py`, `run_keypoint_replay.py` |
| Execution laws, the surrogate plant and their sweep | `tpgpt/policy/rollout.py`, `tpgpt/experiments/sweep_execution.py`, `identity_execution.py` |
| Which stage of a run failed, and by how much | `tpgpt/experiments/diagnose.py` |
| What code produced a result | `tpgpt/reporting/provenance.py` |

## Scope

**Implemented and validated:** all of Sec. III, Appendix A, uncertainty
propagation, the policy refit, and the Sec. V-A reshelving result. Beyond the
paper: grasp generation and filtering, keypoint extraction for scenes with
objects and grippers the demonstration never saw, and a study of how the
transported policy should be executed.

**Deferred:** the Sec. IV baseline comparison (Reshaped-KMP, Laplacian Editing,
LWT, Ensemble-NN, TP-GMM, TP-HMM) and Table I; the dressing, surface-cleaning and
DINO-keypoint experiments of Sec. V-B to V-D. `tpgpt/metrics/` already provides
what the baselines would need. See `ROBOTICS_NOTES.md` §8 for the current open
items.

## Where the experimental results live

Three documents, in decreasing order of how much detail they carry. Each is
written to be readable on its own, with every table's columns defined and every
number's provenance stated.

| document | what it answers |
|---|---|
| **`outputs/keypoints_sweep/FINDINGS.md`** | **Which keypoints should pin the transportation map?** Experiments A–S. The answer is the *grasp-pose cube*: Experiment R places **16 of 20** across five grippers and four objects from a single Panda demonstration, and all of the remaining failures are the gripping mechanism rather than the map. |
| **`docs/dynamics_execution.md`** | **How should the transported policy be executed?** Experiments A–H, comparing nine attractor laws and two query sites. Query the policy at the attractor, not the measured arm; keep the shipped integrator, because in physics no law beats another. Includes four figures in `docs/figures/`. |
| **`ROBOTICS_NOTES.md`** | The running log behind both — every finding, every deviation from the paper with the measurement that motivated it, and every retraction. **Read this first if you want to know *why* the code is the way it is.** Sections 7.26 and 7.27 in particular are about measurements that turned out to be wrong and how. |

Raw run data sits under `outputs/<experiment>/`. Only the prose and the
`manifest.json` files are tracked — each manifest records the git commit, the
dirty and untracked files, and the complete settings behind that run, so a table
can always be traced to the code that produced it. Bulk arrays are gitignored.

- **`CLAUDE.md`** — environment, constraints, and working rules for this box.

## Requirements

CPU only, Python 3.10, MuJoCo 3.7.0 and robosuite 1.5.2. See
`requirements.txt`. Grasp generation additionally needs the external GraspGen-X
server, which runs in its own conda environment because its dependencies
conflict with this project's — `tests/integration/test_graspgen.py` skips when
it is unreachable, so the suite still runs without it.

Memory is the binding constraint rather than CPU: headless physics costs about
800 MB per environment and the GraspGen-X model is 2.6 GB resident, so check
`free -h` before a long campaign and run pytest serially (no `pytest-xdist`).
