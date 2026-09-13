# One demonstration, many hands and many objects

**The question.** A robot is shown a pick-and-place **once**, with one gripper,
on one object. Can that single demonstration be made to work for a *different
object*, picked up by a *different hand*, and put somewhere *different* — with no
retraining, no second demonstration, and no hand-tuning per case?

This repository is the attempt, in MuJoCo. It is **not** a reimplementation of a
paper. It is built **on top of** the transportation-map machinery of Franzese,
Prakash, Della Santina and Kober, *"Generalizable Motion Policies through
Keypoint Parameterization and Transportation Maps"*
([arXiv:2404.13458v2](https://arxiv.org/abs/2404.13458)) — that method is
implemented faithfully here and serves as the substrate — but the research
question is a different and larger one. The paper transports a policy between
scenes; this asks whether a policy can also be transported across **embodiments**
and across **objects the demonstration never saw**, which needs machinery the
paper does not supply: perceiving an unseen object, choosing a grasp for a hand
with different geometry, and deciding what the keypoints should even be when the
two objects share no correspondence.

**Where it stands.** From a single Panda demonstration on one object:

| | result |
|---|---|
| the paper's own setting — same object and hand, 20 randomised scenes | **17/20**, 7.6 mm mean placement error, against 4/20 without the nonlinear stage (p = 3.3e-5) |
| **five different hands × four different objects**, position-control replay | **16/20** |
| the same, executed by the learned policy under impedance control | **15/20** |

The five hands span 24 to 61 mm of tool offset and 50 to 125 mm of jaw
aperture, and include revolute linkages as well as parallel jaws; the four
objects are a milk carton, a can, a 15 cm cereal box and a loaf of bread. Only
one hand-and-object pair — the Panda on its own object — is the one the
demonstration was recorded with.

## The three pieces the question breaks into

**1. Transportation** — the inherited substrate, and the part that is settled.
Given a set of *source keypoints* (where things were in the demonstration) and
*target keypoints* (where the corresponding things are now), fit a smooth
nonlinear map `phi` that carries one onto the other, push every policy label
through it — position, velocity, end-effector orientation, Cartesian stiffness
and damping — and refit a policy on the result. This is the paper's method, and
it reproduces: 17/20 at 7.6 mm in the paper's own setting, against 4/20 with the
nonlinear stage ablated (p = 3.3e-5), so the map is doing the work rather than a
rigid alignment hiding inside it.

**2. Deciding what the keypoints should be** — the part the paper does not have
to answer, because its source and target are the same object. When the target is
a *different* object held by a *different* hand, there is no natural
correspondence, and the choice of keypoints decides everything: a bad set folds
the map (`min det(J)` collapsing from 0.68 to 0.007) while still fitting its own
keypoints perfectly, so it fails silently. Nineteen candidate constructions were
swept; the winner is the **grasp-pose cube**, at **16/20** across five hands and
four objects. `outputs/keypoints_sweep/FINDINGS.md`.

**3. Executing the transported policy** — turning the warped labels into a moving
attractor an impedance-controlled arm chases. Nine candidate attractor laws and
two query sites, on a physics-free bed and then in simulation: query the policy
at the attractor rather than the measured arm, and keep the shipped integrator,
because in physics no law beats another by more than the re-measurement noise.
**15/20** end-to-end, i.e. the executor costs about one cell against the 16/20
replay ceiling. `docs/dynamics_execution.md`.

Getting the object seen and grasped at all sits underneath all three: a text
prompt resolves to an object and a destination, the object is segmented into a
point cloud from simulated depth cameras, and GraspGen-X generates 6-DoF grasps
for whichever gripper is fitted.

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

## What lives where

The first block is the inherited transportation machinery, named by the paper
section it implements — that mapping is kept exact because every function
implementing theory cites its equation. The second block is this project's own
work, which the paper has no counterpart for.

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

| This project | Module |
|---|---|
| Keypoint constructions and their sweep | `tpgpt/experiments/run_keypoint_transport.py`, `run_keypoint_replay.py` |
| Execution laws, the surrogate plant and their sweep | `tpgpt/policy/rollout.py`, `tpgpt/experiments/sweep_execution.py`, `identity_execution.py` |
| Which stage of a run failed, and by how much | `tpgpt/experiments/diagnose.py` |
| What code produced a result | `tpgpt/reporting/provenance.py` |

## Scope

**Built and measured:** the whole cross-embodiment, cross-object pipeline above,
and the transportation substrate it rests on — Sec. III, Appendix A, uncertainty
propagation, the policy refit and the Sec. V-A reshelving result, all of which
had to be correct before the harder question could be asked at all.

**Not attempted, and out of scope:** the paper's Sec. IV baseline comparison
(Reshaped-KMP, Laplacian Editing, LWT, Ensemble-NN, TP-GMM, TP-HMM) and Table I,
and its Sec. V-B to V-D dressing, surface-cleaning and DINO-keypoint
experiments. Those benchmark the *paper's* contribution, which is not the
question here; `tpgpt/metrics/` already provides what they would need if anyone
wants them. See `ROBOTICS_NOTES.md` §8 for the open items that *are* on this
project's critical path — chiefly the residual aim error, a front-approach
demonstration for enclosed shelves, and the `umi` hand failing at `approach` in
every campaign.

## Where the experimental results live

Three documents, in decreasing order of how much detail they carry. Each is
written to be readable on its own, with every table's columns defined and every
number's provenance stated.

| document | what it answers |
|---|---|
| **`outputs/keypoints_sweep/FINDINGS.md`** | **Which keypoints should pin the transportation map?** Experiments A–S. The answer is the *grasp-pose cube*: Experiment R places **16 of 20** across five grippers and four objects from a single Panda demonstration, and all of the remaining failures are the gripping mechanism rather than the map. |
| **`docs/dynamics_execution.md`** | **How should the transported policy be executed?** Experiments A–H, comparing nine attractor laws and two query sites. Query the policy at the attractor, not the measured arm; keep the shipped integrator, because in physics no law beats another by more than the re-measurement noise. The chosen law then runs the same five-hand, four-object grid end-to-end at **15 of 20** — one cell below the executor-free replay ceiling, which is what the executor costs. Includes four figures in `docs/figures/`. |
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
