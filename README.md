# One demonstration, many hands and many objects

## Reproduce the frozen simulation results

The current paper evaluation uses **one fixed scripted Panda/box demonstration**, five
grippers on the same Panda arm, four YCB objects, and six fresh shared pickup/shelf
configurations. Development and evaluation are separate.

| Executor | Success / all scheduled | Rate | Execution failures | Refusals / setup errors |
|---|---:|---:|---:|---:|
| GP reference (`joint_policy`) | 111/120 | 92.5% | 9 | 0 / 0 |
| Direct position replay (`replay`) | 113/120 | 94.2% | 7 | 0 / 0 |

These are **released, supported, stable placement** rates in a development-selected
local workspace, not collision-free rates or results over the full YCB collection.
Target geometry uses full scans at simulator poses. The `task` filter retains legacy
fallbacks. GP execution uses fixed joint-position gains, not transported compliance.
Replay differs in timing and grasp confirmation as well as policy refitting; the
results do not establish a GP success-rate advantage.

- [Results and failure analysis](docs/simulation_evaluation.md)
- [Results LaTeX](docs/simulation_results.tex) and [compiled PDF](outputs/paper_evaluation_v1/manuscript/simulation_results_preview.pdf)
- [Synchronized evidence viewer](outputs/paper_evaluation_v1/evidence_index.html)
- [Development ledger](docs/simulation_development.md) and [original filter investigation](docs/grasp_selection_audit.md)

### 1. Install the evaluated environment

The evaluated host was **Ubuntu 22.04.5, x86-64, Python 3.10.12**, using CPU
MuJoCo 3.7.0 and robosuite 1.5.2 with headless EGL rendering. The repository's
older `requirements.txt` is not the lock for this evaluation. Use
[`requirements-reproduction.lock`](requirements-reproduction.lock), copied from
[the actual installed package inventory](outputs/paper_evaluation_v1/environment.txt).
System package names below are for Ubuntu; on another Linux distribution install
the equivalent OpenGL/EGL and compiler libraries. Obtain the repository checkout
that contains this README, the reproduction script, and the evaluated core changes;
the older base commit alone will fail the frozen code-hash check.

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
  build-essential git libegl1 libgl1 libglfw3 ffmpeg

# First clone the commit containing this README and the evaluated code, then:
# cd TPGPT_multi_obj_and_embodiment
python3.10 -m venv .venv-reproduction
export PY="$PWD/.venv-reproduction/bin/python"
"$PY" -m pip install --no-deps -r requirements-reproduction.lock
"$PY" -m pip install --no-deps --no-build-isolation -e .
"$PY" -c 'import sys,mujoco,robosuite; print(sys.version, mujoco.__version__, robosuite.__version__)'
```

The `--no-deps` flags preserve the recorded versions. The package lock includes
build tools and ffmpeg's Python bridge. The evaluated environment's `pip check`
reports that robosuite's optional `mink` and `qpsolvers` are absent; the Panda
joint-position controller used here does not use them. A fresh install is a
proposed reconstruction until the archive verifier and rendered smoke run pass.
On the evaluated host the interpreter already exists at
`/home/sra/mujoco_env/bin/python`; you may point `PY` there instead of making
a virtual environment.

```bash
export MUJOCO_GL=egl
export GRASPGENX_GRIPPER_CFG_DIR=/home/sra/gripper_descriptions_hf
export NUMBA_CACHE_DIR=/tmp/tpgpt-numba
export MPLCONFIGDIR=/tmp/tpgpt-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
```

Set `GRASPGENX_GRIPPER_CFG_DIR` to the **checkout root** containing
`gripper_descriptions/assets/x_grippers/`, not to that inner directory.
`MUJOCO_GL=egl` must be set before importing MuJoCo or robosuite. The cached
reproduction uses CPU physics and needs no live GraspGen-X server.

### 2. Restore the ignored inputs and external assets

The following artifacts must be present. **A Git clone alone is insufficient:** bulk
outputs and assets are ignored by Git and must accompany the checkout. There is no
published download URL for this local evaluation bundle.

| Input | Purpose |
|---|---|
| `outputs/paper_evaluation_v1/protocol.json` | Frozen settings, code/cell hashes and sampling conditions |
| `outputs/paper_evaluation_v1/cells_{joint_policy,replay}.json` | Original 120 cells per executor |
| `outputs/paper_evaluation_v1/source/` | Successful source manifest, labels, placement and source video |
| `outputs/paper_evaluation_v1/candidates/*.npz` | Original candidate draws; do not regenerate for exact reproduction |
| `outputs/paper_evaluation_v1/{joint_policy,replay}/` | Configurations, per-cell results and summaries used to pin and compare reproductions |
| `outputs/paper_evaluation_v1/{asset_inventory,grasp_assets_inventory}.json` | Required asset paths and SHA-256 hashes |
| `assets/ycb/`, `assets/ycb_mjcf/`, external gripper-description checkout | Scan meshes, collision assets and hand descriptions named in the inventories |
| `outputs/paper_evaluation_v1/code/` and `code.patch` | Exact evaluated code snapshot and tracked diff; the original Git commit alone is insufficient |

To transfer the study from the original evaluated machine, create archives there;
allow several gigabytes for the archives and unpacked files. Transfer these tarballs by your
normal file-copy method. **Do not add them to Git:** they contain large and
third-party assets. From the original repository root:

```bash
tar -czf /path/to/study-data.tar.gz \
  outputs/paper_evaluation_v1 assets/ycb assets/ycb_mjcf
tar -C /home/sra -czf /path/to/external-assets.tar.gz \
  gripper_descriptions_hf GraspGenX/ext
sha256sum /path/to/study-data.tar.gz /path/to/external-assets.tar.gz \
  > /path/to/reproduction-tarballs.sha256
```

On the new host, in the checked-out repository, restore the study data and put
the external assets next to it:

```bash
tar -xzf /path/to/study-data.tar.gz
mkdir -p external
tar -C external -xzf /path/to/external-assets.tar.gz
export GRASPGENX_GRIPPER_CFG_DIR="$PWD/external/gripper_descriptions_hf"
# Compare these digests to reproduction-tarballs.sha256 from the original host.
sha256sum /path/to/study-data.tar.gz /path/to/external-assets.tar.gz
```

The external archive includes the exact gripper descriptions and archived
GraspGen-X files listed in the inventory. The cached runner does not execute
the external model, but the verifier checks these files. If you already have
those assets, copy them to the corresponding locations instead. Do not rebuild
convex collision meshes for exact reproduction; use the archived
`assets/ycb_mjcf` geometry. The raw `assets/ycb` scans and all candidate NPZ
files must also match their recorded hashes.

The archive verifier checks both asset inventories, including archived external
planner files. Cached simulation does **not** need a live grasp-generation server,
but it still needs the gripper geometry. New candidate generation uses an external
GraspGen-X environment and is a new experiment, not an exact reproduction. The
original environment seed does not reproduce the external planner's random draws.

### 3. Verify the archive without running simulation

```bash
"$PY" scripts/reproduce_evaluation.py verify \
  --archive outputs/paper_evaluation_v1
```

This checks the frozen code, cell manifests, asset hashes, successful source,
complete original checkpoints and canonical candidate-array hashes. It should print
`joint_policy: 111/120 (92.5%)` and `replay: 113/120 (94.2%)`.
Additional scripts added after the freeze are allowed; changes to any file listed
in the frozen code inventory are rejected. If you changed evaluated code, use the
archived snapshot in a **separate checkout**, preserving your current edits.

If asset directories are elsewhere, repeat `--asset-map OLD=NEW` for each
changed inventory prefix and point `GRASPGENX_GRIPPER_CFG_DIR` to the restored
checkout root. The two mappings below cover the archive layout just described.

If you restored the two tarballs into `external/` as above, use both mappings:

```bash
"$PY" scripts/reproduce_evaluation.py verify \
  --asset-map /home/sra/gripper_descriptions_hf="$PWD/external/gripper_descriptions_hf" \
  --asset-map /home/sra/GraspGenX/ext="$PWD/external/GraspGenX/ext"
```

This remaps verification paths; it does not rewrite mesh references or runtime
configuration. Supply the same two mappings on every `run` command on that host.
A missing/changed asset is an error to resolve, not permission to substitute a
different object. The examples below omit mappings because they use the original
machine's asset paths. For the restored layout, append both mapping flags to a
run, for example:

```bash
"$PY" scripts/reproduce_evaluation.py run \
  --out outputs/reproduction_smoke --cells 0 --render \
  --asset-map /home/sra/gripper_descriptions_hf="$PWD/external/gripper_descriptions_hf" \
  --asset-map /home/sra/GraspGenX/ext="$PWD/external/GraspGenX/ext"
```

### 4. Run a small reproduction, then the full paired suite

The runner uses normal selection with expected candidate index, candidate-cache
hash, effective grasp pose and transported-label hash. It never forces a rejected
grasp and never requests fresh candidates. Outputs must be outside the original
archive. Physics runs **serially**, GP first and replay second.

```bash
# One success control, both executors, with overview and side videos.
"$PY" scripts/reproduce_evaluation.py run \
  --out outputs/reproduction_smoke --cells 0 --render

# All 120 cells per executor: 240 rollouts, two separate denominators.
"$PY" scripts/reproduce_evaluation.py run \
  --out outputs/reproduction_120

# Representative failure pairs; rendering adds substantial runtime/storage.
"$PY" scripts/reproduce_evaluation.py run \
  --out outputs/reproduction_failures --cells 7,29,31,67,108,110,113,114,118,119 --render
```

Use `--executor joint_policy` or `--executor replay` for one executor. Cell indices
are **zero-based indices in the original 120-cell manifests**. A subset's local
`cell_000` is not necessarily original cell 000; `reproduction.json` records the
mapping. Smoke/subset results must not be presented as the 120-trial rate.

Repeat the **identical command** to resume after interruption. Completed per-cell
`result.json` checkpoints are retained, including failures and setup errors; they
are not silently retried. Changed source, code, executor, cells or render settings
require a new output directory. Do not run concurrent processes against the same
output directory. Inspect progress with, for example:

```bash
tail -f outputs/reproduction_120/joint_policy.log
```

When finished, the runner exports a CSV, summary, paired transitions and success
matrix under `report/`, and writes `reproduction_verification.json`. A full exact
reproduction should match **240/240** original outcomes/input identities. Any
mismatch produces a nonzero exit status and keeps the results for investigation;
do not tune or discard the mismatched cells to recover the headline rate.

### 5. Regenerate reports and inspect videos

```bash
# Rebuild a completed reproduction's report and compare it with the archive.
"$PY" scripts/reproduce_evaluation.py report --out outputs/reproduction_120

# Recompute the original aggregate results in a NEW report directory; no physics.
"$PY" scripts/paper_results.py \
  --run joint_policy=outputs/paper_evaluation_v1/joint_policy \
  --run replay=outputs/paper_evaluation_v1/replay \
  --out outputs/recomputed_paper_report --plot
```

`report/index.html` links each cell to its results, traces and available videos.
Rendered runs save `workspace.mp4`, `sideview.mp4`, event PNGs, `events.json` and
`simulation_steps.json`. Both camera videos use the same 20 Hz frame index;
`events.json` maps events to frames and simulation time. Inspect both cameras
alongside the contacts and trace: the robot can occlude either camera. The final
two-second open-jaw observation also occurs after a stalled run; it is not proof
that the executor reached its release phase.

The original [evidence viewer](outputs/paper_evaluation_v1/evidence_index.html)
contains 30 matched, manually reviewed reproductions: ten failure pairs and five
success controls. Its play/seek controls synchronize the cameras within a run;
event buttons align each executor to its own event. Those reproductions are not
additional headline evaluation trials.

### 6. Tests, manuscript build and original completion audit

```bash
# Run serially. The archived tested result was 832 passed, 7 warnings, no skips.
# Added reproducibility-runner tests increase today's total.
"$PY" -m pytest tests -q

# Compile the results section, table and matrix. On Ubuntu, first install
# texlive-latex-base if pdflatex is absent.
mkdir -p outputs/reproduced_manuscript
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=outputs/reproduced_manuscript docs/simulation_results_preview.tex

# Optional: validate the COMPLETE original evidence package (requires ffprobe).
"$PY" outputs/paper_evaluation_v1/completion_audit.py
```

The full suite contains live GraspGen integration tests; without that server some
checks skip. Record skips rather than claiming the archived zero-skip result.
The completion audit additionally requires the original rendered videos, manual
review records, source/asset inventories, test log and compiled PDF; it writes an
updated receipt in the archive. It is not needed to run a small reproduction, and
automated hash checks do not replace visual review.

### Validation of this reproducibility entry point

The runner was checked against the original archive, then used for a rendered
cell-000 reproduction under both executors: **2/2 outcomes and pinned inputs
matched**. Repeating the command preserved both checkpoint modification times;
report regeneration also passed. All four new videos matched their control-step
counts at 20 fps. The runner and report unit tests passed **11/11**. This smoke
check does not claim a fresh rerun of all 240 rollouts. Local verification output:
[`outputs/reproduction_readme_smoke/reproduction_verification.json`](outputs/reproduction_readme_smoke/reproduction_verification.json).

### Script roles and protocol details

| Script | Role |
|---|---|
| `scripts/reproduce_evaluation.py` | Supported archive verification, cache-only reproduction, resume and original-versus-reproduction comparison |
| `tpgpt/experiments/forensics.py` | Simulation engine, candidate evidence, traces, videos and atomic cell checkpoints |
| `scripts/paper_results.py` | Aggregate complete checkpoints; count all scheduled cells and export paired results |
| `scripts/freeze_evaluation.py` | Historical prospective-freeze procedure; requires development archives and a new output directory |
| `outputs/paper_evaluation_v1/*.py` | Archived study-specific analysis, contact audit, video review and publication builders |

Do not use the archived `run_serial.sh` or `run_evidence_serial.sh` as portable
entry points: they contain original machine paths, output destinations and, in the
latter, a historical process-ID wait. Use the supported reproduction runner above.
Do not rerun `freeze_evaluation.py` over the original archive or treat newly drawn
grasps as a reproduction of its reported numbers.

The final grid is Panda, Robotiq85, Robotiq140, XArm and Rethink × pudding, gelatin,
brick and meat × E01–E06, targeting `top_middle`. The `task` variant corrects the
target-cloud input, samples stride-four path waypoints plus grasp/release/final,
requires task-waypoint reachability, and retains legacy fallbacks. Numerical IK
uses a nominal seed; the physical robot still starts from its ordinary home pose.
GP uses reference-state queries, a 30 mm tracking gate, phase speed 0.25 and a
2400-step budget. Replay holds each label for eight control steps. The frozen
protocol and cell manifests are authoritative for every setting.

Success uses the original <60 mm horizontal / 30 mm base-height slot criterion,
then two seconds of open-jaw settling; during the final second the object must
remain in the slot, touch the destination board, avoid robot contact and stay
within a 2 mm position span. All scheduled cells—including errors, refusals and
execution failures—remain in the denominator. Seven cells per executor used
legacy path fallback; GP succeeded in four and replay in six. Do not relabel this
as a strict-filter evaluation.

---

## Earlier experiments and project background

The results and original machine setup below describe earlier campaigns, **not**
the frozen 120-cell study above. In particular, the older impedance executor,
synthetic objects, seven-hand grid and `/home/ishita` paths are historical.


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
| the paper's own setting — same object and hand, 20 randomised scenes | **20/20**, 7.1 mm median placement error, against 5/20 without the nonlinear stage (p = 5.5e-6) |
| **seven different hands × four different objects**, position-control replay | **19/28** |
| the same 28 cells, executed by the learned policy under impedance control | **17/28** |

Those last two are paired **grasp for grasp** — the replay is pinned to the
campaign's own chosen grasp, and 0 of 28 cells differ in the grasp executed or
in the map it produced. The 17-vs-19 gap is not significant (Fisher p = 0.78),
but on the 15 cells both placed, the policy executor is **6.1 mm less
accurate** (p = 0.048).

The seven hands span 24 to 61 mm of tool offset and 50 to 125 mm of jaw
aperture, include revolute linkages as well as parallel jaws, and one of them
(`robotiq3f`) has **three fingers** rather than two; the four objects are a milk
carton, a can, a 15 cm cereal box and a loaf of bread. Only one hand-and-object
pair — the Panda on its own object — is the one the demonstration was recorded
with.

The three-finger hand is the honest weak spot: it places **0 of 4**, and it does
so under *both* controllers with among the best-conditioned maps in the run
(`min det(J)` 0.86–0.98), so the failure is neither the transport nor the
executor. It sits between the grasp and the grip, and it is open.

## The three pieces the question breaks into

**1. Transportation** — the inherited substrate, and the part that is settled.
Given a set of *source keypoints* (where things were in the demonstration) and
*target keypoints* (where the corresponding things are now), fit a smooth
nonlinear map `phi` that carries one onto the other, push every policy label
through it — position, velocity, end-effector orientation, Cartesian stiffness
and damping — and refit a policy on the result. This is the paper's method, and
it reproduces: 20/20 at 7.1 mm in the paper's own setting, against 5/20 with the
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
**17/28** end-to-end. Against a grasp-for-grasp-paired position-control replay
the executor costs no success rate but about 6 mm of placement precision.
`docs/dynamics_execution.md`.

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
| **`docs/dynamics_execution.md`** | **How should the transported policy be executed?** Experiments A–H, comparing nine attractor laws and two query sites. Query the policy at the attractor, not the measured arm; keep the shipped integrator, because in physics no law beats another by more than the re-measurement noise. The chosen law then runs a seven-hand, four-object grid end-to-end at **17 of 28**, paired grasp for grasp against a position-control replay at 19/28: no success-rate difference (p = 0.78), but 6.1 mm less placement precision (p = 0.048). Includes four figures in `docs/figures/`. |
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
