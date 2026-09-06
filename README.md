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

**Grasping and scene understanding** (in progress): a text prompt resolves to an
object and a destination, the object is segmented into a point cloud from
simulated depth cameras, and GraspGen-X generates 6-DoF grasps for whichever
gripper is fitted. This is the groundwork for transporting a policy to a scene
where both the object *and* the hand are different.

## Quick start

```bash
export MUJOCO_GL=egl
PY=/home/ishita/mujoco_env/bin/python

$PY -m pytest tests/unit -q                                  # fast, no simulator
$PY -m pytest tests -q                                       # everything (~75 s)
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

## Scope

Implemented: all of Sec. III, Appendix A, uncertainty propagation, the policy
refit, and the Sec. V-A reshelving validation.

Deferred (see `ROBOTICS_NOTES.md` §6): grasp filtering and selection, keypoint
extraction for a new-object scene, the Sec. IV baseline comparison, and the
dressing, surface-cleaning and DINO experiments of Sec. V-B to V-D.

## Documentation

- **`ROBOTICS_NOTES.md`** — findings, measurements, and every deviation from the
  paper with its justification. Read this first if you want to know *why* the
  code is the way it is.
- **`CLAUDE.md`** — environment, constraints, and working rules.

## Requirements

CPU only, Python 3.10, MuJoCo 3.7 and robosuite 1.5.2. See `requirements.txt`.
