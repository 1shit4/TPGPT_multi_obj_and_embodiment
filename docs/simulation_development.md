# Simulation development ledger

These are development experiments, not a frozen evaluation. Each rate has its own scheduled denominator. Repeated cases and render reproductions are not independent trials and must not be pooled to claim a 100-run result.

| Batch | Complete / scheduled | Successes | Rate (complete batches only) | Refusals | Setup / diagnostic errors |
|---|---:|---:|---:|---:|---:|
| [replay_24](../outputs/paper_development/replay_24/summary.json) | 24 / 24 | 12 | 50.0% | 0 | 0 / 0 |
| [near_open_replay_24](../outputs/paper_development/near_open_replay_24/summary.json) | 24 / 24 | 11 | 45.8% | 0 | 0 / 0 |
| [edge_path_6](../outputs/paper_development/edge_path_6/summary.json) | 6 / 6 | 2 | 33.3% | 4 | 0 / 0 |
| [joint_measured_gate_6](../outputs/paper_development/joint_measured_gate_6/summary.json) | 6 / 6 | 4 | 66.7% | 0 | 0 / 0 |
| [joint_attractor_30mm_6](../outputs/paper_development/joint_attractor_30mm_6/summary.json) | 6 / 6 | 4 | 66.7% | 0 | 0 / 0 |
| [small_objects_replay_15](../outputs/paper_development/small_objects_replay_15/summary.json) | 15 / 15 | 6 | 40.0% | 0 | 0 / 0 |
| [small_objects_joint_15](../outputs/paper_development/small_objects_joint_15/summary.json) | 15 / 15 | 4 | 26.7% | 0 | 0 / 0 |
| [upright_replay_15](../outputs/paper_development/upright_replay_15/summary.json) | 15 / 15 | 13 | 86.7% | 0 | 0 / 0 |
| [benchmark_replay_16](../outputs/paper_development/benchmark_replay_16/summary.json) | 16 / 16 | 7 | 43.8% | 0 | 0 / 0 |
| [upright_variation_replay_40](../outputs/paper_development/upright_variation_replay_40/summary.json) | 40 / 40 | 26 | 65.0% | 0 | 0 / 0 |
| [upright_joint_15](../outputs/paper_development/upright_joint_15/summary.json) | 15 / 15 | 6 | 40.0% | 0 | 0 / 0 |
| [upright_attractor_15](../outputs/paper_development/upright_attractor_15/summary.json) | 15 / 15 | 11 | 73.3% | 0 | 0 / 0 |
| [compact_development_replay_55](../outputs/paper_development/compact_development_replay_55/summary.json) | 55 / 55 | 41 | 74.5% | 0 | 0 / 0 |
| [nominal_target_6](../outputs/paper_development/nominal_target_6/summary.json) | 6 / 6 | 4 | 66.7% | 0 | 0 / 0 |
| [task_current_6](../outputs/paper_development/task_current_6/summary.json) | 6 / 6 | 1 | 16.7% | 0 | 0 / 0 |
| [nominal_task_replay_55](../outputs/paper_development/nominal_task_replay_55/summary.json) | 55 / 55 | 48 | 87.3% | 0 | 0 / 0 |
| [nominal_task_joint_55](../outputs/paper_development/nominal_task_joint_55/summary.json) | 55 / 55 | 48 | 87.3% | 0 | 0 / 0 |

## What the experiments establish

The upright batch changes the initial rest orientation of three downloaded YCB objects (pudding box, gelatin box, foam brick), then physically settles them. It uses five grippers on the same Panda arm, the original frozen one-box Panda source, target-cloud correction, legacy fallback, stride-four discrete path checking, and direct transported-label replay. New object poses produce new candidate draws; its gain over the flat-object batch is a combined setup-and-selection effect, not an isolated executor improvement.

The 13/15 upright result uses the existing position-placement criterion: XY error below 60 mm and collision geometry base height within 30 mm of the shelf. It does not require preservation of an upright orientation. XArm/pudding finishes at 59.925 mm XY error, just inside this threshold. All 13 successes end with open commands, no held-object flag, and shelf contact; terminal contact evidence is in [upright_release_audit.json](../outputs/paper_development/upright_release_audit.json). Some objects still move late in the trace; a separate post-release settling check is needed before freezing the evaluation.

## Inspected failures and controls

Five normal-selection render reproductions preserve their candidate-cache hashes and selected indices. Both camera timelines and event frames were inspected. The side camera is partly occluded by the arm late in execution; contact traces establish support where the image is obscured. This limitation must not be hidden.

| Case | Observation | Filter evidence | Evidence |
|---|---|---|---|
| Panda/pudding | Grasp, lift and placement observed; terminal trace shows shelf support and no hand contact. | First path accepted, no fallback. | [video](../outputs/paper_development/upright_inspect_5/cell_000/sideview.mp4), [timeline](../outputs/paper_development/upright_inspect_5/cell_000/timeline.jpg), [trace](../outputs/paper_development/upright_inspect_5/cell_000/simulation_steps.json) |
| Robotiq85/pudding | Box lifts but remains against the hand after opening; final object origin is above the destination. | First path accepted; simulator path diagnostic reports no fault at its discrete checked poses. Actual contacts include palm/knuckle geometry. | [video](../outputs/paper_development/upright_inspect_5/cell_001/workspace.mp4), [decision](../outputs/paper_development/upright_inspect_5/cell_001/result.json), [trace](../outputs/paper_development/upright_inspect_5/cell_001/simulation_steps.json) |
| Robotiq140/pudding | Hand misses the grasp and completes an empty motion; box remains on the table. | Legacy fallback restores a kinematically rejected path, joined at waypoint 112. Only 53.8% of the tested portion is reachable. Grasp gate times out at 200 steps. | [video](../outputs/paper_development/upright_inspect_5/cell_002/workspace.mp4), [decision](../outputs/paper_development/upright_inspect_5/cell_002/result.json), [trace](../outputs/paper_development/upright_inspect_5/cell_002/simulation_steps.json) |
| XArm/pudding | Object transported to the shelf; terminal position is a boundary success. | Accepted sampled path, 66 unreachable replay waypoints. | [timeline](../outputs/paper_development/upright_inspect_5/cell_003/timeline.jpg), [trace](../outputs/paper_development/upright_inspect_5/cell_003/simulation_steps.json) |
| Rethink/pudding | Object transported; terminal pose has toppled on the shelf and passes the position criterion. | Accepted sampled path, 68 unreachable replay waypoints. | [timeline](../outputs/paper_development/upright_inspect_5/cell_004/timeline.jpg), [trace](../outputs/paper_development/upright_inspect_5/cell_004/simulation_steps.json) |

## Draft corrections

- State full YCB scan geometry and simulator-derived object pose where used; obstacle clouds are camera observations. Do not call the whole pipeline observed-only.
- State whether rejected candidates were restored and executed. A successful fallback execution is not collision certification.
- State the waypoint stride and endpoint coverage; even every-waypoint checks are discrete.
- State the actual local scripted Panda/one-box source and its hashes. This checkout does not reproduce an unavailable historic campaign.
- Direct replay does not refit a state-dependent policy. The joint-policy prototype does, but uses fixed position-control gains and does not execute transported compliance.
- Changing grippers on a Panda arm is not evidence of transfer across different robot arms.
- Report all scheduled cells, refusals, setup errors, and execution failures. Keep development and evaluation separate.

## Source provenance

Original source: [manifest](../outputs/grasp_audit/source/source.json), labels SHA-256 `da3cd63e566b157d83cee37699ebe24b4340ca6f31914592f2aa4cab4f598dd5`, placement SHA-256 `e004e828e4e8f8917914c7f26f8f0dfd7062ba0132c7f2334b3bad5d3796dc83`. A second compact, zero-carry-rotation source was explored in development; its six pinned transfers scored 3/6 versus 5/6 with the original replay source. A subsequent 55-cell paired source comparison scored 41/55 with the compact source versus 39/55 with the original (seven improvements, five regressions). The compact source was then frozen for the prospective 120-cell evaluation after the nominal-seed development comparison reached 48/55. Final source: [manifest](../outputs/paper_evaluation_v1/source/source.json), labels SHA-256 `1c52fdb032db6a82f2b7b718f5035a772ac63511063fe374ba02daa4cff96038`, placement SHA-256 `b4bb10d0a619d158bf57e7a4be78f902feb37a70867db464eae22c0b2cf7814a`. Every final evaluation cell uses this same demonstration; original-source development runs remain separately identified.

Each run directory stores the exact code snapshot, settings, source-manifest hash, per-cell checkpoints, candidate-cache hash, effective grasp pose, transported labels and simulation-step contacts. The old [grasp audit](grasp_selection_audit.md) remains a separate investigation.
